#include "wifi_mgr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_wifi.h"

#include "cJSON.h"
#include "dns_server.h"
#include "settings.h"
#include "outputs.h"

static const char *TAG = "wifi_mgr";

#define GOT_IP_BIT           BIT0
#define CONNECT_TIMEOUT_MS   25000
#define SETUP_IDLE_REBOOT_US (180LL * 1000000LL)
#define SETUP_AP_SSID        "S3-Radar-Setup"
#define RADAR_TX_SSID        "WiFi-Radar-Net"
#define RADAR_TX_PASSWORD    "12345678"

static EventGroupHandle_t s_eg;
static esp_netif_t *s_sta;
static esp_netif_t *s_ap;
static volatile bool s_try;
static volatile bool s_connected;
static bool s_started;
static netmode_t s_mode = NETMODE_STA;
static char s_ip[16] = "";
static uint8_t s_channel;
static volatile int64_t s_activity_us;
static char *s_scan_cache;

static void set_egypt_country_profile(void)
{
    /*
     * ESP-IDF 5.5.6 country-code table omits EG. Configure a conservative
     * manual profile for 2.4 GHz channels 1..13; NTRA 2024 guidance lists
     * 2400..2483.5 MHz for WLAN with 100 mW EIRP max.
     *
     * esp_wifi_set_country() does not validate regional rules; this explicit
     * profile must be re-reviewed if RF hardware/antenna or regulation changes.
     */
    const wifi_country_t country = {
        .cc = "EG",
        .schan = 1,
        .nchan = 13,
        .max_tx_power = 11,
        .policy = WIFI_COUNTRY_POLICY_MANUAL,
    };
    ESP_ERROR_CHECK(esp_wifi_set_country(&country));
}

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;

    if (base == WIFI_EVENT) {
        if (id == WIFI_EVENT_STA_START) {
            if (s_try) {
                ESP_LOGI(TAG, "station started; connecting");
                (void)esp_wifi_connect();
            }
        } else if (id == WIFI_EVENT_STA_DISCONNECTED) {
            s_connected = false;
            s_ip[0] = '\0';
            if (s_try) {
                outputs_set_net_state(OUT_NET_CONNECTING);
                ESP_LOGW(TAG, "link disconnected; automatic reconnect");
                (void)esp_wifi_connect();
            }
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *event = (const ip_event_got_ip_t *)data;
        snprintf(s_ip, sizeof(s_ip), IPSTR, IP2STR(&event->ip_info.ip));
        s_connected = true;
        (void)esp_wifi_set_ps(WIFI_PS_NONE);

        wifi_ap_record_t ap = {0};
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
            s_channel = ap.primary;
            ESP_LOGI(TAG,
                     "connected SSID='%s' channel=%u BSSID=%02x:%02x:%02x:%02x:%02x:%02x RSSI=%d",
                     (const char *)ap.ssid, ap.primary,
                     ap.bssid[0], ap.bssid[1], ap.bssid[2],
                     ap.bssid[3], ap.bssid[4], ap.bssid[5], ap.rssi);

            settings_t *cfg = settings_get();
            if (cfg->dedicated_mode && cfg->wifi_channel != ap.primary) {
                ESP_LOGW(TAG,
                         "configured channel hint %u differs from TX AP channel %u; using actual AP channel",
                         cfg->wifi_channel, ap.primary);
            }
        }

        xEventGroupSetBits(s_eg, GOT_IP_BIT);
    }
}

/* ------------------------------------------------------------- scanning */

static void do_scan(void)
{
    const wifi_scan_config_t scan = {0};
    if (esp_wifi_scan_start(&scan, true) != ESP_OK) return;

    uint16_t count = 0;
    if (esp_wifi_scan_get_ap_num(&count) != ESP_OK) return;
    if (count > 24) count = 24;

    wifi_ap_record_t *records = calloc(count ? count : 1, sizeof(*records));
    if (records == NULL) return;

    if (esp_wifi_scan_get_ap_records(&count, records) != ESP_OK) {
        free(records);
        return;
    }

    cJSON *array = cJSON_CreateArray();
    if (array == NULL) {
        free(records);
        return;
    }

    for (int i = 0; i < count; i++) {
        const char *ssid = (const char *)records[i].ssid;
        if (ssid[0] == '\0') continue;

        bool duplicate = false;
        cJSON *item;
        cJSON_ArrayForEach(item, array) {
            const cJSON *saved_ssid = cJSON_GetObjectItemCaseSensitive(item, "ssid");
            if (saved_ssid && cJSON_IsString(saved_ssid) &&
                strcmp(saved_ssid->valuestring, ssid) == 0) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        cJSON *entry = cJSON_CreateObject();
        if (entry == NULL) continue;
        cJSON_AddStringToObject(entry, "ssid", ssid);
        cJSON_AddNumberToObject(entry, "rssi", records[i].rssi);
        cJSON_AddBoolToObject(entry, "open",
                              records[i].authmode == WIFI_AUTH_OPEN);
        cJSON_AddNumberToObject(entry, "channel", records[i].primary);
        cJSON_AddItemToArray(array, entry);
    }

    free(s_scan_cache);
    s_scan_cache = cJSON_PrintUnformatted(array);
    cJSON_Delete(array);
    free(records);
}

const char *wifi_mgr_scan(bool refresh)
{
    if (refresh || s_scan_cache == NULL) do_scan();
    return s_scan_cache ? s_scan_cache : "[]";
}

/* --------------------------------------------------------------- portal */

static void idle_task(void *arg)
{
    (void)arg;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        if ((settings_get()->dedicated_mode || settings_has_wifi()) &&
            (esp_timer_get_time() - s_activity_us) > SETUP_IDLE_REBOOT_US) {
            ESP_LOGW(TAG, "setup portal idle; retry saved network settings");
            esp_restart();
        }
    }
}

static esp_err_t start_portal(void)
{
    ESP_LOGW(TAG, "starting Wi-Fi setup portal");
    s_try = false;
    s_connected = false;
    s_mode = NETMODE_SETUP;
    s_activity_us = esp_timer_get_time();
    outputs_set_net_state(OUT_NET_SETUP);

    wifi_config_t ap = {0};
    strlcpy((char *)ap.ap.ssid, SETUP_AP_SSID, sizeof(ap.ap.ssid));
    ap.ap.ssid_len = strlen(SETUP_AP_SSID);
    ap.ap.channel = 1;
    ap.ap.max_connection = 4;
    ap.ap.authmode = WIFI_AUTH_OPEN;

    if (s_started) (void)esp_wifi_disconnect();
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    if (!s_started) {
        ESP_ERROR_CHECK(esp_wifi_start());
        s_started = true;
    }

    do_scan();

    static dns_server_config_t dns_cfg =
        DNS_SERVER_CONFIG_SINGLE("*", "WIFI_AP_DEF");
    if (start_dns_server(&dns_cfg) == NULL) {
        ESP_LOGE(TAG, "captive DNS server failed");
    }

    (void)xTaskCreate(idle_task, "portal_idle", 2048, NULL, 3, NULL);
    ESP_LOGI(TAG, "join '%s', then open http://192.168.4.1", SETUP_AP_SSID);
    return ESP_OK;
}

/* ----------------------------------------------------------- public API */

netmode_t wifi_mgr_start(void)
{
    settings_t *cfg = settings_get();

    s_eg = xEventGroupCreate();
    if (s_eg == NULL) return NETMODE_SETUP;

    s_sta = esp_netif_create_default_wifi_sta();
    s_ap = esp_netif_create_default_wifi_ap();
    if (s_sta == NULL || s_ap == NULL) return NETMODE_SETUP;

    const wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_netif_set_hostname(s_sta, cfg->hostname));

    set_egypt_country_profile();

    ESP_ERROR_CHECK(esp_wifi_set_protocol(
        WIFI_IF_STA,
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));

    wifi_config_t sta = {0};
    if (cfg->dedicated_mode) {
        strlcpy((char *)sta.sta.ssid, RADAR_TX_SSID, sizeof(sta.sta.ssid));
        strlcpy((char *)sta.sta.password, RADAR_TX_PASSWORD,
                sizeof(sta.sta.password));
        sta.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
        ESP_LOGI(TAG, "dedicated mode: joining '%s' (channel hint %u)",
                 RADAR_TX_SSID, cfg->wifi_channel);
    } else if (settings_has_wifi()) {
        strlcpy((char *)sta.sta.ssid, cfg->wifi_ssid, sizeof(sta.sta.ssid));
        strlcpy((char *)sta.sta.password, cfg->wifi_pass,
                sizeof(sta.sta.password));
        sta.sta.threshold.authmode = cfg->wifi_pass[0] ?
                                     WIFI_AUTH_WPA_PSK : WIFI_AUTH_OPEN;
        ESP_LOGI(TAG, "router mode: joining saved SSID '%s'", cfg->wifi_ssid);
    } else {
        ESP_LOGW(TAG, "router mode selected but no router credentials are saved");
        start_portal();
        return NETMODE_SETUP;
    }

    /* Channel 0 lets the driver scan and locate TX even after its Kconfig channel changes. */
    sta.sta.channel = cfg->dedicated_mode ? cfg->wifi_channel : 0;
    sta.sta.scan_method = WIFI_FAST_SCAN;
    sta.sta.sort_method = WIFI_CONNECT_AP_BY_SIGNAL;
    sta.sta.pmf_cfg.capable = true;
    sta.sta.pmf_cfg.required = false;

    outputs_set_net_state(OUT_NET_CONNECTING);
    s_try = true;
    s_mode = NETMODE_STA;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
    ESP_ERROR_CHECK(esp_wifi_start());
    s_started = true;

    const EventBits_t bits = xEventGroupWaitBits(
        s_eg, GOT_IP_BIT, pdFALSE, pdFALSE,
        pdMS_TO_TICKS(CONNECT_TIMEOUT_MS));

    if ((bits & GOT_IP_BIT) != 0) {
        (void)esp_wifi_set_ps(WIFI_PS_NONE);
        (void)esp_wifi_set_max_tx_power(44);
        outputs_set_net_state(OUT_NET_OK);

        wifi_ap_record_t ap = {0};
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
            s_channel = ap.primary;
        }
        return NETMODE_STA;
    }

    ESP_LOGW(TAG, "network connection timeout after %d ms", CONNECT_TIMEOUT_MS);
    start_portal();
    return NETMODE_SETUP;
}

netmode_t wifi_mgr_mode(void) { return s_mode; }
const char *wifi_mgr_ip(void)
{
    return s_mode == NETMODE_SETUP ? "192.168.4.1" : s_ip;
}
bool wifi_mgr_connected(void) { return s_connected; }
uint8_t wifi_mgr_channel(void) { return s_channel; }
void wifi_mgr_touch(void) { s_activity_us = esp_timer_get_time(); }
