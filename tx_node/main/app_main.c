/*
 * WiFi Radar — Transmitter TX (ESP32-S3, ESP-IDF 5.5+)
 * SoftAP مستقل + UDP heartbeat كل 10ms. مهمة الإرسال مثبتة على Core 1.
 *
 * EG profile is supplied with esp_wifi_set_country() because EG is not in
 * the built-in esp_wifi_set_country_code() country-code table in IDF 5.5.6.
 * Custom country data does not get validated by ESP-IDF; verify NTRA rules
 * and antenna EIRP for the exact board before deployment.
 */

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "lwip/inet.h"
#include "lwip/sockets.h"

#define AP_SSID       "WiFi-Radar-Net"
#define AP_PASSWORD   "12345678"
#define AP_CLIENTS    4
#ifndef CONFIG_RADAR_AP_CHANNEL
/* Safe default for builds where the optional project setting is unavailable. */
#define CONFIG_RADAR_AP_CHANNEL 1
#endif
#define AP_CHANNEL    CONFIG_RADAR_AP_CHANNEL
#define UDP_BROADCAST "192.168.4.255"
#define UDP_PORT      3333
#define PERIOD_US     10000LL

static const char *TAG = "radar_tx";
static QueueHandle_t s_ticks;
static esp_timer_handle_t s_timer;
static volatile uint32_t s_sequence;
static volatile uint32_t s_timer_queue_drops;
static volatile uint32_t s_udp_errors;

static void configure_egypt_country(void)
{
    /*
     * Channel plan: 1..13 (2.4 GHz only).
     * NTRA SRD/Wi-Fi document lists 2400..2483.5 MHz and 100mW EIRP maximum.
     * 11dBm device power cap is applied separately below. Antenna gain must
     * still be included in the effective radiated power calculation.
     */
    const wifi_country_t eg = {
        .cc = "EG",
        .schan = 1,
        .nchan = 13,
        .max_tx_power = 11,
        .policy = WIFI_COUNTRY_POLICY_MANUAL,
    };
    ESP_ERROR_CHECK(esp_wifi_set_country(&eg));
}

/* ESP Timer task callback: no socket calls, no blocking. */
static void timer_callback(void *arg)
{
    (void)arg;
    const uint32_t seq = s_sequence++;
    if (xQueueSend(s_ticks, &seq, 0) != pdTRUE) {
        s_timer_queue_drops++;
    }
}

/* The actual network I/O is done on Core 1, not in the timer callback. */
static void udp_sender_task(void *arg)
{
    (void)arg;

    const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "failed to create UDP socket");
        vTaskDelete(NULL);
        return;
    }

    const int broadcast = 1;
    (void)setsockopt(sock, SOL_SOCKET, SO_BROADCAST,
                     &broadcast, sizeof(broadcast));

    const struct sockaddr_in peer = {
        .sin_family = AF_INET,
        .sin_port = htons(UDP_PORT),
        .sin_addr.s_addr = inet_addr(UDP_BROADCAST),
    };

    uint32_t previous_seq = 0;
    int64_t previous_send_us = 0;
    uint32_t sent = 0;
    uint32_t jitter_max_us = 0;
    uint64_t jitter_sum_us = 0;
    uint32_t jitter_samples = 0;

    for (;;) {
        uint32_t seq = 0;
        if (xQueueReceive(s_ticks, &seq, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        const uint32_t payload = htonl(seq); /* exactly 4 bytes */
        const int64_t now_us = esp_timer_get_time();

        const int written = sendto(sock, &payload, sizeof(payload), 0,
                                   (const struct sockaddr *)&peer,
                                   sizeof(peer));
        if (written != (int)sizeof(payload)) {
            s_udp_errors++;
            continue;
        }

        if (previous_send_us != 0) {
            const uint32_t seq_delta = seq - previous_seq;
            const int64_t expected_us = (int64_t)seq_delta * PERIOD_US;
            const int64_t actual_us = now_us - previous_send_us;
            const uint32_t jitter = (uint32_t)llabs(actual_us - expected_us);

            jitter_sum_us += jitter;
            jitter_samples++;
            if (jitter > jitter_max_us) jitter_max_us = jitter;
        }

        previous_seq = seq;
        previous_send_us = now_us;
        sent++;

        if ((sent % 1000U) == 0U) {
            const float average = jitter_samples
                ? (float)jitter_sum_us / (float)jitter_samples : 0.0f;

            ESP_LOGI(TAG,
                     "TX sent=%" PRIu32 " avg_jitter=%.1f us max_jitter=%" PRIu32
                     " us timer_drops=%" PRIu32 " udp_errors=%" PRIu32,
                     sent, average, jitter_max_us,
                     s_timer_queue_drops, s_udp_errors);
        }
    }
}

static void start_100hz_heartbeat(void)
{
    s_ticks = xQueueCreate(64, sizeof(uint32_t));
    if (s_ticks == NULL) {
        ESP_LOGE(TAG, "failed to allocate heartbeat queue");
        return;
    }

    if (xTaskCreatePinnedToCore(udp_sender_task, "radar_udp_tx",
                                4096, NULL, 6, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "failed to create Core 1 sender task");
        vQueueDelete(s_ticks);
        s_ticks = NULL;
        return;
    }

    const esp_timer_create_args_t args = {
        .callback = timer_callback,
        .arg = NULL,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "radar_100hz",
        .skip_unhandled_events = true,
    };
    ESP_ERROR_CHECK(esp_timer_create(&args, &s_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(s_timer, PERIOD_US));
    ESP_LOGI(TAG, "100Hz timer schedule started (period=%lld us)", PERIOD_US);
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES ||
        err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    (void)esp_netif_create_default_wifi_ap();

    const wifi_init_config_t wifi_init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_init));
    configure_egypt_country();

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));

    wifi_config_t ap = {0};
    strlcpy((char *)ap.ap.ssid, AP_SSID, sizeof(ap.ap.ssid));
    strlcpy((char *)ap.ap.password, AP_PASSWORD, sizeof(ap.ap.password));
    ap.ap.ssid_len = strlen(AP_SSID);
    ap.ap.channel = AP_CHANNEL;
    ap.ap.max_connection = AP_CLIENTS;
    ap.ap.authmode = WIFI_AUTH_WPA2_PSK;
    ap.ap.ssid_hidden = 0;
    ap.ap.beacon_interval = 100;
    ap.ap.pairwise_cipher = WIFI_CIPHER_TYPE_CCMP;
    ap.ap.pmf_cfg.capable = true;
    ap.ap.pmf_cfg.required = false;

    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    ESP_ERROR_CHECK(esp_wifi_set_protocol(
        WIFI_IF_AP, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT20));

    /* Fixed HT20 MCS0-SGI: must be configured before esp_wifi_start(). */
    ESP_ERROR_CHECK(esp_wifi_config_80211_tx_rate(
        WIFI_IF_AP, WIFI_PHY_RATE_MCS0_SGI));

    ESP_ERROR_CHECK(esp_wifi_start());

    /* esp_wifi_set_max_tx_power uses quarter-dBm: 44 = 11 dBm. */
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(44));

    /*
     * Power-save applies to station operation; some IDF targets report
     * NOT_SUPPORTED for a SoftAP-only device. The AP remains operational.
     */
    err = esp_wifi_set_ps(WIFI_PS_NONE);
    if (err != ESP_OK) {
        ESP_LOGI(TAG, "WIFI_PS_NONE skipped for SoftAP-only mode (%s)",
                 esp_err_to_name(err));
    }

    uint8_t mac[6] = {0};
    ESP_ERROR_CHECK(esp_wifi_get_mac(WIFI_IF_AP, mac));

    ESP_LOGI(TAG, "SoftAP ready: %s (192.168.4.1)", AP_SSID);
    ESP_LOGI(TAG, "channel=%u HT20, max clients=%u, password=%s",
             (unsigned)AP_CHANNEL, (unsigned)AP_CLIENTS, AP_PASSWORD);
    ESP_LOGI(TAG, "country=EG, AP MAC=%02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    ESP_LOGI(TAG, "fixed PHY MCS0-SGI, power cap 11 dBm");
    start_100hz_heartbeat();
}
