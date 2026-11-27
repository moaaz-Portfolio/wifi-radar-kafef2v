#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "mdns.h"

#include "app.h"
#include "settings.h"
#include "outputs.h"
#include "wifi_mgr.h"
#include "web_server.h"
#include "sensing.h"
#include "notify.h"
#include "csi_stream.h"
#include "door_sensor.h"

static const char *TAG = "app";

static void reboot_task(void *arg)
{
    vTaskDelay(pdMS_TO_TICKS((uint32_t)(uintptr_t)arg));
    esp_restart();
}

void app_reboot_later(uint32_t ms)
{
    xTaskCreate(reboot_task, "reboot", 2048, (void *)(uintptr_t)ms, 5, NULL);
}

static void start_mdns(const settings_t *c)
{
    if (mdns_init() != ESP_OK) {
        ESP_LOGW(TAG, "mdns init failed");
        return;
    }
    mdns_hostname_set(c->hostname);
    mdns_instance_name_set(FW_NAME);
    mdns_service_add(FW_NAME, "_http", "_tcp", 80, NULL, 0);
    ESP_LOGI(TAG, "open http://%s.local", c->hostname);
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

    ESP_ERROR_CHECK(settings_load());
    settings_t *cfg = settings_get();

    /*
     * Dedicated mode always uses the CSI Analyzer role.
     * This is RAM-only: switching back to router mode restores the user's saved role.
     * Do not clear stream_host; it may intentionally point at a laptop address.
     */
    if (cfg->dedicated_mode && cfg->stream_port == 0) {
        cfg->stream_port = 5005;
    }

    outputs_init(cfg);

    const netmode_t network_mode = wifi_mgr_start();

    /* Door sensing stays active in both dedicated and router modes. */
    ESP_ERROR_CHECK(door_sensor_start());
    ESP_ERROR_CHECK(web_server_start());

    if (network_mode == NETMODE_STA) {
        start_mdns(cfg);

        if (cfg->dedicated_mode) {
            ESP_ERROR_CHECK(csi_stream_start());
        } else {
            if (cfg->role == 1) {
                ESP_ERROR_CHECK(sender_start());
            } else if (cfg->role == 2) {
                ESP_ERROR_CHECK(csi_stream_start());
            } else {
                ESP_ERROR_CHECK(sensing_start());
            }
            if (cfg->role != 2) notify_start();
        }

        ESP_LOGI(TAG, "application started: dedicated=%s role=%u",
                 cfg->dedicated_mode ? "yes" : "no", cfg->role);
    } else {
        ESP_LOGW(TAG, "setup mode: connect to S3-Radar-Setup and open http://192.168.4.1");
    }
}
