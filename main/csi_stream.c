/*
 * WiFi Radar Analyzer RX — CSI + heartbeat receiver
 *
 * Maintains the original Python CSI1 packet exactly:
 * "CSI1" | little-endian seq u32 | little-endian ts_ms u32 |
 * signed RSSI i8 | feature count 64 | amplitude[64] bytes
 *
 * UDP CSI: 5005
 * TX heartbeat monitor: UDP 3333
 * Door telemetry remains independently transmitted by door_sensor.c on 5006.
 */

#include "csi_stream.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"

#include "lwip/inet.h"
#include "lwip/sockets.h"

#include "settings.h"

#define CSI_SUBCARRIERS       64
#define CSI_EXPECTED_BYTES    (CSI_SUBCARRIERS * 2)
#define CSI_RSSI_MIN_DBM      (-75)
#define TX_HEARTBEAT_PORT     3333
#define TX_HEARTBEAT_PERIOD_US 10000U
#define TX_BROADCAST_ADDRESS  "192.168.4.255"

#pragma pack(push, 1)
typedef struct {
    char magic[4];
    uint32_t seq;
    uint32_t ts_ms;
    int8_t rssi;
    uint8_t n;
    uint8_t amp[CSI_SUBCARRIERS];
} csi1_packet_t;
#pragma pack(pop)

typedef struct {
    int8_t raw[CSI_EXPECTED_BYTES];
    int8_t rssi;
    uint32_t timestamp_ms;
} csi_sample_t;

static const char *TAG = "csi_stream";
static QueueHandle_t s_sample_queue;
static uint8_t s_tx_bssid[6];
static uint32_t s_csi_sequence;
static portMUX_TYPE s_stats_mux = portMUX_INITIALIZER_UNLOCKED;
static csi_stream_stats_t s_stats;
static uint64_t s_heartbeat_jitter_sum;
static uint32_t s_heartbeat_jitter_samples;

/* Per-subcarrier temporal filter state; only send_task accesses these arrays. */
static bool s_filter_initialized[CSI_SUBCARRIERS];
static float s_ema[CSI_SUBCARRIERS];
static float s_deadband_reference[CSI_SUBCARRIERS];
static bool s_deadband_initialized[CSI_SUBCARRIERS];
static float s_last_accepted[CSI_SUBCARRIERS];
static float s_pending_spike[CSI_SUBCARRIERS];
static bool s_has_pending_spike[CSI_SUBCARRIERS];

uint32_t csi_stream_sent(void)
{
    csi_stream_stats_t snapshot;
    csi_stream_get_stats(&snapshot);
    return snapshot.csi_sent;
}

void csi_stream_get_stats(csi_stream_stats_t *out)
{
    if (out == NULL) return;
    portENTER_CRITICAL(&s_stats_mux);
    *out = s_stats;
    portEXIT_CRITICAL(&s_stats_mux);
}

static void stat_increment(uint32_t *field)
{
    portENTER_CRITICAL(&s_stats_mux);
    (*field)++;
    portEXIT_CRITICAL(&s_stats_mux);
}

static bool is_null_subcarrier(int index)
{
    return (index >= 0 && index <= 5) ||
           index == 32 ||
           (index >= 59 && index <= 63);
}

/*
 * Wi-Fi callback stays short:
 * verify BSSID/length/RSSI and copy CSI bytes to a queue.
 * DSP and socket operations happen in the worker task.
 */
static void csi_rx_callback(void *context, wifi_csi_info_t *info)
{
    (void)context;

    if (info == NULL || info->buf == NULL) return;

    if (memcmp(info->mac, s_tx_bssid, sizeof(s_tx_bssid)) != 0) {
        stat_increment(&s_stats.csi_reject_mac);
        return;
    }

    if (info->len != CSI_EXPECTED_BYTES) {
        stat_increment(&s_stats.csi_reject_len);
        return;
    }

    if (info->rx_ctrl.rssi <= CSI_RSSI_MIN_DBM) {
        stat_increment(&s_stats.csi_reject_rssi);
        return;
    }

    csi_sample_t sample = {0};
    memcpy(sample.raw, info->buf, CSI_EXPECTED_BYTES);
    sample.rssi = info->rx_ctrl.rssi;
    sample.timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000U);

    stat_increment(&s_stats.csi_rx);
    if (xQueueSend(s_sample_queue, &sample, 0) != pdTRUE) {
        stat_increment(&s_stats.csi_queue_drops);
    }
}

/* Map only the permitted non-null subcarriers to their amplitude features. */
static void dsp_amplitude_filter(const csi_sample_t *sample,
                                 uint8_t output[CSI_SUBCARRIERS])
{
    const settings_t *cfg = settings_get();
    const float alpha = cfg->csi_ema_alpha;
    const float jump_threshold = (float)cfg->csi_jump_threshold;

    for (int i = 0; i < CSI_SUBCARRIERS; i++) {
        if (is_null_subcarrier(i)) {
            output[i] = 0;
            s_filter_initialized[i] = false;
            s_ema[i] = 0.0f;
            s_deadband_reference[i] = 0.0f;
            s_deadband_initialized[i] = false;
            s_last_accepted[i] = 0.0f;
            s_has_pending_spike[i] = false;
            continue;
        }

        /* ESP-IDF CSI LLTF stores each complex tone as imag, real int8 pair. */
        const float imag = (float)sample->raw[2 * i];
        const float real = (float)sample->raw[2 * i + 1];
        const float amplitude = sqrtf((real * real) + (imag * imag));

        if (!s_filter_initialized[i]) {
            s_filter_initialized[i] = true;
            s_last_accepted[i] = amplitude;
            s_ema[i] = amplitude;
            s_deadband_reference[i] = amplitude;
            s_deadband_initialized[i] = true;
            s_has_pending_spike[i] = false;
        } else {
            float accepted = amplitude;
            const float delta = fabsf(amplitude - s_last_accepted[i]);

            if (jump_threshold > 0.0f && delta > jump_threshold) {
                /*
                 * Reject a one-frame jump. If the next frame confirms a
                 * similar new level, accept it as a real sustained change.
                 */
                const float confirm_tolerance = fmaxf(4.0f, jump_threshold * 0.25f);
                if (s_has_pending_spike[i] &&
                    fabsf(amplitude - s_pending_spike[i]) <= confirm_tolerance) {
                    s_has_pending_spike[i] = false;
                } else {
                    s_pending_spike[i] = amplitude;
                    s_has_pending_spike[i] = true;
                    accepted = s_last_accepted[i];
                    stat_increment(&s_stats.csi_spike_holds);
                }
            } else {
                s_has_pending_spike[i] = false;
            }

            s_last_accepted[i] = accepted;
            s_ema[i] = alpha * accepted + (1.0f - alpha) * s_ema[i];
        }

        /* Tiny deadband <= 2 amplitude units; default is 0.5. */
        const float deadband = cfg->csi_deadband;
        if (!s_deadband_initialized[i] || deadband <= 0.0f ||
            fabsf(s_ema[i] - s_deadband_reference[i]) >= deadband) {
            s_deadband_reference[i] = s_ema[i];
            s_deadband_initialized[i] = true;
        }

        long quantized = lrintf(s_deadband_reference[i]);
        if (quantized < 0) quantized = 0;
        if (quantized > 255) quantized = 255;
        output[i] = (uint8_t)quantized;
    }
}

/* Count gaps in TX's monotonically increasing 4-byte heartbeat sequence. */
static void heartbeat_monitor_task(void *arg)
{
    (void)arg;

    uint32_t previous_sequence = 0;
    int64_t previous_time_us = 0;
    bool first_packet = true;

    for (;;) {
        const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
        if (sock < 0) {
            ESP_LOGE(TAG, "heartbeat monitor socket failed");
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        int reuse = 1;
        (void)setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        const struct sockaddr_in local = {
            .sin_family = AF_INET,
            .sin_port = htons(TX_HEARTBEAT_PORT),
            .sin_addr.s_addr = htonl(INADDR_ANY),
        };

        if (bind(sock, (const struct sockaddr *)&local, sizeof(local)) != 0) {
            ESP_LOGE(TAG, "heartbeat monitor bind failed");
            close(sock);
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        ESP_LOGI(TAG, "listening for TX heartbeat on UDP/%u",
                 (unsigned)TX_HEARTBEAT_PORT);

        for (;;) {
            uint32_t network_sequence = 0;
            const int received = recvfrom(sock, &network_sequence,
                                          sizeof(network_sequence), 0,
                                          NULL, NULL);
            if (received != (int)sizeof(network_sequence)) continue;

            const uint32_t sequence = ntohl(network_sequence);
            const int64_t now_us = esp_timer_get_time();

            if (!first_packet) {
                const uint32_t delta = sequence - previous_sequence;
                if (delta > 1U && delta < 0x80000000U) {
                    portENTER_CRITICAL(&s_stats_mux);
                    s_stats.heartbeat_lost += delta - 1U;
                    portEXIT_CRITICAL(&s_stats_mux);
                }

                const uint32_t interval_us =
                    (uint32_t)(now_us - previous_time_us);
                const uint32_t jitter_us =
                    (uint32_t)llabs((int64_t)interval_us -
                                    (int64_t)TX_HEARTBEAT_PERIOD_US);

                portENTER_CRITICAL(&s_stats_mux);
                s_stats.heartbeat_interval_us = interval_us;
                s_heartbeat_jitter_sum += jitter_us;
                s_heartbeat_jitter_samples++;
                s_stats.heartbeat_jitter_avg_us =
                    (float)s_heartbeat_jitter_sum /
                    (float)s_heartbeat_jitter_samples;
                if (jitter_us > s_stats.heartbeat_jitter_max_us) {
                    s_stats.heartbeat_jitter_max_us = jitter_us;
                }
                s_stats.heartbeat_rx++;
                portEXIT_CRITICAL(&s_stats_mux);
            } else {
                first_packet = false;
                portENTER_CRITICAL(&s_stats_mux);
                s_stats.heartbeat_rx++;
                portEXIT_CRITICAL(&s_stats_mux);
            }

            previous_sequence = sequence;
            previous_time_us = now_us;
        }

        close(sock);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

static bool make_csi_destination(struct sockaddr_in *destination)
{
    const settings_t *cfg = settings_get();

    memset(destination, 0, sizeof(*destination));
    destination->sin_family = AF_INET;
    destination->sin_port = htons(cfg->stream_port ? cfg->stream_port : 5005);

    if (!cfg->dedicated_mode && cfg->stream_host[0] != '\0' &&
        inet_pton(AF_INET, cfg->stream_host, &destination->sin_addr) == 1) {
        ESP_LOGI(TAG, "CSI1 destination %s:%u",
                 cfg->stream_host, (unsigned)ntohs(destination->sin_port));
        return true;
    }

    if (cfg->dedicated_mode) {
        destination->sin_addr.s_addr = inet_addr(TX_BROADCAST_ADDRESS);
        ESP_LOGI(TAG, "CSI1 broadcast %s:%u",
                 TX_BROADCAST_ADDRESS,
                 (unsigned)ntohs(destination->sin_port));
    } else {
        destination->sin_addr.s_addr = htonl(INADDR_BROADCAST);
        ESP_LOGI(TAG, "CSI1 subnet broadcast port %u",
                 (unsigned)ntohs(destination->sin_port));
    }
    return true;
}

static void csi_send_task(void *arg)
{
    (void)arg;

    int sock = -1;
    struct sockaddr_in destination;
    make_csi_destination(&destination);

    csi_sample_t sample;
    csi1_packet_t packet = {0};
    memcpy(packet.magic, "CSI1", 4);
    packet.n = CSI_SUBCARRIERS;

    for (;;) {
        if (sock < 0) {
            sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
            if (sock < 0) {
                ESP_LOGE(TAG, "CSI UDP socket creation failed");
                vTaskDelay(pdMS_TO_TICKS(500));
                continue;
            }

            int broadcast = 1;
            (void)setsockopt(sock, SOL_SOCKET, SO_BROADCAST,
                             &broadcast, sizeof(broadcast));
        }

        if (xQueueReceive(s_sample_queue, &sample, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        packet.seq = s_csi_sequence++;
        packet.ts_ms = sample.timestamp_ms;
        packet.rssi = sample.rssi;
        dsp_amplitude_filter(&sample, packet.amp);

        const int sent = sendto(sock, &packet, sizeof(packet), 0,
                                (const struct sockaddr *)&destination,
                                sizeof(destination));
        if (sent == (int)sizeof(packet)) {
            stat_increment(&s_stats.csi_sent);
        } else {
            stat_increment(&s_stats.csi_send_errors);
            close(sock);
            sock = -1;
        }
    }
}

esp_err_t csi_stream_start(void)
{
    wifi_ap_record_t ap = {0};
    esp_err_t err = esp_wifi_sta_get_ap_info(&ap);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Analyzer is not associated with an AP");
        return err;
    }

    memcpy(s_tx_bssid, ap.bssid, sizeof(s_tx_bssid));
    ESP_LOGI(TAG, "CSI paired AP BSSID=%02x:%02x:%02x:%02x:%02x:%02x channel=%u RSSI=%d",
             s_tx_bssid[0], s_tx_bssid[1], s_tx_bssid[2],
             s_tx_bssid[3], s_tx_bssid[4], s_tx_bssid[5],
             ap.primary, ap.rssi);

    if (s_sample_queue != NULL) return ESP_ERR_INVALID_STATE;
    s_sample_queue = xQueueCreate(64, sizeof(csi_sample_t));
    if (s_sample_queue == NULL) return ESP_ERR_NO_MEM;

    if (xTaskCreate(csi_send_task, "csi_send", 4096, NULL, 6, NULL) != pdPASS) {
        vQueueDelete(s_sample_queue);
        s_sample_queue = NULL;
        return ESP_ERR_NO_MEM;
    }

    if (xTaskCreate(heartbeat_monitor_task, "tx_heartbeat_rx",
                    3072, NULL, 4, NULL) != pdPASS) {
        ESP_LOGE(TAG, "could not create heartbeat monitor task");
        return ESP_ERR_NO_MEM;
    }

    const wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = false,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
        .dump_ack_en = false,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_rx_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG,
             "CSI active: exactly 128 raw bytes, RSSI > -75 dBm, null mask 0..5/32/59..63");
    ESP_LOGI(TAG, "DSP: EMA alpha=%.2f, spike threshold=%u, deadband=%.2f amplitude units",
             settings_get()->csi_ema_alpha,
             settings_get()->csi_jump_threshold,
             settings_get()->csi_deadband);
    return ESP_OK;
}
