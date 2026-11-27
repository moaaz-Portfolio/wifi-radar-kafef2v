#pragma once
#include <stdint.h>
#include "esp_err.h"

typedef struct {
    uint32_t csi_rx;
    uint32_t csi_sent;
    uint32_t csi_send_errors;
    uint32_t csi_queue_drops;
    uint32_t csi_reject_mac;
    uint32_t csi_reject_len;
    uint32_t csi_reject_rssi;
    uint32_t csi_spike_holds;
    uint32_t heartbeat_rx;
    uint32_t heartbeat_lost;
    uint32_t heartbeat_interval_us;
    float heartbeat_jitter_avg_us;
    uint32_t heartbeat_jitter_max_us;
} csi_stream_stats_t;

/* Dedicated CSI link: captures the paired AP only and emits the athena CSI1 protocol. */
esp_err_t csi_stream_start(void);
uint32_t csi_stream_sent(void);
void csi_stream_get_stats(csi_stream_stats_t *out);
