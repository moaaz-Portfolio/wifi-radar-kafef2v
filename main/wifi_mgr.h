#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

typedef enum { NETMODE_STA = 0, NETMODE_SETUP = 1 } netmode_t;

/* Connect to the dedicated TX SoftAP by default, or saved router in router mode. */
netmode_t wifi_mgr_start(void);
netmode_t wifi_mgr_mode(void);
const char *wifi_mgr_ip(void);
bool wifi_mgr_connected(void);
uint8_t wifi_mgr_channel(void);

/* JSON array text: [{"ssid":"..","rssi":-50,"open":false}, ...]. Valid until next call. */
const char *wifi_mgr_scan(bool refresh);

/* Called on every portal request so the idle-reboot timer is reset. */
void wifi_mgr_touch(void);
