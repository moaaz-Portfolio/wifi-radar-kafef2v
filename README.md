# WiFi Radar — Dedicated Two-ESP32-S3 CSI Sensing

WiFi Radar is a research prototype for indoor presence and activity sensing using Wi-Fi Channel State Information (CSI). The dedicated TX/RX build does not need a home router, internet connection, camera, or cloud service during local operation.

> Research boundary: presence, activity, and possible fall indications are experimental estimates. This is not a medical device or an emergency/life-safety system.

## Dedicated network architecture

The default ESP32-S3 firmware now uses two boards:

1. **Transmitter TX** (folder tx_node/) creates the local Wi-Fi network WiFi-Radar-Net (password 12345678) in SoftAP mode. It sends a four-byte UDP heartbeat to 192.168.4.255:3333 on a 10 ms timer schedule.
2. **Analyzer RX** (folder main/) joins that network as a station, filters CSI by the TX access point BSSID, applies the DSP preprocessing, and streams the 64-byte CSI feature vector to UDP port 5005.
3. **Laptop** joins WiFi-Radar-Net from the room doorway or another location where its Wi-Fi adapter can maintain the link. It runs the Python Factory or User Edition.
4. **Door sensing** remains on the Analyzer RX ESP32-S3: HC-SR04 on GPIO4/GPIO5 and active buzzer driver on GPIO6. Door telemetry is sent to UDP port 5006.

The TX SoftAP currently uses channel 1, HT20, fixed WIFI_PHY_RATE_MCS0_SGI, and an 11 dBm transmit-power cap. Channel 14 is deliberately not forced: it is Japan-specific 802.11b/DSSS spectrum and is incompatible with the requested HT MCS0-SGI mode. Follow local NTRA limits and verify EIRP including antenna gain.

The nominal 100 Hz value is a timer schedule target. Wi-Fi/FreeRTOS scheduling and RF retransmissions mean it is not a guarantee of zero packet jitter. BSSID filtering rejects other transmitters as CSI sources, but it cannot remove physical RF interference on the same channel.

## Flash the two boards

GitHub Actions builds two complete flash images:

- docs/firmware/transmitter_tx_merged.bin — flash this to the new TX board first.
- docs/firmware/analyzer_rx_merged.bin — flash this to the existing RX board second.

Open the GitHub Pages installer in Chrome or Edge on a computer and choose the matching ESP Web Tools button. Connect only one ESP32-S3 board by USB during each flashing operation. Each manifest uses the merged image at flash offset 0x0.

The merge layout is bootloader at 0x0000, partition table at 0xA000, and application at 0x20000.

## Run the Python processor

Connect the laptop to WiFi-Radar-Net, then from the repository:

    cd athena
    python -m pip install -r requirements.txt
    python -m athena --edition factory --host 0.0.0.0

Use Factory Edition to record independent takes for the approved classes, review the metrics, and train the model. The existing four-class configuration is empty_room, person_static, movement, and fall_detection.

After reviewing the independent test metrics, run:

    python -m athena --edition user --host 0.0.0.0

User Edition performs inference and room calibration; it does not record training data or retrain the model. The laptop is the local processing computer in this prototype. A dedicated low-power computer can replace it only after model memory, latency, temperature, and sustained operation have been measured.

## Data protocol

The Analyzer RX keeps the Python CSI1 packet layout unchanged:

- ASCII magic CSI1
- sequence number: little-endian uint32
- timestamp: little-endian uint32 milliseconds
- RSSI: signed int8
- feature count: uint8, exactly 64
- 64 feature bytes: uint8

Valid DSP zero is encoded as byte 128; null/guard subcarriers are encoded as byte 0. Door packets remain on UDP port 5006.

## Research validation

Record across multiple rooms, people, sessions, and days. Compare per-class precision/recall/F1, false alarms, misses, packet loss, and leave-one-room/person-out cross-validation. Do not treat the amplitude ratio as a guaranteed percentage of Bluetooth-noise removal, and do not infer breathing or falls from one signal feature alone.

See docs/DEDICATED_LINK_AR.md, docs/HARDWARE_HC-SR04_BUZZER_AR.md, docs/EDITION_WORKFLOW.md, and athena/EXPERIMENTS.md.

## License

Apache-2.0. The DNS server component is based on Espressif ESP-IDF examples (Unlicense/CC0).
