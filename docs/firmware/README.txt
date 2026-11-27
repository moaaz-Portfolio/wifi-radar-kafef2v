This folder is populated by .github/workflows/bulid.yml.

analyzer_rx_merged.bin      -> complete first-install image for the Analyzer RX ESP32-S3 (offset 0)
transmitter_tx_merged.bin   -> complete first-install image for the Transmitter TX ESP32-S3 (offset 0)
esp32s3-wifi-radar.bin      -> Analyzer RX application image for OTA updates
radar-s3-merged.bin         -> compatibility copy of analyzer_rx_merged.bin

Both merged images include:
- bootloader at 0x0000
- partition table at 0xA000
- application at 0x20000

Install TX first, then RX. The laptop joins WiFi-Radar-Net (password 12345678).
