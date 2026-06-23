# Troubleshooting Guide
# Edge AI Navigation System — docs/TROUBLESHOOTING.md

## Quick Diagnostics

```bash
# Check service status
sudo systemctl status edge-ai-navigation

# Live logs (last 50 lines + follow)
sudo journalctl -u edge-ai-navigation -n 50 -f

# Check all hardware
bash scripts/check_hardware.sh

# Test REST API
curl -s http://localhost:8080/api/v1/status | python3 -m json.tool
```

---

## Camera Issues

### Camera not detected

```bash
# List V4L2 devices
v4l2-ctl --list-devices

# Test capture with libcamera
libcamera-hello --timeout 2000

# Check camera ribbon cable
vcgencmd get_camera
# Expected: supported=1 detected=1
```

**Fix**: Enable camera in raspi-config:
```bash
sudo raspi-config  → Interface Options → Camera → Enable
sudo reboot
```

### Low FPS from camera

Check the capture resolution is achievable at target FPS:

```bash
# Camera Module 3 supported modes
libcamera-hello --list-cameras
```

Camera Module 3 at 1280×720 supports up to **60 fps** on full sensor.
Reduce resolution if needed:
```yaml
# config/settings.yaml
camera:
  width: 640
  height: 480
  fps: 30
```

### JPEG encoding slow

```bash
# Check OpenCV build has JPEG hardware acceleration
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i jpeg
```

Enable TurboJPEG on Pi:
```bash
sudo apt install libturbojpeg0-dev
pip install PyTurboJPEG
```

---

## LiDAR Issues

### `/dev/ttyACM0` not found

```bash
# Check USB connection
lsusb | grep -i "15d1"   # Hokuyo VID
# Expected: Bus 001 Device 003: ID 15d1:0000 Hokuyo Data Flex for USB

# Check udev symlink
ls -la /dev/lidar
# Should exist if udev rules installed

# Check permissions
ls -la /dev/ttyACM0
# Expected: crw-rw---- 1 root dialout
```

**Fix**: Add user to dialout group:
```bash
sudo usermod -aG dialout edgeai
sudo udevadm trigger
```

### LiDAR scanning but all distances invalid

Verify SCIP 2.0 protocol:
```bash
# Raw serial test (disconnect from service first)
sudo systemctl stop edge-ai-navigation
python3 -c "
import serial, time
s = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
s.write(b'SCIP2.0\n')
time.sleep(0.1)
s.write(b'VV\n')
time.sleep(0.5)
print(s.read(500))
"
```

### LiDAR range limits

The URG-04LX minimum range is **60 mm**. Objects closer than 6 cm will be reported as invalid.

Check config:
```yaml
lidar:
  min_range_m: 0.06   # 60 mm minimum
  max_range_m: 5.5    # 5500 mm maximum
```

---

## Hailo / Inference Issues

### System falls back to CPU unexpectedly

```bash
# Check Hailo device
ls /dev/hailo*
hailortcli fw-control identify

# Check Python SDK
python3 -c "from hailo_platform import VDevice; print('OK')"
```

Check system log for the specific error:
```bash
sudo journalctl -u edge-ai-navigation | grep -i "hailo\|inference"
```

### Inference FPS lower than expected

Expected Hailo-8L inference rates:

| Model    | Expected FPS | Minimum Acceptable |
|----------|-------------|-------------------|
| YOLOv8n  | 18–22       | 12                |
| YOLOv8s  | 10–14       | 7                 |

Tuning steps:
1. Verify PCIe Gen 3: `grep pciex1 /boot/firmware/config.txt`
2. Check thermal throttling: `vcgencmd measure_temp`
3. Reduce input resolution in config: `input_width: 416`
4. Reduce camera resolution: `width: 640, height: 480`

### Model file not found / wrong format

```bash
ls -lh models/
# yolov8n.hef should be ~4 MB
# yolov8n.pt  should be ~6 MB (CPU fallback)

file models/yolov8n.hef
# Should show: data (not empty)
```

---

## WebSocket / Dashboard Issues

### Dashboard loads but video is static

```bash
# Test WebSocket manually
python3 -c "
import asyncio, websockets

async def test():
    async with websockets.connect('ws://localhost:8080/api/v1/ws/camera') as ws:
        data = await asyncio.wait_for(ws.recv(), timeout=5.0)
        print(f'Received {len(data)} bytes')

asyncio.run(test())
"
```

### High WebSocket latency

Check network path. For LAN use, disable network proxies.

Reduce JPEG quality for lower bandwidth:
```yaml
camera:
  jpeg_quality: 65   # Default: 80
```

### Dashboard not loading at all

```bash
# Check if server is up
curl -s http://localhost:8080/api/v1/status
# If 200 OK but dashboard 404:
ls dashboard/templates/index.html   # Must exist
ls dashboard/static/css/dashboard.css
```

---

## Performance Issues

### High CPU usage (>80%)

```bash
# Identify hot threads
sudo top -H -p $(pgrep -f uvicorn)

# Reduce inference load
# 1. Lower camera FPS
# 2. Increase inference queue maxsize (drops frames rather than blocking)
# 3. Ensure Hailo is being used (not CPU fallback)
```

### Memory leak symptoms

```bash
# Monitor process memory over time
watch -n5 "ps aux | grep uvicorn | grep -v grep | awk '{print \$6/1024 \" MB\"}'"
```

Expected steady-state memory: **300–600 MB** depending on model and resolution.

### High temperature (>75°C)

```bash
# Current temperature
vcgencmd measure_temp

# Throttling status
vcgencmd get_throttled
# 0x0 = no throttling; non-zero = throttled
```

Cooling solutions:
1. Add active cooling (Raspberry Pi Active Cooler)
2. Reduce camera FPS from 30 to 15
3. Set inference to YOLOv8n (smallest model)
4. Reduce `scan_frequency_hz` to 5

---

## Service Management

```bash
# Start / Stop / Restart
sudo systemctl start  edge-ai-navigation
sudo systemctl stop   edge-ai-navigation
sudo systemctl restart edge-ai-navigation

# Enable / Disable auto-start
sudo systemctl enable  edge-ai-navigation
sudo systemctl disable edge-ai-navigation

# View last 100 log lines
sudo journalctl -u edge-ai-navigation -n 100 --no-pager

# Clear old logs
sudo journalctl --vacuum-size=100M
```

---

## Resetting to Defaults

```bash
# Stop service
sudo systemctl stop edge-ai-navigation

# Reset config to defaults
cd /opt/edge-ai-navigation
sudo -u edgeai cp config/settings.yaml config/settings.yaml.bak
sudo -u edgeai python3 -c "
from config.config_loader import AppConfig
import yaml
cfg = AppConfig()
with open('config/settings.yaml', 'w') as f:
    yaml.dump(cfg.model_dump(), f, default_flow_style=False)
print('Config reset to defaults.')
"

# Clear logs
sudo -u edgeai rm -f logs/edge_ai_nav.log logs/snapshots/*

# Restart
sudo systemctl start edge-ai-navigation
```
