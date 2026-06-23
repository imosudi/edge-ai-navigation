# Performance Optimisation Guide
# Edge AI Navigation System - docs/PERFORMANCE.md

## Baseline Performance Targets (Raspberry Pi 5 + Hailo-8L)

| Metric              | Target      | Acceptable  | Poor       |
|--------------------|-------------|-------------|------------|
| Camera FPS          | 25–30       | 15–24       | < 15       |
| Inference FPS       | 15–22       | 10–14       | < 10       |
| LiDAR scan rate     | 10 Hz       | 5–9 Hz      | < 5 Hz     |
| Fusion latency      | < 50 ms     | 50–100 ms   | > 100 ms   |
| End-to-end latency  | < 120 ms    | 120–250 ms  | > 250 ms   |
| CPU utilisation     | 40–65%      | 65–80%      | > 80%      |
| Memory usage        | 400–700 MB  | 700 MB–1 GB | > 1 GB     |
| CPU temperature     | < 60°C      | 60–70°C     | > 75°C     |

---

## Layer 1 - Hardware Configuration

### PCIe Gen 3 (critical for Hailo throughput)

```bash
# /boot/firmware/config.txt
dtparam=pciex1_gen=3
```

Verify active mode:
```bash
sudo lspci -vv | grep -A2 "Hailo"
# LnkSta: Speed 8GT/s (ok), Width x1 (ok)  ← Gen 3 confirmed
```

### CPU frequency governor

```bash
# Performance mode (eliminates frequency ramp-up latency)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
# Should be close to max_freq (2400000 on Pi 5)
```

### GPU memory allocation

For camera processing without a desktop environment:
```
gpu_mem=128   # /boot/firmware/config.txt
```

### USB buffer size (for reliable LiDAR)

```
# /boot/firmware/cmdline.txt (append)
usbcore.usbfs_memory_mb=256
```

---

## Layer 2 - Application Configuration

### Inference pipeline

```yaml
# config/settings.yaml - high-performance profile
inference:
  model_name: "yolov8n"          # Smallest/fastest model
  input_width: 640               # Native YOLO input
  input_height: 640
  confidence_threshold: 0.45    # Filter weak detections early
  queue_maxsize: 2               # Small queue = lower latency, more drops OK
  draw_overlays: true            # Set false if CPU is the bottleneck

camera:
  width: 1280
  height: 720
  fps: 30
  jpeg_quality: 75               # Reduce for lower network bandwidth
```

### For constrained environments (< 4 GB RAM or thermal throttling)

```yaml
camera:
  width: 640
  height: 480
  fps: 15
  jpeg_quality: 65

inference:
  input_width: 416
  input_height: 416
  queue_maxsize: 2

lidar:
  scan_frequency_hz: 5.0         # Half scan rate saves CPU on scan processing

telemetry:
  interval_seconds: 2.0          # Less frequent telemetry updates
```

---

## Layer 3 - OpenCV Optimisations

### Use NEON SIMD (enabled by default on ARM64)

```python
# Verify optimisations are compiled in
import cv2
print(cv2.getBuildInformation())
# Should show: NEON: YES, USE_NEON: YES
```

### Disable unnecessary colour conversions

The Hailo engine expects RGB input. If your camera outputs BGR, one conversion is
needed. Avoid double-converting:

```python
# BAD: BGR → RGB → BGR → RGB
# GOOD: BGR → RGB (once, at input to inference)
rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
```

### Pre-allocate output buffers

```python
# Pre-allocate JPEG output buffer to avoid repeated malloc
encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
buf = np.empty((1280 * 720 * 3,), dtype=np.uint8)
```

### Reduce overlay rendering cost

If FPS drops below target, disable bounding box rendering:
```yaml
inference:
  draw_overlays: false  # Raw frames - zero annotation overhead
```

---

## Layer 4 - Async Pipeline Tuning

### Frame drop policy

The system intentionally drops frames when the inference queue is full.
This prevents memory accumulation. The queue depth controls the trade-off:

| `queue_maxsize` | Latency  | Memory | Behaviour on slow inference |
|----------------|----------|--------|-----------------------------|
| 1              | Lowest   | Minimal| Aggressive drop             |
| 2–4 (default)  | Low      | Low    | Balanced                    |
| 8–16           | Higher   | Higher | Smoothed, fewer drops       |

### asyncio event loop

The system uses a single uvicorn worker with one asyncio event loop.
Keep all I/O operations async - never call blocking code on the event loop:

```python
# BAD: blocks event loop
result = slow_sync_function()

# GOOD: runs in thread pool, event loop stays free
result = await asyncio.get_event_loop().run_in_executor(None, slow_sync_function)
```

### WebSocket client impact

Each WebSocket connection adds overhead per broadcast. Limit concurrent
dashboard clients for maximum inference performance:

Typical overhead per client: ~1–2 ms per JPEG frame broadcast at 1280×720×80%.

---

## Layer 5 - System-Level Tuning

### Disable unnecessary services

```bash
# Disable Bluetooth (if not needed)
sudo systemctl disable bluetooth

# Disable Wi-Fi (use Ethernet for reliability)
sudo rfkill block wifi

# Disable swap (prevent latency spikes)
sudo swapoff -a
sudo systemctl mask dphys-swapfile
```

### Memory limits

```bash
# /etc/systemd/system/edge-ai-navigation.service
MemoryMax=2G
MemorySwapMax=0    # No swap for this service
```

### IRQ affinity (advanced)

Pin the Hailo PCIe IRQ to CPU core 3 (isolated from camera/LiDAR):
```bash
# Find Hailo IRQ number
cat /proc/interrupts | grep hailo

# Pin to core 3 (0-indexed)
echo 8 | sudo tee /proc/irq/<IRQ_NUMBER>/smp_affinity
# 8 = binary 1000 = core 3
```

### Real-time scheduling (experimental)

```bash
# Give the inference process real-time priority
sudo chrt -f -p 50 $(pgrep -f uvicorn)
# Note: requires CONFIG_PREEMPT_RT kernel
```

---

## Profiling

### Python profiling

```bash
# Profile startup overhead
sudo -u edgeai /opt/edge-ai-navigation/venv/bin/python -m cProfile \
    -o logs/profile.out -m uvicorn app.main:app &
sleep 30 && kill %1

python3 -c "
import pstats, io
p = pstats.Stats('logs/profile.out')
p.sort_stats('cumulative')
p.print_stats(20)
"
```

### Frame timing instrumentation

Add `time.perf_counter()` measurements to the pipeline:
```python
t0 = time.perf_counter()
detections = await hailo.infer(frame)
t1 = time.perf_counter()
logger.debug("Inference: %.1f ms", (t1 - t0) * 1000)
```

### Memory profiling

```bash
pip install memory-profiler
sudo -u edgeai venv/bin/python -m memory_profiler app/main.py
```

---

## Benchmark Script

```bash
# Run a 60-second performance benchmark and report
bash scripts/benchmark.sh

# Expected output:
# Camera FPS: 29.8
# Inference FPS: 19.2
# LiDAR FPS: 9.9
# CPU mean: 58.3%, peak: 74.1%
# Memory mean: 512 MB
# Temperature peak: 63.2°C
# Total inferences: 1152
# Hailo avg latency: 48.3 ms
```
