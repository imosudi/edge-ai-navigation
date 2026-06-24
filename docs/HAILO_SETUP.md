# Hailo SDK Installation Guide
# Edge AI Navigation System - docs/HAILO_SETUP.md

## Overview

The Hailo-8L accelerator requires two software components:

| Component | Installed On | Purpose |
|-----------|-------------|---------|
| HailoRT runtime + PCIe driver | Raspberry Pi 5 | Hardware communication |
| Hailo Model Zoo + compiler    | x86 Linux host  | .hef model compilation |

---

## Part 1 - Raspberry Pi 5: Runtime Installation

### 1.1 Download HailoRT

Register at the Hailo Developer Zone:
> https://hailo.ai/developer-zone/software-downloads/

Download for **aarch64 / ARM64**:
- `hailort_4.18.x_arm64.deb`  - PCIe driver + runtime
- `hailort-4.18.x-cp311-cp311-linux_aarch64.whl`  - Python SDK

### 1.2 Install PCIe driver and runtime

```bash
# Install the runtime package
sudo dpkg -i hailort_4.18.x_arm64.deb

# Load the PCIe driver
sudo modprobe hailo_pci

# Verify device is detected
ls -la /dev/hailo0
# Expected: crw-rw---- 1 root video ... /dev/hailo0

# Make driver load on boot
echo "hailo_pci" | sudo tee /etc/modules-load.d/hailo.conf
```

### 1.3 Install Python SDK into virtualenv

```bash
cd /opt/edge-ai-navigation
sudo -u edgeai venv/bin/pip install hailort-4.18.x-cp311-cp311-linux_aarch64.whl
```

### 1.4 Verify Hailo installation

```bash
sudo -u edgeai /opt/edge-ai-navigation/venv/bin/python -c "
from hailo_platform import VDevice
vd = VDevice()
info = vd.get_target_information()
print('Hailo-8L detected:', info)
vd.release()
"
```

Expected output:
```
Hailo-8L detected: {'arch': 'hailo8l', 'fw_version': '4.18.x', ...}
```

### 1.5 PCIe Gen 3 configuration (performance)

Edit `/boot/firmware/config.txt`:
```
# Hailo HAT+ PCIe Gen 3 (doubles bandwidth vs Gen 2)
dtparam=pciex1_gen=3
```
Reboot after editing.

---

## Part 2 - x86 Host: Model Compilation

The `.hef` compilation process must run on an **x86 Linux machine** (not the Pi).

### 2.1 Install Hailo Model Zoo

```bash
# Python 3.8–3.11 on x86 Linux
pip install hailo-model-zoo

# Verify
hailomz --help
```

### 2.2 Prepare Calibration Dataset
The compiler needs a small calibration dataset (COCO subset). See [MODEL_COMPILE.md](MODEL_COMPILE.md) for how to set up the dataset quickly using `wget` and a subset of validation images.

### 2.3 Compile YOLOv8n → .hef
Export the PyTorch weights to ONNX (with `opset=11`), and run the compiler. 

> [!NOTE]
> You must explicitly specify the 6 output convolution end nodes via `--end-node-names` to bypass the `depth_to_space` allocator exception during parsing. For full details and required python package setup, see the comprehensive [MODEL_COMPILE.md](MODEL_COMPILE.md).

```bash
# Export from ultralytics to ONNX
python -c "
from ultralytics import YOLO
yolo = YOLO('yolov8n.pt')
yolo.export(format='onnx', imgsz=640, opset=11, simplify=True)
"

# Compile to .hef for Hailo-8L / Hailo-15L
hailomz compile yolov8n \
    --ckpt yolov8n.onnx \
    --hw-arch hailo8l \
    --calib-path datasets/coco_calib/ \
    --performance \
    --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv /model.22/cv2.1/cv2.1.2/Conv /model.22/cv3.1/cv3.1.2/Conv /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv
```

Compilation takes approximately **15–45 minutes** on a modern x86 host.

### 2.4 Copy .hef to Raspberry Pi
```bash
scp yolov8n.hef pi@raspberrypi.local:/opt/edge-ai-navigation/models/
```

### 2.5 Update configuration
Edit `config/settings.yaml`:
```yaml
inference:
  model_path: "models/yolov8n.hef"
  device: "hailo"
```

Restart the service:
```bash
sudo systemctl restart edge-ai-navigation
```

---

## Part 3 - Upgrading to YOLOv8s

YOLOv8s offers higher accuracy at the cost of ~2× inference latency.

```bash
# Export YOLOv8s
python -c "
from ultralytics import YOLO
YOLO('yolov8s.pt').export(format='onnx', imgsz=640, opset=11)
"

# Compile for Hailo-8L
hailomz compile yolov8s \
    --ckpt yolov8s.onnx \
    --hw-arch hailo8l \
    --calib-path datasets/coco_calib/ \
    --performance \
    --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv /model.22/cv2.1/cv2.1.2/Conv /model.22/cv3.1/cv3.1.2/Conv /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv
```

Update `config/settings.yaml`:
```yaml
inference:
  model_name: "yolov8s"
  model_path: "models/yolov8s.hef"
```

Expected performance on Hailo-8L:

| Model    | Input  | Hailo-8L FPS | CPU FPS (Pi 5) |
|----------|--------|-------------|----------------|
| YOLOv8n  | 640×640 | ~18–22 fps  | ~3–4 fps       |
| YOLOv8s  | 640×640 | ~10–14 fps  | ~1–2 fps       |

---

## Troubleshooting

### `/dev/hailo0` not found

```bash
# Check PCIe is detected by kernel
lspci | grep -i hailo
# Should show: Hailo Technologies Ltd. Hailo-8 AI Processor

# Reload driver
sudo modprobe -r hailo_pci && sudo modprobe hailo_pci

# Check dmesg
dmesg | grep -i hailo
```

### HailoRT version mismatch

The Python wheel version must match the deb package version exactly.

```bash
hailortcli fw-control identify   # Shows firmware version
python -c "import hailo_platform; print(hailo_platform.__version__)"
```

### Thermal throttling

The Hailo-8L can generate heat. Monitor:
```bash
watch -n1 "vcgencmd measure_temp && hailortcli monitor"
```

Add a heatsink to the HAT+ and ensure adequate airflow.
