# Hailo-8L Model Compilation Guide (ONNX → HEF)

This guide describes how to compile PyTorch/ONNX YOLO models into the **Hailo Executable Format (`.hef`)** required for hardware acceleration on the Raspberry Pi 5 + Hailo AI HAT+.

> [!IMPORTANT]
> The model compilation process is computationally heavy. While `hailo-model-zoo` is installed as part of the project dependencies (on both x86_64 and ARM64/Raspberry Pi 5), actual ONNX-to-HEF compilation requires the **Hailo Dataflow Compiler (DFC)**. Since DFC is proprietary and only compiled for x86_64 architectures, compilation runs on an x86_64 Linux host or inside a virtualized container environment.

---

## 1. Prerequisites & Environment Setup

1. **System Requirements**: Ubuntu 22.04/24.04 LTS (recommended) x86_64 machine, or Raspberry Pi 5 (ARM64) for runtime/local configurations.
2. **HailoRT & PCIe Drivers**: Install the matching HailoRT version (matching the target Pi package, e.g., v4.18+).
3. **Python Virtual Environment & Dependencies**:
   Install the project dependencies, which automatically pull in `hailo-model-zoo` from the official git repository:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip wheel setuptools
   pip install -r requirements.txt
   ```

---

## 2. Compilation Workflow

### Step 1: Export ONNX model from Raspberry Pi 5 or Host
First, download the PyTorch weights and export them to an optimized ONNX format. You can run this command directly on your Pi or host machine:
```bash
bash scripts/download_model.sh yolov8n
# Or for YOLOv8s:
# bash scripts/download_model.sh yolov8s
```
This generates `models/yolov8n.onnx` (or `models/yolov8s.onnx`).

### Step 2: Prepare Calibration Dataset (on x86_64 Host)
Quantization requires a small, representative dataset (usually 100–1000 images) to calibrate the model's weights to 8-bit integers without losing accuracy:
1. Download a subset of the COCO validation dataset.
2. Store the calibration images in `datasets/coco_calib/`.

### Step 3: Run the Hailo Compiler (on x86_64 Host)
Run the compilation task using the Hailo Model Zoo command-line interface. This step parses the ONNX model, matches it to the Hailo hardware constraints, quantizes the weights, and compiles the `.hef` binary.

#### For YOLOv8n:
```bash
hailomz compile yolov8n \
    --ckpt models/yolov8n.onnx \
    --hw-arch hailo8l \
    --calib-path datasets/coco_calib/ \
    --classes 80 \
    --performance
```

#### For YOLOv8s:
```bash
hailomz compile yolov8s \
    --ckpt models/yolov8s.onnx \
    --hw-arch hailo8l \
    --calib-path datasets/coco_calib/ \
    --classes 80 \
    --performance
```

The output file (`yolov8n.hef` or `yolov8s.hef`) will be saved in the current directory.

### Step 4: Transfer the compiled HEF to the Raspberry Pi 5
Copy the compiled HEF binary to the models directory on the Raspberry Pi:
```bash
scp yolov8n.hef pi@<pi-ip>:/opt/edge-ai-navigation/models/
```

### Step 5: Update Runtime Configuration
Edit `/opt/edge-ai-navigation/config/settings.yaml` on the Raspberry Pi 5 to use the new NPU model:
```yaml
inference:
  model_name: "yolov8n"          # Or "yolov8s"
  model_path: "models/yolov8n.hef" # Path to target HEF file
  device: "hailo"                # Set to "hailo" (or "auto") to enable NPU
```

Restart the systemd service to apply the change:
```bash
sudo systemctl restart edge-ai-navigation
```
