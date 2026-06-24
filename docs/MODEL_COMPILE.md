# Hailo AI Model Compilation Guide (ONNX → HEF)

This guide describes how to compile PyTorch/ONNX YOLO models into the **Hailo Executable Format (`.hef`)** required for hardware acceleration on the Raspberry Pi 5 + Hailo AI HAT+ (using the Hailo-8L / Hailo-15L arch).

> [!IMPORTANT]
> Model compilation (quantization and partitioning) is computationally heavy and requires the **Hailo Dataflow Compiler (DFC)**. Since the DFC is proprietary and only built for x86_64, you must run this compilation process on an **x86_64 Linux host machine** (Ubuntu 22.04 or 24.04 LTS is recommended).

---

## 1. Prerequisites & Environment Setup (x86_64 Host)

Before compiling the model, you need to prepare the x86_64 host environment with Python 3.8–3.12, the correct system libraries, and the Hailo SDK components.

### 1.1 Install System Dependencies
Pillow and the DFC require specific image processing and graph libraries to build and run correctly. Install them using your system package manager:
```bash
sudo apt-get update
sudo apt-get install -y \
  graphviz \
  libgraphviz-dev \
  pkg-config \
  libjpeg-dev \
  zlib1g-dev \
  libpng-dev \
  libtiff-dev \
  libwebp-dev \
  liblcms2-dev \
  libfreetype6-dev
```

### 1.2 Set up the Python Virtual Environment
Initialize a virtual environment and install the Hailo Dataflow Compiler wheel along with the required version of the Hailo Model Zoo:

1. Register and download the Dataflow Compiler wheel (`hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl` or similar) from the [Hailo Developer Zone](https://hailo.ai/developer-zone/).
2. Create the virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip wheel setuptools

   # Install the compiler wheel
   pip install hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl

   # Install Model Zoo with PyPI simple index constraint to avoid Pillow version conflicts
   pip install git+https://github.com/hailo-ai/hailo_model_zoo.git \
     --extra-index-url https://pypi.org/simple \
     "Pillow!=9.3.0"

   # Install Ultralytics for PyTorch model loading and ONNX exporting
   pip install ultralytics
   ```

---

## 2. Compilation Workflow

Follow these steps on your x86_64 host machine to download, export, calibrate, and compile the model.

### Step 1: Export the YOLO Model to ONNX
We load the PyTorch weights and export them to an optimized ONNX model. **You must specify `opset=11` and `imgsz=640`** to match the Hailo Model Zoo parser expectations:

```python
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads pretrained YOLOv8n weights
model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=640)
"
# Move the exported ONNX model to the models directory
mkdir -p models
mv yolov8n.onnx models/yolov8n.onnx
```

### Step 2: Prepare the Calibration Dataset
Quantization translates 32-bit floating-point weights into 8-bit integers (`INT8`). To calibrate without losing accuracy, the compiler needs a representative subset of validation images (64–128 images):

```bash
# Create directory
mkdir -p datasets/coco_calib

# Download subset of COCO val2017 dataset
wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
unzip val2017.zip -d val2017

# Copy the first 128 images into datasets/coco_calib/ for calibration
ls val2017/val2017/ | head -128 | xargs -I{} cp val2017/val2017/{} datasets/coco_calib/

# Clean up temporary downloads
rm -rf val2017 val2017.zip
```

### Step 3: Setup NMS Post-Processing Config (Workaround)
The Hailo Model Zoo scripts look for post-processing configurations relative to the script directory using `../../postprocess_config/`. Due to directory layout differences in python package installations, this path lookup fails. Fix it by creating a symbolic link within the virtual environment:

```bash
ln -s ../postprocess_config venv/lib/python3.12/site-packages/hailo_model_zoo/cfg/alls/postprocess_config
```

### Step 4: Run the Hailo Compiler
To compile the model, run `hailomz compile`.

> [!WARNING]
> **Allocator Script Parser Error (`depth_to_space`)**
> During ONNX simplification, certain post-processing sub-graphs in the YOLOv8 head are converted into operations like `depth_to_space`. If you run compile without specifying output boundaries, the parser attempts to include these downstream layers. Because the post-processor expects raw convolutional layers at the output, it throws an `AllocatorScriptParserException: Error in the last layers of the model, expected conv but found LayerType.depth_to_space layer.`
> 
> **Solution:** You must explicitly pass the names of the **6 output convolution layers** (representing the bounding box and classification predictions for the three output strides) using the `--end-node-names` parameter.

Run the compiler with `--end-node-names`:

#### For Raspberry Pi 5 + Hailo AI HAT+ (hailo15l):
```bash
hailomz compile yolov8n \
    --ckpt models/yolov8n.onnx \
    --hw-arch hailo15l \
    --calib-path datasets/coco_calib/ \
    --performance \
    --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv /model.22/cv2.1/cv2.1.2/Conv /model.22/cv3.1/cv3.1.2/Conv /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv
```

#### For Hailo-8 (hailo8l):
```bash
hailomz compile yolov8n \
    --ckpt models/yolov8n.onnx \
    --hw-arch hailo8l \
    --calib-path datasets/coco_calib/ \
    --performance \
    --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv /model.22/cv2.1/cv2.1.2/Conv /model.22/cv3.1/cv3.1.2/Conv /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv
```

The compiled binary (`yolov8n.hef`) will be saved in your working directory.

---

## 3. Deploy to Raspberry Pi 5

Once compilation completes, transfer the `.hef` file to the target device and load it.

### Step 1: Transfer the `.hef` File
Copy the compiled HEF binary to the Pi:
```bash
mv yolov8n.hef models/
scp models/yolov8n.hef pi@raspberrypi.local:/opt/edge-ai-navigation/models/
```

### Step 2: Update runtime configuration
On the Raspberry Pi 5, edit `/opt/edge-ai-navigation/config/settings.yaml`:
```yaml
inference:
  model_name: "yolov8n"
  model_path: "models/yolov8n.hef"
  device: "hailo"
```

### Step 3: Restart the Service
```bash
sudo systemctl restart edge-ai-navigation
```
Verify the logs to ensure the device is running inference on the NPU:
```bash
sudo journalctl -u edge-ai-navigation -n 100 -f
```
