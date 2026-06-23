#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/download_model.sh
# Download YOLOv8 weights and optionally compile to .hef for Hailo-8L
#
# Usage:
#   bash scripts/download_model.sh [yolov8n|yolov8s]   # default: yolov8n
#
# Compilation requires hailo_model_zoo installed on a Linux x86 host machine
# (not the Pi itself). See docs/MODEL_COMPILE.md for the full workflow.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="${1:-yolov8n}"
MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
VENV="${MODELS_DIR}/../venv/bin/python"

[[ -f "${VENV}" ]] || VENV="python3"

mkdir -p "${MODELS_DIR}"
cd "${MODELS_DIR}"

echo "=== Downloading ${MODEL} (PyTorch weights) ==="
"${VENV}" -c "
from ultralytics import YOLO
import shutil, pathlib

model_name = '${MODEL}.pt'
yolo = YOLO(model_name)

# Export to ONNX (required for Hailo Model Zoo compilation)
print('Exporting to ONNX...')
onnx_path = yolo.export(format='onnx', imgsz=640, opset=11, simplify=True)
print(f'ONNX saved: {onnx_path}')

# Move to models dir
src = pathlib.Path(onnx_path)
dst = pathlib.Path('${MODELS_DIR}') / src.name
shutil.move(str(src), str(dst))
print(f'Moved to: {dst}')
"

echo ""
echo "=== ONNX export complete ==="
echo ""
echo "To compile to .hef for Hailo-8L, run on an x86 Linux host:"
echo ""
echo "  # Install Hailo Model Zoo (x86 only)"
echo "  pip install hailo-model-zoo"
echo ""
echo "  # Compile (takes 10–30 minutes)"
echo "  hailomz compile yolov8n \\"
echo "      --ckpt models/${MODEL}.onnx \\"
echo "      --hw-arch hailo8l \\"
echo "      --calib-path datasets/coco_calib/ \\"
echo "      --classes 80 \\"
echo "      --performance"
echo ""
echo "  # Copy .hef to Pi"
echo "  scp ${MODEL}.hef pi@raspberrypi.local:/opt/edge-ai-navigation/models/"
echo ""
echo "Then update config/settings.yaml:"
echo "  inference:"
echo "    model_path: models/${MODEL}.hef"
echo "    device: hailo"
