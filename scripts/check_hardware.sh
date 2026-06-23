#!/usr/bin/env bash
# scripts/check_hardware.sh
# Quick hardware verification script - run before starting the service.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC}  $*"; }
fail() { echo -e "  ${RED}✗${NC}  $*"; FAILURES=$((FAILURES+1)); }
warn() { echo -e "  ${YELLOW}⚠${NC}  $*"; }
FAILURES=0

echo ""
echo "════════════════════════════════════════"
echo "  Edge AI Navigation - Hardware Check"
echo "════════════════════════════════════════"
echo ""

# ── Platform ──────────────────────────────────────────────────────────────────
echo "Platform:"
ARCH=$(uname -m)
if [[ "${ARCH}" == "aarch64" ]]; then
    ok "Architecture: aarch64 (ARM64)"
else
    warn "Architecture: ${ARCH} (expected aarch64)"
fi

# Pi 5 detection
if grep -q "Raspberry Pi 5" /proc/device-tree/model 2>/dev/null; then
    ok "Raspberry Pi 5 detected"
else
    warn "Not a Raspberry Pi 5 (or /proc/device-tree unavailable)"
fi

# ── Camera ────────────────────────────────────────────────────────────────────
echo ""
echo "Camera:"
if ls /dev/video0 &>/dev/null; then
    ok "/dev/video0 exists"
    # Test capture capability
    if v4l2-ctl --device=/dev/video0 --list-formats &>/dev/null; then
        FMTS=$(v4l2-ctl --device=/dev/video0 --list-formats 2>/dev/null | grep "BGR3\|YUYV\|MJPG" | wc -l)
        ok "Capture formats available: ${FMTS}"
    fi
else
    fail "/dev/video0 not found (camera not connected or not enabled)"
fi

if command -v libcamera-hello &>/dev/null; then
    # Quick non-blocking camera test
    if timeout 3 libcamera-hello --timeout 100 --nopreview 2>/dev/null; then
        ok "libcamera-hello test passed"
    else
        warn "libcamera-hello test failed (may need reboot after enabling camera)"
    fi
fi

# picamera2
if python3 -c "import picamera2" 2>/dev/null; then
    ok "picamera2 Python library available"
else
    warn "picamera2 not installed (install: sudo apt install python3-picamera2)"
fi

# ── Hailo-8L ──────────────────────────────────────────────────────────────────
echo ""
echo "Hailo AI HAT+:"
if ls /dev/hailo0 &>/dev/null; then
    ok "/dev/hailo0 exists"
else
    fail "/dev/hailo0 not found (check PCIe connection and driver)"
fi

if lspci 2>/dev/null | grep -qi "hailo"; then
    ok "Hailo PCIe device detected by lspci"
else
    warn "Hailo not found in lspci (PCIe may not be active)"
fi

if python3 -c "from hailo_platform import VDevice" 2>/dev/null; then
    ok "HailoRT Python SDK importable"
    # Quick connection test
    if python3 -c "
from hailo_platform import VDevice
vd = VDevice()
vd.release()
print('OK')
" 2>/dev/null | grep -q "OK"; then
        ok "Hailo-8L connection test passed"
    else
        warn "Hailo-8L connection test failed (check permissions: groups | grep video)"
    fi
else
    warn "HailoRT Python SDK not installed (CPU fallback will be used)"
fi

# PCIe Gen 3
if grep -q "pciex1_gen=3" /boot/firmware/config.txt 2>/dev/null; then
    ok "PCIe Gen 3 enabled in config.txt"
else
    warn "PCIe Gen 3 not enabled (add 'dtparam=pciex1_gen=3' to /boot/firmware/config.txt)"
fi

# ── LiDAR ────────────────────────────────────────────────────────────────────
echo ""
echo "Hokuyo LiDAR:"
if ls /dev/ttyACM0 &>/dev/null; then
    ok "/dev/ttyACM0 exists"
elif ls /dev/lidar &>/dev/null; then
    ok "/dev/lidar symlink exists (udev rule installed)"
else
    fail "/dev/ttyACM0 not found (check USB connection)"
fi

if lsusb 2>/dev/null | grep -q "15d1"; then
    ok "Hokuyo USB device detected (VID: 15d1)"
else
    warn "Hokuyo not found in lsusb (may be different VID or not connected)"
fi

if python3 -c "import serial" 2>/dev/null; then
    ok "pyserial available"
fi
if python3 -c "import serial_asyncio" 2>/dev/null; then
    ok "pyserial-asyncio available"
else
    warn "pyserial-asyncio not installed (pip install pyserial-asyncio)"
fi

# ── Python environment ────────────────────────────────────────────────────────
echo ""
echo "Python environment:"
PY_VER=$(python3 --version 2>&1)
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"; then
    ok "Python version: ${PY_VER}"
else
    fail "Python 3.11+ required (found: ${PY_VER})"
fi

for pkg in fastapi uvicorn cv2 numpy psutil yaml; do
    if python3 -c "import ${pkg}" 2>/dev/null; then
        ok "Python package: ${pkg}"
    else
        fail "Python package missing: ${pkg}"
    fi
done

# ── Models ────────────────────────────────────────────────────────────────────
echo ""
echo "Models:"
MODELS_DIR="$(dirname "${BASH_SOURCE[0]}")/../models"
if [[ -f "${MODELS_DIR}/yolov8n.hef" ]]; then
    SIZE=$(du -h "${MODELS_DIR}/yolov8n.hef" | cut -f1)
    ok "yolov8n.hef found (${SIZE})"
else
    warn "yolov8n.hef not found - compile from ONNX (see docs/HAILO_SETUP.md)"
fi

if [[ -f "${MODELS_DIR}/yolov8n.pt" ]]; then
    SIZE=$(du -h "${MODELS_DIR}/yolov8n.pt" | cut -f1)
    ok "yolov8n.pt found (${SIZE}) - CPU fallback available"
else
    warn "yolov8n.pt not found (download: bash scripts/download_model.sh)"
fi

# ── System temperature ────────────────────────────────────────────────────────
echo ""
echo "Thermals:"
if command -v vcgencmd &>/dev/null; then
    TEMP=$(vcgencmd measure_temp | grep -oP '[0-9.]+')
    if (( $(echo "${TEMP} < 70" | bc -l) )); then
        ok "CPU temperature: ${TEMP}°C"
    elif (( $(echo "${TEMP} < 80" | bc -l) )); then
        warn "CPU temperature: ${TEMP}°C (getting warm - check cooling)"
    else
        fail "CPU temperature: ${TEMP}°C (TOO HOT - add cooling)"
    fi
fi

THROTTLE=$(vcgencmd get_throttled 2>/dev/null | grep -oP '0x[0-9a-f]+' || echo "unknown")
if [[ "${THROTTLE}" == "0x0" ]]; then
    ok "No CPU throttling detected"
else
    warn "Throttle status: ${THROTTLE} (non-zero = throttling has occurred)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
if [[ ${FAILURES} -eq 0 ]]; then
    echo -e "${GREEN}All hardware checks passed!${NC}"
    echo "Run: sudo systemctl start edge-ai-navigation"
else
    echo -e "${RED}${FAILURES} check(s) failed - resolve before starting.${NC}"
    echo "See docs/TROUBLESHOOTING.md for fixes."
fi
echo "════════════════════════════════════════"
echo ""
exit ${FAILURES}
