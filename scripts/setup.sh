#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/setup.sh
# Complete Raspberry Pi 5 setup for Edge AI Navigation System
#
# Run as the pi user (NOT root):
#   chmod +x scripts/setup.sh
#   bash scripts/setup.sh
#
# Tested on: Raspberry Pi OS Bookworm (64-bit) - May 2025
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*" >&2; exit 1; }
section() { echo -e "\n${GREEN}══ $* ══${NC}"; }

# ── Config ────────────────────────────────────────────────────────────────────
INSTALL_DIR="/opt/edge-ai-navigation"
VENV_DIR="${INSTALL_DIR}/venv"
SERVICE_USER="edgeai"
PYTHON=""
HAILO_REPO="https://hailo.ai/developer-zone/software-downloads/"   # requires registration
REQUIRED_PYTHON_VERSION=""

# ── Checks ────────────────────────────────────────────────────────────────────
section "Pre-flight checks"

[[ $(id -u) -eq 0 ]] && error "Do NOT run as root. Run as the pi user."
[[ $(uname -m) == "aarch64" ]] || warn "Not running on ARM64 — some steps may fail."

info "Running on: $(uname -srm)"
info "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"

# ── System packages ───────────────────────────────────────────────────────────
section "Installing system packages"

sudo apt-get update -qq

PYTHON_CANDIDATES=(python3.11 python3.12)
COMMON_PKG_LIST=(
    python3-pip
    git
    wget
    curl
    build-essential
    libgl1-mesa-dev
    libglib2.0-dev
    libsm6
    libxrender1
    libxext6
    libjpeg-dev
    libpng-dev
    libopencv-dev
    v4l-utils
    udev
    pkg-config
    cmake
    python3-serial
    setserial
    lm-sensors
    htop
    iotop
    nethogs
)
PI_ONLY_PKG_LIST=(
    python3-picamera2
    libcamera-tools
    libraspberrypi-bin
)

if [[ $(uname -m) != "aarch64" ]]; then
    warn "Non-ARM64 detected; skipping Raspberry Pi-specific packages: ${PI_ONLY_PKG_LIST[*]}"
    PACKAGE_LIST=("${COMMON_PKG_LIST[@]}")
else
    PACKAGE_LIST=("${COMMON_PKG_LIST[@]}" "${PI_ONLY_PKG_LIST[@]}")
fi

install_packages() {
    local py="$1"
    sudo apt-get install -y --no-install-recommends \
        "${py}" "${py}-venv" "${py}-dev" "${PACKAGE_LIST[@]}"
}

for candidate in "${PYTHON_CANDIDATES[@]}"; do
    info "Attempting to install packages with ${candidate}."
    if install_packages "${candidate}"; then
        PYTHON="${candidate}"
        REQUIRED_PYTHON_VERSION="${candidate#python3.}"
        break
    fi
    warn "${candidate} packages not available; trying next candidate."
done

if [[ -z "${PYTHON}" ]]; then
    error "Could not install python3.11 or python3.12. Please install a supported Python version manually."
fi

info "System packages installed."

# ── Raspberry Pi 5 optimisations ─────────────────────────────────────────────
if [[ $(uname -m) != "aarch64" ]]; then
    warn "Non-ARM64 detected; skipping Raspberry Pi 5 optimisations."
else
    section "Applying Raspberry Pi 5 optimisations"

    # Enable PCIe Gen 3 (improves Hailo HAT+ throughput)
    if ! grep -q "dtparam=pciex1_gen=3" /boot/firmware/config.txt 2>/dev/null; then
        echo "" | sudo tee -a /boot/firmware/config.txt
        echo "# Edge AI Navigation — PCIe Gen 3 for Hailo HAT+" | sudo tee -a /boot/firmware/config.txt
        echo "dtparam=pciex1_gen=3" | sudo tee -a /boot/firmware/config.txt
        info "PCIe Gen 3 enabled (reboot required)."
    else
        info "PCIe Gen 3 already enabled."
    fi

    # Enable camera (CSI)
    if ! grep -q "camera_auto_detect=1" /boot/firmware/config.txt 2>/dev/null; then
        echo "camera_auto_detect=1" | sudo tee -a /boot/firmware/config.txt
    fi

    # Set GPU memory split (128 MB for camera processing)
    if ! grep -q "gpu_mem=128" /boot/firmware/config.txt 2>/dev/null; then
        echo "gpu_mem=128" | sudo tee -a /boot/firmware/config.txt
    fi

    # Increase USB buffer for LiDAR
    if ! grep -q "usbcore.usbfs_memory_mb" /boot/firmware/cmdline.txt 2>/dev/null; then
        sudo sed -i 's/$/ usbcore.usbfs_memory_mb=256/' /boot/firmware/cmdline.txt
        info "USB memory buffer increased."
    fi

    # CPU governor: performance (sustained throughput > power saving)
    echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    info "CPU governor set to 'performance'."
fi

# ── User and groups ───────────────────────────────────────────────────────────
section "Creating service user"

SERVICE_GROUPS=(dialout video plugdev)
if [[ $(uname -m) == "aarch64" ]]; then
    SERVICE_GROUPS+=(gpio i2c spi)
else
    warn "Non-ARM64 detected; skipping Raspberry Pi-specific groups: gpio, i2c, spi"
fi

GROUPS_TO_ADD=()
for group in "${SERVICE_GROUPS[@]}"; do
    if getent group "${group}" >/dev/null 2>&1; then
        GROUPS_TO_ADD+=("${group}")
    else
        warn "Group '${group}' does not exist; skipping"
    fi
done

if ! id "${SERVICE_USER}" &>/dev/null; then
    if [[ ${#GROUPS_TO_ADD[@]} -gt 0 ]]; then
        IFS=, GROUP_LIST="${GROUPS_TO_ADD[*]}"
        sudo useradd -r -m -d "${INSTALL_DIR}" \
            -G "${GROUP_LIST}" \
            -s /bin/bash "${SERVICE_USER}"
    else
        sudo useradd -r -m -d "${INSTALL_DIR}" \
            -s /bin/bash "${SERVICE_USER}"
    fi
    info "User '${SERVICE_USER}' created."
else
    info "User '${SERVICE_USER}' already exists."
fi

# Allow current user to act as edgeai for deployment
sudo usermod -aG "${SERVICE_USER}" "$(whoami)"

# ── udev rules ────────────────────────────────────────────────────────────────
section "Installing udev rules"

sudo tee /etc/udev/rules.d/99-edge-ai-nav.rules > /dev/null <<'UDEV'
# Hokuyo URG-04LX-UG-01 LiDAR
SUBSYSTEM=="tty", ATTRS{idVendor}=="15d1", ATTRS{idProduct}=="0000", \
    SYMLINK+="lidar", GROUP="dialout", MODE="0664"

# Hailo-8L (PCIe)
SUBSYSTEM=="hailo_chardev", KERNEL=="hailo*", \
    GROUP="video", MODE="0664"

# Raspberry Pi Camera (v4l2)
SUBSYSTEM=="video4linux", KERNEL=="video0", \
    GROUP="video", MODE="0664"
UDEV

sudo udevadm control --reload-rules
sudo udevadm trigger
info "udev rules installed."

# ── Install directory ─────────────────────────────────────────────────────────
section "Setting up install directory"

sudo mkdir -p "${INSTALL_DIR}"
sudo chown "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"

# Copy project files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

sudo rsync -a --exclude=".git" --exclude="venv" --exclude="__pycache__" \
    "${PROJECT_DIR}/" "${INSTALL_DIR}/"
sudo chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"

# Create runtime directories
sudo -u "${SERVICE_USER}" mkdir -p \
    "${INSTALL_DIR}/logs/snapshots" \
    "${INSTALL_DIR}/models" \
    "${INSTALL_DIR}/datasets"

info "Files deployed to ${INSTALL_DIR}."

# ── Python virtual environment ────────────────────────────────────────────────
section "Creating Python virtual environment"

sudo -u "${SERVICE_USER}" bash -c "
    ${PYTHON} -m venv '${VENV_DIR}' --system-site-packages
    '${VENV_DIR}/bin/pip' install --upgrade pip wheel setuptools
    '${VENV_DIR}/bin/pip' install --no-cache-dir -r '${INSTALL_DIR}/requirements.txt'
"
info "Virtual environment ready at ${VENV_DIR}."

# ── Hailo SDK ─────────────────────────────────────────────────────────────────
section "Hailo SDK setup"

warn "Hailo SDK requires manual download from ${HAILO_REPO}"
warn "After downloading, run:"
warn "  sudo dpkg -i hailort_<version>_arm64.deb"
warn "  ${VENV_DIR}/bin/pip install hailort-<version>-cp311-cp311-linux_aarch64.whl"
warn "Then run: bash ${SCRIPT_DIR}/download_model.sh yolov8n"

# If already installed, verify
if python3 -c "import hailo_platform" 2>/dev/null; then
    info "HailoRT Python SDK already installed."
else
    warn "HailoRT not yet installed — system will use CPU fallback."
fi

# ── Download YOLO model ───────────────────────────────────────────────────────
section "Downloading YOLOv8n model (CPU fallback)"

MODEL_DIR="${INSTALL_DIR}/models"
if [[ ! -f "${MODEL_DIR}/yolov8n.pt" ]]; then
    sudo -u "${SERVICE_USER}" bash -c "
        cd '${INSTALL_DIR}'
        '${VENV_DIR}/bin/python' -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"
        mv ~/.config/Ultralytics/yolov8n.pt '${MODEL_DIR}/yolov8n.pt' 2>/dev/null || true
        cp yolov8n.pt '${MODEL_DIR}/yolov8n.pt' 2>/dev/null || true
    " || warn "Could not download YOLOv8n model — download manually."
fi

# ── systemd service ───────────────────────────────────────────────────────────
section "Installing systemd service"

sudo cp "${INSTALL_DIR}/systemd/edge-ai-navigation.service" \
        /etc/systemd/system/edge-ai-navigation.service

# Patch WorkingDirectory path in service file
sudo sed -i "s|/opt/edge-ai-navigation|${INSTALL_DIR}|g" \
    /etc/systemd/system/edge-ai-navigation.service

sudo systemctl daemon-reload
sudo systemctl enable edge-ai-navigation
info "systemd service installed and enabled."

# ── Environment file ──────────────────────────────────────────────────────────
section "Creating .env file"

if [[ ! -f "${INSTALL_DIR}/.env" ]]; then
    sudo -u "${SERVICE_USER}" tee "${INSTALL_DIR}/.env" > /dev/null <<ENV
# Edge AI Navigation — Runtime environment
# Generated by setup.sh — edit to customise

EDGE_AI_API_KEY=$(openssl rand -hex 24)
EDGE_AI_LOG_LEVEL=INFO
EDGE_AI_INFERENCE_DEVICE=auto
ENV
    info ".env file created at ${INSTALL_DIR}/.env"
    warn "Your API key is: $(sudo grep EDGE_AI_API_KEY "${INSTALL_DIR}/.env" | cut -d= -f2)"
else
    info ".env already exists — skipping."
fi

# ── Firewall recommendations ──────────────────────────────────────────────────
section "Firewall setup (optional)"

if command -v ufw &>/dev/null; then
    warn "Consider running:"
    warn "  sudo ufw allow 22/tcp    # SSH"
    warn "  sudo ufw allow 8080/tcp  # Dashboard (restrict to LAN)"
    warn "  sudo ufw enable"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "Setup complete!"
echo ""
info "Installation directory : ${INSTALL_DIR}"
info "Virtual environment    : ${VENV_DIR}"
info "Service user           : ${SERVICE_USER}"
info "Dashboard port         : 8080"
echo ""
warn "NEXT STEPS:"
warn "  1. Install Hailo SDK (see docs/HAILO_SETUP.md)"
warn "  2. Compile YOLOv8n to .hef (see docs/MODEL_COMPILE.md)"
warn "  3. Reboot: sudo reboot"
warn "  4. Start: sudo systemctl start edge-ai-navigation"
warn "  5. View dashboard: http://$(hostname -I | awk '{print $1}'):8080"
warn "  6. Monitor: sudo journalctl -u edge-ai-navigation -f"
