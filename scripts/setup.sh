#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/setup.sh
# Cross-platform development and production setup for Edge AI Navigation System
# Supports Raspberry Pi 5 (production) & x86_64 Laptop (development/debugging)
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*" >&2; exit 1; }
section() { echo -e "\n${GREEN}══ $* ══${NC}"; }

# ── Detect hardware platform ──────────────────────────────────────────────────
IS_PI=false
if [[ $(uname -m) == "aarch64" ]] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
    IS_PI=true
fi

# ── Config ────────────────────────────────────────────────────────────────────
if $IS_PI; then
    INSTALL_DIR="/opt/edge-ai-navigation"
    SERVICE_USER="edgeai"
else
    # Dev mode on laptop: install inside local workspace folder
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    INSTALL_DIR="${PROJECT_DIR}"
    SERVICE_USER="$(whoami)"
fi

VENV_DIR="${INSTALL_DIR}/venv"
HAILO_REPO="https://hailo.ai/developer-zone/software-downloads/"   # requires registration

PYTHON=""
PYTHON_PACKAGES=()

# ── Checks ────────────────────────────────────────────────────────────────────
section "Pre-flight checks"

[[ $(id -u) -eq 0 ]] && error "Do NOT run as root. Run as a non-root user."

if $IS_PI; then
    info "Detected platform: Raspberry Pi (production)"
else
    warn "Detected platform: Laptop/x86 (development mode)"
fi

info "Running on: $(uname -srm)"
info "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"

# ── System packages ───────────────────────────────────────────────────────────
section "Installing system packages"

sudo apt-get update -qq

# Prefer a supported installed interpreter if present
if command -v python3.12 >/dev/null 2>&1; then
    PYTHON="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON="python3.11"
fi

# Detect supported Python versions from apt packages if no supported interpreter exists
if [[ -z "${PYTHON}" ]] && command -v apt-cache >/dev/null 2>&1; then
    for ver in 3.12 3.11; do
        if apt-cache show "python${ver}" >/dev/null 2>&1 && apt-cache show "python${ver}-venv" >/dev/null 2>&1; then
            PYTHON="python${ver}"
            PYTHON_PACKAGES+=("python${ver}" "python${ver}-venv" "python${ver}-dev")
            break
        fi
    done
fi

# Fall back to system python3 if its version is supported
if [[ -z "${PYTHON}" && $(command -v python3 >/dev/null 2>&1; echo $?) -eq 0 ]]; then
    python3_version="$(python3 --version 2>&1 | awk '{print $2}')"
    python3_major="${python3_version%%.*}"
    python3_minor="${python3_version#*.}"
    python3_minor="${python3_minor%%.*}"
    if (( python3_major == 3 && python3_minor >= 11 && python3_minor < 13 )); then
        PYTHON="python3"
    fi
fi

if [[ -n "${PYTHON}" && ${#PYTHON_PACKAGES[@]} -eq 0 ]]; then
    if [[ "${PYTHON}" == "python3" ]]; then
        PYTHON_PACKAGES+=(python3 python3-venv python3-dev)
    else
        PYTHON_PACKAGES+=("${PYTHON}" "${PYTHON}-venv" "${PYTHON}-dev")
    fi
fi

if [[ ${#PYTHON_PACKAGES[@]} -eq 0 ]]; then
    python3_version="$(python3 --version 2>&1 | awk '{print $2}' 2>/dev/null || echo 'unknown')"
    error "Detected unsupported Python version ${python3_version}."
    error "The project requires Python 3.11 or 3.12 because numba is not compatible with Python 3.13."
    error "Install a supported Python version before rerunning setup."
    error "Example: sudo apt-get install python3.11 python3.11-venv python3.11-dev"
    error "If your Debian 13 repo does not provide 3.11, use a compatible Python distribution or switch to a repo that provides the needed packages."
fi

if [[ ${#PYTHON_PACKAGES[@]} -eq 0 ]]; then
    error "No supported Python version found. Install Python 3.11 or 3.12 and rerun setup."
fi

# Base packages required on all systems
PACKAGES=(
    "${PYTHON_PACKAGES[@]}"
    python3-pip
    git wget curl
    build-essential
    libgl1-mesa-dev
    libglib2.0-dev
    libsm6 libxrender1 libxext6
    libjpeg-dev libpng-dev
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

# Add Pi-specific packages only on Raspberry Pi
if $IS_PI; then
    PACKAGES+=(python3-picamera2 libcamera-tools ) #libraspberrypi-bin)
fi

info "Installing system packages via apt..."
sudo apt-get install -y --no-install-recommends "${PACKAGES[@]}"

info "System packages installed."

# ── Raspberry Pi 5 optimisations ─────────────────────────────────────────────
if $IS_PI; then
    section "Applying Raspberry Pi 5 optimisations"

    # Enable PCIe Gen 3 (improves Hailo HAT+ throughput)
    if ! grep -q "dtparam=pciex1_gen=3" /boot/firmware/config.txt 2>/dev/null; then
        echo "" | sudo tee -a /boot/firmware/config.txt
        echo "# Edge AI Navigation - PCIe Gen 3 for Hailo HAT+" | sudo tee -a /boot/firmware/config.txt
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
    echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
    info "CPU governor set to 'performance'."
else
    section "Optimisations"
    warn "Skipping Raspberry Pi 5 optimisations (not running on Raspberry Pi)."
fi

# ── User and groups ───────────────────────────────────────────────────────────
section "Creating service user"

if $IS_PI; then
    if ! id "${SERVICE_USER}" &>/dev/null; then
        sudo useradd -r -m -d "${INSTALL_DIR}" \
            -G dialout,video,gpio,i2c,spi,plugdev \
            -s /bin/bash "${SERVICE_USER}"
        info "User '${SERVICE_USER}' created."
    else
        info "User '${SERVICE_USER}' already exists."
    fi

    # Allow current user to act as edgeai for deployment
    sudo usermod -aG "${SERVICE_USER}" "$(whoami)"
else
    info "Development mode: using current user '${SERVICE_USER}' (skipping service user creation)."
fi

# ── udev rules ────────────────────────────────────────────────────────────────
if $IS_PI; then
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
else
    section "udev rules"
    warn "Skipping hardware udev rules installation."
fi

# ── Install directory ─────────────────────────────────────────────────────────
section "Setting up install directory"

if $IS_PI; then
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
else
    # Create runtime directories in the local project workspace
    mkdir -p \
        "${INSTALL_DIR}/logs/snapshots" \
        "${INSTALL_DIR}/models" \
        "${INSTALL_DIR}/datasets"
    info "Local workspace runtime directories prepared."
fi

# ── Python virtual environment ────────────────────────────────────────────────
section "Creating Python virtual environment"

if $IS_PI; then
    sudo -u "${SERVICE_USER}" bash -c "
        ${PYTHON} -m venv '${VENV_DIR}' --system-site-packages
        '${VENV_DIR}/bin/pip' install --upgrade pip wheel setuptools
        '${VENV_DIR}/bin/pip' install --no-cache-dir -r '${INSTALL_DIR}/requirements.txt'
    "
else
    if [[ ! -d "${VENV_DIR}" ]]; then
        ${PYTHON} -m venv "${VENV_DIR}"
        info "Virtual environment created at ${VENV_DIR}."
    fi
    "${VENV_DIR}/bin/pip" install --upgrade pip wheel setuptools
    "${VENV_DIR}/bin/pip" install --no-cache-dir -r "${INSTALL_DIR}/requirements.txt"
fi
info "Virtual environment ready at ${VENV_DIR}."

# ── Hailo SDK ─────────────────────────────────────────────────────────────────
section "Hailo SDK setup"

if $IS_PI; then
    warn "Hailo SDK requires manual download from ${HAILO_REPO}"
    warn "After downloading, run:"
    warn "  sudo dpkg -i hailort_<version>_arm64.deb"
    warn "  ${VENV_DIR}/bin/pip install hailort-<version>-cp311-cp311-linux_aarch64.whl"
    warn "Then run: bash ${SCRIPT_DIR}/download_model.sh yolov8n"

    # If already installed, verify
    if python3 -c "import hailo_platform" 2>/dev/null; then
        info "HailoRT Python SDK already installed."
    else
        warn "HailoRT not yet installed - system will use CPU fallback."
    fi
else
    warn "Skipping Hailo SDK setup for development laptop (CPU fallback default)."
fi

# ── Download YOLO model ───────────────────────────────────────────────────────
section "Downloading YOLOv8n model (CPU fallback)"

MODEL_DIR="${INSTALL_DIR}/models"
if [[ ! -f "${MODEL_DIR}/yolov8n.pt" ]]; then
    if $IS_PI; then
        sudo -u "${SERVICE_USER}" bash -c "
            cd '${INSTALL_DIR}'
            '${VENV_DIR}/bin/python' -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"
            mv ~/.config/Ultralytics/yolov8n.pt '${MODEL_DIR}/yolov8n.pt' 2>/dev/null || true
            cp yolov8n.pt '${MODEL_DIR}/yolov8n.pt' 2>/dev/null || true
        " || warn "Could not download YOLOv8n model - download manually."
    else
        "${VENV_DIR}/bin/python" -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
        mv ~/.config/Ultralytics/yolov8n.pt "${MODEL_DIR}/yolov8n.pt" 2>/dev/null || true
        cp yolov8n.pt "${MODEL_DIR}/yolov8n.pt" 2>/dev/null || true
    fi
    info "YOLOv8n CPU fallback model placed in ${MODEL_DIR}/yolov8n.pt"
else
    info "YOLOv8n CPU fallback model already exists."
fi

# ── systemd service ───────────────────────────────────────────────────────────
if $IS_PI; then
    section "Installing systemd service"

    sudo cp "${INSTALL_DIR}/systemd/edge-ai-navigation.service" \
            /etc/systemd/system/edge-ai-navigation.service

    # Patch WorkingDirectory path in service file
    sudo sed -i "s|/opt/edge-ai-navigation|${INSTALL_DIR}|g" \
        /etc/systemd/system/edge-ai-navigation.service

    sudo systemctl daemon-reload
    sudo systemctl enable edge-ai-navigation
    info "systemd service installed and enabled."
else
    section "systemd service"
    warn "Skipping systemd service installation for development host."
fi

# ── Environment file ──────────────────────────────────────────────────────────
section "Creating .env file"

if [[ ! -f "${INSTALL_DIR}/.env" ]]; then
    if $IS_PI; then
        sudo -u "${SERVICE_USER}" tee "${INSTALL_DIR}/.env" > /dev/null <<ENV
# Edge AI Navigation - Runtime environment
# Generated by setup.sh - edit to customise

EDGE_AI_API_KEY=$(openssl rand -hex 24)
EDGE_AI_LOG_LEVEL=INFO
EDGE_AI_INFERENCE_DEVICE=auto
ENV
    else
        tee "${INSTALL_DIR}/.env" > /dev/null <<ENV
# Edge AI Navigation - Runtime environment
# Generated by setup.sh - edit to customise

EDGE_AI_API_KEY=$(openssl rand -hex 24)
EDGE_AI_LOG_LEVEL=INFO
EDGE_AI_INFERENCE_DEVICE=cpu
ENV
    fi
    info ".env file created at ${INSTALL_DIR}/.env"
    warn "Your API key is: $(grep EDGE_AI_API_KEY "${INSTALL_DIR}/.env" | cut -d= -f2)"
else
    info ".env already exists - skipping."
fi

# ── Firewall recommendations ──────────────────────────────────────────────────
if $IS_PI; then
    section "Firewall setup (optional)"

    if command -v ufw &>/dev/null; then
        warn "Consider running:"
        warn "  sudo ufw allow 22/tcp    # SSH"
        warn "  sudo ufw allow 8080/tcp  # Dashboard (restrict to LAN)"
        warn "  sudo ufw enable"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "Setup complete!"
echo ""
info "Installation directory : ${INSTALL_DIR}"
info "Virtual environment    : ${VENV_DIR}"
info "Service user           : ${SERVICE_USER}"
if $IS_PI; then
    info "Dashboard port         : 8080"
    echo ""
    warn "NEXT STEPS:"
    warn "  1. Install Hailo SDK (see docs/HAILO_SETUP.md)"
    warn "  2. Compile YOLOv8n to .hef (see docs/MODEL_COMPILE.md)"
    warn "  3. Reboot: sudo reboot"
    warn "  4. Start: sudo systemctl start edge-ai-navigation"
    warn "  5. View dashboard: http://$(hostname -I | awk '{print $1}'):8080"
    warn "  6. Monitor: sudo journalctl -u edge-ai-navigation -f"
else
    info "Development port       : 8080"
    echo ""
    warn "NEXT STEPS:"
    warn "  1. Activate virtual environment: source venv/bin/activate"
    warn "  2. Run development server: uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload"
    warn "  3. Open dashboard: http://localhost:8080"
    warn "  Note: Mock fallback streams will automatically run if camera/LiDAR hardware is missing."
fi
echo ""
