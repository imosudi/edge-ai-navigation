#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/deploy.sh
# Remote deployment to Raspberry Pi 5 via rsync + SSH
#
# Usage:
#   bash scripts/deploy.sh [PI_HOST] [PI_USER]
#
# Defaults:
#   PI_HOST = raspberrypi.local
#   PI_USER = pi
#
# Prerequisites:
#   - SSH key-based auth set up for PI_HOST
#   - setup.sh already run on the Pi (creates /opt/edge-ai-navigation)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PI_HOST="${1:-raspberrypi.local}"
PI_USER="${2:-pi}"
REMOTE="${PI_USER}@${PI_HOST}"
REMOTE_DIR="/opt/edge-ai-navigation"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[DEPLOY]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}   $*"; }

info "Deploying to ${REMOTE}:${REMOTE_DIR}"

# ── SSH connectivity check ─────────────────────────────────────────────────────
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE}" "echo ok" &>/dev/null; then
    echo "ERROR: Cannot SSH to ${REMOTE}. Check host/credentials."
    exit 1
fi

# ── Sync application code ──────────────────────────────────────────────────────
info "Syncing files…"
rsync -avz --delete \
    --exclude=".git" \
    --exclude="venv" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".env" \
    --exclude="logs/*" \
    --exclude="models/*.pt" \
    --exclude="datasets/*" \
    --exclude="*.egg-info" \
    "${LOCAL_DIR}/" \
    "${REMOTE}:${REMOTE_DIR}/"

info "Sync complete."

# ── Update Python dependencies ─────────────────────────────────────────────────
info "Updating dependencies…"
ssh "${REMOTE}" bash -s <<REMOTE_SCRIPT
    cd "${REMOTE_DIR}"
    sudo -u edgeai "${REMOTE_DIR}/venv/bin/pip" install \
        --no-cache-dir -q -r requirements.txt
REMOTE_SCRIPT

# ── Restart service ────────────────────────────────────────────────────────────
info "Restarting edge-ai-navigation service…"
ssh "${REMOTE}" "sudo systemctl restart edge-ai-navigation"

# ── Wait and check status ──────────────────────────────────────────────────────
sleep 4
STATUS=$(ssh "${REMOTE}" "sudo systemctl is-active edge-ai-navigation 2>&1" || echo "failed")

if [[ "${STATUS}" == "active" ]]; then
    info "Service is ACTIVE ✓"
    DASHBOARD_IP=$(ssh "${REMOTE}" "hostname -I | awk '{print \$1}'" 2>/dev/null || echo "${PI_HOST}")
    info "Dashboard: http://${DASHBOARD_IP}:8080"
else
    warn "Service status: ${STATUS}"
    warn "Check logs: ssh ${REMOTE} 'sudo journalctl -u edge-ai-navigation -n 50'"
    exit 1
fi
