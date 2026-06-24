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

# Custom SSH options (e.g. SSH_OPTS="-o PubkeyAuthentication=no" or "-o IdentitiesOnly=yes")
SSH_OPTS="${SSH_OPTS:-}"

PI_HOST="${1:-raspberrypi.local}"
PI_USER="${2:-pi}"
REMOTE="${PI_USER}@${PI_HOST}"
REMOTE_DIR="/opt/edge-ai-navigation"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[DEPLOY]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}   $*"; }
error() { echo -e "${RED}[ERR]${NC}    $*" >&2; exit 1; }

ssh_run() {
    ssh -o ConnectTimeout=10 ${SSH_OPTS} "$@"
}

info "Deploying to ${REMOTE}:${REMOTE_DIR}"

# ── SSH connectivity check ─────────────────────────────────────────────────────
# Try batch mode first (fast, passwordless). If it fails, fall back to interactive (prompts for password).
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes ${SSH_OPTS} "${REMOTE}" "echo ok" &>/dev/null; then
    warn "Passwordless SSH check failed. Retrying with interactive/password prompt..."
    if ! ssh_run "${REMOTE}" "echo ok" &>/dev/null; then
        echo "ERROR: Cannot SSH to ${REMOTE}. Check host/credentials."
        exit 1
    fi
fi

# ── Sync application code ──────────────────────────────────────────────────────
info "Syncing files…"
rsync -avz --delete \
    -e "ssh ${SSH_OPTS}" \
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
ssh_run "${REMOTE}" bash -s <<REMOTE_SCRIPT
    cd "${REMOTE_DIR}"
    sudo -u edgeai "${REMOTE_DIR}/venv/bin/pip" install \
        --no-cache-dir -q -r requirements.txt
REMOTE_SCRIPT

# ── Restart service ────────────────────────────────────────────────────────────
info "Restarting edge-ai-navigation service…"
ssh_run "${REMOTE}" "sudo systemctl restart edge-ai-navigation"

# ── Wait and check status ──────────────────────────────────────────────────────
sleep 4
STATUS=$(ssh_run "${REMOTE}" "sudo systemctl is-active edge-ai-navigation 2>&1" || echo "failed")

if [[ "${STATUS}" == "active" ]]; then
    info "Service is ACTIVE ✓"
    DASHBOARD_IP=$(ssh_run "${REMOTE}" "hostname -I | awk '{print \$1}'" 2>/dev/null || echo "${PI_HOST}")
    info "Dashboard: http://${DASHBOARD_IP}:8080"
else
    warn "Service status: ${STATUS}"
    warn "Check logs: ssh_run ${REMOTE} 'sudo journalctl -u edge-ai-navigation -n 50'"
    exit 1
fi
