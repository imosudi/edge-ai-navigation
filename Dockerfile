# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Edge AI Navigation System
# Base: python:3.11-slim-bookworm  (official multi-arch: amd64 + arm64)
#
# Build (Pi 5 native or any host):
#   docker compose up --build
#
# Cross-build for arm64 from x86:
#   docker buildx build --platform linux/arm64 -t edge-ai-nav:latest --load .
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm

LABEL maintainer="edge-ai-nav" \
      description="Edge AI Navigation System (Hailo-8L + Hokuyo LiDAR)" \
      version="1.0.0"

# ── Step 1: Packages available on ALL Debian Bookworm architectures ──────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        udev \
        procps \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# ── Step 2: Raspberry Pi–specific packages (skip silently on x86) ───────────
# libcamera0.2 and libraspberrypi-bin only exist in the Pi OS apt mirror.
# On a standard Debian/Ubuntu host they are simply not available, which is
# fine — picamera2 is injected via the host venv mount on the Pi anyway.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcamera0.2 libraspberrypi-bin \
    || echo "[INFO] Pi-specific packages not available on this platform — skipping." \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd -f gpio \
    && useradd -m -u 1000 -G dialout,video,gpio edgeai

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY --chown=edgeai:edgeai . .

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p logs/snapshots models datasets \
    && chown -R edgeai:edgeai logs models datasets

USER edgeai

ENV EDGE_AI_CONFIG=/app/config/settings.yaml \
    EDGE_AI_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCV_IO_ENABLE_OPENEXR=0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/status')" \
    || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--loop", "asyncio", \
     "--log-level", "info", \
     "--no-access-log"]