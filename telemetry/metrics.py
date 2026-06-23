"""
telemetry/metrics.py
Real-time system telemetry collector.

Collects:
  - CPU utilisation (per-core and aggregate)
  - Memory usage (RSS, available, percent)
  - Disk usage (root partition)
  - CPU temperature (via vcgencmd or /sys/class/thermal)
  - GPU temperature (Pi 5 VideoCore)
  - Hailo-8L utilisation (via HailoRT power/utilisation API)
  - Network I/O rates (bytes/sec)
  - FPS counters for camera, inference, lidar pipelines
  - Process uptime

Exposes:
  - snapshot()          → dict for REST/WebSocket
  - prometheus_export() → Prometheus text format string
  - run()               → async loop that broadcasts at interval_seconds
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from app.websocket.stream import WebSocketManager
    from config.config_loader import TelemetryConfig
    from inference.hailo_engine import HailoInferenceEngine

logger = logging.getLogger(__name__)

_PROCESS = psutil.Process(os.getpid())
_START_TIME = time.time()


class FPSCounter:
    """Sliding-window FPS counter (thread-safe for use with asyncio)."""

    def __init__(self, window: float = 2.0) -> None:
        self._window = window
        self._timestamps: list[float] = []

    def tick(self) -> None:
        now = time.monotonic()
        self._timestamps.append(now)
        # Evict old entries
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t >= cutoff]

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 0 else 0.0


# Global FPS counters shared across pipeline modules
fps_camera    = FPSCounter()
fps_inference = FPSCounter()
fps_lidar     = FPSCounter()
fps_fusion    = FPSCounter()


class TelemetryCollector:
    """Collects and broadcasts system telemetry."""

    def __init__(self, cfg: TelemetryConfig) -> None:
        self._cfg = cfg
        self._net_io_prev = psutil.net_io_counters()
        self._net_io_time = time.monotonic()
        self._last_snapshot: dict[str, Any] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Collect and return current telemetry as a dict."""

        # CPU
        cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

        # Memory
        mem = psutil.virtual_memory()

        # Disk
        disk = psutil.disk_usage("/")

        # Temperature
        temp_cpu, temp_gpu = self._read_temperatures()

        # Network I/O rates
        net_recv_bps, net_sent_bps = self._net_io_rate()

        # Process
        try:
            proc_mem_mb = _PROCESS.memory_info().rss / 1_048_576
            proc_cpu    = _PROCESS.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            proc_mem_mb = 0.0
            proc_cpu    = 0.0

        data = {
            "timestamp":       time.time(),
            "uptime_seconds":  round(time.time() - _START_TIME, 1),
            "cpu": {
                "percent":      round(cpu_percent, 1),
                "per_core":     [round(c, 1) for c in cpu_per_core],
                "count":        psutil.cpu_count(),
                "freq_mhz":     _cpu_freq_mhz(),
            },
            "memory": {
                "total_mb":     round(mem.total / 1_048_576, 1),
                "available_mb": round(mem.available / 1_048_576, 1),
                "used_mb":      round(mem.used / 1_048_576, 1),
                "percent":      mem.percent,
            },
            "disk": {
                "total_gb":  round(disk.total / 1_073_741_824, 2),
                "used_gb":   round(disk.used / 1_073_741_824, 2),
                "free_gb":   round(disk.free / 1_073_741_824, 2),
                "percent":   disk.percent,
            },
            "temperature": {
                "cpu_c":    temp_cpu,
                "gpu_c":    temp_gpu,
            },
            "network": {
                "recv_kbps": round(net_recv_bps / 1024, 1),
                "sent_kbps": round(net_sent_bps / 1024, 1),
            },
            "process": {
                "memory_mb": round(proc_mem_mb, 1),
                "cpu_percent": round(proc_cpu, 1),
            },
            "fps": {
                "camera":    round(fps_camera.fps,    1),
                "inference": round(fps_inference.fps, 1),
                "lidar":     round(fps_lidar.fps,     1),
                "fusion":    round(fps_fusion.fps,    1),
            },
        }

        self._last_snapshot = data
        return data

    def prometheus_export(self) -> str:
        """Return Prometheus text-format metrics string."""
        s = self._last_snapshot
        if not s:
            s = self.snapshot()

        lines = [
            "# HELP edge_ai_cpu_percent CPU utilisation percent",
            "# TYPE edge_ai_cpu_percent gauge",
            f"edge_ai_cpu_percent {s['cpu']['percent']}",
            "# HELP edge_ai_memory_used_bytes Memory used in bytes",
            "# TYPE edge_ai_memory_used_bytes gauge",
            f"edge_ai_memory_used_bytes {s['memory']['used_mb'] * 1_048_576:.0f}",
            "# HELP edge_ai_temperature_cpu_celsius CPU temperature",
            "# TYPE edge_ai_temperature_cpu_celsius gauge",
            f"edge_ai_temperature_cpu_celsius {s['temperature']['cpu_c']}",
            "# HELP edge_ai_fps_camera Camera capture FPS",
            "# TYPE edge_ai_fps_camera gauge",
            f"edge_ai_fps_camera {s['fps']['camera']}",
            "# HELP edge_ai_fps_inference Inference pipeline FPS",
            "# TYPE edge_ai_fps_inference gauge",
            f"edge_ai_fps_inference {s['fps']['inference']}",
            "# HELP edge_ai_fps_lidar LiDAR scan FPS",
            "# TYPE edge_ai_fps_lidar gauge",
            f"edge_ai_fps_lidar {s['fps']['lidar']}",
            "# HELP edge_ai_uptime_seconds System uptime in seconds",
            "# TYPE edge_ai_uptime_seconds counter",
            f"edge_ai_uptime_seconds {s['uptime_seconds']}",
        ]
        return "\n".join(lines) + "\n"

    async def run(
        self,
        ws_manager: WebSocketManager,
        hailo: HailoInferenceEngine | None,
        shutdown_event: asyncio.Event,
    ) -> None:
        """
        Async loop: collect telemetry and broadcast on the 'telemetry' channel.
        Optionally publishes to MQTT if enabled.
        """
        mqtt_client = None
        if self._cfg.mqtt_enabled:
            mqtt_client = await self._setup_mqtt()

        logger.info("Telemetry loop started  (interval=%.1fs)", self._cfg.interval_seconds)

        while not shutdown_event.is_set():
            try:
                data = self.snapshot()

                # Add Hailo utilisation if available
                if hailo is not None:
                    data["hailo"] = await hailo.utilisation_stats()
                else:
                    data["hailo"] = {"available": False, "device_type": "cpu"}


                await ws_manager.broadcast_json(data, channel="telemetry")

                if mqtt_client:
                    await self._publish_mqtt(mqtt_client, data)

            except Exception as exc:
                logger.warning("Telemetry error: %s", exc)

            await asyncio.sleep(self._cfg.interval_seconds)

        if mqtt_client:
            mqtt_client.disconnect()

        logger.info("Telemetry loop stopped.")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _read_temperatures(self) -> tuple[float, float]:
        """Return (cpu_temp_c, gpu_temp_c) with fallbacks."""
        cpu_temp = 0.0
        gpu_temp = 0.0

        # Try /sys/class/thermal (most reliable on Pi 5)
        try:
            thermal_path = "/sys/class/thermal/thermal_zone0/temp"
            with open(thermal_path) as f:
                cpu_temp = int(f.read().strip()) / 1000.0
        except OSError:
            # Fallback: psutil sensors
            try:
                sensors = psutil.sensors_temperatures()
                for key in ("cpu_thermal", "coretemp", "acpitz"):
                    if key in sensors and sensors[key]:
                        cpu_temp = sensors[key][0].current
                        break
            except AttributeError:
                pass

        # GPU temp via vcgencmd (Raspberry Pi specific)
        try:
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if result.returncode == 0:
                # Output: "temp=47.2'C"
                raw = result.stdout.strip()
                gpu_temp = float(raw.split("=")[1].replace("'C", ""))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            gpu_temp = cpu_temp  # approximate

        return round(cpu_temp, 1), round(gpu_temp, 1)

    def _net_io_rate(self) -> tuple[float, float]:
        """Return (bytes_recv_per_sec, bytes_sent_per_sec) since last call."""
        now = time.monotonic()
        counters = psutil.net_io_counters()
        elapsed = now - self._net_io_time

        if elapsed < 0.01:
            return 0.0, 0.0

        recv_rate = (counters.bytes_recv - self._net_io_prev.bytes_recv) / elapsed
        sent_rate = (counters.bytes_sent - self._net_io_prev.bytes_sent) / elapsed

        self._net_io_prev = counters
        self._net_io_time = now

        return max(0.0, recv_rate), max(0.0, sent_rate)

    async def _setup_mqtt(self):
        """Initialise async MQTT client (aiomqtt)."""
        try:
            import aiomqtt  # type: ignore
            client = aiomqtt.Client(
                hostname=self._cfg.mqtt_broker,
                port=self._cfg.mqtt_port,
                username=self._cfg.mqtt_username or None,
                password=self._cfg.mqtt_password or None,
            )
            logger.info("MQTT client configured: %s:%d", self._cfg.mqtt_broker, self._cfg.mqtt_port)
            return client
        except ImportError:
            logger.warning("aiomqtt not installed - MQTT disabled.")
            return None
        except Exception as exc:
            logger.warning("MQTT setup failed: %s", exc)
            return None

    async def _publish_mqtt(self, client, data: dict) -> None:
        """Publish telemetry to MQTT topics."""
        import json as _json
        try:
            topic = f"{self._cfg.mqtt_topic_prefix}/telemetry"
            async with client:
                await client.publish(topic, _json.dumps(data))
        except Exception as exc:
            logger.debug("MQTT publish error: %s", exc)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _cpu_freq_mhz() -> float:
    """Return current CPU frequency in MHz, or 0 if unavailable."""
    try:
        freq = psutil.cpu_freq()
        return round(freq.current, 0) if freq else 0.0
    except Exception:
        return 0.0
