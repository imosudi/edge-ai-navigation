"""
config/config_loader.py
Hierarchical configuration system.

Priority (highest → lowest):
  1. Environment variables  (EDGE_AI_*)
  2. config/settings.yaml
  3. Built-in defaults

All config models use Pydantic v2 for validation and type safety.
Supports runtime mutation of threshold values via the /config API.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(os.environ.get("EDGE_AI_CONFIG", "config/settings.yaml"))


# ─────────────────────────────────────────────
# Sub-config models
# ─────────────────────────────────────────────

class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(8080, ge=1024, le=65535)
    cors_origins: list[str] = ["*"]
    rate_limit_rpm: int = Field(120, ge=10)
    require_api_key: bool = False
    api_key: str = Field(default_factory=lambda: os.environ.get("EDGE_AI_API_KEY", "changeme"))


class CameraConfig(BaseModel):
    device_index: int = 0
    width: int  = 1280
    height: int = 720
    fps: int    = 30
    jpeg_quality: int = Field(80, ge=10, le=100)
    # Horizontal field of view in degrees (Camera Module 3 = ~66°)
    hfov_degrees: float = 66.0
    snapshot_dir: str = "logs/snapshots"
    auto_exposure: bool = True
    auto_white_balance: bool = True


class InferenceConfig(BaseModel):
    model_name: str = "yolov8n"
    model_path: str = "models/yolov8n.hef"
    # CPU fallback ONNX/PT model path (when Hailo unavailable)
    cpu_model_path: str = "models/yolov8n.pt"
    confidence_threshold: float = Field(0.45, ge=0.0, le=1.0)
    nms_threshold: float        = Field(0.45, ge=0.0, le=1.0)
    input_width: int  = 640
    input_height: int = 640
    device: Literal["hailo", "cpu", "auto"] = "auto"
    # Maximum inference queue depth before dropping frames
    queue_maxsize: int = 4
    # COCO class names to detect (empty = all classes)
    target_classes: list[str] = []
    # Draw bounding boxes on streamed frames
    draw_overlays: bool = True
    overlay_thickness: int = 2


class LidarConfig(BaseModel):
    port: str   = "/dev/ttyACM0"
    baudrate: int = 115200
    min_range_m: float = Field(0.06, ge=0.0)
    max_range_m: float = Field(5.5,  ge=0.0)
    # URG-04LX angular range: −120° to +120° (240° total)
    angle_min_deg: float = -120.0
    angle_max_deg: float =  120.0
    # Max scan frequency (Hz); URG-04LX supports up to 10 Hz
    scan_frequency_hz: float = 10.0
    # Number of invalid reads before reconnect attempt
    reconnect_threshold: int = 5


class FusionConfig(BaseModel):
    # Horizontal field of view of camera (must match CameraConfig.hfov_degrees)
    camera_hfov_degrees: float = 66.0
    # Object within this distance is HIGH_THREAT
    threat_distance_m: float = 1.0
    # Object within this distance is MEDIUM_THREAT
    warn_distance_m: float = 2.0
    # Width of LiDAR sector to average for depth estimate (degrees)
    fusion_sector_width_deg: float = 5.0
    # Minimum overlap fraction to match detection bbox to LiDAR sector
    min_overlap_fraction: float = 0.3
    # IOU threshold for object tracker
    tracker_iou_threshold: float = 0.3
    # Frames to retain a tracked object without a new match
    tracker_max_missed: int = 10


class TelemetryConfig(BaseModel):
    # Interval between telemetry broadcasts (seconds)
    interval_seconds: float = 1.0
    # Enable MQTT publishing
    mqtt_enabled: bool = False
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "edge-ai/nav"
    mqtt_username: str = ""
    mqtt_password: str = Field(
        default_factory=lambda: os.environ.get("EDGE_AI_MQTT_PASSWORD", "")
    )


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str  = "logs"
    max_bytes: int = 10 * 1024 * 1024   # 10 MB
    backup_count: int = 5
    json_format: bool = True


# ─────────────────────────────────────────────
# Root config model
# ─────────────────────────────────────────────

class AppConfig(BaseModel):
    api:       APIConfig       = Field(default_factory=lambda: APIConfig())
    camera:    CameraConfig    = Field(default_factory=lambda: CameraConfig())
    inference: InferenceConfig = Field(default_factory=lambda: InferenceConfig())
    lidar:     LidarConfig     = Field(default_factory=lambda: LidarConfig())
    fusion:    FusionConfig    = Field(default_factory=lambda: FusionConfig())
    telemetry: TelemetryConfig = Field(default_factory=lambda: TelemetryConfig())
    logging:   LoggingConfig   = Field(default_factory=lambda: LoggingConfig())

    @field_validator("*", mode="before")
    @classmethod
    def _allow_none_to_default(cls, v):
        # Allow top-level keys to be None in YAML → use defaults
        return v if v is not None else {}


# ─────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────

def load_config(path: Path | None = None) -> AppConfig:
    """
    Load configuration from YAML file, then apply environment variable overrides.

    Environment variables:
      EDGE_AI_API_KEY           → api.api_key
      EDGE_AI_API_PORT          → api.port
      EDGE_AI_CAMERA_FPS        → camera.fps
      EDGE_AI_LIDAR_PORT        → lidar.port
      EDGE_AI_INFERENCE_DEVICE  → inference.device
      EDGE_AI_LOG_LEVEL         → logging.level
      EDGE_AI_MQTT_PASSWORD     → telemetry.mqtt_password
    """
    config_path = path or _CONFIG_PATH
    raw: dict = {}

    if config_path.exists():
        with config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info("Config loaded from %s", config_path)
    else:
        logger.warning(
            "Config file %s not found — using defaults.", config_path
        )

    # Build config from YAML data
    cfg = AppConfig.model_validate(raw)

    # Apply environment variable overrides
    _apply_env_overrides(cfg)

    return cfg


def _apply_env_overrides(cfg: AppConfig) -> None:
    """Mutate config in-place based on environment variables."""
    env_map = {
        "EDGE_AI_API_PORT":           ("api",       "port",               int),
        "EDGE_AI_API_KEY":            ("api",       "api_key",            str),
        "EDGE_AI_CAMERA_FPS":         ("camera",    "fps",                int),
        "EDGE_AI_CAMERA_WIDTH":       ("camera",    "width",              int),
        "EDGE_AI_CAMERA_HEIGHT":      ("camera",    "height",             int),
        "EDGE_AI_LIDAR_PORT":         ("lidar",     "port",               str),
        "EDGE_AI_INFERENCE_DEVICE":   ("inference", "device",             str),
        "EDGE_AI_CONFIDENCE":         ("inference", "confidence_threshold", float),
        "EDGE_AI_LOG_LEVEL":          ("logging",   "level",              str),
        "EDGE_AI_TELEMETRY_INTERVAL": ("telemetry", "interval_seconds",   float),
    }

    for env_key, (section, field, cast) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(getattr(cfg, section), field, cast(val))
                logger.debug("Env override: %s.%s = %s", section, field, val)
            except (ValueError, AttributeError) as exc:
                logger.warning("Invalid env override %s=%s: %s", env_key, val, exc)
