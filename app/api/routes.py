"""
app/api/routes.py
REST API endpoints for the Edge AI Navigation System.

Endpoints:
  GET  /status          - system health & component status
  GET  /detections      - latest object detections
  GET  /lidar/scan      - latest LiDAR scan data
  GET  /fusion/objects  - latest fused objects (camera + LiDAR)
  GET  /telemetry       - current system telemetry snapshot
  GET  /config          - current runtime configuration (read-only)
  POST /config          - update runtime thresholds dynamically
  POST /snapshot        - save annotated frame to disk
  GET  /metrics         - Prometheus-compatible metrics endpoint
  WS   /ws/camera       - WebSocket: JPEG camera stream
  WS   /ws/lidar        - WebSocket: LiDAR scan JSON
  WS   /ws/fusion       - WebSocket: fused object JSON
  WS   /ws/telemetry    - WebSocket: telemetry JSON
  GET  /               - serve dashboard HTML
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from starlette.requests import HTTPConnection

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    components: dict[str, str]
    version: str = "1.0.0"


class ConfigPatch(BaseModel):
    confidence_threshold: float | None = Field(None, ge=0.0, le=1.0)
    nms_threshold: float | None = Field(None, ge=0.0, le=1.0)
    lidar_min_range_m: float | None = Field(None, ge=0.0)
    lidar_max_range_m: float | None = Field(None, ge=0.0)
    threat_distance_m: float | None = Field(None, ge=0.0)


class SnapshotResponse(BaseModel):
    path: str
    timestamp: float


# ─────────────────────────────────────────────
# Dependency: extract shared state from request
# ─────────────────────────────────────────────

def get_state(connection: HTTPConnection) -> Any:
    return connection.app.state


# ─────────────────────────────────────────────
# HTTP endpoints
# ─────────────────────────────────────────────

_start_time = time.monotonic()


@router.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status(state: Any = Depends(get_state)) -> StatusResponse:
    """Return live system health and component availability."""
    components = {
        "camera":  "ok" if getattr(state, "camera", None) else "unavailable",
        "hailo":   "ok" if getattr(state, "hailo",  None) else "cpu_fallback",
        "lidar":   "ok" if getattr(state, "lidar",  None) else "unavailable",
        "fusion":  "ok" if getattr(state, "fusion", None) else "unavailable",
    }
    return StatusResponse(
        status="degraded" if "unavailable" in components.values() else "ok",
        uptime_seconds=round(time.monotonic() - _start_time, 2),
        components=components,
    )
@router.get("/detections", tags=["Vision"])
async def get_detections(state: Any = Depends(get_state)) -> dict:
    """Return the most recent YOLO detection results."""
    camera = getattr(state, "camera", None)
    if camera is None:
        raise HTTPException(status_code=503, detail="Camera unavailable.")
    return {"detections": camera.latest_detections, "timestamp": time.time()}


@router.get("/lidar/scan", tags=["LiDAR"])
async def get_lidar_scan(state: Any = Depends(get_state)) -> dict:
    """Return the most recent LiDAR scan."""
    lidar = getattr(state, "lidar", None)
    if lidar is None:
        raise HTTPException(status_code=503, detail="LiDAR unavailable.")
    scan = lidar.latest_scan
    return {
        "angles_deg":    scan.get("angles", []),
        "distances_m":   scan.get("distances", []),
        "intensities":   scan.get("intensities", []),
        "timestamp":     scan.get("timestamp", time.time()),
        "scan_count":    scan.get("scan_count", 0),
    }


@router.get("/fusion/objects", tags=["Fusion"])
async def get_fused_objects(state: Any = Depends(get_state)) -> dict:
    """Return the most recently fused sensor objects."""
    fusion = getattr(state, "fusion", None)
    if fusion is None:
        raise HTTPException(status_code=503, detail="Fusion engine unavailable.")
    return {"objects": fusion.latest_objects, "timestamp": time.time()}


@router.get("/telemetry", tags=["Telemetry"])
async def get_telemetry(state: Any = Depends(get_state)) -> dict:
    """Return current system telemetry snapshot."""
    tel = getattr(state, "telemetry", None)
    if tel is None:
        raise HTTPException(status_code=503, detail="Telemetry unavailable.")
    return cast(dict[str, Any], tel.snapshot())


@router.get("/config", tags=["Configuration"])
async def get_config(state: Any = Depends(get_state)) -> dict:
    """Return current runtime configuration (read-only view)."""
    cfg = getattr(state, "config", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable.")
    return cast(dict[str, Any], cfg.model_dump(mode="json"))


@router.post("/config", tags=["Configuration"])
async def patch_config(patch: ConfigPatch, state: Any = Depends(get_state)) -> dict:
    """Dynamically update runtime thresholds without restarting."""
    cfg = getattr(state, "config", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable.")

    changed: dict[str, Any] = {}
    if patch.confidence_threshold is not None:
        cfg.inference.confidence_threshold = patch.confidence_threshold
        changed["confidence_threshold"] = patch.confidence_threshold
    if patch.nms_threshold is not None:
        cfg.inference.nms_threshold = patch.nms_threshold
        changed["nms_threshold"] = patch.nms_threshold
    if patch.lidar_min_range_m is not None:
        cfg.lidar.min_range_m = patch.lidar_min_range_m
        changed["lidar_min_range_m"] = patch.lidar_min_range_m
    if patch.lidar_max_range_m is not None:
        cfg.lidar.max_range_m = patch.lidar_max_range_m
        changed["lidar_max_range_m"] = patch.lidar_max_range_m
    if patch.threat_distance_m is not None:
        cfg.fusion.threat_distance_m = patch.threat_distance_m
        changed["threat_distance_m"] = patch.threat_distance_m

    logger.info("Runtime config updated: %s", changed)
    return {"updated": changed, "timestamp": time.time()}


@router.post("/snapshot", response_model=SnapshotResponse, tags=["Vision"])
async def take_snapshot(state: Any = Depends(get_state)) -> SnapshotResponse:
    """Save an annotated snapshot to the logs directory."""
    camera = getattr(state, "camera", None)
    if camera is None:
        raise HTTPException(status_code=503, detail="Camera unavailable.")

    snap_path = await camera.save_snapshot()
    return SnapshotResponse(path=str(snap_path), timestamp=time.time())


@router.get("/metrics", response_class=PlainTextResponse, tags=["Telemetry"])
async def prometheus_metrics(state: Any = Depends(get_state)) -> str:
    """Expose Prometheus-compatible metrics."""
    tel = getattr(state, "telemetry", None)
    if tel is None:
        return "# No telemetry available\n"
    return cast(str, tel.prometheus_export())


# ─────────────────────────────────────────────
# WebSocket endpoints
# ─────────────────────────────────────────────

@router.websocket("/ws/camera")
async def ws_camera(websocket: WebSocket, state: Any = Depends(get_state)) -> None:
    """Stream annotated JPEG frames to connected browser clients."""
    ws_manager = getattr(state, "ws_manager", None)
    if ws_manager is None:
        await websocket.close(code=1011)
        return
    await ws_manager.connect(websocket, channel="camera")
    try:
        while True:
            # Keep-alive ping
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel="camera")
    except Exception as exc:
        logger.warning("Camera WS error: %s", exc)
        ws_manager.disconnect(websocket, channel="camera")


@router.websocket("/ws/lidar")
async def ws_lidar(websocket: WebSocket, state: Any = Depends(get_state)) -> None:
    """Stream LiDAR scan JSON to connected clients."""
    ws_manager = getattr(state, "ws_manager", None)
    if ws_manager is None:
        await websocket.close(code=1011)
        return
    await ws_manager.connect(websocket, channel="lidar")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel="lidar")


@router.websocket("/ws/fusion")
async def ws_fusion(websocket: WebSocket, state: Any = Depends(get_state)) -> None:
    """Stream fused sensor objects JSON to connected clients."""
    ws_manager = getattr(state, "ws_manager", None)
    if ws_manager is None:
        await websocket.close(code=1011)
        return
    await ws_manager.connect(websocket, channel="fusion")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel="fusion")


@router.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket, state: Any = Depends(get_state)) -> None:
    """Stream system telemetry JSON to connected clients."""
    ws_manager = getattr(state, "ws_manager", None)
    if ws_manager is None:
        await websocket.close(code=1011)
        return
    await ws_manager.connect(websocket, channel="telemetry")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel="telemetry")
