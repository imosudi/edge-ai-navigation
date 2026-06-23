"""
edge-ai-navigation · app/main.py
FastAPI application entry point for the Edge AI Indoor Navigation System.

Architecture:
  - FastAPI async server (uvicorn)
  - WebSocket streams for camera, LiDAR, fused objects, telemetry
  - Background tasks: camera capture, LiDAR scan, inference, fusion
  - Graceful shutdown with resource cleanup
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.middleware.auth import APIKeyMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.websocket.stream import WebSocketManager
from camera.capture import CameraCapture
from config.config_loader import AppConfig, load_config
from fusion.sensor_fusion import SensorFusion
from inference.hailo_engine import HailoInferenceEngine
from lidar.urg_driver import URGLidarDriver
from telemetry.logger import setup_logging
from telemetry.metrics import TelemetryCollector

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Global component registry (shared via app.state)
# ─────────────────────────────────────────────

_shutdown_event: asyncio.Event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle manager."""
    cfg: AppConfig = app.state.config

    # ── Setup logging ──────────────────────────────────────────────────────
    setup_logging(cfg.logging)
    logger.info("Edge AI Navigation System starting up …")

    # ── Initialise hardware components ─────────────────────────────────────
    try:
        hailo = HailoInferenceEngine(cfg.inference)
        await hailo.initialise()
        app.state.hailo = hailo
        logger.info("Hailo-8L inference engine ready.")
    except Exception as exc:
        logger.error("Hailo initialisation failed: %s", exc)
        app.state.hailo = None  # graceful degradation to CPU

    try:
        camera = CameraCapture(cfg.camera)
        await camera.start()
        app.state.camera = camera
        logger.info("Camera capture pipeline ready.")
    except Exception as exc:
        logger.error("Camera initialisation failed: %s", exc)
        raise RuntimeError("Camera is required - cannot start.") from exc

    try:
        lidar = URGLidarDriver(cfg.lidar)
        await lidar.connect()
        app.state.lidar = lidar
        logger.info("Hokuyo LiDAR connected.")
    except Exception as exc:
        logger.warning("LiDAR initialisation failed (running without): %s", exc)
        app.state.lidar = None

    # ── Fusion engine ──────────────────────────────────────────────────────
    fusion = SensorFusion(cfg.fusion)
    app.state.fusion = fusion

    # ── WebSocket manager ──────────────────────────────────────────────────
    ws_manager = WebSocketManager()
    app.state.ws_manager = ws_manager

    # ── Telemetry collector ────────────────────────────────────────────────
    telemetry = TelemetryCollector(cfg.telemetry)
    app.state.telemetry = telemetry

    # ── Background processing tasks ────────────────────────────────────────
    tasks: list[asyncio.Task] = []

    async def run_camera_pipeline() -> None:
        from inference.yolo_pipeline import YOLOPipeline
        pipeline = YOLOPipeline(
            hailo_engine=app.state.hailo,
            camera=app.state.camera,
            ws_manager=ws_manager,
            config=cfg.inference,
        )
        await pipeline.run(_shutdown_event)

    async def run_lidar_pipeline() -> None:
        if app.state.lidar is None:
            return
        from lidar.scan_processor import ScanProcessor
        processor = ScanProcessor(
            driver=app.state.lidar,
            ws_manager=ws_manager,
            config=cfg.lidar,
        )
        await processor.run(_shutdown_event)

    async def run_fusion_pipeline() -> None:
        await fusion.run(
            camera=app.state.camera,
            lidar=app.state.lidar,
            ws_manager=ws_manager,
            shutdown_event=_shutdown_event,
        )

    async def run_telemetry() -> None:
        await telemetry.run(
            ws_manager=ws_manager,
            hailo=app.state.hailo,
            shutdown_event=_shutdown_event,
        )

    tasks.append(asyncio.create_task(run_camera_pipeline(),  name="camera"))
    tasks.append(asyncio.create_task(run_lidar_pipeline(),   name="lidar"))
    tasks.append(asyncio.create_task(run_fusion_pipeline(),  name="fusion"))
    tasks.append(asyncio.create_task(run_telemetry(),        name="telemetry"))

    logger.info("All background tasks started.  System is LIVE.")

    yield  # ← application serves requests here

    # ── Graceful shutdown ──────────────────────────────────────────────────
    logger.info("Shutdown signal received - stopping all tasks …")
    _shutdown_event.set()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    if app.state.camera:
        await app.state.camera.stop()
    if app.state.lidar:
        await app.state.lidar.disconnect()
    if app.state.hailo:
        await app.state.hailo.shutdown()

    logger.info("Edge AI Navigation System shut down cleanly.")


def create_application() -> FastAPI:
    """Factory that builds and configures the FastAPI application."""
    cfg = load_config()

    app = FastAPI(
        title="Edge AI Navigation System",
        description=(
            "Real-time edge AI perception system: "
            "YOLOv8 + Hailo-8L + Hokuyo LiDAR + sensor fusion dashboard."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    # Store config on app state so lifespan can access it
    app.state.config = cfg

    # ── Middleware ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, max_requests=cfg.api.rate_limit_rpm)
    if cfg.api.require_api_key:
        app.add_middleware(APIKeyMiddleware, api_key=cfg.api.api_key)

    # ── Routes ─────────────────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_root() -> FileResponse:
        """Serve the main dashboard HTML page."""
        index_path = Path("dashboard/templates/index.html")
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard not found.")
        return FileResponse(index_path)

    # ── Static files (dashboard UI) ────────────────────────────────────────
    app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

    # ── Signal handlers (graceful shutdown on SIGINT / SIGTERM) ───────────
    def _handle_signal(signum: int, frame: object) -> None:
        logger.info("Received signal %s - initiating shutdown.", signum)
        _shutdown_event.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    return app


app = create_application()


if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
        log_level=cfg.logging.level.lower(),
        workers=1,           # Single worker - hardware is shared state
        access_log=True,
        loop="asyncio",
    )
