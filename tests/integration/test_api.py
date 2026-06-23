"""
tests/integration/test_api.py
Integration tests for FastAPI REST endpoints.

Uses httpx.AsyncClient with the FastAPI test client (no real hardware).
Hardware components are mocked.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Mock hardware state ───────────────────────────────────────────────────────

def _mock_app_state():
    """Return a minimal mock application state."""
    state = MagicMock()

    # Camera mock
    camera = MagicMock()
    camera.latest_detections = [
        {"class_name": "person", "confidence": 0.92,
         "bbox": [0.1, 0.2, 0.5, 0.8], "class_id": 0}
    ]
    camera.latest_annotated = b""
    state.camera = camera

    # LiDAR mock
    lidar = MagicMock()
    lidar.latest_scan = {
        "angles": [0.0, 10.0, -10.0],
        "distances": [2.0, 1.5, 3.0],
        "intensities": [1.0, 1.0, 1.0],
        "timestamp": 1e9,
        "scan_count": 42,
    }
    state.lidar = lidar

    # Fusion mock
    fusion = MagicMock()
    fusion.latest_objects = [
        {
            "track_id": 1, "class_name": "person",
            "confidence": 0.92, "bbox": [0.1, 0.2, 0.5, 0.8],
            "bearing_deg": -2.0, "distance_m": 1.8,
            "direction": "centre", "threat_level": "MEDIUM",
            "timestamp": 1e9,
        }
    ]
    state.fusion = fusion

    # Telemetry mock
    telemetry = MagicMock()
    telemetry.snapshot.return_value = {
        "timestamp": 1e9, "uptime_seconds": 120.0,
        "cpu": {"percent": 32.0, "per_core": [30, 34], "count": 4, "freq_mhz": 2400.0},
        "memory": {"total_mb": 8192.0, "available_mb": 6144.0,
                   "used_mb": 2048.0, "percent": 25.0},
        "disk": {"total_gb": 32.0, "used_gb": 8.0, "free_gb": 24.0, "percent": 25.0},
        "temperature": {"cpu_c": 48.5, "gpu_c": 49.0},
        "network": {"recv_kbps": 120.0, "sent_kbps": 30.0},
        "process": {"memory_mb": 350.0, "cpu_percent": 12.0},
        "fps": {"camera": 29.8, "inference": 18.2, "lidar": 9.9, "fusion": 9.8},
    }
    telemetry.prometheus_export.return_value = (
        "# HELP edge_ai_cpu_percent CPU utilisation\n"
        "edge_ai_cpu_percent 32.0\n"
    )
    state.telemetry = telemetry

    # Hailo mock
    hailo = MagicMock()
    state.hailo = hailo

    # WebSocket manager mock
    ws_manager = MagicMock()
    ws_manager.stats.return_value = {"connections": {}, "broadcasts": {}}
    state.ws_manager = ws_manager

    # Config mock
    from config.config_loader import AppConfig
    cfg = AppConfig()
    cfg.api.require_api_key = False
    state.config = cfg

    return state


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_app():
    """Build FastAPI app with mocked hardware."""
    with patch("app.main.lifespan", _null_lifespan):
        from fastapi import FastAPI

        from app.api.routes import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.state = _mock_app_state()
        return app


@asynccontextmanager
async def _null_lifespan(app):
    app.state = _mock_app_state()
    yield


@pytest_asyncio.fixture
async def client(mock_app):
    async with AsyncClient(
        transport=ASGITransport(app=mock_app),
        base_url="http://test"
    ) as ac:
        yield ac


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_ok(self, client):
        r = await client.get("/api/v1/status")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "components" in data

    @pytest.mark.asyncio
    async def test_status_has_version(self, client):
        r = await client.get("/api/v1/status")
        assert r.json()["version"] == "1.0.0"


class TestDetectionsEndpoint:
    @pytest.mark.asyncio
    async def test_detections_returns_list(self, client):
        r = await client.get("/api/v1/detections")
        assert r.status_code == 200
        data = r.json()
        assert "detections" in data
        assert isinstance(data["detections"], list)
        assert len(data["detections"]) >= 1

    @pytest.mark.asyncio
    async def test_detection_fields(self, client):
        r = await client.get("/api/v1/detections")
        det = r.json()["detections"][0]
        assert "class_name" in det
        assert "confidence" in det
        assert "bbox" in det


class TestLidarEndpoint:
    @pytest.mark.asyncio
    async def test_lidar_scan_returns_arrays(self, client):
        r = await client.get("/api/v1/lidar/scan")
        assert r.status_code == 200
        data = r.json()
        assert "angles_deg" in data
        assert "distances_m" in data
        assert isinstance(data["angles_deg"], list)

    @pytest.mark.asyncio
    async def test_lidar_scan_count(self, client):
        r = await client.get("/api/v1/lidar/scan")
        assert r.json()["scan_count"] == 42


class TestFusionEndpoint:
    @pytest.mark.asyncio
    async def test_fusion_objects(self, client):
        r = await client.get("/api/v1/fusion/objects")
        assert r.status_code == 200
        data = r.json()
        assert "objects" in data
        assert len(data["objects"]) >= 1

    @pytest.mark.asyncio
    async def test_fused_object_fields(self, client):
        r = await client.get("/api/v1/fusion/objects")
        obj = r.json()["objects"][0]
        for field in ("track_id", "class_name", "confidence",
                      "bbox", "bearing_deg", "distance_m",
                      "direction", "threat_level"):
            assert field in obj, f"Missing field: {field}"


class TestTelemetryEndpoint:
    @pytest.mark.asyncio
    async def test_telemetry_snapshot(self, client):
        r = await client.get("/api/v1/telemetry")
        assert r.status_code == 200
        data = r.json()
        assert "cpu" in data
        assert "memory" in data
        assert "fps" in data

    @pytest.mark.asyncio
    async def test_cpu_percent_range(self, client):
        r = await client.get("/api/v1/telemetry")
        cpu = r.json()["cpu"]["percent"]
        assert 0.0 <= cpu <= 100.0


class TestConfigEndpoints:
    @pytest.mark.asyncio
    async def test_get_config(self, client):
        r = await client.get("/api/v1/config")
        assert r.status_code == 200
        data = r.json()
        assert "inference" in data
        assert "camera" in data

    @pytest.mark.asyncio
    async def test_patch_confidence(self, client):
        r = await client.post(
            "/api/v1/config",
            json={"confidence_threshold": 0.6}
        )
        assert r.status_code == 200
        assert "confidence_threshold" in r.json()["updated"]

    @pytest.mark.asyncio
    async def test_patch_invalid_confidence(self, client):
        r = await client.post(
            "/api/v1/config",
            json={"confidence_threshold": 1.5}   # > 1.0 - invalid
        )
        assert r.status_code == 422


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_prometheus_format(self, client):
        r = await client.get("/api/v1/metrics")
        assert r.status_code == 200
        assert "edge_ai_cpu_percent" in r.text
        assert "# HELP" in r.text
