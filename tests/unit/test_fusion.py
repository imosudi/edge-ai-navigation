"""
tests/unit/test_fusion.py
Unit tests for the sensor fusion engine.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Minimal config stubs ──────────────────────────────────────────────────────

class _FusionCfg:
    camera_hfov_degrees    = 66.0
    threat_distance_m      = 1.0
    warn_distance_m        = 2.0
    fusion_sector_width_deg = 5.0
    min_overlap_fraction   = 0.3
    tracker_iou_threshold  = 0.3
    tracker_max_missed     = 3


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSensorFusion:
    """Tests for fusion/sensor_fusion.py"""

    def setup_method(self):
        from fusion.sensor_fusion import SensorFusion
        self.fusion = SensorFusion(_FusionCfg())

    def test_bbox_to_bearing_centre(self):
        """A centred bounding box should yield 0° bearing."""
        bbox = [0.25, 0.3, 0.75, 0.7]   # centre_x = 0.5
        bearing = self.fusion._bbox_to_bearing(bbox)
        assert abs(bearing) < 0.1

    def test_bbox_to_bearing_left(self):
        """Left-side bbox → negative bearing (left of centre)."""
        bbox = [0.0, 0.3, 0.2, 0.7]     # centre_x ≈ 0.1 → left
        bearing = self.fusion._bbox_to_bearing(bbox)
        assert bearing < -10.0

    def test_bbox_to_bearing_right(self):
        """Right-side bbox → positive bearing."""
        bbox = [0.8, 0.3, 1.0, 0.7]
        bearing = self.fusion._bbox_to_bearing(bbox)
        assert bearing > 10.0

    def test_bearing_to_direction_centre(self):
        from fusion.sensor_fusion import SensorFusion
        assert SensorFusion._bearing_to_direction(0.0)   == "centre"
        assert SensorFusion._bearing_to_direction(10.0)  == "centre"
        assert SensorFusion._bearing_to_direction(-10.0) == "centre"

    def test_bearing_to_direction_left_right(self):
        from fusion.sensor_fusion import SensorFusion
        assert SensorFusion._bearing_to_direction(-30.0) == "left"
        assert SensorFusion._bearing_to_direction(30.0)  == "right"

    def test_classify_threat_high(self):
        assert self.fusion._classify_threat(0.5) == "HIGH"

    def test_classify_threat_medium(self):
        assert self.fusion._classify_threat(1.5) == "MEDIUM"

    def test_classify_threat_low(self):
        assert self.fusion._classify_threat(3.0) == "LOW"

    def test_classify_threat_unknown(self):
        assert self.fusion._classify_threat(None) == "UNKNOWN"

    def test_iou_identical_boxes(self):
        from fusion.sensor_fusion import _iou
        box = [0.1, 0.1, 0.5, 0.5]
        assert abs(_iou(box, box) - 1.0) < 1e-6

    def test_iou_non_overlapping(self):
        from fusion.sensor_fusion import _iou
        a = [0.0, 0.0, 0.3, 0.3]
        b = [0.7, 0.7, 1.0, 1.0]
        assert _iou(a, b) == 0.0

    def test_iou_partial_overlap(self):
        from fusion.sensor_fusion import _iou
        a = [0.0, 0.0, 0.5, 0.5]
        b = [0.25, 0.25, 0.75, 0.75]
        iou = _iou(a, b)
        assert 0.0 < iou < 1.0

    def test_tracking_creates_new_object(self):
        """New detection should create a TrackedObject."""
        dets = [{"class_name": "person", "bbox": [0.3, 0.1, 0.7, 0.9],
                 "confidence": 0.85}]
        tracks = self.fusion._fuse(dets, {}, [])
        assert len(tracks) == 1
        assert tracks[0].class_name == "person"

    def test_tracking_removes_stale_tracks(self):
        """Tracks not matched for max_missed frames should be removed."""
        # Prime the tracker
        dets = [{"class_name": "person", "bbox": [0.3, 0.1, 0.7, 0.9],
                 "confidence": 0.85}]
        self.fusion._fuse(dets, {}, [])

        # Now stop detecting - run missed_max + 1 times
        for _ in range(_FusionCfg.tracker_max_missed + 2):
            self.fusion._fuse([], {}, [])

        assert len(self.fusion._tracks) == 0

    def test_distance_estimate_with_sector_map(self):
        """Should return the closest sector within the bearing window."""
        sector_angles = [-30.0, 0.0, 30.0]
        sector_map = {0: 3.0, 1: 1.2, 2: 4.0}
        dist = self.fusion._estimate_distance(0.0, sector_map, sector_angles)
        assert dist == pytest.approx(1.2, abs=0.01)

    def test_navigation_command_default_clear(self):
        """No threats/obstacles → FORWARD, speed 1.0."""
        nav = self.fusion._generate_navigation_command([], {}, [])
        assert nav["action"] == "MOVE_FORWARD"
        assert nav["speed"] == 1.0
        assert "Path clear" in nav["reason"]

    def test_navigation_command_high_threat_centre(self):
        """High threat in centre → Steer to avoid."""
        from fusion.sensor_fusion import TrackedObject
        det = {"class_name": "person", "bbox": [0.4, 0.1, 0.6, 0.9], "confidence": 0.85}
        track = TrackedObject(det)
        track.threat_level = "HIGH"
        track.direction = "centre"
        track.distance_m = 0.5

        # Mock sector map where right is clearer than left
        sector_angles = [-30.0, 30.0]
        sector_map = {0: 0.5, 1: 3.0}

        nav = self.fusion._generate_navigation_command([track], sector_map, sector_angles)
        assert nav["action"] == "STEER_RIGHT"
        assert nav["speed"] == 0.25
        assert "Avoid" in nav["reason"]

    def test_navigation_command_lidar_avoidance(self):
        """LiDAR detects close obstacle in front → Steer to avoid."""
        # LiDAR sector angles straight ahead
        sector_angles = [-45.0, 0.0, 45.0]
        # Obstacle in front (distance = 0.5m), left is blocked (0.2m), right is clear (4.0m)
        sector_map = {0: 0.2, 1: 0.5, 2: 4.0}

        nav = self.fusion._generate_navigation_command([], sector_map, sector_angles)
        assert nav["action"] == "STEER_RIGHT"
        assert nav["speed"] == 0.25


class TestScanProcessor:
    """Tests for lidar/scan_processor.py"""

    def setup_method(self):
        class _LidarCfg:
            angle_min_deg = -120.0
            angle_max_deg  =  120.0
            min_range_m    =  0.06
            max_range_m    =  5.5
            reconnect_threshold = 5

        from lidar.scan_processor import ScanProcessor
        driver  = MagicMock()
        ws_mgr  = MagicMock()
        self.proc = ScanProcessor(driver, ws_mgr, _LidarCfg())

    def test_sector_width_computed(self):
        """240° FOV / 72 sectors = 3.33° per sector."""
        assert abs(self.proc._sector_width - 240.0 / 72) < 0.01

    def test_process_scan_returns_required_keys(self):
        scan = {
            "angles":     list(range(-60, 61, 1)),
            "distances":  [2.0] * 121,
            "intensities": [1.0] * 121,
            "timestamp":  1.0,
            "scan_count": 1,
        }
        result = self.proc._process_scan(scan)
        for key in ("angles_deg", "distances_m", "sectors", "obstacle_zones",
                    "min_distance_m", "timestamp"):
            assert key in result

    def test_obstacle_detection_high_threat(self):
        """Close cluster should produce a HIGH threat zone."""
        import numpy as np
        angles = np.array([d * 1.0 for d in range(-5, 6)], dtype=np.float32)
        dists  = np.array([0.2] * len(angles), dtype=np.float32)
        zones  = self.proc._detect_obstacle_zones(angles, dists)
        assert any(z["threat_level"] == "HIGH" for z in zones)

    @pytest.mark.asyncio
    async def test_run_updates_driver_latest_scan(self):
        """ScanProcessor.run should update driver.latest_scan with processed fields."""
        import asyncio
        from unittest.mock import AsyncMock

        self.proc._driver.read_scan = AsyncMock(return_value={
            "angles": [0.0],
            "distances": [1.5],
            "intensities": [1.0],
            "timestamp": 123.45,
            "scan_count": 5,
        })
        self.proc._driver.latest_scan = {}
        self.proc._ws_manager.connection_count.return_value = 0

        shutdown_event = asyncio.Event()

        async def stop_soon():
            await asyncio.sleep(0.01)
            shutdown_event.set()

        asyncio.create_task(stop_soon())
        await self.proc.run(shutdown_event)

        assert "sectors" in self.proc._driver.latest_scan
        assert "sector_angles" in self.proc._driver.latest_scan
        assert self.proc._driver.latest_scan["distances"] == [1.5]
        assert self.proc._driver.latest_scan["scan_count"] == 5


class TestHailoEngine:
    """Tests for inference/hailo_engine.py - CPU path only."""

    def test_coco_classes_length(self):
        from inference.hailo_engine import COCO_CLASSES
        assert len(COCO_CLASSES) == 80

    def test_detection_to_dict(self):
        from inference.hailo_engine import Detection
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.91,
            bbox_xyxy=(0.1, 0.2, 0.5, 0.8),
        )
        d = det.to_dict()
        assert d["class_name"] == "person"
        assert 0.0 <= d["confidence"] <= 1.0
        assert len(d["bbox"]) == 4


class TestWebSocketManager:
    """Tests for app/websocket/stream.py"""

    @pytest.mark.asyncio
    async def test_broadcast_json_no_connections(self):
        from app.websocket.stream import WebSocketManager
        mgr = WebSocketManager()
        # Should not raise even with no connections
        await mgr.broadcast_json({"key": "val"}, channel="telemetry")

    @pytest.mark.asyncio
    async def test_broadcast_bytes_prunes_dead(self):
        from app.websocket.stream import WebSocketManager
        mgr = WebSocketManager()

        # Inject a dead WebSocket mock that raises on send
        dead_ws = AsyncMock()
        dead_ws.send_bytes = AsyncMock(side_effect=Exception("Connection closed"))
        mgr._connections["camera"].append(dead_ws)

        await mgr.broadcast_bytes(b"frame", channel="camera")
        # Dead connection should be pruned
        assert len(mgr._connections["camera"]) == 0

    def test_stats(self):
        from app.websocket.stream import WebSocketManager
        mgr = WebSocketManager()
        stats = mgr.stats()
        assert "connections" in stats
        assert "broadcasts" in stats


class TestConfigLoader:
    """Tests for config/config_loader.py"""

    def test_default_config_valid(self):
        from config.config_loader import AppConfig
        cfg = AppConfig()
        assert cfg.api.port == 8080
        assert cfg.camera.fps == 30
        assert cfg.lidar.max_range_m == 5.5

    def test_confidence_threshold_clamped(self):
        from config.config_loader import InferenceConfig
        cfg = InferenceConfig(confidence_threshold=0.5)
        assert cfg.confidence_threshold == 0.5

    def test_load_from_dict(self):
        from config.config_loader import AppConfig
        raw = {"api": {"port": 9090}, "camera": {"fps": 60}}
        cfg = AppConfig.model_validate(raw)
        assert cfg.api.port == 9090
        assert cfg.camera.fps == 60

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EDGE_AI_API_PORT", "9999")
        import pathlib

        from config.config_loader import load_config
        cfg = load_config(path=pathlib.Path("/nonexistent_path.yaml"))
        assert cfg.api.port == 9999

    def test_dashboard_port_override(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PORT", "8888")
        import pathlib

        from config.config_loader import load_config
        cfg = load_config(path=pathlib.Path("/nonexistent_path.yaml"))
        assert cfg.api.port == 8888

    def test_mqtt_password_override(self, monkeypatch):
        monkeypatch.setenv("EDGE_AI_MQTT_PASSWORD", "secret_pass")
        import pathlib

        from config.config_loader import load_config
        cfg = load_config(path=pathlib.Path("/nonexistent_path.yaml"))
        assert cfg.telemetry.mqtt_password == "secret_pass"


class TestRateLimiter:
    """Tests for app/middleware/rate_limit.py"""

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        from app.middleware.rate_limit import RateLimitMiddleware
        middleware = RateLimitMiddleware(app=None, max_requests=10)

        req = MagicMock()
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "127.0.0.1"
        req.url.path = "/api/v1/status"

        # Under limit - should not rate-limit
        call_next = AsyncMock(return_value=MagicMock(status_code=200))
        result = await middleware.dispatch(req, call_next)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_blocks_excess_requests(self):
        from app.middleware.rate_limit import RateLimitMiddleware
        middleware = RateLimitMiddleware(app=None, max_requests=2)

        req = MagicMock()
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "10.0.0.1"
        req.url.path = "/api/v1/test"

        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        # First two pass
        await middleware.dispatch(req, call_next)
        await middleware.dispatch(req, call_next)

        # Third should be rate-limited (429)
        response = await middleware.dispatch(req, call_next)
        assert response.status_code == 429


class TestTelemetryFPSCounter:
    """Tests for telemetry/metrics.py FPSCounter"""

    def test_fps_zero_initially(self):
        from telemetry.metrics import FPSCounter
        c = FPSCounter()
        assert c.fps == 0.0

    def test_fps_accumulates(self):
        import time

        from telemetry.metrics import FPSCounter
        c = FPSCounter(window=2.0)
        for _ in range(10):
            c.tick()
            time.sleep(0.01)
        assert c.fps > 0.0

    def test_fps_is_approximately_correct(self):
        import time

        from telemetry.metrics import FPSCounter
        c = FPSCounter(window=2.0)
        # Simulate ~20 fps
        for _ in range(20):
            c.tick()
            time.sleep(1.0 / 20)
        # Allow ±5 fps tolerance
        assert 15.0 <= c.fps <= 25.0
