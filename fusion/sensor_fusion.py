"""
fusion/sensor_fusion.py
Camera + LiDAR sensor fusion engine.

Fusion algorithm:
  1. For each YOLO detection (bounding box in image coordinates):
     a. Compute the detection's horizontal centre pixel
     b. Map centre pixel → camera bearing angle (degrees) using HFoV
     c. Query the LiDAR sector range map for that bearing ± fusion_sector_width
     d. Assign distance estimate
     e. Classify threat level (HIGH / MEDIUM / LOW / CLEAR)
  2. Run a simple bounding-box IOU tracker across frames
  3. Broadcast fused object list on "fusion" WebSocket channel

Output per fused object:
  {
    "track_id":     int,
    "class_name":   str,
    "confidence":   float,
    "bbox":         [x1, y1, x2, y2],      # normalised [0–1]
    "bearing_deg":  float,                  # degrees from camera centre
    "distance_m":   float | None,
    "direction":    str,                    # "left" | "centre" | "right"
    "threat_level": str,                    # "HIGH" | "MEDIUM" | "LOW" | "CLEAR"
    "timestamp":    float
  }
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.websocket.stream import WebSocketManager
    from camera.capture import CameraCapture
    from config.config_loader import FusionConfig
    from lidar.urg_driver import URGLidarDriver

logger = logging.getLogger(__name__)


class TrackedObject:
    """Single tracked object with IOU-based matching."""

    _next_id = 0

    def __init__(self, detection: dict[str, Any]) -> None:
        TrackedObject._next_id += 1
        self.track_id    = TrackedObject._next_id
        self.class_name  = detection["class_name"]
        self.bbox        = detection["bbox"]       # normalised [x1,y1,x2,y2]
        self.confidence  = detection["confidence"]
        self.missed      = 0
        self.distance_m: float | None = None
        self.bearing_deg: float = 0.0
        self.threat_level = "CLEAR"
        self.direction    = "centre"
        self.last_seen    = time.time()

    def update(self, detection: dict[str, Any]) -> None:
        """Update from a matched detection."""
        self.bbox        = detection["bbox"]
        self.confidence  = detection["confidence"]
        self.missed      = 0
        self.last_seen   = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id":    self.track_id,
            "class_name":  self.class_name,
            "confidence":  round(self.confidence, 4),
            "bbox":        [round(v, 4) for v in self.bbox],
            "bearing_deg": round(self.bearing_deg, 2),
            "distance_m":  round(self.distance_m, 3) if self.distance_m is not None else None,
            "direction":   self.direction,
            "threat_level": self.threat_level,
            "timestamp":   self.last_seen,
        }


def _iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two bounding boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SensorFusion:
    """
    Fuses camera detections with LiDAR range data.

    Maintains a list of tracked objects with persistent IDs across frames.
    """

    def __init__(self, cfg: FusionConfig) -> None:
        self._cfg     = cfg
        self._tracks: list[TrackedObject] = []
        self.latest_objects: list[dict[str, Any]] = []

    async def run(
        self,
        camera: CameraCapture,
        lidar:  URGLidarDriver | None,
        ws_manager: WebSocketManager,
        shutdown_event: asyncio.Event,
    ) -> None:
        """
        Main fusion loop.  Runs at ~10 Hz (tied to LiDAR scan rate).
        """
        from telemetry.metrics import fps_fusion

        logger.info("Sensor fusion engine started.")

        # Import here to avoid circular imports at module load

        # Get reference to the scan processor's sector map
        # (shared via the lidar object's last scan)
        while not shutdown_event.is_set():
            try:
                detections = list(camera.latest_detections)

                # Get current LiDAR sector map from latest_scan
                sector_map: dict[int, float] = {}
                sector_angles: list[float] = []
                if lidar is not None:
                    raw_scan = lidar.latest_scan
                    sector_map    = raw_scan.get("sectors", {})
                    sector_angles = raw_scan.get("sector_angles", [])

                fused = self._fuse(detections, sector_map, sector_angles)
                self.latest_objects = [obj.to_dict() for obj in fused]

                fps_fusion.tick()

                if ws_manager.connection_count("fusion") > 0:
                    await ws_manager.broadcast_json(
                        {"objects": self.latest_objects, "timestamp": time.time()},
                        channel="fusion",
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Fusion error: %s", exc, exc_info=True)

            # Run at approximately lidar scan rate
            await asyncio.sleep(0.1)

        logger.info("Sensor fusion engine stopped.")

    # ── Fusion logic ────────────────────────────────────────────────────────

    def _fuse(
        self,
        detections: list[dict[str, Any]],
        sector_map: dict[int, float],
        sector_angles: list[float],
    ) -> list[TrackedObject]:
        """
        Match current detections to existing tracks, update distances and threats.

        Args:
            detections:    Latest camera detection dicts.
            sector_map:    sector_index → min_distance_m from LiDAR.
            sector_angles: Centre angle (deg) for each sector.

        Returns:
            List of currently active TrackedObject instances.
        """
        # ── Step 1: Match detections to existing tracks via IOU ──────────
        matched_track_ids: set[int] = set()
        matched_det_idxs:  set[int] = set()

        for ti, track in enumerate(self._tracks):
            best_iou   = self._cfg.tracker_iou_threshold
            best_di    = -1

            for di, det in enumerate(detections):
                if di in matched_det_idxs:
                    continue
                if det["class_name"] != track.class_name:
                    continue
                iou = _iou(track.bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_di  = di

            if best_di >= 0:
                self._tracks[ti].update(detections[best_di])
                matched_track_ids.add(ti)
                matched_det_idxs.add(best_di)

        # ── Step 2: Create new tracks for unmatched detections ───────────
        for di, det in enumerate(detections):
            if di not in matched_det_idxs:
                self._tracks.append(TrackedObject(det))

        # ── Step 3: Increment missed counter; prune dead tracks ──────────
        for ti in range(len(self._tracks) - 1, -1, -1):
            if ti not in matched_track_ids:
                self._tracks[ti].missed += 1
            if self._tracks[ti].missed > self._cfg.tracker_max_missed:
                self._tracks.pop(ti)

        # ── Step 4: Enrich each track with LiDAR distance & threat ───────
        for track in self._tracks:
            bearing = self._bbox_to_bearing(track.bbox)
            track.bearing_deg = bearing
            track.direction   = self._bearing_to_direction(bearing)

            dist = self._estimate_distance(bearing, sector_map, sector_angles)
            track.distance_m   = dist
            track.threat_level = self._classify_threat(dist)

        return self._tracks

    def _bbox_to_bearing(self, bbox: list[float]) -> float:
        """
        Convert normalised bbox centre-x to a horizontal bearing in degrees.

        bbox[0] = x1, bbox[2] = x2 (normalised 0–1)
        Bearing: negative = left, 0 = centre, positive = right
        """
        cx = (bbox[0] + bbox[2]) / 2.0          # 0 to 1
        # Map [0, 1] → [−HFoV/2, +HFoV/2]
        hfov = self._cfg.camera_hfov_degrees
        return (cx - 0.5) * hfov

    @staticmethod
    def _bearing_to_direction(bearing_deg: float) -> str:
        if bearing_deg < -15:
            return "left"
        if bearing_deg > 15:
            return "right"
        return "centre"

    def _estimate_distance(
        self,
        bearing_deg: float,
        sector_map: dict[int, float],
        sector_angles: list[float],
    ) -> float | None:
        """
        Find the nearest LiDAR sector to the camera bearing and return its range.
        """
        if not sector_map or not sector_angles:
            return None

        half = self._cfg.fusion_sector_width_deg / 2.0
        lo   = bearing_deg - half
        hi   = bearing_deg + half

        relevant_dists = [
            sector_map[i]
            for i, angle in enumerate(sector_angles)
            if lo <= angle <= hi and i in sector_map
        ]

        if not relevant_dists:
            # Widen search window × 3 before giving up
            lo = bearing_deg - half * 3
            hi = bearing_deg + half * 3
            relevant_dists = [
                sector_map[i]
                for i, angle in enumerate(sector_angles)
                if lo <= angle <= hi and i in sector_map
            ]

        return min(relevant_dists) if relevant_dists else None

    def _classify_threat(self, distance_m: float | None) -> str:
        """Classify threat level based on estimated distance."""
        if distance_m is None:
            return "UNKNOWN"
        if distance_m <= self._cfg.threat_distance_m:
            return "HIGH"
        if distance_m <= self._cfg.warn_distance_m:
            return "MEDIUM"
        return "LOW"
