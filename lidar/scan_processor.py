"""
lidar/scan_processor.py
LiDAR scan processing pipeline.

Responsibilities:
  1. Continuously read scans from URGLidarDriver
  2. Apply statistical outlier rejection
  3. Build sector-averaged range map (used by fusion engine)
  4. Generate polar obstacle map payload for the dashboard
  5. Broadcast scan data on "lidar" WebSocket channel
  6. Update fps_lidar telemetry counter

Obstacle map format (broadcast JSON):
  {
    "angles_deg":    [float, ...],
    "distances_m":   [float, ...],
    "intensities":   [float, ...],
    "sectors":       {angle_key: min_distance_m, ...},
    "obstacle_zones": [{angle, distance, threat_level}, ...],
    "timestamp":     float,
    "scan_count":    int
  }
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from app.websocket.stream import WebSocketManager
    from config.config_loader import LidarConfig
    from lidar.urg_driver import URGLidarDriver

logger = logging.getLogger(__name__)

# Number of sectors to split the FOV into for fusion matching
_N_SECTORS = 72    # 240° / 72 = 3.33° per sector


class ScanProcessor:
    """
    Processes raw LiDAR scans into structured data for fusion and dashboard.

    Also maintains the sector range map consumed by the sensor fusion engine.
    """

    def __init__(
        self,
        driver: URGLidarDriver,
        ws_manager: WebSocketManager,
        config: LidarConfig,
    ) -> None:
        self._driver     = driver
        self._ws_manager = ws_manager
        self._cfg        = config

        # Sector range map: sector_index → minimum distance (m) in that sector
        self.sector_range_m: dict[int, float] = {}

        # Pre-compute sector angle boundaries
        total_fov = config.angle_max_deg - config.angle_min_deg
        self._sector_width = total_fov / _N_SECTORS
        self._sector_angles = [
            config.angle_min_deg + (i + 0.5) * self._sector_width
            for i in range(_N_SECTORS)
        ]

    async def run(self, shutdown_event: asyncio.Event) -> None:
        """Main scan processing loop."""
        from telemetry.metrics import fps_lidar

        logger.info("LiDAR scan processor started.")

        consecutive_errors = 0

        while not shutdown_event.is_set():
            scan = await self._driver.read_scan()

            if scan is None:
                consecutive_errors += 1
                if consecutive_errors >= self._cfg.reconnect_threshold:
                    logger.warning(
                        "LiDAR: %d consecutive errors - attempting reconnect.",
                        consecutive_errors,
                    )
                    await self._attempt_reconnect()
                    consecutive_errors = 0
                await asyncio.sleep(0.05)
                continue

            consecutive_errors = 0
            fps_lidar.tick()

            # Process and enrich scan
            processed = self._process_scan(scan)

            # Update shared sector map (for fusion engine)
            self.sector_range_m = processed["sectors"]

            # Update driver's latest scan for sensor fusion and REST API compatibility
            self._driver.latest_scan = {
                "angles": processed["angles_deg"],
                "distances": processed["distances_m"],
                "intensities": processed["intensities"],
                "sectors": processed["sectors"],
                "sector_angles": processed["sector_angles"],
                "timestamp": processed["timestamp"],
                "scan_count": processed["scan_count"],
            }

            # Broadcast to WebSocket clients
            if self._ws_manager.connection_count("lidar") > 0:
                await self._ws_manager.broadcast_json(processed, channel="lidar")

        logger.info("LiDAR scan processor stopped.")

    # ── Processing ──────────────────────────────────────────────────────────

    def _process_scan(self, scan: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich a raw scan with:
          - Outlier-rejected angle/distance arrays
          - Sector-averaged range map
          - Obstacle zone list with threat levels
        """
        angles    = np.array(scan["angles"],    dtype=np.float32)
        distances = np.array(scan["distances"], dtype=np.float32)
        intensities = np.array(scan.get("intensities", [1.0] * len(angles)), dtype=np.float32)

        # ── Statistical outlier rejection ────────────────────────────────
        if len(distances) > 10:
            mean = np.mean(distances)
            std  = np.std(distances)
            mask = np.abs(distances - mean) < 3.0 * std
            angles      = angles[mask]
            distances   = distances[mask]
            intensities = intensities[mask]

        # ── Build sector range map ───────────────────────────────────────
        sectors: dict[int, float] = {}
        for i in range(_N_SECTORS):
            lo = self._cfg.angle_min_deg + i * self._sector_width
            hi = lo + self._sector_width
            mask = (angles >= lo) & (angles < hi)
            if np.any(mask):
                sectors[i] = float(np.min(distances[mask]))
            else:
                sectors[i] = float(self._cfg.max_range_m)

        # ── Obstacle zone detection ──────────────────────────────────────
        obstacle_zones = self._detect_obstacle_zones(angles, distances)

        return {
            "angles_deg":     angles.tolist(),
            "distances_m":    distances.tolist(),
            "intensities":    intensities.tolist(),
            "sectors":        sectors,
            "sector_angles":  self._sector_angles,
            "obstacle_zones": obstacle_zones,
            "min_distance_m": float(np.min(distances)) if len(distances) > 0 else self._cfg.max_range_m,
            "timestamp":      scan["timestamp"],
            "scan_count":     scan["scan_count"],
        }

    def _detect_obstacle_zones(
        self,
        angles: np.ndarray,
        distances: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Identify clusters of nearby points and classify threat level.

        Threat classification:
          HIGH   - distance < cfg.lidar.min_range_m * 5  (arbitrary: 0.3 m)
          MEDIUM - distance < cfg.lidar.max_range_m / 3
          LOW    - everything else within range
        """
        zones: list[dict[str, Any]] = []
        if len(distances) == 0:
            return zones

        # Simple clustering: group consecutive points within distance threshold
        cluster_gap_m = 0.3      # Start new cluster if gap > 30 cm
        cluster_min_points = 3   # Ignore noise clusters

        sorted_idx = np.argsort(angles)
        s_angles   = angles[sorted_idx]
        s_dists    = distances[sorted_idx]

        cluster_dists:  list[float] = []
        cluster_angles: list[float] = []

        def _flush_cluster() -> None:
            if len(cluster_dists) < cluster_min_points:
                return
            min_d = min(cluster_dists)
            mean_a = sum(cluster_angles) / len(cluster_angles)

            if min_d < 0.3:
                threat = "HIGH"
            elif min_d < self._cfg.max_range_m / 3:
                threat = "MEDIUM"
            else:
                threat = "LOW"

            zones.append({
                "angle_deg":    round(mean_a, 1),
                "distance_m":   round(min_d, 3),
                "point_count":  len(cluster_dists),
                "threat_level": threat,
            })

        for i in range(len(s_dists)):
            if not cluster_dists:
                cluster_dists.append(s_dists[i])
                cluster_angles.append(s_angles[i])
                continue

            gap = abs(s_dists[i] - cluster_dists[-1])
            if gap > cluster_gap_m:
                _flush_cluster()
                cluster_dists  = []
                cluster_angles = []

            cluster_dists.append(float(s_dists[i]))
            cluster_angles.append(float(s_angles[i]))

        _flush_cluster()
        return zones

    async def _attempt_reconnect(self) -> None:
        """Try to reconnect the LiDAR after persistent errors."""
        try:
            await self._driver.disconnect()
            await asyncio.sleep(2.0)
            await self._driver.connect()
            logger.info("LiDAR reconnected successfully.")
        except Exception as exc:
            logger.error("LiDAR reconnect failed: %s", exc)
            await asyncio.sleep(5.0)

    def get_sector_distance(self, angle_deg: float, width_deg: float = 5.0) -> float | None:
        """
        Return the minimum distance within a sector centred on `angle_deg`.

        Used by the fusion engine to estimate object depth from camera detections.
        """
        if not self.sector_range_m:
            return None

        half = width_deg / 2.0
        lo = angle_deg - half
        hi = angle_deg + half

        relevant = [
            dist for i, dist in self.sector_range_m.items()
            if lo <= self._sector_angles[i] <= hi
        ]
        return min(relevant) if relevant else None
