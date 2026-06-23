"""
fusion/object_tracker.py
Multi-object tracker for the fusion engine.

Implements a lightweight IoU-based SORT-style tracker.
Optionally upgrades to DeepSORT-style embedding matching if
a descriptor model is available.

Public API:
    tracker = ObjectTracker(cfg)
    tracked = tracker.update(detections)   # list[TrackedObject]

Tracking logic:
  1. Predict next position via constant-velocity Kalman filter (optional)
  2. Match detections to existing tracks via Hungarian IoU assignment
  3. Update matched tracks; increment missed counter for unmatched
  4. Initialise new tracks for unmatched detections
  5. Prune tracks missing > max_missed frames
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """
    A single tracked object with persistent identity.

    Attributes:
        track_id:    Unique integer ID assigned at track creation.
        class_name:  COCO class label.
        bbox:        Most recent normalised bbox [x1, y1, x2, y2].
        confidence:  Detection confidence.
        age:         Total frames since creation.
        missed:      Consecutive frames without a match.
        history:     List of past bboxes for velocity estimation.
    """
    track_id:   int
    class_name: str
    bbox:       list[float]
    confidence: float = 0.0
    age:        int   = 0
    missed:     int   = 0
    history:    list[list[float]] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)

    # Enriched by fusion
    bearing_deg:  float       = 0.0
    distance_m:   float | None = None
    direction:    str          = "centre"
    threat_level: str          = "CLEAR"

    def update(self, bbox: list[float], confidence: float) -> None:
        self.history.append(self.bbox)
        if len(self.history) > 10:
            self.history.pop(0)
        self.bbox       = bbox
        self.confidence = confidence
        self.missed     = 0
        self.age       += 1
        self.last_seen  = time.time()

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Estimate (dx, dy) centre velocity from history."""
        if len(self.history) < 2:
            return None
        def _centre(b):
            return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        cx1, cy1 = _centre(self.history[-2])
        cx2, cy2 = _centre(self.history[-1])
        return (cx2 - cx1, cy2 - cy1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id":    self.track_id,
            "class_name":  self.class_name,
            "confidence":  round(self.confidence, 4),
            "bbox":        [round(v, 4) for v in self.bbox],
            "age":         self.age,
            "bearing_deg": round(self.bearing_deg, 2),
            "distance_m":  round(self.distance_m, 3) if self.distance_m is not None else None,
            "direction":   self.direction,
            "threat_level": self.threat_level,
            "velocity":    [round(v, 4) for v in self.velocity] if self.velocity else None,
            "timestamp":   self.last_seen,
        }


def _iou_matrix(tracks: list[Track], dets: list[dict]) -> np.ndarray:
    """
    Compute IoU between all track/detection pairs.

    Returns:
        Matrix of shape [len(tracks), len(dets)].
    """
    n, m = len(tracks), len(dets)
    mat  = np.zeros((n, m), dtype=np.float32)

    for i, tr in enumerate(tracks):
        for j, det in enumerate(dets):
            if tr.class_name != det["class_name"]:
                continue   # Never match different classes
            a, b = tr.bbox, det["bbox"]
            ix1 = max(a[0], b[0])
            iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2])
            iy2 = min(a[3], b[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union  = area_a + area_b - inter
            mat[i, j] = inter / union if union > 0 else 0.0

    return mat


def _hungarian_match(
    iou_mat: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Greedy IoU matching (O(n²)) - sufficient for real-time edge use.
    Returns (matches, unmatched_tracks, unmatched_dets).
    """
    matched: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_dets:   set[int] = set()

    # Sort all pairs by IoU descending
    pairs = [
        (iou_mat[i, j], i, j)
        for i in range(iou_mat.shape[0])
        for j in range(iou_mat.shape[1])
        if iou_mat[i, j] >= threshold
    ]
    pairs.sort(reverse=True)

    for iou, ti, di in pairs:
        if ti in used_tracks or di in used_dets:
            continue
        matched.append((ti, di))
        used_tracks.add(ti)
        used_dets.add(di)

    unmatched_tracks = [i for i in range(iou_mat.shape[0]) if i not in used_tracks]
    unmatched_dets   = [j for j in range(iou_mat.shape[1]) if j not in used_dets]

    return matched, unmatched_tracks, unmatched_dets


class ObjectTracker:
    """
    Stateful multi-object tracker.

    Args:
        iou_threshold:  Minimum IoU to assign a detection to a track.
        max_missed:     Frames before a track is pruned.
        min_age:        Minimum track age before it's reported (reduces FP noise).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 10,
        min_age: int = 2,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed    = max_missed
        self._min_age       = min_age
        self._tracks:  list[Track] = []
        self._next_id: int = 0

    def update(self, detections: list[dict[str, Any]]) -> list[Track]:
        """
        Update tracker state with a new set of detections.

        Args:
            detections: List of detection dicts with keys:
                        class_name, confidence, bbox ([x1,y1,x2,y2]).

        Returns:
            List of currently active tracks (age ≥ min_age).
        """
        if not self._tracks:
            for det in detections:
                self._tracks.append(self._new_track(det))
            return [t for t in self._tracks if t.age >= self._min_age]

        if not detections:
            for t in self._tracks:
                t.missed += 1
            self._prune()
            return [t for t in self._tracks if t.age >= self._min_age]

        # ── Hungarian matching ────────────────────────────────────────────
        iou_mat = _iou_matrix(self._tracks, detections)
        matched, unmatched_tracks, unmatched_dets = _hungarian_match(
            iou_mat, self._iou_threshold
        )

        for ti, di in matched:
            det = detections[di]
            self._tracks[ti].update(det["bbox"], det["confidence"])

        for ti in unmatched_tracks:
            self._tracks[ti].missed += 1

        for di in unmatched_dets:
            self._tracks.append(self._new_track(detections[di]))

        self._prune()

        return [t for t in self._tracks if t.age >= self._min_age]

    def _new_track(self, det: dict[str, Any]) -> Track:
        self._next_id += 1
        return Track(
            track_id=self._next_id,
            class_name=det["class_name"],
            bbox=det["bbox"],
            confidence=det.get("confidence", 0.0),
            age=1,
        )

    def _prune(self) -> None:
        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed]

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks if t.age >= self._min_age)
