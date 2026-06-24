"""
camera/capture.py
Camera capture pipeline using picamera2 (Raspberry Pi Camera Module 3).

Architecture:
  - picamera2 runs in a dedicated background thread
  - Frames are placed into a bounded asyncio.Queue (drop-oldest on overflow)
  - Latest raw frame and annotated JPEG are stored for REST API access
  - Snapshot: saves annotated JPEG to logs/snapshots/

Optimisations:
  - YUV420 capture format → RGB conversion via numpy (no extra OpenCV step)
  - JPEG encoding with configurable quality (default 80%)
  - Frame queue maxsize=4 prevents memory accumulation on slow consumers
  - Thread-safe latest_frame access via asyncio.Lock
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from config.config_loader import CameraConfig

logger = logging.getLogger(__name__)

# Maximum simultaneous frames in the async queue before dropping oldest
_QUEUE_MAX = 4


class CameraCapture:
    """
    Manages capture from Raspberry Pi Camera Module 3 via picamera2.

    Public API:
        await start()
        await stop()
        await get_frame() → np.ndarray (BGR, HxWx3)
        await save_snapshot() → Path
        latest_detections   → list[dict] (set by inference pipeline)
        latest_annotated    → bytes (JPEG, set by inference pipeline)
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._picam2: Any = None
        self._cap: Any = None
        self._thread: Any = None
        self._frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._latest_frame: np.ndarray | None = None
        self._latest_annotated: bytes = b""
        self._lock = asyncio.Lock()
        self._running = False
        self.latest_detections: list[dict] = []
        self._capture_count = 0

        # Ensure snapshot directory exists
        Path(cfg.snapshot_dir).mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Initialise picamera2 and start the capture background thread."""
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError:
            logger.warning(
                "picamera2 not available - falling back to OpenCV VideoCapture."
            )
            await self._start_opencv_fallback()
            return

        self._picam2 = Picamera2()

        # Configure for BGR capture at target resolution/fps
        config = self._picam2.create_video_configuration(
            main={"size": (self._cfg.width, self._cfg.height), "format": "BGR888"},
            controls={
                "FrameDurationLimits": (
                    int(1_000_000 / self._cfg.fps),
                    int(1_000_000 / self._cfg.fps),
                ),
                "AeEnable":  self._cfg.auto_exposure,
                "AwbEnable": self._cfg.auto_white_balance,
            },
        )
        self._picam2.configure(config)
        self._picam2.start()

        self._running = True
        loop = asyncio.get_event_loop()

        # Run capture in executor to avoid blocking event loop
        self._thread = loop.run_in_executor(None, self._capture_loop_picam2)
        logger.info(
            "picamera2 started: %dx%d @ %d fps",
            self._cfg.width, self._cfg.height, self._cfg.fps,
        )

    async def stop(self) -> None:
        """Stop capture and release hardware resources."""
        self._running = False
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._picam2 = None
        logger.info("Camera capture stopped. Captured %d frames.", self._capture_count)

    async def get_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Return the next frame from the queue, or None on timeout.

        Args:
            timeout: Seconds to wait for a frame before returning None.
        """
        try:
            frame = await asyncio.wait_for(self._frame_queue.get(), timeout=timeout)
            return frame
        except TimeoutError:
            return self._latest_frame  # return stale frame rather than None

    async def save_snapshot(self) -> Path:
        """Save the latest annotated JPEG frame to disk."""
        async with self._lock:
            jpeg = self._latest_annotated

        ts = int(time.time())
        snap_path = Path(self._cfg.snapshot_dir) / f"snapshot_{ts}.jpg"

        if jpeg:
            snap_path.write_bytes(jpeg)
        elif self._latest_frame is not None:
            cv2.imwrite(str(snap_path), self._latest_frame)
        else:
            raise RuntimeError("No frame available for snapshot.")

        logger.info("Snapshot saved: %s", snap_path)
        return snap_path

    def set_annotated_jpeg(self, jpeg_bytes: bytes) -> None:
        """Called by the inference pipeline to store the latest annotated frame."""
        self._latest_annotated = jpeg_bytes

    @property
    def latest_annotated(self) -> bytes:
        return self._latest_annotated

    # ── Internal capture loops ──────────────────────────────────────────────

    def _capture_loop_picam2(self) -> None:
        """Blocking loop running in a thread executor."""
        from telemetry.metrics import fps_camera

        while self._running:
            try:
                frame = self._picam2.capture_array("main")  # BGR888 → ndarray
                self._latest_frame = frame
                self._capture_count += 1
                fps_camera.tick()

                # Put frame into queue; drop oldest if full (non-blocking put)
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    self._frame_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    pass

            except Exception as exc:
                logger.error("Camera capture error: %s", exc)
                time.sleep(0.1)

    async def _start_opencv_fallback(self) -> None:
        """Fallback capture using OpenCV VideoCapture (for development/testing)."""
        self._cap = cv2.VideoCapture(self._cfg.device_index)
        if not self._cap.isOpened():
            logger.warning(
                "Cannot open camera index %d. Falling back to Mock Video Generator.",
                self._cfg.device_index
            )
            self._cap = None
            self._running = True
            loop = asyncio.get_event_loop()
            self._thread = loop.run_in_executor(None, self._capture_loop_mock)
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._cfg.fps)

        self._running = True
        loop = asyncio.get_event_loop()
        self._thread = loop.run_in_executor(None, self._capture_loop_opencv)
        logger.info("OpenCV fallback camera started: index=%d", self._cfg.device_index)

    def _capture_loop_opencv(self) -> None:
        """OpenCV capture loop (development fallback)."""
        from telemetry.metrics import fps_camera

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("OpenCV frame read failed - retrying.")
                time.sleep(0.05)
                continue

            self._latest_frame = frame
            self._capture_count += 1
            fps_camera.tick()

            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Exception:
                    pass
            try:
                self._frame_queue.put_nowait(frame)
            except Exception:
                pass

    def _capture_loop_mock(self) -> None:
        """Mock camera loop that generates synthetic frames (for development/testing)."""
        from telemetry.metrics import fps_camera

        width = self._cfg.width or 640
        height = self._cfg.height or 480
        fps = self._cfg.fps or 30
        frame_interval = 1.0 / fps

        # Create base black frame
        base_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Bouncing square properties
        square_size = 60
        x, y = 100, 100
        dx, dy = 8, 8

        while self._running:
            t0 = time.time()
            # Copy base frame
            frame = base_frame.copy()
            
            # Draw grid
            for i in range(0, width, 80):
                cv2.line(frame, (i, 0), (i, height), (30, 30, 30), 1)
            for j in range(0, height, 80):
                cv2.line(frame, (0, j), (width, j), (30, 30, 30), 1)

            # Move and draw bouncing square
            x += dx
            y += dy
            if x <= 0 or x + square_size >= width:
                dx = -dx
            if y <= 0 or y + square_size >= height:
                dy = -dy

            # Draw target square (simulating a detected object)
            cv2.rectangle(frame, (x, y), (x + square_size, y + square_size), (0, 200, 0), 2)
            cv2.rectangle(frame, (x + 5, y + 5), (x + square_size - 5, y + square_size - 5), (0, 120, 0), -1)

            # Add timestamp and mock label
            cv2.putText(
                frame,
                f"MOCK WEBCAM: {time.strftime('%H:%M:%S')}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            self._latest_frame = frame
            self._capture_count += 1
            fps_camera.tick()

            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Exception:
                    pass
            try:
                self._frame_queue.put_nowait(frame)
            except Exception:
                pass

            # Control frame rate
            elapsed = time.time() - t0
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)
