"""
inference/yolo_pipeline.py
YOLO inference pipeline.

Flow:
  1. Pull BGR frames from camera.get_frame()
  2. Run HailoInferenceEngine.infer()
  3. Draw bounding box overlays onto frame
  4. JPEG-encode annotated frame
  5. Broadcast JPEG bytes on "camera" WebSocket channel
  6. Store detections on camera object for REST API

Optimisations:
  - asyncio pipeline - no blocking calls on event loop
  - JPEG encoding in executor thread
  - Per-class colour palette (consistent across frames)
  - FPS counter via telemetry.metrics.fps_inference
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from inference.hailo_engine import COCO_CLASSES, Detection

if TYPE_CHECKING:
    from app.websocket.stream import WebSocketManager
    from camera.capture import CameraCapture
    from config.config_loader import InferenceConfig
    from inference.hailo_engine import HailoInferenceEngine

logger = logging.getLogger(__name__)

# Pre-computed BGR colour palette (one colour per COCO class)
_PALETTE: list[tuple[int, int, int]] = []
for _i in range(len(COCO_CLASSES)):
    _h = int(_i * 180 / len(COCO_CLASSES))
    _hsv = np.array([[[_h, 220, 220]]], dtype=np.uint8)
    _bgr = cv2.cvtColor(_hsv, cv2.COLOR_HSV2BGR)[0][0]
    _PALETTE.append((int(_bgr[0]), int(_bgr[1]), int(_bgr[2])))



class YOLOPipeline:
    """
    End-to-end asynchronous YOLO inference pipeline.

    Args:
        hailo_engine: HailoInferenceEngine (may be None → camera-only mode).
        camera:       CameraCapture instance.
        ws_manager:   WebSocketManager for broadcasting annotated frames.
        config:       InferenceConfig.
    """

    def __init__(
        self,
        hailo_engine: HailoInferenceEngine | None,
        camera: CameraCapture,
        ws_manager: WebSocketManager,
        config: InferenceConfig,
    ) -> None:
        self._hailo      = hailo_engine
        self._camera     = camera
        self._ws_manager = ws_manager
        self._cfg        = config
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self, shutdown_event: asyncio.Event) -> None:
        """
        Main pipeline loop.  Runs until shutdown_event is set.
        """
        from telemetry.metrics import fps_inference

        self._loop = asyncio.get_event_loop()
        logger.info("YOLO inference pipeline started.")

        while not shutdown_event.is_set():
            try:
                # ── 1. Acquire frame ─────────────────────────────────────
                frame = await self._camera.get_frame(timeout=0.5)
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # ── 2. Run inference (non-blocking via executor) ─────────
                detections: list[Detection] = []
                if self._hailo is not None:
                    detections = await self._hailo.infer(frame)
                fps_inference.tick()

                # ── 3. Update camera's detection list (for REST API) ─────
                self._camera.latest_detections = [d.to_dict() for d in detections]

                # ── 4. Annotate frame ────────────────────────────────────
                if self._cfg.draw_overlays:
                    assert self._loop is not None
                    annotated = await self._loop.run_in_executor(
                        None, self._draw_overlays, frame.copy(), detections
                    )
                else:
                    annotated = frame

                # ── 5. JPEG encode ───────────────────────────────────────
                assert self._loop is not None
                jpeg = await self._loop.run_in_executor(
                    None, self._encode_jpeg, annotated
                )
                self._camera.set_annotated_jpeg(jpeg)

                # ── 6. Broadcast to WebSocket clients ────────────────────
                if self._ws_manager.connection_count("camera") > 0:
                    await self._ws_manager.broadcast_bytes(jpeg, channel="camera")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Pipeline error: %s", exc, exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("YOLO inference pipeline stopped.")

    # ── Private helpers ─────────────────────────────────────────────────────

    def _draw_overlays(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Draw YOLO bounding boxes and labels onto the frame.

        Args:
            frame:      BGR frame (mutated in-place).
            detections: List of Detection objects with normalised coords.

        Returns:
            Annotated BGR frame.
        """
        h, w = frame.shape[:2]
        thickness = self._cfg.overlay_thickness
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        for det in detections:
            x1n, y1n, x2n, y2n = det.bbox_xyxy
            x1 = int(x1n * w)
            y1 = int(y1n * h)
            x2 = int(x2n * w)
            y2 = int(y2n * h)

            colour = _PALETTE[det.class_id % len(_PALETTE)]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

            # Label background
            label = f"{det.class_name} {det.confidence:.0%}"
            (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(
                frame,
                (x1, y1 - lh - baseline - 4),
                (x1 + lw, y1),
                colour,
                -1,
            )
            # Label text
            cv2.putText(
                frame, label,
                (x1, y1 - baseline - 2),
                font, font_scale, (255, 255, 255), 1,
                lineType=cv2.LINE_AA,
            )

        # FPS overlay (top-left corner)
        from telemetry.metrics import fps_inference
        fps_text = f"INF {fps_inference.fps:.1f} fps"
        cv2.putText(
            frame, fps_text,
            (10, 24), font, 0.6, (0, 255, 0), 2,
            lineType=cv2.LINE_AA,
        )

        # Timestamp overlay (bottom-right)
        ts_text = time.strftime("%H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts_text, font, 0.45, 1)
        cv2.putText(
            frame, ts_text,
            (w - tw - 8, h - 8), font, 0.45, (200, 200, 200), 1,
            lineType=cv2.LINE_AA,
        )

        return frame

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        """Encode a BGR frame to JPEG bytes."""
        params = [cv2.IMWRITE_JPEG_QUALITY, self._cfg_jpeg_quality]
        success, buf = cv2.imencode(".jpg", frame, params)
        if not success:
            raise RuntimeError("JPEG encoding failed.")
        return buf.tobytes()

    @property
    def _cfg_jpeg_quality(self) -> int:
        # Access via parent config (camera section) if available
        try:
            from config.config_loader import load_config
            return load_config().camera.jpeg_quality
        except Exception:
            return 80
