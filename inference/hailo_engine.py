"""
inference/hailo_engine.py
HailoRT inference engine wrapper for the Hailo-8L accelerator.

Features:
  - Loads compiled .hef model onto Hailo-8L via PCIe
  - Manages input/output virtual streams (VSTREAM)
  - Async inference via asyncio executor (non-blocking event loop)
  - Transparent CPU fallback (ultralytics) when Hailo is unavailable
  - Exposes utilisation_stats() for telemetry
  - Thread-safe with a per-engine asyncio.Lock

Model format:
  - Hailo Executable Format (.hef) compiled with Hailo Model Zoo
  - Input:  RGB uint8  [1, 640, 640, 3]
  - Output: YOLO detection tensors (post-processed by post-processor layer)

HailoRT API version: ≥ 4.17
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from config.config_loader import InferenceConfig

logger = logging.getLogger(__name__)


class Detection:
    """Single detection result from the inference engine."""

    __slots__ = ("class_id", "class_name", "confidence", "bbox_xyxy", "timestamp")

    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> None:
        self.class_id   = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox_xyxy  = bbox_xyxy      # (x1, y1, x2, y2) normalised [0–1]
        self.timestamp  = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox":       [round(v, 4) for v in self.bbox_xyxy],
            "timestamp":  self.timestamp,
        }


# COCO-80 class names (index-matched)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]


class HailoInferenceEngine:
    """
    Wraps HailoRT for async, hardware-accelerated YOLOv8 inference.

    Falls back to CPU (ultralytics) if HailoRT is unavailable or the
    .hef model cannot be loaded.
    """

    def __init__(self, cfg: InferenceConfig) -> None:
        self._cfg      = cfg
        self._lock     = asyncio.Lock()
        self._hailo    = None    # HailoRT Network Group
        self._infer_fn: Callable[[np.ndarray], list[Detection]] | None = None
        self._use_hailo = False

        # Runtime stats
        self._inference_count = 0
        self._total_latency_ms = 0.0
        self._last_latency_ms  = 0.0

    async def initialise(self) -> None:
        """
        Initialise the inference backend.

        Priority:
          1. Hailo-8L (config.device == "hailo" or "auto")
          2. CPU ultralytics (config.device == "cpu" or Hailo fails)
        """
        device = self._cfg.device

        if device in ("hailo", "auto"):
            try:
                await self._init_hailo()
                return
            except Exception as exc:
                if device == "hailo":
                    raise RuntimeError(f"Hailo initialisation required but failed: {exc}") from exc
                logger.warning("Hailo unavailable (%s) — falling back to CPU.", exc)

        await self._init_cpu()

    async def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Run inference on a BGR frame.

        Args:
            frame_bgr: OpenCV-style uint8 BGR image.

        Returns:
            List of Detection objects (confidence-filtered).
        """
        if self._infer_fn is None:
            raise RuntimeError("Inference engine is not initialised.")

        async with self._lock:
            loop = asyncio.get_event_loop()
            t0 = time.monotonic()
            detections = await loop.run_in_executor(
                None, self._infer_fn, frame_bgr
            )
            latency = (time.monotonic() - t0) * 1000
            self._last_latency_ms   = latency
            self._total_latency_ms += latency
            self._inference_count  += 1
            return cast(list[Detection], detections)

    async def utilisation_stats(self) -> dict[str, Any]:
        """Return Hailo-8L utilisation metrics for telemetry."""
        stats: dict[str, Any] = {
            "available":        self._use_hailo,
            "inference_count":  self._inference_count,
            "last_latency_ms":  round(self._last_latency_ms, 2),
        }
        if self._inference_count > 0:
            stats["avg_latency_ms"] = round(
                self._total_latency_ms / self._inference_count, 2
            )

        if self._use_hailo and self._hailo is not None:
            try:
                # HailoRT power/utilisation (available in some SDK versions)
                power = self._hailo.get_power_measurement()
                stats["power_mw"] = round(power, 1) if power else None
            except Exception:
                stats["power_mw"] = None

        return stats

    async def shutdown(self) -> None:
        """Release Hailo hardware resources."""
        if self._hailo is not None:
            try:
                self._hailo.release()
            except Exception as exc:
                logger.debug("Hailo release error: %s", exc)
        logger.info(
            "Hailo engine shut down. Total inferences: %d", self._inference_count
        )

    # ── Private initialisers ────────────────────────────────────────────────

    async def _init_hailo(self) -> None:
        """Load .hef model onto Hailo-8L via HailoRT."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_hailo_sync)
        logger.info("Hailo-8L initialised: model=%s", self._cfg.model_path)

    def _load_hailo_sync(self) -> None:
        """Synchronous Hailo initialisation (runs in executor thread)."""
        # Import HailoRT Python bindings
        try:
            from hailo_platform import (  # type: ignore
                HEF,
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as exc:
            raise ImportError("HailoRT Python SDK not installed.") from exc

        hef_path = self._cfg.model_path
        hef = HEF(hef_path)

        # Open virtual device (PCIe)
        self._vdevice = VDevice()

        # Configure network group
        configure_params = ConfigureParams.create_from_hef(
            hef=hef,
            interface=HailoStreamInterface.PCIe,
        )
        network_groups = self._vdevice.configure(hef, configure_params)
        self._network_group = network_groups[0]
        self._network_group_params = self._network_group.create_params()

        # Input/output stream params
        self._input_vstream_params = InputVStreamParams.make(
            self._network_group,
            format_type=FormatType.UINT8,
        )
        self._output_vstream_params = OutputVStreamParams.make(
            self._network_group,
            format_type=FormatType.FLOAT32,
        )

        # Store reference for utilisation queries
        self._hailo = self._network_group
        self._use_hailo = True

        # Bind the inference function
        self._infer_fn = self._hailo_infer_sync

    def _hailo_infer_sync(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Run one forward pass on Hailo-8L.

        Pre-processing:  BGR → RGB, resize to 640×640, uint8
        Post-processing: decode YOLO output tensors, filter by confidence
        """
        try:
            from hailo_platform import InferVStreams  # type: ignore
        except ImportError:
            return []

        # ── Pre-process ─────────────────────────────────────────────
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._cfg.input_width, self._cfg.input_height))
        batch = np.expand_dims(resized, axis=0)           # [1, 640, 640, 3]

        # ── Inference ────────────────────────────────────────────────
        with InferVStreams(
            self._network_group,
            self._input_vstream_params,
            self._output_vstream_params,
        ) as infer_pipeline:
            with self._network_group.activate(self._network_group_params):
                input_name = next(iter(self._input_vstream_params.keys()))
                infer_data = {input_name: batch.astype(np.uint8)}
                raw_detections = infer_pipeline.infer(infer_data)

        # ── Post-process ─────────────────────────────────────────────
        return self._decode_hailo_output(raw_detections, frame_bgr.shape)

    def _decode_hailo_output(
        self,
        raw: dict[str, np.ndarray],
        orig_shape: tuple,
    ) -> list[Detection]:
        """
        Decode Hailo YOLOv8 output tensors into Detection objects.

        YOLOv8n HEF with the built-in NMS post-processor outputs:
            boxes:     [N, 4]  (x1, y1, x2, y2) in pixel coords on 640×640
            scores:    [N]
            classes:   [N]  (int)

        We normalise bbox coords to [0,1] for resolution-independence.
        """
        detections: list[Detection] = []

        # The HEF compiled with hailo_model_zoo typically has output key
        # matching the layer name — iterate to find detection outputs.
        for key, tensor in raw.items():
            if tensor.ndim < 2:
                continue

            output = tensor.squeeze(0)   # Remove batch dim → [N, ≥6]

            # Expected format per detection: [x1, y1, x2, y2, confidence, class_id]
            if output.shape[-1] < 6:
                continue

            for row in output:
                x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                confidence      = float(row[4])
                class_id        = int(row[5])

                if confidence < self._cfg.confidence_threshold:
                    continue
                if class_id >= len(COCO_CLASSES):
                    continue

                class_name = COCO_CLASSES[class_id]

                # Filter by target classes if specified
                if self._cfg.target_classes and class_name not in self._cfg.target_classes:
                    continue

                # Normalise to [0, 1] relative to 640×640 input
                x1n = np.clip(x1 / self._cfg.input_width,  0.0, 1.0)
                y1n = np.clip(y1 / self._cfg.input_height, 0.0, 1.0)
                x2n = np.clip(x2 / self._cfg.input_width,  0.0, 1.0)
                y2n = np.clip(y2 / self._cfg.input_height, 0.0, 1.0)

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(float(x1n), float(y1n), float(x2n), float(y2n)),
                ))

        return detections

    # ── CPU fallback ────────────────────────────────────────────────────────

    async def _init_cpu(self) -> None:
        """Load YOLOv8 model via ultralytics for CPU inference (dev/fallback)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_cpu_sync)
        logger.info("CPU inference engine initialised: %s", self._cfg.cpu_model_path)

    def _load_cpu_sync(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError("ultralytics not installed for CPU fallback.") from exc

        import cv2 as _cv2  # Ensure cv2 available in this thread
        globals()["cv2"] = _cv2

        model_path = self._cfg.cpu_model_path
        self._cpu_model = YOLO(model_path)
        self._use_hailo = False
        self._infer_fn  = self._cpu_infer_sync
        logger.warning(
            "Using CPU inference — performance will be significantly lower than Hailo-8L."
        )

    def _cpu_infer_sync(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run YOLOv8 inference on CPU via ultralytics."""

        results = self._cpu_model.predict(
            frame_bgr,
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.nms_threshold,
            imgsz=(self._cfg.input_height, self._cfg.input_width),
            verbose=False,
        )

        detections: list[Detection] = []
        h, w = frame_bgr.shape[:2]

        for result in results:
            boxes: Any = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf    = float(box.conf[0])
                cls_id  = int(box.cls[0])
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)

                if self._cfg.target_classes and cls_name not in self._cfg.target_classes:
                    continue

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=(x1 / w, y1 / h, x2 / w, y2 / h),
                ))

        return detections
