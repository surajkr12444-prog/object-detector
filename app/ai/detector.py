from __future__ import annotations

import base64
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .exceptions import InvalidFrameError
from .model_manager import ModelManager
from .schemas import BoundingBox, Detection, DetectionResult

# Consistent color palette for 80 COCO classes
CLASS_COLORS = [
    (56, 189, 248),   # Cyan
    (52, 211, 153),   # Emerald
    (251, 191, 36),   # Amber
    (248, 113, 113),  # Red
    (192, 132, 252),  # Purple
    (244, 114, 182),  # Pink
    (129, 140, 248),  # Indigo
    (74, 222, 128),   # Lime
    (251, 146, 60),   # Orange
    (45, 212, 191),   # Teal
]


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode base64 string or data URL to OpenCV BGR image."""
    if not image_b64:
        raise InvalidFrameError("Base64 image string is empty.")

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise InvalidFrameError("Failed to decode image from buffer.")
        return frame
    except Exception as exc:
        raise InvalidFrameError(f"Invalid base64 image data: {exc}") from exc


def encode_image_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode OpenCV BGR image to base64 JPEG data URL."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def annotate_frame(
    frame: np.ndarray,
    detections: List[Detection],
    draw_labels: bool = True,
    draw_confidence: bool = True,
    draw_tracking_id: bool = True,
) -> np.ndarray:
    """Draw bounding boxes, labels, and tracking IDs on an image."""
    annotated = frame.copy()

    for det in detections:
        color = get_color_for_class(det.class_id)
        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        parts = [det.class_name]
        if draw_tracking_id and det.track_id is not None:
            parts.insert(0, f"#{det.track_id}")
        if draw_confidence:
            parts.append(f"{det.confidence * 100:.0f}%")
        label = " | ".join(parts)

        if draw_labels:
            (w, h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            top_y = max(y1, h + 8)
            # Label background banner
            cv2.rectangle(
                annotated,
                (x1, top_y - h - 6),
                (x1 + w + 8, top_y + 2),
                color,
                -1,
            )
            # Label text in dark contrast
            cv2.putText(
                annotated,
                label,
                (x1 + 4, top_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (15, 23, 42),
                1,
                cv2.LINE_AA,
            )

    return annotated


class YOLODetector:
    """High-level detector and tracker wrapper."""

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._manager = ModelManager.get_instance()

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        self._manager.ensure_loaded(model_name=self._model_name, device=self._device)

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.25,
        enable_tracking: bool = False,
        return_annotated: bool = False,
    ) -> DetectionResult:
        if frame is None:
            raise InvalidFrameError("Input frame is None.")

        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError("Input frame must be a NumPy array.")

        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise InvalidFrameError(
                "Input frame must have shape (H, W, 3) or (H, W, 4)."
            )

        if not (0.0 <= confidence_threshold <= 1.0):
            raise InvalidFrameError("confidence_threshold must be between 0.0 and 1.0")

        model = self._manager.ensure_loaded(
            model_name=self._model_name,
            device=self._device,
        )

        start_time = time.perf_counter()

        # Run tracking or inference
        if enable_tracking and hasattr(model, "track"):
            results = model.track(
                source=frame,
                conf=confidence_threshold,
                device=self._device,
                verbose=False,
                persist=True,
            )
        else:
            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                device=self._device,
                verbose=False,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Detection] = []
        class_counts: Dict[str, int] = {}

        if results:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {}) or {}

            if boxes is not None and len(boxes) > 0:
                cls_ids = boxes.cls.tolist()
                confs = boxes.conf.tolist()
                xyxy = boxes.xyxy.tolist()

                # Tracking IDs (if available)
                track_ids = [None] * len(cls_ids)
                if hasattr(boxes, "id") and boxes.id is not None:
                    try:
                        track_ids = boxes.id.int().tolist()
                    except Exception:
                        pass

                for class_id, conf, coords, t_id in zip(cls_ids, confs, xyxy, track_ids):
                    x1, y1, x2, y2 = coords
                    class_idx = int(class_id)
                    class_name = names.get(class_idx, f"class_{class_idx}")

                    detections.append(
                        Detection(
                            class_id=class_idx,
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=BoundingBox(
                                x1=float(x1),
                                y1=float(y1),
                                x2=float(x2),
                                y2=float(y2),
                            ),
                            track_id=int(t_id) if t_id is not None else None,
                        )
                    )

        counts = Counter(d.class_name for d in detections)
        class_counts = dict(counts)

        annotated_b64: Optional[str] = None
        if return_annotated:
            annotated_frame = annotate_frame(frame, detections)
            annotated_b64 = encode_image_base64(annotated_frame)

        h, w = frame.shape[:2]
        return DetectionResult(
            width=int(w),
            height=int(h),
            confidence_threshold=confidence_threshold,
            model_name=self._model_name,
            inference_time_ms=round(elapsed_ms, 3),
            detections=detections,
            class_counts=class_counts,
            annotated_image_base64=annotated_b64,
        )
