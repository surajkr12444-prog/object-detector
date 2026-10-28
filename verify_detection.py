from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.ai.detector import YOLODetector
from app.ai.exceptions import DetectionEngineError


def annotate_image(image, detections):
    for det in detections:
        x1 = int(det.bbox.x1)
        y1 = int(det.bbox.y1)
        x2 = int(det.bbox.x2)
        y2 = int(det.bbox.y2)
        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone YOLO detection verifier")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or local .pt path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0-1.0)")
    parser.add_argument(
        "--output",
        default="data/temp/annotated_result.jpg",
        help="Output path for annotated image",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError("Unable to read image file. Ensure it is a valid image format.")

    detector = YOLODetector(model_name=args.model, device="cpu")

    try:
        result = detector.detect(frame, confidence_threshold=args.conf)
    except DetectionEngineError as exc:
        print(f"Detection error: {exc}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = annotate_image(frame.copy(), result.detections)
    cv2.imwrite(str(output_path), annotated)

    summary = {
        "model": result.model_name,
        "inference_time_ms": result.inference_time_ms,
        "detections_count": len(result.detections),
        "output_image": str(output_path),
        "detections": [d.model_dump() for d in result.detections],
    }

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
