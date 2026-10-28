from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..ai.detector import YOLODetector, annotate_frame, decode_base64_image, encode_image_base64
from ..ai.model_manager import ModelManager
from ..ai.schemas import DetectionResult, FrameDetectRequest, SnapshotCreateRequest
from ..db import DetectionLog, SnapshotRecord, SNAPSHOTS_DIR, UPLOADS_DIR, get_db

router = APIRouter()

# Global default detector instance
detector = YOLODetector(model_name="yolov8n.pt", device="cpu")


@router.get("/")
def api_root() -> Dict[str, Any]:
    return {
        "name": "VisionX AI Detection & Tracking API",
        "version": "0.1.0",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "detect_frame": "POST /api/detect/frame",
            "detect_upload": "POST /api/detect/upload",
            "snapshots": "GET, POST /api/history/snapshots",
            "analytics": "GET /api/analytics/stats",
            "models": "GET /api/models",
        },
    }


@router.get("/health")
def health_check() -> Dict[str, Any]:
    manager = ModelManager.get_instance()
    return {
        "status": "ok",
        "service": "visionx-backend",
        "version": "0.1.0",
        "model_loaded": manager.is_loaded(),
        "active_model": manager.model_name or "yolov8n.pt",
    }


@router.post("/api/detect/frame", response_model=DetectionResult)
def detect_frame(
    request: FrameDetectRequest,
    db: Session = Depends(get_db),
) -> DetectionResult:
    """Analyze a single live webcam frame for real-time detection & tracking."""
    try:
        frame = decode_base64_image(request.image_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {exc}",
        )

    result = detector.detect(
        frame=frame,
        confidence_threshold=request.confidence_threshold,
        enable_tracking=request.enable_tracking,
        return_annotated=request.return_annotated,
    )

    # Optional fast log of detections for analytics (sampled to avoid DB flood)
    if result.detections:
        try:
            for det in result.detections:
                log_entry = DetectionLog(
                    model_name=result.model_name,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    track_id=det.track_id,
                    inference_time_ms=result.inference_time_ms,
                )
                db.add(log_entry)
            db.commit()
        except Exception:
            db.rollback()

    return result


@router.post("/api/detect/upload", response_model=DetectionResult)
async def detect_upload(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.25,
    enable_tracking: bool = False,
    db: Session = Depends(get_db),
) -> DetectionResult:
    """Process an uploaded image file and return detections + annotated image."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded image. Please ensure valid JPG or PNG format.",
        )

    result = detector.detect(
        frame=frame,
        confidence_threshold=confidence_threshold,
        enable_tracking=enable_tracking,
        return_annotated=True,
    )

    # Log detections to database
    try:
        for det in result.detections:
            db.add(
                DetectionLog(
                    model_name=result.model_name,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    track_id=det.track_id,
                    inference_time_ms=result.inference_time_ms,
                )
            )
        db.commit()
    except Exception:
        db.rollback()

    return result


@router.post("/api/history/snapshots")
def create_snapshot(
    payload: SnapshotCreateRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Save a snapshot frame and its detection metadata."""
    try:
        frame = decode_base64_image(payload.image_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {exc}",
        )

    filename = f"snap_{uuid.uuid4().hex[:12]}_{int(datetime.utcnow().timestamp())}.jpg"
    file_path = SNAPSHOTS_DIR / filename

    # Annotate if detections provided
    if payload.detections:
        annotated = annotate_frame(frame, payload.detections)
        cv2.imwrite(str(file_path), annotated)
    else:
        cv2.imwrite(str(file_path), frame)

    detections_list = [d.model_dump() for d in payload.detections]
    class_counts = {}
    for d in payload.detections:
        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

    record = SnapshotRecord(
        image_filename=filename,
        source_type=payload.source_type,
        model_name=payload.model_name,
        confidence_threshold=payload.confidence_threshold,
        detections_count=len(payload.detections),
        inference_time_ms=payload.inference_time_ms,
        detections_json=json.dumps(detections_list),
        class_counts_json=json.dumps(class_counts),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "image_filename": record.image_filename,
        "created_at": record.created_at.isoformat(),
        "detections_count": record.detections_count,
        "status": "saved",
    }


@router.get("/api/history/snapshots")
def list_snapshots(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    records = (
        db.query(SnapshotRecord)
        .order_by(SnapshotRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    results = []
    for r in records:
        try:
            dets = json.loads(r.detections_json or "[]")
        except Exception:
            dets = []
        try:
            counts = json.loads(r.class_counts_json or "{}")
        except Exception:
            counts = {}

        results.append(
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "image_filename": r.image_filename,
                "image_url": f"/api/history/snapshots/{r.id}/image",
                "source_type": r.source_type,
                "model_name": r.model_name,
                "confidence_threshold": r.confidence_threshold,
                "detections_count": r.detections_count,
                "inference_time_ms": r.inference_time_ms,
                "detections": dets,
                "class_counts": counts,
            }
        )
    return results


@router.get("/api/history/snapshots/{snapshot_id}/image")
def get_snapshot_image(
    snapshot_id: int,
    db: Session = Depends(get_db),
):
    record = db.query(SnapshotRecord).filter(SnapshotRecord.id == snapshot_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Snapshot record not found")

    file_path = SNAPSHOTS_DIR / record.image_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Snapshot image file not found on disk")

    return FileResponse(str(file_path), media_type="image/jpeg")


@router.delete("/api/history/snapshots/{snapshot_id}")
def delete_snapshot(
    snapshot_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    record = db.query(SnapshotRecord).filter(SnapshotRecord.id == snapshot_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    file_path = SNAPSHOTS_DIR / record.image_filename
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    db.delete(record)
    db.commit()
    return {"status": "deleted", "id": str(snapshot_id)}


@router.post("/api/history/clear")
def clear_history(
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    # Clear snapshot images
    for file in SNAPSHOTS_DIR.glob("*.jpg"):
        try:
            file.unlink()
        except Exception:
            pass

    db.query(SnapshotRecord).delete()
    db.query(DetectionLog).delete()
    db.commit()
    return {"status": "cleared"}


@router.get("/api/analytics/stats")
def get_analytics_stats(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    from sqlalchemy import func

    total_snapshots = db.query(func.count(SnapshotRecord.id)).scalar() or 0
    total_detections = db.query(func.count(DetectionLog.id)).scalar() or 0

    avg_inference = (
        db.query(func.avg(DetectionLog.inference_time_ms)).scalar() or 0.0
    )

    # Class distribution breakdown
    class_group = (
        db.query(DetectionLog.class_name, func.count(DetectionLog.id))
        .group_by(DetectionLog.class_name)
        .order_by(func.count(DetectionLog.id).desc())
        .limit(10)
        .all()
    )

    class_distribution = [
        {"name": name, "count": count} for name, count in class_group
    ]

    # Recent snapshots preview
    recent_snapshots = (
        db.query(SnapshotRecord)
        .order_by(SnapshotRecord.created_at.desc())
        .limit(6)
        .all()
    )
    snapshots_preview = [
        {
            "id": s.id,
            "created_at": s.created_at.isoformat(),
            "image_url": f"/api/history/snapshots/{s.id}/image",
            "detections_count": s.detections_count,
            "inference_time_ms": s.inference_time_ms,
        }
        for s in recent_snapshots
    ]

    return {
        "total_snapshots": total_snapshots,
        "total_detections": total_detections,
        "average_inference_ms": round(float(avg_inference), 1),
        "class_distribution": class_distribution,
        "recent_snapshots": snapshots_preview,
    }


@router.get("/api/models")
def get_models() -> Dict[str, Any]:
    manager = ModelManager.get_instance()
    available = [
        {"id": "yolov8n.pt", "name": "YOLOv8 Nano (Fastest, ~6MB)", "ready": True},
        {"id": "yolov8s.pt", "name": "YOLOv8 Small (Balanced, ~22MB)", "ready": False},
        {"id": "yolov8m.pt", "name": "YOLOv8 Medium (Accurate, ~50MB)", "ready": False},
    ]
    return {
        "active_model": manager.model_name or "yolov8n.pt",
        "device": manager.device,
        "is_loaded": manager.is_loaded(),
        "available_models": available,
    }

