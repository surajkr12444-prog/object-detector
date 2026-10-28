from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from .database import Base


class SnapshotRecord(Base):
    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    image_filename = Column(String(255), nullable=False)
    source_type = Column(String(50), default="webcam")  # webcam, upload
    model_name = Column(String(100), default="yolov8n.pt")
    confidence_threshold = Column(Float, default=0.25)
    detections_count = Column(Integer, default=0)
    inference_time_ms = Column(Float, default=0.0)
    detections_json = Column(Text, default="[]")  # JSON-encoded detection objects
    class_counts_json = Column(Text, default="{}")  # JSON-encoded class counts map


class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_name = Column(String(100), default="yolov8n.pt")
    class_name = Column(String(100), index=True)
    confidence = Column(Float)
    track_id = Column(Integer, nullable=True)
    inference_time_ms = Column(Float)

