from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    track_id: Optional[int] = Field(None, description="Tracking ID if tracking is enabled")


class DetectionResult(BaseModel):
    width: int
    height: int
    confidence_threshold: float
    model_name: str
    inference_time_ms: float
    detections: List[Detection] = Field(default_factory=list)
    class_counts: Dict[str, int] = Field(default_factory=dict)
    annotated_image_base64: Optional[str] = None


class FrameDetectRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG or PNG image or data URI")
    confidence_threshold: float = Field(0.25, ge=0.01, le=1.0)
    enable_tracking: bool = Field(False, description="Use ByteTrack object tracker")
    return_annotated: bool = Field(False, description="Include base64 annotated frame in response")


class SnapshotCreateRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded snapshot frame")
    source_type: str = Field("webcam", description="Origin: webcam, upload, stream")
    confidence_threshold: float = Field(0.25)
    model_name: str = Field("yolov8n.pt")
    detections: List[Detection] = Field(default_factory=list)
    inference_time_ms: float = Field(0.0)
