import numpy as np
import pytest

from app.ai.detector import YOLODetector
from app.ai.exceptions import InvalidFrameError


class _FakeBoxes:
    def __init__(self):
        self.cls = np.array([0])
        self.conf = np.array([0.91])
        self.xyxy = np.array([[10.0, 20.0, 110.0, 220.0]])

    def __len__(self):
        return len(self.cls)


class _FakeResult:
    def __init__(self):
        self.boxes = _FakeBoxes()
        self.names = {0: "person"}


class _FakeModel:
    def predict(self, source, conf, device, verbose):
        return [_FakeResult()]


def test_detect_invalid_frame_none():
    detector = YOLODetector()
    with pytest.raises(InvalidFrameError):
        detector.detect(None)


def test_detect_invalid_frame_shape():
    detector = YOLODetector()
    with pytest.raises(InvalidFrameError):
        detector.detect(np.zeros((100, 100), dtype=np.uint8))


def test_detect_parses_output(monkeypatch):
    detector = YOLODetector()

    def _fake_ensure_loaded(model_name, device):
        return _FakeModel()

    monkeypatch.setattr(detector._manager, "ensure_loaded", _fake_ensure_loaded)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame, confidence_threshold=0.25)

    assert result.width == 640
    assert result.height == 480
    assert len(result.detections) == 1
    assert result.detections[0].class_name == "person"
    assert result.detections[0].confidence == pytest.approx(0.91, abs=1e-5)
