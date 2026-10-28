class DetectionEngineError(Exception):
    """Base exception for the detection engine."""


class DependencyUnavailableError(DetectionEngineError):
    """Raised when required AI dependencies are not available."""


class ModelLoadError(DetectionEngineError):
    """Raised when YOLO model loading fails."""


class InvalidFrameError(DetectionEngineError):
    """Raised when an invalid frame is passed for inference."""
