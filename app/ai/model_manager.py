from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from .exceptions import DependencyUnavailableError, ModelLoadError


class ModelManager:
    """Singleton-style manager to avoid unnecessary YOLO model reloads."""

    _instance: Optional["ModelManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._model_name = ""
        self._device = "cpu"
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self, model_name: str = "yolov8n.pt", device: str = "cpu"):
        """
        Load a model once and reuse it.

        If the same model/device is already loaded, no-op.
        """
        with self._model_lock:
            if (
                self._model is not None
                and self._model_name == model_name
                and self._device == device
            ):
                return self._model

            try:
                import ultralytics.utils.downloads as ud
                _orig_attempt = ud.attempt_download_asset

                def _safe_attempt(file, *args, **kwargs):
                    p = Path(str(file).strip())
                    if p.exists():
                        return str(p)
                    return _orig_attempt(file, *args, **kwargs)

                ud.attempt_download_asset = _safe_attempt
                from ultralytics import YOLO  # local import for clearer dependency errors
            except Exception as exc:  # pragma: no cover - environment-specific
                raise DependencyUnavailableError(
                    "Ultralytics YOLO is unavailable. Install backend dependencies first."
                ) from exc

            try:
                # Allow either a known model alias or a local file path.
                model_ref = model_name
                # Check candidate paths for local weights
                candidates = [
                    Path(model_name),
                    Path("models") / model_name,
                    Path(__file__).resolve().parents[2] / "models" / model_name,
                    Path(__file__).resolve().parents[3] / "models" / model_name,
                ]
                for cand in candidates:
                    if cand.is_file():
                        model_ref = str(cand)
                        break

                model = YOLO(model_ref)

                # A tiny warm-up pass can fail fast on bad/corrupt weights.
                # Using an empty call is enough to initialize internals.
                _ = model.names
            except Exception as exc:
                message = str(exc).lower()
                if "download" in message or "network" in message or "url" in message:
                    raise ModelLoadError(
                        "Failed to download YOLO weights. Check internet access or place model file locally in models/."
                    ) from exc
                raise ModelLoadError(f"Failed to load YOLO model '{model_name}': {exc}") from exc

            self._model = model
            self._model_name = model_name
            self._device = device
            return self._model

    def get_model(self):
        if self._model is None:
            raise ModelLoadError("Model is not loaded. Call ensure_loaded() first.")
        return self._model
