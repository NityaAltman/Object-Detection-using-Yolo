"""Model loading, caching and inference for the detection API.

Two families of weights are served side by side:

* the pretrained COCO models, which Ultralytics downloads on first use and
  which work on ordinary photographs out of the box, and
* a satellite model trained on this repo's own two-class dataset
  (``car`` / ``swimming pool``), which detects overhead imagery that the COCO
  models cannot -- they have no swimming-pool class and were trained on
  ground-level photographs.

Those two come in incompatible checkpoint formats. The satellite weights were
produced by the original YOLOv5 repository, which the modern ``ultralytics``
package explicitly refuses to load, so each format gets its own backend behind
one interface. The format is sniffed from the file rather than configured, so
replacing the satellite checkpoint with a freshly trained YOLOv8 one needs no
code change.

A missing satellite checkpoint is an expected state, not a crash: the API
reports it as unavailable so the UI can say so plainly.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = REPO_ROOT / "weights"
VENDORED_YOLOV5 = REPO_ROOT / "yolov5"

COCO_DOMAIN = "coco"
SATELLITE_DOMAIN = "satellite"

ULTRALYTICS_FORMAT = "ultralytics"
YOLOV5_FORMAT = "yolov5"


class WeightsUnavailable(RuntimeError):
    """A model was requested whose weights are not on disk and cannot be fetched."""


def checkpoint_format(weights: str) -> str:
    """Tell a YOLOv5-repo checkpoint from an Ultralytics one.

    Both are pickles, and the class paths they reference give the origin away:
    the YOLOv5 repository pickles its own top-level ``models`` package, while
    Ultralytics references ``ultralytics.nn``.
    """
    try:
        with open(weights, "rb") as handle:
            head = handle.read(16384)
    except OSError:
        return ULTRALYTICS_FORMAT

    if b"models.yolo" in head and b"ultralytics.nn" not in head:
        return YOLOV5_FORMAT
    return ULTRALYTICS_FORMAT


class _UltralyticsBackend:
    """Wraps a modern Ultralytics model (YOLOv8 and newer)."""

    def __init__(self, weights: str) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)

    def infer(self, image: Image.Image, min_confidence: float) -> list[dict]:
        prediction = self._model.predict(image, conf=min_confidence, verbose=False)[0]

        detections = []
        for box in prediction.boxes:
            x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
            detections.append(
                {
                    "class": prediction.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "box": [x1, y1, x2, y2],
                }
            )
        return detections

    def warm_up(self) -> None:
        self._model.predict(Image.new("RGB", (32, 32)), conf=0.99, verbose=False)


class _YoloV5Backend:
    """Wraps a checkpoint produced by the original YOLOv5 repository.

    The `ultralytics` package rejects these outright, so inference runs through
    the copy of YOLOv5 vendored in this repo: letterbox the image, run the
    network, then NMS and rescale the boxes back to the original frame.
    """

    # The satellite weights were trained with `--img 416`; inferring at the
    # training resolution is what the model is calibrated for.
    IMAGE_SIZE = 416
    IOU_THRESHOLD = 0.45

    def __init__(self, weights: str) -> None:
        import numpy as np
        import torch

        self._np = np
        self._torch = torch

        if str(VENDORED_YOLOV5) not in sys.path:
            sys.path.insert(0, str(VENDORED_YOLOV5))

        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes
        from utils.torch_utils import select_device

        self._letterbox = letterbox
        self._nms = non_max_suppression
        self._scale_boxes = scale_boxes

        self._model = DetectMultiBackend(weights, device=select_device("cpu"))
        self._names = self._model.names
        self._stride = int(self._model.stride)

    def infer(self, image: Image.Image, min_confidence: float) -> list[dict]:
        np, torch = self._np, self._torch

        # The vendored code is written against OpenCV's BGR channel order.
        original = np.array(image)[:, :, ::-1]

        padded = self._letterbox(
            original, self.IMAGE_SIZE, stride=self._stride, auto=True
        )[0]
        tensor = padded.transpose((2, 0, 1))[::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(tensor)).float()[None] / 255

        with torch.no_grad():
            prediction = self._nms(
                self._model(tensor), min_confidence, self.IOU_THRESHOLD
            )[0]

        if not len(prediction):
            return []

        prediction[:, :4] = self._scale_boxes(
            tensor.shape[2:], prediction[:, :4], original.shape
        ).round()

        detections = []
        for *xyxy, confidence, class_id in prediction.tolist():
            detections.append(
                {
                    "class": self._names[int(class_id)],
                    "confidence": round(float(confidence), 4),
                    "box": [round(v, 1) for v in xyxy],
                }
            )
        return detections

    def warm_up(self) -> None:
        self.infer(Image.new("RGB", (self.IMAGE_SIZE, self.IMAGE_SIZE)), 0.99)


BACKENDS = {
    ULTRALYTICS_FORMAT: _UltralyticsBackend,
    YOLOV5_FORMAT: _YoloV5Backend,
}


class Detector:
    """Lazily loads models and keeps them warm in memory."""

    MODELS: dict[str, dict] = {
        "n": {
            "weights": "yolov8n.pt",
            "label": "Nano — fastest",
            "domain": COCO_DOMAIN,
            "downloadable": True,
        },
        "s": {
            "weights": "yolov8s.pt",
            "label": "Small — balanced",
            "domain": COCO_DOMAIN,
            "downloadable": True,
        },
        "m": {
            "weights": "yolov8m.pt",
            "label": "Medium — most accurate",
            "domain": COCO_DOMAIN,
            "downloadable": True,
        },
        "satellite": {
            "weights": str(WEIGHTS_DIR / "satellite.pt"),
            "label": "Satellite — cars & pools",
            "domain": SATELLITE_DOMAIN,
            "downloadable": False,
        },
    }

    DEFAULT_MODEL = "n"

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}
        # Guards cache insertion so two simultaneous first-requests for the
        # same model don't both pay to construct it.
        self._load_lock = threading.Lock()
        # Neither backend is documented as thread-safe, so inference on a given
        # model is serialised. Parallelism comes from gunicorn workers.
        self._infer_locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in self.MODELS
        }

    @classmethod
    def is_known(cls, name: str) -> bool:
        return name in cls.MODELS

    @classmethod
    def catalog(cls) -> dict[str, dict]:
        """Model metadata for the UI, including whether each one is usable."""
        return {
            name: {
                "label": spec["label"],
                "domain": spec["domain"],
                "available": cls._weights_present(spec),
            }
            for name, spec in cls.MODELS.items()
        }

    @staticmethod
    def _display_path(weights: str) -> str:
        path = Path(weights)
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return path.name

    @staticmethod
    def _weights_present(spec: dict) -> bool:
        if spec["downloadable"]:
            return True
        return Path(spec["weights"]).is_file()

    def loaded_models(self) -> list[str]:
        return sorted(self._cache)

    def _get_model(self, name: str):
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        spec = self.MODELS[name]
        if not self._weights_present(spec):
            # Reported relative to the repo so the message is useful to the
            # reader without publishing the server's directory layout.
            raise WeightsUnavailable(
                f"No weights found at {self._display_path(spec['weights'])}. "
                "Train the satellite model with `python train_satellite.py` first."
            )

        with self._load_lock:
            # Re-check: another thread may have loaded it while we waited.
            if name not in self._cache:
                weights = spec["weights"]
                fmt = checkpoint_format(weights)
                logger.info("Loading %s from %s (%s format)", name, weights, fmt)

                backend = BACKENDS[fmt](weights)
                self._warm_up(backend)
                self._cache[name] = backend
        return self._cache[name]

    @staticmethod
    def _warm_up(backend) -> None:
        """Absorb torch's one-off lazy initialisation at load time.

        Without this the first real request reports an inference time roughly
        fifty times its steady-state value, which makes the number shown in the
        UI meaningless on the very first scan.
        """
        try:
            backend.warm_up()
        except Exception:  # pragma: no cover - warmup is best effort
            logger.warning("Warm-up pass failed; first timing may be inflated")

    def run(self, image: Image.Image, name: str, min_confidence: float) -> dict:
        backend = self._get_model(name)
        spec = self.MODELS[name]

        with self._infer_locks[name]:
            start = time.perf_counter()
            detections = backend.infer(image, min_confidence)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return {
            "width": image.width,
            "height": image.height,
            "model": name,
            "domain": spec["domain"],
            "inference_ms": elapsed_ms,
            "detections": detections,
        }
