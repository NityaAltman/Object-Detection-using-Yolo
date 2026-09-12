"""Shared fixtures.

The model itself is always stubbed. That keeps the suite fast and lets it run
in CI without pulling torch or downloading weights, while still exercising the
real request validation, error mapping and JSON shaping in ``app``.
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module  # noqa: E402
from detector import WeightsUnavailable  # noqa: E402


class StubDetector:
    """Stands in for :class:`detector.Detector` without loading any weights."""

    def __init__(self, detections=None, raises=None):
        self._detections = detections if detections is not None else []
        self._raises = raises
        self.calls = []

    def run(self, image, name, min_confidence):
        self.calls.append((name, min_confidence))
        if self._raises is not None:
            raise self._raises
        return {
            "width": image.width,
            "height": image.height,
            "model": name,
            "domain": "coco",
            "inference_ms": 12.3,
            "detections": self._detections,
        }

    def loaded_models(self):
        return [] if self._raises else ["n"]


@pytest.fixture
def stub():
    return StubDetector()


@pytest.fixture
def client(stub, monkeypatch):
    monkeypatch.setattr(app_module, "detector", stub)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def unavailable_client(monkeypatch):
    """A client whose detector reports the satellite weights as missing."""
    failing = StubDetector(
        raises=WeightsUnavailable("No weights found at weights/satellite.pt.")
    )
    monkeypatch.setattr(app_module, "detector", failing)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def png_bytes(size=(64, 48), color=(120, 90, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def image_upload():
    return lambda: {"image": (png_bytes(), "example.png")}
