"""Unit tests for the model wrapper.

Neither torch nor ultralytics is imported here. Both backends import their
heavy dependencies lazily inside ``__init__``, so injecting a fake backend into
the cache exercises the real envelope and ordering logic, and the Ultralytics
result-shaping is tested by constructing that backend without running its
constructor.
"""
import threading
from pathlib import Path

import pytest
from PIL import Image

import detector as detector_module
from detector import (
    ULTRALYTICS_FORMAT,
    YOLOV5_FORMAT,
    Detector,
    WeightsUnavailable,
    checkpoint_format,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    """Stands in for a loaded model backend."""

    def __init__(self, detections=None):
        self._detections = detections or []
        self.calls = []
        self.warmed = False

    def infer(self, image, min_confidence):
        self.calls.append(min_confidence)
        return [dict(d) for d in self._detections]

    def warm_up(self):
        self.warmed = True


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def image():
    return Image.new("RGB", (200, 100), (5, 5, 5))


def test_catalog_marks_satellite_unavailable_when_untrained(tmp_path, monkeypatch):
    monkeypatch.setitem(
        Detector.MODELS["satellite"], "weights", str(tmp_path / "absent.pt")
    )
    catalog = Detector.catalog()

    assert catalog["satellite"]["available"] is False
    # The downloadable COCO models are always considered available.
    assert catalog["n"]["available"] is True


def test_catalog_marks_satellite_available_once_weights_exist(tmp_path, monkeypatch):
    weights = tmp_path / "satellite.pt"
    weights.write_bytes(b"not a real checkpoint")
    monkeypatch.setitem(Detector.MODELS["satellite"], "weights", str(weights))

    assert Detector.catalog()["satellite"]["available"] is True


def test_missing_satellite_weights_raises(detector, tmp_path, monkeypatch, image):
    monkeypatch.setitem(
        Detector.MODELS["satellite"], "weights", str(tmp_path / "absent.pt")
    )
    with pytest.raises(WeightsUnavailable) as excinfo:
        detector.run(image, "satellite", 0.1)

    message = str(excinfo.value)
    assert "train" in message.lower()
    # The message reaches the browser, so it must not disclose the server's
    # absolute directory layout.
    assert str(tmp_path) not in message


def test_is_known():
    assert Detector.is_known("n")
    assert Detector.is_known("satellite")
    assert not Detector.is_known("xl")


def test_run_sorts_detections_by_confidence(detector, image):
    detector._cache["n"] = FakeBackend(
        [
            {"class": "car", "confidence": 0.20, "box": [0, 0, 10, 10]},
            {"class": "swimming pool", "confidence": 0.90, "box": [5, 5, 25, 30]},
            {"class": "car", "confidence": 0.55, "box": [1, 2, 3, 4]},
        ]
    )

    result = detector.run(image, "n", 0.1)

    assert [d["confidence"] for d in result["detections"]] == [0.90, 0.55, 0.20]
    assert [d["class"] for d in result["detections"]] == [
        "swimming pool", "car", "car",
    ]
    assert result["width"] == 200
    assert result["height"] == 100
    assert result["domain"] == "coco"
    assert isinstance(result["inference_ms"], float)


def test_run_passes_the_confidence_floor_through(detector, image):
    backend = FakeBackend()
    detector._cache["n"] = backend

    detector.run(image, "n", 0.25)

    assert backend.calls == [0.25]


def test_satellite_domain_is_reported(detector, image, tmp_path, monkeypatch):
    weights = tmp_path / "satellite.pt"
    weights.write_bytes(b"stub")
    monkeypatch.setitem(Detector.MODELS["satellite"], "weights", str(weights))
    detector._cache["satellite"] = FakeBackend()

    assert detector.run(image, "satellite", 0.1)["domain"] == "satellite"


def test_loaded_models_reflects_the_cache(detector):
    assert detector.loaded_models() == []
    detector._cache["m"] = FakeBackend()
    detector._cache["n"] = FakeBackend()
    assert detector.loaded_models() == ["m", "n"]


def test_concurrent_inference_is_serialised(detector, image):
    """Two threads must not be inside ``infer`` on one model simultaneously."""
    overlaps = []
    active = 0
    guard = threading.Lock()

    class CountingBackend(FakeBackend):
        def infer(self, img, min_confidence):
            nonlocal active
            with guard:
                active += 1
                if active > 1:
                    overlaps.append(active)
            # Long enough that unsynchronised threads would overlap here.
            threading.Event().wait(0.05)
            with guard:
                active -= 1
            return super().infer(img, min_confidence)

    detector._cache["n"] = CountingBackend()
    threads = [
        threading.Thread(target=detector.run, args=(image, "n", 0.1))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert overlaps == []


def test_backend_is_warmed_up_on_load(detector, tmp_path, monkeypatch, image):
    """Loading must burn a throwaway prediction so the first timing is honest."""
    weights = tmp_path / "satellite.pt"
    weights.write_bytes(b"stub")
    monkeypatch.setitem(Detector.MODELS["satellite"], "weights", str(weights))

    created = []

    class RecordingBackend(FakeBackend):
        def __init__(self, _weights):
            super().__init__()
            created.append(self)

    monkeypatch.setitem(detector_module.BACKENDS, ULTRALYTICS_FORMAT, RecordingBackend)
    detector.run(image, "satellite", 0.1)

    assert len(created) == 1
    assert created[0].warmed is True


class TestCheckpointFormat:
    """The sniffing that decides which backend loads a file.

    Getting this wrong is not subtle -- ultralytics raises a TypeError on a
    YOLOv5 checkpoint rather than misbehaving quietly -- but it means a
    retrained YOLOv8 satellite model needs no config change to be picked up.
    """

    def test_yolov5_repo_checkpoint(self, tmp_path):
        path = tmp_path / "v5.pt"
        path.write_bytes(b"PK\x03\x04" + b"models.yolo" + b"models.common.Focus")
        assert checkpoint_format(str(path)) == YOLOV5_FORMAT

    def test_ultralytics_checkpoint(self, tmp_path):
        path = tmp_path / "v8.pt"
        path.write_bytes(b"PK\x03\x04" + b"ultralytics.nn.tasks")
        assert checkpoint_format(str(path)) == ULTRALYTICS_FORMAT

    def test_ultralytics_wins_when_both_appear(self, tmp_path):
        """Ultralytics vendors a `models` module path of its own."""
        path = tmp_path / "both.pt"
        path.write_bytes(b"models.yolo" + b"ultralytics.nn.tasks")
        assert checkpoint_format(str(path)) == ULTRALYTICS_FORMAT

    def test_missing_file_defaults_to_ultralytics(self, tmp_path):
        """Downloadable weights are sniffed before they exist on disk."""
        assert checkpoint_format(str(tmp_path / "nope.pt")) == ULTRALYTICS_FORMAT

    def test_real_satellite_checkpoint_if_present(self):
        weights = REPO_ROOT / "weights" / "satellite.pt"
        if not weights.is_file():
            pytest.skip("satellite.pt not present in this checkout")
        # The recovered checkpoint came from the original YOLOv5 repo.
        assert checkpoint_format(str(weights)) == YOLOV5_FORMAT


class TestUltralyticsResultShaping:
    """The Ultralytics backend's translation of boxes into plain dicts."""

    @staticmethod
    def _backend(boxes, names):
        class FakeRow:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        class FakeBox:
            def __init__(self, xyxy, confidence, class_id):
                self.xyxy = [FakeRow(xyxy)]
                self.conf = [confidence]
                self.cls = [class_id]

        class FakePrediction:
            def __init__(self):
                self.boxes = [FakeBox(*b) for b in boxes]
                self.names = names

        class FakeModel:
            def predict(self, image, conf=None, verbose=False):
                return [FakePrediction()]

        backend = object.__new__(detector_module._UltralyticsBackend)
        backend._model = FakeModel()
        return backend

    def test_maps_class_ids_through_the_names_table(self):
        backend = self._backend(
            [([0.0, 0.0, 10.0, 10.0], 0.5, 2)], {0: "person", 2: "car"}
        )
        assert backend.infer(Image.new("RGB", (10, 10)), 0.1) == [
            {"class": "car", "confidence": 0.5, "box": [0.0, 0.0, 10.0, 10.0]}
        ]

    def test_rounds_coordinates_and_confidence(self):
        backend = self._backend(
            [([1.23456, 2.0, 3.0, 4.0], 0.123456789, 0)], {0: "car"}
        )
        (detection,) = backend.infer(Image.new("RGB", (10, 10)), 0.1)
        assert detection["box"][0] == 1.2
        assert detection["confidence"] == 0.1235

    def test_no_boxes_gives_no_detections(self):
        backend = self._backend([], {0: "car"})
        assert backend.infer(Image.new("RGB", (10, 10)), 0.1) == []
