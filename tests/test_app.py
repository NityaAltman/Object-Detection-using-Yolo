"""Route-level tests for the detection API."""
import io

import app as app_module
from conftest import png_bytes


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Nitya Vision System" in response.data


def test_models_endpoint_lists_every_model(client):
    payload = client.get("/api/models").get_json()
    assert set(payload) == {"n", "s", "m", "satellite"}
    assert payload["n"]["available"] is True
    assert payload["satellite"]["domain"] == "satellite"


def test_healthz_reports_availability(client):
    payload = client.get("/healthz").get_json()
    assert payload["status"] == "ok"
    assert "satellite" in payload["available_models"]


def test_detect_returns_detections(client, image_upload, stub):
    response = client.post(
        "/api/detect", data={**image_upload(), "model": "n"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["width"] == 64
    assert payload["height"] == 48
    assert payload["model"] == "n"
    assert payload["detections"] == []

    # The server floor must stay low so the client-side slider has headroom.
    (_, min_confidence), = stub.calls
    assert min_confidence == app_module.MIN_SERVER_CONFIDENCE


def test_detect_defaults_to_the_default_model(client, image_upload, stub):
    client.post("/api/detect", data=image_upload(),
                content_type="multipart/form-data")
    assert stub.calls[0][0] == "n"


def test_detect_requires_an_image(client):
    response = client.post("/api/detect", data={},
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert "No image" in response.get_json()["error"]


def test_detect_rejects_unknown_model(client, image_upload):
    response = client.post(
        "/api/detect", data={**image_upload(), "model": "enormous"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "enormous" in response.get_json()["error"]


def test_detect_rejects_a_file_that_is_not_an_image(client):
    response = client.post(
        "/api/detect",
        data={"image": (io.BytesIO(b"not an image at all"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "couldn't be read" in response.get_json()["error"]


def test_oversized_upload_returns_json_not_html(client):
    """Regression test.

    Werkzeug's default 413 body is an HTML page. The frontend parses every
    response as JSON, so an HTML body surfaced to the user as a parser error
    instead of a message about the size limit.
    """
    oversized = io.BytesIO(b"\0" * (app_module.MAX_UPLOAD_BYTES + 1024))
    response = client.post(
        "/api/detect", data={"image": (oversized, "huge.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    assert response.is_json
    assert "16MB" in response.get_json()["error"]


def test_missing_satellite_weights_is_a_503_with_guidance(
    unavailable_client, image_upload
):
    response = unavailable_client.post(
        "/api/detect", data={**image_upload(), "model": "satellite"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    assert "weights" in response.get_json()["error"].lower()


def test_train_endpoint_is_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("ENABLE_TRAIN_ENDPOINT", raising=False)
    response = client.post("/train")
    assert response.status_code == 403
    assert "disabled" in response.get_json()["error"]


def test_train_endpoint_rejects_get(client):
    assert client.get("/train").status_code == 405


def test_detect_handles_a_jpeg(client):
    buffer = io.BytesIO()
    from PIL import Image

    Image.new("RGB", (20, 10), (10, 200, 10)).save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/api/detect", data={"image": (buffer, "photo.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["width"] == 20


def test_detect_converts_transparent_png(client):
    """RGBA input must not blow up on the way to the model."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (12, 12), (0, 0, 0, 0)).save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/api/detect", data={"image": (buffer, "transparent.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200


def _jpeg_with_orientation(orientation, size=(40, 20)):
    """A JPEG whose stored pixels need rotating to display correctly."""
    from PIL import Image

    image = Image.new("RGB", size, (200, 40, 40))
    exif = image.getexif()
    exif[0x0112] = orientation  # EXIF Orientation tag
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    buffer.seek(0)
    return buffer


def test_exif_orientation_is_applied_before_inference(client):
    """Regression test.

    Orientation 6 means "rotate 90° clockwise to display", so a 40x20 stored
    image is really a 20x40 picture. Browsers honour that tag when rendering,
    so feeding the model the raw pixels meant it saw every phone photo on its
    side: confident labels turned into unrelated low-confidence ones, and the
    dimensions reported back disagreed with what the canvas had drawn.
    """
    response = client.post(
        "/api/detect",
        data={"image": (_jpeg_with_orientation(6), "sideways.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert (payload["width"], payload["height"]) == (20, 40)


def test_exif_orientation_1_is_left_alone(client):
    """Orientation 1 is "already upright" and must not transpose anything."""
    response = client.post(
        "/api/detect",
        data={"image": (_jpeg_with_orientation(1), "upright.jpg")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    assert (payload["width"], payload["height"]) == (40, 20)


def test_image_without_exif_is_unchanged(client):
    """A PNG carries no orientation tag; dimensions must pass through as-is."""
    response = client.post(
        "/api/detect",
        data={"image": (png_bytes(size=(30, 15)), "plain.png")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    assert (payload["width"], payload["height"]) == (30, 15)


def test_unexpected_inference_error_is_a_500(client, monkeypatch, image_upload):
    def explode(*_args, **_kwargs):
        raise RuntimeError("CUDA is on fire")

    monkeypatch.setattr(app_module.detector, "run", explode)
    response = client.post("/api/detect", data=image_upload(),
                           content_type="multipart/form-data")

    assert response.status_code == 500
    # The underlying message must not leak to the client.
    assert "CUDA" not in response.get_data(as_text=True)


def test_png_helper_produces_a_readable_image():
    from PIL import Image

    assert Image.open(png_bytes(size=(8, 8))).size == (8, 8)
