"""Satellite / general-purpose object detection web app.

HTTP layer only: request validation, error mapping and JSON shaping. All model
work lives in :mod:`detector`.

The client uploads an image once and receives every detection down to a low
confidence floor; the confidence slider in the browser then filters that same
response, so sweeping the threshold costs no further requests.
"""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, ImageOps

from detector import Detector, WeightsUnavailable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 16 * 1024 * 1024

# Detections below this are never returned. The client-side slider filters
# everything above it, so it can sit low without flooding the response.
MIN_SERVER_CONFIDENCE = 0.10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app)

detector = Detector()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@app.errorhandler(413)
def upload_too_large(_error):
    """Return JSON rather than Werkzeug's HTML page.

    The frontend parses every response as JSON, so an HTML body here surfaces
    as a parser error instead of a usable message.
    """
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify({"error": f"That image is larger than {limit_mb}MB."}), 413


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=Detector.catalog(),
        default_model=Detector.DEFAULT_MODEL,
        max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
    )


@app.route("/api/models")
def models():
    return jsonify(Detector.catalog())


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image was attached to the request."}), 400

    upload = request.files["image"]
    model_name = request.form.get("model", Detector.DEFAULT_MODEL)

    if not Detector.is_known(model_name):
        return jsonify({"error": f"Unknown model '{model_name}'."}), 400

    try:
        image = Image.open(upload.stream)
        image.load()
        # Cameras record the sensor's raw orientation and attach an EXIF tag
        # telling viewers how to rotate it. Browsers honour that tag, so
        # without this the model is fed a sideways image while the user looks
        # at an upright one -- which both garbles the class predictions and
        # makes the reported width/height disagree with what the canvas draws.
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        return jsonify({"error": "That file couldn't be read as an image."}), 400

    try:
        result = detector.run(image, model_name, MIN_SERVER_CONFIDENCE)
    except WeightsUnavailable as exc:
        # Expected when the satellite model hasn't been trained yet.
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("Inference failed for model %s", model_name)
        return jsonify({"error": "Detection failed on the server. Check the logs."}), 500

    return jsonify(result)


@app.route("/train", methods=["POST"])
def train():
    """Kick off the satellite training pipeline.

    Disabled unless ENABLE_TRAIN_ENDPOINT is set, because it is unauthenticated
    and runs for a long time: left open on a deployed instance, anyone who can
    reach the URL can saturate the box. It also blocks the worker it runs on,
    so it is meant for local use rather than production traffic.
    """
    if not _env_flag("ENABLE_TRAIN_ENDPOINT"):
        return (
            jsonify(
                {
                    "error": "Training endpoint is disabled. Set "
                    "ENABLE_TRAIN_ENDPOINT=1 to enable it, or run "
                    "`python train_satellite.py` directly."
                }
            ),
            403,
        )

    try:
        from satelliteDetection.pipeline.training_pipeline import TrainPipeline

        TrainPipeline().run_pipeline()
    except Exception:
        logger.exception("Training pipeline failed")
        return jsonify({"error": "Training failed. Check the logs."}), 500

    return jsonify({"status": "training complete"})


@app.route("/healthz")
def healthz():
    return jsonify(
        {
            "status": "ok",
            "loaded_models": detector.loaded_models(),
            "available_models": {
                name: meta["available"] for name, meta in Detector.catalog().items()
            },
        }
    )


if __name__ == "__main__":
    # Defaults match this repo's existing deployment constants. Port 5000 is
    # deliberately avoided: macOS binds it to AirPlay Receiver by default.
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        debug=_env_flag("FLASK_DEBUG"),
    )
