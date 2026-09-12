# Python 3.11: current torch wheels no longer target 3.8, which the previous
# base image pinned.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    # Keep Ultralytics' config and weight cache inside the app directory so it
    # stays writable for the non-root user below.
    YOLO_CONFIG_DIR=/app/.cache \
    MPLCONFIGDIR=/app/.cache

WORKDIR /app

# libGL and libglib are required by opencv, which ultralytics depends on.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the default weights into the image so the first request doesn't have to
# reach the network, which would otherwise make cold starts slow and make the
# container unusable in an egress-restricted environment.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# A real WSGI server: app.py's built-in server is for local development only.
# One worker because each one holds its own copy of the model in memory.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app"]
