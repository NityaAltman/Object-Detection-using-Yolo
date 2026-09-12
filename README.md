# Nitya Vision System

Object detection over HTTP: drop in an image, get bounding boxes back, and
sweep the confidence threshold live in the browser without re-running
inference.

Two families of weights are served from the same UI:

| Model | Classes | Best for |
|---|---|---|
| `n` / `s` / `m` | COCO's 80 general classes (person, car, bus, dog, …) | Ordinary ground-level photographs |
| `sat` | 2 classes: `car`, `swimming pool` | Overhead / satellite imagery |

The COCO models are downloaded on first use, so the app works the moment you
clone it. They are useless on aerial imagery though — they have no
swimming-pool class at all, and having been trained on ground-level photos
they will confidently label an overhead car a "cell phone", which is roughly
what a small dark rectangle seen from above looks like in their world.

The `sat` model is the one this project exists for. It needs
`weights/satellite.pt`; without it that option reports itself as untrained
rather than failing silently.

### Checkpoint formats

`weights/satellite.pt` is a checkpoint from the **original YOLOv5 repository**,
which the modern `ultralytics` package refuses to load outright. `detector.py`
therefore keeps two backends and picks one by sniffing the file, so dropping in
a freshly trained YOLOv8 checkpoint needs no code change. Serving the YOLOv5
one runs the vendored `yolov5/` code, which is why `setuptools<81` is pinned —
that code still imports `pkg_resources`.

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Then open <http://localhost:8080>.

Port 8080 matches the deployment in `.github/workflows/main.yaml`. Override
with `PORT`, `HOST` and `FLASK_DEBUG` if needed. Note that **port 5000 is not
a good default on macOS** — AirPlay Receiver binds it, so you get
`Address already in use`.

With Docker:

```bash
docker build -t vision .
docker run -p 8080:8080 vision
```

The image bakes in the `yolov8n` weights, so a cold container answers its
first request without reaching the network, and it serves through gunicorn
rather than Flask's development server.

## Training the satellite model

The current `weights/satellite.pt` is the YOLOv5s model from
`research/SatelliteObjectDetection.ipynb`, trained with
`--img 416 --epochs 50`. Its validation scores were:

| | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| car | 0.570 | 0.604 | 0.475 | 0.222 |
| swimming pool | 0.829 | 0.757 | 0.778 | 0.396 |

Pools do well; cars are weak — a recall of 0.604 means it misses about 40% of
them. Retraining is worthwhile, and the dataset is already in the repo
(`data/train`, `data/valid` — about 3,400 images; `data.zip` is the original):

```bash
python train_satellite.py --epochs 50
```

This overwrites `weights/satellite.pt`. Expect hours on CPU — it wants a GPU.

Two levers matter most, because **the median object is only 11x11 px** on a
224x224 tile and 77% of boxes are under 16px on a side:

- **Input resolution.** Upscaling is what makes these objects survive the
  network's downsampling: at 416px an 11px car is ~20px, at 640px it's ~31px.
  The old model used 416, so `train_satellite.py`'s default of 640 should beat
  it on cars for that reason alone.
- **Rotation augmentation.** Ultralytics defaults to `degrees=0.0` and
  `flipud=0.0` because ground-level photos have a definite "up". Aerial images
  don't, so `degrees=180` and `flipud=0.5` are close to free accuracy here.

Two things worth knowing about the data:

- `data/data.yaml` declares `train: ../train/images`, which resolves outside
  the data directory and breaks Ultralytics' loader. `train_satellite.py`
  writes a corrected config with absolute paths instead of editing that file.
- 200 training and 63 validation images have no corresponding label file. YOLO
  trains those as background examples rather than erroring, so the script
  prints a count on startup instead of letting it pass unnoticed.
- The validation split is lopsided: 3,275 cars but only 144 swimming pools,
  against a 2.4:1 ratio in training. Pool metrics from it are therefore noisy,
  and worth re-splitting before trusting them.

## Layout

- **`app.py`** — HTTP layer only: validation, error mapping, JSON shaping.
- **`detector.py`** — model registry, lazy loading, caching, inference.
- **`templates/`, `static/`** — single-page frontend, no build step and no
  framework.
- **`train_satellite.py`** — fine-tunes a COCO checkpoint on the 2-class data.
- **`satelliteDetection/`** — the original data ingestion / validation /
  training pipeline package.
- **`tests/`** — route and detector tests. The model is stubbed, so the suite
  runs in about a second and CI needs neither torch nor any weights.

```bash
pip install -r requirements-dev.txt
pytest
```

## Design notes

**The confidence slider costs nothing.** `/api/detect` returns every detection
down to 10% confidence once. The slider filters that same response client-side
and redraws, so sweeping the threshold triggers no further requests.

**Inference timing is measured after a warm-up pass.** Torch does a chunk of
lazy initialisation on its first `predict` call. Timing that would report
~2,600 ms on a first scan against ~43 ms steady-state, so `detector.py` burns
a throwaway 32×32 prediction at load time and the number shown in the UI is
the honest one.

**EXIF orientation is applied before inference.** Phone cameras store pixels
in the sensor's orientation and attach a tag saying how to rotate them.
Browsers honour that tag, so passing the raw pixels to the model meant it saw
sideways photos while the user saw upright ones — a confident `bus 96%` became
`train 24%` / `chair 25%`, and the reported dimensions disagreed with what the
canvas drew, putting the boxes in the wrong places. Ultralytics handles this
for file paths but not for an already-decoded PIL image, which is what the API
passes it.

**Inference is serialised per model.** Ultralytics models aren't documented as
thread-safe, so each is guarded by a lock; parallelism is meant to come from
running more gunicorn workers, not more threads against one model.

**`/train` is disabled by default.** It's unauthenticated and long-running, so
an open instance would let anyone saturate the box. Set
`ENABLE_TRAIN_ENDPOINT=1` to turn it on, or just run `train_satellite.py`.

## Deployment (AWS ECR + EC2)

`.github/workflows/main.yaml` builds the image, pushes it to ECR, and runs it
on a self-hosted EC2 runner on port 8080.

One-time setup:

1. Create an IAM user with `AmazonEC2ContainerRegistryFullAccess` and
   `AmazonEC2FullAccess`.
2. Create an ECR repository and note its URI.
3. Launch an Ubuntu EC2 instance and install Docker:

   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

4. Register that instance as a self-hosted runner under
   *Settings → Actions → Runners → New self-hosted runner*.
5. Add these repository secrets: `AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ECR_LOGIN_URI`,
   `ECR_REPOSITORY_NAME`.

## Limitations

**Large images need tiling.** Inference downscales the whole image to the
model's input size, so on a 2400x1839 satellite photo an 11px car becomes
about 3px and disappears — `satellite1.png` returns zero detections at full
size while the 224px tiles it was cut from detect fine. Small crops work; whole
satellite scenes need slicing into overlapping windows and merging the results
(what the SAHI library does). Not implemented yet.

**Pick the model to match the viewpoint.** The COCO models have no
swimming-pool class and mislabel overhead cars; `sat` knows nothing but cars
and pools and is calibrated for 416px aerial tiles. Neither is a general
detector, which is why the selector exposes both instead of pretending there
is one.
