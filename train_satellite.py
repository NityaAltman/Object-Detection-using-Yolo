"""Train the two-class satellite detector (``car`` / ``swimming pool``).

Produces ``weights/satellite.pt``, which the web app picks up automatically —
the "sat" model option turns from unavailable to available once this finishes.

    python train_satellite.py --epochs 50

Ultralytics resolves the dataset paths in a data YAML relative to that file's
own location, and the committed ``data/data.yaml`` points at ``../train/images``
which lands outside the data directory. Rather than edit that file in place,
this script writes a corrected copy with absolute paths before training.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "data"
WEIGHTS_DIR = REPO_ROOT / "weights"
CLASS_NAMES = ["car", "swimming pool"]


def build_dataset_config(dataset_dir: Path, destination: Path) -> Path:
    """Write a data YAML with absolute train/val paths."""
    train_images = dataset_dir / "train" / "images"
    val_images = dataset_dir / "valid" / "images"

    missing = [p for p in (train_images, val_images) if not p.is_dir()]
    if missing:
        raise SystemExit(
            "Dataset is incomplete; these directories are missing:\n  "
            + "\n  ".join(str(p) for p in missing)
            + f"\n\nUnzip the dataset first:  unzip data.zip -d {dataset_dir}"
        )

    config = {
        "path": str(dataset_dir),
        "train": str(train_images),
        "val": str(val_images),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    return destination


def report_label_coverage(dataset_dir: Path) -> None:
    """Warn about images with no label file.

    YOLO treats those as background-only examples rather than erroring, which
    is easy to miss: this dataset ships a few hundred of them.
    """
    for split in ("train", "valid"):
        images = dataset_dir / split / "images"
        labels = dataset_dir / split / "labels"
        if not (images.is_dir() and labels.is_dir()):
            continue

        image_stems = {p.stem for p in images.iterdir() if p.is_file()}
        label_stems = {p.stem for p in labels.iterdir() if p.suffix == ".txt"}
        unlabelled = len(image_stems - label_stems)
        if unlabelled:
            print(
                f"  note: {split} has {unlabelled} of {len(image_stems)} images "
                "with no label file; they will train as background examples."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Directory holding train/ and valid/ (default: data)")
    parser.add_argument("--base-model", default="yolov8s.pt",
                        help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None,
                        help="e.g. 0 for the first GPU, or cpu. Defaults to auto.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        )

    dataset_dir = args.dataset.resolve()
    print(f"Dataset: {dataset_dir}")
    report_label_coverage(dataset_dir)

    config_path = build_dataset_config(
        dataset_dir, REPO_ROOT / "artifacts" / "satellite_data.yaml"
    )
    print(f"Wrote dataset config: {config_path}")

    results = YOLO(args.base_model).train(
        data=str(config_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(REPO_ROOT / "artifacts" / "satellite"),
        name="train",
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Training finished but {best} was not produced.")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS_DIR / "satellite.pt"
    shutil.copy2(best, target)

    print(f"\nDone. Copied {best} -> {target}")
    print("Restart the app and the 'sat' model will be selectable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
