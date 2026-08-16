"""Portable configuration for the pneumonia classification experiment."""

from __future__ import annotations

import os
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(
    os.environ.get("PNEUMONIA_DATASET_DIR", REPO_DIR / "dataset" / "chest_xray")
).expanduser().resolve()
MODELS_DIR = REPO_DIR / "saved_models"
CHARTS_DIR = REPO_DIR / "report_charts"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

RESNET_PATH = MODELS_DIR / "best_resnet50.keras"
MOBILENET_PATH = MODELS_DIR / "best_mobilenetv2.keras"
CNN_PATH = MODELS_DIR / "best_custom_cnn.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.2


def ensure_output_dirs() -> None:
    """Create local model and chart directories when a run begins."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_dataset_layout() -> None:
    """Fail early when required dataset folders are missing."""

    required = [
        split / label
        for split in (TRAIN_DIR, VAL_DIR, TEST_DIR)
        for label in ("NORMAL", "PNEUMONIA")
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        joined = "\n- ".join(missing)
        raise FileNotFoundError(
            "Dataset layout is incomplete. Missing:\n- " + joined
            + "\nSet PNEUMONIA_DATASET_DIR if the dataset is elsewhere."
        )
