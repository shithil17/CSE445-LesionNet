import os
from pathlib import Path

import torch

CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]  # ImageFolder order

IMAGE_SIZE = 224

_REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get(
    "LESIONNET_MODEL_PATH",
    str(_REPO_ROOT / "Model" / "efficientnetb4_classifier.pth"),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}