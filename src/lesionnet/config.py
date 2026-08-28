import torch
from pathlib import Path

CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]  # ImageFolder order

# EfficientNetV2-S training/eval geometry (Model/Final/EfficientNetV2S_HAM10K.py).
IMAGE_SIZE = 380
EVAL_RESIZE = 380
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_DIR = Path(__file__).resolve().parent

# F1-weighted ensemble, mirroring the shipped config in
# efficientnetv2s_training_metrics.pth (members = top-2 val Macro-F1).
ENSEMBLE_MEMBERS = [
    "efficientnetv2s_best_model_epoch22.pth",
    "efficientnetv2s_best_model_epoch19.pth",
]

# Shipped ensemble temperature (grid search on val ECE). Applied to the
# averaged probs via log(probs)/T, exactly like the training pipeline.
ENSEMBLE_TEMPERATURE = 0.6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}