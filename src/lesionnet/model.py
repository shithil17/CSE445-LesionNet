import functools

import torch
import torch.nn as nn
from torchvision import models

from lesionnet.config import CLASS_NAMES, DEVICE, ENSEMBLE_MEMBERS, MODEL_DIR


def build_model(num_classes: int = 7) -> nn.Module:
    """Mirror of the image-only path of Model/Final/EfficientNetV2S_HAM10K.py
    build_model() (lines 737-781): EfficientNetV2-S + Dropout(0.2) +
    Linear(1280, 7)."""
    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, num_classes),
    )
    return model


@functools.lru_cache(maxsize=1)
def load_ensemble() -> list[tuple[nn.Module, float]]:
    """Load the F1-weighted ensemble members once and cache them (eval mode).

    Returns [(model, weight)] with weight = each checkpoint's saved val
    Macro-F1, matching _ensemble_probs() in the training script
    (EfficientNetV2S_HAM10K.py:1356).
    """
    ensemble = []
    for name in ENSEMBLE_MEMBERS:
        path = MODEL_DIR / name
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        if checkpoint.get("backbone", "v2s") != "v2s":
            raise ValueError(f"Checkpoint {name} is not an EfficientNet-V2S model")
        model = build_model(num_classes=len(CLASS_NAMES)).to(DEVICE)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"])
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/architecture mismatch in {name}: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys"
            )
        model.eval()
        f1 = float(checkpoint.get("best_macro_f1", 1.0))
        ensemble.append((model, max(f1, 1e-6)))
    return ensemble