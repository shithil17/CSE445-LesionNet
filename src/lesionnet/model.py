import functools

import torch
import torch.nn as nn
from torchvision import models

from lesionnet.config import CLASS_NAMES, DEVICE, MODEL_PATH


def build_model(num_classes: int = 7) -> nn.Module:
    """Literal mirror of Model/EfficientNetB4_HAM10K.py:121-135."""
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, num_classes),
    )
    return model


@functools.lru_cache(maxsize=1)
def load_model() -> nn.Module:
    """Load the trained checkpoint once and cache it (eval mode)."""
    model = build_model(num_classes=len(CLASS_NAMES)).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/architecture mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys"
        )
    model.eval()
    return model