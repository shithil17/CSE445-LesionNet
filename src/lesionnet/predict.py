from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms

from lesionnet.config import (
    CLASS_NAMES,
    DEVICE,
    ENSEMBLE_TEMPERATURE,
    EVAL_RESIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from lesionnet.model import load_ensemble
from lesionnet.xai import GradCAM, render_overlay

# Deterministic eval transform, mirror of _build_transforms() in
# Model/Final/EfficientNetV2S_HAM10K.py (lines 561-566).
_transform = transforms.Compose(
    [
        transforms.Resize(EVAL_RESIZE),  # shorter side, aspect preserved
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def _tta_views(tensor):
    """Deterministic TTA views, mirror of predict_with_tta() n_augments=4:
    identity, horizontal flip, +5° rotation, -5° rotation."""
    return (
        tensor,
        torch.flip(tensor, dims=[2]),
        transforms.functional.rotate(tensor, 5),
        transforms.functional.rotate(tensor, -5),
    )


def _apply_temperature(logits, temperature):
    """Mirror of EfficientNetV2S_HAM10K._apply_temperature (line 1139):
    softmax(logits / T) with max-subtraction stability."""
    shifted = (logits - logits.max(axis=1, keepdims=True)) / temperature
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _ensemble_probs(tensor):
    """F1-weighted TTA ensemble probs (numpy, 7,). Each member contributes
    softmax(mean TTA logits) * val-Macro-F1 weight, like _ensemble_probs()."""
    tensor = tensor.to(DEVICE)
    probs_sum = None
    weight_sum = 0.0
    for model, weight in load_ensemble():
        logits_sum = None
        n_views = 0
        with torch.no_grad():
            for view in _tta_views(tensor):
                logits = model(view.unsqueeze(0))
                logits_sum = logits if logits_sum is None else logits_sum + logits
                n_views += 1
        probs = F.softmax(logits_sum / n_views, dim=1)[0]
        probs_sum = probs * weight if probs_sum is None else probs_sum + probs * weight
        weight_sum += weight
    return (probs_sum / weight_sum).cpu().numpy()


def _temperature_scaled(probs):
    """Apply the shipped ensemble temperature to the averaged probs via
    log(probs)/T, exactly like fit_temperature_on_probs() /
    _grid_search_temperature(is_probs=True). Argmax must be unchanged."""
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))[None, :]
    scaled = _apply_temperature(log_probs, ENSEMBLE_TEMPERATURE)[0]
    if int(np.argmax(scaled)) != int(np.argmax(probs)):
        raise RuntimeError("temperature scaling changed the argmax prediction")
    return scaled


def predict(image):
    """Return {'probs': [...7...], 'pred_class': str, 'pred_conf': float}.

    probs are in CLASS_NAMES order (index == class_to_idx).
    """
    image = image.convert("RGB")
    tensor = _transform(image)
    probs = _temperature_scaled(_ensemble_probs(tensor))
    idx = int(probs.argmax())
    return {
        "probs": probs,
        "pred_class": CLASS_NAMES[idx],
        "pred_conf": float(probs[idx]),
    }


def build_result_image(original, overlay, probs, gender, age):
    """Compose original + overlay side by side with a prediction panel below."""
    original = original.convert("RGB")
    overlay = overlay.convert("RGB").resize(original.size)

    w, h = original.size
    side = PIL.Image.new("RGB", (w * 2, h))
    side.paste(original, (0, 0))
    side.paste(overlay, (w, 0))

    top_idx = int(np.argmax(probs))
    header = f"Prediction: {CLASS_NAMES[top_idx]}  ({probs[top_idx] * 100:.1f}%)"
    meta = f"Gender: {gender or '-'}   Age: {age if age else '-'}"
    panel_h = 24 + 7 * 20 + 26
    panel = PIL.Image.new("RGB", (side.width, panel_h), "white")
    draw = PIL.ImageDraw.Draw(panel)
    font = PIL.ImageFont.load_default(size=14)

    draw.text((12, 6), header, fill="black", font=font)
    draw.text((12, 26), meta, fill="black", font=font)

    bar_x0, bar_w = 160, side.width - 190
    for rank, i in enumerate(np.argsort(probs)[::-1]):
        y = 48 + rank * 20
        draw.text((12, y), f"{CLASS_NAMES[i]}:  {probs[i] * 100:5.1f}%", fill="black", font=font)
        draw.rectangle([bar_x0, y + 2, bar_x0 + bar_w, y + 8], fill="lightgray")
        draw.rectangle([bar_x0, y + 2, bar_x0 + int(bar_w * probs[i]), y + 8], fill="red")

    composite = PIL.Image.new("RGB", (side.width, side.height + panel.height))
    composite.paste(side, (0, 0))
    composite.paste(panel, (0, side.height))
    return composite


def predict_full(image, gender, age, output_dir=None):
    """predict() + Grad-CAM overlay + composite image + written file paths."""
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        image = PIL.Image.open(image_path)
        image.filename = str(image_path)
    image_filename = getattr(image, "filename", None)
    original = image.convert("RGB")
    result = predict(original)

    cam_model = GradCAM(load_ensemble()[0][0])
    try:
        tensor = _transform(original)
        cam = cam_model.generate(tensor, int(np.argmax(result["probs"])))
    finally:
        cam_model.remove()

    overlay = render_overlay(original, cam)
    composite = build_result_image(original, overlay, result["probs"], gender, age)

    stem = Path(image_filename).stem if image_filename else "result"
    out_dir = Path(output_dir) if output_dir else Path("/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{stem}_overlay.png"
    composite_path = out_dir / f"{stem}_result.png"
    overlay.save(overlay_path)
    composite.save(composite_path)

    result["overlay"] = str(overlay_path)
    result["composite"] = composite
    result["composite_path"] = str(composite_path)
    return result