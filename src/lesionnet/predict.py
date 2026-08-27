from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms

from lesionnet.config import CLASS_NAMES, IMAGE_SIZE
from lesionnet.model import load_model
from lesionnet.xai import GradCAM, render_overlay

_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


def predict(image):
    """Return {'probs': [...7...], 'pred_class': str, 'pred_conf': float}.

    probs are in CLASS_NAMES order (index == class_to_idx).
    """
    image = image.convert("RGB")
    model = load_model()
    device = next(model.parameters()).device
    tensor = _transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits[0], dim=0).cpu().numpy()
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


def predict_full(image, gender, age):
    """predict() + Grad-CAM overlay + composite image + written file path."""
    image_filename = getattr(image, "filename", None)
    original = image.convert("RGB")
    result = predict(original)

    cam_model = GradCAM(load_model())
    try:
        tensor = _transform(original)
        cam = cam_model.generate(tensor, int(np.argmax(result["probs"])))
    finally:
        cam_model.remove()

    overlay = render_overlay(original, cam)
    composite = build_result_image(original, overlay, result["probs"], gender, age)

    stem = Path(image_filename).stem if image_filename else "result"
    overlay_path = Path("/tmp") / f"{stem}_overlay.png"
    composite_path = Path("/tmp") / f"{stem}_result.png"
    overlay.save(overlay_path)
    composite.save(composite_path)

    result["overlay"] = str(overlay_path)
    result["composite"] = composite
    result["composite_path"] = str(composite_path)
    return result