import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from matplotlib import cm


def find_last_conv(model):
    """Return the last nn.Conv2d in model.features (reverse-walk)."""
    target = None
    for module in model.features.modules():
        if isinstance(module, nn.Conv2d):
            target = module
    return target


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target = find_last_conv(model)
        self._activations = None
        self._gradients = None
        self._forward_handle = self.target.register_forward_hook(self._save_activations)
        self._backward_handle = self.target.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx):
        """Generate a [0,1] Grad-CAM map. Autograd must stay on."""
        self._activations = self._gradients = None
        device = next(self.model.parameters()).device
        image = image_tensor.detach().unsqueeze(0).to(device).requires_grad_(True)
        logits = self.model(image)
        logits[0, class_idx].backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        vmax = cam.max()
        return (cam - cam.min()) / (vmax - cam.min()) if vmax > 0 else cam

    def remove(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


def render_overlay(original_pil, cam):
    """Blend the jet-colormapped CAM over the original image at full resolution."""
    original = original_pil.convert("RGB")
    heatmap = PIL.Image.fromarray((cm.jet(cam) * 255).astype(np.uint8))
    heatmap = heatmap.resize(original.size).convert("RGB")
    return PIL.Image.blend(original, heatmap, alpha=0.5)