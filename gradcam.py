"""
utils/gradcam.py
────────────────
Gradient-weighted Class Activation Mapping (Grad-CAM) for DenseNet-121.

Usage
-----
    from utils.gradcam import GradCAM, overlay_heatmap
    gcam   = GradCAM(model, target_layer="features.denseblock4.denselayer16.conv2")
    heatmap = gcam(input_tensor, class_idx)          # numpy H×W in [0,1]
    result_img = overlay_heatmap(original_pil, heatmap)
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Computes Grad-CAM for a given model and target convolutional layer.

    Parameters
    ----------
    model       : nn.Module   – The loaded PyTorch model (eval mode).
    target_layer: str         – Dot-path to the convolutional layer, e.g.
                                "model.features.denseblock4.denselayer16.conv2"
    """

    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.model.eval()
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        # Resolve the layer by traversing the module tree
        layer = self._get_layer(model, target_layer)
        layer.register_forward_hook(self._save_activation)
        layer.register_backward_hook(self._save_gradient)

    # ── Hooks ─────────────────────────────────────────────────
    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ── Helper ────────────────────────────────────────────────
    @staticmethod
    def _get_layer(model, dotpath: str):
        parts = dotpath.split(".")
        m = model
        for p in parts:
            m = getattr(m, p)
        return m

    # ── Main ──────────────────────────────────────────────────
    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Parameters
        ----------
        input_tensor : torch.Tensor  shape (1, C, H, W), already normalised
        class_idx    : int           index of the target class

        Returns
        -------
        heatmap : np.ndarray  float32, shape (H, W) in [0, 1]
        """
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        logits = self.model(input_tensor)          # (1, num_classes)
        score  = logits[0, class_idx]

        # Backward for the selected class
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Pool gradients over spatial dims → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.astype(np.float32)


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Blend a Grad-CAM heatmap onto the original X-ray image.

    Parameters
    ----------
    original_image : PIL.Image   – The chest X-ray (any size, RGB or L).
    heatmap        : np.ndarray  – Float32 H×W array in [0, 1].
    alpha          : float       – Transparency of the heatmap overlay (0–1).
    colormap       : int         – OpenCV colormap constant.

    Returns
    -------
    PIL.Image  – Blended RGB image, same size as original_image.
    """
    orig_w, orig_h = original_image.size
    orig_rgb = np.array(original_image.convert("RGB"))

    # Resize heatmap to original image dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (orig_w, orig_h),
                                  interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap(heatmap_resized, colormap)   # BGR
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(orig_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(blended)


def get_top_predictions(
    logits: torch.Tensor,
    class_names: list[str],
    top_k: int = 5,
    threshold: float = 0.15,
) -> list[dict]:
    """
    Convert raw logits to a ranked list of predictions with probabilities.

    Parameters
    ----------
    logits      : torch.Tensor  shape (num_classes,)
    class_names : list[str]
    top_k       : int           number of top classes to return
    threshold   : float         minimum sigmoid probability to include

    Returns
    -------
    list of dicts: [{"label": str, "probability": float, "index": int}, ...]
    sorted descending by probability, filtered by threshold.
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    results = []
    for idx, prob in ranked[:top_k]:
        if prob >= threshold:
            results.append({
                "label":       class_names[idx],
                "probability": float(prob),
                "index":       idx,
            })
    return results
