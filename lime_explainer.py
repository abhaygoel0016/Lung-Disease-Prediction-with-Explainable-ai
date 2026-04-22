"""
utils/lime_explainer.py
───────────────────────
LIME (Local Interpretable Model-agnostic Explanations) for chest X-ray
classification using DenseNet-121.

LIME works by:
  1. Segmenting the input image into superpixels (using SLIC).
  2. Creating ~N perturbed versions by randomly masking superpixels.
  3. Getting the model's prediction probabilities for each perturbed image.
  4. Fitting a weighted linear model to learn which superpixels matter most.
  5. Highlighting the top positive / negative contributing regions.

Usage
-----
    from utils.lime_explainer import LIMEExplainer
    lime = LIMEExplainer(model, class_names)
    result = lime.explain(pil_image, class_idx, num_samples=500)
    result["overlay"].save("lime_overlay.png")
"""

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as T
from skimage.segmentation import slic
from skimage.color import label2rgb
from sklearn.linear_model import Ridge


# ── ImageNet normalisation (must match training) ───────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_TO_TENSOR = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])


def _pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB → float32 numpy H×W×3 in [0, 1]."""
    return np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0


def _numpy_to_tensor(img_np: np.ndarray, device) -> torch.Tensor:
    """
    img_np : H×W×3 float32 in [0, 1]
    Returns: normalised tensor (1, 3, 224, 224) on device.
    """
    img_resized = cv2.resize(img_np, (224, 224))
    img_norm    = (img_resized - _MEAN) / _STD
    tensor      = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0).to(device)


class LIMEExplainer:
    """
    LIME explainer for a multi-label chest X-ray classifier.

    Parameters
    ----------
    model       : torch.nn.Module  – Loaded model in eval mode.
    class_names : list[str]        – Ordered list of disease labels.
    device      : torch.device     – CPU or CUDA.
    n_segments  : int              – Number of SLIC superpixels (default 60).
    compactness : float            – SLIC compactness (default 20).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        device: torch.device,
        n_segments: int = 60,
        compactness: float = 20.0,
    ):
        self.model       = model
        self.model.eval()
        self.class_names = class_names
        self.device      = device
        self.n_segments  = n_segments
        self.compactness = compactness

    # ─────────────────────────────────────────────────────────
    def _segment(self, img_np: np.ndarray) -> np.ndarray:
        """Run SLIC on the [0,1] float image; return integer segment map."""
        return slic(
            img_np,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=0,
            channel_axis=2,
        )

    @torch.no_grad()
    def _predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Run the model on a list of H×W×3 float images.
        Returns sigmoid probabilities as numpy (N, num_classes).
        """
        tensors = torch.cat(
            [_numpy_to_tensor(img, self.device) for img in images], dim=0
        )
        logits = self.model(tensors)                  # (N, C)
        probs  = torch.sigmoid(logits).cpu().numpy()  # (N, C)
        return probs

    # ─────────────────────────────────────────────────────────
    def explain(
        self,
        pil_image: Image.Image,
        class_idx: int,
        num_samples: int = 500,
        batch_size: int  = 32,
        top_k_segments: int = 10,
        positive_only: bool = False,
    ) -> dict:
        """
        Generate a LIME explanation for `class_idx`.

        Parameters
        ----------
        pil_image      : PIL.Image   – The input chest X-ray.
        class_idx      : int         – Target class index to explain.
        num_samples    : int         – Number of perturbed samples (↑ = more accurate, slower).
        batch_size     : int         – GPU batch size for inference.
        top_k_segments : int         – How many superpixels to highlight.
        positive_only  : bool        – Show only positively contributing regions.

        Returns
        -------
        dict with keys:
          overlay         : PIL.Image  – LIME mask overlaid on the X-ray.
          segments        : np.ndarray – Integer superpixel map.
          weights         : np.ndarray – LIME linear model coefficients per segment.
          top_segments    : list[int]  – Indices of most influential superpixels.
          r2_score        : float      – Fit quality of the surrogate model.
          num_segments    : int        – Total superpixels used.
        """
        img_np   = _pil_to_numpy(pil_image)
        segments = self._segment(img_np)
        n_segs   = segments.max() + 1

        # ── Perturb ────────────────────────────────────────────
        # Each row in Z is a binary mask over superpixels (1 = keep, 0 = grey)
        rng  = np.random.default_rng(seed=42)
        Z    = rng.integers(0, 2, size=(num_samples, n_segs)).astype(np.float32)
        # Always include the original (all segments ON)
        Z[0] = 1.0

        # Build perturbed images
        grey_fill = np.ones_like(img_np) * 0.5   # neutral grey replacement

        perturbed_imgs = []
        for mask_vec in Z:
            img_p = img_np.copy()
            for seg_id in range(n_segs):
                if mask_vec[seg_id] == 0:
                    img_p[segments == seg_id] = grey_fill[segments == seg_id]
            perturbed_imgs.append(img_p)

        # ── Inference in batches ───────────────────────────────
        all_probs = []
        for start in range(0, num_samples, batch_size):
            batch = perturbed_imgs[start : start + batch_size]
            all_probs.append(self._predict_batch(batch))
        all_probs = np.concatenate(all_probs, axis=0)   # (N, C)

        target_probs = all_probs[:, class_idx]           # (N,)

        # ── Distance weighting (cosine kernel) ────────────────
        original_mask = Z[0]                             # all ones
        distances     = np.sqrt(
            ((Z - original_mask) ** 2).sum(axis=1)
        )
        kernel_width = 0.25 * np.sqrt(n_segs)
        weights      = np.exp(-distances ** 2 / (2 * kernel_width ** 2))

        # ── Surrogate linear model ─────────────────────────────
        model_surrogate = Ridge(alpha=1.0, fit_intercept=True)
        model_surrogate.fit(Z, target_probs, sample_weight=weights)

        coefs    = model_surrogate.coef_                 # (n_segs,)
        r2       = model_surrogate.score(Z, target_probs, sample_weight=weights)

        # ── Select top segments ────────────────────────────────
        if positive_only:
            ranked = np.argsort(coefs)[::-1]             # highest positive first
        else:
            ranked = np.argsort(np.abs(coefs))[::-1]     # absolute magnitude

        top_segs = ranked[:top_k_segments].tolist()

        # ── Build overlay image ────────────────────────────────
        overlay_img = self._build_overlay(img_np, segments, coefs, top_segs)

        return {
            "overlay":      overlay_img,
            "segments":     segments,
            "weights":      coefs,
            "top_segments": top_segs,
            "r2_score":     float(r2),
            "num_segments": int(n_segs),
        }

    # ─────────────────────────────────────────────────────────
    def _build_overlay(
        self,
        img_np: np.ndarray,
        segments: np.ndarray,
        coefs: np.ndarray,
        top_segs: list[int],
        alpha: float = 0.55,
    ) -> Image.Image:
        """
        Draw LIME explanation on the original image.
        Green tint  → positively contributing superpixel.
        Red tint    → negatively contributing superpixel.
        Grey mask   → segment not selected for highlight.
        """
        orig_h, orig_w = img_np.shape[:2]
        overlay = (img_np * 255).astype(np.uint8).copy()

        for seg_id in top_segs:
            mask = (segments == seg_id)
            coef = coefs[seg_id]

            if coef > 0:
                color = np.array([0, 220, 100], dtype=np.uint8)    # green
            else:
                color = np.array([220, 60, 60], dtype=np.uint8)    # red

            # Intensity proportional to |coefficient|
            intensity = min(abs(coef) / (np.abs(coefs).max() + 1e-8), 1.0)
            tint = (color * intensity).astype(np.uint8)

            for c in range(3):
                channel = overlay[:, :, c].copy()
                channel[mask] = (
                    (1 - alpha) * channel[mask] + alpha * tint[c]
                ).clip(0, 255).astype(np.uint8)
                overlay[:, :, c] = channel

        # Draw superpixel boundaries for selected segments
        boundary_mask = np.zeros(segments.shape, dtype=bool)
        for seg_id in top_segs:
            seg_mask = (segments == seg_id).astype(np.uint8)
            kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated  = cv2.dilate(seg_mask, kernel)
            boundary_mask |= (dilated - seg_mask).astype(bool)

        overlay[boundary_mask] = [255, 255, 255]

        return Image.fromarray(overlay)


# ── Convenience: combine Grad-CAM + LIME into a single summary ─
def build_dual_xai_summary(
    gradcam_heatmap: np.ndarray,
    lime_result: dict,
    class_name: str,
    probability: float,
) -> str:
    """
    Generate a human-readable explanation combining Grad-CAM and LIME insights.
    """
    r2   = lime_result["r2_score"]
    nsegs = lime_result["num_segments"]
    top_k = len(lime_result["top_segments"])

    lines = [
        f"The model predicted {class_name} with {probability*100:.1f}% confidence. "
        f"Two complementary XAI techniques were applied to explain this decision:\n",

        f"• Grad-CAM (gradient-based): Backpropagates the class score through the "
        f"final DenseNet-121 convolutional block to produce a pixel-level heatmap. "
        f"Warm (red/yellow) regions in the heatmap indicate where the network's attention "
        f"was concentrated when making this prediction.\n",

        f"• LIME (model-agnostic): Divided the X-ray into {nsegs} superpixels (coherent "
        f"image regions), then tested {500} random perturbations by masking groups of "
        f"superpixels with grey. A surrogate linear model was fitted to these results "
        f"(R² = {r2:.3f}). The {top_k} most influential regions are highlighted — "
        f"green = pushes prediction higher, red = pushes prediction lower.",
    ]
    return "\n".join(lines)
