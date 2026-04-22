"""
utils/inference.py
──────────────────
Single-image inference pipeline: Grad-CAM + LIME dual XAI.
"""

import os, time
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

from utils.gradcam import GradCAM, overlay_heatmap, get_top_predictions
from utils.lime_explainer import LIMEExplainer, build_dual_xai_summary


class CheXNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        densenet = models.densenet121(weights=None)
        in_features = densenet.classifier.in_features
        densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.model = densenet

    def forward(self, x):
        return self.model(x)


GRADCAM_LAYER = "model.features.denseblock4.denselayer16.conv2"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class LungDiagnosisPredictor:
    def __init__(self, model_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint["labels"]

        self.model = CheXNet(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.gradcam = GradCAM(self.model, GRADCAM_LAYER)
        self.lime    = LIMEExplainer(self.model, self.class_names, self.device, n_segments=60)
        print(f"[Predictor] Loaded → {self.device} | {len(self.class_names)} classes")

    def predict(
        self,
        image_path: str,
        top_k: int = 5,
        heatmap_alpha: float = 0.45,
        lime_samples: int = 500,
        save_gradcam_path=None,
        save_lime_path=None,
    ) -> dict:
        t0 = time.perf_counter()

        original_pil = Image.open(image_path).convert("RGB")
        tensor       = TRANSFORM(original_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor).squeeze(0)

        predictions  = get_top_predictions(logits, self.class_names, top_k=top_k)

        gradcam_path = lime_path = None
        lime_r2 = lime_num_segs = 0
        xai_summary = ""

        if predictions:
            top_idx   = predictions[0]["index"]
            top_label = predictions[0]["label"]
            top_prob  = predictions[0]["probability"]

            # Grad-CAM
            heatmap = self.gradcam(tensor.clone(), class_idx=top_idx)
            gc_overlay = overlay_heatmap(original_pil, heatmap, alpha=heatmap_alpha)
            if save_gradcam_path:
                gc_overlay.save(save_gradcam_path)
                gradcam_path = save_gradcam_path

            # LIME
            lime_result = self.lime.explain(
                pil_image=original_pil,
                class_idx=top_idx,
                num_samples=lime_samples,
                top_k_segments=10,
            )
            if save_lime_path:
                lime_result["overlay"].save(save_lime_path)
                lime_path = save_lime_path

            lime_r2       = lime_result["r2_score"]
            lime_num_segs = lime_result["num_segments"]
            xai_summary   = build_dual_xai_summary(heatmap, lime_result, top_label, top_prob)

        return {
            "predictions":   predictions,
            "top_label":     predictions[0]["label"] if predictions else "No Finding",
            "top_prob":      predictions[0]["probability"] if predictions else 0.0,
            "gradcam_path":  gradcam_path,
            "lime_path":     lime_path,
            "lime_r2":       lime_r2,
            "lime_num_segs": lime_num_segs,
            "xai_summary":   xai_summary,
            "inference_ms":  round((time.perf_counter() - t0) * 1000, 1),
        }

    def get_disease_info(self, label: str) -> dict:
        INFO = {
            "Atelectasis":        {"desc": "Partial or complete collapse of the lung.", "severity": "moderate"},
            "Cardiomegaly":       {"desc": "Abnormal enlargement of the heart.", "severity": "high"},
            "Effusion":           {"desc": "Excess fluid between pleural layers.", "severity": "moderate"},
            "Infiltration":       {"desc": "Fluid or pus filling lung tissue.", "severity": "moderate"},
            "Mass":               {"desc": "Lung lesion > 3 cm — may be tumour or benign.", "severity": "high"},
            "Nodule":             {"desc": "Small opacity ≤ 3 cm — needs follow-up.", "severity": "moderate"},
            "Pneumonia":          {"desc": "Infection inflaming the air sacs.", "severity": "high"},
            "Pneumothorax":       {"desc": "Collapsed lung from air in pleural space.", "severity": "critical"},
            "Consolidation":      {"desc": "Lung air replaced by fluid or solid material.", "severity": "moderate"},
            "Edema":              {"desc": "Excess fluid in the lungs.", "severity": "high"},
            "Emphysema":          {"desc": "Damaged air sacs — often due to smoking.", "severity": "high"},
            "Fibrosis":           {"desc": "Scarring of lung tissue reducing function.", "severity": "high"},
            "Pleural_Thickening": {"desc": "Scarring on the pleural lining.", "severity": "low"},
            "Hernia":             {"desc": "Abdominal contents protruding into chest.", "severity": "moderate"},
            "No Finding":         {"desc": "No pathological findings detected.", "severity": "none"},
        }
        return INFO.get(label, {"desc": "No additional information available.", "severity": "unknown"})
