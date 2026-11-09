"""Unified inference engine for clinical imaging analysis."""

import base64
import io
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

# Configure headless matplotlib environment before importing pyplot
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .networks import get_classifier, get_monai_segmenter, GradCAM


class MedicalImageInferenceEngine:
    """Provides combined classification, segmentation, and Grad-CAM explainability."""

    def __init__(
        self,
        classifier_path: Optional[Union[str, Path]] = None,
        segmenter_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        model_name: str = "resnet18",
        pretrained: bool = False,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize Models (pretrained=False by default to allow fully offline / sandboxed execution)
        torch.set_grad_enabled(True)
        torch.set_num_threads(1)  # Limit thread overhead on single-core / low-memory containers
        self.classifier = get_classifier(model_name=model_name, num_classes=2, pretrained=pretrained)
        self.segmenter = get_monai_segmenter(spatial_dims=2, in_channels=3, out_channels=1)

        if classifier_path and Path(classifier_path).exists():
            state_dict = torch.load(classifier_path, map_location=self.device)
            self.classifier.load_state_dict(state_dict)

        if segmenter_path and Path(segmenter_path).exists():
            state_dict = torch.load(segmenter_path, map_location=self.device)
            self.segmenter.load_state_dict(state_dict)

        self.classifier.to(self.device).eval()
        self.segmenter.to(self.device).eval()

        # Initialize Grad-CAM
        self.grad_cam = GradCAM(self.classifier)

    def preprocess_image(self, pil_image: Image.Image, target_size: int = 256) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocesses a PIL Image into standardized PyTorch tensor and clean RGB numpy array."""
        image_resized = pil_image.convert("RGB").resize((target_size, target_size), Image.Resampling.BILINEAR)
        img_np = np.asarray(image_resized, dtype=np.float32) / 255.0

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        normalized_t = (img_t - mean) / std

        return normalized_t.unsqueeze(0).to(self.device), img_np

    def analyze(self, image_input: Union[Image.Image, str, Path]) -> Dict[str, Any]:
        """Executes full diagnostic analysis pipeline on the medical scan."""
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input)
        else:
            pil_image = image_input

        tensor, raw_np = self.preprocess_image(pil_image)

        # 1. Classification
        with torch.no_grad():
            cls_logits = self.classifier(tensor)
            probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()

        p_benign = float(probs[0])
        p_malignant = float(probs[1])
        prediction = "malignant" if p_malignant >= 0.5 else "benign"
        confidence = p_malignant if prediction == "malignant" else p_benign

        # 2. Lesion Segmentation via MONAI UNet
        with torch.no_grad():
            seg_logits = self.segmenter(tensor)
            seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
            binary_mask = (seg_prob >= 0.5).astype(np.uint8)

        # 3. Explainability via Grad-CAM
        heatmap = self.grad_cam.generate(tensor, target_class=1 if prediction == "malignant" else 0)

        # Clear gradient graph and cache immediately to stay below container memory limit
        self.classifier.zero_grad(set_to_none=True)
        del tensor
        import gc
        gc.collect()

        # 4. Generate Base64 Visualization Images
        scan_b64 = self._encode_image_b64((raw_np * 255).astype(np.uint8))
        mask_b64 = self._encode_image_b64((binary_mask * 255).astype(np.uint8), mode="L")
        overlay_b64 = self._generate_overlay_b64(raw_np, binary_mask, heatmap)

        lesion_pixel_count = int(np.sum(binary_mask))
        total_pixels = int(binary_mask.size)
        lesion_area_percentage = round((lesion_pixel_count / total_pixels) * 100, 2)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {
                "benign": round(p_benign, 4),
                "malignant": round(p_malignant, 4),
            },
            "lesion_metrics": {
                "area_pixels": lesion_pixel_count,
                "area_percentage": lesion_area_percentage,
                "detected": lesion_pixel_count > 20,
            },
            "visualizations": {
                "scan_image": f"data:image/jpeg;base64,{scan_b64}",
                "mask_image": f"data:image/png;base64,{mask_b64}",
                "gradcam_overlay": f"data:image/jpeg;base64,{overlay_b64}",
            },
            "clinical_notice": "For research and educational assistance only. Not approved for primary diagnosis without clinician validation.",
        }

    def _encode_image_b64(self, arr: np.ndarray, mode: str = "RGB") -> str:
        buffered = io.BytesIO()
        img = Image.fromarray(arr, mode=mode)
        img.save(buffered, format="PNG" if mode == "L" else "JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _generate_overlay_b64(self, raw_rgb: np.ndarray, mask: np.ndarray, heatmap: np.ndarray) -> str:
        # Fast Pillow/NumPy blending to avoid matplotlib heavy memory allocation & font cache stalls on container platforms
        try:
            h, w = raw_rgb.shape[:2]
            # Normalize heatmap to [0, 255]
            hm_norm = np.clip(heatmap, 0, 1)
            # Apply color mapping (jet-like: blue -> cyan -> yellow -> red)
            # Or use matplotlib colormap array directly without creating a Figure
            cmap = matplotlib.colormaps.get_cmap("jet")
            colored_heatmap = (cmap(hm_norm)[:, :, :3] * 255).astype(np.uint8)

            raw_uint8 = (np.clip(raw_rgb, 0, 1) * 255).astype(np.uint8)
            blended = (0.55 * raw_uint8 + 0.45 * colored_heatmap).astype(np.uint8)

            # Draw green outline on lesion mask boundaries
            if np.any(mask):
                # Simple boundary detection using difference with shifted masks
                m = mask.astype(bool)
                # Dilate/erode or simple neighbor difference
                pad = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
                edge = m & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
                # Dilate edge slightly for visibility (2px boundary)
                edge_pad = np.pad(edge, ((1, 1), (1, 1)), mode="constant", constant_values=False)
                thick_edge = edge | edge_pad[:-2, 1:-1] | edge_pad[2:, 1:-1] | edge_pad[1:-1, :-2] | edge_pad[1:-1, 2:]
                blended[thick_edge] = [0, 255, 0]  # Vibrant lime boundary

            out_img = Image.fromarray(blended, mode="RGB")
            buffered = io.BytesIO()
            out_img.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception:
            # Fallback to pure raw RGB if overlay fails
            raw_uint8 = (np.clip(raw_rgb, 0, 1) * 255).astype(np.uint8)
            out_img = Image.fromarray(raw_uint8, mode="RGB")
            buffered = io.BytesIO()
            out_img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
