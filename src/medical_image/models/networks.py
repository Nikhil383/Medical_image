"""Classification and Segmentation models using PyTorch & MONAI with Grad-CAM support."""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from monai.networks.nets import UNet


def get_classifier(model_name: str = "resnet18", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Instantiate a classification model (e.g. ResNet18 or DenseNet121)."""
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:  # default to resnet18 matching the notebook
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


def get_monai_segmenter(spatial_dims: int = 2, in_channels: int = 3, out_channels: int = 1) -> nn.Module:
    """Instantiate a MONAI UNet architecture for lesion segmentation."""
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )
    return model


class GradCAM:
    """Grad-CAM: Gradient-weighted Class Activation Mapping for visual explanations."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        if target_layer is None:
            # Automatic heuristic for ResNet / DenseNet
            if hasattr(model, "layer4"):
                target_layer = model.layer4[-1]
            elif hasattr(model, "features"):
                target_layer = model.features[-1]
            else:
                raise ValueError("Could not automatically locate the target conv layer. Pass target_layer explicitly.")

        self.target_layer = target_layer
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate a 2D Grad-CAM heatmap normalized to [0, 1].

        input_tensor: [1, C, H, W]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Channel-wise pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)

        # Upsample to input tensor resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def cleanup(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()
