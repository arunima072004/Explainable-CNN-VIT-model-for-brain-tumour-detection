"""
model.py - Model loading and inference for Brain Tumor Detection & Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import resnet50, ResNet50_Weights


TUMOR_CLASSES = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
VIT_MODEL_NAME = 'vit_base_patch16_224'
VIT_IMG_SIZE = 224


class ViTBinaryClassifier(nn.Module):
    """Vision Transformer for binary tumor detection (tumor vs no_tumor)."""

    def __init__(self, model_name=VIT_MODEL_NAME, pretrained=False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.vit.num_features

        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

        self.attention_weights = None
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.attention_weights = output

        if hasattr(self.vit, 'blocks'):
            self.vit.blocks[-1].attn.register_forward_hook(hook_fn)

    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)

    def get_attention_map(self, x):
        """Extract attention map from the last ViT block."""
        self.eval()
        with torch.no_grad():
            _ = self.forward(x)

            if self.attention_weights is None:
                return None

            # Shape: [Batch, Tokens, Features] → remove CLS token, avg over features
            attn_map = self.attention_weights[:, 1:, :].mean(dim=-1)

            patch_size = 16
            grid_size = VIT_IMG_SIZE // patch_size  # 14

            try:
                attn_map = attn_map.reshape(-1, grid_size, grid_size)
            except RuntimeError:
                return None

            return attn_map


class CNNTumorClassifier(nn.Module):
    """ResNet-50 for multi-class tumor type classification."""

    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def load_models(vit_weights_path: str, cnn_weights_path: str, device: torch.device):
    """Load both models from saved weight files."""
    vit_model = ViTBinaryClassifier(pretrained=False)
    vit_model.load_state_dict(torch.load(vit_weights_path, map_location=device, weights_only=False))
    vit_model.to(device)
    vit_model.eval()

    cnn_model = CNNTumorClassifier(num_classes=3, pretrained=False)
    cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location=device, weights_only=False))
    cnn_model.to(device)
    cnn_model.eval()

    return vit_model, cnn_model
