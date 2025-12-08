import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNextClassifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=True, freeze_backbone=False, dropout=0.15):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.base_model = convnext_tiny(weights=weights)

        if freeze_backbone:
            for param in self.base_model.features.parameters():
                param.requires_grad = False
            for param in self.base_model.features[-1].parameters():
                param.requires_grad = True

        in_features = self.base_model.classifier[2].in_features

        self.base_model.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.base_model.features(x)
        x = x.mean(dim=(-2, -1))
        x = self.base_model.classifier(x)
        return x
