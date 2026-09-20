"""
Slice-level 2D Vision Backbones for Knee MRI.
Extracts D-dimensional feature embeddings from individual 2D MRI slices.
Supports ConvNeXt, EfficientNet, ResNet, and DINOv2 adapters.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SliceEncoder(nn.Module):
    """
    Encodes 2D MRI slices into feature vectors of dimension `embed_dim`.
    Supports single-channel (grayscale MRI) or 3-channel inputs.
    """
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        in_channels: int = 1,
        embed_dim: int = 512,
        drop_rate: float = 0.2
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if "convnext" in self.backbone_name:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            in_features = base.classifier[2].in_features
            
            # Adapt first convolution if grayscale MRI
            if in_channels != 3:
                orig_conv = base.features[0][0]
                new_conv = nn.Conv2d(
                    in_channels, orig_conv.out_channels,
                    kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride,
                    padding=orig_conv.padding
                )
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
                base.features[0][0] = new_conv
                
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(in_features),
                nn.Dropout(drop_rate),
                nn.Linear(in_features, embed_dim)
            )

        elif "efficientnet" in self.backbone_name:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            
            if in_channels != 3:
                orig_conv = base.features[0][0]
                new_conv = nn.Conv2d(
                    in_channels, orig_conv.out_channels,
                    kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride,
                    padding=orig_conv.padding,
                    bias=False
                )
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
                base.features[0][0] = new_conv
                
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(in_features),
                nn.Dropout(drop_rate),
                nn.Linear(in_features, embed_dim)
            )

        elif "resnet" in self.backbone_name:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base = models.resnet34(weights=weights)
            in_features = base.fc.in_features
            
            if in_channels != 3:
                orig_conv = base.conv1
                new_conv = nn.Conv2d(
                    in_channels, orig_conv.out_channels,
                    kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride,
                    padding=orig_conv.padding,
                    bias=False
                )
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
                base.conv1 = new_conv
                
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(in_features),
                nn.Dropout(drop_rate),
                nn.Linear(in_features, embed_dim)
            )
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B * K, C, H, W) where B is batch size and K is number of slices.
        Returns:
            (B * K, embed_dim) slice feature embeddings.
        """
        feat = self.features(x)
        pooled = self.pool(feat)
        out = self.proj(pooled)
        return out
