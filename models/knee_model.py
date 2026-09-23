"""
End-to-End Knee MRI Abnormality Detection Model.
Combines 2D slice encoder, slice-level attention pooling, cross-plane fusion, and 12 multi-label heads.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .backbones import SliceEncoder
from .pooling import GatedAttentionPooling, MultiHeadSlicePooling
from .fusion import ConcatFusion, CrossPlaneTransformerFusion


# Must match the Kaggle train.csv / submission.csv column names and order exactly
ABNORMALITIES = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]


class KneeMultiSeriesModel(nn.Module):
    """
    Multi-Series, Multi-Label Knee MRI Classifier.
    
    Accepts arbitrary patient studies with variable slice counts across Sagittal, Coronal, and Axial planes.
    Outputs:
        1. Logits of shape (B, 12)
        2. Probabilities of shape (B, 12) via Sigmoid
        3. Slice attention maps {plane: (B, K_plane)} for radiological interpretability
    """
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        in_channels: int = 1,
        embed_dim: int = 512,
        pooling_type: str = "gated",  # 'gated' or 'mha'
        fusion_type: str = "concat",  # 'concat' or 'transformer'
        num_targets: int = 12,
        planes: List[str] = ("sagittal", "coronal", "axial"),
        dropout: float = 0.2
    ):
        super().__init__()
        self.planes = list(planes)
        self.num_targets = num_targets
        self.embed_dim = embed_dim

        # 1. 2D Slice Encoder (shared across all slices & series)
        self.slice_encoder = SliceEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            embed_dim=embed_dim,
            drop_rate=dropout
        )

        # 2. Slice-Level Attention Pooling per plane
        self.pooling_layers = nn.ModuleDict()
        for p in self.planes:
            if pooling_type == "gated":
                self.pooling_layers[p] = GatedAttentionPooling(
                    in_features=embed_dim,
                    hidden_dim=embed_dim // 4,
                    dropout=dropout
                )
            elif pooling_type == "mha":
                self.pooling_layers[p] = MultiHeadSlicePooling(
                    in_features=embed_dim,
                    num_heads=4,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown pooling type: {pooling_type}")

        # 3. Cross-Plane Fusion
        fused_dim = embed_dim
        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                embed_dim=embed_dim,
                out_dim=fused_dim,
                planes=self.planes,
                dropout=dropout
            )
        elif fusion_type == "transformer":
            self.fusion = CrossPlaneTransformerFusion(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                planes=self.planes,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # 4. Multi-Label Classification Heads (12 independent binary classifiers)
        # Each head specializes in its specific abnormality
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fused_dim, fused_dim // 2),
                nn.LayerNorm(fused_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim // 2, 1)
            )
            for _ in range(num_targets)
        ])

    def forward(
        self,
        series_dict: Dict[str, torch.Tensor],
        masks_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            series_dict: Dictionary mapping plane name -> Tensor of shape (B, K, C, H, W)
                         K may vary between planes (e.g. Sagittal has 36 slices, Coronal has 28).
            masks_dict: Optional dictionary mapping plane name -> boolean mask of shape (B, K)
        Returns:
            logits: (B, 12) unnormalized prediction scores
            probs:  (B, 12) calibrated probabilities in [0, 1]
            attn_maps: Dict[plane, (B, K)] attention weights indicating key diagnostic slices
        """
        series_embeddings = {}
        attn_maps = {}

        for plane_name, series_tensor in series_dict.items():
            if series_tensor is None or plane_name not in self.pooling_layers:
                continue

            B, K, C, H, W = series_tensor.shape
            
            # Reshape all slices across batch to pass through 2D CNN in parallel
            slices_flat = series_tensor.view(B * K, C, H, W)
            slice_features = self.slice_encoder(slices_flat)  # (B * K, embed_dim)
            slice_features = slice_features.view(B, K, self.embed_dim)  # (B, K, embed_dim)

            # Attention pool over K slices
            mask = masks_dict.get(plane_name) if masks_dict is not None else None
            pooled, attn = self.pooling_layers[plane_name](slice_features, mask=mask)
            
            series_embeddings[plane_name] = pooled
            attn_maps[plane_name] = attn.squeeze(-1) if attn.dim() == 3 else attn

        # Fuse across available planes
        study_embedding = self.fusion(series_embeddings)  # (B, fused_dim)

        # 12 Independent heads
        logits_list = [head(study_embedding) for head in self.heads]  # 12 x (B, 1)
        logits = torch.cat(logits_list, dim=-1)  # (B, 12)
        probs = torch.sigmoid(logits)           # (B, 12)

        return logits, probs, attn_maps
