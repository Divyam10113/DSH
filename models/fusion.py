"""
Multi-Series & Cross-Plane Fusion Modules.
Fuses per-series embeddings (Sagittal, Coronal, Axial, fluid-sensitive) into a study-level vector.
Gracefully handles missing planes/series across different patient studies.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ConcatFusion(nn.Module):
    """
    Concatenates available plane embeddings and projects through an MLP with residual connection.
    Missing planes are zero-filled and tagged with a binary presence indicator.
    """
    def __init__(
        self,
        embed_dim: int,
        out_dim: int = 512,
        planes: List[str] = ("sagittal", "coronal", "axial"),
        dropout: float = 0.2
    ):
        super().__init__()
        self.planes = list(planes)
        self.embed_dim = embed_dim
        num_planes = len(self.planes)

        # Input dimension: (embed_dim * num_planes) + (num_planes presence flags)
        in_dim = (embed_dim * num_planes) + num_planes

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, series_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            series_embeddings: dict mapping plane name ('sagittal', 'coronal', 'axial') -> (B, embed_dim)
        Returns:
            fused: (B, out_dim) study-level embedding.
        """
        # Determine batch size and device from any available embedding
        first_plane = next(iter(series_embeddings.values()))
        B = first_plane.shape[0]
        device = first_plane.device

        fused_components = []
        presence_flags = []

        for p in self.planes:
            if p in series_embeddings and series_embeddings[p] is not None:
                fused_components.append(series_embeddings[p])
                presence_flags.append(torch.ones((B, 1), device=device, dtype=torch.float32))
            else:
                # Zero-fill missing plane
                fused_components.append(torch.zeros((B, self.embed_dim), device=device, dtype=torch.float32))
                presence_flags.append(torch.zeros((B, 1), device=device, dtype=torch.float32))

        # Concatenate features + presence indicators
        cat_features = torch.cat(fused_components + presence_flags, dim=-1)
        out = self.mlp(cat_features)
        return out


class CrossPlaneTransformerFusion(nn.Module):
    """
    Transformer-based cross-attention fusion.
    Each plane is treated as a token with a learnable plane-type embedding.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        planes: List[str] = ("sagittal", "coronal", "axial"),
        dropout: float = 0.1
    ):
        super().__init__()
        self.planes = list(planes)
        self.plane_to_idx = {p: i for i, p in enumerate(self.planes)}
        self.plane_embeddings = nn.Parameter(torch.randn(len(self.planes), embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.study_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, series_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            series_embeddings: dict mapping plane name -> (B, embed_dim)
        Returns:
            fused: (B, embed_dim) study-level representation.
        """
        first_plane = next(iter(series_embeddings.values()))
        B = first_plane.shape[0]
        device = first_plane.device

        tokens = [self.study_token.expand(B, -1, -1)]  # (B, 1, embed_dim)
        mask = [torch.zeros((B, 1), dtype=torch.bool, device=device)]  # False = attend

        for p in self.planes:
            p_idx = self.plane_to_idx[p]
            p_pos = self.plane_embeddings[p_idx].unsqueeze(0).unsqueeze(0).expand(B, -1, -1)

            if p in series_embeddings and series_embeddings[p] is not None:
                token = (series_embeddings[p].unsqueeze(1) + p_pos)
                tokens.append(token)
                mask.append(torch.zeros((B, 1), dtype=torch.bool, device=device))
            else:
                tokens.append(p_pos)  # dummy token
                mask.append(torch.ones((B, 1), dtype=torch.bool, device=device))  # True = ignore

        all_tokens = torch.cat(tokens, dim=1)      # (B, 1 + num_planes, embed_dim)
        src_key_padding_mask = torch.cat(mask, dim=1)  # (B, 1 + num_planes)

        encoded = self.transformer(all_tokens, src_key_padding_mask=src_key_padding_mask)
        study_emb = self.norm(encoded[:, 0, :])  # Extract pooled study token
        return study_emb
