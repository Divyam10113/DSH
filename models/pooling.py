"""
Slice-level Attention Pooling Modules (Multiple Instance Learning).
Aggregates variable K slices (20 to 300) into a single fixed-size series embedding.
Learns which specific slices contain the abnormality (e.g. ACL tear, meniscus tear).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling (Ilse et al., ICML 2018 Multiple Instance Learning).
    
    a_k = softmax( w^T (tanh(V * h_k) * sigmoid(U * h_k)) )
    z = sum_k ( a_k * h_k )
    
    Provides interpretable attention weights showing which slices the network focused on.
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, K, in_features) where K is number of slices.
            mask: Optional (B, K) boolean mask where True indicates padded slices to ignore.
        Returns:
            pooled: (B, in_features) aggregated series embedding.
            attn_weights: (B, K, 1) normalized attention scores per slice.
        """
        v = self.attention_V(x)          # (B, K, hidden_dim)
        u = self.attention_U(x)          # (B, K, hidden_dim)
        gated = v * u                    # (B, K, hidden_dim)
        scores = self.attention_weights(gated)  # (B, K, 1)

        if mask is not None:
            # Mask out padding slices before softmax
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))

        attn_weights = F.softmax(scores, dim=1)  # (B, K, 1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of slice features
        pooled = torch.sum(attn_weights * x, dim=1)  # (B, in_features)
        return pooled, attn_weights


class MultiHeadSlicePooling(nn.Module):
    """
    Transformer-style Multi-Head Attention Pooling using a learnable query token.
    Allows capturing multiple distinct features (e.g. bone vs cartilage vs fluid) across slices.
    """
    def __init__(self, in_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.query = nn.Parameter(torch.randn(1, 1, in_features))
        self.mha = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(in_features)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, K, in_features)
            mask: (B, K) boolean mask for padded slices (True = ignore)
        Returns:
            pooled: (B, in_features)
            attn_weights: (B, 1, K)
        """
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, in_features)
        
        # Multihead attention
        out, attn_weights = self.mha(
            query=q,
            key=x,
            value=x,
            key_padding_mask=mask
        )
        pooled = self.norm(out.squeeze(1))  # (B, in_features)
        return pooled, attn_weights
