"""
Loss Functions Optimized for Multi-Label Macro-AUC Evaluation.
Implements Multi-Label Focal Loss and Asymmetric Loss (ASL) to handle extreme class imbalance
across the 12 knee abnormalities (e.g. rare fractures vs common effusions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label Focal Loss.
    Downweights well-classified easy negative examples to focus training on hard diagnostic boundaries.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) unnormalized predictions
            targets: (B, num_classes) binary ground-truth labels in {0, 1}
        """
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * torch.pow((1 - p_t), self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification (Ridnik et al., ICCV 2021).
    Applies asymmetric focusing (gamma_neg > gamma_pos) and probability margin shifting on negatives.
    Highly effective for medical multi-label imaging where negative labels dominate.
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes)
            targets: (B, num_classes)
        """
        probs = torch.sigmoid(logits)
        
        # Positive and negative components
        pos_probs = probs
        neg_probs = 1 - probs

        # Asymmetric margin clipping for negatives
        if self.clip is not None and self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1)

        # Basic Cross Entropy
        pos_loss = targets * torch.log(pos_probs.clamp(min=self.eps))
        neg_loss = (1 - targets) * torch.log(neg_probs.clamp(min=self.eps))

        # Asymmetric weights
        pos_weight = torch.pow(1 - pos_probs, self.gamma_pos)
        neg_weight = torch.pow(1 - neg_probs, self.gamma_neg)

        loss = - (pos_weight * pos_loss + neg_weight * neg_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
