"""
Sanity check script verifying Divyam's end-to-end knee MRI model architecture.
Tests variable slice counts, missing planes, gradient flow, and loss calculation on MPS/CPU.
"""

import sys
import torch
from models import KneeMultiSeriesModel, ABNORMALITIES
from losses import AsymmetricLoss, MultiLabelFocalLoss


def run_sanity_check():
    print("=" * 60)
    print("RSNA Knee Abnormality Detection — M3 Architecture Verification")
    print("=" * 60)

    # 1. Device selection (Apple Silicon MPS or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[1] Running on device: {device}")

    # 2. Instantiate Model
    print("\n[2] Initializing KneeMultiSeriesModel...")
    model = KneeMultiSeriesModel(
        backbone_name="resnet34",
        pretrained=False,     # False for fast offline test without downloading weights
        in_channels=1,        # Grayscale MRI slice
        embed_dim=256,
        pooling_type="gated", # Gated attention pooling (MIL)
        fusion_type="concat",
        num_targets=12,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total trainable parameters: {total_params:,}")

    # 3. Create synthetic multi-series study with variable slice counts
    # Patient 1 and Patient 2 in batch
    # Sagittal has 32 slices, Coronal has 24 slices, Axial has 38 slices!
    B, C, H, W = 2, 1, 128, 128
    series_dict = {
        "sagittal": torch.randn(B, 32, C, H, W, device=device),
        "coronal": torch.randn(B, 24, C, H, W, device=device),
        "axial": torch.randn(B, 38, C, H, W, device=device),
    }
    print("\n[3] Synthetic Input Batch:")
    for plane, t in series_dict.items():
        print(f"    {plane.capitalize():<10}: {list(t.shape)} (Variable slices: {t.shape[1]})")

    # 4. Forward Pass
    print("\n[4] Running Forward Pass...")
    logits, probs, attn_maps = model(series_dict)
    
    print(f"    Logits shape:      {list(logits.shape)} (Expected: [2, 12])")
    print(f"    Probs shape:       {list(probs.shape)}  (Expected: [2, 12])")
    print(f"    Prob values range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
    
    assert logits.shape == (B, 12), f"Logits shape mismatch: {logits.shape}"
    assert probs.shape == (B, 12), f"Probs shape mismatch: {probs.shape}"
    assert len(attn_maps) == 3, f"Expected 3 attention maps, got {len(attn_maps)}"
    
    print("    Attention map shapes:")
    for plane, a in attn_maps.items():
        print(f"      - {plane}: {list(a.shape)} (sum across slices = {a[0].sum().item():.2f})")

    # 5. Loss and Backward Pass (Gradient Flow)
    print("\n[5] Testing Loss Calculation and Gradient Flow...")
    dummy_targets = torch.tensor([
        [1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # Patient 1
        [0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1.],  # Patient 2
    ], device=device)

    loss_fn = AsymmetricLoss()
    loss = loss_fn(logits, dummy_targets)
    print(f"    Asymmetric Loss: {loss.item():.4f}")

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"    Non-zero gradient tensors: {len(grad_norms)}/{len(list(model.parameters()))}")
    assert len(grad_norms) > 0, "No gradients were computed!"

    # 6. Test Missing Plane Robustness (e.g. Axial plane missing from study)
    print("\n[6] Testing Robustness to Missing Series (Study with only Sagittal + Coronal)...")
    partial_series = {
        "sagittal": torch.randn(B, 30, C, H, W, device=device),
        "coronal": torch.randn(B, 20, C, H, W, device=device),
    }
    logits_partial, probs_partial, _ = model(partial_series)
    assert logits_partial.shape == (B, 12)
    print(f"    Output shape with missing Axial plane: {list(logits_partial.shape)} (OK!)")

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED! Divyam's architecture is 100% operational.")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_check()
