#!/usr/bin/env python3
"""
Gradient Redistribution Analysis for Table 9 (tab:gradient_redistribution).

Measures per-patch gradient magnitude under mean-only vs tail-aware loss
using V48_01 checkpoint (Task 1, i.e. cable class).

Output: Grad at Tail, Grad at Non-Tail, Ratio for both loss configurations.
"""
import sys, os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / "logs" / "V48_01_H04_highres_clean" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "5_Analysis" / "Gradient_HR"
DATA_PATH = "/Volume/MVTecAD"


def setup_model_and_data(device, task_id=1):
    """Load model from checkpoint and prepare data for gradient analysis."""
    from decoflow.extractors import create_feature_extractor
    from decoflow.models.decoflow_nf import DeCoFlowNF
    from decoflow.config.ablation import AblationConfig
    from decoflow.data.mvtec import MVTEC

    # Feature extractor
    feature_extractor = create_feature_extractor(
        backbone_name='wide_resnet50_2',
        input_shape=(3, 224, 224),
        target_embed_dimension=768,
        device=device,
        patch_size=3, patch_stride=1, use_high_res=True,
    )
    for p in feature_extractor.parameters():
        p.requires_grad = False
    feature_extractor.eval()

    # Model
    ablation = AblationConfig(
        use_lora=True, use_acl=True, acl_n_layers=2,
        use_tsa=True,
        scale_context_kernel=5,
    )
    nf_model = DeCoFlowNF(
        embed_dim=768, coupling_layers=6, clamp_alpha=1.9,
        lora_rank=64, lora_alpha=1.0, device=device,
        ablation_config=ablation,
    )

    # Add tasks and load checkpoint
    task_classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                    'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    for tid in range(task_id + 1):
        nf_model.add_task(tid)

    ckpt = torch.load(
        CHECKPOINT_DIR / f"task_{task_id}" / "nf_model.pth",
        map_location=device, weights_only=False
    )
    nf_model.load_state_dict(ckpt, strict=False)
    nf_model.set_active_task(task_id)
    nf_model.to(device)

    # Data: use Task 1 (cable) training data
    class_name = task_classes[task_id]
    dataset = MVTEC(root=DATA_PATH, class_name=class_name, train=True,
                    img_size=224, crp_size=224, msk_size=256)
    loader = DataLoader(dataset, batch_size=16, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=True)

    return feature_extractor, nf_model, loader, class_name


def compute_patch_gradients(feature_extractor, nf_model, loader, device,
                            tail_weight=0.0, tail_ratio=0.02, num_batches=20):
    """
    Compute per-patch gradient magnitudes under given loss configuration.

    Args:
        tail_weight: 0.0 = mean-only, 0.85 = tail-aware (SOTA)
    """
    nf_model.train()

    all_tail_grads = []
    all_nontail_grads = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        images = batch[0].to(device)

        with torch.no_grad():
            features = feature_extractor(images)

        # Zero grads
        nf_model.zero_grad()

        # Forward
        z, logdet_patch = nf_model(features)

        # Per-patch NLL
        log_pz = -0.5 * torch.sum(z ** 2, dim=-1)  # (B, H, W)
        nll_patch = -(log_pz + logdet_patch)  # (B, H, W)

        B, H, W = nll_patch.shape

        # Identify tail patches (top tail_ratio by NLL)
        nll_flat = nll_patch.reshape(B, -1)  # (B, H*W)
        k = max(1, int(H * W * tail_ratio))
        _, top_indices = nll_flat.topk(k, dim=-1)

        tail_mask = torch.zeros_like(nll_flat, dtype=torch.bool)
        tail_mask.scatter_(1, top_indices, True)
        tail_mask = tail_mask.reshape(B, H, W)

        # Compute loss based on configuration
        if tail_weight > 0:
            nll_flat_all = nll_patch.reshape(B, -1)
            mean_loss = nll_flat_all.mean(dim=-1)
            tail_values = nll_patch[tail_mask].reshape(B, k) if k > 0 else mean_loss
            tail_loss = tail_values.mean(dim=-1)
            loss = (1 - tail_weight) * mean_loss + tail_weight * tail_loss
        else:
            loss = nll_patch.mean(dim=(1, 2))

        loss = loss.mean()
        loss.backward()

        # Collect per-patch gradient magnitudes from LoRA layers
        grad_magnitudes = []
        for name, param in nf_model.named_parameters():
            if 'lora_' in name and param.grad is not None:
                grad_magnitudes.append(param.grad.norm().item())

        total_grad = sum(grad_magnitudes) / max(len(grad_magnitudes), 1)

        # To get per-patch gradient, we use the z gradient
        if z.grad is not None:
            z_grad = z.grad  # (B, H, W, D)
            patch_grad_mag = z_grad.norm(dim=-1)  # (B, H, W)
        else:
            # Re-compute with z requiring grad
            nf_model.zero_grad()
            z2, logdet2 = nf_model(features)
            z2.retain_grad()
            log_pz2 = -0.5 * torch.sum(z2 ** 2, dim=-1)
            nll2 = -(log_pz2 + logdet2)

            nll_flat2 = nll2.reshape(B, -1)
            _, top_idx2 = nll_flat2.topk(k, dim=-1)
            tail_mask2 = torch.zeros_like(nll_flat2, dtype=torch.bool)
            tail_mask2.scatter_(1, top_idx2, True)
            tail_mask2 = tail_mask2.reshape(B, H, W)

            if tail_weight > 0:
                mean_l = nll2.reshape(B, -1).mean(dim=-1)
                tail_l = nll2[tail_mask2].reshape(B, k).mean(dim=-1)
                loss2 = ((1 - tail_weight) * mean_l + tail_weight * tail_l).mean()
            else:
                loss2 = nll2.mean()

            loss2.backward()
            patch_grad_mag = z2.grad.norm(dim=-1)  # (B, H, W)

        # Separate tail vs non-tail
        tail_grad = patch_grad_mag[tail_mask].mean().item()
        nontail_grad = patch_grad_mag[~tail_mask].mean().item()

        all_tail_grads.append(tail_grad)
        all_nontail_grads.append(nontail_grad)

    return {
        'grad_at_tail': float(np.mean(all_tail_grads)),
        'grad_at_nontail': float(np.mean(all_nontail_grads)),
        'ratio': float(np.mean(all_tail_grads) / max(np.mean(all_nontail_grads), 1e-12)),
        'tail_grads_per_batch': all_tail_grads,
        'nontail_grads_per_batch': all_nontail_grads,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Gradient Redistribution Analysis for Table 9")
    print(f"Device: {device}")
    print("=" * 60)

    # Setup
    print("\n[1/3] Loading model and data...")
    extractor, model, loader, class_name = setup_model_and_data(device, task_id=1)
    print(f"  Analyzing on class: {class_name}")

    # Mean-only
    print("\n[2/3] Computing gradients with Mean-only loss...")
    mean_results = compute_patch_gradients(
        extractor, model, loader, device,
        tail_weight=0.0, tail_ratio=0.02, num_batches=20
    )

    # Tail-Aware
    print("[3/3] Computing gradients with Tail-Aware loss (tw=0.85)...")
    tail_results = compute_patch_gradients(
        extractor, model, loader, device,
        tail_weight=0.85, tail_ratio=0.02, num_batches=20
    )

    # Summary
    print("\n" + "=" * 60)
    print("=== RESULTS (for Table 9) ===")
    print("=" * 60)
    print(f"{'Configuration':<20} {'Grad@Tail':>12} {'Grad@NonTail':>14} {'Ratio':>8}")
    print("-" * 56)
    print(f"{'Mean-only':<20} {mean_results['grad_at_tail']:>12.4f} "
          f"{mean_results['grad_at_nontail']:>14.4f} {mean_results['ratio']:>7.1f}x")
    print(f"{'Tail-Aware':<20} {tail_results['grad_at_tail']:>12.4f} "
          f"{tail_results['grad_at_nontail']:>14.4f} {tail_results['ratio']:>7.1f}x")

    amplification_tail = tail_results['grad_at_tail'] / max(mean_results['grad_at_tail'], 1e-12)
    amplification_nontail = tail_results['grad_at_nontail'] / max(mean_results['grad_at_nontail'], 1e-12)
    amplification_ratio = tail_results['ratio'] / max(mean_results['ratio'], 1e-12)

    print(f"{'Amplification':<20} {amplification_tail:>11.1f}x "
          f"{amplification_nontail:>13.2f}x {amplification_ratio:>7.1f}x")

    # Save
    output = {
        'mean_only': mean_results,
        'tail_aware': tail_results,
        'amplification': {
            'tail': amplification_tail,
            'nontail': amplification_nontail,
            'ratio': amplification_ratio,
        },
        'config': {
            'task_id': 1,
            'class_name': class_name,
            'tail_weight': 0.85,
            'tail_ratio': 0.02,
            'num_batches': 20,
        }
    }
    with open(OUTPUT_DIR / 'gradient_analysis_hr.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR / 'gradient_analysis_hr.json'}")


if __name__ == '__main__':
    main()
