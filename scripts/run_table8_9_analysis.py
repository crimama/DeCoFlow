#!/usr/bin/env python3
"""
Table 8 (SVD Analysis) + Table 9 (Gradient Redistribution) Analysis Script.

Uses V48_01 checkpoint (15-class HR SOTA) to measure:
  - Table 8: SVD of weight deltas (ΔW = W_task1 - W_base) → effective rank, energy spectrum
  - Table 9: Per-patch gradient magnitude under Mean-only vs Tail-Aware loss

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/run_table8_9_analysis.py \
      --checkpoint_dir logs/V48_01_H04_highres_clean/checkpoints \
      --data_path /Volume/MVTecAD
"""

import os
import sys
import math
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from decoflow.extractors import create_feature_extractor, get_backbone_type
from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.config.ablation import AblationConfig
from decoflow.data.mvtec import MVTEC


MVTEC_CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]


def create_model_and_extractor(args, device):
    """Create model and feature extractor matching V48_01 config."""
    backbone_name = 'wide_resnet50_2'
    embed_dim = 768  # WRN50 HR embed dim

    feature_extractor = create_feature_extractor(
        backbone_name=backbone_name,
        input_shape=(3, 224, 224),  # Match V48_01 training config
        target_embed_dimension=embed_dim,
        device=device,
        patch_size=3,
        patch_stride=1,
        use_high_res=True,
    )

    ablation_config = AblationConfig(
        use_lora=True,
        use_tsa=True,
        use_acl=True,
        acl_n_layers=2,
        use_tail_aware_loss=True,
        tail_weight=0.85,
        tail_top_k_ratio=0.02,
        score_aggregation_mode='top_k',
        score_aggregation_top_k=3,
        lambda_logdet=1e-4,
        scale_context_kernel=5,
        score_smooth_sigma=0.0,
    )

    nf_model = DeCoFlowNF(
        embed_dim=embed_dim,
        coupling_layers=6,
        clamp_alpha=1.9,
        lora_rank=64,
        lora_alpha=1.0,
        device=device,
        ablation_config=ablation_config,
    )

    return feature_extractor, nf_model, ablation_config


def load_checkpoint_for_task(nf_model, checkpoint_dir, task_id, device):
    """Load checkpoint for a specific task."""
    task_dir = Path(checkpoint_dir) / f"task_{task_id}"
    model_path = task_dir / "nf_model.pth"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        nf_model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint: task_{task_id}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")


# =============================================================================
# Table 8: SVD Analysis
# =============================================================================

def run_svd_analysis(nf_model, checkpoint_dir, device):
    """
    SVD analysis of weight deltas between base (task_0) and adapted (task_1+).

    Measures:
    - Effective rank of ΔW
    - Energy captured by rank-r approximation
    """
    print("\n" + "=" * 70)
    print("TABLE 8: SVD Analysis — Weight Delta Decomposition")
    print("=" * 70)

    # Step 1: Load task_0 weights (base)
    # We need to compare base subnet weights vs full finetune delta
    # In DeCoFlow, base weights are frozen after task_0.
    # The LoRA adapters capture the task-specific deltas.

    # Approach: For each coupling layer subnet, extract:
    #   - base weights W_base (frozen after task_0)
    #   - LoRA matrices A_t, B_t for task t
    #   - Effective delta: ΔW_t = (alpha/r) * B_t @ A_t
    #   - SVD of ΔW_t to measure effective rank and energy

    # Load task_0 checkpoint to get base weights
    for tid in range(15):
        nf_model.add_task(tid)
    load_checkpoint_for_task(nf_model, checkpoint_dir, 14, device)  # load final checkpoint
    nf_model.to(device)
    nf_model.eval()

    # Collect LoRA deltas across all coupling layers for each task
    all_results = {}

    for task_id in range(1, 15):  # Skip task_0 (base)
        nf_model.set_active_task(task_id)
        deltas = []

        task_key = str(task_id)
        for name, module in nf_model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if task_key in module.lora_A and task_key in module.lora_B:
                    A = module.lora_A[task_key].data  # nn.Parameter → (r, d_in)
                    B = module.lora_B[task_key].data  # nn.Parameter → (d_out, r)
                    alpha = getattr(module, 'lora_alpha', 1.0)
                    rank = A.shape[0]
                    scaling = alpha / rank

                    # Compute effective delta
                    delta_W = scaling * (B @ A)  # (d_out, d_in)
                    deltas.append(delta_W.detach().cpu())

        if not deltas:
            continue

        # Per-module SVD, then aggregate singular value spectra
        all_singular_values = []
        for d in deltas:
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            all_singular_values.append(S.numpy())

        # Pool all singular values across modules, sort descending
        S_np = np.sort(np.concatenate(all_singular_values))[::-1].copy()

        # Compute metrics
        total_energy = (S_np ** 2).sum()
        cumulative_energy = np.cumsum(S_np ** 2) / total_energy

        # Effective rank (Shannon entropy)
        p = (S_np ** 2) / total_energy
        p = p[p > 1e-10]
        entropy = -np.sum(p * np.log(p))
        effective_rank = np.exp(entropy)

        # Energy captured at various ranks
        energy_at_16 = cumulative_energy[min(15, len(cumulative_energy) - 1)] * 100
        energy_at_64 = cumulative_energy[min(63, len(cumulative_energy) - 1)] * 100

        all_results[str(task_id)] = {
            'effective_rank': float(effective_rank),
            'total_singular_values': int(len(S_np)),
            'energy_at_rank16': float(energy_at_16),
            'energy_at_rank64': float(energy_at_64),
            'top10_singular_values': [float(x) for x in S_np[:10]],
        }

    # Aggregate across tasks
    eff_ranks = [r['effective_rank'] for r in all_results.values()]
    energies_16 = [r['energy_at_rank16'] for r in all_results.values()]
    energies_64 = [r['energy_at_rank64'] for r in all_results.values()]
    total_svs = [r['total_singular_values'] for r in all_results.values()]

    print(f"\n--- SVD Results (averaged over {len(all_results)} tasks) ---")
    print(f"  Total singular values (matrix size): {np.mean(total_svs):.0f}")
    print(f"  Effective Rank:    {np.mean(eff_ranks):.1f} ± {np.std(eff_ranks):.1f}")
    print(f"  Energy at rank 16: {np.mean(energies_16):.1f}% ± {np.std(energies_16):.1f}%")
    print(f"  Energy at rank 64: {np.mean(energies_64):.1f}% ± {np.std(energies_64):.1f}%")

    print(f"\n--- Table 8 Values ---")
    print(f"  Full (Eff. Rank {np.mean(total_svs):.0f}): 100%")
    print(f"  Rank 16: {np.mean(energies_16):.1f}%")
    print(f"  Rank 64: {np.mean(energies_64):.1f}%")

    return {
        'per_task': all_results,
        'mean_effective_rank': float(np.mean(eff_ranks)),
        'mean_total_sv': float(np.mean(total_svs)),
        'mean_energy_at_16': float(np.mean(energies_16)),
        'mean_energy_at_64': float(np.mean(energies_64)),
    }


# =============================================================================
# Table 9: Gradient Redistribution Analysis
# =============================================================================

def run_gradient_analysis(nf_model, feature_extractor, checkpoint_dir, args, device):
    """
    Measure per-patch gradient magnitude under Mean-only vs Tail-Aware loss.

    For each batch:
    1. Forward pass → get z, logdet, nll per patch
    2. Identify tail patches (top-k by NLL)
    3. Compute loss (mean-only or tail-aware)
    4. Backward → measure gradient magnitude at tail vs non-tail patches
    """
    print("\n" + "=" * 70)
    print("TABLE 9: Gradient Redistribution Analysis")
    print("=" * 70)

    # Load task_1 checkpoint (cable) — a representative non-base task
    # Use task_1 because it's the first adapted task
    for tid in range(15):
        nf_model.add_task(tid)
    load_checkpoint_for_task(nf_model, checkpoint_dir, 14, device)
    nf_model.to(device)

    # We analyze task_1 (cable) as a representative
    analysis_task = 1
    analysis_class = MVTEC_CLASSES[analysis_task]
    nf_model.set_active_task(analysis_task)

    print(f"\n  Analyzing task {analysis_task} ({analysis_class})")

    # Create dataloader
    dataset = MVTEC(
        root=args.data_path,
        class_name=analysis_class,
        train=True,
        img_size=224,
        crp_size=224,
        msk_size=256,
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    tail_ratio = 0.02
    num_batches = 30

    configs = [
        {'name': 'mean_only', 'tail_weight': 0.0},
        {'name': 'tail_aware', 'tail_weight': 0.85},
    ]

    results = {c['name']: [] for c in configs}

    feature_extractor.eval()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images = batch[0].to(device)

        with torch.no_grad():
            features = feature_extractor(images)  # (B, H, W, D)

        for config in configs:
            # Clone features and enable grad
            feat = features.clone().detach().requires_grad_(True)

            nf_model.train()
            nf_model.zero_grad()
            if feat.grad is not None:
                feat.grad = None

            # Forward (set_active_task already called above)
            z, logdet_patch = nf_model(feat)
            B, H, W, D = z.shape

            # NLL per patch
            log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            nll_patch = -(log_pz + logdet_patch)  # (B, H, W)

            flat_nll = nll_patch.reshape(B, -1)
            num_patches = H * W

            # Mean loss
            mean_loss = flat_nll.mean()

            # Tail loss
            k = max(1, int(num_patches * tail_ratio))
            top_k_nll, top_k_idx = torch.topk(flat_nll, k, dim=1)
            tail_loss = top_k_nll.mean()

            # Combined loss
            tw = config['tail_weight']
            loss = (1 - tw) * mean_loss + tw * tail_loss

            # Backward
            loss.backward()

            # Measure gradients at input features
            grad = feat.grad  # (B, H, W, D)
            grad_mag = grad.norm(dim=-1)  # (B, H, W)
            flat_grad = grad_mag.reshape(B, -1)  # (B, num_patches)

            # Compute mean grad at tail vs non-tail
            # Use first sample for tail identification
            batch_grad_tail = []
            batch_grad_nontail = []
            for b in range(B):
                tail_indices = top_k_idx[b]
                all_indices = torch.arange(num_patches, device=device)
                mask = torch.ones(num_patches, dtype=torch.bool, device=device)
                mask[tail_indices] = False
                nontail_indices = all_indices[mask]

                g_tail = flat_grad[b, tail_indices].mean().item()
                g_nontail = flat_grad[b, nontail_indices].mean().item()
                batch_grad_tail.append(g_tail)
                batch_grad_nontail.append(g_nontail)

            results[config['name']].append({
                'grad_tail': np.mean(batch_grad_tail),
                'grad_nontail': np.mean(batch_grad_nontail),
                'loss': loss.item(),
            })

            # Detach to free graph
            del z, logdet_patch, nll_patch, loss, grad
            feat.grad = None

        if (batch_idx + 1) % 5 == 0:
            print(f"    Batch {batch_idx + 1}/{num_batches}")

    # Aggregate
    print(f"\n--- Gradient Redistribution Results ---")
    print(f"{'Config':<15} {'Grad@Tail':<15} {'Grad@NonTail':<15} {'Ratio':<10}")
    print("-" * 55)

    table9 = {}
    for config_name, batch_results in results.items():
        g_tail = np.mean([r['grad_tail'] for r in batch_results])
        g_nontail = np.mean([r['grad_nontail'] for r in batch_results])
        ratio = g_tail / (g_nontail + 1e-10)
        print(f"{config_name:<15} {g_tail:<15.4f} {g_nontail:<15.4f} {ratio:<10.1f}x")
        table9[config_name] = {
            'grad_tail': float(g_tail),
            'grad_nontail': float(g_nontail),
            'ratio': float(ratio),
        }

    # Amplification
    if 'mean_only' in table9 and 'tail_aware' in table9:
        amp_tail = table9['tail_aware']['grad_tail'] / (table9['mean_only']['grad_tail'] + 1e-10)
        amp_nontail = table9['tail_aware']['grad_nontail'] / (table9['mean_only']['grad_nontail'] + 1e-10)
        amp_ratio = table9['tail_aware']['ratio'] / (table9['mean_only']['ratio'] + 1e-10)
        print(f"{'Amplification':<15} {amp_tail:<15.1f}x {amp_nontail:<15.2f}x {amp_ratio:<10.1f}x")
        table9['amplification'] = {
            'tail': float(amp_tail),
            'nontail': float(amp_nontail),
            'ratio': float(amp_ratio),
        }

    return table9


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str,
                        default='logs/V48_01_H04_highres_clean/checkpoints')
    parser.add_argument('--data_path', type=str, default='/Volume/MVTecAD')
    parser.add_argument('--output_dir', type=str, default='logs/5_Analysis/Table8_9')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    feature_extractor, nf_model, ablation_config = create_model_and_extractor(args, device)

    # Table 8: SVD Analysis
    svd_results = run_svd_analysis(nf_model, args.checkpoint_dir, device)
    with open(output_dir / 'table8_svd_results.json', 'w') as f:
        json.dump(svd_results, f, indent=2)
    print(f"\nSVD results saved to {output_dir / 'table8_svd_results.json'}")

    # Re-create model for gradient analysis (clean state)
    feature_extractor, nf_model, ablation_config = create_model_and_extractor(args, device)

    # Table 9: Gradient Analysis
    grad_results = run_gradient_analysis(nf_model, feature_extractor, args.checkpoint_dir, args, device)
    with open(output_dir / 'table9_gradient_results.json', 'w') as f:
        json.dump(grad_results, f, indent=2)
    print(f"Gradient results saved to {output_dir / 'table9_gradient_results.json'}")

    # Print final summary for LaTeX
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Copy to main.tex")
    print("=" * 70)

    print("\n--- Table 8 (SVD) ---")
    print(f"Full (Eff. Rank {svd_results['mean_effective_rank']:.0f}) & 100% & 98.47% & Full capacity")
    print(f"16 & {svd_results['mean_energy_at_16']:.1f}% & TBD & Low-rank sufficiency")
    print(f"64 & {svd_results['mean_energy_at_64']:.1f}% & 98.47% & Diminishing returns")

    print("\n--- Table 9 (Gradient) ---")
    m = grad_results['mean_only']
    t = grad_results['tail_aware']
    a = grad_results.get('amplification', {})
    print(f"Mean-only & {m['grad_tail']:.4f} & {m['grad_nontail']:.4f} & {m['ratio']:.1f}x")
    print(f"Tail-Aware & {t['grad_tail']:.4f} & {t['grad_nontail']:.4f} & {t['ratio']:.1f}x")
    print(f"Amplification & {a.get('tail', 0):.1f}x & {a.get('nontail', 0):.2f}x & {a.get('ratio', 0):.1f}x")


if __name__ == '__main__':
    main()
