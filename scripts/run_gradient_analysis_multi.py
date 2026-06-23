#!/usr/bin/env python3
"""
Multi-Task Gradient Redistribution Analysis (Table 9) — Definitive Version.

Runs gradient analysis on 4 diverse tasks (cable, grid, metal_nut, transistor)
with 30 batches each, then reports mean +/- std for all gradient ratios.

Uses EXACT V48_01 config: img_size=224, use_high_res=True, embed_dim=768,
NCL6+ACB2, rank64, tail_weight=0.85, tail_top_k_ratio=0.02.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/run_gradient_analysis_multi.py \
      --checkpoint_dir logs/V48_01_H04_highres_clean/checkpoints \
      --data_path /Volume/MVTecAD \
      --output_dir logs/5_Analysis/Table8_9_rerun
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

# Tasks to analyze: cable(1), grid(4), metal_nut(7), transistor(12)
ANALYSIS_TASKS = [1, 4, 7, 12]


def create_model_and_extractor(device):
    """Create model and feature extractor matching EXACT V48_01 config."""
    backbone_name = 'wide_resnet50_2'
    embed_dim = 768  # WRN50 HR embed dim
    img_size = 224   # EXACT match to training config

    feature_extractor = create_feature_extractor(
        backbone_name=backbone_name,
        input_shape=(3, img_size, img_size),  # Match training exactly
        target_embed_dimension=embed_dim,
        device=device,
        patch_size=3,
        patch_stride=1,
        use_high_res=True,
    )

    ablation_config = AblationConfig(
        use_lora=True,
        use_tsa=True,
        use_acb=True,
        acb_n_blocks=2,
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


def load_checkpoint(nf_model, checkpoint_dir, device):
    """Load final checkpoint (task_14) with all tasks registered."""
    for tid in range(15):
        nf_model.add_task(tid)

    task_dir = Path(checkpoint_dir) / "task_14"
    model_path = task_dir / "nf_model.pth"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        nf_model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint: task_14 (final)")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    nf_model.to(device)
    return nf_model


def analyze_single_task(nf_model, feature_extractor, task_id, data_path, device,
                        num_batches=30, batch_size=16):
    """
    Run gradient analysis for a single task.

    Returns dict with mean_only and tail_aware gradient stats.
    """
    class_name = MVTEC_CLASSES[task_id]
    img_size = 224  # Match V48_01 training config exactly

    nf_model.set_active_task(task_id)
    print(f"\n  Task {task_id} ({class_name})")

    # Create dataloader — match training config: img_size=224, crp_size=224
    dataset = MVTEC(
        root=data_path,
        class_name=class_name,
        train=True,
        img_size=img_size,
        crp_size=img_size,
        msk_size=256,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    tail_ratio = 0.02

    configs = [
        {'name': 'mean_only', 'tail_weight': 0.0},
        {'name': 'tail_aware', 'tail_weight': 0.85},
    ]

    results = {c['name']: [] for c in configs}

    feature_extractor.eval()

    batches_processed = 0
    for batch_idx, batch in enumerate(dataloader):
        if batches_processed >= num_batches:
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

            # Forward
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

            # Compute mean grad at tail vs non-tail per sample
            batch_grad_tail = []
            batch_grad_nontail = []
            for b in range(B):
                tail_indices = top_k_idx[b]
                mask = torch.ones(num_patches, dtype=torch.bool, device=device)
                mask[tail_indices] = False
                nontail_indices = torch.arange(num_patches, device=device)[mask]

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

        batches_processed += 1
        if (batches_processed) % 10 == 0:
            print(f"    Batch {batches_processed}/{num_batches}")

    # Aggregate per-task
    task_results = {}
    for config_name, batch_results in results.items():
        g_tail = np.mean([r['grad_tail'] for r in batch_results])
        g_nontail = np.mean([r['grad_nontail'] for r in batch_results])
        ratio = g_tail / (g_nontail + 1e-10)
        task_results[config_name] = {
            'grad_tail': float(g_tail),
            'grad_nontail': float(g_nontail),
            'ratio': float(ratio),
            'n_batches': len(batch_results),
        }

    # Print per-task summary
    m = task_results['mean_only']
    t = task_results['tail_aware']
    print(f"    mean_only:  tail={m['grad_tail']:.6f}  nontail={m['grad_nontail']:.6f}  ratio={m['ratio']:.2f}x")
    print(f"    tail_aware: tail={t['grad_tail']:.6f}  nontail={t['grad_nontail']:.6f}  ratio={t['ratio']:.2f}x")

    return task_results


def main():
    parser = argparse.ArgumentParser(description="Multi-task Gradient Redistribution Analysis (Table 9)")
    parser.add_argument('--checkpoint_dir', type=str,
                        default='logs/V48_01_H04_highres_clean/checkpoints')
    parser.add_argument('--data_path', type=str, default='/Volume/MVTecAD')
    parser.add_argument('--output_dir', type=str, default='logs/5_Analysis/Table8_9_rerun')
    parser.add_argument('--num_batches', type=int, default=30,
                        help='Number of batches per task (default: 30)')
    parser.add_argument('--tasks', nargs='+', type=int, default=ANALYSIS_TASKS,
                        help='Task IDs to analyze (default: 1 4 7 12)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Tasks to analyze: {args.tasks} ({[MVTEC_CLASSES[t] for t in args.tasks]})")
    print(f"Batches per task: {args.num_batches}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    feature_extractor, nf_model, ablation_config = create_model_and_extractor(device)

    # Load checkpoint
    nf_model = load_checkpoint(nf_model, args.checkpoint_dir, device)

    # =========================================================================
    # Run gradient analysis for each task
    # =========================================================================
    print("\n" + "=" * 70)
    print("TABLE 9: Multi-Task Gradient Redistribution Analysis")
    print("=" * 70)

    all_task_results = {}
    for task_id in args.tasks:
        task_results = analyze_single_task(
            nf_model, feature_extractor, task_id, args.data_path, device,
            num_batches=args.num_batches,
        )
        all_task_results[str(task_id)] = task_results

    # =========================================================================
    # Aggregate across all tasks
    # =========================================================================
    print("\n" + "=" * 70)
    print("CROSS-TASK AGGREGATION")
    print("=" * 70)

    # Collect per-task values
    mean_only_tails = [all_task_results[str(t)]['mean_only']['grad_tail'] for t in args.tasks]
    mean_only_nontails = [all_task_results[str(t)]['mean_only']['grad_nontail'] for t in args.tasks]
    mean_only_ratios = [all_task_results[str(t)]['mean_only']['ratio'] for t in args.tasks]

    tail_aware_tails = [all_task_results[str(t)]['tail_aware']['grad_tail'] for t in args.tasks]
    tail_aware_nontails = [all_task_results[str(t)]['tail_aware']['grad_nontail'] for t in args.tasks]
    tail_aware_ratios = [all_task_results[str(t)]['tail_aware']['ratio'] for t in args.tasks]

    # Per-task amplification
    amp_tails = [all_task_results[str(t)]['tail_aware']['grad_tail'] /
                 (all_task_results[str(t)]['mean_only']['grad_tail'] + 1e-10)
                 for t in args.tasks]
    amp_nontails = [all_task_results[str(t)]['tail_aware']['grad_nontail'] /
                    (all_task_results[str(t)]['mean_only']['grad_nontail'] + 1e-10)
                    for t in args.tasks]
    amp_ratios = [all_task_results[str(t)]['tail_aware']['ratio'] /
                  (all_task_results[str(t)]['mean_only']['ratio'] + 1e-10)
                  for t in args.tasks]

    # Print per-task table
    print(f"\n{'Task':<15} {'Config':<12} {'Grad@Tail':<14} {'Grad@NonTail':<14} {'Ratio':<10}")
    print("-" * 65)
    for task_id in args.tasks:
        class_name = MVTEC_CLASSES[task_id]
        t_res = all_task_results[str(task_id)]
        m = t_res['mean_only']
        t = t_res['tail_aware']
        print(f"{class_name:<15} {'mean_only':<12} {m['grad_tail']:<14.6f} {m['grad_nontail']:<14.6f} {m['ratio']:<10.2f}x")
        print(f"{'':<15} {'tail_aware':<12} {t['grad_tail']:<14.6f} {t['grad_nontail']:<14.6f} {t['ratio']:<10.2f}x")

    # Compute aggregated statistics
    agg = {
        'mean_only': {
            'grad_tail': float(np.mean(mean_only_tails)),
            'grad_tail_std': float(np.std(mean_only_tails)),
            'grad_nontail': float(np.mean(mean_only_nontails)),
            'grad_nontail_std': float(np.std(mean_only_nontails)),
            'ratio': float(np.mean(mean_only_ratios)),
            'ratio_std': float(np.std(mean_only_ratios)),
        },
        'tail_aware': {
            'grad_tail': float(np.mean(tail_aware_tails)),
            'grad_tail_std': float(np.std(tail_aware_tails)),
            'grad_nontail': float(np.mean(tail_aware_nontails)),
            'grad_nontail_std': float(np.std(tail_aware_nontails)),
            'ratio': float(np.mean(tail_aware_ratios)),
            'ratio_std': float(np.std(tail_aware_ratios)),
        },
        'amplification': {
            'tail': float(np.mean(amp_tails)),
            'tail_std': float(np.std(amp_tails)),
            'nontail': float(np.mean(amp_nontails)),
            'nontail_std': float(np.std(amp_nontails)),
            'ratio': float(np.mean(amp_ratios)),
            'ratio_std': float(np.std(amp_ratios)),
        },
    }

    # =========================================================================
    # Print definitive values
    # =========================================================================
    print("\n" + "=" * 70)
    print("=== DEFINITIVE Table 9 VALUES ===")
    print("=" * 70)

    m = agg['mean_only']
    t = agg['tail_aware']
    a = agg['amplification']

    print(f"Mean-only:  grad_tail={m['grad_tail']:.6f}  grad_nontail={m['grad_nontail']:.6f}  ratio={m['ratio']:.2f}x")
    print(f"            (std: tail={m['grad_tail_std']:.6f}  nontail={m['grad_nontail_std']:.6f}  ratio={m['ratio_std']:.2f})")
    print(f"Tail-Aware: grad_tail={t['grad_tail']:.6f}  grad_nontail={t['grad_nontail']:.6f}  ratio={t['ratio']:.1f}x")
    print(f"            (std: tail={t['grad_tail_std']:.6f}  nontail={t['grad_nontail_std']:.6f}  ratio={t['ratio_std']:.1f})")
    print(f"Amplification: tail={a['tail']:.1f}x  nontail={a['nontail']:.2f}x  ratio={a['ratio']:.1f}x")
    print(f"               (std: tail={a['tail_std']:.1f}  nontail={a['nontail_std']:.2f}  ratio={a['ratio_std']:.1f})")

    # Print LaTeX-ready values
    print("\n--- LaTeX Table 9 Values ---")
    print(f"Mean-only & {m['grad_tail']:.6f} & {m['grad_nontail']:.6f} & {m['ratio']:.2f}$\\times$ \\\\")
    print(f"Tail-Aware & {t['grad_tail']:.6f} & {t['grad_nontail']:.6f} & {t['ratio']:.1f}$\\times$ \\\\")
    print(f"Amplification & {a['tail']:.1f}$\\times$ & {a['nontail']:.2f}$\\times$ & {a['ratio']:.1f}$\\times$ \\\\")

    # =========================================================================
    # Save results
    # =========================================================================
    full_results = {
        'config': {
            'checkpoint_dir': str(args.checkpoint_dir),
            'tasks': args.tasks,
            'task_names': [MVTEC_CLASSES[t] for t in args.tasks],
            'num_batches_per_task': args.num_batches,
            'batch_size': 16,
            'img_size': 224,
            'tail_ratio': 0.02,
            'tail_weight': 0.85,
        },
        'per_task': all_task_results,
        'aggregated': agg,
    }

    output_path = output_dir / 'table9_gradient_multi_task.json'
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
