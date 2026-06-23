#!/usr/bin/env python
"""
Efficiency Profiling and Visualization for DeCoFlow Paper.

Three tasks:
  A) Parse training log to extract per-task training time.
  B) Measure peak GPU memory for feature extractor, Task 0, and Task N>0 training.
  C) Generate two half-width figures:
     1. Training time per task (bar chart)
     2. Memory usage breakdown (bar chart)

Usage:
    # Full run (parse log + memory profiling + visualization)
    CUDA_VISIBLE_DEVICES=0 python scripts/analyze_efficiency.py

    # Log parsing + visualization only (no GPU needed)
    python scripts/analyze_efficiency.py --skip_memory

    # Memory profiling only
    CUDA_VISIBLE_DEVICES=0 python scripts/analyze_efficiency.py --memory_only
"""

import sys
import os
import re
import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Volume/DeCoFlow')


# ============================================================================
# Task A: Training Time Extraction from Log
# ============================================================================

def parse_training_times(log_path: str) -> dict:
    """
    Parse the training log to extract per-task training time.

    Timestamps follow the format:
        2026-02-10 22:10:29 | INFO | ... Training Task N: ['class_name']

    The time from "Training Task N" to "Training Task N+1" gives the total
    training+eval time for task N. For the last task, we use the last
    timestamp in the log.

    Returns:
        dict with keys: 'task_times_minutes', 'task_classes', 'task_timestamps'
    """
    timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    task_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| INFO \| .*Training Task (\d+): \['([^']+)'\]"
    )

    task_timestamps = []  # (task_id, datetime, class_name)
    last_timestamp = None

    with open(log_path, 'r') as f:
        for line in f:
            # Track the last timestamp in the file
            ts_match = timestamp_pattern.match(line.strip())
            if ts_match:
                last_timestamp = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')

            # Find task start lines
            task_match = task_pattern.search(line)
            if task_match:
                ts = datetime.strptime(task_match.group(1), '%Y-%m-%d %H:%M:%S')
                task_id = int(task_match.group(2))
                class_name = task_match.group(3)
                task_timestamps.append((task_id, ts, class_name))

    if not task_timestamps:
        raise ValueError(f"No 'Training Task' entries found in {log_path}")

    # Compute per-task durations
    task_times_minutes = []
    task_classes = []

    for i, (task_id, ts, cls) in enumerate(task_timestamps):
        task_classes.append(cls)
        if i + 1 < len(task_timestamps):
            # Duration = next task start - this task start
            duration = task_timestamps[i + 1][1] - ts
        else:
            # Last task: use last timestamp in log
            duration = last_timestamp - ts
        task_times_minutes.append(duration.total_seconds() / 60.0)

    return {
        'task_times_minutes': task_times_minutes,
        'task_classes': task_classes,
        'task_timestamps': [(tid, ts.isoformat(), cls) for tid, ts, cls in task_timestamps],
        'total_time_minutes': sum(task_times_minutes),
    }


# ============================================================================
# Task B: GPU Memory Profiling
# ============================================================================

def profile_memory(data_path: str = '/Volume/MVTecAD') -> dict:
    """
    Measure peak GPU memory for each component of DeCoFlow.

    Uses V48_01 configuration:
        backbone: wide_resnet50_2, embed_dim: 768, NCL: 6, ACL: 2,
        rank: 64, batch_size: 16, use_high_res: True, img_size: 224

    Returns:
        dict with memory measurements in GB.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from decoflow.extractors import create_feature_extractor
    from decoflow.models.decoflow_nf import DeCoFlowNF
    from decoflow.models.position_embedding import PositionalEmbeddingGenerator
    from decoflow.config.ablation import AblationConfig
    from decoflow.data import create_task_dataset
    from decoflow.utils.config import get_config
    from decoflow.utils.helpers import init_seeds, setting_lr_parameters

    device = torch.device('cuda')
    init_seeds(0)

    results = {}

    # --- V48_01 configuration ---
    backbone_name = 'wide_resnet50_2'
    embed_dim = 768
    num_coupling_layers = 6
    acl_n_layers = 2
    lora_rank = 64
    batch_size = 16
    img_size = 224
    msk_size = 256
    lr = 3e-4

    ablation_config = AblationConfig(
        use_lora=True,
        use_router=True,
        use_task_adapter=True,
        use_pos_embedding=True,
        use_tsa=True,
        use_acl=True,
        acl_n_layers=acl_n_layers,
        use_spatial_context=True,
        use_scale_context=True,
        scale_context_kernel=5,
        lora_rank=lora_rank,
        lambda_logdet=1e-4,
        use_tail_aware_loss=True,
        tail_weight=0.7,
        tail_top_k_ratio=0.02,
        score_aggregation_mode='top_k',
        score_aggregation_top_k=3,
    )

    # ---------------------------------------------------------------
    # 1. Feature Extractor memory (frozen)
    # ---------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    feature_extractor = create_feature_extractor(
        backbone_name=backbone_name,
        input_shape=(3, img_size, img_size),
        target_embed_dimension=embed_dim,
        device='cuda',
        patch_size=3,
        patch_stride=1,
        use_high_res=True,
    )
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = False

    torch.cuda.synchronize()
    mem_extractor = torch.cuda.memory_allocated() - mem_before
    results['extractor_gb'] = mem_extractor / (1024 ** 3)
    print(f"[Memory] Feature Extractor (frozen): {results['extractor_gb']:.3f} GB")

    # ---------------------------------------------------------------
    # 2. NF model + Task 0 training step (peak)
    # ---------------------------------------------------------------
    nf_model = DeCoFlowNF(
        embed_dim=embed_dim,
        coupling_layers=num_coupling_layers,
        clamp_alpha=1.9,
        lora_rank=lora_rank,
        lora_alpha=1.0,
        device='cuda',
        ablation_config=ablation_config,
    ).to(device)

    pos_embed_gen = PositionalEmbeddingGenerator(device=device)

    # Add Task 0
    nf_model.add_task(task_id=0, class_name='bottle')
    nf_model.set_active_task(0)
    nf_model.train()

    # Prepare a real batch for realistic memory measurement
    config = get_config(
        img_size=img_size,
        data_path=data_path,
        msk_size=msk_size,
        batch_size=batch_size,
        seed=0,
        lr=lr,
    )
    setting_lr_parameters(config)
    config.dataset = 'mvtec'

    ALL_CLASSES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}

    train_dataset = create_task_dataset(
        config, ['bottle'], GLOBAL_CLASS_TO_IDX, train=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False, drop_last=True
    )

    # Get one batch
    images, class_ids, masks, names, paths = next(iter(train_loader))
    images = images.to(device)

    # Set up optimizer (Task 0 = base + LoRA)
    optimizer = torch.optim.Adam(
        [p for p in nf_model.parameters() if p.requires_grad],
        lr=lr
    )

    # --- Task 0 peak memory (base + LoRA training) ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Forward pass
    with torch.no_grad():
        patch_embeddings, spatial_shape = feature_extractor(images, return_spatial_shape=True)
    patch_embeddings_with_pos = pos_embed_gen(spatial_shape, patch_embeddings)

    z, logdet_patch = nf_model(patch_embeddings_with_pos)
    B, H, W, D = z.shape

    # Compute loss (standard NLL)
    log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
    log_px_patch = log_pz_patch + logdet_patch
    nll_patch = -log_px_patch  # (B, H, W)
    loss = nll_patch.mean()

    # Backward + step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    peak_task0 = torch.cuda.max_memory_allocated()
    results['task0_peak_gb'] = peak_task0 / (1024 ** 3)
    print(f"[Memory] Task 0 training peak: {results['task0_peak_gb']:.3f} GB")

    # ---------------------------------------------------------------
    # 3. Task N>0 training step (LoRA only, base frozen)
    # ---------------------------------------------------------------
    # Freeze base, add Task 1
    nf_model.add_task(task_id=1, class_name='cable')
    nf_model.set_active_task(1)
    nf_model.train()

    # Rebuild optimizer with only trainable params (LoRA + adapter + ACL for task 1)
    optimizer_task1 = torch.optim.Adam(
        [p for p in nf_model.parameters() if p.requires_grad],
        lr=lr
    )

    # Load cable data
    train_dataset_1 = create_task_dataset(
        config, ['cable'], GLOBAL_CLASS_TO_IDX, train=True
    )
    train_loader_1 = DataLoader(
        train_dataset_1, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False, drop_last=True
    )
    images_1, _, _, _, _ = next(iter(train_loader_1))
    images_1 = images_1.to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Forward
    with torch.no_grad():
        patch_embeddings_1, spatial_shape_1 = feature_extractor(images_1, return_spatial_shape=True)
    patch_embeddings_with_pos_1 = pos_embed_gen(spatial_shape_1, patch_embeddings_1)

    z1, logdet_patch_1 = nf_model(patch_embeddings_with_pos_1)
    log_pz_patch_1 = -0.5 * (z1 ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
    log_px_patch_1 = log_pz_patch_1 + logdet_patch_1
    nll_patch_1 = -log_px_patch_1
    loss_1 = nll_patch_1.mean()

    optimizer_task1.zero_grad()
    loss_1.backward()
    optimizer_task1.step()

    torch.cuda.synchronize()
    peak_taskN = torch.cuda.max_memory_allocated()
    results['taskN_peak_gb'] = peak_taskN / (1024 ** 3)
    print(f"[Memory] Task N>0 training peak: {results['taskN_peak_gb']:.3f} GB")

    # Cleanup
    del nf_model, feature_extractor, optimizer, optimizer_task1
    del images, images_1, patch_embeddings, patch_embeddings_1
    del z, z1, logdet_patch, logdet_patch_1
    torch.cuda.empty_cache()

    return results


# ============================================================================
# Task C: Visualization
# ============================================================================

def plot_training_time(time_data: dict, output_dir: str):
    """
    Create a half-width bar chart of per-task training time.

    Task 0 is highlighted (base + LoRA); a dashed line shows Task 1-14 average.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
    })

    classes = time_data['task_classes']
    times = time_data['task_times_minutes']
    n = len(classes)

    # Colors: Task 0 emphasized, rest in lighter tone
    colors = ['#2171b5'] + ['#6baed6'] * (n - 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x = np.arange(n)
    bars = ax.bar(x, times, width=0.7, color=colors, edgecolor='white', linewidth=0.3)

    # Average line for Task 1-14
    if n > 1:
        avg_time = np.mean(times[1:])
        ax.axhline(y=avg_time, color='#d94801', linestyle='--', linewidth=1.2,
                    label=f'Avg. Task 1\u201314: {avg_time:.1f} min')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    ax.set_xlabel('Task (class)')
    ax.set_ylabel('Time (min)')
    ax.set_title('Training Time per Task')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=55, ha='right', fontsize=7.5)
    ax.set_ylim(0, max(times) * 1.15)

    # Annotate Task 0
    ax.annotate('Base+LoRA', xy=(0, times[0]), xytext=(1.5, times[0] * 1.02),
                fontsize=7, color='#2171b5', ha='center',
                arrowprops=dict(arrowstyle='->', color='#2171b5', lw=0.8))

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'efficiency_training_time.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close(fig)


def plot_memory_usage(mem_data: dict, output_dir: str):
    """
    Create a half-width bar chart of GPU memory usage.

    Three bars: Feature Extractor (frozen), Task 0 training, Task N>0 training.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 10,
    })

    labels = [
        'Feature\nExtractor\n(frozen)',
        'Task 0\nTraining\n(Base+LoRA)',
        'Task N>0\nTraining\n(LoRA only)',
    ]
    values = [
        mem_data['extractor_gb'],
        mem_data['task0_peak_gb'],
        mem_data['taskN_peak_gb'],
    ]
    colors = ['#74c476', '#2171b5', '#6baed6']

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.55, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Peak GPU Memory (GB)')
    ax.set_title('GPU Memory Usage')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, max(values) * 1.2)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'efficiency_memory_usage.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DeCoFlow Efficiency Profiling')
    parser.add_argument('--log_path', type=str,
                        default='/Volume/DeCoFlow/logs/V48_01_H04_highres_clean/V48_01_H04_highres_clean.log',
                        help='Path to training log')
    parser.add_argument('--data_path', type=str, default='/Volume/MVTecAD',
                        help='Path to MVTec AD dataset')
    parser.add_argument('--output_dir', type=str,
                        default='/Volume/DeCoFlow/Paper_works/figures',
                        help='Directory to save figures')
    parser.add_argument('--skip_memory', action='store_true',
                        help='Skip GPU memory profiling (log parsing + viz only)')
    parser.add_argument('--memory_only', action='store_true',
                        help='Only run memory profiling (skip log parsing + viz)')
    parser.add_argument('--cache_file', type=str,
                        default='/Volume/DeCoFlow/Paper_works/figures/efficiency_data.json',
                        help='Cache file for profiling results')
    args = parser.parse_args()

    print("=" * 70)
    print("DeCoFlow Efficiency Profiling")
    print("=" * 70)

    cached_data = {}
    if os.path.exists(args.cache_file):
        with open(args.cache_file, 'r') as f:
            cached_data = json.load(f)
        print(f"Loaded cached data from: {args.cache_file}")

    # ------------------------------------------------------------------
    # Task A: Parse training log
    # ------------------------------------------------------------------
    if not args.memory_only:
        print("\n--- Task A: Training Time Extraction ---")
        time_data = parse_training_times(args.log_path)
        cached_data['time_data'] = time_data

        print(f"Parsed {len(time_data['task_classes'])} tasks from log.")
        print(f"Total training time: {time_data['total_time_minutes']:.1f} min "
              f"({time_data['total_time_minutes']/60:.1f} h)")
        print(f"\nPer-task breakdown:")
        for i, (cls, t) in enumerate(zip(time_data['task_classes'], time_data['task_times_minutes'])):
            tag = " (base+LoRA)" if i == 0 else ""
            print(f"  Task {i:2d} [{cls:12s}]: {t:6.1f} min{tag}")

        if len(time_data['task_times_minutes']) > 1:
            avg = np.mean(time_data['task_times_minutes'][1:])
            print(f"\n  Avg (Task 1-14): {avg:.1f} min")
            print(f"  Task 0 overhead: {time_data['task_times_minutes'][0] - avg:.1f} min "
                  f"({time_data['task_times_minutes'][0] / avg:.1f}x)")
    else:
        time_data = cached_data.get('time_data')
        if time_data is None:
            print("ERROR: No cached time data found. Run without --memory_only first.")
            return

    # ------------------------------------------------------------------
    # Task B: Memory profiling
    # ------------------------------------------------------------------
    if not args.skip_memory:
        print("\n--- Task B: GPU Memory Profiling ---")
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Skipping memory profiling.")
            mem_data = cached_data.get('mem_data')
        else:
            mem_data = profile_memory(data_path=args.data_path)
            cached_data['mem_data'] = mem_data
    else:
        print("\n--- Task B: Skipped (--skip_memory) ---")
        mem_data = cached_data.get('mem_data')

    # Save cache
    os.makedirs(os.path.dirname(args.cache_file), exist_ok=True)
    with open(args.cache_file, 'w') as f:
        json.dump(cached_data, f, indent=2)
    print(f"\nCached profiling data to: {args.cache_file}")

    # ------------------------------------------------------------------
    # Task C: Visualization
    # ------------------------------------------------------------------
    if not args.memory_only:
        print("\n--- Task C: Visualization ---")

        # Figure 1: Training time per task
        plot_training_time(time_data, args.output_dir)

        # Figure 2: Memory usage
        if mem_data is not None:
            plot_memory_usage(mem_data, args.output_dir)
        else:
            print("WARNING: No memory data available. Skipping memory figure.")
            print("  Run with CUDA_VISIBLE_DEVICES=0 (without --skip_memory) to profile memory.")

    print("\nDone.")


if __name__ == '__main__':
    main()
