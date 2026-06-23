#!/usr/bin/env python
"""
Enhanced Block-wise Analysis (Fig. 6 — Multi-Panel)

Three complementary metrics measured after each DCL/ACL block:
  (a) Q-Q Correlation — marginal Gaussianity (existing metric)
  (b) Cross-Covariance Off-Diagonal Norm — cross-dimensional independence
  (c) Per-block |log-det J| — transformation magnitude

Architecture: 6 DCL blocks (FrEIA) + 2 ACL blocks (ACB AffineCoupling)

Key Hypothesis:
  - DCL primarily does cross-dimensional disentangling (visible in metric b)
  - ACL primarily refines marginal Gaussianity (visible in metric a)
  → Complementary roles, not redundant

Usage:
  CUDA_VISIBLE_DEVICES=5 python scripts/analyze_blockwise_enhanced.py

Output:
  - Paper_works/figures/blockwise_enhanced_analysis.{pdf,png}
  - logs/5_Analysis/blockwise_enhanced_data.json
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from collections import OrderedDict

PROJECT_ROOT = "/Volume/DeCoFlow"
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.models.dcl import DCLContextSubnet
from decoflow.models.position_embedding import PositionalEmbeddingGenerator
from decoflow.extractors import create_feature_extractor, get_backbone_type
from decoflow.config.ablation import AblationConfig
from decoflow.data.mvtec import MVTEC, MVTEC_CLASS_NAMES
from decoflow.utils.checkpoint import load_checkpoint
from decoflow.utils.helpers import init_seeds

# ─── Config ───────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "logs/V48_01_H04_highres_clean/checkpoints")
DATA_PATH = "/Volume/MVTecAD"
BACKBONE_NAME = "wide_resnet50_2"
EMBED_DIM = 768
IMG_SIZE = 224
NUM_COUPLING_LAYERS = 6
ACB_N_BLOCKS = 2
LORA_RANK = 64
LORA_ALPHA = 1.0
USE_HIGH_RES = True
BATCH_SIZE = 16
NUM_BATCHES = 3
N_QQ_SAMPLES = 50000
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIG_DIR = os.path.join(PROJECT_ROOT, "Paper_works/figures")
DATA_DIR = os.path.join(PROJECT_ROOT, "logs/5_Analysis")

ALL_CLASSES = MVTEC_CLASS_NAMES
BLOCK_LABELS = [
    "Input", "DCL1", "DCL2", "DCL3", "DCL4", "DCL5", "DCL6", "ACL1", "ACL2"
]


# ─── Metrics ──────────────────────────────────────────────────────────

def compute_qq_correlation(z_tensor, n_samples=N_QQ_SAMPLES):
    """Q-Q correlation with N(0,1) — marginal Gaussianity."""
    z_flat = z_tensor.reshape(-1).numpy()
    if len(z_flat) > n_samples:
        rng = np.random.RandomState(SEED)
        indices = rng.choice(len(z_flat), n_samples, replace=False)
        z_flat = z_flat[indices]
    z_sorted = np.sort(z_flat)
    n = len(z_sorted)
    theoretical = stats.norm.ppf(np.linspace(1 / (n + 1), n / (n + 1), n))
    corr = np.corrcoef(theoretical, z_sorted)[0, 1]
    return float(corr)


def compute_offdiag_norm(z_tensor, n_samples=20000):
    """
    Off-diagonal Frobenius norm of the covariance matrix.

    Measures cross-dimensional dependence:
    - High value → dimensions are correlated (not independent)
    - Low value → dimensions are approximately independent

    This metric captures what Q-Q correlation misses:
    DCL may disentangle cross-dimensional correlations while
    not affecting marginal distributions.
    """
    # z_tensor: (B, H, W, D) → flatten to (N, D)
    z_flat = z_tensor.reshape(-1, z_tensor.shape[-1]).numpy()

    if len(z_flat) > n_samples:
        rng = np.random.RandomState(SEED)
        indices = rng.choice(len(z_flat), n_samples, replace=False)
        z_flat = z_flat[indices]

    # Compute covariance matrix (D x D)
    cov = np.cov(z_flat, rowvar=False)  # (D, D)

    # Extract off-diagonal elements
    D = cov.shape[0]
    mask = ~np.eye(D, dtype=bool)
    offdiag = cov[mask]

    # Frobenius norm of off-diagonal (normalized by D for comparability)
    offdiag_norm = np.sqrt(np.sum(offdiag ** 2)) / D

    return float(offdiag_norm)


# ─── Model Setup ──────────────────────────────────────────────────────

def build_ablation_config():
    config = AblationConfig()
    config.use_lora = True
    config.use_task_adapter = True
    config.use_task_bias = True
    config.use_pos_embedding = True
    config.use_router = True
    config.use_spatial_context = True
    config.spatial_context_kernel = 3
    config.use_scale_context = True
    config.scale_context_kernel = 5
    config.use_acb = True
    config.acb_n_blocks = ACB_N_BLOCKS
    config.use_tsa = True
    config.use_tail_aware_loss = True
    config.tail_weight = 0.85
    config.score_aggregation_mode = "top_k"
    config.score_aggregation_top_k = 3
    config.score_smooth_sigma = 0.0
    return config


def forward_with_hooks(nf_model, features, task_id):
    """
    Run forward pass capturing intermediate z AND per-block log-det
    after each DCL and ACL block.

    Returns:
        intermediates: dict of block_name → (z_cpu, log_det_cpu)
    """
    B, H, W, D = features.shape
    nf_model.set_active_task(task_id)
    intermediates = OrderedDict()

    x = features.clone()

    # Pre-flow: input adapter
    if nf_model.use_task_adapter and nf_model.current_task_id is not None:
        task_key = str(nf_model.current_task_id)
        if task_key in nf_model.input_adapters:
            x = nf_model.input_adapters[task_key](x)

    # Pre-flow: spatial context mixing
    if nf_model.spatial_mixer is not None:
        if (nf_model.use_task_adaptive_context
                and nf_model.task_adaptive_mixer is not None):
            x = nf_model.task_adaptive_mixer(x)
        else:
            x = nf_model.spatial_mixer(x)

    intermediates["input"] = (x.detach().cpu(), None)

    # --- DCL blocks (FrEIA SequenceINN) ---
    x_flat = x.reshape(B * H * W, D)

    if nf_model.use_scale_context:
        DCLContextSubnet._spatial_info = (B, H, W)

    x_tuple = (x_flat,)
    for i, module in enumerate(nf_model.flow.module_list):
        x_tuple, log_jac = module(x_tuple, jac=True, rev=False)
        z_flat = x_tuple[0]
        z_spatial = z_flat.reshape(B, H, W, D)
        # log_jac from FrEIA module: shape (B*H*W,) — per-element log-det
        log_det_spatial = log_jac.reshape(B, H, W) if log_jac is not None else None
        intermediates["dcl" + str(i + 1)] = (
            z_spatial.detach().cpu(),
            log_det_spatial.detach().cpu() if log_det_spatial is not None else None,
        )

    if nf_model.use_scale_context:
        DCLContextSubnet._spatial_info = None

    # --- ACL blocks (ACB) ---
    if nf_model.use_acb and nf_model.current_task_id is not None:
        task_key = str(nf_model.current_task_id)
        if task_key in nf_model.acb_adapters:
            acb = nf_model.acb_adapters[task_key]
            z = x_tuple[0].reshape(B, H, W, D)
            for j, block in enumerate(acb.coupling_blocks):
                z, block_logdet = block(z, reverse=False)
                intermediates["acl" + str(j + 1)] = (
                    z.detach().cpu(),
                    block_logdet.detach().cpu(),
                )

    return intermediates


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Enhanced Block-wise Analysis (Fig. 6 — Multi-Panel)")
    print("=" * 70)
    print("  Device:", DEVICE)
    print("  Checkpoint:", CHECKPOINT_DIR)
    print("  Metrics: Q-Q Correlation, Off-Diagonal Cov Norm, |log-det|")
    print()

    init_seeds(SEED)

    # 1. Build model
    print("[1/4] Building model...")
    ablation_config = build_ablation_config()

    nf_model = DeCoFlowNF(
        embed_dim=EMBED_DIM,
        coupling_layers=NUM_COUPLING_LAYERS,
        clamp_alpha=1.9,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        device=DEVICE,
        ablation_config=ablation_config,
    )

    for task_id in range(len(ALL_CLASSES)):
        nf_model.add_task(task_id)

    load_checkpoint(
        nf_model=nf_model,
        router=None,
        checkpoint_dir=CHECKPOINT_DIR,
        device=DEVICE,
        task_id=14,
    )
    nf_model.eval()
    print("  Model loaded and set to eval mode.")

    # 2. Build feature extractor
    print("[2/4] Initializing feature extractor...")

    feature_extractor = create_feature_extractor(
        backbone_name=BACKBONE_NAME,
        input_shape=(3, IMG_SIZE, IMG_SIZE),
        target_embed_dimension=EMBED_DIM,
        device=DEVICE,
        patch_size=3,
        patch_stride=1,
        use_high_res=USE_HIGH_RES,
    )
    feature_extractor.eval()

    pos_embed_gen = PositionalEmbeddingGenerator(device=DEVICE)

    # 3. Collect metrics per class, per block
    print("[3/4] Computing metrics...")
    all_qq = OrderedDict()
    all_offdiag = OrderedDict()
    all_logdet = OrderedDict()

    for task_id, class_name in enumerate(ALL_CLASSES):
        print("  Task %2d (%12s): " % (task_id, class_name), end="", flush=True)

        dataset = MVTEC(
            root=DATA_PATH,
            class_name=class_name,
            train=True,
            img_size=IMG_SIZE,
            crp_size=IMG_SIZE,
            msk_size=256,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        block_z_accum = {label: [] for label in BLOCK_LABELS}
        block_logdet_accum = {label: [] for label in BLOCK_LABELS}
        n_batches_done = 0

        with torch.no_grad():
            for images, _, _, _, _ in loader:
                if n_batches_done >= NUM_BATCHES:
                    break
                images = images.to(DEVICE)

                patch_embeddings, spatial_shape = feature_extractor(
                    images, return_spatial_shape=True
                )
                features_with_pos = pos_embed_gen(spatial_shape, patch_embeddings)

                intermediates = forward_with_hooks(
                    nf_model, features_with_pos, task_id
                )

                for label in BLOCK_LABELS:
                    key = label.lower()
                    if key in intermediates:
                        z_cpu, ld_cpu = intermediates[key]
                        block_z_accum[label].append(z_cpu)
                        if ld_cpu is not None:
                            block_logdet_accum[label].append(ld_cpu)

                n_batches_done += 1

        # Compute metrics per block
        qq_vals = []
        offdiag_vals = []
        logdet_vals = []

        for label in BLOCK_LABELS:
            if block_z_accum[label]:
                z_cat = torch.cat(block_z_accum[label], dim=0)
                qq_vals.append(compute_qq_correlation(z_cat))
                offdiag_vals.append(compute_offdiag_norm(z_cat))
            else:
                qq_vals.append(float("nan"))
                offdiag_vals.append(float("nan"))

            if block_logdet_accum[label]:
                ld_cat = torch.cat(block_logdet_accum[label], dim=0)
                # Mean absolute log-det per spatial location
                logdet_vals.append(float(ld_cat.abs().mean().item()))
            else:
                logdet_vals.append(0.0)

        all_qq[class_name] = qq_vals
        all_offdiag[class_name] = offdiag_vals
        all_logdet[class_name] = logdet_vals

        print("QQ=%.4f→%.4f  OffDiag=%.4f→%.4f  |LD|=%.2f/%.2f" % (
            qq_vals[0], qq_vals[-1],
            offdiag_vals[0], offdiag_vals[-1],
            logdet_vals[6] if len(logdet_vals) > 6 else 0,  # DCL6
            logdet_vals[8] if len(logdet_vals) > 8 else 0,  # ACL2
        ))

    # 4. Plot multi-panel figure
    print("[4/4] Generating multi-panel plot...")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    qq_matrix = np.array([all_qq[c] for c in ALL_CLASSES])
    offdiag_matrix = np.array([all_offdiag[c] for c in ALL_CLASSES])
    logdet_matrix = np.array([all_logdet[c] for c in ALL_CLASSES])

    mean_qq = np.nanmean(qq_matrix, axis=0)
    std_qq = np.nanstd(qq_matrix, axis=0)
    mean_offdiag = np.nanmean(offdiag_matrix, axis=0)
    std_offdiag = np.nanstd(offdiag_matrix, axis=0)
    mean_logdet = np.nanmean(logdet_matrix, axis=0)
    std_logdet = np.nanstd(logdet_matrix, axis=0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    x_positions = np.arange(len(BLOCK_LABELS))
    boundary_x = 6.5

    colors = {"main": "#2563EB", "fill": "#2563EB", "gray": "gray"}

    # --- Panel (a): Q-Q Correlation ---
    ax = axes[0]
    for class_name in ALL_CLASSES:
        ax.plot(x_positions, all_qq[class_name],
                color="gray", alpha=0.2, linewidth=0.7, zorder=1)
    ax.plot(x_positions, mean_qq,
            color=colors["main"], linewidth=2.2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=0.8,
            label="Mean (15 classes)", zorder=3)
    ax.fill_between(x_positions, mean_qq - std_qq, mean_qq + std_qq,
                    alpha=0.12, color=colors["fill"], zorder=2)
    ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(BLOCK_LABELS, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(r"Q-Q Corr. with $\mathcal{N}(0,1)$", fontsize=10)
    ax.set_title("(a) Marginal Gaussianity", fontsize=11, fontweight="bold")
    ax.set_ylim(top=1.005)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    # Add DCL/ACL labels
    ymin, ymax = ax.get_ylim()
    label_y = ymin + 0.03 * (ymax - ymin)
    ax.text(3.0, label_y, "DCL", ha="center", fontsize=8, color="gray", style="italic")
    ax.text(7.5, label_y, "ACL", ha="center", fontsize=8, color="gray", style="italic")

    # --- Panel (b): Off-Diagonal Covariance Norm ---
    ax = axes[1]
    for class_name in ALL_CLASSES:
        ax.plot(x_positions, all_offdiag[class_name],
                color="gray", alpha=0.2, linewidth=0.7, zorder=1)
    ax.plot(x_positions, mean_offdiag,
            color="#DC2626", linewidth=2.2, marker="s", markersize=5,
            markeredgecolor="white", markeredgewidth=0.8,
            label="Mean (15 classes)", zorder=3)
    ax.fill_between(x_positions, mean_offdiag - std_offdiag, mean_offdiag + std_offdiag,
                    alpha=0.12, color="#DC2626", zorder=2)
    ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(BLOCK_LABELS, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Off-Diag. Cov. Norm (×D⁻¹)", fontsize=10)
    ax.set_title("(b) Cross-Dim. Independence", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ymin, ymax = ax.get_ylim()
    label_y = ymin + 0.03 * (ymax - ymin)
    ax.text(3.0, label_y, "DCL", ha="center", fontsize=8, color="gray", style="italic")
    ax.text(7.5, label_y, "ACL", ha="center", fontsize=8, color="gray", style="italic")

    # --- Panel (c): Per-block |log-det| ---
    ax = axes[2]
    # Bar chart for log-det (more intuitive for per-block magnitude)
    dcl_mask = [i for i, l in enumerate(BLOCK_LABELS) if l.startswith("DCL")]
    acl_mask = [i for i, l in enumerate(BLOCK_LABELS) if l.startswith("ACL")]
    bar_colors = []
    for i in range(len(BLOCK_LABELS)):
        if BLOCK_LABELS[i] == "Input":
            bar_colors.append("#9CA3AF")
        elif i in dcl_mask:
            bar_colors.append("#2563EB")
        else:
            bar_colors.append("#DC2626")

    ax.bar(x_positions, mean_logdet, yerr=std_logdet,
           color=bar_colors, alpha=0.75, edgecolor="white", linewidth=0.5,
           capsize=3, error_kw={"linewidth": 1.0, "capthick": 1.0})
    ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(BLOCK_LABELS, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Mean |log det J|", fontsize=10)
    ax.set_title("(c) Transformation Magnitude", fontsize=11, fontweight="bold")
    ymin, ymax = ax.get_ylim()
    label_y = ymin + 0.03 * (ymax - ymin)
    ax.text(3.0, label_y, "DCL", ha="center", fontsize=8, color="gray", style="italic")
    ax.text(7.5, label_y, "ACL", ha="center", fontsize=8, color="gray", style="italic")

    plt.tight_layout(w_pad=2.0)

    pdf_path = os.path.join(FIG_DIR, "blockwise_enhanced_analysis.pdf")
    png_path = os.path.join(FIG_DIR, "blockwise_enhanced_analysis.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", pdf_path)
    print("  Saved:", png_path)

    # Save data
    data_out = {
        "block_labels": BLOCK_LABELS,
        "classes": list(ALL_CLASSES),
        "metrics": {
            "qq_correlation": {
                "per_class": {c: all_qq[c] for c in ALL_CLASSES},
                "mean": mean_qq.tolist(),
                "std": std_qq.tolist(),
            },
            "offdiag_cov_norm": {
                "per_class": {c: all_offdiag[c] for c in ALL_CLASSES},
                "mean": mean_offdiag.tolist(),
                "std": std_offdiag.tolist(),
            },
            "abs_logdet": {
                "per_class": {c: all_logdet[c] for c in ALL_CLASSES},
                "mean": mean_logdet.tolist(),
                "std": std_logdet.tolist(),
            },
        },
        "config": {
            "backbone": BACKBONE_NAME,
            "embed_dim": EMBED_DIM,
            "num_dcl_blocks": NUM_COUPLING_LAYERS,
            "num_acl_blocks": ACB_N_BLOCKS,
            "lora_rank": LORA_RANK,
            "checkpoint": CHECKPOINT_DIR,
            "num_batches": NUM_BATCHES,
            "batch_size": BATCH_SIZE,
            "n_qq_samples": N_QQ_SAMPLES,
        },
    }
    json_path = os.path.join(DATA_DIR, "blockwise_enhanced_data.json")
    with open(json_path, "w") as f:
        json.dump(data_out, f, indent=2)
    print("  Saved:", json_path)

    # Summary
    print()
    print("=" * 70)
    print("Summary of Mean Metrics:")
    print("-" * 70)
    header = "%8s" % ""
    for label in BLOCK_LABELS:
        header += " %8s" % label
    print(header)
    print("-" * 70)

    row = "%8s" % "QQ"
    for v in mean_qq:
        row += " %8.4f" % v
    print(row)

    row = "%8s" % "OffDiag"
    for v in mean_offdiag:
        row += " %8.4f" % v
    print(row)

    row = "%8s" % "|LogDet|"
    for v in mean_logdet:
        row += " %8.4f" % v
    print(row)

    print("=" * 70)

    # Key interpretation
    print()
    print("Key Findings:")
    dcl_qq_change = mean_qq[6] - mean_qq[0]
    acl_qq_change = mean_qq[8] - mean_qq[6]
    dcl_offdiag_change = mean_offdiag[6] - mean_offdiag[0]
    acl_offdiag_change = mean_offdiag[8] - mean_offdiag[6]
    dcl_total_logdet = sum(mean_logdet[1:7])
    acl_total_logdet = sum(mean_logdet[7:9])

    print("  DCL (Input→DCL6):")
    print("    Q-Q Change:     %+.4f (%.3f → %.3f)" % (dcl_qq_change, mean_qq[0], mean_qq[6]))
    print("    OffDiag Change: %+.4f (%.4f → %.4f)" % (dcl_offdiag_change, mean_offdiag[0], mean_offdiag[6]))
    print("    Total |LogDet|: %.4f" % dcl_total_logdet)
    print("  ACL (DCL6→ACL2):")
    print("    Q-Q Change:     %+.4f (%.3f → %.3f)" % (acl_qq_change, mean_qq[6], mean_qq[8]))
    print("    OffDiag Change: %+.4f (%.4f → %.4f)" % (acl_offdiag_change, mean_offdiag[6], mean_offdiag[8]))
    print("    Total |LogDet|: %.4f" % acl_total_logdet)
    print()

    if dcl_offdiag_change < 0 and acl_qq_change > 0:
        print("  ✓ CONFIRMS complementary roles:")
        print("    DCL → cross-dim disentangling (OffDiag↓)")
        print("    ACL → marginal Gaussianity refinement (Q-Q↑)")
    else:
        print("  ⚠ Results need further interpretation:")
        print("    DCL OffDiag: %s" % ("↓ (disentangling)" if dcl_offdiag_change < 0 else "↑ (unexpected)"))
        print("    ACL Q-Q: %s" % ("↑ (Gaussianizing)" if acl_qq_change > 0 else "↓ (unexpected)"))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
