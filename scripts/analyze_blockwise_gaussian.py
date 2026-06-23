#!/usr/bin/env python
"""
Block-wise Gaussian Convergence Analysis (Fig. 6)

Analyzes how the latent distribution progressively converges to N(0,I)
as data passes through DCL1->DCL6->ACL1->ACL2 blocks.

Architecture:
  - 6 DCL blocks: FrEIA SequenceINN with AllInOneBlock modules
  - 2 ACL (ACB) blocks: AuxiliaryCouplingBlocks with AffineCouplingBlock modules

Produces:
  - Line plot: Q-Q correlation vs block index
  - X-axis: Input, DCL1..DCL6, ACL1, ACL2
  - 15 thin gray lines (per-class) + 1 thick colored mean line

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/analyze_blockwise_gaussian.py

Output:
  - Paper_works/figures/blockwise_gaussian_convergence.{pdf,png}
  - logs/5_Analysis/blockwise_gaussian_data.json
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

from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.models.dcl import DCLContextSubnet
from decoflow.models.position_embedding import PositionalEmbeddingGenerator
from decoflow.extractors import create_feature_extractor, get_backbone_type
from decoflow.config.ablation import AblationConfig
from decoflow.data.mvtec import MVTEC, MVTEC_CLASS_NAMES
from decoflow.utils.checkpoint import load_checkpoint
from decoflow.utils.helpers import init_seeds

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


def compute_qq_correlation(z_tensor, n_samples=N_QQ_SAMPLES):
    """Compute Q-Q correlation between z and standard normal N(0,1)."""
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


def build_ablation_config():
    """Build AblationConfig matching V48_01_H04_highres_clean."""
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
    Run forward pass capturing intermediate z after each DCL and ACL block.

    Returns dict with keys input, dcl1..dcl6, acl1, acl2,
    each mapping to a CPU tensor of shape (B, H, W, D).
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

    intermediates["input"] = x.detach().cpu()

    # --- DCL blocks (FrEIA SequenceINN) ---
    x_flat = x.reshape(B * H * W, D)

    if nf_model.use_scale_context:
        DCLContextSubnet._spatial_info = (B, H, W)

    # FrEIA SequenceINN wraps input as tuple
    x_tuple = (x_flat,)
    for i, module in enumerate(nf_model.flow.module_list):
        x_tuple, _ = module(x_tuple, jac=True, rev=False)
        z_flat = x_tuple[0]
        z_spatial = z_flat.reshape(B, H, W, D)
        intermediates["dcl" + str(i + 1)] = z_spatial.detach().cpu()

    if nf_model.use_scale_context:
        DCLContextSubnet._spatial_info = None

    # --- ACL blocks (ACB) ---
    if nf_model.use_acb and nf_model.current_task_id is not None:
        task_key = str(nf_model.current_task_id)
        if task_key in nf_model.acb_adapters:
            acb = nf_model.acb_adapters[task_key]
            z = x_tuple[0].reshape(B, H, W, D)
            for j, block in enumerate(acb.coupling_blocks):
                z, _ = block(z, reverse=False)
                intermediates["acl" + str(j + 1)] = z.detach().cpu()

    return intermediates


def main():
    print("=" * 70)
    print("Block-wise Gaussian Convergence Analysis (Fig. 6)")
    print("=" * 70)
    print("  Device:", DEVICE)
    print("  Checkpoint:", CHECKPOINT_DIR)
    print("  Backbone:", BACKBONE_NAME)
    print("  Embed dim:", EMBED_DIM)
    print("  Architecture: %d DCL + %d ACL blocks" % (NUM_COUPLING_LAYERS, ACB_N_BLOCKS))
    print("  Classes:", len(ALL_CLASSES))
    print("  Batches per class: %d (batch_size=%d)" % (NUM_BATCHES, BATCH_SIZE))
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

    # 3. Collect Q-Q correlations per class, per block
    print("[3/4] Computing Q-Q correlations...")
    all_correlations = OrderedDict()

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
                        block_z_accum[label].append(intermediates[key])

                n_batches_done += 1

        correlations = []
        for label in BLOCK_LABELS:
            if block_z_accum[label]:
                z_cat = torch.cat(block_z_accum[label], dim=0)
                corr = compute_qq_correlation(z_cat)
                correlations.append(corr)
            else:
                correlations.append(float("nan"))

        all_correlations[class_name] = correlations
        corr_str = " ".join(["%.4f" % c for c in correlations])
        print(corr_str)

    # 4. Plot and save
    print("[4/4] Generating plot...")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    corr_matrix = np.array([all_correlations[c] for c in ALL_CLASSES])
    mean_corr = np.nanmean(corr_matrix, axis=0)
    std_corr = np.nanstd(corr_matrix, axis=0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x_positions = np.arange(len(BLOCK_LABELS))

    for class_name in ALL_CLASSES:
        corrs = all_correlations[class_name]
        ax.plot(
            x_positions, corrs,
            color="gray", alpha=0.25, linewidth=0.8, zorder=1,
        )

    ax.plot(
        x_positions, mean_corr,
        color="#2563EB", linewidth=2.5, marker="o", markersize=6,
        markeredgecolor="white", markeredgewidth=1.0,
        label="Mean (15 classes)", zorder=3,
    )

    ax.fill_between(
        x_positions,
        mean_corr - std_corr,
        mean_corr + std_corr,
        alpha=0.15, color="#2563EB", zorder=2,
    )

    boundary_x = 6.5
    ax.axvline(
        x=boundary_x, color="gray", linestyle="--",
        linewidth=1.0, alpha=0.7, zorder=0,
    )

    ymin, ymax = ax.get_ylim()
    label_y = ymin + 0.02 * (ymax - ymin)
    ax.text(
        3.0, label_y, "DCL (Normalizing Flow)",
        ha="center", va="bottom", fontsize=9, color="gray", style="italic",
    )
    ax.text(
        7.5, label_y, "ACL",
        ha="center", va="bottom", fontsize=9, color="gray", style="italic",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(BLOCK_LABELS, fontsize=10)
    ax.set_xlabel("Block", fontsize=12)
    ax.set_ylabel(
        r"Q-Q Correlation with $\mathcal{N}(0, I)$", fontsize=12
    )
    ax.set_title(
        "Block-wise Gaussian Convergence", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_ylim(top=1.002)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()

    pdf_path = os.path.join(FIG_DIR, "blockwise_gaussian_convergence.pdf")
    png_path = os.path.join(FIG_DIR, "blockwise_gaussian_convergence.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", pdf_path)
    print("  Saved:", png_path)

    data_out = {
        "block_labels": BLOCK_LABELS,
        "classes": list(ALL_CLASSES),
        "per_class_correlations": {
            c: all_correlations[c] for c in ALL_CLASSES
        },
        "mean_correlation": mean_corr.tolist(),
        "std_correlation": std_corr.tolist(),
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
    json_path = os.path.join(DATA_DIR, "blockwise_gaussian_data.json")
    with open(json_path, "w") as f:
        json.dump(data_out, f, indent=2)
    print("  Saved:", json_path)

    # Summary table
    print()
    print("=" * 90)
    header = "%12s" % "Block"
    for label in BLOCK_LABELS:
        header += " %8s" % label
    print(header)
    print("-" * 90)
    for class_name in ALL_CLASSES:
        row = "%12s" % class_name
        for corr in all_correlations[class_name]:
            row += " %8.5f" % corr
        print(row)
    print("-" * 90)
    row = "%12s" % "Mean"
    for m in mean_corr:
        row += " %8.5f" % m
    print(row)
    row = "%12s" % "Std"
    for s in std_corr:
        row += " %8.5f" % s
    print(row)
    print("=" * 90)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
