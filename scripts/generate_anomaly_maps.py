#!/usr/bin/env python
"""
Generate anomaly map visualizations for DeCoFlow paper (Fig. 4).

For top-4 Pixel-AP MVTec classes (metal_nut, bottle, pill, carpet),
generate heatmap overlays: Input | GT mask | DeCoFlow heatmap.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_anomaly_maps.py
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
PROJECT_ROOT = '/Volume/DeCoFlow'
sys.path.insert(0, PROJECT_ROOT)

from decoflow.extractors import create_feature_extractor
from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.models.position_embedding import PositionalEmbeddingGenerator
from decoflow.config.ablation import AblationConfig
from decoflow.utils.checkpoint import load_checkpoint
from decoflow.data.mvtec import MVTEC

# ============================================================================
# Configuration
# ============================================================================

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'logs/V48_01_H04_highres_clean/checkpoints')
DATA_PATH = '/Volume/MVTecAD'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Paper_works/figures')

# V48_01 config
BACKBONE_NAME = 'wide_resnet50_2'
EMBED_DIM = 768
NUM_COUPLING_LAYERS = 6
ACL_N_LAYERS = 2
LORA_RANK = 64
LORA_ALPHA = 1.0
IMG_SIZE = 224
MSK_SIZE = 256
SCORE_SMOOTH_SIGMA = 0.0  # No smoothing

# All 15 MVTec classes in training order (task IDs)
ALL_CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]
NUM_TASKS = len(ALL_CLASSES)

# Classes to visualize with their task IDs — Top 4 by Pixel-AP
VIS_CLASSES = {
    'metal_nut': 7,
    'bottle': 0,
    'pill': 8,
    'carpet': 3,
}
# Order for figure rows
VIS_ORDER = ['metal_nut', 'bottle', 'pill', 'carpet']

# ImageNet normalization for de-normalizing
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Number of images per class: 1 normal + 2 anomaly
N_NORMAL = 1
N_ANOMALY = 2


def denormalize_image(img_tensor):
    """Convert normalized tensor (C, H, W) to displayable numpy array (H, W, C) in [0, 1]."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img


def build_model(device):
    """Build feature extractor and NF model, load checkpoint."""
    print("Building feature extractor...")
    feature_extractor = create_feature_extractor(
        backbone_name=BACKBONE_NAME,
        input_shape=(3, IMG_SIZE, IMG_SIZE),
        target_embed_dimension=EMBED_DIM,
        device=device,
        patch_size=3,
        patch_stride=1,
        use_high_res=True,
    )

    print("Building NF model...")
    ablation_config = AblationConfig(
        use_lora=True,
        use_tsa=True,
        use_acl=True,
        acl_n_layers=ACL_N_LAYERS,
        use_tail_aware_loss=True,
        tail_weight=0.85,
        tail_top_k_ratio=0.02,
        score_aggregation_mode='top_k',
        score_aggregation_top_k=3,
        lambda_logdet=1e-4,
        scale_context_kernel=5,
        score_smooth_sigma=SCORE_SMOOTH_SIGMA,
    )

    nf_model = DeCoFlowNF(
        embed_dim=EMBED_DIM,
        coupling_layers=NUM_COUPLING_LAYERS,
        clamp_alpha=1.9,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        device=device,
        ablation_config=ablation_config,
    )

    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    # Add all 15 tasks (must happen before loading checkpoint)
    print("Adding all 15 tasks...")
    for task_id in range(NUM_TASKS):
        nf_model.add_task(task_id)
        nf_model.set_active_task(task_id)

    # Load checkpoint (task_14 = final)
    print("Loading checkpoint from: %s" % CHECKPOINT_DIR)
    load_checkpoint(
        nf_model=nf_model,
        router=None,  # Not needed for visualization
        checkpoint_dir=CHECKPOINT_DIR,
        device=str(device),
        task_id=14,  # Load final checkpoint
    )
    print("Checkpoint loaded successfully.")

    nf_model.eval()
    feature_extractor.eval()

    return feature_extractor, nf_model, pos_embed_generator


def compute_anomaly_map(images, feature_extractor, nf_model, pos_embed_generator, task_id, device):
    """
    Compute anomaly score maps for a batch of images.

    Returns:
        anomaly_maps: (B, MSK_SIZE, MSK_SIZE) numpy array
        raw_maps: (B, H, W) raw score maps before upscaling
    """
    with torch.no_grad():
        images = images.to(device)

        # Set active task
        nf_model.set_active_task(task_id)

        # Extract features
        patch_embeddings, spatial_shape = feature_extractor(images, return_spatial_shape=True)
        B = patch_embeddings.shape[0]
        H, W = spatial_shape

        # Add positional embeddings
        patch_embeddings_with_pos = pos_embed_generator(spatial_shape, patch_embeddings)

        # Forward through NF
        z, logdet_patch = nf_model.forward(patch_embeddings_with_pos, reverse=False)
        D = z.shape[-1]

        # Compute NLL per patch: -log p(z) - log|det J|
        nll = 0.5 * (z ** 2).sum(dim=-1) + 0.5 * D * math.log(2 * math.pi) - logdet_patch
        # nll shape: (B, H, W)

        raw_maps = nll.cpu()

        # Upscale to MSK_SIZE using bicubic interpolation
        nll_up = F.interpolate(
            nll.unsqueeze(1),
            size=(MSK_SIZE, MSK_SIZE),
            mode='bicubic',
            align_corners=False,
        ).squeeze(1)  # (B, MSK_SIZE, MSK_SIZE)

        anomaly_maps = nll_up.cpu().numpy()

    return anomaly_maps, raw_maps.numpy()


def select_images(class_name, n_normal=1, n_anomaly=2):
    """
    Select representative normal and anomaly images from test set.

    Returns (normals, anomalies) as lists of (image, mask, label, img_type, filename).
    """
    dataset = MVTEC(
        root=DATA_PATH,
        class_name=class_name,
        train=False,
        img_size=IMG_SIZE,
        crp_size=IMG_SIZE,
        msk_size=MSK_SIZE,
    )

    normals = []
    anomalies = []

    for i in range(len(dataset)):
        image, label, mask, filename, img_type = dataset[i]
        if label == 0:
            normals.append((image, mask, label, img_type, filename))
        else:
            anomalies.append((image, mask, label, img_type, filename))

    # Select diverse anomaly types
    anomaly_by_type = {}
    for item in anomalies:
        img_type = item[3]
        if img_type not in anomaly_by_type:
            anomaly_by_type[img_type] = []
        anomaly_by_type[img_type].append(item)

    sorted_types = sorted(anomaly_by_type.keys())

    selected_anomalies = []
    for atype in sorted_types:
        if len(selected_anomalies) >= n_anomaly:
            break
        selected_anomalies.append(anomaly_by_type[atype][0])

    # Fill if not enough types
    if len(selected_anomalies) < n_anomaly:
        for item in anomalies:
            if item not in selected_anomalies:
                selected_anomalies.append(item)
            if len(selected_anomalies) >= n_anomaly:
                break

    selected_normals = normals[:n_normal]

    return selected_normals, selected_anomalies


def create_figure(all_data, feature_extractor, nf_model, pos_embed_generator, device):
    """
    Create the full figure with layout:
        Columns: Normal (Input|GT|Heatmap), Anomaly1 (Input|GT|Heatmap), Anomaly2 (Input|GT|Heatmap)
        Rows: screw, grid, metal_nut, leather
    """
    n_rows = len(VIS_ORDER)
    n_sample_groups = 3  # Normal, Anomaly1, Anomaly2
    n_subcols = 3  # Input, GT, Heatmap per group
    total_cols = n_sample_groups * n_subcols  # 9

    fig_width = total_cols * 1.8
    fig_height = n_rows * 2.0 + 0.8  # extra for titles

    fig, axes = plt.subplots(
        n_rows, total_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={'wspace': 0.02, 'hspace': 0.15},
    )

    # Subcol titles
    subcol_titles = ['Input', 'GT', 'Ours']
    group_titles = ['Normal', 'Anomaly 1', 'Anomaly 2']

    for row_idx, class_name in enumerate(VIS_ORDER):
        task_id = VIS_CLASSES[class_name]
        normals, anomalies = all_data[class_name]

        # Combine: [normal_0, anomaly_0, anomaly_1]
        samples = normals[:N_NORMAL] + anomalies[:N_ANOMALY]

        # --- Phase 1: Compute all anomaly maps for this row ---
        row_anom_maps = []
        row_images = []
        row_masks = []
        for (image, mask, label, img_type, filename) in samples:
            img_batch = image.unsqueeze(0)
            anom_maps, _ = compute_anomaly_map(
                img_batch, feature_extractor, nf_model, pos_embed_generator, task_id, device
            )
            row_anom_maps.append(anom_maps[0])  # (MSK_SIZE, MSK_SIZE)

            img_display = denormalize_image(image)
            img_pil = Image.fromarray((img_display * 255).astype(np.uint8))
            img_pil = img_pil.resize((MSK_SIZE, MSK_SIZE), Image.LANCZOS)
            row_images.append(np.array(img_pil) / 255.0)

            row_masks.append(mask.squeeze(0).numpy())

        # --- Per-row global scale: use shared vmin/vmax across all 3 images ---
        all_vals = np.concatenate([m.ravel() for m in row_anom_maps])
        row_vmin = all_vals.min()
        row_vmax = all_vals.max()

        # --- Phase 2: Plot with shared scale ---
        for group_idx in range(len(samples)):
            anom_map = row_anom_maps[group_idx]
            img_resized = row_images[group_idx]
            gt_mask = row_masks[group_idx]

            # Normalize using per-row global scale
            if row_vmax > row_vmin:
                anom_map_norm = (anom_map - row_vmin) / (row_vmax - row_vmin)
            else:
                anom_map_norm = np.zeros_like(anom_map)

            # Column indices
            base_col = group_idx * n_subcols

            # --- Input image ---
            ax = axes[row_idx, base_col + 0]
            ax.imshow(img_resized)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Row label on leftmost column
            if group_idx == 0:
                display_name = class_name.replace('_', ' ')
                ax.set_ylabel(display_name, fontsize=12, fontweight='bold', labelpad=8)

            # --- GT mask ---
            ax = axes[row_idx, base_col + 1]
            ax.imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # --- Heatmap overlay ---
            ax = axes[row_idx, base_col + 2]
            heatmap_colored = plt.cm.jet(anom_map_norm)[:, :, :3]
            overlay = 0.5 * img_resized + 0.5 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Add subcol titles on top row
    for group_idx in range(n_sample_groups):
        base_col = group_idx * n_subcols
        for sub_idx, sub_title in enumerate(subcol_titles):
            col = base_col + sub_idx
            axes[0, col].set_title(sub_title, fontsize=10, pad=4)

    # Add group-level titles above subcol titles
    fig.canvas.draw()  # needed to get positions
    for group_idx, gtitle in enumerate(group_titles):
        base_col = group_idx * n_subcols
        pos_left = axes[0, base_col].get_position()
        pos_right = axes[0, base_col + n_subcols - 1].get_position()
        x_center = (pos_left.x0 + pos_right.x1) / 2
        y_top = pos_left.y1 + 0.055
        fig.text(x_center, y_top, gtitle, ha='center', va='bottom',
                 fontsize=13, fontweight='bold')

    return fig


def create_figure_2col(all_data, feature_extractor, nf_model, pos_embed_generator, device):
    """
    Create 2-macro-column figure layout (for ECCV single-column width):
        Left macro-column:  metal_nut, bottle  (Normal + Anomaly 1)
        Right macro-column: pill, carpet       (Normal + Anomaly 1)
    Each macro-column has 6 sub-columns: Input|GT|Ours × 2 groups.
    """
    import matplotlib.gridspec as gridspec

    n_subcols = 3  # Input, GT, Ours per group
    n_groups = 2   # Normal, Anomaly 1
    cols_per_macro = n_groups * n_subcols  # 6
    n_macro_cols = 2
    n_rows_per_macro = 2

    left_classes = ['metal_nut', 'bottle']
    right_classes = ['pill', 'carpet']
    macro_classes = [left_classes, right_classes]

    subcol_titles = ['Input', 'GT', 'Ours']
    group_titles = ['Normal', 'Anomaly 1']

    fig_width = n_macro_cols * cols_per_macro * 1.5 + 1.0
    fig_height = n_rows_per_macro * 2.0 + 0.8

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = gridspec.GridSpec(1, n_macro_cols, figure=fig, wspace=0.08)

    # Store axes for title placement: axes_store[macro_idx][row_idx][col_idx]
    axes_store = {}

    for macro_idx, classes in enumerate(macro_classes):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows_per_macro, cols_per_macro,
            subplot_spec=outer_gs[macro_idx],
            wspace=0.02, hspace=0.15,
        )
        axes_store[macro_idx] = {}

        for row_idx, class_name in enumerate(classes):
            task_id = VIS_CLASSES[class_name]
            normals, anomalies = all_data[class_name]
            samples = normals[:1] + anomalies[:1]
            axes_store[macro_idx][row_idx] = {}

            # Compute anomaly maps
            row_anom_maps, row_images, row_masks = [], [], []
            for (image, mask, label, img_type, filename) in samples:
                img_batch = image.unsqueeze(0)
                anom_maps, _ = compute_anomaly_map(
                    img_batch, feature_extractor, nf_model, pos_embed_generator, task_id, device
                )
                row_anom_maps.append(anom_maps[0])
                img_display = denormalize_image(image)
                img_pil = Image.fromarray((img_display * 255).astype(np.uint8))
                img_pil = img_pil.resize((MSK_SIZE, MSK_SIZE), Image.LANCZOS)
                row_images.append(np.array(img_pil) / 255.0)
                row_masks.append(mask.squeeze(0).numpy())

            all_vals = np.concatenate([m.ravel() for m in row_anom_maps])
            row_vmin, row_vmax = all_vals.min(), all_vals.max()

            for group_idx in range(len(samples)):
                anom_map = row_anom_maps[group_idx]
                img_resized = row_images[group_idx]
                gt_mask = row_masks[group_idx]

                if row_vmax > row_vmin:
                    anom_map_norm = (anom_map - row_vmin) / (row_vmax - row_vmin)
                else:
                    anom_map_norm = np.zeros_like(anom_map)

                base_col = group_idx * n_subcols

                # Input
                ax = fig.add_subplot(inner_gs[row_idx, base_col + 0])
                ax.imshow(img_resized)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if group_idx == 0:
                    ax.set_ylabel(class_name.replace('_', ' '),
                                  fontsize=12, fontweight='bold', labelpad=8)
                axes_store[macro_idx][row_idx][base_col + 0] = ax

                # GT
                ax = fig.add_subplot(inner_gs[row_idx, base_col + 1])
                ax.imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                axes_store[macro_idx][row_idx][base_col + 1] = ax

                # Heatmap
                ax = fig.add_subplot(inner_gs[row_idx, base_col + 2])
                heatmap_colored = plt.cm.jet(anom_map_norm)[:, :, :3]
                overlay = np.clip(0.5 * img_resized + 0.5 * heatmap_colored, 0, 1)
                ax.imshow(overlay)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                axes_store[macro_idx][row_idx][base_col + 2] = ax

    # Subcol titles on top row
    for macro_idx in range(n_macro_cols):
        for group_idx in range(n_groups):
            base_col = group_idx * n_subcols
            for sub_idx, sub_title in enumerate(subcol_titles):
                axes_store[macro_idx][0][base_col + sub_idx].set_title(
                    sub_title, fontsize=10, pad=4)

    # Group-level titles
    fig.canvas.draw()
    for macro_idx in range(n_macro_cols):
        for group_idx, gtitle in enumerate(group_titles):
            base_col = group_idx * n_subcols
            pos_left = axes_store[macro_idx][0][base_col].get_position()
            pos_right = axes_store[macro_idx][0][base_col + n_subcols - 1].get_position()
            x_center = (pos_left.x0 + pos_right.x1) / 2
            y_top = pos_left.y1 + 0.055
            fig.text(x_center, y_top, gtitle, ha='center', va='bottom',
                     fontsize=13, fontweight='bold')

    return fig


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: %s" % device)

    # Build model
    feature_extractor, nf_model, pos_embed_generator = build_model(device)

    # Collect selected images for each class
    print("\nSelecting representative images...")
    all_data = {}
    for class_name in VIS_ORDER:
        sys.stdout.write("  %s: " % class_name)
        normals, anomalies = select_images(class_name, N_NORMAL, N_ANOMALY)
        all_data[class_name] = (normals, anomalies)
        anomaly_types = [a[3] for a in anomalies]
        print("%d normal, %d anomaly (types: %s)" % (len(normals), len(anomalies), anomaly_types))

    # Generate figure
    print("\nGenerating anomaly map figure...")
    fig = create_figure(all_data, feature_extractor, nf_model, pos_embed_generator, device)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_path = os.path.join(OUTPUT_DIR, 'anomaly_maps.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'anomaly_maps.png')

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nFigure saved to:")
    print("  PDF: %s" % pdf_path)
    print("  PNG: %s" % png_path)

    # Generate 2-column version
    print("\nGenerating 2-column anomaly map figure...")
    fig_2col = create_figure_2col(all_data, feature_extractor, nf_model, pos_embed_generator, device)

    pdf_2col = os.path.join(OUTPUT_DIR, 'anomaly_maps_2col.pdf')
    jpg_2col = os.path.join(OUTPUT_DIR, 'anomaly_maps_2col.jpg')

    fig_2col.savefig(pdf_2col, dpi=300, bbox_inches='tight')
    fig_2col.savefig(jpg_2col, dpi=300, bbox_inches='tight')
    plt.close(fig_2col)

    print("  PDF: %s" % pdf_2col)
    print("  JPG: %s" % jpg_2col)


if __name__ == '__main__':
    main()
