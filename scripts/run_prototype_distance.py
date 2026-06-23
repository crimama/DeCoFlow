"""
Prototype Distance Matrix Visualization.

Loads the router prototypes from a checkpoint and computes pairwise
symmetric Mahalanobis distances between all task prototypes.
Generates a heatmap showing inter-class feature separation.

Usage:
    python scripts/run_prototype_distance.py
    python scripts/run_prototype_distance.py --checkpoint_path logs/V48_01_H04_highres_clean/checkpoints/latest/
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "V48_01_H04_highres_clean", "checkpoints", "latest",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Paper_works", "figures",
)

MVTEC_CLASS_NAMES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype distance matrix visualization"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to checkpoint directory containing router.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for tensor operations (cpu recommended for this script)",
    )
    return parser.parse_args()


def load_prototypes(checkpoint_path, device="cpu"):
    """
    Load router prototypes from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory containing router.pth
        device: Device to load tensors onto

    Returns:
        prototypes: dict mapping task_id (int) -> {
            'task_classes': list[str],
            'mean': Tensor (D,),
            'covariance': Tensor (D, D),
            'precision': Tensor (D, D),
            'n_samples': int,
        }
    """
    router_path = os.path.join(checkpoint_path, "router.pth")
    if not os.path.exists(router_path):
        raise FileNotFoundError(f"router.pth not found at {router_path}")

    raw = torch.load(router_path, map_location=device, weights_only=False)

    prototypes = {}
    for tid_str, pdata in raw.items():
        tid = int(tid_str)
        prototypes[tid] = {
            "task_classes": pdata["task_classes"],
            "mean": pdata["mean"].to(device).double(),
            "covariance": pdata["covariance"].to(device).double(),
            "precision": pdata["precision"].to(device).double(),
            "n_samples": pdata["n_samples"],
        }

    return prototypes


def compute_symmetric_mahalanobis(mu_i, cov_i, mu_j, cov_j):
    """
    Compute symmetric Mahalanobis distance between two prototypes.

    d(i,j) = sqrt( (mu_i - mu_j)^T  Sigma_avg^{-1}  (mu_i - mu_j) )
    where Sigma_avg = (Sigma_i + Sigma_j) / 2

    Args:
        mu_i: (D,) mean of prototype i
        cov_i: (D, D) covariance of prototype i
        mu_j: (D,) mean of prototype j
        cov_j: (D, D) covariance of prototype j

    Returns:
        Scalar distance value
    """
    delta = mu_i - mu_j  # (D,)
    cov_avg = (cov_i + cov_j) / 2.0  # (D, D)

    # Regularize for numerical stability
    reg = 1e-6 * torch.eye(cov_avg.shape[0], device=cov_avg.device, dtype=cov_avg.dtype)
    cov_avg = cov_avg + reg

    # Solve cov_avg @ x = delta  instead of explicit inverse for stability
    # Then distance = sqrt(delta^T x)
    try:
        L = torch.linalg.cholesky(cov_avg)
        x = torch.cholesky_solve(delta.unsqueeze(1), L).squeeze(1)  # (D,)
    except RuntimeError:
        # Fallback: pseudo-inverse
        prec_avg = torch.linalg.pinv(cov_avg)
        x = prec_avg @ delta

    dist_sq = torch.dot(delta, x)
    dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))
    return dist.item()


def build_distance_matrix(prototypes):
    """
    Build the full pairwise symmetric Mahalanobis distance matrix.

    Args:
        prototypes: dict from load_prototypes()

    Returns:
        dist_matrix: np.ndarray of shape (N, N)
        class_names: list of class names in task-id order
    """
    task_ids = sorted(prototypes.keys())
    n = len(task_ids)

    class_names = []
    for tid in task_ids:
        class_names.append(prototypes[tid]["task_classes"][0])

    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = task_ids[i], task_ids[j]
            d = compute_symmetric_mahalanobis(
                prototypes[ti]["mean"], prototypes[ti]["covariance"],
                prototypes[tj]["mean"], prototypes[tj]["covariance"],
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix, class_names


def plot_heatmap(dist_matrix, class_names, output_dir):
    """
    Plot and save the prototype distance heatmap.

    Args:
        dist_matrix: (N, N) distance matrix
        class_names: list of class name strings
        output_dir: directory to save PDF and PNG
    """
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(9, 8))

    # Use a perceptually uniform colormap
    im = ax.imshow(dist_matrix, cmap="viridis", aspect="equal")

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Move x-axis ticks to top as well (for readability)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Annotate cells with distance values
    for i in range(n):
        for j in range(n):
            val = dist_matrix[i, j]
            # Choose text color for readability
            text_color = "white" if val < (dist_matrix.max() + dist_matrix.min()) / 2 else "black"
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                fontsize=6.5, color=text_color,
            )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Symmetric Mahalanobis Distance", fontsize=10)

    ax.set_title("Prototype Distance Matrix (15 MVTec Classes)", fontsize=12, pad=12)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "prototype_distance_matrix.pdf")
    png_path = os.path.join(output_dir, "prototype_distance_matrix.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    return pdf_path, png_path


def print_summary(dist_matrix, class_names):
    """Print distance statistics to stdout."""
    n = len(class_names)

    # Extract upper-triangle values (excluding diagonal)
    upper = dist_matrix[np.triu_indices(n, k=1)]

    print("\n=== Prototype Distance Summary ===")
    print(f"Number of classes: {n}")
    print(f"Feature dimension: inferred from checkpoint")
    print(f"Min distance:  {upper.min():.2f}")
    print(f"Max distance:  {upper.max():.2f}")
    print(f"Mean distance: {upper.mean():.2f}")
    print(f"Std distance:  {upper.std():.2f}")

    # Closest pair
    min_idx = np.argmin(upper)
    i_min, j_min = np.triu_indices(n, k=1)
    print(f"\nClosest pair:  {class_names[i_min[min_idx]]} <-> "
          f"{class_names[j_min[min_idx]]} (d={upper.min():.2f})")

    # Farthest pair
    max_idx = np.argmax(upper)
    print(f"Farthest pair: {class_names[i_min[max_idx]]} <-> "
          f"{class_names[j_min[max_idx]]} (d={upper.max():.2f})")

    # Per-class: average distance to all others
    print("\nPer-class mean distance to others:")
    for i in range(n):
        others = [dist_matrix[i, j] for j in range(n) if j != i]
        print(f"  {class_names[i]:>12s}: {np.mean(others):.2f}")


def main():
    args = parse_args()

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output dir: {args.output_dir}")

    # Load prototypes
    prototypes = load_prototypes(args.checkpoint_path, device=args.device)
    print(f"Loaded {len(prototypes)} prototypes")

    feat_dim = prototypes[0]["mean"].shape[0]
    print(f"Feature dimension: {feat_dim}")

    # Build distance matrix
    dist_matrix, class_names = build_distance_matrix(prototypes)

    # Print summary statistics
    print_summary(dist_matrix, class_names)

    # Plot and save
    plot_heatmap(dist_matrix, class_names, args.output_dir)

    # Also save raw data as JSON for later use
    data_path = os.path.join(args.output_dir, "prototype_distance_data.json")
    data_out = {
        "class_names": class_names,
        "distance_matrix": dist_matrix.tolist(),
        "stats": {
            "min": float(dist_matrix[np.triu_indices(len(class_names), k=1)].min()),
            "max": float(dist_matrix[np.triu_indices(len(class_names), k=1)].max()),
            "mean": float(dist_matrix[np.triu_indices(len(class_names), k=1)].mean()),
            "std": float(dist_matrix[np.triu_indices(len(class_names), k=1)].std()),
        },
    }
    with open(data_path, "w") as f:
        json.dump(data_out, f, indent=2)
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
