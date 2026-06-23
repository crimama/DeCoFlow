"""
Flow Diagnostics for scale(s) analysis.

This module provides tools to analyze whether the Flow's scale(s) is:
- Alive: responds meaningfully to anomalies
- Dead: no response regardless of input
- Noisy: responds randomly, even on normal images

Key diagnostics:
1. logdet std comparison (normal vs anomaly)
2. logdet vs ||z|| correlation (meaningful deformation check)
3. Spatial heatmaps (where does scale respond?)
4. z-term vs logdet-term contribution analysis
"""

import os
import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Diagnostics plots will be disabled.")


class FlowDiagnostics:
    """
    Diagnostics for analyzing Flow scale(s) behavior.

    Collects z and logdet_patch during evaluation, then generates
    diagnostic plots to understand if scale(s) is working properly.

    Usage:
        diagnostics = FlowDiagnostics(save_dir="./logs/diagnostics")

        # During evaluation
        for batch in test_loader:
            z, logdet_patch = model.forward(x)
            diagnostics.collect(z, logdet_patch, is_anomaly=labels)

        # After evaluation
        diagnostics.analyze_and_save()
    """

    def __init__(self, save_dir: str, max_samples: int = 1000):
        """
        Initialize diagnostics collector.

        Args:
            save_dir: Directory to save diagnostic plots
            max_samples: Maximum samples to collect per category (normal/anomaly)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples

        # Storage for collected data
        self.reset()

    def reset(self):
        """Reset collected data."""
        self.data = {
            'normal': {
                'z': [],
                'logdet_patch': [],
                'images': [],
                'masks': []
            },
            'anomaly': {
                'z': [],
                'logdet_patch': [],
                'images': [],
                'masks': []
            }
        }
        self.class_names = []

    def collect(self, z: torch.Tensor, logdet_patch: torch.Tensor,
                is_anomaly: torch.Tensor, images: torch.Tensor = None,
                masks: torch.Tensor = None, class_name: str = None):
        """
        Collect z and logdet_patch for analysis.

        Args:
            z: Latent tensor (B, H, W, D)
            logdet_patch: Patch-wise log|det J| (B, H, W)
            is_anomaly: Binary labels (B,) - 1 for anomaly, 0 for normal
            images: Original images (B, C, H, W) - optional, for visualization
            masks: GT masks (B, H, W) - optional, for anomaly localization
            class_name: Class name for grouping
        """
        z = z.detach().cpu()
        logdet_patch = logdet_patch.detach().cpu()
        is_anomaly = is_anomaly.detach().cpu()

        if class_name and class_name not in self.class_names:
            self.class_names.append(class_name)

        for i in range(z.shape[0]):
            category = 'anomaly' if is_anomaly[i].item() else 'normal'

            if len(self.data[category]['z']) < self.max_samples:
                self.data[category]['z'].append(z[i])
                self.data[category]['logdet_patch'].append(logdet_patch[i])

                if images is not None:
                    self.data[category]['images'].append(images[i].detach().cpu())
                if masks is not None:
                    self.data[category]['masks'].append(masks[i].detach().cpu())

    def _stack_data(self, category: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack collected data into tensors."""
        if not self.data[category]['z']:
            return None, None
        z = torch.stack(self.data[category]['z'])
        logdet = torch.stack(self.data[category]['logdet_patch'])
        return z, logdet

    def compute_statistics(self) -> Dict:
        """
        Compute diagnostic statistics.

        Returns:
            Dictionary with computed statistics
        """
        stats = {}

        for category in ['normal', 'anomaly']:
            z, logdet = self._stack_data(category)
            if z is None:
                continue

            # z norm per patch
            z_norm = torch.norm(z, dim=-1)  # (N, H, W)

            # log p(z) per patch
            D = z.shape[-1]
            log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

            stats[category] = {
                'n_samples': z.shape[0],
                # Per-image std of logdet
                'logdet_std_per_image': logdet.std(dim=(1, 2)),  # (N,)
                # Per-image mean of logdet
                'logdet_mean_per_image': logdet.mean(dim=(1, 2)),  # (N,)
                # Per-image max abs logdet
                'logdet_max_per_image': logdet.abs().amax(dim=(1, 2)),  # (N,)
                # Variance decomposition
                'var_logdet_per_image': logdet.var(dim=(1, 2)),  # (N,)
                'var_log_pz_per_image': log_pz_patch.var(dim=(1, 2)),  # (N,)
                # Flattened for correlation
                'z_norm_flat': z_norm.flatten(),
                'logdet_flat': logdet.flatten(),
                # Raw data for heatmaps
                'logdet_patches': logdet,
                'z_norm_patches': z_norm,
            }

        return stats

    def analyze_and_save(self, epoch: int = None, task_id: int = None):
        """
        Generate and save all diagnostic plots.

        Args:
            epoch: Current epoch (for filename)
            task_id: Current task ID (for filename)
        """
        if not HAS_MATPLOTLIB:
            print("Skipping diagnostics plots (matplotlib not available)")
            return

        stats = self.compute_statistics()

        if not stats:
            print("No data collected for diagnostics")
            return

        suffix = ""
        if task_id is not None:
            suffix += f"_task{task_id}"
        if epoch is not None:
            suffix += f"_epoch{epoch}"

        # 1. Logdet std comparison (normal vs anomaly)
        self._plot_logdet_std_comparison(stats, suffix)

        # 2. Logdet vs ||z|| correlation
        self._plot_logdet_z_correlation(stats, suffix)

        # 3. Variance contribution (z vs logdet)
        self._plot_variance_contribution(stats, suffix)

        # 4. Max logdet distribution
        self._plot_max_logdet_distribution(stats, suffix)

        # 5. Spatial heatmaps (sample images)
        self._plot_spatial_heatmaps(stats, suffix)

        # 6. Summary statistics to text file
        self._save_summary_stats(stats, suffix)

        print(f"Diagnostics saved to {self.save_dir}")

    def _plot_logdet_std_comparison(self, stats: Dict, suffix: str):
        """Plot 1: logdet std comparison between normal and anomaly."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax = axes[0]
        for category, color in [('normal', 'blue'), ('anomaly', 'red')]:
            if category in stats:
                data = stats[category]['logdet_std_per_image'].numpy()
                ax.hist(data, bins=30, alpha=0.6, label=f'{category} (n={len(data)})',
                       color=color, density=True)
        ax.set_xlabel('logdet std per image')
        ax.set_ylabel('Density')
        ax.set_title('logdet_patch Std Distribution\n(Good: normal=low, anomaly=high)')
        ax.legend()

        # Boxplot
        ax = axes[1]
        data_to_plot = []
        labels = []
        for category in ['normal', 'anomaly']:
            if category in stats:
                data_to_plot.append(stats[category]['logdet_std_per_image'].numpy())
                labels.append(f"{category}\n(n={stats[category]['n_samples']})")

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
        ax.set_ylabel('logdet std per image')
        ax.set_title('logdet_patch Std Boxplot')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'1_logdet_std_comparison{suffix}.png', dpi=150)
        plt.close()

    def _plot_logdet_z_correlation(self, stats: Dict, suffix: str):
        """Plot 2: logdet vs ||z|| scatter plot."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, category in enumerate(['normal', 'anomaly']):
            if category not in stats:
                continue
            ax = axes[idx]

            z_norm = stats[category]['z_norm_flat'].numpy()
            logdet = stats[category]['logdet_flat'].numpy()

            # Subsample for visualization
            n_points = min(10000, len(z_norm))
            indices = np.random.choice(len(z_norm), n_points, replace=False)

            ax.scatter(z_norm[indices], logdet[indices], alpha=0.3, s=1,
                      color='blue' if category == 'normal' else 'red')

            # Correlation coefficient
            corr = np.corrcoef(z_norm, logdet)[0, 1]
            ax.set_xlabel('||z|| (patch norm)')
            ax.set_ylabel('logdet_patch')
            ax.set_title(f'{category.upper()}: ||z|| vs logdet\n'
                        f'Correlation: {corr:.4f}\n'
                        f'(Good: positive correlation for anomaly)')

            # Add trend line
            z = np.polyfit(z_norm[indices], logdet[indices], 1)
            p = np.poly1d(z)
            x_line = np.linspace(z_norm.min(), z_norm.max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.5, label=f'trend (slope={z[0]:.4f})')
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / f'2_logdet_z_correlation{suffix}.png', dpi=150)
        plt.close()

    def _plot_variance_contribution(self, stats: Dict, suffix: str):
        """Plot 3: Variance contribution of z-term vs logdet-term."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        data = []
        labels = []

        for category in ['normal', 'anomaly']:
            if category not in stats:
                continue
            var_logdet = stats[category]['var_logdet_per_image'].numpy()
            var_log_pz = stats[category]['var_log_pz_per_image'].numpy()

            data.append(var_log_pz)
            labels.append(f'{category}\nlog_pz var')
            data.append(var_logdet)
            labels.append(f'{category}\nlogdet var')

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = ['lightblue', 'skyblue', 'lightcoral', 'salmon']
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])

        ax.set_ylabel('Variance (per image)')
        ax.set_title('Variance Contribution: log_pz vs logdet\n'
                    '(Good: similar variance, or logdet ↑ for anomaly only)')
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'3_variance_contribution{suffix}.png', dpi=150)
        plt.close()

    def _plot_max_logdet_distribution(self, stats: Dict, suffix: str):
        """Plot 4: Max |logdet| distribution."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for category, color in [('normal', 'blue'), ('anomaly', 'red')]:
            if category in stats:
                data = stats[category]['logdet_max_per_image'].numpy()
                ax.hist(data, bins=30, alpha=0.6, label=f'{category} (n={len(data)})',
                       color=color, density=True)

        ax.set_xlabel('max |logdet| per image')
        ax.set_ylabel('Density')
        ax.set_title('Max |logdet_patch| Distribution\n'
                    '(Good: normal << anomaly)')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / f'4_max_logdet_distribution{suffix}.png', dpi=150)
        plt.close()

    def _plot_spatial_heatmaps(self, stats: Dict, suffix: str, n_samples: int = 4):
        """Plot 5: Spatial heatmaps of logdet and ||z||."""
        for category in ['normal', 'anomaly']:
            if category not in stats:
                continue

            logdet_patches = stats[category]['logdet_patches']
            z_norm_patches = stats[category]['z_norm_patches']
            n = min(n_samples, logdet_patches.shape[0])

            fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
            if n == 1:
                axes = axes.reshape(1, -1)

            for i in range(n):
                # Original image (if available)
                if self.data[category]['images']:
                    img = self.data[category]['images'][i]
                    if img.shape[0] == 3:
                        img = img.permute(1, 2, 0).numpy()
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    axes[i, 0].imshow(img)
                else:
                    axes[i, 0].text(0.5, 0.5, 'No image', ha='center', va='center')
                axes[i, 0].set_title('Original Image')
                axes[i, 0].axis('off')

                # GT mask (if available)
                if self.data[category]['masks']:
                    mask = self.data[category]['masks'][i].numpy()
                    # Squeeze channel dimension if present (e.g., (1, H, W) -> (H, W))
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)
                    axes[i, 1].imshow(mask, cmap='gray')
                else:
                    axes[i, 1].text(0.5, 0.5, 'No mask', ha='center', va='center')
                axes[i, 1].set_title('GT Mask')
                axes[i, 1].axis('off')

                # logdet heatmap
                logdet = logdet_patches[i].numpy()
                im = axes[i, 2].imshow(logdet, cmap='hot')
                axes[i, 2].set_title(f'logdet_patch\n(std={logdet.std():.4f})')
                axes[i, 2].axis('off')
                plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

                # ||z|| heatmap
                z_norm = z_norm_patches[i].numpy()
                im = axes[i, 3].imshow(z_norm, cmap='hot')
                axes[i, 3].set_title(f'||z|| (patch norm)\n(mean={z_norm.mean():.4f})')
                axes[i, 3].axis('off')
                plt.colorbar(im, ax=axes[i, 3], fraction=0.046)

            plt.suptitle(f'{category.upper()} Samples - Spatial Analysis', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.save_dir / f'5_spatial_heatmap_{category}{suffix}.png', dpi=150)
            plt.close()

    def _save_summary_stats(self, stats: Dict, suffix: str):
        """Save summary statistics to text file."""
        lines = []
        lines.append("=" * 60)
        lines.append("Flow Diagnostics Summary")
        lines.append("=" * 60)
        lines.append("")

        for category in ['normal', 'anomaly']:
            if category not in stats:
                continue
            s = stats[category]
            lines.append(f"[{category.upper()}] (n={s['n_samples']})")
            lines.append("-" * 40)

            # logdet std
            logdet_std = s['logdet_std_per_image']
            lines.append(f"  logdet_std per image:")
            lines.append(f"    mean: {logdet_std.mean():.6f}")
            lines.append(f"    std:  {logdet_std.std():.6f}")
            lines.append(f"    min:  {logdet_std.min():.6f}")
            lines.append(f"    max:  {logdet_std.max():.6f}")

            # max logdet
            logdet_max = s['logdet_max_per_image']
            lines.append(f"  max |logdet| per image:")
            lines.append(f"    mean: {logdet_max.mean():.6f}")
            lines.append(f"    std:  {logdet_max.std():.6f}")

            # Variance contribution
            var_logdet = s['var_logdet_per_image']
            var_log_pz = s['var_log_pz_per_image']
            lines.append(f"  Variance contribution:")
            lines.append(f"    logdet var (mean): {var_logdet.mean():.6f}")
            lines.append(f"    log_pz var (mean): {var_log_pz.mean():.6f}")
            lines.append(f"    ratio (logdet/log_pz): {(var_logdet.mean() / (var_log_pz.mean() + 1e-8)):.4f}")

            # Correlation
            z_norm = s['z_norm_flat'].numpy()
            logdet = s['logdet_flat'].numpy()
            corr = np.corrcoef(z_norm, logdet)[0, 1]
            lines.append(f"  ||z|| vs logdet correlation: {corr:.6f}")

            lines.append("")

        # Interpretation guide
        lines.append("=" * 60)
        lines.append("INTERPRETATION GUIDE")
        lines.append("=" * 60)
        lines.append("")
        lines.append("1. logdet_std comparison:")
        lines.append("   - Good: normal_std << anomaly_std")
        lines.append("   - Bad: normal_std ≈ anomaly_std (scale is dead)")
        lines.append("   - Bad: normal_std is large (scale is noisy)")
        lines.append("")
        lines.append("2. ||z|| vs logdet correlation:")
        lines.append("   - Good: positive correlation (esp. for anomaly)")
        lines.append("   - Bad: no correlation (scale is random)")
        lines.append("")
        lines.append("3. Variance contribution:")
        lines.append("   - Good: similar, or logdet ↑ for anomaly only")
        lines.append("   - Bad: logdet >> log_pz everywhere (noisy)")
        lines.append("   - Bad: logdet << log_pz everywhere (dead)")
        lines.append("")

        # Suggested actions
        if 'normal' in stats and 'anomaly' in stats:
            normal_std = stats['normal']['logdet_std_per_image'].mean().item()
            anomaly_std = stats['anomaly']['logdet_std_per_image'].mean().item()
            ratio = anomaly_std / (normal_std + 1e-8)

            lines.append("=" * 60)
            lines.append("DIAGNOSIS")
            lines.append("=" * 60)
            lines.append(f"  logdet_std ratio (anomaly/normal): {ratio:.4f}")

            if ratio < 1.2:
                lines.append("  Status: scale(s) appears DEAD or nearly uniform")
                lines.append("  Suggested: Add spatial context (3x3 conv)")
            elif normal_std > 1.0:
                lines.append("  Status: scale(s) appears NOISY on normal images")
                lines.append("  Suggested: Regularize scale or change aggregation")
            else:
                lines.append("  Status: scale(s) appears HEALTHY")

        # Save to file
        with open(self.save_dir / f'diagnostics_summary{suffix}.txt', 'w') as f:
            f.write('\n'.join(lines))

        # Also print to console
        print('\n'.join(lines))


def run_diagnostics_on_model(trainer, args, task_id: int = None,
                             save_dir: str = None, max_samples: int = 500):
    """
    Convenience function to run diagnostics on a trained model.

    Args:
        trainer: DeCoFlowContinualTrainer instance
        args: Training arguments (with data_path, etc.)
        task_id: Task to evaluate (None = all tasks)
        save_dir: Directory to save diagnostics
        max_samples: Max samples per category
    """
    from decoflow.data import create_task_dataset
    from torch.utils.data import DataLoader

    if save_dir is None:
        save_dir = "./diagnostics"

    diagnostics = FlowDiagnostics(save_dir=save_dir, max_samples=max_samples)

    # Get task classes
    if task_id is not None:
        task_classes = trainer.task_classes.get(task_id, [])
    else:
        task_classes = list(trainer.task_classes.values())
        task_classes = [c for sublist in task_classes for c in sublist]

    trainer.nf_model.eval()
    trainer.vit_extractor.eval()

    device = trainer.device

    for cls_name in task_classes:
        print(f"Collecting diagnostics for: {cls_name}")

        # Create test dataset
        test_args = type(args)()
        for k, v in vars(args).items():
            setattr(test_args, k, v)
        test_args.class_to_idx = {cls_name: 0}
        test_args.n_classes = 1

        try:
            test_dataset = create_task_dataset(test_args, [cls_name], {cls_name: 0}, train=False)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        except Exception as e:
            print(f"Failed to create dataset for {cls_name}: {e}")
            continue

        # Set active task
        if task_id is not None:
            trainer.nf_model.set_active_task(task_id)

        with torch.no_grad():
            for batch in test_loader:
                images, labels, masks, _, _ = batch
                images = images.to(device)

                # Extract features
                patch_embeddings, spatial_shape = trainer.vit_extractor(
                    images, return_spatial_shape=True
                )

                if trainer.use_pos_embedding:
                    patch_embeddings_with_pos = trainer.pos_embed_generator(
                        spatial_shape, patch_embeddings
                    )
                else:
                    B = patch_embeddings.shape[0]
                    H, W = spatial_shape
                    patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)

                # Forward through NF
                z, logdet_patch = trainer.nf_model.forward(
                    patch_embeddings_with_pos, reverse=False
                )

                # Collect
                is_anomaly = (labels > 0).long()
                diagnostics.collect(
                    z=z,
                    logdet_patch=logdet_patch,
                    is_anomaly=is_anomaly,
                    images=images,
                    masks=masks,
                    class_name=cls_name
                )

    # Generate plots
    diagnostics.analyze_and_save(task_id=task_id)

    return diagnostics
