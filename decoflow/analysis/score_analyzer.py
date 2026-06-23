"""
Score Distribution Analyzer for Anomaly Detection.

This module analyzes the distribution of anomaly scores,
comparing normal vs anomaly patches/images to understand
how well the model separates them.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt


class ScoreDistributionAnalyzer:
    """
    Analyzer for anomaly score distributions.

    Key analyses:
    1. Score distribution visualization
    2. Separation metrics (Fisher ratio, Cohen's d)
    3. Threshold analysis
    4. Per-class breakdown
    """

    def __init__(self,
                 trainer,
                 device: str = 'cuda'):
        """
        Initialize score analyzer.

        Args:
            trainer: DeCoFlowContinualTrainer instance
            device: Device to run analysis on
        """
        self.trainer = trainer
        self.device = device

    def collect_scores(self,
                       test_loader: DataLoader,
                       task_id: int) -> Dict:
        """
        Collect anomaly scores for all test samples.

        Args:
            test_loader: Test data loader with masks
            task_id: Task ID

        Returns:
            Dict with normal and anomaly scores at patch and image level
        """
        self.trainer.nf_model.eval()
        self.trainer.vit_extractor.eval()

        patch_normal_scores = []
        patch_anomaly_scores = []
        image_normal_scores = []
        image_anomaly_scores = []

        with torch.no_grad():
            for batch in test_loader:
                images, labels, masks = batch[0], batch[1], batch[2]
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.numpy()

                # Get scores
                patch_scores, image_scores, _ = self.trainer.inference(
                    images, task_id=task_id
                )
                B, H, W = patch_scores.shape

                # Resize masks
                masks_resized = F.interpolate(
                    masks.float(),
                    size=(H, W),
                    mode='nearest'
                ).squeeze(1)

                # Patch-level
                for b in range(B):
                    flat_scores = patch_scores[b].reshape(-1).cpu().numpy()
                    flat_mask = masks_resized[b].reshape(-1).cpu().numpy()

                    anomaly_mask = flat_mask > 0.5
                    normal_mask = ~anomaly_mask

                    if normal_mask.sum() > 0:
                        patch_normal_scores.extend(flat_scores[normal_mask].tolist())
                    if anomaly_mask.sum() > 0:
                        patch_anomaly_scores.extend(flat_scores[anomaly_mask].tolist())

                # Image-level
                for b in range(B):
                    score = image_scores[b].item()
                    if labels[b] == 0:  # Normal
                        image_normal_scores.append(score)
                    else:  # Anomaly
                        image_anomaly_scores.append(score)

        return {
            'patch_normal': np.array(patch_normal_scores),
            'patch_anomaly': np.array(patch_anomaly_scores),
            'image_normal': np.array(image_normal_scores),
            'image_anomaly': np.array(image_anomaly_scores),
        }

    def compute_separation_metrics(self,
                                    normal_scores: np.ndarray,
                                    anomaly_scores: np.ndarray) -> Dict:
        """
        Compute separation metrics between normal and anomaly scores.

        Args:
            normal_scores: Scores for normal samples
            anomaly_scores: Scores for anomaly samples

        Returns:
            Dict with separation metrics
        """
        if len(normal_scores) == 0 or len(anomaly_scores) == 0:
            return {'error': 'empty_scores'}

        # Basic statistics
        normal_mean = normal_scores.mean()
        normal_std = normal_scores.std()
        anomaly_mean = anomaly_scores.mean()
        anomaly_std = anomaly_scores.std()

        # Fisher's Discriminant Ratio
        mean_diff = anomaly_mean - normal_mean
        pooled_var = (normal_std ** 2 + anomaly_std ** 2) / 2
        fisher_ratio = (mean_diff ** 2) / (pooled_var + 1e-10)

        # Cohen's d (effect size)
        pooled_std = np.sqrt(pooled_var)
        cohens_d = mean_diff / (pooled_std + 1e-10)

        # Distribution overlap (histogram intersection)
        min_val = min(normal_scores.min(), anomaly_scores.min())
        max_val = max(normal_scores.max(), anomaly_scores.max())
        bins = np.linspace(min_val, max_val, 100)

        hist_normal, _ = np.histogram(normal_scores, bins=bins, density=True)
        hist_anomaly, _ = np.histogram(anomaly_scores, bins=bins, density=True)

        bin_width = bins[1] - bins[0]
        overlap = np.minimum(hist_normal, hist_anomaly).sum() * bin_width

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(normal_scores, anomaly_scores)

        # Mann-Whitney U test
        mw_stat, mw_pvalue = stats.mannwhitneyu(anomaly_scores, normal_scores,
                                                  alternative='greater')

        # AUROC
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([np.zeros(len(normal_scores)),
                                     np.ones(len(anomaly_scores))])
        auroc = roc_auc_score(all_labels, all_scores)

        # Find optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]

        return {
            'normal_mean': normal_mean,
            'normal_std': normal_std,
            'anomaly_mean': anomaly_mean,
            'anomaly_std': anomaly_std,
            'fisher_ratio': fisher_ratio,
            'cohens_d': cohens_d,
            'overlap': overlap,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mw_statistic': mw_stat,
            'mw_pvalue': mw_pvalue,
            'auroc': auroc,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': optimal_tpr,
            'optimal_fpr': optimal_fpr,
            'n_normal': len(normal_scores),
            'n_anomaly': len(anomaly_scores),
        }

    def analyze_full(self,
                      test_loader: DataLoader,
                      task_id: int,
                      save_path: Optional[str] = None) -> Dict:
        """
        Run full score distribution analysis.

        Args:
            test_loader: Test data loader
            task_id: Task ID
            save_path: Path to save visualization

        Returns:
            Dict with complete analysis results
        """
        print("\n" + "=" * 70)
        print("Score Distribution Analysis")
        print("=" * 70)

        # Collect scores
        print("  Collecting scores...")
        scores = self.collect_scores(test_loader, task_id)

        results = {}

        # Pixel-level analysis
        print("\n--- Pixel-Level Analysis ---")
        results['pixel'] = self.compute_separation_metrics(
            scores['patch_normal'],
            scores['patch_anomaly']
        )
        self._print_metrics(results['pixel'], "Pixel")

        # Image-level analysis
        print("\n--- Image-Level Analysis ---")
        results['image'] = self.compute_separation_metrics(
            scores['image_normal'],
            scores['image_anomaly']
        )
        self._print_metrics(results['image'], "Image")

        # Store raw scores for visualization
        results['raw_scores'] = scores

        if save_path:
            self._plot_distributions(scores, results, save_path)

        return results

    def _print_metrics(self, metrics: Dict, level: str):
        """Print separation metrics."""
        if 'error' in metrics:
            print(f"  Error: {metrics['error']}")
            return

        print(f"  {level} Normal:  mean={metrics['normal_mean']:.4f}, std={metrics['normal_std']:.4f}, n={metrics['n_normal']}")
        print(f"  {level} Anomaly: mean={metrics['anomaly_mean']:.4f}, std={metrics['anomaly_std']:.4f}, n={metrics['n_anomaly']}")
        print(f"\n  Separation Metrics:")
        print(f"    Fisher's Ratio: {metrics['fisher_ratio']:.4f}")
        print(f"    Cohen's d: {metrics['cohens_d']:.4f}")
        print(f"    Overlap: {metrics['overlap']:.4f}")
        print(f"    AUROC: {metrics['auroc']:.4f}")
        print(f"\n  Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"    TPR: {metrics['optimal_tpr']:.4f}, FPR: {metrics['optimal_fpr']:.4f}")

        # Interpretation
        if metrics['cohens_d'] > 0.8:
            print(f"\n  Interpretation: Large effect size - strong separation")
        elif metrics['cohens_d'] > 0.5:
            print(f"\n  Interpretation: Medium effect size - moderate separation")
        else:
            print(f"\n  Interpretation: Small effect size - weak separation")

    def _plot_distributions(self,
                             scores: Dict,
                             results: Dict,
                             save_path: str):
        """Plot score distributions."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Pixel-level histogram
        ax = axes[0, 0]
        ax.hist(scores['patch_normal'], bins=50, alpha=0.5, label='Normal', density=True, color='blue')
        ax.hist(scores['patch_anomaly'], bins=50, alpha=0.5, label='Anomaly', density=True, color='red')
        ax.axvline(results['pixel']['normal_mean'], color='blue', linestyle='--')
        ax.axvline(results['pixel']['anomaly_mean'], color='red', linestyle='--')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title(f"Pixel-Level (Cohen's d={results['pixel']['cohens_d']:.2f})")
        ax.legend()

        # Pixel-level boxplot
        ax = axes[0, 1]
        ax.boxplot([scores['patch_normal'], scores['patch_anomaly']],
                   labels=['Normal', 'Anomaly'])
        ax.set_ylabel('Anomaly Score')
        ax.set_title(f"Pixel-Level (Fisher={results['pixel']['fisher_ratio']:.2f})")

        # Pixel-level ROC
        ax = axes[0, 2]
        all_scores = np.concatenate([scores['patch_normal'], scores['patch_anomaly']])
        all_labels = np.concatenate([np.zeros(len(scores['patch_normal'])),
                                     np.ones(len(scores['patch_anomaly']))])
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        ax.plot(fpr, tpr, label=f'AUROC={results["pixel"]["auroc"]:.4f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Pixel-Level ROC Curve')
        ax.legend()

        # Image-level histogram
        ax = axes[1, 0]
        ax.hist(scores['image_normal'], bins=30, alpha=0.5, label='Normal', density=True, color='blue')
        ax.hist(scores['image_anomaly'], bins=30, alpha=0.5, label='Anomaly', density=True, color='red')
        ax.axvline(results['image']['normal_mean'], color='blue', linestyle='--')
        ax.axvline(results['image']['anomaly_mean'], color='red', linestyle='--')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title(f"Image-Level (Cohen's d={results['image']['cohens_d']:.2f})")
        ax.legend()

        # Image-level boxplot
        ax = axes[1, 1]
        ax.boxplot([scores['image_normal'], scores['image_anomaly']],
                   labels=['Normal', 'Anomaly'])
        ax.set_ylabel('Anomaly Score')
        ax.set_title(f"Image-Level (Fisher={results['image']['fisher_ratio']:.2f})")

        # Image-level ROC
        ax = axes[1, 2]
        all_scores = np.concatenate([scores['image_normal'], scores['image_anomaly']])
        all_labels = np.concatenate([np.zeros(len(scores['image_normal'])),
                                     np.ones(len(scores['image_anomaly']))])
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        ax.plot(fpr, tpr, label=f'AUROC={results["image"]["auroc"]:.4f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Image-Level ROC Curve')
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to {save_path}")

    def compare_configurations(self,
                                configs: List[Dict],
                                test_loader: DataLoader,
                                task_id: int,
                                save_path: Optional[str] = None) -> Dict:
        """
        Compare score distributions across different training configurations.

        Args:
            configs: List of config dicts with 'name' and 'trainer' keys
            test_loader: Test data loader
            task_id: Task ID
            save_path: Path to save comparison plot

        Returns:
            Dict with comparison results
        """
        print("\n" + "=" * 70)
        print("Configuration Comparison")
        print("=" * 70)

        results = {}

        for config in configs:
            name = config['name']
            print(f"\n  Analyzing: {name}")

            # Temporarily set trainer
            original_trainer = self.trainer
            self.trainer = config['trainer']

            scores = self.collect_scores(test_loader, task_id)
            pixel_metrics = self.compute_separation_metrics(
                scores['patch_normal'],
                scores['patch_anomaly']
            )
            image_metrics = self.compute_separation_metrics(
                scores['image_normal'],
                scores['image_anomaly']
            )

            results[name] = {
                'pixel': pixel_metrics,
                'image': image_metrics,
                'scores': scores,
            }

            self.trainer = original_trainer

        # Print comparison table
        print("\n--- Comparison Table ---")
        print(f"{'Config':<20} {'Pixel AUROC':<12} {'Pixel d':<10} {'Image AUROC':<12} {'Image d':<10}")
        print("-" * 65)
        for name, res in results.items():
            print(f"{name:<20} {res['pixel']['auroc']:<12.4f} {res['pixel']['cohens_d']:<10.2f} "
                  f"{res['image']['auroc']:<12.4f} {res['image']['cohens_d']:<10.2f}")

        if save_path:
            self._plot_comparison(results, save_path)

        return results

    def _plot_comparison(self, results: Dict, save_path: str):
        """Plot configuration comparison."""
        config_names = list(results.keys())
        n_configs = len(config_names)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Pixel AUROC comparison
        ax = axes[0, 0]
        aurocs = [results[name]['pixel']['auroc'] for name in config_names]
        ax.bar(config_names, aurocs, color='steelblue')
        ax.set_ylabel('AUROC')
        ax.set_title('Pixel-Level AUROC')
        ax.set_ylim(0.9, 1.0)

        # Pixel Cohen's d comparison
        ax = axes[0, 1]
        ds = [results[name]['pixel']['cohens_d'] for name in config_names]
        ax.bar(config_names, ds, color='coral')
        ax.set_ylabel("Cohen's d")
        ax.set_title("Pixel-Level Effect Size")

        # Image AUROC comparison
        ax = axes[1, 0]
        aurocs = [results[name]['image']['auroc'] for name in config_names]
        ax.bar(config_names, aurocs, color='steelblue')
        ax.set_ylabel('AUROC')
        ax.set_title('Image-Level AUROC')
        ax.set_ylim(0.9, 1.0)

        # Image Cohen's d comparison
        ax = axes[1, 1]
        ds = [results[name]['image']['cohens_d'] for name in config_names]
        ax.bar(config_names, ds, color='coral')
        ax.set_ylabel("Cohen's d")
        ax.set_title("Image-Level Effect Size")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison plot to {save_path}")


def analyze_score_by_defect_type(analyzer: ScoreDistributionAnalyzer,
                                   test_loader: DataLoader,
                                   task_id: int,
                                   defect_info: Optional[Dict] = None) -> Dict:
    """
    Analyze score distribution broken down by defect type.

    Args:
        analyzer: ScoreDistributionAnalyzer instance
        test_loader: Test data loader
        task_id: Task ID
        defect_info: Optional dict mapping sample indices to defect types

    Returns:
        Dict with per-defect-type analysis
    """
    print("\n" + "=" * 70)
    print("Per-Defect-Type Score Analysis")
    print("=" * 70)

    # This is a placeholder - actual implementation would need
    # defect type annotations which may not be available in all datasets

    print("  Note: This analysis requires defect type annotations")
    print("  Currently showing aggregate analysis")

    return analyzer.analyze_full(test_loader, task_id)
