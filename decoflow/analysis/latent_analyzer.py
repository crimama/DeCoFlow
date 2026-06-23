"""
Latent Space Analyzer for Normalizing Flow models.

This module analyzes the latent space properties, particularly:
1. Gaussianity of the latent distribution
2. Tail calibration (QQ-plot analysis)
3. Per-dimension statistics

Hypothesis H7: Tail training improves latent space calibration.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.stats import norm, shapiro, kstest, anderson
import matplotlib.pyplot as plt


class LatentSpaceAnalyzer:
    """
    Analyzer for latent space properties of normalizing flows.

    Key analyses:
    1. Per-dimension normality tests
    2. Multivariate Gaussianity assessment
    3. Tail calibration analysis (QQ-plot)
    4. Latent space coverage analysis
    """

    def __init__(self,
                 trainer,
                 device: str = 'cuda'):
        """
        Initialize latent space analyzer.

        Args:
            trainer: DeCoFlowContinualTrainer instance
            device: Device to run analysis on
        """
        self.trainer = trainer
        self.device = device

    def collect_latent_samples(self,
                               data_loader: DataLoader,
                               task_id: int,
                               num_batches: int = 50,
                               max_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect latent samples z and log-determinants from the flow.

        Args:
            data_loader: Data loader
            task_id: Task ID
            num_batches: Maximum number of batches
            max_samples: Maximum number of samples to collect

        Returns:
            z_samples: (N, D) array of latent samples
            logdet_samples: (N,) array of log-determinants
        """
        self.trainer.nf_model.eval()
        self.trainer.vit_extractor.eval()

        z_list = []
        logdet_list = []
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= num_batches or total_samples >= max_samples:
                    break

                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)

                features = self.trainer.vit_extractor(images)
                if self.trainer.use_pos_embedding and self.trainer.pos_embed_generator is not None:
                    pos_embed = self.trainer.pos_embed_generator(features)
                    features = features + pos_embed

                z, logdet_patch = self.trainer.nf_model(features, task_id=task_id)
                B, H, W, D = z.shape

                # Flatten spatial dimensions
                z_flat = z.reshape(-1, D).cpu().numpy()  # (B*H*W, D)
                logdet_flat = logdet_patch.reshape(-1).cpu().numpy()  # (B*H*W,)

                z_list.append(z_flat)
                logdet_list.append(logdet_flat)
                total_samples += len(z_flat)

        z_samples = np.concatenate(z_list, axis=0)[:max_samples]
        logdet_samples = np.concatenate(logdet_list, axis=0)[:max_samples]

        return z_samples, logdet_samples

    def analyze_gaussianity(self,
                            z_samples: np.ndarray,
                            num_dims_to_test: int = 50) -> Dict:
        """
        Analyze Gaussianity of latent samples.

        Tests:
        1. Shapiro-Wilk test per dimension
        2. Kolmogorov-Smirnov test per dimension
        3. Skewness and kurtosis

        Args:
            z_samples: (N, D) latent samples
            num_dims_to_test: Number of dimensions to test

        Returns:
            Dict with Gaussianity metrics
        """
        print("\n" + "=" * 70)
        print("Latent Space Gaussianity Analysis")
        print("=" * 70)

        N, D = z_samples.shape
        print(f"  Samples: {N}, Dimensions: {D}")

        # Subsample for efficiency
        subsample_size = min(5000, N)
        indices = np.random.choice(N, subsample_size, replace=False)
        z_sub = z_samples[indices]

        # Select dimensions to test
        dims_to_test = min(num_dims_to_test, D)
        dim_indices = np.linspace(0, D - 1, dims_to_test, dtype=int)

        shapiro_pvalues = []
        ks_pvalues = []
        skewnesses = []
        kurtoses = []

        for d in dim_indices:
            z_dim = z_sub[:, d]

            # Normalize for testing
            z_norm = (z_dim - z_dim.mean()) / (z_dim.std() + 1e-10)

            # Shapiro-Wilk test (H0: data is normally distributed)
            # Only use up to 5000 samples (Shapiro limit)
            try:
                _, sw_p = shapiro(z_norm[:5000])
                shapiro_pvalues.append(sw_p)
            except:
                shapiro_pvalues.append(np.nan)

            # KS test against standard normal
            _, ks_p = kstest(z_norm, 'norm')
            ks_pvalues.append(ks_p)

            # Moments
            skewnesses.append(stats.skew(z_norm))
            kurtoses.append(stats.kurtosis(z_norm))

        # Overall statistics
        shapiro_pvalues = np.array(shapiro_pvalues)
        ks_pvalues = np.array(ks_pvalues)
        skewnesses = np.array(skewnesses)
        kurtoses = np.array(kurtoses)

        # Fraction passing normality test (p > 0.05)
        shapiro_pass_rate = np.nanmean(shapiro_pvalues > 0.05)
        ks_pass_rate = np.mean(ks_pvalues > 0.05)

        results = {
            'shapiro_pass_rate': shapiro_pass_rate,
            'shapiro_mean_p': np.nanmean(shapiro_pvalues),
            'ks_pass_rate': ks_pass_rate,
            'ks_mean_p': np.mean(ks_pvalues),
            'mean_skewness': np.mean(np.abs(skewnesses)),
            'mean_kurtosis': np.mean(np.abs(kurtoses)),
            'std_skewness': np.std(skewnesses),
            'std_kurtosis': np.std(kurtoses),
            'dims_tested': dims_to_test,
            'n_samples': N,
        }

        # Print summary
        print("\n--- Normality Test Results ---")
        print(f"  Shapiro-Wilk pass rate (p > 0.05): {shapiro_pass_rate:.2%}")
        print(f"  KS test pass rate (p > 0.05): {ks_pass_rate:.2%}")
        print(f"  Mean |skewness|: {results['mean_skewness']:.4f} (ideal: 0)")
        print(f"  Mean |kurtosis|: {results['mean_kurtosis']:.4f} (ideal: 0)")

        # Interpretation
        print("\n--- Interpretation ---")
        if shapiro_pass_rate > 0.8 and results['mean_skewness'] < 0.5:
            print("  GOOD: Latent space is approximately Gaussian")
        elif shapiro_pass_rate > 0.5:
            print("  MODERATE: Some deviation from Gaussianity")
        else:
            print("  POOR: Significant deviation from Gaussianity")

        return results

    def analyze_tail_calibration(self,
                                  z_samples: np.ndarray,
                                  save_path: Optional[str] = None) -> Dict:
        """
        Analyze tail calibration using QQ-plot analysis.

        Compares empirical quantiles with theoretical N(0,1) quantiles,
        particularly focusing on tail regions.

        Args:
            z_samples: (N, D) latent samples
            save_path: Path to save QQ-plot

        Returns:
            Dict with tail calibration metrics
        """
        print("\n" + "=" * 70)
        print("Tail Calibration Analysis (QQ-Plot)")
        print("=" * 70)

        N, D = z_samples.shape

        # Flatten all samples for overall analysis
        z_flat = z_samples.flatten()

        # Theoretical quantiles
        percentiles = np.linspace(1, 99, 99)
        theoretical_q = norm.ppf(percentiles / 100)
        empirical_q = np.percentile(z_flat, percentiles)

        # QQ correlation (overall)
        qq_correlation = np.corrcoef(theoretical_q, empirical_q)[0, 1]

        # Focus on tail regions
        tail_percentiles = [1, 2, 5, 95, 98, 99]
        tail_theoretical = norm.ppf(np.array(tail_percentiles) / 100)
        tail_empirical = np.percentile(z_flat, tail_percentiles)

        # Tail calibration error
        tail_error = np.abs(tail_theoretical - tail_empirical)
        mean_tail_error = tail_error.mean()

        # Extreme tail analysis (0.1%, 99.9%)
        extreme_percentiles = [0.1, 0.5, 99.5, 99.9]
        extreme_theoretical = norm.ppf(np.array(extreme_percentiles) / 100)
        extreme_empirical = np.percentile(z_flat, extreme_percentiles)
        extreme_error = np.abs(extreme_theoretical - extreme_empirical)
        mean_extreme_error = extreme_error.mean()

        results = {
            'qq_correlation': qq_correlation,
            'mean_tail_error': mean_tail_error,
            'mean_extreme_error': mean_extreme_error,
            'tail_errors': dict(zip(tail_percentiles, tail_error.tolist())),
            'extreme_errors': dict(zip(extreme_percentiles, extreme_error.tolist())),
            'theoretical_quantiles': theoretical_q.tolist(),
            'empirical_quantiles': empirical_q.tolist(),
        }

        # Print summary
        print(f"  QQ correlation: {qq_correlation:.4f}")
        print(f"  Mean tail error (1,2,5,95,98,99%): {mean_tail_error:.4f}")
        print(f"  Mean extreme error (0.1,0.5,99.5,99.9%): {mean_extreme_error:.4f}")
        print("\n  Tail Error Details:")
        for p, err in zip(tail_percentiles, tail_error):
            print(f"    {p}th percentile: error = {err:.4f}")

        # Interpretation
        print("\n--- Interpretation ---")
        if qq_correlation > 0.99 and mean_tail_error < 0.2:
            print("  EXCELLENT: Very well calibrated, including tails")
        elif qq_correlation > 0.95 and mean_tail_error < 0.5:
            print("  GOOD: Well calibrated, minor tail deviation")
        else:
            print("  MODERATE: Some calibration issues in tails")

        if save_path:
            self._plot_qq(theoretical_q, empirical_q, results, save_path)

        return results

    def _plot_qq(self,
                 theoretical_q: np.ndarray,
                 empirical_q: np.ndarray,
                 results: Dict,
                 save_path: str):
        """Plot QQ-plot."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Main QQ-plot
        ax = axes[0]
        ax.scatter(theoretical_q, empirical_q, alpha=0.5, s=20)
        ax.plot([-4, 4], [-4, 4], 'r--', label='Ideal')
        ax.set_xlabel('Theoretical Quantiles (N(0,1))')
        ax.set_ylabel('Empirical Quantiles')
        ax.set_title(f'QQ-Plot (correlation = {results["qq_correlation"]:.4f})')
        ax.legend()
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)

        # Tail deviation plot
        ax = axes[1]
        deviation = empirical_q - theoretical_q
        percentiles = np.linspace(1, 99, 99)
        ax.bar(percentiles, deviation, width=1, color='steelblue', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Deviation (Empirical - Theoretical)')
        ax.set_title(f'Tail Deviation (mean tail error = {results["mean_tail_error"]:.4f})')
        ax.grid(True, alpha=0.3)

        # Highlight tail regions
        ax.axvspan(0, 5, alpha=0.2, color='red', label='Lower tail')
        ax.axvspan(95, 100, alpha=0.2, color='red', label='Upper tail')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved QQ-plot to {save_path}")

    def analyze_logdet_distribution(self,
                                     logdet_samples: np.ndarray,
                                     save_path: Optional[str] = None) -> Dict:
        """
        Analyze the distribution of log-determinants.

        The log-det distribution affects score distribution and
        should ideally be stable.

        Args:
            logdet_samples: (N,) log-determinant samples
            save_path: Path to save histogram

        Returns:
            Dict with log-det statistics
        """
        print("\n" + "=" * 70)
        print("Log-Determinant Distribution Analysis")
        print("=" * 70)

        results = {
            'mean': logdet_samples.mean(),
            'std': logdet_samples.std(),
            'min': logdet_samples.min(),
            'max': logdet_samples.max(),
            'skewness': stats.skew(logdet_samples),
            'kurtosis': stats.kurtosis(logdet_samples),
        }

        # Percentiles
        for p in [1, 5, 25, 50, 75, 95, 99]:
            results[f'p{p}'] = np.percentile(logdet_samples, p)

        # Print summary
        print(f"  Mean: {results['mean']:.4f}")
        print(f"  Std: {results['std']:.4f}")
        print(f"  Range: [{results['min']:.4f}, {results['max']:.4f}]")
        print(f"  Skewness: {results['skewness']:.4f}")
        print(f"  Kurtosis: {results['kurtosis']:.4f}")

        # Check for numerical issues
        if results['std'] > 100:
            print("\n  WARNING: High variance in log-det (potential numerical instability)")
        if np.abs(results['skewness']) > 2:
            print("\n  WARNING: Highly skewed log-det distribution")

        if save_path:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.hist(logdet_samples, bins=100, density=True, alpha=0.7)
            ax.axvline(results['mean'], color='r', linestyle='--', label=f'Mean = {results["mean"]:.2f}')
            ax.set_xlabel('Log-Determinant')
            ax.set_ylabel('Density')
            ax.set_title('Log-Determinant Distribution')
            ax.legend()
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved histogram to {save_path}")

        return results

    def run_full_analysis(self,
                           data_loader: DataLoader,
                           task_id: int,
                           output_dir: str,
                           num_batches: int = 50) -> Dict:
        """
        Run complete latent space analysis.

        Args:
            data_loader: Data loader
            task_id: Task ID
            output_dir: Directory to save results
            num_batches: Number of batches to analyze

        Returns:
            Dict with all analysis results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("Full Latent Space Analysis")
        print("=" * 70)

        # Collect samples
        print("\nCollecting latent samples...")
        z_samples, logdet_samples = self.collect_latent_samples(
            data_loader, task_id, num_batches
        )
        print(f"  Collected {len(z_samples)} samples")

        results = {}

        # Gaussianity analysis
        results['gaussianity'] = self.analyze_gaussianity(z_samples)

        # Tail calibration
        results['tail_calibration'] = self.analyze_tail_calibration(
            z_samples,
            save_path=os.path.join(output_dir, 'qq_plot.png')
        )

        # Log-det distribution
        results['logdet'] = self.analyze_logdet_distribution(
            logdet_samples,
            save_path=os.path.join(output_dir, 'logdet_histogram.png')
        )

        return results


def compare_latent_spaces(analyzer: LatentSpaceAnalyzer,
                           train_loader: DataLoader,
                           test_loader: DataLoader,
                           task_id: int,
                           output_dir: str) -> Dict:
    """
    Compare latent space properties between train and test data.

    Args:
        analyzer: LatentSpaceAnalyzer instance
        train_loader: Training data loader
        test_loader: Test data loader
        task_id: Task ID
        output_dir: Output directory

    Returns:
        Dict with comparison results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("Train vs Test Latent Space Comparison")
    print("=" * 70)

    # Collect train samples
    print("\nCollecting train latent samples...")
    train_z, train_logdet = analyzer.collect_latent_samples(
        train_loader, task_id, num_batches=30
    )

    # Collect test samples
    print("Collecting test latent samples...")
    test_z, test_logdet = analyzer.collect_latent_samples(
        test_loader, task_id, num_batches=30
    )

    # Compare distributions
    results = {
        'train': {
            'z_mean': train_z.mean(),
            'z_std': train_z.std(),
            'logdet_mean': train_logdet.mean(),
            'logdet_std': train_logdet.std(),
        },
        'test': {
            'z_mean': test_z.mean(),
            'z_std': test_z.std(),
            'logdet_mean': test_logdet.mean(),
            'logdet_std': test_logdet.std(),
        }
    }

    # 2-sample KS test
    # Test if train and test z come from same distribution
    z_ks_stat, z_ks_p = stats.ks_2samp(train_z.flatten()[:10000],
                                        test_z.flatten()[:10000])
    results['z_ks_statistic'] = z_ks_stat
    results['z_ks_pvalue'] = z_ks_p

    logdet_ks_stat, logdet_ks_p = stats.ks_2samp(train_logdet[:10000],
                                                   test_logdet[:10000])
    results['logdet_ks_statistic'] = logdet_ks_stat
    results['logdet_ks_pvalue'] = logdet_ks_p

    # Print comparison
    print("\n--- Distribution Comparison ---")
    print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'z mean':<20} {results['train']['z_mean']:<15.4f} {results['test']['z_mean']:<15.4f}")
    print(f"{'z std':<20} {results['train']['z_std']:<15.4f} {results['test']['z_std']:<15.4f}")
    print(f"{'logdet mean':<20} {results['train']['logdet_mean']:<15.4f} {results['test']['logdet_mean']:<15.4f}")
    print(f"{'logdet std':<20} {results['train']['logdet_std']:<15.4f} {results['test']['logdet_std']:<15.4f}")

    print("\n--- Distribution Shift Tests ---")
    print(f"  z KS test: stat={z_ks_stat:.4f}, p={z_ks_p:.4e}")
    print(f"  logdet KS test: stat={logdet_ks_stat:.4f}, p={logdet_ks_p:.4e}")

    if z_ks_p < 0.05:
        print("\n  WARNING: Significant distribution shift detected in z")
    if logdet_ks_p < 0.05:
        print("\n  WARNING: Significant distribution shift detected in log-det")

    return results
