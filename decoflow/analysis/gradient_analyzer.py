"""
Gradient Dynamics Analyzer for Tail-Aware Loss.

This module analyzes how Tail-Aware Loss affects gradient distribution
during training, helping understand the learning dynamics difference
between mean-only and tail-aware training.

Hypothesis H3: Mean-only training dilutes gradients from tail patches.
Hypothesis H6: Tail training improves Jacobian precision.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class GradientAnalyzer:
    """
    Analyzer for gradient dynamics in Tail-Aware Loss training.

    Key analyses:
    1. Per-patch gradient magnitude distribution
    2. Layer-wise gradient statistics
    3. Gradient concentration (Gini coefficient)
    4. Effective gradient rank
    """

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda'):
        """
        Initialize gradient analyzer.

        Args:
            model: The normalizing flow model
            device: Device to run analysis on
        """
        self.model = model
        self.device = device
        self._gradient_hooks = []
        self._gradient_storage = defaultdict(list)

    def register_gradient_hooks(self, layer_names: Optional[List[str]] = None):
        """
        Register hooks to capture gradients during backward pass.

        Args:
            layer_names: Specific layer names to monitor (None = all)
        """
        self.clear_hooks()

        for name, module in self.model.named_modules():
            if layer_names is not None and name not in layer_names:
                continue

            # Register hook for modules with parameters
            if len(list(module.parameters(recurse=False))) > 0:
                hook = module.register_full_backward_hook(
                    self._make_gradient_hook(name)
                )
                self._gradient_hooks.append(hook)

    def _make_gradient_hook(self, layer_name: str):
        """Create a gradient capture hook for a specific layer."""
        def hook(module, grad_input, grad_output):
            # Capture gradient magnitudes for each parameter
            for param_name, param in module.named_parameters(recurse=False):
                if param.grad is not None:
                    grad_mag = param.grad.abs().mean().item()
                    self._gradient_storage[f"{layer_name}.{param_name}"].append(grad_mag)
        return hook

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._gradient_hooks:
            hook.remove()
        self._gradient_hooks = []
        self._gradient_storage.clear()

    def analyze_single_batch(self,
                             features: torch.Tensor,
                             task_id: int,
                             tail_weight: float,
                             tail_ratio: float) -> Dict:
        """
        Analyze gradients for a single batch with different loss configurations.

        Args:
            features: Input features (B, H, W, D)
            task_id: Task ID for the model
            tail_weight: Weight for tail loss (0 = mean only, 1 = tail only)
            tail_ratio: Ratio of patches considered as tail

        Returns:
            Dict with gradient statistics
        """
        self.model.train()
        self._gradient_storage.clear()

        # Forward pass
        z, logdet_patch = self.model(features, task_id=task_id)
        B, H, W, D = z.shape

        # Compute NLL
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
        nll_patch = -(log_pz_patch + logdet_patch)

        # Compute loss based on configuration
        flat_nll = nll_patch.reshape(B, -1)
        num_patches = H * W

        # Mean loss
        mean_loss = flat_nll.mean()

        # Tail loss
        k = max(1, int(num_patches * tail_ratio))
        top_k_nll, _ = torch.topk(flat_nll, k, dim=1)
        tail_loss = top_k_nll.mean()

        # Combined loss
        loss = (1 - tail_weight) * mean_loss + tail_weight * tail_loss

        # Backward pass (captures gradients via hooks)
        loss.backward()

        # Collect statistics
        results = {
            'loss': loss.item(),
            'mean_loss': mean_loss.item(),
            'tail_loss': tail_loss.item(),
            'gradient_stats': {}
        }

        for layer_name, grad_list in self._gradient_storage.items():
            if len(grad_list) > 0:
                results['gradient_stats'][layer_name] = {
                    'mean': np.mean(grad_list),
                    'std': np.std(grad_list),
                    'max': np.max(grad_list),
                    'min': np.min(grad_list),
                }

        return results

    def compare_loss_configurations(self,
                                    train_loader: DataLoader,
                                    extractor,
                                    pos_embed_generator,
                                    task_id: int,
                                    tail_ratio: float = 0.02,
                                    num_batches: int = 20,
                                    device: str = 'cuda') -> Dict:
        """
        Compare gradient dynamics between mean-only and tail-aware training.

        Args:
            train_loader: Training data loader
            extractor: Feature extractor
            pos_embed_generator: Positional embedding generator
            task_id: Task ID
            tail_ratio: Ratio of tail patches
            num_batches: Number of batches to analyze
            device: Device

        Returns:
            Dict with comparison results
        """
        print("\n" + "=" * 70)
        print("Gradient Dynamics Comparison: Mean-Only vs Tail-Aware")
        print("=" * 70)

        configurations = [
            {'name': 'mean_only', 'tail_weight': 0.0},
            {'name': 'tail_aware', 'tail_weight': 1.0},
            {'name': 'balanced', 'tail_weight': 0.7},
        ]

        results = {config['name']: [] for config in configurations}

        # Register hooks
        self.register_gradient_hooks()

        with torch.enable_grad():
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(device)

                # Extract features
                with torch.no_grad():
                    features = extractor(images)
                    if pos_embed_generator is not None:
                        pos_embed = pos_embed_generator(features)
                        features = features + pos_embed

                # Analyze each configuration
                for config in configurations:
                    # Clone features to allow multiple backward passes
                    features_clone = features.clone().detach().requires_grad_(False)

                    # Zero gradients
                    self.model.zero_grad()

                    # Analyze
                    batch_results = self.analyze_single_batch(
                        features_clone,
                        task_id=task_id,
                        tail_weight=config['tail_weight'],
                        tail_ratio=tail_ratio
                    )
                    results[config['name']].append(batch_results)

                if (batch_idx + 1) % 5 == 0:
                    print(f"  Processed {batch_idx + 1}/{num_batches} batches")

        self.clear_hooks()

        # Aggregate results
        aggregated = {}
        for config_name, batch_results in results.items():
            aggregated[config_name] = self._aggregate_results(batch_results)

        # Print summary
        self._print_comparison_summary(aggregated)

        return aggregated

    def _aggregate_results(self, batch_results: List[Dict]) -> Dict:
        """Aggregate results across batches."""
        aggregated = {
            'mean_loss': np.mean([r['mean_loss'] for r in batch_results]),
            'tail_loss': np.mean([r['tail_loss'] for r in batch_results]),
            'loss': np.mean([r['loss'] for r in batch_results]),
            'gradient_stats': {}
        }

        # Aggregate gradient stats per layer
        all_layers = set()
        for r in batch_results:
            all_layers.update(r['gradient_stats'].keys())

        for layer in all_layers:
            means = []
            maxs = []
            for r in batch_results:
                if layer in r['gradient_stats']:
                    means.append(r['gradient_stats'][layer]['mean'])
                    maxs.append(r['gradient_stats'][layer]['max'])

            if len(means) > 0:
                aggregated['gradient_stats'][layer] = {
                    'mean': np.mean(means),
                    'std': np.std(means),
                    'max_mean': np.mean(maxs),
                }

        return aggregated

    def _print_comparison_summary(self, aggregated: Dict):
        """Print comparison summary."""
        print("\n--- Loss Comparison ---")
        print(f"{'Configuration':<15} {'Total Loss':<12} {'Mean Loss':<12} {'Tail Loss':<12}")
        print("-" * 55)
        for config_name, stats in aggregated.items():
            print(f"{config_name:<15} {stats['loss']:<12.4f} {stats['mean_loss']:<12.4f} {stats['tail_loss']:<12.4f}")

        print("\n--- Gradient Magnitude Comparison (Top Layers) ---")

        # Find layers with largest gradient difference
        if 'mean_only' in aggregated and 'tail_aware' in aggregated:
            mean_stats = aggregated['mean_only']['gradient_stats']
            tail_stats = aggregated['tail_aware']['gradient_stats']

            common_layers = set(mean_stats.keys()) & set(tail_stats.keys())

            differences = []
            for layer in common_layers:
                diff = tail_stats[layer]['mean'] - mean_stats[layer]['mean']
                ratio = tail_stats[layer]['mean'] / (mean_stats[layer]['mean'] + 1e-10)
                differences.append((layer, diff, ratio, mean_stats[layer]['mean'], tail_stats[layer]['mean']))

            # Sort by absolute difference
            differences.sort(key=lambda x: abs(x[1]), reverse=True)

            print(f"{'Layer':<50} {'Mean-Only':<12} {'Tail-Aware':<12} {'Ratio':<10}")
            print("-" * 85)
            for layer, diff, ratio, mean_val, tail_val in differences[:10]:
                # Shorten layer name
                short_name = layer[-47:] if len(layer) > 47 else layer
                print(f"{short_name:<50} {mean_val:<12.2e} {tail_val:<12.2e} {ratio:<10.2f}")

    def compute_gradient_concentration(self,
                                       gradient_magnitudes: List[float]) -> Dict:
        """
        Compute gradient concentration metrics.

        Args:
            gradient_magnitudes: List of gradient magnitudes

        Returns:
            Dict with Gini coefficient and effective rank
        """
        magnitudes = np.array(gradient_magnitudes)
        magnitudes = magnitudes / (magnitudes.sum() + 1e-10)  # Normalize

        # Gini coefficient
        n = len(magnitudes)
        sorted_mags = np.sort(magnitudes)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_mags) / (n * np.sum(sorted_mags))) - (n + 1) / n

        # Effective rank (entropy-based)
        magnitudes_clipped = np.clip(magnitudes, 1e-10, 1.0)
        entropy = -np.sum(magnitudes_clipped * np.log(magnitudes_clipped))
        effective_rank = np.exp(entropy)

        return {
            'gini': gini,
            'effective_rank': effective_rank,
            'max_rank': n,
            'concentration_ratio': 1 - effective_rank / n
        }

    def analyze_per_patch_gradients(self,
                                     features: torch.Tensor,
                                     task_id: int,
                                     tail_weight: float,
                                     tail_ratio: float) -> Dict:
        """
        Analyze gradient magnitude per patch.

        This helps understand which patches receive stronger gradients.

        Args:
            features: Input features (B, H, W, D)
            task_id: Task ID
            tail_weight: Tail loss weight
            tail_ratio: Tail ratio

        Returns:
            Dict with per-patch gradient analysis
        """
        self.model.train()
        features = features.clone().detach().requires_grad_(True)

        # Forward pass
        z, logdet_patch = self.model(features, task_id=task_id)
        B, H, W, D = z.shape

        # Compute NLL
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
        nll_patch = -(log_pz_patch + logdet_patch)

        # Compute loss
        flat_nll = nll_patch.reshape(B, -1)
        num_patches = H * W

        mean_loss = flat_nll.mean()
        k = max(1, int(num_patches * tail_ratio))
        top_k_nll, top_k_indices = torch.topk(flat_nll, k, dim=1)
        tail_loss = top_k_nll.mean()

        loss = (1 - tail_weight) * mean_loss + tail_weight * tail_loss

        # Backward to get gradients w.r.t. features
        loss.backward()

        # Compute per-patch gradient magnitude
        grad_features = features.grad  # (B, H, W, D)
        grad_magnitude = grad_features.norm(dim=-1)  # (B, H, W)

        # Analyze gradient distribution
        flat_grad = grad_magnitude.reshape(B, -1)  # (B, H*W)

        # Which patches have highest gradients?
        _, high_grad_indices = torch.topk(flat_grad.mean(dim=0), k)

        # Overlap with tail patches
        tail_set = set(top_k_indices[0].tolist())  # Use first sample
        high_grad_set = set(high_grad_indices.tolist())
        overlap = len(tail_set & high_grad_set) / k

        results = {
            'grad_magnitude_mean': grad_magnitude.mean().item(),
            'grad_magnitude_std': grad_magnitude.std().item(),
            'grad_at_tail_mean': flat_grad[0, top_k_indices[0]].mean().item(),
            'grad_at_nontail_mean': flat_grad[0, ~torch.isin(torch.arange(num_patches, device=features.device), top_k_indices[0])].mean().item(),
            'high_grad_tail_overlap': overlap,
            'grad_concentration': self.compute_gradient_concentration(
                flat_grad.mean(dim=0).cpu().numpy().tolist()
            )
        }

        return results


def plot_gradient_comparison(mean_results: Dict,
                             tail_results: Dict,
                             save_path: str):
    """
    Plot gradient comparison between mean-only and tail-aware training.

    Args:
        mean_results: Results from mean-only training
        tail_results: Results from tail-aware training
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Loss comparison
    ax = axes[0]
    labels = ['Mean Loss', 'Tail Loss', 'Total Loss']
    mean_vals = [mean_results['mean_loss'], mean_results['tail_loss'], mean_results['loss']]
    tail_vals = [tail_results['mean_loss'], tail_results['tail_loss'], tail_results['loss']]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, mean_vals, width, label='Mean-Only Training')
    ax.bar(x + width / 2, tail_vals, width, label='Tail-Aware Training')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Plot 2: Gradient magnitude per layer
    ax = axes[1]
    mean_grads = mean_results.get('gradient_stats', {})
    tail_grads = tail_results.get('gradient_stats', {})

    common_layers = list(set(mean_grads.keys()) & set(tail_grads.keys()))[:10]
    if common_layers:
        mean_vals = [mean_grads[l]['mean'] for l in common_layers]
        tail_vals = [tail_grads[l]['mean'] for l in common_layers]

        x = np.arange(len(common_layers))
        ax.bar(x - width / 2, mean_vals, width, label='Mean-Only')
        ax.bar(x + width / 2, tail_vals, width, label='Tail-Aware')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title('Per-Layer Gradients')
        ax.set_xticks(x)
        ax.set_xticklabels([l.split('.')[-1][:10] for l in common_layers], rotation=45)
        ax.legend()

    # Plot 3: Gradient ratio (tail/mean)
    ax = axes[2]
    if common_layers:
        ratios = [tail_grads[l]['mean'] / (mean_grads[l]['mean'] + 1e-10)
                  for l in common_layers]
        ax.bar(range(len(ratios)), ratios)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Equal')
        ax.set_ylabel('Gradient Ratio (Tail/Mean)')
        ax.set_title('Gradient Amplification by Tail-Aware Loss')
        ax.set_xticks(range(len(common_layers)))
        ax.set_xticklabels([l.split('.')[-1][:10] for l in common_layers], rotation=45)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved gradient comparison plot to {save_path}")
