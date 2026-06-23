#!/usr/bin/env python3
"""
Continual Learning Baselines for DeCoFlow comparison.

Implements:
1. Fine-tune: Sequential training without any CL strategy (catastrophic forgetting)
2. EWC: Elastic Weight Consolidation (Kirkpatrick et al., 2017)

These baselines demonstrate the necessity of DeCoFlow's approach.

Usage:
  python scripts/run_cl_baselines.py --method finetune --num_tasks 15
  python scripts/run_cl_baselines.py --method ewc --num_tasks 15
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from decoflow.extractors import create_feature_extractor, get_backbone_type
from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.data.mvtec import MVTEC
from sklearn.metrics import roc_auc_score


# =============================================================================
# Configuration
# =============================================================================

MVTEC_CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

DEFAULT_CONFIG = {
    'backbone': 'wide_resnet50_2',
    'data_path': '/Data/MVTecAD',
    'img_size': 256,
    'batch_size': 16,
    'num_epochs': 60,
    'lr': 2e-4,
    'num_coupling_layers': 8,
    'target_embed_dim': 1024,
    'ewc_lambda': 1000,  # EWC importance weight
    'ewc_samples': 200,  # Samples for Fisher computation
    'seed': 42,
}


# =============================================================================
# EWC Implementation
# =============================================================================

class EWC:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.

    Adds a quadratic penalty to the loss for changes to important parameters:
    L_total = L_current + (lambda/2) * sum_i F_i * (theta_i - theta_i*)^2

    Where F_i is the Fisher information for parameter i, and theta_i* is the
    optimal parameter value after the previous task.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.tasks_trained = 0

    def compute_fisher(
        self,
        dataloader: DataLoader,
        feature_extractor,
        device: str,
        num_samples: int = 200
    ):
        """
        Compute Fisher Information Matrix diagonal.

        Uses empirical Fisher: F_i = E[(dL/dθ_i)^2]
        """
        self.model.train()

        # Initialize Fisher accumulator
        fisher_accum = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        samples_seen = 0
        for batch in dataloader:
            if samples_seen >= num_samples:
                break

            images, _, _, _, _ = batch
            images = images.to(device)
            batch_size = images.size(0)

            # Get features
            with torch.no_grad():
                features = feature_extractor(images)

            # Enable grad for Fisher computation
            self.model.zero_grad()

            # Forward pass
            z, logdet_patch = self.model(features)

            # NLL loss
            log_pz_patch = -0.5 * torch.sum(z ** 2, dim=-1)
            log_px_patch = log_pz_patch + logdet_patch
            nll = -log_px_patch.sum(dim=(1, 2))
            loss = nll.mean()

            # Backward to get gradients
            loss.backward()

            # Accumulate squared gradients
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_accum[n] += (p.grad.detach() ** 2) * batch_size

            samples_seen += batch_size

        # Average Fisher
        for n in fisher_accum:
            fisher_accum[n] /= samples_seen

        # Consolidate Fisher (accumulate across tasks)
        for n in fisher_accum:
            if n in self.fisher:
                self.fisher[n] = self.fisher[n] + fisher_accum[n]
            else:
                self.fisher[n] = fisher_accum[n]

        # Store optimal parameters
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.optimal_params[n] = p.detach().clone()

        self.tasks_trained += 1

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term.

        Returns:
            Scalar tensor with EWC penalty
        """
        if self.tasks_trained == 0:
            return torch.tensor(0.0)

        penalty = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.optimal_params:
                diff = p - self.optimal_params[n]
                penalty += (self.fisher[n] * (diff ** 2)).sum()

        return (self.lambda_ewc / 2) * penalty


# =============================================================================
# Training Functions
# =============================================================================

def create_dataloader(config: dict, class_name: str, train: bool = True) -> DataLoader:
    """Create DataLoader for a specific class."""
    dataset = MVTEC(
        root=config['data_path'],
        class_name=class_name,
        train=train,
        img_size=config['img_size'],
        crp_size=config['img_size'],
        msk_size=config['img_size']
    )

    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        drop_last=train and len(dataset) >= config['batch_size']
    )


def create_model(config: dict, device: str) -> Tuple[nn.Module, nn.Module]:
    """Create feature extractor and NF model."""
    # Feature extractor
    feature_extractor = create_feature_extractor(
        backbone_name=config['backbone'],
        input_shape=(3, config['img_size'], config['img_size']),
        target_embed_dimension=config['target_embed_dim'],
        device=device
    )

    # Freeze backbone
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()

    # Get feature dimensions
    with torch.no_grad():
        dummy = torch.randn(1, 3, config['img_size'], config['img_size']).to(device)
        dummy_features = feature_extractor(dummy)
        _, H, W, C = dummy_features.shape

    # Create ablation config for baseline (no LoRA)
    from decoflow.config.ablation import AblationConfig
    ablation_config = AblationConfig(
        use_lora=False,  # No LoRA for baselines
        use_tsa=False,
        use_acb=False,
    )

    # Create NF model (without LoRA - plain shared weights)
    nf_model = DeCoFlowNF(
        embed_dim=C,
        coupling_layers=config['num_coupling_layers'],
        device=device,
        ablation_config=ablation_config,
    ).to(device)

    return feature_extractor, nf_model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    device: str
) -> float:
    """Evaluate model and return Image AUC."""
    model.eval()

    all_labels = []
    all_scores = []

    for batch in dataloader:
        images, labels, _, _, _ = batch
        images = images.to(device)

        # Get features
        features = feature_extractor(images)

        # Forward pass
        z, logdet_patch = model(features)  # z: (B, H, W, D), logdet_patch: (B, H, W)

        # Anomaly score = NLL
        log_pz_patch = -0.5 * torch.sum(z ** 2, dim=-1)  # (B, H, W)
        log_px_patch = log_pz_patch + logdet_patch  # (B, H, W)
        nll = -log_px_patch.sum(dim=(1, 2))  # (B,)
        scores = nll.cpu().numpy()

        all_labels.append(labels.numpy())
        all_scores.append(scores)

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)

    # Compute AUC
    if len(np.unique(labels)) < 2:
        return 0.5

    return roc_auc_score(labels, scores)


def train_epoch(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    ewc: Optional[EWC] = None
) -> float:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images, _, _, _, _ = batch
        images = images.to(device)

        # Get features
        with torch.no_grad():
            features = feature_extractor(images)

        # Forward pass
        z, logdet_patch = model(features)  # z: (B, H, W, D), logdet_patch: (B, H, W)

        # NLL loss
        # log p(z) = -0.5 * ||z||^2 for standard Gaussian
        # Patch-wise: sum over D dimension
        log_pz_patch = -0.5 * torch.sum(z ** 2, dim=-1)  # (B, H, W)

        # Patch-wise log p(x) = log p(z) + log|det J|
        log_px_patch = log_pz_patch + logdet_patch  # (B, H, W)

        # Image-level NLL: negative sum of log probs
        nll = -log_px_patch.sum(dim=(1, 2))  # (B,)
        loss = nll.mean()

        # Add EWC penalty if applicable
        if ewc is not None:
            ewc_penalty = ewc.penalty()
            loss = loss + ewc_penalty

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def run_continual_learning(
    method: str,
    config: dict,
    num_tasks: int,
    log_dir: Path,
    device: str
):
    """
    Run continual learning experiment.

    Args:
        method: 'finetune' or 'ewc'
        config: Experiment configuration
        num_tasks: Number of tasks to train
        log_dir: Directory for saving results
        device: Device to use
    """
    print(f"\n{'='*70}")
    print(f"Running {method.upper()} baseline ({num_tasks} tasks)")
    print(f"{'='*70}\n")

    # Create model
    feature_extractor, model = create_model(config, device)

    # Create EWC if needed
    ewc = EWC(model, lambda_ewc=config['ewc_lambda']) if method == 'ewc' else None

    # Results storage
    # results[after_task][eval_task] = auc
    results: Dict[int, Dict[int, float]] = {}

    # Task classes
    task_classes = MVTEC_CLASSES[:num_tasks]

    for task_id, class_name in enumerate(task_classes):
        print(f"\n[Task {task_id}] Training on {class_name}")

        # Create dataloaders
        train_loader = create_dataloader(config, class_name, train=True)
        test_loader = create_dataloader(config, class_name, train=False)

        # Create optimizer (reinitialize for each task for fair comparison)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)

        # Training loop
        for epoch in range(config['num_epochs']):
            loss = train_epoch(model, feature_extractor, train_loader, optimizer, device, ewc)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss:.4f}")

        # After training, compute Fisher for EWC
        if ewc is not None:
            print(f"  Computing Fisher Information...")
            ewc.compute_fisher(train_loader, feature_extractor, device, config['ewc_samples'])

        # Evaluate on all tasks seen so far
        print(f"  Evaluating on all tasks...")
        results[task_id] = {}

        for eval_task_id in range(task_id + 1):
            eval_class = task_classes[eval_task_id]
            eval_loader = create_dataloader(config, eval_class, train=False)
            auc = evaluate(model, feature_extractor, eval_loader, device)
            results[task_id][eval_task_id] = auc
            print(f"    Task {eval_task_id} ({eval_class}): {auc:.4f}")

        # Save intermediate results
        save_results(results, task_classes, log_dir, method)

    # Final summary
    print_summary(results, task_classes, method)

    return results


def save_results(
    results: Dict[int, Dict[int, float]],
    task_classes: List[str],
    log_dir: Path,
    method: str
):
    """Save results to CSV files."""
    log_dir.mkdir(parents=True, exist_ok=True)

    for after_task, task_results in results.items():
        csv_path = log_dir / f"continual_results_after_task_{after_task}.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class_name', 'task_id', 'img_auc', 'pixel_auc', 'is_current_task', 'routing_accuracy'])

            for task_id, auc in task_results.items():
                writer.writerow([
                    task_classes[task_id],
                    task_id,
                    auc,
                    0.0,  # pixel_auc (not computed for baselines)
                    task_id == after_task,
                    1.0  # routing (oracle for baselines)
                ])

            # Average row
            avg_auc = np.mean(list(task_results.values()))
            writer.writerow(['Average', '', avg_auc, 0.0, '', 1.0])


def print_summary(
    results: Dict[int, Dict[int, float]],
    task_classes: List[str],
    method: str
):
    """Print final summary with BWT and FM."""
    num_tasks = len(task_classes)
    final_results = results[num_tasks - 1]

    print(f"\n{'='*70}")
    print(f"Final Results: {method.upper()}")
    print(f"{'='*70}")

    # Per-task results
    print(f"\n{'Task':<15} {'Initial':<12} {'Final':<12} {'BWT':<12} {'FM':<12}")
    print("-" * 63)

    total_bwt = 0.0
    total_fm = 0.0

    for task_id in range(num_tasks - 1):
        initial = results[task_id][task_id]
        final = final_results[task_id]
        bwt = final - initial

        # FM = max performance - final performance
        peak = max(results[t][task_id] for t in range(task_id, num_tasks))
        fm = peak - final

        print(f"{task_classes[task_id]:<15} {initial:<12.4f} {final:<12.4f} {bwt:<+12.4f} {fm:<12.4f}")

        total_bwt += bwt
        total_fm += fm

    avg_bwt = total_bwt / (num_tasks - 1)
    avg_fm = total_fm / (num_tasks - 1)

    print("-" * 63)
    print(f"{'Average':<15} {'':<12} {'':<12} {avg_bwt:<+12.4f} {avg_fm:<12.4f}")

    # Overall metrics
    avg_final_auc = np.mean(list(final_results.values()))
    print(f"\n[Summary]")
    print(f"  Average Final I-AUC: {avg_final_auc:.4f}")
    print(f"  Average BWT:         {avg_bwt:+.4f}")
    print(f"  Average FM:          {avg_fm:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run CL baselines for DeCoFlow comparison")
    parser.add_argument("--method", type=str, required=True, choices=['finetune', 'ewc'],
                        help="CL method to run")
    parser.add_argument("--num_tasks", type=int, default=15, help="Number of tasks")
    parser.add_argument("--num_epochs", type=int, default=60, help="Epochs per task")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--ewc_lambda", type=float, default=1000, help="EWC importance weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="./logs/CL_Baselines", help="Log directory")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Config
    config = DEFAULT_CONFIG.copy()
    config['num_epochs'] = args.num_epochs
    config['lr'] = args.lr
    config['ewc_lambda'] = args.ewc_lambda
    config['seed'] = args.seed

    # Log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"{args.method}_{args.num_tasks}tasks_{timestamp}"

    # Save config
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Run experiment
    results = run_continual_learning(
        method=args.method,
        config=config,
        num_tasks=args.num_tasks,
        log_dir=log_dir,
        device=device
    )

    print(f"\nResults saved to: {log_dir}")


if __name__ == "__main__":
    main()
