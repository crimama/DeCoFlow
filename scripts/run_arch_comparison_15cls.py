#!/usr/bin/env python3
"""
Cross-Architecture Comparison (Table 3) — 15-class V48 config.

Runs AE, VAE, Teacher-Student baselines with the same "frozen base + task-specific adapter"
decomposition strategy, using V48 config (WRN50, 768-dim, high_res, 60ep).

NF (DeCoFlow) result is taken from V48_01 checkpoint (no re-run needed).

Usage:
    # Run all baselines on separate GPUs
    CUDA_VISIBLE_DEVICES=2 python scripts/run_arch_comparison_15cls.py --model ae
    CUDA_VISIBLE_DEVICES=3 python scripts/run_arch_comparison_15cls.py --model vae
    CUDA_VISIBLE_DEVICES=4 python scripts/run_arch_comparison_15cls.py --model ts

    # Run all sequentially on one GPU
    CUDA_VISIBLE_DEVICES=2 python scripts/run_arch_comparison_15cls.py --model all
"""

import argparse
import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

PROJECT_ROOT = Path('/Volume/DeCoFlow')
sys.path.insert(0, str(PROJECT_ROOT))

from decoflow.extractors import create_feature_extractor
from decoflow.data.mvtec import MVTEC, MVTEC_CLASS_NAMES
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================================
# V48_01 Configuration (matching DeCoFlow best)
# ============================================================================

ALL_CLASSES = list(MVTEC_CLASS_NAMES)  # 15 classes
BACKBONE_NAME = 'wide_resnet50_2'
EMBED_DIM = 768
IMG_SIZE = 224
MSK_SIZE = 256
NUM_EPOCHS = 60
BATCH_SIZE = 16
LR = 3e-4
DATA_PATH = '/Volume/MVTecAD'
LOG_DIR = PROJECT_ROOT / 'logs' / '5_Analysis' / 'Architecture_Comparison' / '15cls_V48config'
SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============================================================================
# Feature Extractor (shared, frozen)
# ============================================================================

def build_feature_extractor():
    """Build feature extractor matching V48_01."""
    extractor = create_feature_extractor(
        backbone_name=BACKBONE_NAME,
        input_shape=(3, IMG_SIZE, IMG_SIZE),
        target_embed_dimension=EMBED_DIM,
        device=DEVICE,
        patch_size=3,
        patch_stride=1,
        use_high_res=True,
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor


# ============================================================================
# Baseline Models — "Frozen Base + Task-Specific Adapter" Decomposition
# ============================================================================

class AutoencoderBaseline(nn.Module):
    """AE with frozen encoder after Task 0, task-specific LoRA-like adapters."""

    def __init__(self, feature_dim=EMBED_DIM, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, feature_dim),
        )

        # Task-specific adapters (LoRA-like: low-rank residual in bottleneck)
        self.task_adapters = nn.ModuleDict()
        self.current_task = 0

    def add_task(self, task_id: int):
        self.current_task = task_id
        adapter = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.hidden_dim),
        ).to(next(self.parameters()).device)
        nn.init.zeros_(adapter[2].weight)
        nn.init.zeros_(adapter[2].bias)
        self.task_adapters[str(task_id)] = adapter

        if task_id > 0:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

    def forward(self, features):
        B, H, W, D = features.shape
        x = features.reshape(B * H * W, D)
        z = self.encoder(x)
        if str(self.current_task) in self.task_adapters:
            z = z + self.task_adapters[str(self.current_task)](z)
        x_rec = self.decoder(z)
        return x_rec.reshape(B, H, W, D)

    def get_trainable_params(self):
        if self.current_task == 0:
            return list(self.parameters())
        return list(self.task_adapters[str(self.current_task)].parameters())


class VAEBaseline(nn.Module):
    """VAE with frozen encoder after Task 0."""

    def __init__(self, feature_dim=EMBED_DIM, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, feature_dim),
        )

        self.task_adapters = nn.ModuleDict()
        self.current_task = 0

    def add_task(self, task_id: int):
        self.current_task = task_id
        adapter = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.latent_dim),
        ).to(next(self.parameters()).device)
        nn.init.zeros_(adapter[2].weight)
        nn.init.zeros_(adapter[2].bias)
        self.task_adapters[str(task_id)] = adapter

        if task_id > 0:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.fc_mu.parameters():
                p.requires_grad = False
            for p in self.fc_logvar.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features):
        B, H, W, D = features.shape
        x = features.reshape(B * H * W, D)
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        if str(self.current_task) in self.task_adapters:
            z = z + self.task_adapters[str(self.current_task)](z)
        x_rec = self.decoder(z)
        return x_rec.reshape(B, H, W, D), mu, logvar

    def get_trainable_params(self):
        if self.current_task == 0:
            return list(self.parameters())
        return list(self.task_adapters[str(self.current_task)].parameters())


class TeacherStudentBaseline(nn.Module):
    """Teacher-Student with frozen teacher after Task 0."""

    def __init__(self, feature_dim=EMBED_DIM, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.teacher = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.student = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.task_adapters = nn.ModuleDict()
        self.current_task = 0

    def add_task(self, task_id: int):
        self.current_task = task_id
        adapter = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.hidden_dim),
        ).to(next(self.parameters()).device)
        nn.init.zeros_(adapter[2].weight)
        nn.init.zeros_(adapter[2].bias)
        self.task_adapters[str(task_id)] = adapter

        if task_id > 0:
            for p in self.teacher.parameters():
                p.requires_grad = False

    def forward(self, features):
        B, H, W, D = features.shape
        x = features.reshape(B * H * W, D)
        with torch.no_grad():
            t_out = self.teacher(x)
        s_out = self.student(x)
        if str(self.current_task) in self.task_adapters:
            s_out = s_out + self.task_adapters[str(self.current_task)](s_out)
        return t_out, s_out

    def get_trainable_params(self):
        if self.current_task == 0:
            return list(self.parameters())
        params = list(self.student.parameters())
        params += list(self.task_adapters[str(self.current_task)].parameters())
        return params


# ============================================================================
# Training and Evaluation
# ============================================================================

def create_dataloader(class_name, train=True):
    dataset = MVTEC(
        root=DATA_PATH,
        class_name=class_name,
        train=train,
        img_size=IMG_SIZE,
        crp_size=IMG_SIZE,
        msk_size=MSK_SIZE,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        drop_last=train,
    )


def train_one_task(model, model_name, task_id, class_name, feature_extractor):
    """Train model on a single task."""
    model.add_task(task_id)
    model.train()

    train_loader = create_dataloader(class_name, train=True)
    params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            images = batch[0].to(DEVICE)

            with torch.no_grad():
                features, spatial_shape = feature_extractor(images, return_spatial_shape=True)

            if model_name == 'ae':
                reconstructed = model(features)
                loss = F.mse_loss(reconstructed, features)
            elif model_name == 'vae':
                reconstructed, mu, logvar = model(features)
                recon_loss = F.mse_loss(reconstructed, features)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.001 * kl_loss
            elif model_name == 'ts':
                t_out, s_out = model(features)
                loss = F.mse_loss(s_out, t_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"    Epoch {epoch+1:3d}/{NUM_EPOCHS}: loss={avg_loss:.6f}")


@torch.no_grad()
def evaluate_task(model, model_name, task_id, class_name, feature_extractor):
    """Evaluate model on a single task (test set)."""
    model.eval()
    model.current_task = task_id

    test_loader = create_dataloader(class_name, train=False)

    all_image_scores = []
    all_pixel_scores = []
    all_labels = []
    all_masks = []

    for batch in test_loader:
        images, labels, masks, _, _ = batch
        images = images.to(DEVICE)

        features, spatial_shape = feature_extractor(images, return_spatial_shape=True)
        H, W = spatial_shape

        if model_name == 'ae':
            reconstructed = model(features)
            patch_scores = ((features - reconstructed) ** 2).mean(dim=-1)  # (B, H, W)
        elif model_name == 'vae':
            reconstructed, mu, logvar = model(features)
            patch_scores = ((features - reconstructed) ** 2).mean(dim=-1)
        elif model_name == 'ts':
            t_out, s_out = model(features)
            B_cur = features.shape[0]
            diff = (t_out - s_out).reshape(B_cur, H, W, -1)
            patch_scores = (diff ** 2).mean(dim=-1)  # (B, H, W)

        # Image score: mean of top-k patches
        flat_scores = patch_scores.reshape(patch_scores.shape[0], -1)
        topk_scores, _ = flat_scores.topk(k=min(3, flat_scores.shape[1]), dim=1)
        image_scores = topk_scores.mean(dim=1)

        # Upscale for pixel metrics
        pixel_scores = F.interpolate(
            patch_scores.unsqueeze(1), size=(MSK_SIZE, MSK_SIZE),
            mode='bicubic', align_corners=False
        ).squeeze(1)

        all_image_scores.append(image_scores.cpu().numpy())
        all_pixel_scores.append(pixel_scores.cpu().numpy())
        all_labels.append(labels.numpy())
        all_masks.append(masks.numpy())

    all_image_scores = np.concatenate(all_image_scores)
    all_pixel_scores = np.concatenate(all_pixel_scores)
    all_labels = np.concatenate(all_labels)
    all_masks = np.concatenate(all_masks)

    # Image AUC
    image_auc = roc_auc_score(all_labels, all_image_scores) if len(np.unique(all_labels)) > 1 else 0.5

    # Pixel AP
    masks_flat = all_masks.reshape(-1)
    scores_flat = all_pixel_scores.reshape(-1)
    if masks_flat.sum() > 0:
        pixel_ap = average_precision_score(masks_flat, scores_flat)
    else:
        pixel_ap = 0.0

    return image_auc, pixel_ap


def run_experiment(model_name: str):
    """Run full 15-class continual experiment for one architecture."""
    print(f"\n{'='*70}")
    print(f"  Cross-Architecture Comparison: {model_name.upper()}")
    print(f"  15 classes, V48 config (WRN50, 768-dim, high_res, 60ep)")
    print(f"{'='*70}\n")

    set_seed(SEED)

    # Build feature extractor
    print("Building feature extractor (WRN50 + high_res)...")
    feature_extractor = build_feature_extractor()

    # Build model
    if model_name == 'ae':
        model = AutoencoderBaseline().to(DEVICE)
    elif model_name == 'vae':
        model = VAEBaseline().to(DEVICE)
    elif model_name == 'ts':
        model = TeacherStudentBaseline().to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Track metrics
    results = {
        'model': model_name,
        'config': {
            'backbone': BACKBONE_NAME,
            'embed_dim': EMBED_DIM,
            'img_size': IMG_SIZE,
            'num_epochs': NUM_EPOCHS,
            'lr': LR,
            'classes': ALL_CLASSES,
        },
        'per_task_after_all': {},
        'initial_aucs': {},
    }

    initial_aucs = {}

    for task_id, class_name in enumerate(ALL_CLASSES):
        print(f"\n--- Task {task_id}: {class_name} ---")

        # Train
        t0 = time.time()
        train_one_task(model, model_name, task_id, class_name, feature_extractor)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate on current task (initial AUC)
        iauc, pap = evaluate_task(model, model_name, task_id, class_name, feature_extractor)
        initial_aucs[task_id] = iauc
        print(f"  Initial I-AUC: {iauc*100:.2f}%, P-AP: {pap*100:.2f}%")

    # Final evaluation on ALL tasks
    print(f"\n{'='*50}")
    print("Final evaluation on all 15 tasks...")
    print(f"{'='*50}")

    final_aucs = []
    final_paps = []
    forgetting_measures = []

    for task_id, class_name in enumerate(ALL_CLASSES):
        iauc, pap = evaluate_task(model, model_name, task_id, class_name, feature_extractor)
        final_aucs.append(iauc)
        final_paps.append(pap)

        fm = max(0, initial_aucs[task_id] - iauc)
        forgetting_measures.append(fm)

        results['per_task_after_all'][class_name] = {
            'image_auc': iauc,
            'pixel_ap': pap,
            'initial_auc': initial_aucs[task_id],
            'forgetting': fm,
        }

        print(f"  Task {task_id:2d} [{class_name:12s}]: "
              f"I-AUC={iauc*100:.2f}% (init={initial_aucs[task_id]*100:.2f}%), "
              f"P-AP={pap*100:.2f}%, FM={fm*100:.2f}pp")

    avg_iauc = np.mean(final_aucs) * 100
    avg_pap = np.mean(final_paps) * 100
    avg_fm = np.mean(forgetting_measures) * 100

    results['summary'] = {
        'avg_image_auc': avg_iauc,
        'avg_pixel_ap': avg_pap,
        'avg_forgetting_measure': avg_fm,
    }

    print(f"\n{'='*50}")
    print(f"  {model_name.upper()} SUMMARY:")
    print(f"  Avg I-AUC: {avg_iauc:.2f}%")
    print(f"  Avg P-AP:  {avg_pap:.2f}%")
    print(f"  Avg FM:    {avg_fm:.2f}%p")
    print(f"{'='*50}")

    # Save results
    out_dir = LOG_DIR / f'{model_name}_baseline'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_dir / 'results.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Cross-Architecture Comparison (Table 3)')
    parser.add_argument('--model', type=str, required=True,
                        choices=['ae', 'vae', 'ts', 'all'],
                        help='Model to run')
    args = parser.parse_args()

    if args.model == 'all':
        for m in ['ae', 'vae', 'ts']:
            run_experiment(m)
    else:
        run_experiment(args.model)


if __name__ == '__main__':
    main()
