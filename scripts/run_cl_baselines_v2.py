#!/usr/bin/env python3
"""
CL Baselines Script for DeCoFlow.

Implements standard continual learning baselines (Finetune, EWC, LwF, Replay)
on the same NF backbone used by DeCoFlow, but WITHOUT LoRA/ACL/TaskSpecificAlignment.
This provides a fair comparison: same architecture, same training setup,
different CL strategy.

Methods:
  - finetune: Naive sequential training (no CL mechanism, catastrophic forgetting baseline)
  - ewc: Elastic Weight Consolidation (Fisher-based quadratic penalty)
  - lwf: Learning without Forgetting (MSE distillation on z and logdet)
  - replay: Experience Replay (store feature exemplars, replay during training)

Usage:
  python scripts/run_cl_baselines_v2.py --method ewc --dataset mvtec --use_high_res
"""

import os
import sys
import copy
import math
import json
import time
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoflow.extractors import create_feature_extractor
from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.models.position_embedding import positionalencoding2d
from decoflow.config.ablation import AblationConfig
from decoflow.data.datasets import get_dataset_class


# ============================================================================
# Constants
# ============================================================================

MVTEC_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
    "pcb4", "pipe_fryum"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logger with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("cl_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================================
# EWC: Fisher Information
# ============================================================================

class EWCRegularizer:
    """Elastic Weight Consolidation with diagonal Fisher information."""

    def __init__(self, ewc_lambda: float = 1000.0):
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = {}
        self.param_star_dict = {}

    @torch.no_grad()
    def compute_fisher(self, model, dataloader, feature_extractor, device, spatial_shape, pos_embed):
        """Compute diagonal Fisher information from training data."""
        model.train()
        fisher_diag = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_diag[name] = torch.zeros_like(param)

        n_samples = 0
        for batch in dataloader:
            images = batch[0].to(device)
            with torch.no_grad():
                features = feature_extractor(images)
                B, H, W, D = features.shape
                features_with_pos = features + pos_embed[:, :H, :W, :].to(device)

            # Need gradients for Fisher
            torch.set_grad_enabled(True)
            z, logdet_patch = model(features_with_pos, reverse=False)
            log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            nll = -(log_pz + logdet_patch).mean()

            model.zero_grad()
            nll.backward()
            torch.set_grad_enabled(False)

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_diag[name] += param.grad.data ** 2

            n_samples += images.shape[0]

        for name in fisher_diag:
            fisher_diag[name] /= max(n_samples, 1)

        for name in fisher_diag:
            if name in self.fisher_dict:
                self.fisher_dict[name] = self.fisher_dict[name] + fisher_diag[name]
            else:
                self.fisher_dict[name] = fisher_diag[name].clone()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_star_dict[name] = param.data.clone()

    def penalty(self, model):
        """Compute EWC penalty: sum_i F_i * (theta_i - theta_star_i)^2."""
        if not self.fisher_dict:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher_dict and param.requires_grad:
                fisher = self.fisher_dict[name]
                param_star = self.param_star_dict[name]
                loss = loss + (fisher * (param - param_star) ** 2).sum()

        return 0.5 * self.ewc_lambda * loss


# ============================================================================
# LwF: Learning without Forgetting
# ============================================================================

class LwFDistiller:
    """Learning without Forgetting via MSE distillation on z and logdet."""

    def __init__(self, lwf_lambda: float = 1.0):
        self.lwf_lambda = lwf_lambda
        self.old_model = None

    def snapshot(self, model):
        """Take a snapshot of the current model as the teacher."""
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

    def distillation_loss(self, model, features_with_pos):
        """Compute MSE distillation loss between old and new model outputs."""
        if self.old_model is None:
            return torch.tensor(0.0, device=features_with_pos.device)

        with torch.no_grad():
            z_old, logdet_old = self.old_model(features_with_pos, reverse=False)

        z_new, logdet_new = model(features_with_pos, reverse=False)

        z_loss = F.mse_loss(z_new, z_old)
        logdet_loss = F.mse_loss(logdet_new, logdet_old)

        return self.lwf_lambda * (z_loss + logdet_loss)


# ============================================================================
# Replay: Feature Exemplar Memory
# ============================================================================

class FeatureReplayBuffer:
    """Store feature exemplars from previous tasks for replay."""

    def __init__(self, replay_size: int = 500):
        self.replay_size = replay_size
        self.buffer = {}

    @torch.no_grad()
    def store(self, task_id, dataloader, feature_extractor, device, pos_embed):
        """Store random feature exemplars from a task's training data."""
        all_features = []
        for batch in dataloader:
            images = batch[0].to(device)
            features = feature_extractor(images)
            B, H, W, D = features.shape
            features_with_pos = features + pos_embed[:, :H, :W, :].to(device)
            all_features.append(features_with_pos.cpu())

        all_features = torch.cat(all_features, dim=0)
        n_total = all_features.shape[0]
        n_store = min(self.replay_size, n_total)

        indices = torch.randperm(n_total)[:n_store]
        self.buffer[task_id] = all_features[indices]

    def get_replay_batch(self, batch_size, device):
        """Sample a random replay batch from all stored tasks."""
        if not self.buffer:
            return None

        all_stored = torch.cat(list(self.buffer.values()), dim=0)
        n = all_stored.shape[0]
        if n == 0:
            return None

        indices = torch.randperm(n)[:min(batch_size, n)]
        return all_stored[indices].to(device)


# ============================================================================
# Training Utilities
# ============================================================================

def compute_nll_loss(z, logdet_patch, D, lambda_logdet=1e-4):
    """Compute NLL loss from NF output."""
    log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
    nll_patch = -(log_pz + logdet_patch)
    loss = nll_patch.mean()

    if lambda_logdet > 0:
        logdet_reg = lambda_logdet * (logdet_patch ** 2).mean()
        loss = loss + logdet_reg

    return loss


def compute_anomaly_scores(z, logdet_patch, D):
    """Compute anomaly scores from NF output."""
    log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
    patch_scores = -(log_pz + logdet_patch)
    B = patch_scores.shape[0]
    flat_scores = patch_scores.reshape(B, -1)
    topk_scores, _ = flat_scores.topk(min(3, flat_scores.shape[1]), dim=1)
    image_scores = topk_scores.mean(dim=1)
    return patch_scores, image_scores


# ============================================================================
# Training Loop
# ============================================================================

def train_task(model, feature_extractor, train_loader, device, args, logger,
               task_id, class_name, spatial_shape, pos_embed,
               ewc_reg=None, lwf_dist=None, replay_buf=None):
    """Train the model on a single task."""
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    D = model.embed_dim
    H, W = spatial_shape

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_reg = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch[0].to(device)

            with torch.no_grad():
                features = feature_extractor(images)
                B_cur, H_cur, W_cur, D_cur = features.shape
                features_with_pos = features + pos_embed[:, :H_cur, :W_cur, :].to(device)

            z, logdet_patch = model(features_with_pos, reverse=False)
            nll_loss = compute_nll_loss(z, logdet_patch, D, args.lambda_logdet)
            total_loss = nll_loss
            epoch_nll += nll_loss.item()

            reg_loss = torch.tensor(0.0, device=device)

            if args.method == "ewc" and ewc_reg is not None:
                ewc_penalty = ewc_reg.penalty(model)
                reg_loss = reg_loss + ewc_penalty

            if args.method == "lwf" and lwf_dist is not None:
                distill_loss = lwf_dist.distillation_loss(model, features_with_pos)
                reg_loss = reg_loss + distill_loss

            if args.method == "replay" and replay_buf is not None and task_id > 0:
                replay_batch = replay_buf.get_replay_batch(args.batch_size, device)
                if replay_batch is not None:
                    z_rep, logdet_rep = model(replay_batch, reverse=False)
                    replay_nll = compute_nll_loss(z_rep, logdet_rep, D, args.lambda_logdet)
                    reg_loss = reg_loss + replay_nll

            total_loss = total_loss + reg_loss
            epoch_reg += reg_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_nll = epoch_nll / max(n_batches, 1)
        avg_reg = epoch_reg / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_current = scheduler.get_last_lr()[0]
            logger.info(
                "  Task %d [%s] Epoch %d/%d: Loss=%.4f (NLL=%.4f, Reg=%.4f) LR=%.2e",
                task_id, class_name, epoch + 1, args.num_epochs,
                avg_loss, avg_nll, avg_reg, lr_current
            )

    return model


@torch.no_grad()
def evaluate_task(model, feature_extractor, test_loader, device, class_name,
                  spatial_shape, pos_embed, msk_size):
    """Evaluate the model on a single task's test set."""
    model.eval()
    D = model.embed_dim
    H, W = spatial_shape

    all_image_scores = []
    all_image_labels = []
    all_pixel_scores = []
    all_pixel_labels = []

    for batch in test_loader:
        images, labels, masks, _, _ = batch
        images = images.to(device)

        features = feature_extractor(images)
        B, H_cur, W_cur, D_cur = features.shape
        features_with_pos = features + pos_embed[:, :H_cur, :W_cur, :].to(device)

        z, logdet_patch = model(features_with_pos, reverse=False)
        patch_scores, image_scores = compute_anomaly_scores(z, logdet_patch, D)

        all_image_scores.append(image_scores.cpu().numpy())
        all_image_labels.append(labels.numpy())

        score_map = patch_scores.unsqueeze(1)
        score_map_up = F.interpolate(
            score_map, size=(msk_size, msk_size),
            mode="bicubic", align_corners=False
        ).squeeze(1)

        mask_flat = masks.squeeze(1).numpy()
        score_flat = score_map_up.cpu().numpy()

        for i in range(B):
            all_pixel_scores.append(score_flat[i].ravel())
            all_pixel_labels.append((mask_flat[i] > 0.5).astype(int).ravel())

    all_image_scores = np.concatenate(all_image_scores)
    all_image_labels = np.concatenate(all_image_labels)
    all_pixel_scores = np.concatenate(all_pixel_scores)
    all_pixel_labels = np.concatenate(all_pixel_labels)

    image_labels_bin = (all_image_labels > 0).astype(int)

    results = {}
    try:
        results["image_auc"] = roc_auc_score(image_labels_bin, all_image_scores) * 100
    except ValueError:
        results["image_auc"] = 0.0
    try:
        results["pixel_auc"] = roc_auc_score(all_pixel_labels, all_pixel_scores) * 100
    except ValueError:
        results["pixel_auc"] = 0.0
    try:
        results["image_ap"] = average_precision_score(image_labels_bin, all_image_scores) * 100
    except ValueError:
        results["image_ap"] = 0.0
    try:
        results["pixel_ap"] = average_precision_score(all_pixel_labels, all_pixel_scores) * 100
    except ValueError:
        results["pixel_ap"] = 0.0

    return results


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CL Baselines for DeCoFlow NF Backbone")

    parser.add_argument("--method", type=str, required=True,
                        choices=["finetune", "ewc", "lwf", "replay"],
                        help="CL method: finetune, ewc, lwf, or replay")
    parser.add_argument("--dataset", type=str, default="mvtec",
                        choices=["mvtec", "visa"],
                        help="Dataset to use (default: mvtec)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset root")
    parser.add_argument("--use_high_res", action="store_true",
                        help="Use high-resolution features (56x56 from layer1)")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2",
                        help="Backbone name (default: wide_resnet50_2)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--msk_size", type=int, default=224,
                        help="Mask size for pixel-level evaluation (default: 224)")
    parser.add_argument("--num_coupling_layers", type=int, default=6,
                        help="Number of coupling layers in NF (default: 6)")
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="Embedding dimension (default: 768, auto-detected from backbone)")
    parser.add_argument("--num_epochs", type=int, default=60,
                        help="Number of training epochs per task (default: 60)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lambda_logdet", type=float, default=1e-4,
                        help="Logdet regularization weight (default: 1e-4)")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0,
                        help="EWC regularization strength (default: 1000)")
    parser.add_argument("--lwf_lambda", type=float, default=1.0,
                        help="LwF distillation loss weight (default: 1.0)")
    parser.add_argument("--replay_size", type=int, default=500,
                        help="Number of exemplars per task for replay (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (auto-generated if None)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (default: ./logs/{experiment_name})")

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine dataset
    if args.dataset == "mvtec":
        task_classes = MVTEC_CLASSES
        if args.data_path is None:
            args.data_path = "/Volume/MVTecAD"
    else:
        task_classes = VISA_CLASSES
        if args.data_path is None:
            args.data_path = "/Volume/VisA"

    # Experiment name
    res_tag = "HR" if args.use_high_res else "LR"
    if args.experiment_name is None:
        args.experiment_name = "V54_%s_%s_%s" % (args.method, args.dataset, res_tag)

    if args.log_dir is None:
        args.log_dir = os.path.join("./logs", args.experiment_name)

    logger = setup_logging(args.log_dir)
    separator = "=" * 70
    logger.info(separator)
    logger.info("CL Baseline: %s", args.method.upper())
    logger.info("Dataset: %s (%d classes)", args.dataset, len(task_classes))
    logger.info("Resolution: %s | Backbone: %s", res_tag, args.backbone)
    logger.info("Epochs: %d | LR: %s | Batch: %d", args.num_epochs, args.lr, args.batch_size)
    logger.info("Experiment: %s", args.experiment_name)
    logger.info("Log dir: %s", args.log_dir)
    logger.info(separator)

    # Save config
    config_dict = vars(args)
    config_dict["task_classes"] = task_classes
    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Create feature extractor (frozen)
    logger.info("Creating feature extractor: %s (high_res=%s)", args.backbone, args.use_high_res)
    input_shape = (3, args.img_size, args.img_size)
    feature_extractor = create_feature_extractor(
        backbone_name=args.backbone,
        input_shape=input_shape,
        target_embed_dimension=args.embed_dim,
        device=str(device),
        use_high_res=args.use_high_res,
    )
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = False

    # Determine embed_dim and spatial_shape from a dummy forward pass
    logger.info("Running dummy forward pass to determine feature dimensions...")
    dummy_input = torch.randn(1, *input_shape).to(device)
    with torch.no_grad():
        dummy_features = feature_extractor(dummy_input)
    _, H_feat, W_feat, D_feat = dummy_features.shape
    spatial_shape = (H_feat, W_feat)
    embed_dim = D_feat
    logger.info("Feature shape: (B, %d, %d, %d)", H_feat, W_feat, embed_dim)
    del dummy_input, dummy_features

    # Create positional embedding
    pos_embed_hw = max(H_feat, W_feat)
    pos_embed_2d = positionalencoding2d(embed_dim, pos_embed_hw, pos_embed_hw)
    pos_embed = pos_embed_2d.permute(1, 2, 0).unsqueeze(0).to(device)

    # Create NF model (baseline: no LoRA, no ACL, no TaskSpecificAlignment)
    ablation_cfg = AblationConfig(
        use_lora=False,
        use_tsa=False,
        use_acl=False,
        use_router=False,
        use_task_adapter=False,
        use_pos_embedding=True,
        use_spatial_context=True,
        use_scale_context=True,
        scale_context_kernel=5,
        lambda_logdet=args.lambda_logdet,
        score_smooth_sigma=0.0,
    )

    model = DeCoFlowNF(
        embed_dim=embed_dim,
        coupling_layers=args.num_coupling_layers,
        clamp_alpha=1.9,
        lora_rank=64,
        lora_alpha=1.0,
        device=str(device),
        ablation_config=ablation_cfg,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("NF Model: %d total params", total_params)
    logger.info("Trainable: %d params", trainable_params)

    # Initialize CL method
    ewc_reg = None
    lwf_dist = None
    replay_buf = None

    if args.method == "ewc":
        ewc_reg = EWCRegularizer(ewc_lambda=args.ewc_lambda)
        logger.info("EWC: lambda=%s", args.ewc_lambda)
    elif args.method == "lwf":
        lwf_dist = LwFDistiller(lwf_lambda=args.lwf_lambda)
        logger.info("LwF: lambda=%s", args.lwf_lambda)
    elif args.method == "replay":
        replay_buf = FeatureReplayBuffer(replay_size=args.replay_size)
        logger.info("Replay: %d exemplars/task", args.replay_size)

    # Dataset class
    dataset_cls = get_dataset_class(args.dataset)

    # Results storage
    all_results = {}
    task_train_times = {}

    # ========================================================================
    # Continual Learning Loop
    # ========================================================================
    for task_id, class_name in enumerate(task_classes):
        sep60 = "=" * 60
        logger.info("")
        logger.info(sep60)
        logger.info("Task %d/%d: %s", task_id, len(task_classes) - 1, class_name)
        logger.info(sep60)

        # Add task to model (handles internal state)
        model.add_task(task_id, class_name=class_name)
        model.current_task_id = task_id

        # Create train dataset & loader
        train_dataset = dataset_cls(
            root=args.data_path,
            class_name=class_name,
            train=True,
            img_size=args.img_size,
            crp_size=args.img_size,
            msk_size=args.msk_size,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True
        )

        logger.info("  Train samples: %d", len(train_dataset))

        # LwF: snapshot model before training on new task
        if args.method == "lwf" and task_id > 0:
            lwf_dist.snapshot(model)

        # Train
        t_start = time.time()
        model = train_task(
            model, feature_extractor, train_loader, device, args, logger,
            task_id, class_name, spatial_shape, pos_embed,
            ewc_reg=ewc_reg, lwf_dist=lwf_dist, replay_buf=replay_buf
        )
        t_elapsed = time.time() - t_start
        task_train_times[task_id] = t_elapsed
        logger.info("  Training time: %.1fs", t_elapsed)

        # EWC: compute Fisher after training
        if args.method == "ewc":
            logger.info("  Computing Fisher information...")
            ewc_reg.compute_fisher(model, train_loader, feature_extractor, device, spatial_shape, pos_embed)

        # Replay: store exemplars after training
        if args.method == "replay":
            logger.info("  Storing %d replay exemplars...", args.replay_size)
            replay_buf.store(task_id, train_loader, feature_extractor, device, pos_embed)

        # Evaluate on ALL tasks seen so far
        logger.info("  Evaluating on tasks 0..%d:", task_id)
        task_results = {}
        for eval_tid in range(task_id + 1):
            eval_class = task_classes[eval_tid]
            model.current_task_id = eval_tid

            test_dataset = dataset_cls(
                root=args.data_path,
                class_name=eval_class,
                train=False,
                img_size=args.img_size,
                crp_size=args.img_size,
                msk_size=args.msk_size,
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=4, pin_memory=True
            )

            metrics = evaluate_task(
                model, feature_extractor, test_loader, device,
                eval_class, spatial_shape, pos_embed, args.msk_size
            )
            task_results[eval_class] = metrics
            logger.info(
                "    [%s] ImgAUC=%.2f%% PixAP=%.2f%% PixAUC=%.2f%%",
                eval_class, metrics["image_auc"], metrics["pixel_ap"], metrics["pixel_auc"]
            )

        all_results[task_id] = task_results

    # ========================================================================
    # Final Evaluation (all classes after all tasks trained)
    # ========================================================================
    separator70 = "=" * 70
    logger.info("")
    logger.info(separator70)
    logger.info("FINAL EVALUATION (after all %d tasks)", len(task_classes))
    logger.info(separator70)

    final_metrics = {}
    for task_id, class_name in enumerate(task_classes):
        model.current_task_id = task_id

        test_dataset = dataset_cls(
            root=args.data_path,
            class_name=class_name,
            train=False,
            img_size=args.img_size,
            crp_size=args.img_size,
            msk_size=args.msk_size,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )

        metrics = evaluate_task(
            model, feature_extractor, test_loader, device,
            class_name, spatial_shape, pos_embed, args.msk_size
        )
        final_metrics[class_name] = metrics
        final_metrics[class_name]["task_id"] = task_id
        logger.info(
            "  Task %2d [%-15s] ImgAUC=%.2f%% PixAUC=%.2f%% ImgAP=%.2f%% PixAP=%.2f%%",
            task_id, class_name,
            metrics["image_auc"], metrics["pixel_auc"],
            metrics["image_ap"], metrics["pixel_ap"]
        )

    # Compute averages
    avg_img_auc = np.mean([m["image_auc"] for m in final_metrics.values()])
    avg_pix_auc = np.mean([m["pixel_auc"] for m in final_metrics.values()])
    avg_img_ap = np.mean([m["image_ap"] for m in final_metrics.values()])
    avg_pix_ap = np.mean([m["pixel_ap"] for m in final_metrics.values()])

    logger.info("")
    logger.info(
        "  AVERAGE: ImgAUC=%.2f%% PixAUC=%.2f%% ImgAP=%.2f%% PixAP=%.2f%%",
        avg_img_auc, avg_pix_auc, avg_img_ap, avg_pix_ap
    )

    # ========================================================================
    # Save Results: final_results.csv
    # ========================================================================
    results_path = os.path.join(args.log_dir, "final_results.csv")
    with open(results_path, "w") as f:
        f.write("Task ID,Class Name,Routing Acc,Image AUC,Pixel AUC,Image AP,Pixel AP\n")
        for class_name, m in final_metrics.items():
            f.write("%s,%s,N/A,%.4f,%.4f,%.4f,%.4f\n" % (
                m["task_id"], class_name,
                m["image_auc"], m["pixel_auc"],
                m["image_ap"], m["pixel_ap"]
            ))
        f.write("AVG,AVERAGE,N/A,%.4f,%.4f,%.4f,%.4f\n" % (
            avg_img_auc, avg_pix_auc, avg_img_ap, avg_pix_ap
        ))

    logger.info("")
    logger.info("Final results saved to: %s", results_path)

    # ========================================================================
    # Save Results: forgetting_analysis.csv
    # ========================================================================
    forgetting_path = os.path.join(args.log_dir, "forgetting_analysis.csv")
    with open(forgetting_path, "w") as f:
        f.write("Task ID,Class Name,ImgAUC After Own Training,ImgAUC Final,Forgetting (pp)\n")
        for task_id, class_name in enumerate(task_classes):
            if task_id in all_results and class_name in all_results[task_id]:
                after_own = all_results[task_id][class_name]["image_auc"]
            else:
                after_own = float("nan")
            final_val = final_metrics[class_name]["image_auc"]
            forgetting = after_own - final_val
            f.write("%d,%s,%.4f,%.4f,%.4f\n" % (task_id, class_name, after_own, final_val, forgetting))

        forgetting_vals = []
        for task_id, class_name in enumerate(task_classes):
            if task_id in all_results and class_name in all_results[task_id]:
                after_own = all_results[task_id][class_name]["image_auc"]
                final_val = final_metrics[class_name]["image_auc"]
                forgetting_vals.append(after_own - final_val)
        avg_forgetting = np.mean(forgetting_vals) if forgetting_vals else float("nan")
        f.write("AVG,AVERAGE,,,%.4f\n" % avg_forgetting)

    logger.info("Forgetting analysis saved to: %s", forgetting_path)
    logger.info("Average forgetting: %.2f pp", avg_forgetting)
    logger.info("")
    logger.info("Done! Total training time: %.1fs", sum(task_train_times.values()))


if __name__ == "__main__":
    main()
