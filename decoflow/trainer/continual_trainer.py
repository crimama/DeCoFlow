"""
DeCoFlow Continual Learning Trainer.

Implements Stage-Separated Training:
- Stage 1 (FAST): LoRA + InputAdapter adaptation (base frozen)
- Stage 2 (SLOW): Base NF consolidation (optional)

Ablation Support:
- use_lora: Enable/disable LoRA adaptation
- use_router: Enable/disable prototype router
- use_pos_embedding: Enable/disable positional embedding
- use_task_adapter: Enable/disable task input adapters

Optimizer: AdamP (Adaptive Momentum with Decoupled Weight Decay)
- Better generalization than AdamW through projection-based gradient correction
- Hyperparameters tuned for normalizing flow training
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, TYPE_CHECKING

try:
    from adamp import AdamP
except ImportError:
    # Fallback to AdamW if adamp is not installed
    AdamP = None
    print("Warning: adamp not installed. Install with: pip install adamp")

from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.models.routing import TaskPrototype, PrototypeRouter, ClassPrototype, ClassRouter
from decoflow.utils.logger import TrainingLogger
from decoflow.utils.replay import OrthogonalGradientProjection

if TYPE_CHECKING:
    from decoflow.config.ablation import AblationConfig


# AdamP hyperparameters for DeCoFlow
ADAMP_CONFIG = {
    'betas': (0.9, 0.999),      # Momentum coefficients
    'weight_decay': 0.01,        # Decoupled weight decay (higher than AdamW default)
    'delta': 0.1,                # Threshold for gradient projection
    'wd_ratio': 0.1,             # Weight decay ratio for decoupled WD
    'nesterov': False,           # Nesterov momentum
}


def create_optimizer(params, lr: float, weight_decay_scale: float = 1.0):
    """
    Create optimizer for training.

    Uses AdamP if available, otherwise falls back to AdamW.

    Args:
        params: Model parameters to optimize
        lr: Learning rate
        weight_decay_scale: Scale factor for weight decay (default 1.0)

    Returns:
        Optimizer instance
    """
    if AdamP is not None:
        return AdamP(
            params,
            lr=lr,
            betas=ADAMP_CONFIG['betas'],
            weight_decay=ADAMP_CONFIG['weight_decay'] * weight_decay_scale,
            delta=ADAMP_CONFIG['delta'],
            wd_ratio=ADAMP_CONFIG['wd_ratio'],
            nesterov=ADAMP_CONFIG['nesterov']
        )
    else:
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4 * weight_decay_scale)


class DeCoFlowContinualTrainer:
    """
    Slow-Fast DeCoFlow Continual Learning Trainer.

    Implements Stage-Separated Training:
    - Stage 1 (FAST): LoRA + InputAdapter adaptation (base frozen)
    - Stage 2 (SLOW): Base NF consolidation (optional)

    Key Design Philosophy:
    - NF base = Slow learner (global normality manifold)
    - LoRA + InputAdapter = Fast learner (task-specific adaptation)

    Ablation Support:
    - use_lora: If False, skip LoRA (train shared NF)
    - use_router: If False, use oracle task_id
    - use_pos_embedding: If False, skip positional embedding
    - use_mahalanobis: If False, use Euclidean distance in router
    """

    def __init__(self,
                 vit_extractor,
                 pos_embed_generator,
                 nf_model: DeCoFlowNF,
                 args,
                 device: str = 'cuda',
                 slow_lr_ratio: float = 0.2,
                 slow_blocks_k: int = 2,
                 ablation_config: 'AblationConfig' = None,
                 logger: Optional[TrainingLogger] = None):

        self.vit_extractor = vit_extractor
        self.pos_embed_generator = pos_embed_generator
        self.nf_model = nf_model
        self.args = args
        self.device = device

        # Ablation configuration
        self.ablation_config = ablation_config
        if ablation_config is None:
            # Default: all components enabled
            self.use_lora = True
            self.use_router = True
            self.use_pos_embedding = True
            self.use_mahalanobis = True
            self.lambda_logdet = 0.0
            # V3: OGP defaults
            self.use_ogp = False
            self.ogp_threshold = 0.99
            self.ogp_max_rank = 50
            self.ogp_n_samples = 300
            # V5: Score aggregation defaults
            self.score_aggregation_mode = "percentile"
            self.score_aggregation_percentile = 0.99
            self.score_aggregation_top_k = 10
            self.score_aggregation_top_k_percent = 0.05
            # V5: Structural improvements defaults
            self.use_tail_aware_loss = False
            self.tail_weight = 0.3
            self.tail_top_k_ratio = 0.05
            self.tail_match_eval_topk = False
            self.tail_loss_blend_mode = 'mixed'
            self.focal_gamma = 0.0
            self.cluster_weight = 0.5
            self.cluster_high_score_percentile = 0.9
            # V5.5: Position-Agnostic defaults
            self.use_relative_position = False
            self.use_dual_branch = False
            self.use_local_consistency = False
            self.local_consistency_kernel = 3
            self.local_consistency_temperature = 1.0
            # V5.6: Improved Position-Agnostic defaults
            self.use_improved_dual_branch = False
            self.use_score_guided_dual = False
            self.use_multiscale_consistency = False
            # V5.7: Rotation-Invariant PE defaults
            self.use_multi_orientation = False
            self.multi_orientation_n = 4
            self.use_content_based_pe = False
            self.use_hybrid_pe = False
            # V5.8: TAPE defaults
            self.use_tape = False
            # V6.1: STN defaults
            self.use_stn = False
            self.stn_mode = 'rotation'
            self.stn_hidden_dim = 128
            self.stn_dropout = 0.1
            self.stn_rotation_reg_weight = 0.01
            self.stn_pretrain_epochs = 0
            # V6: Task-separated defaults
            self.use_task_separated = False
            self.use_regular_linear = False
            self.use_spectral_norm = False
            # V7: DSM defaults
            self.use_dsm = False
            self.dsm_mode = 'hybrid'
            self.dsm_alpha = 0.5
            self.dsm_sigma_min = 0.01
            self.dsm_sigma_max = 1.0
            self.dsm_n_projections = 1
            self.dsm_use_sliced = True
            self.dsm_noise_mode = 'geometric'
            self.dsm_clean_penalty = 0.0
        else:
            self.use_lora = ablation_config.use_lora
            self.use_router = ablation_config.use_router
            self.use_pos_embedding = ablation_config.use_pos_embedding
            self.use_mahalanobis = ablation_config.use_mahalanobis
            self.lambda_logdet = ablation_config.lambda_logdet
            # V45: ACL gating and regularization
            self.acl_gate_l1_lambda = getattr(ablation_config, 'acl_gate_l1_lambda', 0.0)
            self.acl_weight_decay = getattr(ablation_config, 'acl_weight_decay', 0.0)
            self.spatial_var_lambda = getattr(ablation_config, 'spatial_var_lambda', 0.0)
            # V3: OGP settings
            self.use_ogp = ablation_config.use_ogp
            self.ogp_threshold = ablation_config.ogp_threshold
            self.ogp_max_rank = ablation_config.ogp_max_rank
            self.ogp_n_samples = ablation_config.ogp_n_samples
            # V5: Score aggregation settings
            self.score_aggregation_mode = ablation_config.score_aggregation_mode
            self.score_aggregation_percentile = ablation_config.score_aggregation_percentile
            self.score_aggregation_top_k = ablation_config.score_aggregation_top_k
            self.score_aggregation_top_k_percent = ablation_config.score_aggregation_top_k_percent
            # V5: Structural improvements settings
            self.use_tail_aware_loss = ablation_config.use_tail_aware_loss
            self.tail_weight = ablation_config.tail_weight
            self.tail_top_k_ratio = ablation_config.tail_top_k_ratio
            self.tail_match_eval_topk = getattr(ablation_config, 'tail_match_eval_topk', False)
            self.tail_loss_blend_mode = getattr(ablation_config, 'tail_loss_blend_mode', 'mixed')
            self.focal_gamma = getattr(ablation_config, 'focal_gamma', 0.0)
            self.cluster_weight = ablation_config.cluster_weight
            self.cluster_high_score_percentile = ablation_config.cluster_high_score_percentile
            # Multi-task loss settings
            self.use_multitask_loss = getattr(ablation_config, 'use_multitask_loss', False)
            self.multitask_alpha = getattr(ablation_config, 'multitask_alpha', 0.7)
            self.multitask_beta = getattr(ablation_config, 'multitask_beta', 0.3)
            # V5.5: Position-Agnostic settings
            self.use_relative_position = ablation_config.use_relative_position
            self.use_dual_branch = ablation_config.use_dual_branch
            self.use_local_consistency = ablation_config.use_local_consistency
            self.local_consistency_kernel = ablation_config.local_consistency_kernel
            self.local_consistency_temperature = ablation_config.local_consistency_temperature
            # V5.6: Improved Position-Agnostic settings
            self.use_improved_dual_branch = ablation_config.use_improved_dual_branch
            self.use_score_guided_dual = ablation_config.use_score_guided_dual
            self.use_multiscale_consistency = ablation_config.use_multiscale_consistency
            # V5.7: Rotation-Invariant PE settings
            self.use_multi_orientation = ablation_config.use_multi_orientation
            self.multi_orientation_n = ablation_config.multi_orientation_n
            self.use_content_based_pe = ablation_config.use_content_based_pe
            self.use_hybrid_pe = ablation_config.use_hybrid_pe
            # V5.8: TAPE settings
            self.use_tape = ablation_config.use_tape
            # V6.1: STN settings
            self.use_stn = ablation_config.use_stn
            self.stn_mode = ablation_config.stn_mode
            self.stn_hidden_dim = ablation_config.stn_hidden_dim
            self.stn_dropout = ablation_config.stn_dropout
            self.stn_rotation_reg_weight = ablation_config.stn_rotation_reg_weight
            self.stn_pretrain_epochs = ablation_config.stn_pretrain_epochs
            # V6: Task-separated training settings
            self.use_task_separated = getattr(ablation_config, 'use_task_separated', False)
            self.use_regular_linear = getattr(ablation_config, 'use_regular_linear', False)
            self.use_spectral_norm = getattr(ablation_config, 'use_spectral_norm', False)
            # V7: Denoising Score Matching (DSM) settings
            self.use_dsm = getattr(ablation_config, 'use_dsm', False)
            self.dsm_mode = getattr(ablation_config, 'dsm_mode', 'hybrid')
            self.dsm_alpha = getattr(ablation_config, 'dsm_alpha', 0.5)
            self.dsm_sigma_min = getattr(ablation_config, 'dsm_sigma_min', 0.01)
            self.dsm_sigma_max = getattr(ablation_config, 'dsm_sigma_max', 1.0)
            self.dsm_n_projections = getattr(ablation_config, 'dsm_n_projections', 1)
            self.dsm_use_sliced = getattr(ablation_config, 'dsm_use_sliced', True)
            self.dsm_noise_mode = getattr(ablation_config, 'dsm_noise_mode', 'geometric')
            self.dsm_clean_penalty = getattr(ablation_config, 'dsm_clean_penalty', 0.0)

        # Slow-Fast hyperparameters
        self.slow_lr_ratio = slow_lr_ratio
        self.slow_blocks_k = slow_blocks_k
        self.enable_slow_stage = getattr(args, 'enable_slow_stage', False)

        # Router for inference (only if enabled)
        if self.use_router:
            self.router = PrototypeRouter(device=device, use_mahalanobis=self.use_mahalanobis)
        else:
            self.router = None

        # Class-level adapter management
        self.use_class_level = ablation_config.use_class_level_adapters if ablation_config else False
        if self.use_class_level and self.use_router:
            self.class_router = ClassRouter(device=device, use_mahalanobis=self.use_mahalanobis)
        else:
            self.class_router = None

        # V3: Initialize OGP (Orthogonal Gradient Projection) if enabled
        if self.use_ogp:
            self.ogp = OrthogonalGradientProjection(
                threshold=self.ogp_threshold,
                max_rank_per_task=self.ogp_max_rank,
                device=device
            )
            print(f"✅ [V3] OGP enabled: threshold={self.ogp_threshold}, max_rank={self.ogp_max_rank}")
        else:
            self.ogp = None

        # V7: Initialize DSM (Denoising Score Matching) if enabled
        if self.use_dsm:
            from decoflow.models.dsm import DSMLoss
            self.dsm_loss_fn = DSMLoss(
                sigma_min=self.dsm_sigma_min,
                sigma_max=self.dsm_sigma_max,
                use_sliced=self.dsm_use_sliced,
                n_projections=self.dsm_n_projections,
                noise_mode=self.dsm_noise_mode
            )
            print(f"✅ [V7] DSM enabled: mode={self.dsm_mode}, alpha={self.dsm_alpha}, "
                  f"sigma=[{self.dsm_sigma_min}, {self.dsm_sigma_max}], sliced={self.dsm_use_sliced}")
        else:
            self.dsm_loss_fn = None

        # V6.1: Initialize STN (Spatial Transformer Network) if enabled
        if self.use_stn:
            from decoflow.models.adapters import SpatialTransformerNetwork
            img_size = getattr(args, 'img_size', 224)
            self.stn = SpatialTransformerNetwork(
                input_channels=3,
                input_size=img_size,
                mode=self.stn_mode,
                hidden_dim=self.stn_hidden_dim,
                dropout=self.stn_dropout,
                rotation_reg_weight=self.stn_rotation_reg_weight
            ).to(device)
            print(f"✅ [V6.1] STN enabled: mode={self.stn_mode}, hidden_dim={self.stn_hidden_dim}")
        else:
            self.stn = None

        # Task information
        self.task_classes: Dict[int, List[str]] = {}

        # Store train loaders
        self.task_loaders: Dict[int, DataLoader] = {}

        # Training logger
        self.logger: Optional[TrainingLogger] = logger

        # Log ablation status
        self._log_ablation_status()

    @property
    def feature_extractor(self):
        """Alias for vit_extractor (supports both ViT and CNN backbones)."""
        return self.vit_extractor

    def set_logger(self, logger: TrainingLogger):
        """Set training logger."""
        self.logger = logger

    def _log_ablation_status(self):
        """Log which components are enabled/disabled for ablation."""
        disabled = []
        if not self.use_lora:
            disabled.append("LoRA")
        if not self.use_router:
            disabled.append("Router")
        if not self.use_pos_embedding:
            disabled.append("PosEmbed")
        if not self.use_mahalanobis:
            disabled.append("Mahalanobis")

        if disabled:
            print(f"[Ablation] Disabled components: {', '.join(disabled)}")
        else:
            print("[Ablation] Full model (all components enabled)")

        # Log logdet regularization
        if self.lambda_logdet > 0:
            print(f"[Regularization] Logdet L2: lambda={self.lambda_logdet:.2e}")

        # Log tail-aware loss details (important for train/eval objective alignment)
        if self.use_tail_aware_loss:
            print(
                "[V5] TailAware: "
                f"weight={self.tail_weight:.3f}, "
                f"ratio={self.tail_top_k_ratio:.4f}, "
                f"match_eval_topk={self.tail_match_eval_topk}, "
                f"blend_mode={self.tail_loss_blend_mode}"
            )

        # V3: Log OGP status
        if self.use_ogp:
            print(f"[V3] OGP: threshold={self.ogp_threshold}, max_rank={self.ogp_max_rank}, n_samples={self.ogp_n_samples}")

        # V7: Log DSM status
        if self.use_dsm:
            print(f"[V7] DSM: mode={self.dsm_mode}, alpha={self.dsm_alpha}, sigma=[{self.dsm_sigma_min}, {self.dsm_sigma_max}]")

    def _log(self, message: str, level: str = 'info'):
        """Log message using logger if available, otherwise print."""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
        else:
            print(message)

    def _compute_nll_loss(self, z: torch.Tensor, logdet_patch: torch.Tensor,
                          return_logdet_reg: bool = False):
        """
        Compute NLL loss using patch-wise log-likelihood.

        Standard NLL: L = -log p(x) = -log p(z) - log |det J|
        Now computed properly at patch level before aggregation.

        Optional: Logdet L2 regularization to stabilize image-level scores
        L_reg = lambda * (logdet_patch ** 2).mean()

        Args:
            z: Latent tensor [B, H, W, D]
            logdet_patch: Patch-wise log Jacobian determinant [B, H, W]
            return_logdet_reg: If True, also return logdet regularization term

        Returns:
            nll_loss (scalar), or (nll_loss, logdet_reg) if return_logdet_reg=True
        """
        B, H, W, D = z.shape

        # Patch-wise log p(z): (B, H, W)
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Patch-wise log p(x) = log p(z) + log|det J|: (B, H, W)
        log_px_patch = log_pz_patch + logdet_patch

        # Image-level log-likelihood: sum over all patches
        log_px_image = log_px_patch.sum(dim=(1, 2))  # (B,)

        # NLL = -log p(x), averaged over batch
        nll_loss = -log_px_image.mean()

        if return_logdet_reg:
            # Logdet L2 regularization: encourages log|det J| to be small
            # This reduces variance in log_det across patches, stabilizing image scores
            logdet_reg = (logdet_patch ** 2).mean()
            return nll_loss, logdet_reg

        return nll_loss

    def _compute_tail_aware_loss(self, z: torch.Tensor, logdet_patch: torch.Tensor):
        """
        V5: Compute Tail-Aware Loss to align training with evaluation objective.

        Problem: Standard NLL minimizes mean, but evaluation uses top-k/percentile.
        Solution: Combined loss = (1-w) * mean_loss + w * tail_loss

        This encourages the model to also minimize the high-scoring patches,
        which are what determine the image-level anomaly score during evaluation.

        Args:
            z: Latent tensor [B, H, W, D]
            logdet_patch: Patch-wise log Jacobian determinant [B, H, W]

        Returns:
            combined_loss: Tail-aware NLL loss (scalar)
        """
        B, H, W, D = z.shape

        # Patch-wise log p(z): (B, H, W)
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Patch-wise log p(x) = log p(z) + log|det J|: (B, H, W)
        log_px_patch = log_pz_patch + logdet_patch

        # Patch-wise NLL: (B, H, W)
        nll_patch = -log_px_patch

        # Focal weighting: up-weight hard patches, down-weight easy patches
        focal_gamma = getattr(self, 'focal_gamma', 0.0)
        if focal_gamma > 0:
            with torch.no_grad():
                # Normalize NLL to [0, 1] range per image for stable focal weights
                flat_for_weight = nll_patch.reshape(B, -1)
                nll_min = flat_for_weight.min(dim=1, keepdim=True)[0].unsqueeze(-1)  # (B,1,1)
                nll_max = flat_for_weight.max(dim=1, keepdim=True)[0].unsqueeze(-1)  # (B,1,1)
                nll_norm = (nll_patch - nll_min) / (nll_max - nll_min + 1e-8)  # (B,H,W) in [0,1]
                focal_weight = nll_norm ** focal_gamma  # hard patches get weight ~1, easy ~0
            nll_patch = focal_weight * nll_patch

        # Flatten for aggregation
        flat_nll = nll_patch.reshape(B, -1)  # (B, H*W)
        num_patches = flat_nll.shape[1]

        # Standard mean loss (averaged over all patches and batch)
        mean_loss = flat_nll.mean()

        # Tail loss: average of highest-NLL patches per image
        # Optional alignment mode: use eval top-k count directly during training.
        if self.tail_match_eval_topk and self.score_aggregation_mode == "top_k":
            k = min(max(1, int(self.score_aggregation_top_k)), num_patches)
        else:
            k = max(1, int(num_patches * self.tail_top_k_ratio))

        top_k_nll, _ = torch.topk(flat_nll, k, dim=1)  # (B, k)
        tail_loss = top_k_nll.mean()

        # Blend behavior:
        # - mixed: (1-w)*mean + w*tail (legacy behavior)
        # - tail_only: pure top-k tail loss
        if self.tail_loss_blend_mode == "tail_only":
            return tail_loss

        w = self.tail_weight
        combined_loss = (1 - w) * mean_loss + w * tail_loss
        return combined_loss

    def _compute_dsm_hybrid_loss(self,
                                  patch_embeddings_with_pos: torch.Tensor,
                                  z: torch.Tensor,
                                  logdet_patch: torch.Tensor) -> tuple:
        """
        V7: Compute hybrid NLL + DSM loss.

        L_total = alpha * L_NLL + (1-alpha) * L_DSM + lambda_clean * L_clean

        This combines:
        1. Standard NLL loss (density estimation)
        2. DSM loss (score matching for robustness)
        3. Optional clean data penalty (MULDE-style)

        Args:
            patch_embeddings_with_pos: (B, H, W, D) input features (for DSM)
            z: (B, H, W, D) latent from forward pass
            logdet_patch: (B, H, W) log det from forward pass

        Returns:
            total_loss: Combined loss
            info: Dict with component losses
        """
        B, H, W, D = z.shape
        info = {}

        # 1. NLL loss (standard)
        nll_loss = self._compute_nll_loss(z, logdet_patch)
        info['nll_loss'] = nll_loss.item()

        # 2. DSM loss
        if self.dsm_mode in ['dsm_only', 'hybrid'] and self.dsm_loss_fn is not None:
            dsm_loss, dsm_info = self.dsm_loss_fn(
                self.nf_model,
                patch_embeddings_with_pos
            )
            info.update(dsm_info)
        else:
            dsm_loss = torch.tensor(0.0, device=z.device)
            info['dsm_loss'] = 0.0

        # 3. Clean data penalty (MULDE-style) - optional
        if self.dsm_clean_penalty > 0:
            # Encourage high likelihood for clean samples
            log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            log_px = log_pz + logdet_patch
            clean_penalty = -log_px.mean()  # Minimize negative log-likelihood
            info['clean_penalty'] = clean_penalty.item()
        else:
            clean_penalty = torch.tensor(0.0, device=z.device)

        # 4. Multi-task loss (pixel-level + image-level)
        if self.use_multitask_loss:
            # Pixel-level loss (current NLL on patches)
            pixel_loss = nll_loss

            # Image-level loss (average NLL per image)
            # z: (B, H, W, D), logdet_patch: (B, H, W)
            log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            log_px_patch = log_pz_patch + logdet_patch
            log_px_image = log_px_patch.view(B, -1).mean(dim=1)  # (B,)
            image_loss = -log_px_image.mean()

            # Combine pixel and image losses
            nll_loss = self.multitask_alpha * pixel_loss + self.multitask_beta * image_loss
            info['pixel_loss'] = pixel_loss.item()
            info['image_loss'] = image_loss.item()
            info['multitask_alpha'] = self.multitask_alpha
            info['multitask_beta'] = self.multitask_beta

        # 5. Combine losses based on mode
        if self.dsm_mode == 'nll_only':
            total_loss = nll_loss
        elif self.dsm_mode == 'dsm_only':
            total_loss = dsm_loss
        else:  # hybrid
            alpha = self.dsm_alpha
            total_loss = alpha * nll_loss + (1 - alpha) * dsm_loss
            info['dsm_alpha'] = alpha

        # Add clean penalty
        if self.dsm_clean_penalty > 0:
            total_loss = total_loss + self.dsm_clean_penalty * clean_penalty

        info['total_loss'] = total_loss.item()

        return total_loss, info

    def train_task(self,
                   task_id: int,
                   task_classes: List[str],
                   train_loader: DataLoader,
                   num_epochs: int = 10,
                   lr: float = 1e-4,
                   log_interval: int = 10,
                   global_class_to_idx: dict = None):
        """
        Stage-Separated Slow-Fast Training.

        Task 0: Base NF training + feature statistics collection
        Task > 0:
            Stage 1 (FAST): LoRA + InputAdapter adaptation (base frozen)
            Stage 2 (SLOW): Base NF consolidation with EWC protection (optional)

        Class-Level Mode (use_class_level_adapters=True):
            Each class within the task is trained separately with its own
            LoRA, ACL, InputAdapter, and Prototype.

        Args:
            task_id: Task identifier
            task_classes: List of class names in this task
            train_loader: DataLoader for training data
            num_epochs: Number of training epochs
            lr: Learning rate
            log_interval: Logging interval
            global_class_to_idx: Mapping from class name to global index (required for class-level mode)
        """
        # Log task start
        if self.logger:
            self.logger.log_task_start(task_id, task_classes, num_epochs, lr)
        else:
            print(f"\n{'='*70}")
            print(f"🚀 Training Task {task_id}: {task_classes}")
            if self.use_class_level:
                print(f"   📋 Class-Level Mode: Each class trained separately")
            print(f"{'='*70}")

        self.task_classes[task_id] = task_classes
        self.task_loaders[task_id] = train_loader

        # =================================================================
        # Class-Level Training Mode
        # =================================================================
        if self.use_class_level and global_class_to_idx is not None:
            self._train_task_class_level(
                task_id=task_id,
                task_classes=task_classes,
                train_loader=train_loader,
                num_epochs=num_epochs,
                lr=lr,
                log_interval=log_interval,
                global_class_to_idx=global_class_to_idx
            )
            return

        # =================================================================
        # Standard Task-Level Training Mode
        # =================================================================
        # Add task to NF model (pass class_name for per-class ACL support)
        cls_name = task_classes[0] if len(task_classes) == 1 else None
        self.nf_model.add_task(task_id, class_name=cls_name)

        if task_id == 0:
            # Task 0: Base NF Training
            self._train_base_task(task_id, task_classes, train_loader,
                                  num_epochs, lr, log_interval)

        else:
            # V6: Task-Separated Mode - Train each task independently (like Task 0)
            if self.use_task_separated:
                if self.logger:
                    self.logger.log_stage_start(
                        "Task-Separated Training (Full NF)",
                        num_epochs,
                        {"Base NF": "TRAINING", "Task Layers": "TRAINING"}
                    )
                else:
                    print(f"\n{'─'*70}")
                    print(f"🔀 [V6] Task-Separated: Full NF Training ({num_epochs} epochs)")
                    print(f"{'─'*70}")

                # Use base task training style for task-separated mode
                self._train_base_task(task_id, task_classes, train_loader,
                                      num_epochs, lr, log_interval)
            else:
                # Standard: Two-Stage Slow-Fast Training
                if self.enable_slow_stage:
                    fast_epochs = int(num_epochs * 0.85)
                    slow_epochs = num_epochs - fast_epochs
                else:
                    fast_epochs = num_epochs
                    slow_epochs = 0

                # Stage 1: FAST Adaptation
                if self.logger:
                    self.logger.log_stage_start(
                        "Stage 1: FAST Adaptation",
                        fast_epochs,
                        {"Base NF": "FROZEN", "LoRA + InputAdapter": "TRAINING"}
                    )
                else:
                    print(f"\n{'─'*70}")
                    print(f"📌 Stage 1: FAST Adaptation ({fast_epochs} epochs)")
                    print(f"{'─'*70}")

                self._train_fast_stage(task_id, task_classes, train_loader,
                                       fast_epochs, lr, log_interval)

                # Stage 2: SLOW Consolidation (optional, only for non-task-separated mode)
                if self.enable_slow_stage and slow_epochs > 0:
                    if self.logger:
                        self.logger.log_stage_start(
                            "Stage 2: SLOW Consolidation",
                            slow_epochs,
                            {
                                f"Last {self.slow_blocks_k} blocks": f"TRAINING (LR={lr * self.slow_lr_ratio:.2e})",
                                "LoRA + InputAdapter": "FROZEN"
                            }
                        )

                    self._train_slow_stage(task_id, task_classes, train_loader,
                                           slow_epochs, lr * self.slow_lr_ratio, log_interval)

                    self.nf_model.unfreeze_fast_params(task_id)

        # Build prototype for routing (only if router is enabled)
        if self.use_router:
            self._build_prototype(task_id, task_classes, train_loader)

        # V3: Compute and store OGP basis after task training
        # This captures the important gradient subspace for the completed task
        if self.use_ogp and self.ogp is not None:
            self._compute_ogp_basis(task_id, train_loader)

        # V5.8: Log TAPE PE strength after training
        if self.use_tape and hasattr(self.nf_model, 'tape') and self.nf_model.tape is not None:
            pe_strength = self.nf_model.tape.get_pe_strength(task_id)
            tape_msg = f"   📐 [V5.8] TAPE PE strength for Task {task_id}: {pe_strength:.4f}"
            if self.logger:
                self.logger.info(tape_msg)
            else:
                print(tape_msg)

        # Log task completion
        if self.logger:
            self.logger.log_task_complete(task_id)
        else:
            print(f"\n✅ Task {task_id} training completed!")

    # =========================================================================
    # Class-Level Training Methods
    # =========================================================================

    def _train_task_class_level(self,
                                task_id: int,
                                task_classes: List[str],
                                train_loader: DataLoader,
                                num_epochs: int,
                                lr: float,
                                log_interval: int,
                                global_class_to_idx: dict):
        """
        Train each class in the task separately (class-level mode).

        This method:
        1. Filters the train_loader to get class-specific samples
        2. Adds class adapter to NF model
        3. Trains the class-specific adapter
        4. Builds class prototype for routing

        Args:
            task_id: Task identifier
            task_classes: List of class names in this task
            train_loader: DataLoader for all classes in this task
            num_epochs: Number of training epochs per class
            lr: Learning rate
            log_interval: Logging interval
            global_class_to_idx: Mapping from class name to global index
        """
        print(f"\n📋 Class-Level Training: {len(task_classes)} classes in Task {task_id}")

        for cls_idx, cls_name in enumerate(task_classes):
            class_id = global_class_to_idx[cls_name]

            print(f"\n{'─'*60}")
            print(f"📌 Training Class {class_id}: {cls_name} ({cls_idx+1}/{len(task_classes)})")
            print(f"{'─'*60}")

            # Add class adapter to NF model
            self.nf_model.add_class(class_id, cls_name, task_id)

            # Get class-specific data loader
            class_loader = self._filter_loader_by_class(train_loader, cls_name)

            if len(class_loader.dataset) == 0:
                print(f"   ⚠️  No samples for class {cls_name}, skipping...")
                continue

            print(f"   📊 Samples: {len(class_loader.dataset)}")

            # Train this class
            if task_id == 0:
                self._train_class_base(class_id, cls_name, class_loader,
                                       num_epochs, lr, log_interval)
            else:
                self._train_class_fast(class_id, cls_name, class_loader,
                                       num_epochs, lr, log_interval)

            # Build class prototype for routing
            if self.use_router and self.class_router is not None:
                self._build_class_prototype(class_id, cls_name, task_id, class_loader)

            # Collect reference statistics from Task 0 classes
            if task_id == 0:
                self._collect_class_reference_stats(class_id, class_loader)

        # Finalize reference statistics after all Task 0 classes
        if task_id == 0:
            self.nf_model.finalize_reference_stats()

        print(f"\n✅ Task {task_id} class-level training completed!")
        print(f"   📊 Trained {len(task_classes)} class adapters")

    def _filter_loader_by_class(self, train_loader: DataLoader, target_class: str) -> DataLoader:
        """
        Filter a DataLoader to only include samples from a specific class.

        Args:
            train_loader: Original DataLoader with all classes
            target_class: Class name to filter for

        Returns:
            DataLoader containing only samples from target_class
        """
        from torch.utils.data import Subset, DataLoader as TorchDataLoader

        # Get indices for the target class
        indices = []
        for idx in range(len(train_loader.dataset)):
            # Dataset returns (image, label, class_name, ...) or similar
            sample = train_loader.dataset[idx]

            target_class_id = None
            class_to_idx = getattr(train_loader.dataset, 'class_to_idx', {})
            if target_class in class_to_idx:
                target_class_id = int(class_to_idx[target_class])

            # TaskDataset stores samples as (image, class_id, mask, name, path).
            # Some older datasets expose class_name directly.
            if len(sample) >= 2 and target_class_id is not None:
                sample_class = int(sample[1])
                is_match = sample_class == target_class_id
            elif hasattr(train_loader.dataset, 'class_name'):
                # Single-class dataset
                sample_class = train_loader.dataset.class_name
                is_match = sample_class == target_class
            else:
                # Fallback: can't determine class, include all
                sample_class = target_class
                is_match = True

            if is_match:
                indices.append(idx)

        # Create subset dataset
        subset = Subset(train_loader.dataset, indices)

        # Create new DataLoader with same settings
        return TorchDataLoader(
            subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=getattr(train_loader, 'num_workers', 0),
            pin_memory=getattr(train_loader, 'pin_memory', False),
            drop_last=len(indices) >= train_loader.batch_size
        )

    def _train_class_base(self, class_id: int, class_name: str,
                          train_loader: DataLoader, num_epochs: int,
                          lr: float, log_interval: int):
        """
        Train a class adapter with base NF (Task 0 class).

        Similar to _train_base_task but for a single class.
        """
        self.nf_model.train()
        self.nf_model.set_active_class(class_id)

        # Get trainable parameters for this class
        params = self.nf_model.get_trainable_params_for_class(class_id)
        optimizer = create_optimizer(params, lr)

        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch[0].to(self.device)

                # Forward pass
                loss, nll_loss, _ = self._compute_training_loss_for_images(images)
                if torch.isnan(loss) or loss.item() > 1e8:
                    continue

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
                optimizer.step()

                epoch_loss += nll_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"   Epoch [{epoch+1:3d}/{num_epochs}] Loss: {avg_loss:.4f}")

    def _train_class_fast(self, class_id: int, class_name: str,
                          train_loader: DataLoader, num_epochs: int,
                          lr: float, log_interval: int):
        """
        Train a class adapter (Task > 0 class) - FAST stage only.

        Similar to _train_fast_stage but for a single class.
        """
        self.nf_model.train()
        self.nf_model.set_active_class(class_id)

        # Get trainable parameters for this class (LoRA + adapters only)
        params = self.nf_model.get_trainable_params_for_class(class_id)
        optimizer = create_optimizer(params, lr)

        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch[0].to(self.device)

                # Forward pass
                loss, nll_loss, _ = self._compute_training_loss_for_images(images)
                if torch.isnan(loss) or loss.item() > 1e8:
                    continue

                # Apply OGP if enabled (project gradients)
                if self.use_ogp and self.ogp is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    if hasattr(self.ogp, 'project_gradient'):
                        self.ogp.project_gradient(self.nf_model)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
                    optimizer.step()

                epoch_loss += nll_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"   Epoch [{epoch+1:3d}/{num_epochs}] Loss: {avg_loss:.4f}")

    def _build_class_prototype(self, class_id: int, class_name: str,
                               task_id: int, train_loader: DataLoader):
        """
        Build prototype for a single class.

        Args:
            class_id: Global class index
            class_name: Class name
            task_id: Task this class belongs to
            train_loader: DataLoader for this class only
        """
        self.nf_model.eval()
        self.nf_model.set_active_class(class_id)

        all_features = []

        with torch.no_grad():
            for batch in train_loader:
                images = batch[0].to(self.device)

                # Extract image-level features for routing
                image_features = self._extract_image_features(images)
                all_features.append(image_features)

        all_features = torch.cat(all_features, dim=0)

        # Create class prototype
        prototype = ClassPrototype(class_id, class_name, task_id, self.device)
        prototype.update(all_features)
        prototype.finalize()

        # Add to class router
        self.class_router.add_prototype(class_id, prototype)

        print(f"   🎯 Class prototype built: {all_features.shape[0]} samples, "
              f"mean_norm={prototype.mean.norm():.4f}")

    def _collect_class_reference_stats(self, class_id: int, train_loader: DataLoader):
        """
        Collect reference statistics from a Task 0 class.

        These statistics are used to initialize adapters for later tasks.
        """
        self.nf_model.eval()
        self.nf_model.set_active_class(class_id)

        with torch.no_grad():
            for batch in train_loader:
                images = batch[0].to(self.device)

                # Extract features and update reference statistics
                features = self._extract_patch_features(images)
                self.nf_model.update_reference_stats(features)

    def _extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image-level features for routing."""
        # Use ViT CLS token or global pooled features
        with torch.no_grad():
            if hasattr(self.vit_extractor, 'get_cls_token'):
                return self.vit_extractor.get_cls_token(images)
            elif hasattr(self.vit_extractor, 'get_image_level_features'):
                return self.vit_extractor.get_image_level_features(images)
            else:
                # Fallback: extract patch features and pool
                features = self.vit_extractor(images)
                return features.mean(dim=(1, 2))  # (B, D)

    def _extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features for NF training."""
        with torch.no_grad():
            patch_embeddings, spatial_shape = self.vit_extractor(
                images, return_spatial_shape=True
            )
            use_custom_pe = self.use_content_based_pe or self.use_hybrid_pe or self.use_tape
            if use_custom_pe:
                B = patch_embeddings.shape[0]
                H, W = spatial_shape
                return patch_embeddings.reshape(B, H, W, -1)
            if self.use_pos_embedding and self.pos_embed_generator is not None:
                return self.pos_embed_generator(spatial_shape, patch_embeddings)
            B = patch_embeddings.shape[0]
            H, W = spatial_shape
            return patch_embeddings.reshape(B, H, W, -1)

    def _compute_training_loss_for_images(self, images: torch.Tensor):
        """Compute the standard NF training loss for an image batch."""
        if self.use_stn and self.stn is not None:
            images, _ = self.stn(images)

        with torch.no_grad():
            patch_embeddings, spatial_shape = self.vit_extractor(
                images, return_spatial_shape=True
            )
            use_custom_pe = self.use_content_based_pe or self.use_hybrid_pe or self.use_tape
            if use_custom_pe:
                B = patch_embeddings.shape[0]
                H, W = spatial_shape
                patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)
            elif self.use_pos_embedding:
                patch_embeddings_with_pos = self.pos_embed_generator(
                    spatial_shape, patch_embeddings
                )
            else:
                B = patch_embeddings.shape[0]
                H, W = spatial_shape
                patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)

        z, logdet_patch = self.nf_model.forward(patch_embeddings_with_pos, reverse=False)

        if self.use_dsm and self.dsm_loss_fn is not None:
            total_loss, dsm_info = self._compute_dsm_hybrid_loss(
                patch_embeddings_with_pos, z, logdet_patch
            )
            nll_loss = torch.tensor(dsm_info['nll_loss'], device=total_loss.device)
            logdet_reg = None
            if self.lambda_logdet > 0:
                logdet_reg = (logdet_patch ** 2).mean()
                total_loss = total_loss + self.lambda_logdet * logdet_reg
        elif self.use_tail_aware_loss:
            nll_loss = self._compute_tail_aware_loss(z, logdet_patch)
            total_loss = nll_loss
            logdet_reg = None
            if self.lambda_logdet > 0:
                logdet_reg = (logdet_patch ** 2).mean()
                total_loss = total_loss + self.lambda_logdet * logdet_reg
        elif self.lambda_logdet > 0:
            nll_loss, logdet_reg = self._compute_nll_loss(
                z, logdet_patch, return_logdet_reg=True
            )
            total_loss = nll_loss + self.lambda_logdet * logdet_reg
        else:
            nll_loss = self._compute_nll_loss(z, logdet_patch)
            total_loss = nll_loss
            logdet_reg = None

        if self.spatial_var_lambda > 0 and self.nf_model._acl_spatial_var_loss is not None:
            total_loss = total_loss + self.spatial_var_lambda * self.nf_model._acl_spatial_var_loss
        if self.acl_gate_l1_lambda > 0:
            gate_l1 = self.nf_model.get_acl_gate_l1_loss()
            if gate_l1 is not None:
                total_loss = total_loss + self.acl_gate_l1_lambda * gate_l1
        if self.use_stn and self.stn is not None:
            stn_reg = self.stn.get_rotation_regularization_loss()
            if stn_reg.device != total_loss.device:
                stn_reg = stn_reg.to(total_loss.device)
            total_loss = total_loss + stn_reg

        return total_loss, nll_loss, logdet_reg

    def _train_base_task(self, task_id: int, task_classes: List[str],
                         train_loader: DataLoader, num_epochs: int,
                         lr: float, log_interval: int):
        """Train base NF + LoRA for Task 0, or independent NF for Complete Separated mode."""
        trainable_params = self.nf_model.get_trainable_params(task_id)
        num_params = sum(p.numel() for p in trainable_params)

        # Check if LoRA is included (handles both DCLSubnet and DCLContextSubnet)
        def _get_first_layer(subnet):
            return subnet.s_layer1 if hasattr(subnet, 's_layer1') else subnet.layer1

        def _has_lora_for_task(subnet, tid):
            layer = _get_first_layer(subnet)
            if hasattr(layer, 'lora_A'):
                return str(tid) in layer.lora_A
            return False  # regular_linear mode has no lora_A

        # For Complete Separated mode (task > 0), check task-specific subnets
        task_key = str(task_id)
        if self.use_task_separated and task_id > 0 and task_key in self.nf_model.task_subnets:
            subnets_to_check = self.nf_model.task_subnets[task_key]
            has_lora = self.use_lora and any(
                _has_lora_for_task(subnet, task_id) for subnet in subnets_to_check
            )
            phase_name = "Complete Separated NF Training" if not has_lora else "Complete Separated NF + LoRA Training"
        else:
            has_lora = self.use_lora and any(
                _has_lora_for_task(subnet, task_id) for subnet in self.nf_model.subnets
            )
            use_regular_linear = getattr(self.ablation_config, 'use_regular_linear', False)
            if use_regular_linear:
                phase_name = "Base + Full-Rank Task Weights"
            elif has_lora:
                phase_name = "Base + LoRA Training"
            else:
                phase_name = "Base Training"

        if self.logger:
            self.logger.log_model_info(
                f"{phase_name} + Feature Statistics Collection",
                num_params,
                {"Classes": task_classes, "Epochs": num_epochs, "Learning Rate": f"{lr:.2e}"}
            )
        else:
            print(f"   - Phase: {phase_name} + Feature Statistics Collection")
            print(f"   - Trainable Parameters: {num_params:,}")

        # V6.1: Include STN parameters in optimization if enabled
        all_params = list(trainable_params)
        if self.use_stn and self.stn is not None:
            all_params.extend(self.stn.parameters())

        # Create optimizer (AdamP if available, AdamW fallback)
        # V45: Separate ACL params for custom weight decay if needed
        task_key = str(task_id)
        if self.acl_weight_decay > 0 and task_key in self.nf_model.acl_adapters:
            acl_params = list(self.nf_model.acl_adapters[task_key].parameters())
            acl_ids = {id(p) for p in acl_params}
            other_params = [p for p in all_params if id(p) not in acl_ids]
            optimizer = create_optimizer(
                [{'params': other_params}, {'params': acl_params, 'weight_decay': self.acl_weight_decay}],
                lr=lr
            )
        else:
            optimizer = create_optimizer(all_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        self.nf_model.train()
        self.vit_extractor.eval()
        if self.use_stn and self.stn is not None:
            self.stn.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, labels, _, _, _) in enumerate(train_loader):
                images = images.to(self.device)

                # V6.1: Apply STN for image alignment before feature extraction
                if self.use_stn and self.stn is not None:
                    images, stn_theta = self.stn(images)

                with torch.no_grad():
                    patch_embeddings, spatial_shape = self.vit_extractor(
                        images, return_spatial_shape=True
                    )
                    # V5.7/V5.8: Skip standard PE when custom PE is enabled
                    # NF will apply PE internally (ContentBasedPE, HybridPE, or TAPE)
                    use_custom_pe = self.use_content_based_pe or self.use_hybrid_pe or self.use_tape
                    if use_custom_pe:
                        # Pass raw features (no PE) - NF will apply custom PE
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)
                    elif self.use_pos_embedding:
                        # Standard: Apply grid positional embedding
                        patch_embeddings_with_pos = self.pos_embed_generator(
                            spatial_shape, patch_embeddings
                        )
                    else:
                        # Reshape to (B, H, W, D) without positional embedding
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)
                    # Collect feature statistics
                    self.nf_model.update_reference_stats(patch_embeddings_with_pos)

                # Forward through NF to get latent z and patch-wise logdet
                z, logdet_patch = self.nf_model.forward(patch_embeddings_with_pos, reverse=False)

                # V7: Compute loss with DSM if enabled
                dsm_info = None
                if self.use_dsm and self.dsm_loss_fn is not None:
                    total_loss, dsm_info = self._compute_dsm_hybrid_loss(
                        patch_embeddings_with_pos, z, logdet_patch
                    )
                    nll_loss = torch.tensor(dsm_info['nll_loss'])
                    logdet_reg = None
                    # Add logdet regularization if enabled
                    if self.lambda_logdet > 0:
                        logdet_reg = (logdet_patch ** 2).mean()
                        total_loss = total_loss + self.lambda_logdet * logdet_reg
                # V5: Compute loss (tail-aware or standard NLL)
                elif self.use_tail_aware_loss:
                    nll_loss = self._compute_tail_aware_loss(z, logdet_patch)
                    total_loss = nll_loss
                    logdet_reg = None
                    # Add logdet regularization if enabled
                    if self.lambda_logdet > 0:
                        logdet_reg = (logdet_patch ** 2).mean()
                        total_loss = total_loss + self.lambda_logdet * logdet_reg
                elif self.lambda_logdet > 0:
                    nll_loss, logdet_reg = self._compute_nll_loss(z, logdet_patch, return_logdet_reg=True)
                    total_loss = nll_loss + self.lambda_logdet * logdet_reg
                else:
                    nll_loss = self._compute_nll_loss(z, logdet_patch)
                    total_loss = nll_loss
                    logdet_reg = None

                # V45: Spatial variance preservation loss
                if self.spatial_var_lambda > 0 and self.nf_model._acl_spatial_var_loss is not None:
                    total_loss = total_loss + self.spatial_var_lambda * self.nf_model._acl_spatial_var_loss
                # V45: ACL gate L1 regularization
                if self.acl_gate_l1_lambda > 0:
                    gate_l1 = self.nf_model.get_acl_gate_l1_loss()
                    if gate_l1 is not None:
                        total_loss = total_loss + self.acl_gate_l1_lambda * gate_l1

                # V6.1: Add STN rotation regularization to prevent extreme rotations
                stn_reg = None
                if self.use_stn and self.stn is not None:
                    stn_reg = self.stn.get_rotation_regularization_loss()
                    if stn_reg.device != total_loss.device:
                        stn_reg = stn_reg.to(total_loss.device)
                    total_loss = total_loss + stn_reg

                if torch.isnan(total_loss) or total_loss.item() > 1e8:
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
                optimizer.step()

                epoch_loss += nll_loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    extra_info = {}
                    if self.use_tail_aware_loss:
                        extra_info["tail_aware"] = True
                    if logdet_reg is not None:
                        extra_info["logdet_reg"] = logdet_reg.item()
                    if self.logger:
                        self.logger.log_batch(
                            task_id, epoch, num_epochs, batch_idx+1, len(train_loader),
                            nll_loss.item(), avg_loss, stage="BASE", extra_info=extra_info if extra_info else None
                        )
                    else:
                        reg_str = f" | LogdetReg: {logdet_reg.item():.4f}" if logdet_reg is not None else ""
                        print(f"  [BASE] Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                              f"Loss: {nll_loss.item():.4f} | Avg: {avg_loss:.4f}{reg_str}")

            scheduler.step()
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr

            # Get context gate/alpha info for logging
            extra_info = {"LR": current_lr}
            context_info = self.nf_model.get_context_info()
            if context_info:
                extra_info.update(context_info)

            if self.logger:
                self.logger.log_epoch(
                    task_id, epoch, num_epochs, avg_epoch_loss,
                    stage="BASE", extra_info=extra_info
                )
            else:
                ctx_str = ""
                if 'gate_mean' in context_info:
                    ctx_str = f" | Gate: {context_info['gate_mean']:.4f}±{context_info['gate_std']:.4f}"
                elif 'alpha_mean' in context_info:
                    ctx_str = f" | Alpha: {context_info['alpha_mean']:.4f}"
                print(f"  📊 [BASE] Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}{ctx_str}")

        # Finalize feature statistics
        self.nf_model.finalize_reference_stats()

        # Log feature statistics
        if self.logger and hasattr(self.nf_model.reference_stats, 'mean'):
            stats = self.nf_model.reference_stats
            if stats.is_initialized:
                self.logger.log_feature_stats(
                    stats.n_samples,
                    stats.mean.norm().item(),
                    stats.std.mean().item(),
                    stats.std.min().item(),
                    stats.std.max().item()
                )

    def _train_fast_stage(self, task_id: int, task_classes: List[str],
                          train_loader: DataLoader, num_epochs: int,
                          lr: float, log_interval: int):
        """Stage 1: FAST Adaptation (LoRA + InputAdapter, base frozen)."""
        # Freeze base NF unless no_freeze_base is enabled (for sequential training ablation)
        if not self.ablation_config.no_freeze_base:
            self.nf_model.freeze_all_base()

        fast_params = self.nf_model.get_fast_params(task_id)
        num_params = sum(p.numel() for p in fast_params)

        if self.logger:
            self.logger.log_model_info(
                "FAST Adaptation",
                num_params,
                {"Learning Rate": f"{lr:.2e}"}
            )
        else:
            print(f"   - Trainable Parameters: {num_params:,}")

        # V6.1: Include STN parameters in optimization if enabled
        all_fast_params = list(fast_params)
        if self.use_stn and self.stn is not None:
            all_fast_params.extend(self.stn.parameters())

        # Sequential Training ablation: include base NF parameters
        if self.ablation_config.no_freeze_base:
            base_params = [p for p in self.nf_model.parameters() if p.requires_grad]
            all_fast_params.extend(base_params)

        # Create optimizer (AdamP if available, AdamW fallback)
        # V45: Separate ACL params for custom weight decay if needed
        task_key = str(task_id)
        if self.acl_weight_decay > 0 and task_key in self.nf_model.acl_adapters:
            acl_params = list(self.nf_model.acl_adapters[task_key].parameters())
            acl_ids = {id(p) for p in acl_params}
            other_fast = [p for p in all_fast_params if id(p) not in acl_ids]
            optimizer = create_optimizer(
                [{'params': other_fast}, {'params': acl_params, 'weight_decay': self.acl_weight_decay}],
                lr=lr
            )
        else:
            optimizer = create_optimizer(all_fast_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        self.nf_model.train()
        self.vit_extractor.eval()
        if self.use_stn and self.stn is not None:
            self.stn.train()

        warmup_epochs = 2

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Warmup
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * warmup_factor

            for batch_idx, (images, labels, _, _, _) in enumerate(train_loader):
                images = images.to(self.device)

                # V6.1: Apply STN for image alignment before feature extraction
                if self.use_stn and self.stn is not None:
                    images, stn_theta = self.stn(images)

                with torch.no_grad():
                    patch_embeddings, spatial_shape = self.vit_extractor(
                        images, return_spatial_shape=True
                    )
                    # V5.7/V5.8: Skip standard PE when custom PE is enabled
                    use_custom_pe = self.use_content_based_pe or self.use_hybrid_pe or self.use_tape
                    if use_custom_pe:
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)
                    elif self.use_pos_embedding:
                        patch_embeddings_with_pos = self.pos_embed_generator(
                            spatial_shape, patch_embeddings
                        )
                    else:
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)

                # Forward through NF to get latent z and patch-wise logdet
                z, logdet_patch = self.nf_model.forward(patch_embeddings_with_pos, reverse=False)

                # V7: Compute loss with DSM if enabled
                dsm_info = None
                if self.use_dsm and self.dsm_loss_fn is not None:
                    total_loss, dsm_info = self._compute_dsm_hybrid_loss(
                        patch_embeddings_with_pos, z, logdet_patch
                    )
                    nll_loss = torch.tensor(dsm_info['nll_loss'])
                    logdet_reg = None
                    # Add logdet regularization if enabled
                    if self.lambda_logdet > 0:
                        logdet_reg = (logdet_patch ** 2).mean()
                        total_loss = total_loss + self.lambda_logdet * logdet_reg
                # V5: Compute loss (tail-aware or standard NLL)
                elif self.use_tail_aware_loss:
                    nll_loss = self._compute_tail_aware_loss(z, logdet_patch)
                    total_loss = nll_loss
                    logdet_reg = None
                    # Add logdet regularization if enabled
                    if self.lambda_logdet > 0:
                        logdet_reg = (logdet_patch ** 2).mean()
                        total_loss = total_loss + self.lambda_logdet * logdet_reg
                elif self.lambda_logdet > 0:
                    nll_loss, logdet_reg = self._compute_nll_loss(z, logdet_patch, return_logdet_reg=True)
                    total_loss = nll_loss + self.lambda_logdet * logdet_reg
                else:
                    nll_loss = self._compute_nll_loss(z, logdet_patch)
                    total_loss = nll_loss
                    logdet_reg = None

                # V45: Spatial variance preservation loss
                if self.spatial_var_lambda > 0 and self.nf_model._acl_spatial_var_loss is not None:
                    total_loss = total_loss + self.spatial_var_lambda * self.nf_model._acl_spatial_var_loss
                # V45: ACL gate L1 regularization
                if self.acl_gate_l1_lambda > 0:
                    gate_l1 = self.nf_model.get_acl_gate_l1_loss()
                    if gate_l1 is not None:
                        total_loss = total_loss + self.acl_gate_l1_lambda * gate_l1

                # V6.1: Add STN rotation regularization to prevent extreme rotations
                stn_reg = None
                if self.use_stn and self.stn is not None:
                    stn_reg = self.stn.get_rotation_regularization_loss()
                    if stn_reg.device != total_loss.device:
                        stn_reg = stn_reg.to(total_loss.device)
                    total_loss = total_loss + stn_reg

                if torch.isnan(total_loss) or total_loss.item() > 1e8:
                    continue

                optimizer.zero_grad()
                total_loss.backward()

                # V3: Apply OGP gradient projection for Task > 0
                # This projects gradients to be orthogonal to important subspaces from previous tasks
                if self.use_ogp and self.ogp is not None and self.ogp.is_initialized:
                    self.ogp.project_gradient(self.nf_model)

                torch.nn.utils.clip_grad_norm_(all_fast_params, max_norm=0.5)
                optimizer.step()

                epoch_loss += nll_loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = optimizer.param_groups[0]['lr']
                    extra_info = {"LR": current_lr}
                    if self.use_tail_aware_loss:
                        extra_info["tail_aware"] = True
                    if epoch < warmup_epochs:
                        extra_info["Warmup"] = True
                    if logdet_reg is not None:
                        extra_info["logdet_reg"] = logdet_reg.item()
                    if self.use_ogp and self.ogp is not None and self.ogp.is_initialized:
                        extra_info["OGP"] = True

                    if self.logger:
                        self.logger.log_batch(
                            task_id, epoch, num_epochs, batch_idx+1, len(train_loader),
                            nll_loss.item(), avg_loss, stage="FAST", extra_info=extra_info
                        )
                    else:
                        warmup_str = " (warmup)" if epoch < warmup_epochs else ""
                        reg_str = f" | LogdetReg: {logdet_reg.item():.4f}" if logdet_reg is not None else ""
                        ogp_str = " | OGP" if (self.use_ogp and self.ogp is not None and self.ogp.is_initialized) else ""
                        print(f"  [FAST] Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                              f"Loss: {nll_loss.item():.4f} | Avg: {avg_loss:.4f}{warmup_str}{reg_str}{ogp_str}")

            if epoch >= warmup_epochs:
                scheduler.step()

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']

            # Get context gate/alpha info for logging
            extra_info = {"LR": current_lr}
            context_info = self.nf_model.get_context_info()
            if context_info:
                extra_info.update(context_info)

            if self.logger:
                self.logger.log_epoch(
                    task_id, epoch, num_epochs, avg_epoch_loss,
                    stage="FAST", extra_info=extra_info
                )
            else:
                ctx_str = ""
                if 'gate_mean' in context_info:
                    ctx_str = f" | Gate: {context_info['gate_mean']:.4f}±{context_info['gate_std']:.4f}"
                elif 'alpha_mean' in context_info:
                    ctx_str = f" | Alpha: {context_info['alpha_mean']:.4f}"
                print(f"  📊 [FAST] Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}{ctx_str}")

    def _train_slow_stage(self, task_id: int, task_classes: List[str],
                          train_loader: DataLoader, num_epochs: int,
                          lr: float, log_interval: int):
        """Stage 2: SLOW Consolidation (last K blocks of base NF)."""
        self.nf_model.freeze_fast_params(task_id)
        self.nf_model.unfreeze_last_k_blocks(self.slow_blocks_k)

        slow_params = self.nf_model.get_base_params_for_slow_update(self.slow_blocks_k)
        num_params = sum(p.numel() for p in slow_params)

        if self.logger:
            self.logger.log_model_info(
                "SLOW Consolidation",
                num_params,
                {"Learning Rate": f"{lr:.2e}"}
            )
        else:
            print(f"   - Trainable Parameters: {num_params:,}")

        # Create optimizer (lower weight decay for slow consolidation)
        optimizer = create_optimizer(slow_params, lr=lr, weight_decay_scale=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        self.nf_model.train()
        self.vit_extractor.eval()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, labels, _, _, _) in enumerate(train_loader):
                images = images.to(self.device)

                with torch.no_grad():
                    patch_embeddings, spatial_shape = self.vit_extractor(
                        images, return_spatial_shape=True
                    )
                    # V5.7/V5.8: Skip standard PE when custom PE is enabled
                    use_custom_pe = self.use_content_based_pe or self.use_hybrid_pe or self.use_tape
                    if use_custom_pe:
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)
                    elif self.use_pos_embedding:
                        patch_embeddings_with_pos = self.pos_embed_generator(
                            spatial_shape, patch_embeddings
                        )
                    else:
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)

                # NLL loss
                log_prob = self.nf_model.log_prob(patch_embeddings_with_pos)
                nll_loss = -log_prob.mean()

                if torch.isnan(nll_loss) or nll_loss.item() > 1e8:
                    continue

                optimizer.zero_grad()
                nll_loss.backward()
                torch.nn.utils.clip_grad_norm_(slow_params, max_norm=0.5)
                optimizer.step()

                epoch_loss += nll_loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = optimizer.param_groups[0]['lr']

                    if self.logger:
                        self.logger.log_batch(
                            task_id, epoch, num_epochs, batch_idx+1, len(train_loader),
                            nll_loss.item(), avg_loss, stage="SLOW",
                            extra_info={"LR": current_lr}
                        )
                    else:
                        print(f"  [SLOW] Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                              f"Loss: {nll_loss.item():.4f} | Avg: {avg_loss:.4f}")

            scheduler.step()
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']

            # Get context gate/alpha info for logging
            extra_info = {"LR": current_lr}
            context_info = self.nf_model.get_context_info()
            if context_info:
                extra_info.update(context_info)

            if self.logger:
                self.logger.log_epoch(
                    task_id, epoch, num_epochs, avg_epoch_loss,
                    stage="SLOW", extra_info=extra_info
                )
            else:
                ctx_str = ""
                if 'gate_mean' in context_info:
                    ctx_str = f" | Gate: {context_info['gate_mean']:.4f}±{context_info['gate_std']:.4f}"
                elif 'alpha_mean' in context_info:
                    ctx_str = f" | Alpha: {context_info['alpha_mean']:.4f}"
                print(f"  📊 [SLOW] Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}{ctx_str}")

    def _build_prototype(self, task_id: int, task_classes: List[str],
                         train_loader: DataLoader):
        """Build prototype for routing."""
        self._log(f"\n📦 Building prototype for Task {task_id}...")

        all_image_features = []
        self.nf_model.eval()
        self.vit_extractor.eval()
        if self.use_stn and self.stn is not None:
            self.stn.eval()

        with torch.no_grad():
            for images, _, _, _, _ in train_loader:
                images = images.to(self.device)
                # V6.1: Apply STN for image alignment
                if self.use_stn and self.stn is not None:
                    images, _ = self.stn(images)
                image_features = self.vit_extractor.get_image_level_features(images)
                all_image_features.append(image_features)

        prototype = TaskPrototype(task_id, task_classes, self.device)
        all_features = torch.cat(all_image_features, dim=0)
        prototype.update(all_features)
        prototype.finalize()

        self.router.add_prototype(task_id, prototype)

        if self.logger:
            self.logger.log_prototype_creation(
                task_id, prototype.n_samples, prototype.mean.shape
            )
        else:
            print(f"   ✅ Prototype stored: μ shape={prototype.mean.shape}, "
                  f"n_samples={prototype.n_samples}")

    def _compute_ogp_basis(self, task_id: int, train_loader: DataLoader):
        """
        V3: Compute and store OGP gradient basis for completed task.

        This should be called AFTER training on a task is complete.
        The gradient subspace captures the important directions for this task.
        For Task > current, gradients will be projected to be orthogonal.
        """
        self._log(f"\n📐 [V3] Computing OGP basis for Task {task_id}...")

        # Create a wrapper that provides features in the expected format
        class FeatureDataLoader:
            def __init__(self, trainer, data_loader, device):
                self.trainer = trainer
                self.data_loader = data_loader
                self.device = device
                self._iter = None

            def __iter__(self):
                for images, _, _, _, _ in self.data_loader:
                    images = images.to(self.device)
                    # V6.1: Apply STN for image alignment
                    if self.trainer.use_stn and self.trainer.stn is not None:
                        with torch.no_grad():
                            images, _ = self.trainer.stn(images)
                    with torch.no_grad():
                        patch_embeddings, spatial_shape = self.trainer.vit_extractor(
                            images, return_spatial_shape=True
                        )
                        # V5.7: Skip standard PE when ContentBasedPE or HybridPE is enabled
                        use_v57_pe = self.trainer.use_content_based_pe or self.trainer.use_hybrid_pe
                        if use_v57_pe:
                            B = patch_embeddings.shape[0]
                            H, W = spatial_shape
                            features = patch_embeddings.reshape(B, H, W, -1)
                        elif self.trainer.use_pos_embedding:
                            features = self.trainer.pos_embed_generator(
                                spatial_shape, patch_embeddings
                            )
                        else:
                            B = patch_embeddings.shape[0]
                            H, W = spatial_shape
                            features = patch_embeddings.reshape(B, H, W, -1)
                    yield (features,)

        feature_loader = FeatureDataLoader(self, train_loader, self.device)

        # Compute and store basis
        self.ogp.compute_and_store_basis(
            model=self.nf_model,
            data_loader=feature_loader,
            task_id=task_id,
            n_samples=self.ogp_n_samples
        )

        # Log memory usage
        mem_info = self.ogp.get_memory_usage()
        self._log(f"   📊 OGP Memory: {mem_info['memory_mb']:.2f} MB "
                  f"({mem_info['n_params']} params, {mem_info['n_tasks']} tasks)")

    def inference(self, images: torch.Tensor, task_id: Optional[int] = None):
        """
        Inference with automatic routing or specified task.

        Args:
            images: (B, C, H, W) input images
            task_id: If None, use router to predict task (if enabled)

        Returns:
            anomaly_scores: (B, H_patch, W_patch) spatial anomaly map
            image_scores: (B,) image-level scores
            predicted_tasks: (B,) predicted task IDs in task-level mode.
                In class-level mode, this contains predicted global class IDs.

        Respects ablation flags:
        - use_router: If False, always use task_id (oracle mode)
        - use_pos_embedding: If False, skip positional embedding
        """
        self.nf_model.eval()
        self.vit_extractor.eval()
        if self.use_stn and self.stn is not None:
            self.stn.eval()

        with torch.no_grad():
            images = images.to(self.device)

            # V6.1: Apply STN for image alignment before feature extraction
            if self.use_stn and self.stn is not None:
                images, _ = self.stn(images)

            # Route if task_id not specified and router is enabled. In
            # class-level mode, the component ID is a global class ID.
            use_class_router = (
                self.use_class_level
                and self.class_router is not None
                and len(self.class_router.prototypes) > 0
            )
            if task_id is None and self.use_router and use_class_router:
                image_features = self.vit_extractor.get_image_level_features(images)
                predicted_tasks = self.class_router.route(image_features)
            elif task_id is None and self.use_router and self.router is not None and len(self.router.prototypes) > 0:
                image_features = self.vit_extractor.get_image_level_features(images)
                predicted_tasks = self.router.route(image_features)
            else:
                predicted_tasks = torch.full((images.shape[0],),
                                            task_id if task_id is not None else 0,
                                            dtype=torch.long, device=self.device)

            # Process each unique task
            unique_tasks = predicted_tasks.unique()

            B = images.shape[0]
            anomaly_scores = torch.zeros(B, 37, 37, device=self.device)
            image_scores = torch.zeros(B, device=self.device)

            for t_id in unique_tasks:
                mask = (predicted_tasks == t_id)
                task_images = images[mask]

                if task_images.shape[0] == 0:
                    continue

                # Set active component. Class-level mode uses class IDs as
                # adapter keys, while the standard path uses task IDs.
                if self.use_class_level and use_class_router:
                    self.nf_model.set_active_class(t_id.item())
                else:
                    self.nf_model.set_active_task(t_id.item())

                # Extract features
                patch_embeddings, spatial_shape = self.vit_extractor(
                    task_images, return_spatial_shape=True
                )
                B_task = patch_embeddings.shape[0]
                H, W = spatial_shape

                # V5.7/V5.8: Skip standard PE when custom PE is enabled
                use_custom_pe = self.use_content_based_pe or self.use_hybrid_pe or self.use_tape
                if use_custom_pe:
                    patch_embeddings_with_pos = patch_embeddings.reshape(B_task, H, W, -1)
                elif self.use_pos_embedding:
                    patch_embeddings_with_pos = self.pos_embed_generator(
                        spatial_shape, patch_embeddings
                    )
                else:
                    patch_embeddings_with_pos = patch_embeddings.reshape(B_task, H, W, -1)

                # V5.5/V5.6: For dual-branch, also create no-position version
                patch_embeddings_nopos = None
                use_any_dual = (self.use_dual_branch or self.use_improved_dual_branch or self.use_score_guided_dual)
                if use_any_dual:
                    patch_embeddings_nopos = patch_embeddings.reshape(B_task, H, W, -1)

                # Compute anomaly scores
                task_anomaly_scores, task_image_scores = self._compute_anomaly_scores(
                    patch_embeddings_with_pos, patch_embeddings_nopos
                )

                # Resize if needed
                H, W = spatial_shape
                if anomaly_scores.shape[1] != H or anomaly_scores.shape[2] != W:
                    anomaly_scores = torch.zeros(B, H, W, device=self.device)

                anomaly_scores[mask] = task_anomaly_scores
                image_scores[mask] = task_image_scores

        return anomaly_scores, image_scores, predicted_tasks

    def _compute_anomaly_scores(self, patch_embeddings_with_pos: torch.Tensor,
                                  patch_embeddings_nopos: torch.Tensor = None):
        """
        Compute anomaly scores from patch embeddings.

        Now uses patch-wise log|det J| directly for proper spatial localization.
        This preserves the Flow's spatial information in the anomaly map.

        V5.5: Optional dual-branch scoring when patch_embeddings_nopos is provided.
        V5.7: Multi-Orientation Ensemble for rotation-invariant detection.
        """
        B, H, W, D = patch_embeddings_with_pos.shape

        # V5.7: Multi-Orientation Ensemble (Direction C)
        # Test-time rotation ensemble: take minimum score across orientations
        if self.use_multi_orientation and hasattr(self.nf_model, 'multi_orientation') and self.nf_model.multi_orientation is not None:
            return self._compute_multi_orientation_scores(patch_embeddings_with_pos, patch_embeddings_nopos)

        # Forward through NF - now returns patch-wise logdet: (B, H, W)
        z_pos, logdet_pos = self.nf_model.forward(patch_embeddings_with_pos, reverse=False)

        # Patch-wise log p(z): (B, H, W)
        log_pz_pos = -0.5 * (z_pos ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Patch-wise log p(x) = log p(z) + log|det J|: (B, H, W)
        patch_log_prob_pos = log_pz_pos + logdet_pos

        # Anomaly score = -log p(x)
        anomaly_scores_pos = -patch_log_prob_pos

        # V5.5/V5.6: Dual-Branch Scoring (Direction 2)
        # Run NF twice: with and without positional embedding, blend scores
        use_any_dual = (self.use_dual_branch or self.use_improved_dual_branch or self.use_score_guided_dual)
        if use_any_dual and patch_embeddings_nopos is not None:
            # Forward through NF without positional embedding
            z_nopos, logdet_nopos = self.nf_model.forward(patch_embeddings_nopos, reverse=False)

            # Compute no-position branch scores
            log_pz_nopos = -0.5 * (z_nopos ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            patch_log_prob_nopos = log_pz_nopos + logdet_nopos
            anomaly_scores_nopos = -patch_log_prob_nopos

            # V5.6: Use improved dual branch scorer (anti-collapse)
            if self.use_improved_dual_branch and hasattr(self.nf_model, 'improved_dual_branch') and self.nf_model.improved_dual_branch is not None:
                anomaly_scores = self.nf_model.improved_dual_branch(
                    z_pos, z_nopos, anomaly_scores_pos, anomaly_scores_nopos
                )
            # V5.6: Use score-guided dual branch (simpler alternative)
            elif self.use_score_guided_dual and hasattr(self.nf_model, 'score_guided_dual') and self.nf_model.score_guided_dual is not None:
                anomaly_scores = self.nf_model.score_guided_dual(
                    z_pos, z_nopos, anomaly_scores_pos, anomaly_scores_nopos
                )
            # V5.5: Original dual branch scorer
            elif self.use_dual_branch and hasattr(self.nf_model, 'dual_branch_scorer') and self.nf_model.dual_branch_scorer is not None:
                anomaly_scores = self.nf_model.dual_branch_scorer(
                    z_pos, z_nopos, anomaly_scores_pos, anomaly_scores_nopos
                )
            else:
                # Fallback: simple average
                anomaly_scores = 0.5 * anomaly_scores_pos + 0.5 * anomaly_scores_nopos
        else:
            anomaly_scores = anomaly_scores_pos

        # V5: Image-level score aggregation (configurable)
        image_scores = self._aggregate_patch_scores(anomaly_scores)

        return anomaly_scores, image_scores

    def _aggregate_patch_scores(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """
        Aggregate patch-level anomaly scores to image-level scores.

        V5 Feature: Multiple aggregation modes for better image-level detection.
        V5.5 Feature: Local consistency calibration before aggregation.

        Args:
            patch_scores: (B, H, W) patch-level anomaly scores

        Returns:
            image_scores: (B,) image-level anomaly scores
        """
        # V5.6: Apply Multi-Scale Consistency (Improved Direction 3)
        if self.use_multiscale_consistency and hasattr(self.nf_model, 'multiscale_consistency') and self.nf_model.multiscale_consistency is not None:
            patch_scores = self.nf_model.multiscale_consistency(patch_scores)
        # V5.5: Apply Local Consistency Calibration (Direction 3)
        # Down-weights isolated high scores that are likely rotation noise
        elif self.use_local_consistency and hasattr(self.nf_model, 'local_consistency') and self.nf_model.local_consistency is not None:
            patch_scores = self.nf_model.local_consistency(patch_scores)

        B = patch_scores.shape[0]
        flat_scores = patch_scores.reshape(B, -1)  # (B, H*W)
        num_patches = flat_scores.shape[1]

        mode = self.score_aggregation_mode

        if mode == "percentile":
            # Original method: p-th percentile
            p = self.score_aggregation_percentile
            image_scores = torch.quantile(flat_scores, p, dim=1)

        elif mode == "top_k":
            # Average of top-K highest scoring patches
            k = min(self.score_aggregation_top_k, num_patches)
            top_k_scores, _ = torch.topk(flat_scores, k, dim=1)
            image_scores = top_k_scores.mean(dim=1)

        elif mode == "top_k_percent":
            # Average of top K% highest scoring patches
            k = max(1, int(num_patches * self.score_aggregation_top_k_percent))
            top_k_scores, _ = torch.topk(flat_scores, k, dim=1)
            image_scores = top_k_scores.mean(dim=1)

        elif mode == "max":
            # Maximum patch score
            image_scores = flat_scores.max(dim=1)[0]

        elif mode == "mean":
            # Mean of all patch scores
            image_scores = flat_scores.mean(dim=1)

        elif mode == "spatial_cluster":
            # V5: Spatial Clustering Score
            # Real defects form spatial clusters, noise is randomly distributed
            # Score = cluster_weight * clustered_score + (1 - cluster_weight) * top_k_score
            image_scores = self._compute_spatial_cluster_score(patch_scores)

        else:
            # Default fallback to percentile
            image_scores = torch.quantile(flat_scores, 0.99, dim=1)

        return image_scores

    def _compute_spatial_cluster_score(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """
        V5: Compute spatial clustering-aware anomaly score.

        Key Insight: Real defects form spatially clustered high-score regions,
        while noise produces randomly scattered high scores.

        Algorithm:
        1. Identify high-scoring patches (above threshold)
        2. Compute connectivity score based on 8-connected neighbors
        3. Weight clustered high scores more heavily

        Args:
            patch_scores: (B, H, W) patch-level anomaly scores

        Returns:
            image_scores: (B,) cluster-aware image-level scores
        """
        B, H, W = patch_scores.shape
        device = patch_scores.device

        # Step 1: Determine high-score threshold per image
        flat_scores = patch_scores.reshape(B, -1)
        threshold = torch.quantile(flat_scores, self.cluster_high_score_percentile, dim=1, keepdim=True)
        threshold = threshold.reshape(B, 1, 1)  # (B, 1, 1)

        # Step 2: Create binary mask for high-scoring patches
        high_score_mask = (patch_scores > threshold).float()  # (B, H, W)

        # Step 3: Count connected high-scoring neighbors (8-connectivity)
        # Pad to handle boundary conditions
        padded_mask = torch.nn.functional.pad(high_score_mask, (1, 1, 1, 1), mode='constant', value=0)

        # Sum of 8 neighbors (excluding self)
        neighbor_count = torch.zeros_like(high_score_mask)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbor_count += padded_mask[:, 1+di:1+di+H, 1+dj:1+dj+W]

        # Connectivity score: high when patch is part of a cluster
        # Range: 0 (isolated) to 8 (fully surrounded)
        connectivity = neighbor_count * high_score_mask  # Only count for high-scoring patches
        max_connectivity = connectivity.reshape(B, -1).max(dim=1)[0].clamp(min=1)  # Prevent div by zero

        # Step 4: Compute cluster-weighted scores
        # Normalize connectivity to [0, 1] range
        normalized_connectivity = connectivity / 8.0

        # Cluster-weighted score: high scores with high connectivity count more
        weight = 1.0 + normalized_connectivity  # Range: [1, 2]
        weighted_scores = patch_scores * weight
        weighted_flat = weighted_scores.reshape(B, -1)

        # Take top-k weighted scores
        k = max(1, int(H * W * self.tail_top_k_ratio))
        top_k_weighted, _ = torch.topk(weighted_flat, k, dim=1)
        clustered_score = top_k_weighted.mean(dim=1)

        # Also compute standard top-k for comparison
        top_k_standard, _ = torch.topk(flat_scores, k, dim=1)
        standard_score = top_k_standard.mean(dim=1)

        # Step 5: Combine scores
        w = self.cluster_weight
        image_scores = w * clustered_score + (1 - w) * standard_score

        return image_scores

    def _compute_multi_orientation_scores(self, patch_embeddings_with_pos: torch.Tensor,
                                           patch_embeddings_nopos: torch.Tensor = None):
        """
        V5.7 Direction C: Multi-Orientation Ensemble for rotation-invariant detection.

        핵심 아이디어:
        - Test time에 여러 회전 방향에서 NF 적용
        - 각 방향에서 anomaly score 계산
        - 가장 낮은 score 선택 (가장 "정상"인 방향)

        직관:
        - 정상 이미지: 최소 1개 방향에서 낮은 score → min 낮음
        - 비정상 이미지: 모든 방향에서 높은 score → min도 높음

        이점:
        - 학습 변경 없음 (inference만 수정)
        - 완벽한 rotation invariance 보장
        - 단점: N배 inference 비용

        Args:
            patch_embeddings_with_pos: (B, H, W, D) features with positional embedding
            patch_embeddings_nopos: (B, H, W, D) features without positional embedding (optional)

        Returns:
            anomaly_scores: (B, H, W) patch-level anomaly scores
            image_scores: (B,) image-level anomaly scores
        """
        B, H, W, D = patch_embeddings_with_pos.shape
        multi_orientation = self.nf_model.multi_orientation

        # Get rotation angles
        angles = multi_orientation.get_orientations()

        # Store scores for each orientation
        all_anomaly_scores = []

        for angle in angles:
            # Rotate features
            rotated_features = multi_orientation.rotate_features(patch_embeddings_with_pos, angle)

            # Forward through NF
            z, logdet = self.nf_model.forward(rotated_features, reverse=False)

            # Compute anomaly scores
            log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            patch_log_prob = log_pz + logdet
            anomaly_scores = -patch_log_prob  # (B, H, W)

            # Inverse rotate scores to original orientation
            anomaly_scores = multi_orientation.inverse_rotate_scores(anomaly_scores, angle)

            all_anomaly_scores.append(anomaly_scores)

        # Stack: (N_orientations, B, H, W)
        stacked_scores = torch.stack(all_anomaly_scores, dim=0)

        # Take minimum across orientations (most "normal" score)
        # 정상: 최소 1개 방향에서 잘 맞음 → min 낮음
        # 비정상: 모든 방향에서 안 맞음 → min도 높음
        min_scores, _ = stacked_scores.min(dim=0)  # (B, H, W)

        # Aggregate to image-level
        image_scores = self._aggregate_patch_scores(min_scores)

        return min_scores, image_scores
