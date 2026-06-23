"""
Feature Statistics and Task Input Adapters.

Provides distribution alignment for cross-task consistency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class FeatureStatistics:
    """
    Store reference feature statistics from Task 0 for distribution alignment.

    Key Insight: We DON'T normalize new task features using Task 0 statistics directly.
    Instead, we store these statistics and pass them to TaskInputAdapter,
    which learns to transform new task features to match the reference distribution.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.is_initialized = False
        self.n_samples = 0

        # Welford's online algorithm for stable mean/variance computation
        self._M2: Optional[torch.Tensor] = None  # Sum of squared deviations

    def update(self, features: torch.Tensor):
        """
        Update running statistics using Welford's online algorithm.
        More stable than simple EMA for computing variance.

        Args:
            features: (B, H, W, D) or (N, D) features
        """
        # Flatten to (N, D)
        if features.dim() == 4:
            B, H, W, D = features.shape
            features = features.reshape(-1, D)

        features = features.detach()
        batch_size = features.shape[0]

        if self.mean is None:
            self.mean = features.mean(dim=0)
            self._M2 = ((features - self.mean.unsqueeze(0)) ** 2).sum(dim=0)
            self.n_samples = batch_size
        else:
            # Welford's online update
            for i in range(batch_size):
                self.n_samples += 1
                delta = features[i] - self.mean
                self.mean = self.mean + delta / self.n_samples
                delta2 = features[i] - self.mean
                self._M2 = self._M2 + delta * delta2

    def finalize(self):
        """
        Compute final std and mark as initialized.
        """
        if self._M2 is not None and self.n_samples > 1:
            variance = self._M2 / (self.n_samples - 1)
            self.std = torch.sqrt(variance + 1e-6)
        else:
            self.std = torch.ones_like(self.mean)

        self.is_initialized = True
        print(f"📊 Feature Statistics Finalized:")
        print(f"   - n_samples: {self.n_samples}")
        print(f"   - mean_norm: {self.mean.norm():.4f}")
        print(f"   - std_mean: {self.std.mean():.4f}")
        print(f"   - std_min: {self.std.min():.4f}, std_max: {self.std.max():.4f}")

    def get_reference_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get reference mean and std for TaskInputAdapter initialization.

        Returns:
            (mean, std) tensors
        """
        if not self.is_initialized:
            raise ValueError("Statistics not finalized. Call finalize() first.")
        return self.mean.clone(), self.std.clone()


class TaskInputAdapter(nn.Module):
    """
    FiLM-style Task Input Adapter (v3).

    Key Design Principles:
    1. FiLM (Feature-wise Linear Modulation): y = gamma * x + beta
    2. Layer Norm option to preserve spatial information
    3. Larger MLP capacity with active residual gate
    4. Can work with or without reference statistics

    v3 Changes from v2:
    - Instance Norm → Layer Norm (preserves spatial info)
    - residual_gate: 0 → 0.5 (MLP actively used from start)
    - Larger hidden_dim for more capacity
    - Optional use_norm flag for Task 0 self-adaptation
    """

    def __init__(self, channels: int, reference_mean: torch.Tensor = None,
                 reference_std: torch.Tensor = None, use_norm: bool = True):
        super(TaskInputAdapter, self).__init__()

        self.channels = channels
        self.eps = 1e-6
        self.use_norm = use_norm

        # Store reference statistics (from Task 0) - used for target distribution
        if reference_mean is not None:
            self.register_buffer('reference_mean', reference_mean.clone())
            self.register_buffer('reference_std', reference_std.clone())
            self.has_reference = True
        else:
            self.register_buffer('reference_mean', torch.zeros(channels))
            self.register_buffer('reference_std', torch.ones(channels))
            self.has_reference = False

        # FiLM parameters: y = gamma * x + beta
        # Initialize gamma=1, beta=0 for identity start
        self.film_gamma = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.film_beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

        # Larger MLP for stronger feature transformation
        hidden_dim = max(channels // 2, 128)  # Increased from channels//4
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )

        # Initialize last layer to small values (not zero) for faster learning
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

        # Residual gate - starts at 0.5 for active MLP contribution
        # Changed from 0.0 to enable MLP from the beginning
        self.residual_gate = nn.Parameter(torch.tensor([0.5]))

        # Optional Layer Norm (preserves spatial info better than Instance Norm)
        if use_norm:
            self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FiLM-style feature modulation with residual MLP.

        Flow: x -> (optional LayerNorm) -> FiLM -> MLP residual

        Args:
            x: (B, H, W, D) input features

        Returns:
            Transformed features with same shape
        """
        B, H, W, D = x.shape
        identity = x

        # 1. Optional normalization (Layer Norm preserves spatial structure)
        if self.use_norm:
            x_flat = x.reshape(-1, D)
            x_normed = self.layer_norm(x_flat)
            x = x_normed.reshape(B, H, W, D)

        # 2. FiLM modulation: y = gamma * x + beta
        x = self.film_gamma * x + self.film_beta

        # 3. MLP residual with learnable gate
        gate = torch.sigmoid(self.residual_gate)  # 0 ~ 1
        x_flat = x.reshape(-1, D)
        mlp_out = self.mlp(x_flat).reshape(B, H, W, D)
        x = x + gate * mlp_out

        # 4. Optional: blend with identity for stability
        # This helps Task 0 where we want minimal transformation
        if not self.has_reference:
            # For Task 0 self-adapter: stronger identity connection
            x = 0.9 * identity + 0.1 * x

        return x


class SimpleTaskAdapter(nn.Module):
    """
    Simpler adapter using stored Task 0 statistics for alignment.

    Transforms new task features to have similar distribution to Task 0.
    """

    def __init__(self, channels: int, reference_stats: FeatureStatistics):
        super(SimpleTaskAdapter, self).__init__()

        self.channels = channels
        self.reference_stats = reference_stats

        # Learnable transformation to match reference distribution
        self.target_mean = nn.Parameter(torch.zeros(channels))
        self.target_std = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize to zero mean/unit std, then transform to target distribution.

        Args:
            x: (B, H, W, D) input features

        Returns:
            Aligned features
        """
        B, H, W, D = x.shape

        # Compute current batch statistics
        x_flat = x.reshape(-1, D)
        batch_mean = x_flat.mean(dim=0, keepdim=True)
        batch_std = x_flat.std(dim=0, keepdim=True) + 1e-6

        # Normalize to zero mean, unit std
        x_normalized = (x_flat - batch_mean) / batch_std

        # Transform to target distribution (learned)
        # Initialize target to reference stats if available
        if self.reference_stats.is_initialized:
            target_mean = self.reference_stats.mean + self.target_mean
            target_std = self.reference_stats.std * torch.exp(self.target_std - 1)
        else:
            target_mean = self.target_mean
            target_std = torch.exp(self.target_std)

        x_aligned = x_normalized * target_std + target_mean

        return x_aligned.reshape(B, H, W, D)


class SoftLNTaskInputAdapter(nn.Module):
    """
    Task Input Adapter with Soft/Optional LayerNorm.

    Key Design (Baseline 1.4):
    - Task 0: LayerNorm ON (weak) - sets the base density anchor
    - Task > 0: LayerNorm OFF (blend=0, fixed) - let Flow's scale(s) + LoRA handle distribution

    This prevents LN from stealing Flow's role:
    - Flow already has mean shift (t(x)) and scale (s(x))
    - LN before Flow makes scale(s) meaningless → log_det noise
    - Task 0 LN provides the reference point, then Flow takes over

    Parameters:
    - task_id: 0 for base task (LN on), >0 for subsequent tasks (LN off)
    - soft_ln_init_scale: Initial blend for Task 0 (default 0.01 = very weak)
    """

    def __init__(self, channels: int, reference_mean: torch.Tensor = None,
                 reference_std: torch.Tensor = None, task_id: int = 0,
                 soft_ln_init_scale: float = 0.01):
        super(SoftLNTaskInputAdapter, self).__init__()

        self.channels = channels
        self.eps = 1e-6
        self.task_id = task_id
        self.soft_ln_init_scale = soft_ln_init_scale

        # Task 0: LN on (learnable blend starting at soft_ln_init_scale)
        # Task > 0: LN off (blend fixed at 0)
        self.use_ln = (task_id == 0)

        # Store reference statistics (from Task 0)
        if reference_mean is not None:
            self.register_buffer('reference_mean', reference_mean.clone())
            self.register_buffer('reference_std', reference_std.clone())
            self.has_reference = True
        else:
            self.register_buffer('reference_mean', torch.zeros(channels))
            self.register_buffer('reference_std', torch.ones(channels))
            self.has_reference = False

        # FiLM parameters
        self.film_gamma = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.film_beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

        # MLP
        hidden_dim = max(channels // 2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

        self.residual_gate = nn.Parameter(torch.tensor([0.5]))

        # LayerNorm (only used for Task 0)
        self.layer_norm = nn.LayerNorm(channels)

        # Soft LN blend factor
        # Task 0: learnable, starts at soft_ln_init_scale
        # Task > 0: fixed at 0 (LN completely off)
        if self.use_ln:
            self.ln_blend = nn.Parameter(torch.tensor([soft_ln_init_scale]))
        else:
            # Fixed at 0 for Task > 0
            self.register_buffer('ln_blend', torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FiLM-style feature modulation with soft LayerNorm.

        Task 0: Soft LN applied (weak, learnable blend)
        Task > 0: LN skipped (blend=0), only FiLM + MLP applied

        Args:
            x: (B, H, W, D) input features

        Returns:
            Transformed features
        """
        B, H, W, D = x.shape
        identity = x

        # 1. Soft LayerNorm: blend between normalized and original
        # Task 0: learnable blend (starts at soft_ln_init_scale)
        # Task > 0: blend=0 (LN effectively off, but still computes for stability)
        x_flat = x.reshape(-1, D)
        x_normed = self.layer_norm(x_flat).reshape(B, H, W, D)

        # Soft blending: scale is clamped to [0, 1]
        blend = torch.clamp(self.ln_blend, 0.0, 1.0)
        x = blend * x_normed + (1 - blend) * x

        # 2. FiLM modulation
        x = self.film_gamma * x + self.film_beta

        # 3. MLP residual with learnable gate
        gate = torch.sigmoid(self.residual_gate)
        x_flat = x.reshape(-1, D)
        mlp_out = self.mlp(x_flat).reshape(B, H, W, D)
        x = x + gate * mlp_out

        # 4. Blend with identity for stability (for self-adapting Task 0)
        if not self.has_reference:
            x = 0.9 * identity + 0.1 * x

        return x


def create_task_adapter(adapter_mode: str, channels: int,
                        reference_mean: torch.Tensor = None,
                        reference_std: torch.Tensor = None,
                        task_id: int = 0,
                        soft_ln_init_scale: float = 0.01,
                        **kwargs) -> nn.Module:
    """
    Factory function to create the appropriate adapter based on mode.

    Args:
        adapter_mode: "soft_ln", "standard", "no_ln_after_task0", "no_ln", "tsa", "tsa_no_ln"
        channels: Feature dimension
        reference_mean: Reference mean from Task 0
        reference_std: Reference std from Task 0
        task_id: Current task ID
        soft_ln_init_scale: Initial scale for soft LN

    Returns:
        Appropriate adapter module
    """
    if adapter_mode == "tsa":
        # V3: LayerNorm + task-specific affine alignment
        return TaskSpecificAlignment(
            channels=channels,
            task_id=task_id,
            reference_mean=reference_mean,
            reference_std=reference_std,
            gamma_range=kwargs.get('gamma_range', (0.5, 2.0)),
            beta_max=kwargs.get('beta_max', 2.0)
        )
    elif adapter_mode == "tsa_no_ln":
        # Ablation: Same as TSA but WITHOUT LayerNorm
        # Tests the effect of LayerNorm in isolation
        return TaskSpecificAlignmentNoLN(
            channels=channels,
            task_id=task_id,
            reference_mean=reference_mean,
            reference_std=reference_std,
            gamma_range=kwargs.get('gamma_range', (0.5, 2.0)),
            beta_max=kwargs.get('beta_max', 2.0)
        )
    elif adapter_mode == "soft_ln":
        # SoftLN: Task 0 has learnable LN blend, Task > 0 has blend=0 (LN off)
        return SoftLNTaskInputAdapter(
            channels=channels,
            reference_mean=reference_mean,
            reference_std=reference_std,
            task_id=task_id,  # Critical: controls LN behavior
            soft_ln_init_scale=soft_ln_init_scale
        )
    elif adapter_mode == "no_ln_after_task0":
        # Task 0: use LN, Task > 0: no LN (hard switch)
        use_norm = (task_id == 0)
        return TaskInputAdapter(
            channels=channels,
            reference_mean=reference_mean,
            reference_std=reference_std,
            use_norm=use_norm
        )
    elif adapter_mode == "no_ln":
        # No LN for ALL tasks (ablation study)
        return TaskInputAdapter(
            channels=channels,
            reference_mean=reference_mean,
            reference_std=reference_std,
            use_norm=False
        )
    else:  # "standard"
        return TaskInputAdapter(
            channels=channels,
            reference_mean=reference_mean,
            reference_std=reference_std,
            use_norm=True
        )


# =============================================================================
# V3 Improvements: Task-Specific Alignment
# =============================================================================

class TaskSpecificAlignment(nn.Module):
    """
    Task-Specific Alignment module (V3 Solution 3).

    Key Design:
    1. All tasks go through LayerNorm first (mean=0, std=1)
    2. task-specific affine alignment with constrained parameters
    3. Task 0 stays close to identity (anchor point)

    This ensures:
    - All tasks have consistent intermediate representation
    - Base NF receives well-normalized inputs regardless of task
    - Task-specific adaptation is controlled and stable

    Parameters:
    - gamma: constrained to [0.5, 2.0] via sigmoid
    - beta: constrained to [-2.0, 2.0] via tanh
    """

    def __init__(self, channels: int, task_id: int = 0,
                 reference_mean: torch.Tensor = None,
                 reference_std: torch.Tensor = None,
                 gamma_range: tuple = (0.5, 2.0),
                 beta_max: float = 2.0):
        super(TaskSpecificAlignment, self).__init__()

        self.channels = channels
        self.task_id = task_id
        self.gamma_min, self.gamma_max = gamma_range
        self.beta_max = beta_max

        # TSA layer (shared across all tasks, no learnable affine)
        self.whiten = nn.LayerNorm(channels, elementwise_affine=False)

        # Store reference statistics for logging/debugging
        if reference_mean is not None:
            self.register_buffer('reference_mean', reference_mean.clone())
            self.register_buffer('reference_std', reference_std.clone())
            self.has_reference = True
        else:
            self.register_buffer('reference_mean', torch.zeros(channels))
            self.register_buffer('reference_std', torch.ones(channels))
            self.has_reference = False

        if task_id == 0:
            # Task 0: Start very close to identity, learnable but regularized
            # gamma_raw=0 → sigmoid=0.5 → gamma = 0.5 + 1.5*0.5 = 1.25
            # We want gamma ≈ 1.0, so init gamma_raw to make sigmoid ≈ 0.33
            # sigmoid(x) = 0.33 → x ≈ -0.7
            init_gamma_raw = -0.7 * torch.ones(1, 1, 1, channels)
            self.gamma_raw = nn.Parameter(init_gamma_raw)
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.identity_reg_weight = 0.1  # Regularize toward identity
        else:
            # Task 1+: Learnable, initialized based on reference stats
            self.gamma_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.identity_reg_weight = 0.0

    @property
    def gamma(self):
        """Constrained gamma in [gamma_min, gamma_max]."""
        gamma_range = self.gamma_max - self.gamma_min
        return self.gamma_min + gamma_range * torch.sigmoid(self.gamma_raw)

    @property
    def beta(self):
        """Constrained beta in [-beta_max, beta_max]."""
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm followed by task-specific affine alignment.

        Args:
            x: (B, H, W, D) input features

        Returns:
            Transformed features with same shape
        """
        B, H, W, D = x.shape

        # 1. TSA: normalize to N(0, 1)
        x_flat = x.reshape(-1, D)
        x_white = self.whiten(x_flat).reshape(B, H, W, D)

        # 2. task-specific affine alignment
        x_out = self.gamma * x_white + self.beta

        return x_out

    def identity_regularization(self) -> torch.Tensor:
        """
        Regularization loss to keep Task 0 adapter close to identity.

        Returns:
            Regularization loss term
        """
        if self.identity_reg_weight > 0:
            # gamma should be close to 1.0
            gamma_reg = ((self.gamma - 1.0) ** 2).mean()
            # beta should be close to 0.0
            beta_reg = (self.beta ** 2).mean()
            return self.identity_reg_weight * (gamma_reg + beta_reg)
        return torch.tensor(0.0, device=self.gamma_raw.device)

    def get_stats(self) -> dict:
        """Get current adapter statistics for logging."""
        with torch.no_grad():
            return {
                'gamma_mean': self.gamma.mean().item(),
                'gamma_std': self.gamma.std().item(),
                'beta_mean': self.beta.mean().item(),
                'beta_std': self.beta.std().item(),
            }


class TaskSpecificAlignmentNoLN(nn.Module):
    """
    TaskSpecificAlignment WITHOUT LayerNorm (Ablation Study).

    Same structure as TaskSpecificAlignment but without the LayerNorm step.
    This allows testing the effect of LayerNorm in isolation.

    Forward: x -> gamma * x + beta (NO LayerNorm)

    Compare with TaskSpecificAlignment:
    Forward: x -> LayerNorm(x) -> gamma * LN(x) + beta
    """

    def __init__(self, channels: int, task_id: int = 0,
                 reference_mean: torch.Tensor = None,
                 reference_std: torch.Tensor = None,
                 gamma_range: tuple = (0.5, 2.0),
                 beta_max: float = 2.0):
        super(TaskSpecificAlignmentNoLN, self).__init__()

        self.channels = channels
        self.task_id = task_id
        self.gamma_min, self.gamma_max = gamma_range
        self.beta_max = beta_max

        # NO LayerNorm layer (key difference from TaskSpecificAlignment)

        # Store reference statistics for logging/debugging
        if reference_mean is not None:
            self.register_buffer('reference_mean', reference_mean.clone())
            self.register_buffer('reference_std', reference_std.clone())
            self.has_reference = True
        else:
            self.register_buffer('reference_mean', torch.zeros(channels))
            self.register_buffer('reference_std', torch.ones(channels))
            self.has_reference = False

        if task_id == 0:
            # Task 0: Start very close to identity
            init_gamma_raw = -0.7 * torch.ones(1, 1, 1, channels)
            self.gamma_raw = nn.Parameter(init_gamma_raw)
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.identity_reg_weight = 0.1
        else:
            # Task 1+: Learnable, initialized at midpoint
            self.gamma_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.identity_reg_weight = 0.0

    @property
    def gamma(self):
        """Constrained gamma in [gamma_min, gamma_max]."""
        gamma_range = self.gamma_max - self.gamma_min
        return self.gamma_min + gamma_range * torch.sigmoid(self.gamma_raw)

    @property
    def beta(self):
        """Constrained beta in [-beta_max, beta_max]."""
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply task-specific scaling WITHOUT LayerNorm.

        Args:
            x: (B, H, W, D) input features

        Returns:
            Transformed features with same shape
        """
        # NO LayerNorm - direct gamma/beta application
        # This preserves: ||x||, mean(x), std(x) information
        x_out = self.gamma * x + self.beta

        return x_out

    def identity_regularization(self) -> torch.Tensor:
        """Regularization loss to keep Task 0 adapter close to identity."""
        if self.identity_reg_weight > 0:
            gamma_reg = ((self.gamma - 1.0) ** 2).mean()
            beta_reg = (self.beta ** 2).mean()
            return self.identity_reg_weight * (gamma_reg + beta_reg)
        return torch.tensor(0.0, device=self.gamma_raw.device)

    def get_stats(self) -> dict:
        """Get current adapter statistics for logging."""
        with torch.no_grad():
            return {
                'gamma_mean': self.gamma.mean().item(),
                'gamma_std': self.gamma.std().item(),
                'beta_mean': self.beta.mean().item(),
                'beta_std': self.beta.std().item(),
            }


class SpatialContextMixer(nn.Module):
    """
    Spatial Context Mixer for NF input preprocessing.

    Key Design (Baseline 1.5):
    - Current problem: scale(s) only sees individual patch features
    - Solution: Add shallow spatial mixing so scale(s) can see local context

    This allows the Flow to detect:
    - "Is this patch abnormal COMPARED TO neighbors?" (local contrast)
    - Not just "Is this patch abnormal in isolation?"

    Implementation:
    - 3x3 depthwise conv (channel-wise, preserves spatial structure)
    - Optional: local statistics (mean, std) concatenation
    - Residual connection for stability

    Modes:
    - "depthwise": 3x3 depthwise conv only (preserves dim)
    - "depthwise_residual": depthwise conv + residual
    - "local_stats": concat local mean/std (doubles dim, needs projection)
    - "full": depthwise + local_stats + projection back to original dim
    """

    def __init__(self, channels: int, mode: str = "depthwise_residual",
                 kernel_size: int = 3, learnable: bool = True):
        super(SpatialContextMixer, self).__init__()

        self.channels = channels
        self.mode = mode
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 3x3 Depthwise conv: each channel has its own 3x3 kernel
        # groups=channels means each channel is convolved independently
        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            groups=channels,  # Depthwise: each channel separately
            bias=True
        )

        # Initialize to identity-like (center = 1, others = small)
        if not learnable:
            # Fixed averaging kernel
            with torch.no_grad():
                self.depthwise_conv.weight.fill_(1.0 / (kernel_size * kernel_size))
                self.depthwise_conv.bias.zero_()
            for param in self.depthwise_conv.parameters():
                param.requires_grad = False
        else:
            # Learnable, initialized to slight smoothing
            nn.init.constant_(self.depthwise_conv.weight, 1.0 / (kernel_size * kernel_size))
            nn.init.zeros_(self.depthwise_conv.bias)

        # Residual gate (learnable blend between original and context)
        self.residual_gate = nn.Parameter(torch.tensor([0.5]))

        # For "local_stats" or "full" mode: projection to preserve dimensions
        if mode in ["local_stats", "full"]:
            # Local stats doubles the channels (concat mean, std)
            # Project back to original dimension
            self.stats_proj = nn.Sequential(
                nn.Linear(channels * 3 if mode == "full" else channels * 2, channels),
                nn.GELU(),
                nn.Linear(channels, channels)
            )
            nn.init.zeros_(self.stats_proj[-1].weight)
            nn.init.zeros_(self.stats_proj[-1].bias)

    def _compute_local_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute local mean and std using average pooling.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            local_mean: (B, C, H, W)
            local_std: (B, C, H, W)
        """
        # Use average pooling to get local mean
        local_mean = F.avg_pool2d(
            x, kernel_size=self.kernel_size, stride=1, padding=self.padding
        )

        # Compute local variance: E[X^2] - E[X]^2
        local_sq_mean = F.avg_pool2d(
            x ** 2, kernel_size=self.kernel_size, stride=1, padding=self.padding
        )
        local_var = local_sq_mean - local_mean ** 2
        local_std = torch.sqrt(local_var.clamp(min=1e-6))

        return local_mean, local_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial context mixing.

        Args:
            x: (B, H, W, D) input features (note: H, W, D order from ViT)

        Returns:
            Mixed features with same shape (B, H, W, D)
        """
        B, H, W, D = x.shape
        identity = x

        # Convert to (B, D, H, W) for conv2d
        x_conv = x.permute(0, 3, 1, 2)  # (B, D, H, W)

        if self.mode == "depthwise":
            # Simple depthwise conv
            x_mixed = self.depthwise_conv(x_conv)
            x_out = x_mixed.permute(0, 2, 3, 1)  # Back to (B, H, W, D)

        elif self.mode == "depthwise_residual":
            # Depthwise conv with residual
            x_context = self.depthwise_conv(x_conv)
            gate = torch.sigmoid(self.residual_gate)
            x_mixed = (1 - gate) * x_conv + gate * x_context
            x_out = x_mixed.permute(0, 2, 3, 1)  # Back to (B, H, W, D)

        elif self.mode == "local_stats":
            # Concat local mean and std, then project back
            local_mean, local_std = self._compute_local_stats(x_conv)
            # Concat: (B, 2D, H, W)
            x_concat = torch.cat([x_conv, local_std], dim=1)
            # Permute to (B, H, W, 2D) for linear
            x_concat = x_concat.permute(0, 2, 3, 1)
            # Project back to D
            x_proj = self.stats_proj(x_concat)
            # Residual
            gate = torch.sigmoid(self.residual_gate)
            x_out = (1 - gate) * identity + gate * x_proj

        elif self.mode == "full":
            # Depthwise conv + local stats
            x_context = self.depthwise_conv(x_conv)
            local_mean, local_std = self._compute_local_stats(x_conv)
            # Concat original, context, local_std: (B, 3D, H, W)
            x_concat = torch.cat([x_conv, x_context, local_std], dim=1)
            # Permute to (B, H, W, 3D)
            x_concat = x_concat.permute(0, 2, 3, 1)
            # Project back to D
            x_proj = self.stats_proj(x_concat)
            # Residual
            gate = torch.sigmoid(self.residual_gate)
            x_out = (1 - gate) * identity + gate * x_proj

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return x_out


# =============================================================================
# V5 Structural Improvements: Semantic Projector & Context Modules
# =============================================================================

class SemanticProjector(nn.Module):
    """
    V5 Phase 2: Position-Agnostic Semantic Projector.

    Problem: Standard features are position-entangled, making the model
    sensitive to geometric variations (rotation, translation).

    Solution: Learn a semantic representation that is invariant to position
    through permutation-invariant pooling operations.

    Key Design:
    1. Extract position-agnostic statistics (mean, max, std)
    2. Project to semantic bottleneck (compressed representation)
    3. Broadcast back to spatial features as semantic context

    This allows the model to:
    - Recognize "screw thread" regardless of rotation angle
    - Focus on semantic patterns rather than exact positions
    """

    def __init__(self, channels: int, bottleneck_ratio: float = 0.5):
        super(SemanticProjector, self).__init__()

        self.channels = channels
        bottleneck_dim = int(channels * bottleneck_ratio)

        # Permutation-invariant feature aggregation
        # Input: (mean, max, std) = 3 * channels
        self.semantic_encoder = nn.Sequential(
            nn.Linear(channels * 3, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )

        # Project semantic features back to channel space
        self.semantic_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, channels),
            nn.LayerNorm(channels)
        )

        # Learnable gate for semantic influence
        self.semantic_gate = nn.Parameter(torch.tensor([0.3]))

        # Initialize last layer small for gradual learning
        nn.init.zeros_(self.semantic_decoder[-2].weight)
        nn.init.zeros_(self.semantic_decoder[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add position-agnostic semantic context to spatial features.

        Args:
            x: (B, H, W, D) spatial features

        Returns:
            Enhanced features with semantic context (B, H, W, D)
        """
        B, H, W, D = x.shape
        identity = x

        # Flatten spatial dimensions
        x_flat = x.reshape(B, -1, D)  # (B, H*W, D)

        # Compute permutation-invariant statistics
        x_mean = x_flat.mean(dim=1)  # (B, D)
        x_max = x_flat.max(dim=1)[0]  # (B, D)
        x_std = x_flat.std(dim=1)  # (B, D)

        # Concatenate for semantic encoding
        x_stats = torch.cat([x_mean, x_max, x_std], dim=-1)  # (B, 3*D)

        # Encode to semantic bottleneck
        semantic = self.semantic_encoder(x_stats)  # (B, bottleneck_dim)

        # Decode back to channel space
        semantic_ctx = self.semantic_decoder(semantic)  # (B, D)

        # Broadcast to all spatial positions
        semantic_ctx = semantic_ctx.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
        semantic_ctx = semantic_ctx.expand(-1, H, W, -1)  # (B, H, W, D)

        # Blend with original features
        gate = torch.sigmoid(self.semantic_gate)
        x_out = identity + gate * semantic_ctx

        return x_out

    def get_semantic_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract only the semantic features (for analysis/visualization).

        Args:
            x: (B, H, W, D) spatial features

        Returns:
            Semantic features (B, bottleneck_dim)
        """
        B, H, W, D = x.shape
        x_flat = x.reshape(B, -1, D)

        x_mean = x_flat.mean(dim=1)
        x_max = x_flat.max(dim=1)[0]
        x_std = x_flat.std(dim=1)

        x_stats = torch.cat([x_mean, x_max, x_std], dim=-1)
        semantic = self.semantic_encoder(x_stats)

        return semantic


class TaskAdaptiveContextMixer(nn.Module):
    """
    V5 Phase 2: Task-Adaptive Context Mixer.

    Problem: SpatialMixer is frozen after Task 0, limiting adaptation
    to new task contexts (e.g., different object sizes, spatial patterns).

    Solution: Add lightweight task-specific adapters that modify the
    mixing behavior without changing the frozen base mixer.

    Key Design:
    1. Keep the base SpatialContextMixer frozen (preserves Task 0 knowledge)
    2. Add per-task lightweight scaling and shift
    3. Minimal parameters per task (only gamma, beta per channel)

    This is like LoRA but for spatial mixing:
    - Base mixer: frozen, shared across all tasks
    - Task adapter: learned per task, scales/shifts the mixing output
    """

    def __init__(self, channels: int, base_mixer: SpatialContextMixer):
        super(TaskAdaptiveContextMixer, self).__init__()

        self.channels = channels

        # Frozen base mixer (from Task 0)
        self.base_mixer = base_mixer
        for param in self.base_mixer.parameters():
            param.requires_grad = False

        # Task-specific adapters: dictionary of {task_id: (gamma, beta)}
        self.task_adapters = nn.ModuleDict()
        self.active_task_id = 0

    def add_task(self, task_id: int):
        """Add a new task-specific adapter."""
        if str(task_id) not in self.task_adapters:
            adapter = nn.ModuleDict({
                'gamma': nn.Parameter(torch.ones(1, 1, 1, self.channels)),
                'beta': nn.Parameter(torch.zeros(1, 1, 1, self.channels))
            })
            self.task_adapters[str(task_id)] = adapter

    def set_active_task(self, task_id: int):
        """Set the active task for forward pass."""
        self.active_task_id = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply base mixer followed by task-specific adaptation.

        Args:
            x: (B, H, W, D) input features

        Returns:
            Task-adapted mixed features (B, H, W, D)
        """
        # Apply frozen base mixer
        x_mixed = self.base_mixer(x)

        # Apply task-specific adaptation if available
        task_key = str(self.active_task_id)
        if task_key in self.task_adapters:
            adapter = self.task_adapters[task_key]
            gamma = adapter['gamma']
            beta = adapter['beta']
            x_mixed = gamma * x_mixed + beta

        return x_mixed

    def get_trainable_params(self, task_id: int):
        """Get trainable parameters for a specific task."""
        task_key = str(task_id)
        if task_key in self.task_adapters:
            return self.task_adapters[task_key].parameters()
        return iter([])


class LightweightGlobalContext(nn.Module):
    """
    V5 Phase 3: Lightweight Global Context Module.

    Problem: Current architecture only sees local context (3x3 or 5x5),
    missing global patterns needed for structural defects (e.g., transistor
    with misaligned legs).

    Solution: Efficient global context aggregation without full self-attention.

    Key Design:
    1. Divide spatial map into N regions (e.g., 4x4 = 16 regions)
    2. Compute region-level representations (average pooling)
    3. Simple cross-attention: each patch queries all regions
    4. Add global context as residual

    This captures:
    - "Does this patch differ from the overall pattern?"
    - "Are there structural inconsistencies across regions?"

    Complexity: O(HW * N) instead of O((HW)^2) for full self-attention
    """

    def __init__(self, channels: int, num_regions: int = 4, reduction: int = 4):
        super(LightweightGlobalContext, self).__init__()

        self.channels = channels
        self.num_regions = num_regions  # Regions per dimension (total = num_regions^2)
        self.reduction = reduction

        reduced_dim = channels // reduction

        # Query projection for patches
        self.query_proj = nn.Linear(channels, reduced_dim)

        # Key projection for regions
        self.key_proj = nn.Linear(channels, reduced_dim)

        # Value projection for regions
        self.value_proj = nn.Linear(channels, reduced_dim)

        # Output projection
        self.out_proj = nn.Linear(reduced_dim, channels)

        # Learnable gate for global context influence
        self.context_gate = nn.Parameter(torch.tensor([0.2]))

        # Initialize output projection small
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add global context to spatial features.

        Args:
            x: (B, H, W, D) spatial features

        Returns:
            Features with global context (B, H, W, D)
        """
        B, H, W, D = x.shape
        identity = x

        # Step 1: Compute region representations via adaptive pooling
        # Reshape to (B, D, H, W) for pooling
        x_spatial = x.permute(0, 3, 1, 2)  # (B, D, H, W)

        # Adaptive pooling to (num_regions, num_regions)
        region_size = self.num_regions
        x_regions = F.adaptive_avg_pool2d(x_spatial, (region_size, region_size))  # (B, D, R, R)
        x_regions = x_regions.permute(0, 2, 3, 1)  # (B, R, R, D)
        x_regions = x_regions.reshape(B, -1, D)  # (B, R*R, D)
        num_regions = x_regions.shape[1]

        # Step 2: Project to reduced dimension
        x_flat = x.reshape(B, -1, D)  # (B, H*W, D)

        Q = self.query_proj(x_flat)  # (B, H*W, d)
        K = self.key_proj(x_regions)  # (B, R*R, d)
        V = self.value_proj(x_regions)  # (B, R*R, d)

        # Step 3: Compute attention
        d_k = Q.shape[-1]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)  # (B, H*W, R*R)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H*W, R*R)

        # Step 4: Aggregate values
        context = torch.bmm(attn_weights, V)  # (B, H*W, d)

        # Step 5: Project back and reshape
        context = self.out_proj(context)  # (B, H*W, D)
        context = context.reshape(B, H, W, D)

        # Step 6: Residual with gate
        gate = torch.sigmoid(self.context_gate)
        x_out = identity + gate * context

        return x_out

    def get_region_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.

        Args:
            x: (B, H, W, D) spatial features

        Returns:
            Attention weights (B, H*W, R*R)
        """
        B, H, W, D = x.shape

        x_spatial = x.permute(0, 3, 1, 2)
        region_size = self.num_regions
        x_regions = F.adaptive_avg_pool2d(x_spatial, (region_size, region_size))
        x_regions = x_regions.permute(0, 2, 3, 1).reshape(B, -1, D)

        x_flat = x.reshape(B, -1, D)

        Q = self.query_proj(x_flat)
        K = self.key_proj(x_regions)

        d_k = Q.shape[-1]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        return attn_weights


# =============================================================================
# V5.5: Position-Agnostic Improvements for Rotation-Invariant Classes
# =============================================================================

class RelativePositionEmbedding(nn.Module):
    """
    V5.5 Direction 1: Relative Position Encoding.

    Problem: Absolute positional embeddings encode "pattern at position (x,y)"
    which breaks when objects have geometric variance (rotation, translation).

    Solution: Encode relative positions between patches instead of absolute positions.
    - Each patch learns its relationship with neighbors
    - Rotation-invariant: "patch A is similar to its right neighbor" is preserved after rotation

    Key Design:
    1. Compute pairwise relative distances in a local window
    2. Learn embedding based on relative offset, not absolute position
    3. Aggregate neighbor information with learned weights
    """

    def __init__(self, channels: int, max_relative_distance: int = 7, num_heads: int = 4):
        super(RelativePositionEmbedding, self).__init__()

        self.channels = channels
        self.max_dist = max_relative_distance
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Learnable relative position bias table
        # Range: [-max_dist, max_dist] for both x and y
        table_size = 2 * max_relative_distance + 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size * table_size, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Project to get query, key for relative attention
        self.query_proj = nn.Linear(channels, channels)
        self.key_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        # Learnable gate for blending
        self.blend_gate = nn.Parameter(torch.tensor([0.3]))

    def _get_relative_position_index(self, H: int, W: int, device):
        """Compute relative position index for HxW grid."""
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, H, W)
        coords_flatten = coords.reshape(2, -1)  # (2, H*W)

        # Relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, H*W, H*W)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (H*W, H*W, 2)

        # Shift to start from 0 and clamp to max distance
        relative_coords[:, :, 0] = torch.clamp(relative_coords[:, :, 0] + self.max_dist, 0, 2 * self.max_dist)
        relative_coords[:, :, 1] = torch.clamp(relative_coords[:, :, 1] + self.max_dist, 0, 2 * self.max_dist)

        # Convert to 1D index
        relative_position_index = relative_coords[:, :, 0] * (2 * self.max_dist + 1) + relative_coords[:, :, 1]

        return relative_position_index.long()

    def forward(self, x: torch.Tensor, absolute_pe: torch.Tensor = None) -> torch.Tensor:
        """
        Apply relative position encoding.

        Args:
            x: (B, H, W, D) input features (without absolute PE)
            absolute_pe: (B, H, W, D) absolute positional embedding (optional, for blending)

        Returns:
            Features with relative position encoding (B, H, W, D)
        """
        B, H, W, D = x.shape
        N = H * W

        # Flatten spatial dimensions
        x_flat = x.reshape(B, N, D)  # (B, N, D)

        # Compute Q, K
        Q = self.query_proj(x_flat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key_proj(x_flat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, N, N)

        # Add relative position bias
        relative_position_index = self._get_relative_position_index(H, W, x.device)  # (N, N)
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            N, N, self.num_heads
        )  # (N, N, heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)
        attn = attn + relative_position_bias

        # Softmax and aggregate
        attn = F.softmax(attn, dim=-1)
        V = x_flat.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = torch.matmul(attn, V)  # (B, heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # (B, N, D)
        out = self.out_proj(out)

        # Reshape back
        out = out.reshape(B, H, W, D)

        # Blend with original or absolute PE
        gate = torch.sigmoid(self.blend_gate)
        if absolute_pe is not None:
            # Blend: (1-gate)*absolute + gate*relative
            result = (1 - gate) * (x + absolute_pe) + gate * (x + out)
        else:
            result = x + gate * out

        return result


class DualBranchScorer(nn.Module):
    """
    V5.5 Direction 2: Position-Agnostic Score Branch.

    Problem: Single NF with positional info fails on rotation-variant classes.

    Solution: Two parallel scoring branches:
    1. Position Branch: Standard NF with positional embedding (good for aligned objects)
    2. No-Position Branch: NF without positional embedding (good for rotated objects)

    Final score = α * pos_score + (1-α) * nopos_score
    where α is learned per-patch based on local pattern consistency.

    This allows the model to automatically rely less on position for patches
    where positional information hurts (e.g., rotated thread patterns).
    """

    def __init__(self, channels: int):
        super(DualBranchScorer, self).__init__()

        self.channels = channels

        # Alpha predictor: learns when to trust position vs no-position
        # Input: concatenation of pos and nopos features
        self.alpha_net = nn.Sequential(
            nn.Linear(channels * 2, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
            nn.Sigmoid()
        )

        # Initialize to balanced (α ≈ 0.5)
        nn.init.zeros_(self.alpha_net[-2].weight)
        nn.init.zeros_(self.alpha_net[-2].bias)

    def forward(self, z_pos: torch.Tensor, z_nopos: torch.Tensor,
                score_pos: torch.Tensor, score_nopos: torch.Tensor) -> torch.Tensor:
        """
        Combine scores from position and no-position branches.

        Args:
            z_pos: (B, H, W, D) latent from position branch
            z_nopos: (B, H, W, D) latent from no-position branch
            score_pos: (B, H, W) patch scores from position branch
            score_nopos: (B, H, W) patch scores from no-position branch

        Returns:
            Combined patch scores (B, H, W)
        """
        B, H, W, D = z_pos.shape

        # Concatenate latents for alpha prediction
        z_concat = torch.cat([z_pos, z_nopos], dim=-1)  # (B, H, W, 2D)

        # Predict per-patch alpha
        alpha = self.alpha_net(z_concat).squeeze(-1)  # (B, H, W)

        # Weighted combination
        combined_score = alpha * score_pos + (1 - alpha) * score_nopos

        return combined_score

    def get_alpha_map(self, z_pos: torch.Tensor, z_nopos: torch.Tensor) -> torch.Tensor:
        """Get alpha map for visualization."""
        z_concat = torch.cat([z_pos, z_nopos], dim=-1)
        alpha = self.alpha_net(z_concat).squeeze(-1)
        return alpha


class LocalConsistencyCalibrator(nn.Module):
    """
    V5.5 Direction 3: Local Consistency Score.

    Problem: False positive patches (normal patches with high scores due to rotation)
    create noise that dominates image-level aggregation.

    Key Insight:
    - Real defects: High score AND neighbors also have high scores (consistent)
    - Rotation noise: High score BUT neighbors have low scores (inconsistent)

    Solution: Calibrate patch scores based on local consistency.
    - Compute local score variance in a window
    - Down-weight patches with high inconsistency (likely rotation noise)

    calibrated_score = raw_score * consistency_weight

    This is different from SpatialCluster:
    - SpatialCluster: Binary clustering check
    - LocalConsistency: Continuous consistency measurement with learned calibration
    """

    def __init__(self, kernel_size: int = 3, temperature: float = 1.0):
        super(LocalConsistencyCalibrator, self).__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Learnable temperature for consistency weighting
        self.temperature = nn.Parameter(torch.tensor([temperature]))

        # Learnable bias for minimum weight (don't completely suppress)
        self.min_weight = nn.Parameter(torch.tensor([0.3]))

        # Optional: learned convolution for more flexible consistency computation
        self.use_learned_kernel = True
        if self.use_learned_kernel:
            self.consistency_conv = nn.Conv2d(1, 1, kernel_size, padding=self.padding, bias=False)
            # Initialize as averaging kernel
            nn.init.constant_(self.consistency_conv.weight, 1.0 / (kernel_size * kernel_size))

    def forward(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """
        Calibrate patch scores based on local consistency.

        Args:
            patch_scores: (B, H, W) raw patch anomaly scores

        Returns:
            Calibrated patch scores (B, H, W)
        """
        B, H, W = patch_scores.shape

        # Reshape for conv2d: (B, 1, H, W)
        scores = patch_scores.unsqueeze(1)

        # Compute local mean
        if self.use_learned_kernel:
            local_mean = self.consistency_conv(scores)
        else:
            kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=scores.device)
            kernel = kernel / (self.kernel_size * self.kernel_size)
            local_mean = F.conv2d(scores, kernel, padding=self.padding)

        # Compute local variance (measure of inconsistency)
        local_var = F.conv2d(
            (scores - local_mean) ** 2,
            torch.ones(1, 1, self.kernel_size, self.kernel_size, device=scores.device) / (self.kernel_size ** 2),
            padding=self.padding
        )

        # Normalize variance by score magnitude (relative consistency)
        # Add small epsilon to avoid division by zero
        relative_var = local_var / (scores ** 2 + 1e-6)

        # Consistency weight: high when variance is low
        # weight = sigmoid(-temperature * relative_var)
        temperature = F.softplus(self.temperature)  # Ensure positive
        consistency_weight = torch.sigmoid(-temperature * relative_var.squeeze(1))

        # Apply minimum weight floor
        min_w = torch.sigmoid(self.min_weight)
        consistency_weight = min_w + (1 - min_w) * consistency_weight

        # Calibrate scores
        calibrated_scores = patch_scores * consistency_weight

        return calibrated_scores

    def get_consistency_map(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """Get consistency weight map for visualization."""
        B, H, W = patch_scores.shape
        scores = patch_scores.unsqueeze(1)

        if self.use_learned_kernel:
            local_mean = self.consistency_conv(scores)
        else:
            kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=scores.device)
            kernel = kernel / (self.kernel_size * self.kernel_size)
            local_mean = F.conv2d(scores, kernel, padding=self.padding)

        local_var = F.conv2d(
            (scores - local_mean) ** 2,
            torch.ones(1, 1, self.kernel_size, self.kernel_size, device=scores.device) / (self.kernel_size ** 2),
            padding=self.padding
        )

        relative_var = local_var / (scores ** 2 + 1e-6)
        temperature = F.softplus(self.temperature)
        consistency_weight = torch.sigmoid(-temperature * relative_var.squeeze(1))

        min_w = torch.sigmoid(self.min_weight)
        consistency_weight = min_w + (1 - min_w) * consistency_weight

        return consistency_weight


# =============================================================================
# V5.6 Improved Modules
# =============================================================================

class ImprovedDualBranchScorer(nn.Module):
    """
    V5.6 Improved Direction 2: Dual Branch with Anti-Collapse Mechanism.

    V5.5 실패 원인:
    - α가 학습 초기에 빠르게 0으로 수렴 (no-pos 브랜치가 loss 더 낮음)
    - 일단 α→0이 되면 pos 브랜치 gradient 소실
    - 결과: no-pos만 사용하여 고정 방향 클래스 붕괴

    V5.6 개선:
    1. α 초기값을 0.7로 설정 (pos 브랜치 선호 시작)
    2. min_alpha=0.3 제약으로 pos 브랜치 최소 사용 보장
    3. Score 차이를 추가 입력으로 활용 (informative signal)
    4. α divergence regularization loss 반환
    """

    def __init__(self, channels: int, init_alpha: float = 0.7,
                 min_alpha: float = 0.3, max_alpha: float = 0.9):
        super(ImprovedDualBranchScorer, self).__init__()

        self.channels = channels
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        # Alpha predictor with score difference as additional input
        # Input: z_pos, z_nopos, score_diff (normalized)
        self.alpha_net = nn.Sequential(
            nn.Linear(channels * 2 + 1, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 1),
        )

        # Initialize bias to achieve init_alpha after sigmoid and clamping
        # sigmoid^{-1}(0.7) ≈ 0.847
        init_logit = math.log(init_alpha / (1 - init_alpha))
        nn.init.zeros_(self.alpha_net[-1].weight)
        nn.init.constant_(self.alpha_net[-1].bias, init_logit)

        # Regularization strength
        self.reg_strength = 0.1

    def forward(self, z_pos: torch.Tensor, z_nopos: torch.Tensor,
                score_pos: torch.Tensor, score_nopos: torch.Tensor) -> torch.Tensor:
        """
        Combine scores with anti-collapse α prediction.

        Returns:
            Combined patch scores (B, H, W)
        """
        B, H, W, D = z_pos.shape

        # Compute normalized score difference as additional signal
        score_diff = (score_pos - score_nopos) / (score_pos.abs() + score_nopos.abs() + 1e-6)
        score_diff = score_diff.unsqueeze(-1)  # (B, H, W, 1)

        # Concatenate all inputs
        combined_input = torch.cat([z_pos, z_nopos, score_diff], dim=-1)  # (B, H, W, 2D+1)

        # Predict raw alpha logits
        alpha_logit = self.alpha_net(combined_input).squeeze(-1)  # (B, H, W)

        # Apply sigmoid and clamp to [min_alpha, max_alpha]
        alpha_raw = torch.sigmoid(alpha_logit)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * alpha_raw

        # Store for regularization
        self._last_alpha = alpha

        # Weighted combination
        combined_score = alpha * score_pos + (1 - alpha) * score_nopos

        return combined_score

    def get_alpha_stats(self) -> dict:
        """Get alpha statistics for monitoring."""
        if hasattr(self, '_last_alpha') and self._last_alpha is not None:
            alpha = self._last_alpha
            return {
                'alpha_mean': alpha.mean().item(),
                'alpha_std': alpha.std().item(),
                'alpha_min': alpha.min().item(),
                'alpha_max': alpha.max().item(),
            }
        return {}

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Regularization to prevent α collapse.
        Encourages α to stay near 0.5 (balanced use of both branches).
        """
        if hasattr(self, '_last_alpha') and self._last_alpha is not None:
            alpha = self._last_alpha
            # Encourage variance (don't want all same α)
            # And encourage mean near 0.5
            target = (self.min_alpha + self.max_alpha) / 2
            mean_reg = (alpha.mean() - target) ** 2
            # Also encourage some variance (adaptive α)
            var_reg = -alpha.var().clamp(min=1e-6).log()
            return self.reg_strength * (mean_reg + 0.1 * var_reg)
        return torch.tensor(0.0)


class MultiScaleLocalConsistency(nn.Module):
    """
    V5.6 Improved Direction 3: Multi-Scale Local Consistency.

    V5.5 분석:
    - 단일 3x3 커널은 제한적
    - 결함 크기에 따라 다른 스케일 필요

    V5.6 개선:
    1. Multi-scale consistency (3x3, 5x5, 7x7)
    2. 스케일별 learnable weight
    3. Score-aware adaptive weighting
    """

    def __init__(self, kernel_sizes: list = [3, 5, 7], temperature: float = 1.0):
        super(MultiScaleLocalConsistency, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.n_scales = len(kernel_sizes)

        # Per-scale learnable parameters
        self.temperatures = nn.ParameterList([
            nn.Parameter(torch.tensor([temperature])) for _ in kernel_sizes
        ])

        # Per-scale minimum weights
        self.min_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([0.3])) for _ in kernel_sizes
        ])

        # Scale fusion weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(self.n_scales) / self.n_scales)

        # Optional: score-adaptive scale selection
        self.use_adaptive_fusion = True
        if self.use_adaptive_fusion:
            self.adaptive_net = nn.Sequential(
                nn.Linear(self.n_scales, self.n_scales),
                nn.Softmax(dim=-1)
            )

    def _compute_consistency_at_scale(self, scores: torch.Tensor, kernel_size: int,
                                       temperature: nn.Parameter, min_weight: nn.Parameter) -> torch.Tensor:
        """Compute consistency weight at a single scale."""
        padding = kernel_size // 2
        B, _, H, W = scores.shape

        # Local mean
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=scores.device)
        kernel = kernel / (kernel_size * kernel_size)
        local_mean = F.conv2d(scores, kernel, padding=padding)

        # Local variance
        local_var = F.conv2d(
            (scores - local_mean) ** 2,
            kernel,
            padding=padding
        )

        # Relative variance
        relative_var = local_var / (scores ** 2 + 1e-6)

        # Consistency weight
        temp = F.softplus(temperature)
        consistency_weight = torch.sigmoid(-temp * relative_var)

        # Apply minimum weight
        min_w = torch.sigmoid(min_weight)
        consistency_weight = min_w + (1 - min_w) * consistency_weight

        return consistency_weight.squeeze(1)  # (B, H, W)

    def forward(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale consistency calibration.

        Args:
            patch_scores: (B, H, W) raw patch anomaly scores

        Returns:
            Calibrated patch scores (B, H, W)
        """
        B, H, W = patch_scores.shape
        scores = patch_scores.unsqueeze(1)  # (B, 1, H, W)

        # Compute consistency at each scale
        scale_weights_list = []
        for i, kernel_size in enumerate(self.kernel_sizes):
            weight = self._compute_consistency_at_scale(
                scores, kernel_size, self.temperatures[i], self.min_weights[i]
            )
            scale_weights_list.append(weight)

        # Stack: (B, H, W, n_scales)
        all_weights = torch.stack(scale_weights_list, dim=-1)

        if self.use_adaptive_fusion:
            # Adaptive fusion based on local score statistics
            # Use mean score at each scale as input
            scale_means = all_weights.mean(dim=(1, 2))  # (B, n_scales)
            fusion_weights = self.adaptive_net(scale_means)  # (B, n_scales)
            fusion_weights = fusion_weights.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, n_scales)
        else:
            # Simple learnable fusion
            fusion_weights = F.softmax(self.scale_weights, dim=0)
            fusion_weights = fusion_weights.view(1, 1, 1, -1)

        # Weighted combination of scales
        combined_weight = (all_weights * fusion_weights).sum(dim=-1)  # (B, H, W)

        # Calibrate scores
        calibrated_scores = patch_scores * combined_weight

        return calibrated_scores

    def get_scale_weights(self) -> torch.Tensor:
        """Get current scale fusion weights for monitoring."""
        return F.softmax(self.scale_weights, dim=0)


class ScoreGuidedDualBranch(nn.Module):
    """
    V5.6 Alternative Direction 2: Score-Guided Branch Selection.

    다른 접근: latent 대신 score 통계로 브랜치 선택.

    아이디어:
    - pos_score >> nopos_score: pos 브랜치 신뢰 (고정 방향 클래스)
    - pos_score << nopos_score: nopos 브랜치 신뢰 (회전 클래스)
    - 비슷하면: 평균 사용

    장점:
    - 더 interpretable
    - Gradient가 더 직접적
    """

    def __init__(self, temperature: float = 1.0, min_alpha: float = 0.2):
        super(ScoreGuidedDualBranch, self).__init__()

        self.temperature = nn.Parameter(torch.tensor([temperature]))
        self.min_alpha = min_alpha

        # Bias term to shift default preference
        self.bias = nn.Parameter(torch.tensor([0.5]))  # Slightly prefer pos

    def forward(self, z_pos: torch.Tensor, z_nopos: torch.Tensor,
                score_pos: torch.Tensor, score_nopos: torch.Tensor) -> torch.Tensor:
        """
        Score-guided branch combination.

        Logic:
        - When score_pos < score_nopos: prefer pos (lower score = more normal)
        - When score_pos > score_nopos: prefer nopos
        """
        # Score difference (positive when pos is worse)
        score_diff = score_pos - score_nopos

        # Normalize by score magnitude
        score_magnitude = (score_pos.abs() + score_nopos.abs()) / 2 + 1e-6
        normalized_diff = score_diff / score_magnitude

        # α = sigmoid(temperature * (bias - normalized_diff))
        # When pos is worse (diff > 0), α decreases (use more nopos)
        # When pos is better (diff < 0), α increases (use more pos)
        temp = F.softplus(self.temperature)
        alpha = torch.sigmoid(temp * (self.bias - normalized_diff))

        # Clamp to ensure minimum contribution from both
        alpha = alpha.clamp(min=self.min_alpha, max=1 - self.min_alpha)

        # Store for monitoring
        self._last_alpha = alpha

        # Weighted combination
        combined_score = alpha * score_pos + (1 - alpha) * score_nopos

        return combined_score

    def get_alpha_stats(self) -> dict:
        """Get alpha statistics for monitoring."""
        if hasattr(self, '_last_alpha') and self._last_alpha is not None:
            alpha = self._last_alpha
            return {
                'alpha_mean': alpha.mean().item(),
                'alpha_std': alpha.std().item(),
            }
        return {}


# =============================================================================
# V5.7 Rotation-Invariant Position Encoding
# =============================================================================

class MultiOrientationEnsemble(nn.Module):
    """
    V5.7 Direction C: Multi-Orientation Ensemble.

    핵심 아이디어:
    - Test time에 여러 회전 방향에서 NF 적용
    - 가장 낮은 anomaly score 선택 (가장 "정상"인 방향)

    직관:
    - 정상 이미지: 최소 1개 방향에서 낮은 score
    - 비정상 이미지: 모든 방향에서 높은 score

    장점:
    - 학습 변경 없음 (inference만 수정)
    - 완벽한 rotation invariance 보장

    단점:
    - N배 inference 비용 (N = 회전 개수)
    """

    def __init__(self, n_orientations: int = 4):
        super(MultiOrientationEnsemble, self).__init__()
        self.n_orientations = n_orientations
        # 회전 각도들 (도 단위)
        self.angles = [i * (360 // n_orientations) for i in range(n_orientations)]

    def rotate_features(self, features: torch.Tensor, angle: int) -> torch.Tensor:
        """
        Feature map을 회전.

        Args:
            features: (B, H, W, D)
            angle: 회전 각도 (0, 90, 180, 270)

        Returns:
            Rotated features (B, H, W, D)
        """
        if angle == 0:
            return features

        # (B, H, W, D) -> (B, D, H, W) for torch rotation
        features_permuted = features.permute(0, 3, 1, 2)

        # k = number of 90-degree rotations
        k = angle // 90
        rotated = torch.rot90(features_permuted, k=k, dims=(2, 3))

        # Back to (B, H, W, D)
        return rotated.permute(0, 2, 3, 1)

    def inverse_rotate_scores(self, scores: torch.Tensor, angle: int) -> torch.Tensor:
        """
        Score map을 역회전 (원래 방향으로).

        Args:
            scores: (B, H, W)
            angle: 원래 회전 각도

        Returns:
            Inverse rotated scores (B, H, W)
        """
        if angle == 0:
            return scores

        # 역회전: 360 - angle
        k = (360 - angle) // 90
        return torch.rot90(scores, k=k, dims=(1, 2))

    def get_orientations(self) -> List[int]:
        """Return list of rotation angles."""
        return self.angles


class ContentBasedPositionalEmbedding(nn.Module):
    """
    V5.7 Direction D: Content-Based Positional Embedding.

    핵심 아이디어:
    - 그리드 위치 대신 **의미적 위치** 인코딩
    - "이 패치는 (5,5)에 있다" → "이 패치는 '나사산 영역'에 있다"

    작동 방식:
    1. 학습 가능한 프로토타입 (의미적 앵커) 정의
    2. 각 패치가 어떤 프로토타입과 유사한지 계산
    3. 유사도 기반으로 위치 임베딩 생성

    장점:
    - 회전에 완전 불변 (내용 기반이므로)
    - Task 0에서 프로토타입 자동 학습

    예시:
    - Proto 0: 배경 텍스처
    - Proto 1: 엣지/경계
    - Proto 2: 나사산 패턴 → 회전해도 여전히 "나사산 패턴"!
    """

    def __init__(self, embed_dim: int, n_prototypes: int = 16,
                 temperature: float = 0.1, blend_with_grid: bool = True):
        super(ContentBasedPositionalEmbedding, self).__init__()

        self.embed_dim = embed_dim
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        self.blend_with_grid = blend_with_grid

        # 학습 가능한 의미적 프로토타입
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, embed_dim) * 0.02
        )

        # 프로토타입 유사도 → 위치 임베딩 변환
        self.position_proj = nn.Sequential(
            nn.Linear(n_prototypes, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Grid PE와 블렌딩 비율 (learnable)
        if blend_with_grid:
            self.blend_gate = nn.Parameter(torch.tensor([0.3]))  # 초기: 30% content, 70% grid

        # 초기화
        nn.init.xavier_uniform_(self.position_proj[0].weight)
        nn.init.xavier_uniform_(self.position_proj[3].weight)

    def forward(self, features: torch.Tensor,
                grid_pe: torch.Tensor = None) -> torch.Tensor:
        """
        Content-based positional embedding 적용.

        Args:
            features: (B, H, W, D) patch features (PE 적용 전)
            grid_pe: (B, H, W, D) optional standard grid PE

        Returns:
            features + position_embedding: (B, H, W, D)
        """
        B, H, W, D = features.shape

        # Flatten for batch processing
        feat_flat = features.reshape(B * H * W, D)

        # 정규화된 유사도 계산
        proto_norm = F.normalize(self.prototypes, dim=-1)
        feat_norm = F.normalize(feat_flat, dim=-1)

        # Cosine similarity with temperature scaling
        similarity = feat_norm @ proto_norm.T  # (BHW, n_proto)
        similarity = F.softmax(similarity / self.temperature, dim=-1)

        # 의미적 위치 임베딩 생성
        semantic_pos = self.position_proj(similarity)  # (BHW, D)
        semantic_pos = semantic_pos.reshape(B, H, W, D)

        # Grid PE와 블렌딩 (optional)
        if self.blend_with_grid and grid_pe is not None:
            alpha = torch.sigmoid(self.blend_gate)  # content PE 비율
            position = alpha * semantic_pos + (1 - alpha) * grid_pe
        else:
            position = semantic_pos

        return features + position

    def get_prototype_assignments(self, features: torch.Tensor) -> torch.Tensor:
        """시각화용: 각 패치의 프로토타입 할당."""
        B, H, W, D = features.shape
        feat_flat = features.reshape(B * H * W, D)

        proto_norm = F.normalize(self.prototypes, dim=-1)
        feat_norm = F.normalize(feat_flat, dim=-1)

        similarity = feat_norm @ proto_norm.T
        assignments = similarity.argmax(dim=-1)  # (BHW,)

        return assignments.reshape(B, H, W)


class HybridRotationInvariantPE(nn.Module):
    """
    V5.7 Direction E: Hybrid Approach.

    핵심 아이디어:
    - Content-Based PE + Grid PE를 패치별로 선택적 사용
    - 패치 내용에 따라 어떤 PE가 적합한지 학습

    직관:
    - 구조적 패턴 (나사산): Content PE 사용 → rotation invariant
    - 위치 의존 패턴 (코너): Grid PE 사용 → position aware

    작동 방식:
    1. 두 종류 PE 모두 계산
    2. Selector network가 패치별 가중치 결정
    3. 가중 평균으로 최종 PE 생성
    """

    def __init__(self, embed_dim: int, n_prototypes: int = 16,
                 temperature: float = 0.1):
        super(HybridRotationInvariantPE, self).__init__()

        self.embed_dim = embed_dim

        # Content-Based PE (rotation invariant)
        self.content_pe = ContentBasedPositionalEmbedding(
            embed_dim=embed_dim,
            n_prototypes=n_prototypes,
            temperature=temperature,
            blend_with_grid=False  # 블렌딩은 여기서 직접 처리
        )

        # Selector: 패치별로 content vs grid 선택
        # 입력: 패치 feature, 출력: content PE 가중치
        self.selector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

        # Selector 초기화: 약간 grid 선호 (기존 동작 유지)
        nn.init.zeros_(self.selector[-2].weight)
        nn.init.constant_(self.selector[-2].bias, -0.5)  # sigmoid(-0.5) ≈ 0.38

    def forward(self, features: torch.Tensor,
                grid_pe: torch.Tensor) -> torch.Tensor:
        """
        Hybrid PE 적용.

        Args:
            features: (B, H, W, D) patch features (PE 적용 전)
            grid_pe: (B, H, W, D) standard grid PE

        Returns:
            features + hybrid_position_embedding: (B, H, W, D)
        """
        B, H, W, D = features.shape

        # Content-based position (rotation invariant)
        # content_pe는 내부에서 features + pos를 반환하므로, pos만 추출
        content_pos = self.content_pe.position_proj(
            F.softmax(
                F.normalize(features.reshape(-1, D), dim=-1) @
                F.normalize(self.content_pe.prototypes, dim=-1).T /
                self.content_pe.temperature,
                dim=-1
            )
        ).reshape(B, H, W, D)

        # Selector: 패치별 content PE 가중치
        alpha = self.selector(features)  # (B, H, W, 1)

        # 저장 (모니터링용)
        self._last_alpha = alpha.squeeze(-1)

        # Hybrid PE
        hybrid_pos = alpha * content_pos + (1 - alpha) * grid_pe

        return features + hybrid_pos

    def get_alpha_map(self, features: torch.Tensor) -> torch.Tensor:
        """시각화용: content PE 가중치 맵."""
        return self.selector(features).squeeze(-1)

    def get_alpha_stats(self) -> dict:
        """Alpha 통계 (모니터링용)."""
        if hasattr(self, '_last_alpha') and self._last_alpha is not None:
            alpha = self._last_alpha
            return {
                'content_ratio_mean': alpha.mean().item(),
                'content_ratio_std': alpha.std().item(),
                'content_ratio_min': alpha.min().item(),
                'content_ratio_max': alpha.max().item(),
            }
        return {}


# =============================================================================
# V5.8 Task-Adaptive Position Encoding (TAPE)
# =============================================================================

class TaskAdaptivePositionEncoding(nn.Module):
    """
    V5.8: Task-Adaptive Position Encoding (TAPE).

    핵심 통찰:
    - Patch-level이 아닌 **Task-level**에서 PE 강도 결정
    - Inference가 아닌 **Training** 시 학습
    - NLL loss가 직접 gradient 제공 → 명확한 학습 신호

    작동 원리:
    - 각 Task마다 learnable gate (scalar) 보유
    - gate → sigmoid → PE 강도 (0~1)
    - NLL loss가 최적의 PE 강도로 수렴하도록 gradient 제공

    기대 효과:
    - Screw: PE 강도 → 0.1~0.3 (rotation variance 대응)
    - Leather: PE 강도 → 0.8~1.0 (spatial consistency 유지)
    - 자동으로 각 class에 최적화

    왜 이전 방법들이 실패했나:
    - V5.5/V5.6 Dual Branch: patch-level α 학습 시도
      → 정상 패치는 pos/nopos 둘 다 낮은 score → gradient 없음
    - V5.7 Multi-Orientation: inference 시 rotation ensemble
      → features에 이미 PE 포함되어 있어 의미있는 다른 시점 아님

    TAPE가 작동하는 이유:
    - Task 전체의 NLL loss로 gate 학습
    - PE가 도움되면 gate ↑, 방해되면 gate ↓
    - 명확한 gradient 존재
    """

    def __init__(self, init_value: float = 0.0):
        """
        Args:
            init_value: gate 초기값 (sigmoid 전)
                - 0.0: sigmoid(0) = 0.5 (50% PE로 시작)
                - 2.0: sigmoid(2) ≈ 0.88 (기존 동작에 가깝게)
                - -2.0: sigmoid(-2) ≈ 0.12 (minimal PE)
        """
        super(TaskAdaptivePositionEncoding, self).__init__()
        self.init_value = init_value
        self.pe_gates = nn.ParameterDict()  # {task_id: gate}

    def add_task(self, task_id: int, device: str = 'cuda'):
        """새 Task 추가 시 해당 Task의 PE gate 생성."""
        task_key = str(task_id)
        if task_key not in self.pe_gates:
            # 초기값으로 gate 생성 (지정된 device에)
            self.pe_gates[task_key] = nn.Parameter(
                torch.tensor([self.init_value], device=device)
            )

    def forward(self, features: torch.Tensor, grid_pe: torch.Tensor,
                task_id: int) -> torch.Tensor:
        """
        Task-adaptive PE 적용.

        Args:
            features: (B, H, W, D) raw features (PE 적용 전)
            grid_pe: (B, H, W, D) standard grid positional encoding
            task_id: 현재 task ID

        Returns:
            features_with_pe: (B, H, W, D) = features + alpha * grid_pe
        """
        task_key = str(task_id)
        device = features.device

        # Ensure grid_pe is on the same device as features
        grid_pe = grid_pe.to(device)

        if task_key not in self.pe_gates:
            # Task가 등록되지 않은 경우 기본값 사용
            alpha = torch.tensor(0.5, device=device)
        else:
            gate = self.pe_gates[task_key]
            # Gate should already be on correct device from add_task
            alpha = torch.sigmoid(gate)

        # PE 강도 조절
        features_with_pe = features + alpha * grid_pe

        return features_with_pe

    def get_pe_strength(self, task_id: int) -> float:
        """특정 Task의 학습된 PE 강도 반환."""
        task_key = str(task_id)
        if task_key in self.pe_gates:
            return torch.sigmoid(self.pe_gates[task_key]).item()
        return 0.5  # 기본값

    def get_all_pe_strengths(self) -> dict:
        """모든 Task의 PE 강도 반환."""
        strengths = {}
        for task_key, gate in self.pe_gates.items():
            strengths[int(task_key)] = torch.sigmoid(gate).item()
        return strengths

    def get_trainable_params(self, task_id: int) -> List[nn.Parameter]:
        """특정 Task의 trainable parameters 반환."""
        task_key = str(task_id)
        if task_key in self.pe_gates:
            return [self.pe_gates[task_key]]
        return []

    def __repr__(self):
        strengths = self.get_all_pe_strengths()
        strength_str = ", ".join([f"T{k}={v:.3f}" for k, v in strengths.items()])
        return f"TaskAdaptivePositionEncoding(init={self.init_value}, strengths=[{strength_str}])"


# =============================================================================
# V6.1: Spatial Transformer Network (STN)
# =============================================================================

class SpatialTransformerNetwork(nn.Module):
    """
    V6.1: Spatial Transformer Network for automatic image alignment.

    Learns to align input images to a canonical orientation, solving the
    rotation variance problem in classes like Screw.

    Architecture:
        1. Localization Network: Predicts transformation parameters (θ)
        2. Grid Generator: Creates sampling grid from θ
        3. Sampler: Applies bilinear interpolation to transform image

    Modes:
        - 'rotation': Only learns rotation angle (1 parameter)
        - 'rotation_scale': Rotation + uniform scale (2 parameters)
        - 'affine': Full 6-parameter affine transformation

    Key Design:
        - Initialized to identity transformation (no change initially)
        - Learnable end-to-end with the rest of the model
        - Applied BEFORE feature extraction for spatial alignment
    """

    def __init__(
        self,
        input_channels: int = 3,
        input_size: int = 224,
        mode: str = 'rotation',
        hidden_dim: int = 128,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        rotation_reg_weight: float = 0.01,
    ):
        """
        Args:
            input_channels: Number of input image channels (3 for RGB)
            input_size: Input image size (assumed square)
            mode: Transformation mode ('rotation', 'rotation_scale', 'affine')
            hidden_dim: Hidden dimension for localization network
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate for regularization
            scale_range: Min/max scale for 'rotation_scale' mode
            rotation_reg_weight: Regularization weight for rotation angle
        """
        super(SpatialTransformerNetwork, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.scale_range = scale_range
        self.rotation_reg_weight = rotation_reg_weight

        # Determine number of output parameters based on mode
        if mode == 'rotation':
            self.num_params = 1  # theta (rotation angle)
        elif mode == 'rotation_scale':
            self.num_params = 2  # theta, scale
        elif mode == 'affine':
            self.num_params = 6  # full affine matrix elements
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'rotation', 'rotation_scale', or 'affine'")

        # Localization Network: CNN to predict transformation parameters
        self.localization = self._build_localization_network(
            input_channels, hidden_dim, use_batch_norm, dropout
        )

        # Final fully connected layer to output transformation parameters
        # Calculate feature size after conv layers
        self._dummy_input = torch.zeros(1, input_channels, input_size, input_size)
        with torch.no_grad():
            dummy_features = self.localization(self._dummy_input)
            self.feature_size = dummy_features.view(1, -1).size(1)
        del self._dummy_input

        self.fc_loc = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_params)
        )

        # Initialize to identity transformation
        self._init_identity()

        # For logging/debugging
        self.last_theta = None
        self.last_rotation_angle = None

    def _build_localization_network(
        self,
        in_channels: int,
        hidden_dim: int,
        use_bn: bool,
        dropout: float
    ) -> nn.Sequential:
        """Build CNN for localization (predicting transformation params)."""
        layers = []

        # Downsample progressively: input_size -> input_size/32
        channels = [in_channels, 32, 64, 128, hidden_dim]

        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU(True))
            if i < len(channels) - 2:  # No dropout on last conv
                layers.append(nn.Dropout2d(dropout))

        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())

        return nn.Sequential(*layers)

    def _init_identity(self):
        """Initialize to identity transformation (no change)."""
        # Initialize final layer bias to produce identity transformation
        nn.init.zeros_(self.fc_loc[-1].weight)

        if self.mode == 'rotation':
            # theta = 0 (no rotation)
            nn.init.zeros_(self.fc_loc[-1].bias)
        elif self.mode == 'rotation_scale':
            # theta = 0, scale = 1.0
            # We'll use sigmoid for scale, so init to 0 gives 0.5
            # We need to adjust in forward to map to scale_range
            nn.init.zeros_(self.fc_loc[-1].bias)
        elif self.mode == 'affine':
            # Identity affine: [[1, 0, 0], [0, 1, 0]]
            # Bias for [a, b, tx, c, d, ty] where identity is [1, 0, 0, 0, 1, 0]
            self.fc_loc[-1].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )

    def _params_to_theta(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted parameters to 2x3 affine transformation matrix.

        Args:
            params: (B, num_params) predicted transformation parameters

        Returns:
            theta: (B, 2, 3) affine transformation matrix
        """
        B = params.size(0)
        device = params.device

        if self.mode == 'rotation':
            # params: (B, 1) -> rotation angle in radians
            # Limit rotation range with tanh: [-pi, pi]
            angle = torch.tanh(params[:, 0]) * math.pi

            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            # Rotation matrix (around center)
            # [[cos, -sin, 0], [sin, cos, 0]]
            theta = torch.zeros(B, 2, 3, device=device)
            theta[:, 0, 0] = cos_a
            theta[:, 0, 1] = -sin_a
            theta[:, 1, 0] = sin_a
            theta[:, 1, 1] = cos_a

            self.last_rotation_angle = angle.detach()

        elif self.mode == 'rotation_scale':
            # params: (B, 2) -> rotation angle, scale
            angle = torch.tanh(params[:, 0]) * math.pi

            # Scale: map sigmoid output to scale_range
            scale_raw = torch.sigmoid(params[:, 1])
            scale_min, scale_max = self.scale_range
            scale = scale_min + (scale_max - scale_min) * scale_raw

            cos_a = torch.cos(angle) * scale
            sin_a = torch.sin(angle) * scale

            theta = torch.zeros(B, 2, 3, device=device)
            theta[:, 0, 0] = cos_a
            theta[:, 0, 1] = -sin_a
            theta[:, 1, 0] = sin_a
            theta[:, 1, 1] = cos_a

            self.last_rotation_angle = angle.detach()

        elif self.mode == 'affine':
            # params: (B, 6) -> [a, b, tx, c, d, ty]
            # Full affine: [[a, b, tx], [c, d, ty]]
            theta = params.view(B, 2, 3)

            # Extract approximate rotation angle for logging
            # angle ≈ atan2(c, a) for small shear
            self.last_rotation_angle = torch.atan2(
                theta[:, 1, 0], theta[:, 0, 0]
            ).detach()

        self.last_theta = theta.detach()
        return theta

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial transformation to input images.

        Args:
            x: (B, C, H, W) input images

        Returns:
            x_transformed: (B, C, H, W) transformed images
            theta: (B, 2, 3) transformation matrix (for logging/regularization)
        """
        B, C, H, W = x.shape

        # 1. Localization: predict transformation parameters
        features = self.localization(x)
        params = self.fc_loc(features)  # (B, num_params)

        # 2. Convert to affine transformation matrix
        theta = self._params_to_theta(params)  # (B, 2, 3)

        # 3. Generate sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        # 4. Sample from input using bilinear interpolation
        # padding_mode='border' avoids black borders by extending edge pixels
        x_transformed = F.grid_sample(
            x, grid,
            mode='bilinear',
            padding_mode='border',  # 'zeros', 'border', or 'reflection'
            align_corners=False
        )

        return x_transformed, theta

    def get_rotation_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to encourage small rotations.

        Prevents the STN from learning extreme rotations that might
        destabilize training.

        Returns:
            reg_loss: Scalar regularization loss
        """
        if self.last_rotation_angle is None:
            return torch.tensor(0.0)

        # L2 penalty on rotation angle
        # Encourages staying close to identity (0 rotation)
        reg_loss = self.rotation_reg_weight * (self.last_rotation_angle ** 2).mean()
        return reg_loss

    def get_transform_stats(self) -> dict:
        """Get statistics about the learned transformation for logging."""
        stats = {}

        if self.last_rotation_angle is not None:
            angles_deg = self.last_rotation_angle * 180 / math.pi
            stats['rotation_mean_deg'] = angles_deg.mean().item()
            stats['rotation_std_deg'] = angles_deg.std().item()
            stats['rotation_min_deg'] = angles_deg.min().item()
            stats['rotation_max_deg'] = angles_deg.max().item()

        return stats

    def __repr__(self):
        return (f"SpatialTransformerNetwork(mode={self.mode}, "
                f"input_size={self.input_size}, num_params={self.num_params})")


# =============================================================================
# Feature-Level Adaptation Baselines (for Coupling vs Feature-level comparison)
# =============================================================================

class FeatureLevelPromptAdapter(nn.Module):
    """
    Feature-level Prompt Adapter for baseline comparison.

    This implements the feature-level adaptation approach where learnable
    prompts/adapters are applied BEFORE the NF, modifying the input distribution.

    Equation: x' = x + P_t where P_t is a learnable prompt

    Used to compare against DeCoFlow's coupling-level LoRA adaptation.
    """

    def __init__(self, channels: int, spatial_size: int = 14,
                 target_params: int = 1_930_000):
        """
        Args:
            channels: Feature dimension (e.g., 1024 for WideResNet50)
            spatial_size: Spatial dimension (H=W, e.g., 14 for 224x224 input)
            target_params: Target number of parameters to match LoRA baseline
        """
        super(FeatureLevelPromptAdapter, self).__init__()

        self.channels = channels
        self.spatial_size = spatial_size

        # Calculate hidden dimension to match target parameters
        # Params = channels * hidden + hidden * channels = 2 * channels * hidden
        hidden_dim = target_params // (2 * channels)
        self.hidden_dim = hidden_dim

        # MLP-based adapter (Bottleneck structure)
        self.adapter = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )

        # Initialize to near-identity
        nn.init.normal_(self.adapter[0].weight, std=0.01)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

        # Learnable spatial prompt (optional, adds spatial bias)
        # This captures position-dependent adaptation
        self.use_spatial_prompt = True
        if self.use_spatial_prompt:
            self.spatial_prompt = nn.Parameter(
                torch.zeros(1, spatial_size, spatial_size, channels)
            )

        actual_params = sum(p.numel() for p in self.parameters())
        print(f"📦 FeatureLevelPromptAdapter: {actual_params:,} params "
              f"(target: {target_params:,}, hidden_dim: {hidden_dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature-level adaptation.

        Args:
            x: (B, H, W, D) input features

        Returns:
            Adapted features (B, H, W, D)
        """
        B, H, W, D = x.shape

        # MLP adaptation
        x_flat = x.reshape(-1, D)
        delta = self.adapter(x_flat).reshape(B, H, W, D)

        # Add spatial prompt if enabled (with interpolation for size mismatch)
        if self.use_spatial_prompt:
            if H != self.spatial_size or W != self.spatial_size:
                # Interpolate prompt to match input size
                # prompt: (1, S, S, D) -> (1, D, S, S) -> interpolate -> (1, D, H, W) -> (1, H, W, D)
                prompt = self.spatial_prompt.permute(0, 3, 1, 2)  # (1, D, S, S)
                prompt = torch.nn.functional.interpolate(
                    prompt, size=(H, W), mode='bilinear', align_corners=False
                )
                prompt = prompt.permute(0, 2, 3, 1)  # (1, H, W, D)
                delta = delta + prompt
            else:
                delta = delta + self.spatial_prompt

        return x + delta


class FeatureLevelMLPAdapter(nn.Module):
    """
    Feature-level MLP Adapter with larger capacity.

    This is an alternative to FeatureLevelPromptAdapter with more
    non-linear transformation capacity.
    """

    def __init__(self, channels: int, target_params: int = 1_930_000):
        """
        Args:
            channels: Feature dimension
            target_params: Target number of parameters
        """
        super(FeatureLevelMLPAdapter, self).__init__()

        self.channels = channels

        # Calculate hidden dimensions for 2-layer MLP
        # Params = ch*h1 + h1*h2 + h2*ch ≈ 2*ch*h for h1=h2=h
        hidden_dim = target_params // (3 * channels)

        self.adapter = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )

        # Zero-init last layer for identity start
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        actual_params = sum(p.numel() for p in self.parameters())
        print(f"📦 FeatureLevelMLPAdapter: {actual_params:,} params "
              f"(target: {target_params:,}, hidden_dim: {hidden_dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature-level MLP adaptation."""
        B, H, W, D = x.shape

        x_flat = x.reshape(-1, D)
        delta = self.adapter(x_flat).reshape(B, H, W, D)

        return x + delta
