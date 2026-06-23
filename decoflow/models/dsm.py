"""
Denoising Score Matching (DSM) for DeCoFlow.

V7 Implementation: Combines Normalizing Flow's conservative field properties
with MULDE-style score matching for improved anomaly detection.

Key Components:
- NoiseSchedule: Multi-scale noise sampling (geometric/uniform)
- DSMLoss: Sliced Score Matching implementation

References:
- MULDE (CVPR 2024): Multi-scale Log-Density Estimation via Denoising Score Matching
- Song et al. (ICLR 2020): Sliced Score Matching for efficient gradient estimation
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from decoflow.models.decoflow_nf import DeCoFlowNF


class NoiseSchedule:
    """
    Multi-scale noise schedule for DSM.

    Supports:
    - geometric: sigma ~ LogUniform(sigma_min, sigma_max) - recommended
    - uniform: sigma ~ Uniform(sigma_min, sigma_max)
    - fixed: sigma = sigma_min (single scale)

    Geometric sampling ensures equal representation across noise scales,
    which is important for learning the score function across the manifold.
    """

    def __init__(self,
                 sigma_min: float = 0.01,
                 sigma_max: float = 1.0,
                 mode: str = 'geometric'):
        """
        Args:
            sigma_min: Minimum noise scale
            sigma_max: Maximum noise scale
            mode: 'geometric', 'uniform', or 'fixed'
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mode = mode

        # Pre-compute log bounds for geometric sampling
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noise scales for a batch.

        Args:
            batch_size: Number of samples
            device: Target device

        Returns:
            sigma: (batch_size,) noise scales
        """
        if self.mode == 'geometric':
            # LogUniform: log(sigma) ~ Uniform(log(sigma_min), log(sigma_max))
            log_sigma = torch.rand(batch_size, device=device)
            log_sigma = log_sigma * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
            return torch.exp(log_sigma)
        elif self.mode == 'uniform':
            return torch.rand(batch_size, device=device) * (self.sigma_max - self.sigma_min) + self.sigma_min
        else:  # fixed
            return torch.full((batch_size,), self.sigma_min, device=device)

    def __repr__(self):
        return f"NoiseSchedule(mode={self.mode}, sigma=[{self.sigma_min}, {self.sigma_max}])"


class DSMLoss(nn.Module):
    """
    Denoising Score Matching Loss for Normalizing Flows.

    Computes the DSM objective:
        L_DSM = E[||s_theta(x_noisy) + epsilon/sigma||^2]

    where:
        - s_theta(x) = nabla_x log p_theta(x) is the score function
        - x_noisy = x + sigma * epsilon
        - epsilon ~ N(0, I)
        - sigma ~ NoiseSchedule

    Uses Sliced Score Matching (SSM) for efficiency:
        L_SSM = E_v[(v^T * s_theta(x) + v^T * epsilon/sigma)^2]

    This reduces complexity from O(D) to O(1) per sample.

    Key Insight (MULDE):
        By training with noise-perturbed inputs, the model learns the score
        function not just on the data manifold, but also in the surrounding
        space. This provides better OOD detection and more robust density
        estimates.

    Key Advantage (This Implementation):
        Normalizing Flows guarantee a conservative vector field (curl = 0),
        which is a theoretical requirement for valid score functions that
        MULDE's MLP cannot guarantee.
    """

    def __init__(self,
                 sigma_min: float = 0.01,
                 sigma_max: float = 1.0,
                 use_sliced: bool = True,
                 n_projections: int = 1,
                 noise_mode: str = 'geometric'):
        """
        Args:
            sigma_min: Minimum noise scale
            sigma_max: Maximum noise scale
            use_sliced: Use Sliced Score Matching (recommended for efficiency)
            n_projections: Number of random projections for SSM
            noise_mode: Noise schedule mode ('geometric', 'uniform', 'fixed')
        """
        super().__init__()

        self.noise_schedule = NoiseSchedule(sigma_min, sigma_max, noise_mode)
        self.use_sliced = use_sliced
        self.n_projections = n_projections
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def inject_noise(self, x: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject Gaussian noise scaled by sigma.

        Args:
            x: (B, H, W, D) clean features
            sigma: (B,) noise scales per sample

        Returns:
            x_noisy: (B, H, W, D) noisy features
            epsilon: (B, H, W, D) noise that was added (for DSM target)
        """
        epsilon = torch.randn_like(x)
        # Expand sigma for broadcasting: (B,) -> (B, 1, 1, 1)
        sigma_expanded = sigma.view(-1, 1, 1, 1)
        x_noisy = x + sigma_expanded * epsilon
        return x_noisy, epsilon

    def compute_score_ssm(self,
                          nf_model: 'DeCoFlowNF',
                          x_noisy: torch.Tensor,
                          epsilon: torch.Tensor,
                          sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute Sliced Score Matching loss.

        SSM approximates the full DSM objective via random projections:
            L_SSM = E_v[(v^T * s_theta(x) + v^T * epsilon/sigma)^2]

        This is O(1) in feature dimension vs O(D) for full DSM.

        Args:
            nf_model: DeCoFlowNF model
            x_noisy: (B, H, W, D) noisy features
            epsilon: (B, H, W, D) noise that was added
            sigma: (B,) noise scales

        Returns:
            loss: Scalar SSM loss
        """
        B, H, W, D = x_noisy.shape
        device = x_noisy.device

        total_loss = 0.0

        for _ in range(self.n_projections):
            # Random projection vector (unit norm per patch)
            v = torch.randn(B, H, W, D, device=device)
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

            # Enable gradient tracking for score computation
            x_noisy_grad = x_noisy.detach().requires_grad_(True)

            # Forward through NF to get log p(x)
            z, logdet_patch = nf_model.forward(x_noisy_grad, reverse=False)

            # Compute patch-wise log p(x)
            log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            log_px_patch = log_pz_patch + logdet_patch  # (B, H, W)

            # Sum over spatial dimensions for gradient computation
            log_px = log_px_patch.sum()

            # Compute score: nabla_x log p(x)
            score = torch.autograd.grad(
                outputs=log_px,
                inputs=x_noisy_grad,
                create_graph=True,
                retain_graph=True
            )[0]  # (B, H, W, D)

            # v^T * score (projected score)
            v_dot_score = (v * score).sum(dim=(1, 2, 3))  # (B,)

            # v^T * (-epsilon / sigma) (projected target)
            # Note: score should equal -epsilon/sigma for denoising
            sigma_expanded = sigma.view(-1, 1, 1, 1)
            v_dot_target = -(v * epsilon / sigma_expanded).sum(dim=(1, 2, 3))  # (B,)

            # SSM loss: (v^T * score - v^T * target)^2
            # Which is: (v^T * score + v^T * epsilon/sigma)^2
            ssm_loss = ((v_dot_score - v_dot_target) ** 2).mean()
            total_loss = total_loss + ssm_loss

        return total_loss / self.n_projections

    def compute_score_full(self,
                           nf_model: 'DeCoFlowNF',
                           x_noisy: torch.Tensor,
                           epsilon: torch.Tensor,
                           sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute full Denoising Score Matching loss.

            L_DSM = E[||s_theta(x_noisy) - (-epsilon/sigma)||^2]

        Note: This is expensive (O(D) per sample) but more accurate.

        Args:
            nf_model: DeCoFlowNF model
            x_noisy: (B, H, W, D) noisy features
            epsilon: (B, H, W, D) noise that was added
            sigma: (B,) noise scales

        Returns:
            loss: Scalar DSM loss
        """
        B, H, W, D = x_noisy.shape

        # Enable gradient tracking
        x_noisy_grad = x_noisy.detach().requires_grad_(True)

        # Forward through NF
        z, logdet_patch = nf_model.forward(x_noisy_grad, reverse=False)

        # Compute patch-wise log p(x)
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
        log_px_patch = log_pz_patch + logdet_patch  # (B, H, W)

        # Compute full score via autograd
        score = torch.autograd.grad(
            outputs=log_px_patch.sum(),
            inputs=x_noisy_grad,
            create_graph=True,
            retain_graph=True
        )[0]  # (B, H, W, D)

        # Target: -epsilon/sigma (the direction that removes noise)
        sigma_expanded = sigma.view(-1, 1, 1, 1)
        target = -epsilon / sigma_expanded

        # DSM loss: ||score - target||^2
        # Weight by sigma^2 for scale invariance (optional, MULDE style)
        dsm_loss = ((score - target) ** 2).sum(dim=-1).mean()

        return dsm_loss

    def forward(self,
                nf_model: 'DeCoFlowNF',
                x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute DSM loss for training.

        Args:
            nf_model: DeCoFlowNF model (with active task set)
            x: (B, H, W, D) clean features (after adapter, before NF)

        Returns:
            loss: DSM loss scalar
            info: Dict with diagnostic information
        """
        B = x.shape[0]
        device = x.device

        # Sample noise scales from schedule
        sigma = self.noise_schedule.sample(B, device)

        # Inject noise
        x_noisy, epsilon = self.inject_noise(x, sigma)

        # Compute score matching loss
        if self.use_sliced:
            loss = self.compute_score_ssm(nf_model, x_noisy, epsilon, sigma)
        else:
            loss = self.compute_score_full(nf_model, x_noisy, epsilon, sigma)

        info = {
            'dsm_loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'sigma_mean': sigma.mean().item(),
            'sigma_min': sigma.min().item(),
            'sigma_max': sigma.max().item(),
        }

        return loss, info

    def __repr__(self):
        return (f"DSMLoss(sigma=[{self.sigma_min}, {self.sigma_max}], "
                f"sliced={self.use_sliced}, n_proj={self.n_projections})")
