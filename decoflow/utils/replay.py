"""
Continual Learning Utilities for DeCoFlow (V3).

Provides:
- OrthogonalGradientProjection (OGP): Projects gradients to preserve previous tasks
- FeatureBank: Stores representative features from previous tasks
- DistillationLoss: Knowledge distillation from teacher to student
- EWC (Elastic Weight Consolidation): Protects important parameters

V3 Key Insight:
Instead of Replay (storing/generating data), we use gradient-based
methods that only store small matrices representing important subspaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy


class FeatureBank:
    """
    Feature Bank for Continual Learning (V3 Solution 1).

    Stores representative features from each task using coreset selection.
    This is more reliable than generative replay because:
    1. No generation quality issues
    2. No error accumulation across tasks
    3. Memory efficient (fixed budget per task)

    Usage:
    - After training Task t, call store(task_id, features)
    - During training Task t+1, call sample(task_id) to get replay features
    """

    def __init__(self,
                 max_samples_per_task: int = 500,
                 selection_method: str = 'random',
                 device: str = 'cuda'):
        """
        Args:
            max_samples_per_task: Maximum features to store per task
            selection_method: 'random', 'kmeans', or 'herding'
            device: Device for stored features
        """
        self.max_samples_per_task = max_samples_per_task
        self.selection_method = selection_method
        self.device = device

        # Storage: task_id -> (features, spatial_shape)
        self.banks: Dict[int, Tuple[torch.Tensor, Tuple[int, int]]] = {}

    def store(self, task_id: int, features: torch.Tensor,
              spatial_shape: Tuple[int, int] = None):
        """
        Store representative features for a task.

        Args:
            task_id: Task identifier
            features: (N, H, W, D) features from training data
            spatial_shape: (H, W) spatial dimensions
        """
        N = features.shape[0]

        if spatial_shape is None:
            spatial_shape = (features.shape[1], features.shape[2])

        # Flatten to (N, H*W*D) for selection
        features_flat = features.reshape(N, -1)

        # Select representative samples
        if N <= self.max_samples_per_task:
            selected_indices = torch.arange(N)
        else:
            selected_indices = self._select_samples(
                features_flat, self.max_samples_per_task
            )

        # Store selected features
        selected_features = features[selected_indices].detach().cpu()
        self.banks[task_id] = (selected_features, spatial_shape)

        print(f"   📦 FeatureBank: Stored {len(selected_indices)} samples for Task {task_id}")

    def _select_samples(self, features: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Select representative samples using specified method."""
        N = features.shape[0]

        if self.selection_method == 'random':
            indices = torch.randperm(N)[:n_samples]

        elif self.selection_method == 'kmeans':
            # Simple k-means based selection
            indices = self._kmeans_select(features, n_samples)

        elif self.selection_method == 'herding':
            # Herding-based selection (closer to mean)
            indices = self._herding_select(features, n_samples)

        else:
            indices = torch.randperm(N)[:n_samples]

        return indices

    def _kmeans_select(self, features: torch.Tensor, n_samples: int) -> torch.Tensor:
        """K-means based coreset selection."""
        try:
            from sklearn.cluster import KMeans
            import numpy as np

            features_np = features.detach().cpu().numpy()

            # Cluster into n_samples clusters
            kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
            kmeans.fit(features_np)

            # Select sample closest to each centroid
            indices = []
            for i in range(n_samples):
                cluster_mask = kmeans.labels_ == i
                cluster_features = features_np[cluster_mask]
                centroid = kmeans.cluster_centers_[i]

                # Find closest sample to centroid
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                closest_in_cluster = np.argmin(distances)

                # Map back to original index
                original_indices = np.where(cluster_mask)[0]
                indices.append(original_indices[closest_in_cluster])

            return torch.tensor(indices)

        except ImportError:
            # Fallback to random if sklearn not available
            return torch.randperm(features.shape[0])[:n_samples]

    def _herding_select(self, features: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Herding-based selection (iCaRL style)."""
        features = features.detach()
        mean = features.mean(dim=0, keepdim=True)

        indices = []
        selected_sum = torch.zeros_like(mean)

        for _ in range(n_samples):
            # Find sample that brings running mean closest to true mean
            remaining_mask = torch.ones(features.shape[0], dtype=torch.bool)
            remaining_mask[indices] = False

            if not remaining_mask.any():
                break

            remaining_features = features[remaining_mask]
            remaining_indices = torch.where(remaining_mask)[0]

            # Compute how adding each sample affects the mean
            n_selected = len(indices) + 1
            candidate_means = (selected_sum + remaining_features) / n_selected
            distances = (candidate_means - mean).norm(dim=1)

            best_idx = distances.argmin()
            original_idx = remaining_indices[best_idx]

            indices.append(original_idx.item())
            selected_sum = selected_sum + features[original_idx:original_idx+1]

        return torch.tensor(indices)

    def sample(self, task_id: int, n_samples: int = None,
               device: str = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Sample features from a stored task.

        Args:
            task_id: Task to sample from
            n_samples: Number of samples (None = all stored)
            device: Device to move samples to

        Returns:
            (features, spatial_shape) tuple
        """
        if task_id not in self.banks:
            raise ValueError(f"Task {task_id} not in feature bank")

        features, spatial_shape = self.banks[task_id]

        if device is None:
            device = self.device

        if n_samples is not None and n_samples < features.shape[0]:
            indices = torch.randperm(features.shape[0])[:n_samples]
            features = features[indices]

        return features.to(device), spatial_shape

    def get_all_task_ids(self) -> List[int]:
        """Get list of stored task IDs."""
        return list(self.banks.keys())

    def get_task_sample_count(self, task_id: int) -> int:
        """Get number of stored samples for a task."""
        if task_id in self.banks:
            return self.banks[task_id][0].shape[0]
        return 0

    def clear(self):
        """Clear all stored features."""
        self.banks.clear()


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for NF models (V3 Solution 1).

    Computes distillation loss between teacher and student NF outputs.
    Supports multiple distillation modes:
    - 'logprob': Match log probability distributions
    - 'latent': Match latent space representations
    - 'combined': Both logprob and latent matching
    """

    def __init__(self,
                 mode: str = 'logprob',
                 temperature: float = 2.0,
                 latent_weight: float = 1.0):
        """
        Args:
            mode: 'logprob', 'latent', or 'combined'
            temperature: Temperature for probability softening
            latent_weight: Weight for latent space matching (if combined)
        """
        super(DistillationLoss, self).__init__()
        self.mode = mode
        self.temperature = temperature
        self.latent_weight = latent_weight

    def forward(self,
                student_z: torch.Tensor,
                student_logdet: torch.Tensor,
                teacher_z: torch.Tensor,
                teacher_logdet: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_z: Student latent output (B, H, W, D)
            student_logdet: Student log det Jacobian (B, H, W)
            teacher_z: Teacher latent output (B, H, W, D)
            teacher_logdet: Teacher log det Jacobian (B, H, W)

        Returns:
            Distillation loss scalar
        """
        if self.mode == 'latent':
            return self._latent_loss(student_z, teacher_z)

        elif self.mode == 'logprob':
            return self._logprob_loss(
                student_z, student_logdet,
                teacher_z, teacher_logdet
            )

        else:  # combined
            latent_loss = self._latent_loss(student_z, teacher_z)
            logprob_loss = self._logprob_loss(
                student_z, student_logdet,
                teacher_z, teacher_logdet
            )
            return logprob_loss + self.latent_weight * latent_loss

    def _latent_loss(self, student_z: torch.Tensor,
                     teacher_z: torch.Tensor) -> torch.Tensor:
        """Match latent representations."""
        # MSE loss on latent vectors
        return F.mse_loss(student_z, teacher_z.detach())

    def _logprob_loss(self,
                      student_z: torch.Tensor,
                      student_logdet: torch.Tensor,
                      teacher_z: torch.Tensor,
                      teacher_logdet: torch.Tensor) -> torch.Tensor:
        """Match log probability distributions."""
        import math

        B, H, W, D = student_z.shape

        # Compute log p(z) for both
        student_log_pz = -0.5 * (student_z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
        teacher_log_pz = -0.5 * (teacher_z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Full log probability
        student_log_px = student_log_pz + student_logdet
        teacher_log_px = teacher_log_pz + teacher_logdet

        # Temperature scaling
        student_scaled = student_log_px / self.temperature
        teacher_scaled = teacher_log_px / self.temperature

        # KL divergence on softmaxed probabilities (per image)
        student_probs = F.softmax(student_scaled.reshape(B, -1), dim=1)
        teacher_probs = F.softmax(teacher_scaled.reshape(B, -1), dim=1)

        # KL(teacher || student)
        kl_loss = F.kl_div(
            student_probs.log(),
            teacher_probs.detach(),
            reduction='batchmean'
        )

        return kl_loss * (self.temperature ** 2)


class EWC:
    """
    Elastic Weight Consolidation (V3 Solution 1).

    Protects important parameters from previous tasks by adding
    a quadratic penalty when they deviate from their optimal values.

    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting
    in neural networks", PNAS 2017
    """

    def __init__(self, lambda_ewc: float = 1000.0):
        """
        Args:
            lambda_ewc: Regularization strength
        """
        self.lambda_ewc = lambda_ewc

        # Storage for Fisher information and optimal parameters
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

        self.is_initialized = False

    def compute_fisher(self,
                       model: nn.Module,
                       data_loader,
                       device: str = 'cuda',
                       n_samples: int = 500):
        """
        Compute Fisher information matrix diagonal for current parameters.

        Args:
            model: The model to compute Fisher for
            data_loader: DataLoader for computing gradients
            device: Device to use
            n_samples: Number of samples to use for estimation
        """
        model.eval()

        # Initialize Fisher to zero
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        # Accumulate squared gradients
        n_processed = 0
        for batch in data_loader:
            if n_processed >= n_samples:
                break

            features = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            batch_size = features.shape[0]

            # Forward pass
            model.zero_grad()
            log_prob = model.log_prob(features)
            loss = -log_prob.mean()
            loss.backward()

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad ** 2) * batch_size

            n_processed += batch_size

        # Normalize
        for name in fisher:
            fisher[name] /= n_processed

        # Store Fisher and current parameters
        self.fisher = fisher
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        self.is_initialized = True
        print(f"   📊 EWC: Computed Fisher for {len(fisher)} parameters using {n_processed} samples")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty for current parameters.

        Args:
            model: Current model

        Returns:
            EWC penalty loss
        """
        if not self.is_initialized:
            return torch.tensor(0.0)

        penalty = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher and name in self.optimal_params:
                fisher = self.fisher[name]
                optimal = self.optimal_params[name]

                # Quadratic penalty weighted by Fisher information
                penalty += (fisher * (param - optimal) ** 2).sum()

        return self.lambda_ewc * penalty

    def update(self,
               model: nn.Module,
               data_loader,
               device: str = 'cuda',
               n_samples: int = 500,
               consolidation_mode: str = 'sum'):
        """
        Update Fisher information after learning a new task.

        Args:
            model: Model after learning new task
            data_loader: DataLoader for new task
            device: Device to use
            n_samples: Number of samples for estimation
            consolidation_mode: 'sum' or 'max' for combining old and new Fisher
        """
        # Compute new Fisher
        new_fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_fisher[name] = torch.zeros_like(param)

        model.eval()
        n_processed = 0

        for batch in data_loader:
            if n_processed >= n_samples:
                break

            features = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            batch_size = features.shape[0]

            model.zero_grad()
            log_prob = model.log_prob(features)
            loss = -log_prob.mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    new_fisher[name] += (param.grad ** 2) * batch_size

            n_processed += batch_size

        for name in new_fisher:
            new_fisher[name] /= n_processed

        # Consolidate with old Fisher
        if self.is_initialized:
            for name in new_fisher:
                if name in self.fisher:
                    if consolidation_mode == 'sum':
                        self.fisher[name] = self.fisher[name] + new_fisher[name]
                    else:  # max
                        self.fisher[name] = torch.max(self.fisher[name], new_fisher[name])
                else:
                    self.fisher[name] = new_fisher[name]
        else:
            self.fisher = new_fisher

        # Update optimal parameters
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        self.is_initialized = True


def create_teacher_model(model: nn.Module) -> nn.Module:
    """
    Create a frozen copy of the model to use as teacher.

    Args:
        model: Model to copy

    Returns:
        Frozen copy of model
    """
    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


# =============================================================================
# V3 Solution: Orthogonal Gradient Projection (OGP)
# =============================================================================

class OrthogonalGradientProjection:
    """
    Orthogonal Gradient Projection (OGP) - V3 No-Replay Solution.

    Key Idea:
    After learning Task t, compute the principal subspace of gradients
    (or features) that are important for that task. When learning Task t+1,
    project gradients to be orthogonal to this subspace, ensuring that
    updates don't interfere with previously learned knowledge.

    This is based on GPM (Gradient Projection Memory) from:
    "Continual Learning in Low-rank Orthogonal Subspaces", NeurIPS 2020

    Mathematical Formulation:
    1. After Task t: Compute U_t = SVD(G_t)[:, :k] where G_t is gradient matrix
    2. Store M_t = U_t @ U_t^T (projection matrix onto important subspace)
    3. For Task t+1: g' = g - M_t @ g (project gradient to null space)

    Advantages over Replay:
    - No data storage required
    - Memory: O(d × k) per task where k << d
    - Mathematically guarantees no interference in stored subspace
    """

    def __init__(self,
                 threshold: float = 0.99,
                 max_rank_per_task: int = 50,
                 device: str = 'cuda'):
        """
        Args:
            threshold: Cumulative variance threshold for selecting principal components
            max_rank_per_task: Maximum rank of subspace per task
            device: Device for computations
        """
        self.threshold = threshold
        self.max_rank_per_task = max_rank_per_task
        self.device = device

        # Store projection bases per parameter
        # param_name -> list of basis matrices (one per task)
        self.bases: Dict[str, List[torch.Tensor]] = {}

        self.is_initialized = False
        self.n_tasks = 0

    def compute_and_store_basis(self,
                                 model: nn.Module,
                                 data_loader,
                                 task_id: int,
                                 n_samples: int = 300):
        """
        Compute gradient subspace basis for completed task.

        This should be called AFTER training on a task is complete.

        Args:
            model: Trained model
            data_loader: DataLoader for the completed task
            task_id: Task identifier
            n_samples: Number of samples for gradient collection
        """
        model.eval()

        # Collect gradients for each parameter
        gradient_matrices: Dict[str, List[torch.Tensor]] = {}

        n_processed = 0
        for batch in data_loader:
            if n_processed >= n_samples:
                break

            features = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            batch_size = features.shape[0]

            # Forward and backward
            model.zero_grad()
            try:
                log_prob = model.log_prob(features)
                loss = -log_prob.mean()
                loss.backward()
            except Exception as e:
                continue

            # Collect gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad.detach().flatten()

                    if name not in gradient_matrices:
                        gradient_matrices[name] = []
                    gradient_matrices[name].append(grad.clone())

            n_processed += batch_size

        # Compute SVD for each parameter's gradient matrix
        for name, grads in gradient_matrices.items():
            if len(grads) < 5:  # Need enough samples
                continue

            # Stack gradients: (n_samples, n_params)
            G = torch.stack(grads, dim=0)

            # Compute SVD
            try:
                U, S, Vh = torch.linalg.svd(G, full_matrices=False)
            except Exception:
                continue

            # Select top-k components based on variance threshold
            var_ratio = (S ** 2).cumsum(0) / (S ** 2).sum()
            k = min(
                (var_ratio < self.threshold).sum().item() + 1,
                self.max_rank_per_task,
                S.shape[0]
            )

            # Store basis vectors (columns of V transposed)
            # Vh shape: (min(m,n), n), we want (n, k) basis
            basis = Vh[:k, :].T  # (n_params, k)

            if name not in self.bases:
                self.bases[name] = []
            self.bases[name].append(basis.to(self.device))

        self.n_tasks = task_id + 1
        self.is_initialized = True

        total_bases = sum(len(v) for v in self.bases.values())
        print(f"   📐 OGP: Stored {total_bases} basis matrices for Task {task_id}")
        print(f"      Avg rank per param: {total_bases / max(len(self.bases), 1):.1f}")

    def project_gradient(self, model: nn.Module):
        """
        Project current gradients to be orthogonal to stored subspaces.

        Call this AFTER loss.backward() and BEFORE optimizer.step().

        Args:
            model: Model with computed gradients
        """
        if not self.is_initialized:
            return

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            if name not in self.bases:
                continue

            grad = param.grad.flatten()

            # Project out all stored subspaces for this parameter
            for basis in self.bases[name]:
                # basis: (n_params, k)
                # Project gradient onto subspace: (basis @ basis.T) @ grad
                # Then subtract to get orthogonal component
                proj = basis @ (basis.T @ grad)
                grad = grad - proj

            # Reshape and assign back
            param.grad = grad.reshape(param.shape)

    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        total_elements = 0
        for name, basis_list in self.bases.items():
            for basis in basis_list:
                total_elements += basis.numel()

        return {
            'n_params': len(self.bases),
            'n_tasks': self.n_tasks,
            'total_elements': total_elements,
            'memory_mb': total_elements * 4 / (1024 * 1024)  # Assuming float32
        }


class GradientProjectionHook:
    """
    Hook for automatic gradient projection during training.

    Usage:
        ogp = OrthogonalGradientProjection()
        # ... train task 0 ...
        ogp.compute_and_store_basis(model, loader, task_id=0)

        # For task 1+:
        hook = GradientProjectionHook(ogp, model)
        # Training loop will automatically project gradients
    """

    def __init__(self, ogp: OrthogonalGradientProjection, model: nn.Module):
        self.ogp = ogp
        self.model = model
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register backward hooks for gradient projection."""
        def make_hook(name):
            def hook(grad):
                if name not in self.ogp.bases:
                    return grad

                grad_flat = grad.flatten()
                for basis in self.ogp.bases[name]:
                    proj = basis @ (basis.T @ grad_flat)
                    grad_flat = grad_flat - proj

                return grad_flat.reshape(grad.shape)
            return hook

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.ogp.bases:
                handle = param.register_hook(make_hook(name))
                self.hooks.append(handle)

    def remove(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
