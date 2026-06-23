"""
Task Prototype and Router for Task Selection.

Uses Mahalanobis distance to select the best LoRA expert.

V3 Improvements:
- TwoStageHybridRouter: Combines prototype and likelihood-based routing
- RegionalPrototype: Captures spatial structure for better discrimination
- CalibratedLikelihood: Normalized likelihood for fair comparison

Ablation Support:
- use_mahalanobis: If False, use Euclidean distance instead
- use_hybrid_routing: If True, use two-stage hybrid router
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from decoflow.models.decoflow_nf import DeCoFlowNF


class TaskPrototype:
    """
    Task Prototype for Distance-based Routing.

    Stores:
    - mu_t: Mean of image-level features
    - Sigma_t^{-1}: Precision matrix (inverse of covariance) for Mahalanobis
    """

    def __init__(self, task_id: int, task_classes: List[str], device: str = 'cuda'):
        self.task_id = task_id
        self.task_classes = task_classes
        self.device = device

        self.mean: Optional[torch.Tensor] = None
        self.precision: Optional[torch.Tensor] = None
        self.covariance: Optional[torch.Tensor] = None
        self.n_samples: int = 0

    def update(self, features: torch.Tensor):
        """
        Update prototype statistics with new features.

        Args:
            features: (N, D) image-level features
        """
        features = features.detach()

        if self.mean is None:
            self.mean = features.mean(dim=0)
            self.n_samples = features.shape[0]

            # Compute covariance with regularization
            centered = features - self.mean.unsqueeze(0)
            self.covariance = (centered.T @ centered) / (features.shape[0] - 1)

            # Add regularization for numerical stability
            reg = 1e-5 * torch.eye(features.shape[1], device=features.device)
            self.covariance = self.covariance + reg
        else:
            # Incremental update
            n_new = features.shape[0]
            n_total = self.n_samples + n_new

            new_mean = features.mean(dim=0)
            delta = new_mean - self.mean

            # Update mean
            self.mean = (self.n_samples * self.mean + n_new * new_mean) / n_total

            # Update covariance (Welford's online algorithm)
            centered_new = features - new_mean.unsqueeze(0)
            cov_new = (centered_new.T @ centered_new) / (n_new - 1) if n_new > 1 else torch.zeros_like(self.covariance)

            self.covariance = (self.n_samples * self.covariance + n_new * cov_new +
                              (self.n_samples * n_new / n_total) * torch.outer(delta, delta)) / n_total

            self.n_samples = n_total

    def finalize(self):
        """Compute precision matrix after all updates."""
        if self.covariance is not None:
            try:
                self.precision = torch.linalg.inv(self.covariance)
            except:
                # Fallback: use pseudo-inverse
                self.precision = torch.linalg.pinv(self.covariance)

    def mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance from prototype.

        Args:
            features: (N, D) features

        Returns:
            distances: (N,) Mahalanobis distances
        """
        if self.mean is None or self.precision is None:
            raise ValueError("Prototype not initialized. Call finalize() first.")

        centered = features - self.mean.unsqueeze(0)  # (N, D)
        # D_M = sqrt((x - mu)^T Sigma^{-1} (x - mu))
        # (N, D) @ (D, D) @ (D, N) -> diagonal gives (N,)
        distances = torch.sqrt(torch.sum(centered @ self.precision * centered, dim=1))

        return distances

    def euclidean_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance from prototype.

        Args:
            features: (N, D) features

        Returns:
            distances: (N,) Euclidean distances
        """
        if self.mean is None:
            raise ValueError("Prototype not initialized.")

        centered = features - self.mean.unsqueeze(0)  # (N, D)
        distances = torch.sqrt(torch.sum(centered ** 2, dim=1))

        return distances


class PrototypeRouter:
    """
    Prototype-based Router for Task Selection.

    Uses Mahalanobis or Euclidean distance to select the best LoRA expert.

    Ablation Support:
    - use_mahalanobis: If False, use Euclidean distance instead
    """

    def __init__(self, device: str = 'cuda', use_mahalanobis: bool = True):
        self.device = device
        self.use_mahalanobis = use_mahalanobis
        self.prototypes: Dict[int, TaskPrototype] = {}

    def add_prototype(self, task_id: int, prototype: TaskPrototype):
        """Add a task prototype."""
        self.prototypes[task_id] = prototype

    def route(self, features: torch.Tensor) -> torch.Tensor:
        """
        Route features to the best task based on distance.

        Args:
            features: (N, D) image-level features

        Returns:
            task_ids: (N,) predicted task IDs

        Uses Mahalanobis distance by default, or Euclidean if use_mahalanobis=False.
        """
        if not self.prototypes:
            return torch.zeros(features.shape[0], dtype=torch.long, device=features.device)

        all_distances = []
        task_ids = sorted(self.prototypes.keys())

        for task_id in task_ids:
            if self.use_mahalanobis:
                distances = self.prototypes[task_id].mahalanobis_distance(features)
            else:
                distances = self.prototypes[task_id].euclidean_distance(features)
            all_distances.append(distances)

        # Stack: (num_tasks, N)
        all_distances = torch.stack(all_distances, dim=0)

        # Find minimum distance task
        min_indices = torch.argmin(all_distances, dim=0)
        predicted_tasks = torch.tensor([task_ids[idx] for idx in min_indices],
                                        device=features.device)

        return predicted_tasks


# =============================================================================
# V3 Improvements: Regional Prototype and Hybrid Routing
# =============================================================================

class RegionalPrototype:
    """
    Regional Prototype for capturing spatial structure (V3 Solution 4).

    Unlike global mean, this captures regional feature distributions
    which helps distinguish structurally similar classes.

    Key Design:
    - Divide image into grid_size × grid_size regions
    - Store mean for each region
    - Distance considers all regional similarities
    """

    def __init__(self, task_id: int, task_classes: List[str],
                 grid_size: int = 4, device: str = 'cuda'):
        self.task_id = task_id
        self.task_classes = task_classes
        self.grid_size = grid_size
        self.device = device
        self.n_regions = grid_size * grid_size

        # Regional means: (n_regions, D)
        self.regional_means: Optional[torch.Tensor] = None
        # Global mean for fallback
        self.global_mean: Optional[torch.Tensor] = None
        self.n_samples: int = 0

    def update(self, patch_features: torch.Tensor):
        """
        Update regional prototype with new features.

        Args:
            patch_features: (N, H, W, D) patch-level features
        """
        N, H, W, D = patch_features.shape
        patch_features = patch_features.detach()

        # Compute regional features
        regional = self._compute_regional_features(patch_features)  # (N, n_regions, D)

        if self.regional_means is None:
            self.regional_means = regional.mean(dim=0)  # (n_regions, D)
            self.global_mean = patch_features.mean(dim=(0, 1, 2))  # (D,)
            self.n_samples = N
        else:
            # Incremental update
            n_total = self.n_samples + N
            new_regional_mean = regional.mean(dim=0)
            new_global_mean = patch_features.mean(dim=(0, 1, 2))

            self.regional_means = (
                self.n_samples * self.regional_means + N * new_regional_mean
            ) / n_total
            self.global_mean = (
                self.n_samples * self.global_mean + N * new_global_mean
            ) / n_total
            self.n_samples = n_total

    def _compute_regional_features(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Compute regional features by averaging within each region.

        Args:
            patch_features: (N, H, W, D)

        Returns:
            (N, n_regions, D) regional features
        """
        N, H, W, D = patch_features.shape

        # Reshape to grid
        rh = H // self.grid_size
        rw = W // self.grid_size

        # Handle case where H, W are not divisible by grid_size
        if rh == 0 or rw == 0:
            # Fall back to global pooling
            return patch_features.mean(dim=(1, 2), keepdim=True).expand(N, self.n_regions, D)

        # Truncate to fit grid
        H_trunc = rh * self.grid_size
        W_trunc = rw * self.grid_size
        patch_features = patch_features[:, :H_trunc, :W_trunc, :]

        # Reshape and average
        regional = patch_features.reshape(N, self.grid_size, rh, self.grid_size, rw, D)
        regional = regional.mean(dim=(2, 4))  # (N, grid_size, grid_size, D)
        regional = regional.reshape(N, self.n_regions, D)

        return regional

    def regional_distance(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Compute distance based on regional similarity.

        Args:
            patch_features: (N, H, W, D)

        Returns:
            distances: (N,)
        """
        if self.regional_means is None:
            raise ValueError("Prototype not initialized")

        regional = self._compute_regional_features(patch_features)  # (N, n_regions, D)

        # Distance: average of regional distances
        diff = regional - self.regional_means.unsqueeze(0)  # (N, n_regions, D)
        distances = (diff ** 2).sum(dim=-1).sqrt().mean(dim=-1)  # (N,)

        return distances


class LikelihoodCalibrator:
    """
    Likelihood Calibrator for fair comparison across tasks (V3 Solution 4).

    Different tasks may have different likelihood scales, making
    raw likelihood comparison unfair. This calibrator normalizes
    likelihoods to z-scores based on training data statistics.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        # task_id -> (mean_log_prob, std_log_prob)
        self.stats: Dict[int, Tuple[float, float]] = {}

    def calibrate(self, task_id: int, log_probs: torch.Tensor):
        """
        Store calibration statistics for a task.

        Args:
            task_id: Task identifier
            log_probs: (N,) log probabilities from training data
        """
        log_probs = log_probs.detach()
        mean = log_probs.mean().item()
        std = log_probs.std().item()

        # Ensure std is not zero
        std = max(std, 1e-6)

        self.stats[task_id] = (mean, std)
        print(f"   📊 Calibrator: Task {task_id} log_prob mean={mean:.2f}, std={std:.2f}")

    def normalize(self, task_id: int, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert log probabilities to z-scores.

        Args:
            task_id: Task identifier
            log_probs: (N,) log probabilities

        Returns:
            (N,) z-scores
        """
        if task_id not in self.stats:
            return log_probs

        mean, std = self.stats[task_id]
        return (log_probs - mean) / std

    def is_calibrated(self, task_id: int) -> bool:
        """Check if task is calibrated."""
        return task_id in self.stats


class TwoStageHybridRouter:
    """
    Two-Stage Hybrid Router (V3 Solution 4).

    Combines the efficiency of prototype-based routing with
    the accuracy of likelihood-based routing.

    Stage 1: Fast prototype filtering (select top-K candidates)
    Stage 2: Precise likelihood comparison (among candidates)

    This reduces computation while maintaining accuracy.
    """

    def __init__(self,
                 device: str = 'cuda',
                 use_mahalanobis: bool = True,
                 use_regional: bool = True,
                 top_k: int = 2,
                 prototype_weight: float = 0.6):
        """
        Args:
            device: Device to use
            use_mahalanobis: Use Mahalanobis distance for prototypes
            use_regional: Use regional prototypes
            top_k: Number of candidates for stage 2
            prototype_weight: Weight for prototype score in hybrid mode
        """
        self.device = device
        self.use_mahalanobis = use_mahalanobis
        self.use_regional = use_regional
        self.top_k = top_k
        self.prototype_weight = prototype_weight

        # Standard prototypes (image-level)
        self.prototypes: Dict[int, TaskPrototype] = {}

        # Regional prototypes (patch-level)
        self.regional_prototypes: Dict[int, RegionalPrototype] = {}

        # Likelihood calibrator
        self.calibrator = LikelihoodCalibrator(device)

        # Reference to NF model (set during inference)
        self.nf_model: Optional['DeCoFlowNF'] = None

    def set_nf_model(self, nf_model: 'DeCoFlowNF'):
        """Set NF model for likelihood computation."""
        self.nf_model = nf_model

    def add_prototype(self, task_id: int, prototype: TaskPrototype,
                      regional_prototype: RegionalPrototype = None):
        """Add prototypes for a task."""
        self.prototypes[task_id] = prototype
        if regional_prototype is not None:
            self.regional_prototypes[task_id] = regional_prototype

    def add_calibration(self, task_id: int, log_probs: torch.Tensor):
        """Add likelihood calibration for a task."""
        self.calibrator.calibrate(task_id, log_probs)

    def route(self,
              image_features: torch.Tensor,
              patch_features: torch.Tensor = None,
              use_likelihood: bool = True) -> torch.Tensor:
        """
        Route features to best task using two-stage approach.

        Args:
            image_features: (N, D) image-level features for prototype matching
            patch_features: (N, H, W, D) patch features for likelihood (optional)
            use_likelihood: Whether to use stage 2 likelihood routing

        Returns:
            task_ids: (N,) predicted task IDs
        """
        if not self.prototypes:
            return torch.zeros(image_features.shape[0], dtype=torch.long,
                             device=image_features.device)

        N = image_features.shape[0]
        task_ids = sorted(self.prototypes.keys())
        n_tasks = len(task_ids)

        # =================================================================
        # Stage 1: Prototype-based candidate selection
        # =================================================================
        all_distances = []

        for task_id in task_ids:
            # Compute prototype distance
            if self.use_mahalanobis:
                dist = self.prototypes[task_id].mahalanobis_distance(image_features)
            else:
                dist = self.prototypes[task_id].euclidean_distance(image_features)

            # Add regional distance if available
            if self.use_regional and task_id in self.regional_prototypes and patch_features is not None:
                regional_dist = self.regional_prototypes[task_id].regional_distance(patch_features)
                dist = 0.7 * dist + 0.3 * regional_dist

            all_distances.append(dist)

        # Stack distances: (n_tasks, N)
        distance_matrix = torch.stack(all_distances, dim=0)

        # If only one task or not using likelihood, return prototype result
        if n_tasks == 1 or not use_likelihood or self.nf_model is None or patch_features is None:
            min_indices = torch.argmin(distance_matrix, dim=0)
            return torch.tensor([task_ids[idx] for idx in min_indices],
                              device=image_features.device)

        # Select top-K candidates per sample
        effective_k = min(self.top_k, n_tasks)
        _, top_k_indices = torch.topk(distance_matrix, effective_k, dim=0, largest=False)
        # top_k_indices: (K, N)

        # =================================================================
        # Stage 2: Likelihood-based refinement (only for candidates)
        # =================================================================
        final_predictions = []

        for i in range(N):
            candidates = [task_ids[idx] for idx in top_k_indices[:, i]]

            if len(candidates) == 1:
                final_predictions.append(candidates[0])
                continue

            # Compute likelihood for each candidate
            candidate_scores = []
            for task_id in candidates:
                self.nf_model.set_active_task(task_id)
                with torch.no_grad():
                    log_prob = self.nf_model.log_prob(patch_features[i:i+1])

                # Normalize if calibrated
                if self.calibrator.is_calibrated(task_id):
                    score = self.calibrator.normalize(task_id, log_prob)
                else:
                    score = log_prob

                candidate_scores.append(score.item())

            # Combine with prototype distance
            candidate_distances = [
                distance_matrix[task_ids.index(t), i].item()
                for t in candidates
            ]

            # Normalize distances to scores (lower distance = higher score)
            max_dist = max(candidate_distances) + 1e-6
            dist_scores = [1 - d / max_dist for d in candidate_distances]

            # Normalize likelihood scores
            min_like = min(candidate_scores)
            max_like = max(candidate_scores)
            like_range = max_like - min_like + 1e-6
            like_scores = [(s - min_like) / like_range for s in candidate_scores]

            # Hybrid score
            hybrid_scores = [
                self.prototype_weight * ds + (1 - self.prototype_weight) * ls
                for ds, ls in zip(dist_scores, like_scores)
            ]

            # Select best
            best_idx = hybrid_scores.index(max(hybrid_scores))
            final_predictions.append(candidates[best_idx])

        return torch.tensor(final_predictions, device=image_features.device)

    def route_simple(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Simple prototype-only routing (for backward compatibility).

        Args:
            image_features: (N, D) image-level features

        Returns:
            task_ids: (N,) predicted task IDs
        """
        if not self.prototypes:
            return torch.zeros(image_features.shape[0], dtype=torch.long,
                             device=image_features.device)

        all_distances = []
        task_ids = sorted(self.prototypes.keys())

        for task_id in task_ids:
            if self.use_mahalanobis:
                distances = self.prototypes[task_id].mahalanobis_distance(image_features)
            else:
                distances = self.prototypes[task_id].euclidean_distance(image_features)
            all_distances.append(distances)

        all_distances = torch.stack(all_distances, dim=0)
        min_indices = torch.argmin(all_distances, dim=0)
        predicted_tasks = torch.tensor([task_ids[idx] for idx in min_indices],
                                        device=image_features.device)

        return predicted_tasks


# =============================================================================
# Class-Level Prototype and Router (for use_class_level_adapters mode)
# =============================================================================

class ClassPrototype:
    """
    Class Prototype for Class-Level Distance-based Routing.

    Similar to TaskPrototype, but for individual classes within tasks.
    This enables finer-grained routing when multiple classes are grouped
    into a single task (step).

    Stores:
    - mu_c: Mean of image-level features for this class
    - Sigma_c^{-1}: Precision matrix for Mahalanobis distance
    """

    def __init__(self, class_id: int, class_name: str, task_id: int, device: str = 'cuda'):
        """
        Args:
            class_id: Global class index (unique across all tasks)
            class_name: Class name (e.g., 'leather', 'grid')
            task_id: Task ID this class belongs to (for reference)
            device: Device to use
        """
        self.class_id = class_id
        self.class_name = class_name
        self.task_id = task_id  # Which task this class belongs to
        self.device = device

        self.mean: Optional[torch.Tensor] = None
        self.precision: Optional[torch.Tensor] = None
        self.covariance: Optional[torch.Tensor] = None
        self.n_samples: int = 0

    def update(self, features: torch.Tensor):
        """
        Update prototype statistics with new features.

        Args:
            features: (N, D) image-level features from this class only
        """
        features = features.detach()

        if self.mean is None:
            self.mean = features.mean(dim=0)
            self.n_samples = features.shape[0]

            # Compute covariance with regularization
            if features.shape[0] > 1:
                centered = features - self.mean.unsqueeze(0)
                self.covariance = (centered.T @ centered) / (features.shape[0] - 1)
            else:
                # Single sample: use identity covariance
                self.covariance = torch.eye(features.shape[1], device=features.device)

            # Add regularization for numerical stability
            reg = 1e-5 * torch.eye(features.shape[1], device=features.device)
            self.covariance = self.covariance + reg
        else:
            # Incremental update (same as TaskPrototype)
            n_new = features.shape[0]
            n_total = self.n_samples + n_new

            new_mean = features.mean(dim=0)
            delta = new_mean - self.mean

            # Update mean
            self.mean = (self.n_samples * self.mean + n_new * new_mean) / n_total

            # Update covariance (Welford's online algorithm)
            centered_new = features - new_mean.unsqueeze(0)
            cov_new = (centered_new.T @ centered_new) / (n_new - 1) if n_new > 1 else torch.zeros_like(self.covariance)

            self.covariance = (self.n_samples * self.covariance + n_new * cov_new +
                              (self.n_samples * n_new / n_total) * torch.outer(delta, delta)) / n_total

            self.n_samples = n_total

    def finalize(self):
        """Compute precision matrix after all updates."""
        if self.covariance is not None:
            try:
                self.precision = torch.linalg.inv(self.covariance)
            except:
                # Fallback: use pseudo-inverse
                self.precision = torch.linalg.pinv(self.covariance)

    def mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance from prototype.

        Args:
            features: (N, D) features

        Returns:
            distances: (N,) Mahalanobis distances
        """
        if self.mean is None or self.precision is None:
            raise ValueError("Prototype not initialized. Call finalize() first.")

        centered = features - self.mean.unsqueeze(0)
        distances = torch.sqrt(torch.sum(centered @ self.precision * centered, dim=1) + 1e-8)

        return distances

    def euclidean_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance from prototype.

        Args:
            features: (N, D) features

        Returns:
            distances: (N,) Euclidean distances
        """
        if self.mean is None:
            raise ValueError("Prototype not initialized.")

        centered = features - self.mean.unsqueeze(0)
        distances = torch.sqrt(torch.sum(centered ** 2, dim=1) + 1e-8)

        return distances


class ClassRouter:
    """
    Class-Level Prototype Router for Class Selection.

    Routes images to specific class adapters (not task adapters).
    Uses Mahalanobis or Euclidean distance to select the best class expert.

    This router is used when use_class_level_adapters=True, enabling
    finer-grained routing when multiple classes are grouped into tasks.
    """

    def __init__(self, device: str = 'cuda', use_mahalanobis: bool = True):
        self.device = device
        self.use_mahalanobis = use_mahalanobis
        self.prototypes: Dict[int, ClassPrototype] = {}  # class_id -> ClassPrototype

        # Mapping from class_id to task_id (for reference)
        self.class_to_task: Dict[int, int] = {}
        # Mapping from class_id to class_name
        self.class_to_name: Dict[int, str] = {}

    def add_prototype(self, class_id: int, prototype: ClassPrototype):
        """
        Add a class prototype.

        Args:
            class_id: Global class index
            prototype: ClassPrototype instance
        """
        self.prototypes[class_id] = prototype
        self.class_to_task[class_id] = prototype.task_id
        self.class_to_name[class_id] = prototype.class_name

    def route(self, features: torch.Tensor) -> torch.Tensor:
        """
        Route features to the best class based on distance.

        Args:
            features: (N, D) image-level features

        Returns:
            class_ids: (N,) predicted class IDs (global indices)
        """
        if not self.prototypes:
            return torch.zeros(features.shape[0], dtype=torch.long, device=features.device)

        all_distances = []
        class_ids = sorted(self.prototypes.keys())

        for class_id in class_ids:
            if self.use_mahalanobis:
                distances = self.prototypes[class_id].mahalanobis_distance(features)
            else:
                distances = self.prototypes[class_id].euclidean_distance(features)
            all_distances.append(distances)

        # Stack: (num_classes, N)
        all_distances = torch.stack(all_distances, dim=0)

        # Find minimum distance class
        min_indices = torch.argmin(all_distances, dim=0)
        predicted_classes = torch.tensor([class_ids[idx] for idx in min_indices],
                                          device=features.device)

        return predicted_classes

    def route_with_confidence(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route features with confidence scores.

        Args:
            features: (N, D) image-level features

        Returns:
            class_ids: (N,) predicted class IDs
            confidences: (N,) confidence scores (negative distance, higher is better)
        """
        if not self.prototypes:
            return (
                torch.zeros(features.shape[0], dtype=torch.long, device=features.device),
                torch.zeros(features.shape[0], device=features.device)
            )

        all_distances = []
        class_ids = sorted(self.prototypes.keys())

        for class_id in class_ids:
            if self.use_mahalanobis:
                distances = self.prototypes[class_id].mahalanobis_distance(features)
            else:
                distances = self.prototypes[class_id].euclidean_distance(features)
            all_distances.append(distances)

        all_distances = torch.stack(all_distances, dim=0)  # (num_classes, N)

        # Find minimum distance class
        min_distances, min_indices = torch.min(all_distances, dim=0)
        predicted_classes = torch.tensor([class_ids[idx] for idx in min_indices],
                                          device=features.device)

        # Confidence: negative distance (higher is better)
        confidences = -min_distances

        return predicted_classes, confidences

    def get_task_for_class(self, class_id: int) -> int:
        """Get the task ID that a class belongs to."""
        return self.class_to_task.get(class_id, 0)

    def get_class_name(self, class_id: int) -> str:
        """Get the class name for a class ID."""
        return self.class_to_name.get(class_id, f"class_{class_id}")

    def get_num_classes(self) -> int:
        """Get total number of registered classes."""
        return len(self.prototypes)

    def get_classes_for_task(self, task_id: int) -> List[int]:
        """Get all class IDs belonging to a task."""
        return [c_id for c_id, t_id in self.class_to_task.items() if t_id == task_id]
