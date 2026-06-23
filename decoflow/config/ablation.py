"""
Ablation Configuration for DeCoFlow.

This module provides configuration for ablation studies,
allowing individual components to be enabled/disabled.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AblationConfig:
    """
    Configuration for ablation studies.

    Each flag controls whether a specific component is enabled.
    Set to False to disable the component for ablation study.

    Components:
        use_lora: LoRA (Low-Rank Adaptation) for task-specific adaptation
        use_router: Prototype-based router for task selection
        use_task_adapter: Task Input Adapter (Instance Normalization)
        use_pos_embedding: Positional embedding for spatial awareness
        use_slow_stage: Slow-Fast two-stage training
        use_task_bias: Task-specific bias in LoRA layers
        use_mahalanobis: Mahalanobis distance in router (vs Euclidean)

    Adapter Modes (adapter_mode):
        "soft_ln": SoftLNTaskInputAdapter (default) - optional/weak LayerNorm for Task > 0
        "standard": Original FiLM-style TaskInputAdapter
        "no_ln_after_task0": Disable LN for Task > 0 (quick ablation)
        "no_ln": Disable LN completely for all tasks (ablation study)
        "tsa": V3 TaskSpecificAlignment - LayerNorm + task-specific affine alignment

    V3 Improvements (--use_* flags):
        use_tsa: Use TaskSpecificAlignment instead of SoftLN
        use_ms_context: Use LightweightMSContext (multi-scale dilated context)
        use_feature_bank: Use FeatureBank for replay-based continual learning
        use_distillation: Use knowledge distillation from teacher model
        use_ewc: Use Elastic Weight Consolidation
        use_hybrid_routing: Use TwoStageHybridRouter with likelihood
        use_adaptive_unfreeze: Unfreeze later blocks for Task > 0
    """
    # Core components
    use_lora: bool = True
    use_router: bool = True
    use_task_adapter: bool = True
    use_pos_embedding: bool = True

    # Training strategies
    use_slow_stage: bool = False
    no_freeze_base: bool = False  # If True, base NF is NOT frozen after Task 0 (for sequential training ablation)

    # Fine-grained controls
    use_task_bias: bool = True
    use_mahalanobis: bool = True

    # ==========================================================================
    # Class-Level Adapter Management
    # ==========================================================================
    # When enabled, creates separate adapters (LoRA, ACL, TaskSpecificAlignment, Prototype)
    # for each class instead of each task. This allows finer-grained learning when
    # multiple classes are grouped into a single task (step).
    #
    # Example with use_class_level_adapters=False (default):
    #   Task 0: [leather, grid, transistor] -> 1 LoRA, 1 ACL, 1 Prototype
    #
    # Example with use_class_level_adapters=True:
    #   Task 0: [leather, grid, transistor] -> 3 LoRAs, 3 ACLs, 3 Prototypes
    use_class_level_adapters: bool = False

    # LoRA configuration (only used if use_lora=True)
    lora_rank: int = 32
    lora_alpha: float = 1.0

    # Adapter mode configuration
    adapter_mode: str = "tsa"  # "soft_ln", "standard", "no_ln_after_task0", "tsa"
    soft_ln_init_scale: float = 0.01  # Initial scale for soft LN (used when adapter_mode="soft_ln")

    # Flow scale(s) regularization
    # Regularizes log_det to reduce variance and stabilize image-level scores
    # loss = nll_loss + lambda_logdet * (logdet_patch ** 2).mean()
    lambda_logdet: float = 3e-5  # Recommended: 3e-5 (stabilizes logdet without hurting performance)

    # Spatial Context Mixing
    # Adds local context to features before NF, so scale(s) can detect local contrast
    use_spatial_context: bool = True
    spatial_context_mode: str = "depthwise_residual"  # "depthwise", "depthwise_residual", "local_stats", "full"
    spatial_context_kernel: int = 3  # Kernel size for context mixing

    # ==========================================================================
    # Scale-specific Context Injection
    # ==========================================================================
    # Injects 3x3 local context into s-network ONLY, keeping t-network context-free
    # - scale(s) sees local contrast → anomaly-sensitive
    # - shift(t) sees x only → density-preserving
    use_scale_context: bool = True
    scale_context_kernel: int = 3    # 3x3 depthwise conv for local context
    scale_context_init_scale: float = 0.1  # Initial alpha (learnable)
    scale_context_max_alpha: float = 0.2   # Upper bound via sigmoid
    # Selective scale context placement: which coupling blocks get scale context
    # "all" = every block (default), "first_only" = block 0 only,
    # "except_last" = all except last block, "first_half" = first half of blocks,
    # "last_half" = last half of blocks, or comma-separated indices e.g. "0,1,2"
    scale_context_blocks: str = "all"

    # ==========================================================================
    # V3 Improvements: New Modules (all disabled by default for backward compat)
    # ==========================================================================

    # Solution 3: Task-Specific Alignment
    # Replaces SoftLN with LayerNorm + task-specific affine alignment
    use_tsa: bool = True

    # Solution 2: Multi-Scale Context (Lightweight dilated convolutions)
    # Expands receptive field from 3×3 to 9×9 with attention-weighted fusion
    use_ms_context: bool = False
    ms_context_dilations: tuple = (1, 2, 4)  # Dilation rates
    ms_context_use_regional: bool = True     # Include regional pooling

    # Solution 1: Feature Bank for Replay
    # Stores representative features from previous tasks
    use_feature_bank: bool = False
    feature_bank_size: int = 500             # Samples per task
    feature_bank_selection: str = 'random'   # 'random', 'kmeans', 'herding'

    # Solution 1: Knowledge Distillation
    # Distill knowledge from teacher (previous task model)
    use_distillation: bool = False
    distillation_mode: str = 'logprob'       # 'logprob', 'latent', 'combined'
    distillation_temperature: float = 2.0
    distillation_weight: float = 1.0         # λ_distill

    # Solution 1: Elastic Weight Consolidation
    # Protects important parameters from previous tasks
    use_ewc: bool = False
    ewc_lambda: float = 1000.0               # EWC regularization strength

    # Solution 1: Adaptive Unfreezing
    # Partially unfreeze base NF for Task > 0 (with EWC protection)
    use_adaptive_unfreeze: bool = False
    adaptive_unfreeze_ratio: float = 0.4     # Ratio of later blocks to unfreeze
    adaptive_unfreeze_lr_ratio: float = 0.1  # LR ratio for unfrozen base params

    # Solution 4: Two-Stage Hybrid Routing
    # Combines prototype and likelihood-based routing
    use_hybrid_routing: bool = False
    hybrid_routing_top_k: int = 2            # Top-K candidates for stage 2
    hybrid_routing_proto_weight: float = 0.6 # Weight for prototype vs likelihood

    # Solution 4: Regional Prototypes
    # Captures spatial structure for better discrimination
    use_regional_prototype: bool = False
    regional_prototype_grid: int = 4         # Grid size (4×4 = 16 regions)

    # ==========================================================================
    # V5 Score Aggregation (Patch → Image)
    # ==========================================================================
    # Controls how patch-level anomaly scores are aggregated to image-level score
    #
    # Modes:
    #   "percentile": Use p-th percentile (default, p=0.99)
    #   "top_k": Average of top-K highest scoring patches
    #   "top_k_percent": Average of top K% highest scoring patches
    #   "max": Maximum patch score
    #   "mean": Mean of all patch scores
    #   "spatial_cluster": Cluster-aware aggregation (V5 improvement)
    score_aggregation_mode: str = "percentile"
    score_aggregation_percentile: float = 0.99    # For "percentile" mode
    score_aggregation_top_k: int = 10             # For "top_k" mode
    score_aggregation_top_k_percent: float = 0.05 # For "top_k_percent" mode (top 5%)

    # ==========================================================================
    # V5 Structural Improvements
    # ==========================================================================

    # --- Phase 1: Loss & Scoring ---
    # Tail-Aware Loss: Train with focus on extreme patches (not just mean)
    use_tail_aware_loss: bool = False
    tail_weight: float = 0.3                      # Weight for tail loss (λ)
    tail_top_k_ratio: float = 0.05                # Top k% patches for tail loss
    # If True and eval uses top_k aggregation, train tail k is forced to eval top_k
    tail_match_eval_topk: bool = False
    # "mixed": (1-w)*mean + w*tail, "tail_only": pure top-k tail loss
    tail_loss_blend_mode: str = "mixed"

    # Spatial Clustering Score: Cluster-aware image score aggregation
    # (enabled via score_aggregation_mode="spatial_cluster")
    cluster_weight: float = 0.5                   # Bonus weight for clustered anomalies
    cluster_high_score_percentile: float = 0.9   # Threshold for high-score region

    # --- Phase 2: Semantic & Context ---
    # Semantic Projector: Position-agnostic semantic feature extraction
    use_semantic_projector: bool = False
    semantic_bottleneck_ratio: float = 0.5        # Bottleneck dim = channels * ratio

    # Task-Adaptive Context: Task-specific adaptation of frozen SpatialMixer
    use_task_adaptive_context: bool = False

    # --- Phase 3: Global Context ---
    # Global Context Module: Long-range dependency via cross-attention
    use_global_context: bool = False
    global_context_regions: int = 4               # Number of regional tokens (R×R)
    global_context_reduction: int = 4             # Channel reduction factor

    # ==========================================================================
    # V5.5: Position-Agnostic Improvements (Rotation Invariance)
    # ==========================================================================
    # These address the fundamental problem: absolute position encoding causes
    # failures on rotation-variant classes (e.g., screw with random rotation)

    # --- Direction 1: Relative Position Encoding ---
    # Replace absolute positional embedding with relative position attention
    use_relative_position: bool = False
    relative_position_max_dist: int = 7           # Max relative distance to consider
    relative_position_num_heads: int = 4          # Number of attention heads

    # --- Direction 2: Dual Branch Scoring ---
    # Two parallel NF branches: with position and without position
    # Learns per-patch α to blend: α*pos_score + (1-α)*nopos_score
    use_dual_branch: bool = False

    # --- Direction 3: Local Consistency Calibration ---
    # Calibrate patch scores based on local consistency
    # Down-weights isolated high scores (likely rotation noise)
    use_local_consistency: bool = False
    local_consistency_kernel: int = 3             # Kernel size for consistency check
    local_consistency_temperature: float = 1.0   # Temperature for weighting

    # ==========================================================================
    # V5.6 Improved Position-Agnostic Solutions
    # ==========================================================================

    # --- Improved Direction 2: Anti-Collapse Dual Branch ---
    # Fixes V5.5 α collapse problem with:
    # 1. α initialized to 0.7 (prefer position branch)
    # 2. α clamped to [min_alpha, max_alpha] to prevent collapse
    # 3. Score difference as additional input
    # 4. Regularization loss to encourage balanced α
    use_improved_dual_branch: bool = False
    dual_branch_init_alpha: float = 0.7           # Initial α value (prefer pos)
    dual_branch_min_alpha: float = 0.3            # Minimum α (ensure pos contribution)
    dual_branch_max_alpha: float = 0.9            # Maximum α (ensure nopos contribution)

    # --- Alternative Direction 2: Score-Guided Branch ---
    # Simpler approach: use score difference to guide α directly
    # α = sigmoid(temp * (bias - score_diff))
    use_score_guided_dual: bool = False
    score_guided_temperature: float = 1.0
    score_guided_min_alpha: float = 0.2

    # --- Improved Direction 3: Multi-Scale Consistency ---
    # Multi-scale consistency check (3x3, 5x5, 7x7)
    # Learnable scale fusion weights
    use_multiscale_consistency: bool = False
    multiscale_kernel_sizes: str = "3,5,7"        # Comma-separated kernel sizes

    # ==========================================================================
    # V5.7 Rotation-Invariant Position Encoding
    # ==========================================================================

    # --- Direction C: Multi-Orientation Ensemble ---
    # Test-time rotation ensemble: apply NF at multiple orientations
    # Take minimum score (most "normal" orientation)
    use_multi_orientation: bool = False
    multi_orientation_n: int = 4                  # Number of orientations (4 = 0,90,180,270)

    # --- Direction D: Content-Based PE ---
    # Replace grid PE with semantic prototype-based PE
    # "Where am I in the object?" instead of "Where am I in the grid?"
    use_content_based_pe: bool = False
    content_pe_n_prototypes: int = 16             # Number of semantic prototypes
    content_pe_temperature: float = 0.1           # Softmax temperature
    content_pe_blend_grid: bool = True            # Blend with grid PE

    # --- Direction E: Hybrid PE ---
    # Per-patch selection between content PE and grid PE
    # Learns when to use rotation-invariant vs position-aware encoding
    use_hybrid_pe: bool = False
    hybrid_pe_n_prototypes: int = 16              # Number of prototypes for content PE

    # ==========================================================================
    # V5.8 Task-Adaptive Position Encoding (TAPE)
    # ==========================================================================
    # 핵심 통찰: Task-level에서 PE 강도를 학습 (patch-level 아님)
    # - 각 Task마다 learnable gate (scalar)
    # - NLL loss가 직접 gradient 제공 → 명확한 학습 신호
    # - Screw: PE ↓ (rotation tolerance), Leather: PE ↑ (spatial consistency)
    use_tape: bool = False
    tape_init_value: float = 0.0                  # sigmoid(0) = 0.5 (50% PE로 시작)
                                                  # 2.0 → ~0.88 (기존 동작에 가깝게)
                                                  # -2.0 → ~0.12 (minimal PE)

    # ==========================================================================
    # V6 Data Augmentation
    # ==========================================================================
    # Random rotation augmentation for rotation-invariant learning
    use_rotation_aug: bool = False
    rotation_degrees: float = 180.0               # ±degrees rotation range

    # ==========================================================================
    # V6.1 Spatial Transformer Network (STN)
    # ==========================================================================
    # Learns to align images to canonical orientation before feature extraction
    # Solves rotation variance problem in classes like Screw
    use_stn: bool = False
    stn_mode: str = 'rotation'                    # 'rotation', 'rotation_scale', 'affine'
    stn_hidden_dim: int = 128                     # Hidden dimension for localization net
    stn_dropout: float = 0.1                      # Dropout for regularization
    stn_rotation_reg_weight: float = 0.01         # Regularization to prevent extreme rotations
    stn_pretrain_epochs: int = 0                  # Optional: pretrain STN before main training

    # ==========================================================================
    # V3 No-Replay Solutions: ACL + OGP
    # ==========================================================================

    # Auxiliary Coupling Layer (ACL)
    # Adds a small task-specific flow after base NF for nonlinear manifold adaptation
    use_acl: bool = True
    acl_n_layers: int = 2                    # Number of coupling layers per ACL (MAIN: 2)
    acl_hidden_ratio: float = 0.5            # Hidden dim = channels * hidden_ratio

    # V45: Per-class ACL layers (JSON dict, e.g. '{"grid":2,"cable":2}')
    # Empty string = use acl_n_layers for all classes. Default fallback = 1 layer.
    per_class_acl_layers: str = ""
    # V45: Learnable gate α in ACL coupling layers
    acl_gate: bool = False
    acl_gate_init: float = 0.0               # Raw sigmoid init (2.0→0.88, -2.0→0.12)
    acl_gate_l1_lambda: float = 0.0          # L1 reg on gate values (0=disabled)
    # V45: ACL-specific weight decay (0=use default optimizer WD)
    acl_weight_decay: float = 0.0
    # V45: Spatial variance preservation loss (0=disabled)
    spatial_var_lambda: float = 0.0

    # V46: Spatial ACL — replace FC subnets with depthwise-separable convolutions
    # 'fc' = original SimpleSubnet (MLP), 'spatial' = SpatialSubnet (depthwise conv)
    acl_subnet_type: str = 'fc'
    acl_kernel_size: int = 3                     # Kernel size for spatial ACL (3 or 5)

    # Subnet depth (number of Linear layers in coupling subnet)
    # 2 = default (Linear→ReLU→Linear), 3 = deeper (Linear→ReLU→Linear→ReLU→Linear)
    subnet_depth: int = 2

    # Subnet hidden dimension ratio: hidden_dim = int(hidden_ratio * dims_in)
    # Default 2.0 matches original hardcoded behavior (hidden_dim = 2 * dims_in)
    subnet_hidden_ratio: float = 2.0

    # Subnet residual connection: adds skip connection from input to output
    # Only works when dims_in == dims_out (true for NF coupling blocks)
    subnet_residual: bool = False

    # Subnet activation function: 'relu' (default), 'gelu', or 'silu'
    # GELU/SiLU provide smoother gradients and may improve fine-grained anomaly detection
    activation_fn: str = 'relu'

    # Subnet LayerNorm: applies LayerNorm after each activation (improves training stability)
    use_layernorm: bool = False

    # Focal NLL: down-weights easy (low-NLL) patches, up-weights hard (high-NLL) patches
    # focal_loss = (nll / nll_max)^gamma * nll  — focuses learning on hard patches
    # 0.0 = disabled (standard NLL), typical values: 0.5, 1.0, 2.0
    focal_gamma: float = 0.0

    # Score map smoothing sigma (Gaussian blur applied to anomaly score map at eval)
    # Default 4.0 matches original hardcoded behavior
    score_smooth_sigma: float = 4.0

    # Multi-task loss: combines pixel-level and image-level supervision
    # Pixel-level: standard NLL on patches (fine-grained localization)
    # Image-level: average NLL per image (holistic anomaly scoring)
    # total_loss = alpha * pixel_loss + beta * image_loss
    use_multitask_loss: bool = False
    multitask_alpha: float = 0.7  # Weight for pixel-level loss
    multitask_beta: float = 0.3   # Weight for image-level loss

    # Nonlinear LoRA (Sinter activation between A and B matrices)
    # Adds nonlinear task adaptation: delta_W(x) = B @ sinter(A @ x)
    use_nonlinear_lora: bool = False
    nonlinear_lora_alpha: float = 0.5          # Sinter amplitude: x + alpha * sin(x)

    # Per-Task Latent Affine (applied after base NF, before ACL if both enabled)
    # Learnable scale+shift per task in latent space with exact log-determinant
    use_latent_affine: bool = False

    # Orthogonal Gradient Projection (OGP)
    # Projects gradients to null space of important subspaces from previous tasks
    use_ogp: bool = False
    ogp_threshold: float = 0.99              # Cumulative variance threshold for SVD
    ogp_max_rank: int = 50                   # Maximum rank per task per parameter
    ogp_n_samples: int = 300                 # Samples for gradient collection

    # ==========================================================================
    # V6 Ablation Experiments
    # ==========================================================================

    # Experiment 1: No LoRA - Use regular Linear layers instead of LoRA
    # Each task has its own complete Linear layers (no low-rank constraint)
    use_regular_linear: bool = False

    # Experiment 2: Task-Separated Training
    # Instead of freezing base after Task 0, train complete NF for each task
    # No parameter sharing between tasks (upper bound experiment)
    use_task_separated: bool = False

    # Experiment 3: Spectral Normalization
    # Apply spectral norm to subnet layers for Lipschitz constraint
    use_spectral_norm: bool = False

    # ==========================================================================
    # V3 Task-Conditioned Multi-Scale Context (Fundamental Solution)
    # ==========================================================================
    # Replaces LightweightMSContext with proper task-specific adaptation
    # - Shared base (dilated convs, regional_proj base) frozen after Task 0
    # - Task-specific components: scale_attention, regional LoRA, fusion_gate
    # This solves the catastrophic forgetting issue of the original MSContext
    use_task_conditioned_ms_context: bool = False
    tc_ms_context_dilations: tuple = (1, 2, 4)  # Dilation rates for multi-scale
    tc_ms_context_use_regional: bool = True     # Use regional context pooling
    tc_ms_context_lora_rank: int = 16           # LoRA rank for regional_proj

    # ==========================================================================
    # V7 Denoising Score Matching (DSM)
    # ==========================================================================
    # Combines NF's conservative field with MULDE-style score matching
    # Key insight: NF guarantees conservative vector field (curl=0) for score
    #
    # DSM Loss: L_DSM = E[||∇_x log p(x_noisy) + ε/σ||²]
    # Hybrid:   L = α*L_NLL + (1-α)*L_DSM
    #
    # Benefits:
    # - Improved OOD robustness (learns score on manifold + surroundings)
    # - More consistent anomaly scores across tasks
    # - Better handling of hard classes (screw, capsule)
    use_dsm: bool = False
    dsm_mode: str = "hybrid"              # "dsm_only", "nll_only", "hybrid"
    dsm_alpha: float = 0.5                # NLL weight in hybrid: α*NLL + (1-α)*DSM
    dsm_sigma_min: float = 0.01           # Minimum noise scale
    dsm_sigma_max: float = 1.0            # Maximum noise scale
    dsm_n_projections: int = 1            # SSM projections (1 is usually enough)
    dsm_use_sliced: bool = True           # Use Sliced Score Matching (efficient)
    dsm_noise_mode: str = "geometric"     # "geometric" (LogUniform), "uniform", "fixed"
    dsm_clean_penalty: float = 0.0        # MULDE-style clean data penalty weight

    def __post_init__(self):
        """Validate configuration."""
        if not self.use_lora and self.use_task_bias:
            # Task bias requires LoRA
            self.use_task_bias = False

        # V3: If tsa adapter is enabled, update adapter_mode
        if self.use_tsa:
            self.adapter_mode = "tsa"

        # V3: TaskSpecificAlignment + MS-Context 조합 충돌 방지
        # 두 모듈 모두 입력 분포를 변환하여 함께 사용시 학습 불안정 발생
        if self.use_tsa and self.use_ms_context:
            print("⚠️  Warning: use_tsa + use_ms_context 조합은 학습 불안정을 유발할 수 있습니다.")
            print("   → MS-Context 비활성화 (scale_context가 이미 s-network에 local context 제공)")
            self.use_ms_context = False

        # V3: Adaptive unfreeze requires EWC or distillation for stability
        if self.use_adaptive_unfreeze and not (self.use_ewc or self.use_distillation or self.use_feature_bank):
            print("⚠️  Warning: use_adaptive_unfreeze without EWC/distillation/feature_bank may cause forgetting")

        # V7: DSM auto-disables ACL (Flow itself acts as density function)
        if self.use_dsm and self.use_acl:
            print("⚠️  Warning: DSM enabled → ACL auto-disabled (Flow is the density function)")
            self.use_acl = False

    def get_active_components(self) -> list:
        """Return list of active component names."""
        components = []
        if self.use_lora:
            components.append("LoRA")
        if self.use_router:
            components.append("Router")
        if self.use_task_adapter:
            components.append("TaskAdapter")
        if self.use_pos_embedding:
            components.append("PosEmbed")
        if self.use_slow_stage:
            components.append("SlowStage")
        if self.use_task_bias:
            components.append("TaskBias")
        if self.use_mahalanobis:
            components.append("Mahalanobis")

        # V3 components
        if self.use_tsa:
            components.append("TaskSpecificAlignment")
        if self.use_ms_context:
            components.append("MSContext")
        if self.use_feature_bank:
            components.append("FeatureBank")
        if self.use_distillation:
            components.append("Distillation")
        if self.use_ewc:
            components.append("EWC")
        if self.use_adaptive_unfreeze:
            components.append("AdaptiveUnfreeze")
        if self.use_hybrid_routing:
            components.append("HybridRouting")
        if self.use_regional_prototype:
            components.append("RegionalProto")
        if self.use_acl:
            components.append("ACL")
        if self.use_ogp:
            components.append("OGP")
        if self.use_task_conditioned_ms_context:
            components.append("TCMSContext")
        if self.use_dsm:
            components.append("DSM")

        return components

    def get_disabled_components(self) -> list:
        """Return list of disabled component names."""
        all_components = {
            "LoRA": self.use_lora,
            "Router": self.use_router,
            "TaskAdapter": self.use_task_adapter,
            "PosEmbed": self.use_pos_embedding,
            "SlowStage": self.use_slow_stage,
            "TaskBias": self.use_task_bias,
            "Mahalanobis": self.use_mahalanobis,
        }
        return [name for name, enabled in all_components.items() if not enabled]

    def get_experiment_name(self) -> str:
        """Generate experiment name based on disabled components and adapter mode."""
        disabled = self.get_disabled_components()
        base_name = "full_model" if not disabled else "wo_" + "_".join([c.lower() for c in disabled])

        # V3 components - add as suffixes when enabled
        v3_parts = []
        if self.use_tsa:
            v3_parts.append("tsa")
        if self.use_ms_context:
            v3_parts.append("msctx")
        if self.use_feature_bank:
            v3_parts.append("fbank")
        if self.use_distillation:
            v3_parts.append("distill")
        if self.use_ewc:
            v3_parts.append("ewc")
        if self.use_adaptive_unfreeze:
            v3_parts.append("unfreeze")
        if self.use_hybrid_routing:
            v3_parts.append("hybrid")
        if self.use_regional_prototype:
            v3_parts.append("regional")
        if self.use_acl:
            v3_parts.append("acl")
        if self.use_ogp:
            v3_parts.append("ogp")
        if self.use_task_conditioned_ms_context:
            v3_parts.append("tcmsctx")
        if self.use_dsm:
            v3_parts.append(f"dsm_{self.dsm_mode}_a{self.dsm_alpha}")

        if v3_parts:
            base_name += "_v3_" + "_".join(v3_parts)
        else:
            # Add adapter mode suffix if not soft_ln (default) and no V3 components
            if self.adapter_mode != "soft_ln":
                base_name += f"_{self.adapter_mode}"
            else:
                base_name += f"_soft_ln_s{self.soft_ln_init_scale}"

        # Add logdet regularization suffix if enabled
        if self.lambda_logdet > 0:
            base_name += f"_ldet{self.lambda_logdet:.0e}".replace("-0", "-").replace("+0", "")

        # Add spatial context suffix if enabled (and not using ms_context which supersedes it)
        if self.use_spatial_context and not self.use_ms_context:
            base_name += f"_sctx_{self.spatial_context_mode}"

        # Add scale context suffix if enabled
        if self.use_scale_context:
            base_name += f"_scalectx_k{self.scale_context_kernel}"

        return base_name

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = ["AblationConfig:"]
        lines.append(f"  Active: {', '.join(self.get_active_components())}")
        disabled = self.get_disabled_components()
        if disabled:
            lines.append(f"  Disabled: {', '.join(disabled)}")
        if self.use_task_adapter:
            lines.append(f"  Adapter Mode: {self.adapter_mode}")
            if self.adapter_mode == "soft_ln":
                lines.append(f"    SoftLN Init Scale: {self.soft_ln_init_scale}")
        if self.lambda_logdet > 0:
            lines.append(f"  Logdet Regularization: {self.lambda_logdet:.2e}")
        if self.use_spatial_context:
            lines.append(f"  Spatial Context: {self.spatial_context_mode} (kernel={self.spatial_context_kernel})")
        if self.use_scale_context:
            lines.append(f"  Scale Context: s-net only (kernel={self.scale_context_kernel}, init_alpha={self.scale_context_init_scale}, max_alpha={self.scale_context_max_alpha})")
        return "\n".join(lines)


# Predefined ablation configurations for common experiments
ABLATION_PRESETS = {
    "full": AblationConfig(),

    # Single component ablations
    "wo_lora": AblationConfig(use_lora=False),
    "wo_router": AblationConfig(use_router=False),
    "wo_adapter": AblationConfig(use_task_adapter=False),
    "wo_pos_embed": AblationConfig(use_pos_embedding=False),

    # Combined ablations
    "wo_router_adapter": AblationConfig(use_router=False, use_task_adapter=False),

    # Minimal baselines
    "baseline_nf": AblationConfig(
        use_lora=False,
        use_router=False,
        use_task_adapter=False,
        use_pos_embedding=False,
    ),
    "lora_only": AblationConfig(
        use_router=False,
        use_task_adapter=False,
    ),
}


def get_ablation_config(preset_name: str = None, **kwargs) -> AblationConfig:
    """
    Get ablation configuration.

    Args:
        preset_name: Name of preset configuration (see ABLATION_PRESETS)
        **kwargs: Override specific settings

    Returns:
        AblationConfig instance
    """
    if preset_name and preset_name in ABLATION_PRESETS:
        config = ABLATION_PRESETS[preset_name]
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    return AblationConfig(**kwargs)


def add_ablation_args(parser):
    """
    Add ablation arguments to argument parser.

    Args:
        parser: argparse.ArgumentParser instance

    Returns:
        parser with added arguments
    """
    ablation_group = parser.add_argument_group('Ablation Settings')

    ablation_group.add_argument(
        '--ablation_preset', type=str, default=None,
        choices=list(ABLATION_PRESETS.keys()),
        help='Use a predefined ablation configuration'
    )

    # Core components
    ablation_group.add_argument(
        '--no_lora', action='store_true',
        help='Disable LoRA adaptation (use shared NF weights)'
    )
    ablation_group.add_argument(
        '--no_router', action='store_true',
        help='Disable prototype router (use oracle task_id)'
    )
    ablation_group.add_argument(
        '--no_task_adapter', action='store_true',
        help='Disable task input adapter (instance normalization)'
    )
    ablation_group.add_argument(
        '--no_pos_embedding', action='store_true',
        help='Disable positional embedding'
    )

    # Training strategy
    ablation_group.add_argument(
        '--no_freeze_base', action='store_true',
        help='Do NOT freeze base NF after Task 0 (for sequential training ablation)'
    )

    # Fine-grained controls
    ablation_group.add_argument(
        '--no_task_bias', action='store_true',
        help='Disable task-specific bias in LoRA layers'
    )
    ablation_group.add_argument(
        '--no_mahalanobis', action='store_true',
        help='Use Euclidean distance instead of Mahalanobis in router'
    )

    # Class-level adapter management
    ablation_group.add_argument(
        '--use_class_level_adapters', action='store_true',
        help='Create separate adapters (LoRA, ACL, Prototype) per class instead of per task'
    )

    # Adapter mode options
    ablation_group.add_argument(
        '--adapter_mode', type=str, default='soft_ln',
        choices=['soft_ln', 'standard', 'no_ln_after_task0', 'no_ln', 'tsa_no_ln'],
        help='Adapter mode: soft_ln (default), standard (FiLM), no_ln, tsa_no_ln (tsa structure without LN)'
    )
    ablation_group.add_argument(
        '--soft_ln_init_scale', type=float, default=0.01,
        help='Initial scale for soft LN (used with --adapter_mode soft_ln)'
    )

    # Flow regularization
    ablation_group.add_argument(
        '--lambda_logdet', type=float, default=1e-5,
        help='Logdet L2 regularization weight (0=disabled, recommended: 1e-5 ~ 1e-4)'
    )

    # Spatial context mixing (enabled by default)
    ablation_group.add_argument(
        '--no_spatial_context', action='store_true',
        help='Disable spatial context mixing before NF'
    )
    ablation_group.add_argument(
        '--spatial_context_mode', type=str, default='depthwise_residual',
        choices=['depthwise', 'depthwise_residual', 'local_stats', 'full'],
        help='Spatial context mixing mode'
    )
    ablation_group.add_argument(
        '--spatial_context_kernel', type=int, default=3,
        help='Kernel size for spatial context mixing (default: 3)'
    )

    # Scale-specific context injection (enabled by default)
    ablation_group.add_argument(
        '--no_scale_context', action='store_true',
        help='Disable scale context injection'
    )
    ablation_group.add_argument(
        '--scale_context_kernel', type=int, default=3,
        help='Kernel size for scale context extraction (default: 3)'
    )
    ablation_group.add_argument(
        '--scale_context_init_scale', type=float, default=0.1,
        help='Initial alpha for context scaling (learnable, default: 0.1)'
    )
    ablation_group.add_argument(
        '--scale_context_max_alpha', type=float, default=0.2,
        help='Maximum alpha for context scaling (sigmoid upper bound, default: 0.2)'
    )
    ablation_group.add_argument(
        '--scale_context_blocks', type=str, default='all',
        help='Which blocks get scale context: "all", "first_only", "except_last", '
             '"first_half", "last_half", or comma-separated indices e.g. "0,1,2"'
    )

    # =========================================================================
    # V3 Improvements (--use_* flags for ablation study)
    # =========================================================================
    v3_group = parser.add_argument_group('V3 Improvements (Ablation Study)')

    # Solution 3: TSA Adapter
    v3_group.add_argument(
        '--use_tsa', action='store_true',
        help='[V3] Use TaskSpecificAlignment instead of SoftLN (Solution 3)'
    )
    v3_group.add_argument(
        '--no_tsa', action='store_true',
        help='[V3] Disable TaskSpecificAlignment (use SoftLN instead) - for ablation study'
    )

    # Solution 2: Multi-Scale Context
    v3_group.add_argument(
        '--use_ms_context', action='store_true',
        help='[V3] Use LightweightMSContext with dilated convolutions (Solution 2)'
    )
    v3_group.add_argument(
        '--ms_context_dilations', type=str, default='1,2,4',
        help='Dilation rates for MS context (comma-separated, default: 1,2,4)'
    )

    # Solution 1: Feature Bank
    v3_group.add_argument(
        '--use_feature_bank', action='store_true',
        help='[V3] Use FeatureBank for replay-based continual learning (Solution 1)'
    )
    v3_group.add_argument(
        '--feature_bank_size', type=int, default=500,
        help='Number of features to store per task (default: 500)'
    )
    v3_group.add_argument(
        '--feature_bank_selection', type=str, default='random',
        choices=['random', 'kmeans', 'herding'],
        help='Feature selection method for bank (default: random)'
    )

    # Solution 1: Distillation
    v3_group.add_argument(
        '--use_distillation', action='store_true',
        help='[V3] Use knowledge distillation from teacher model (Solution 1)'
    )
    v3_group.add_argument(
        '--distillation_mode', type=str, default='logprob',
        choices=['logprob', 'latent', 'combined'],
        help='Distillation mode (default: logprob)'
    )
    v3_group.add_argument(
        '--distillation_weight', type=float, default=1.0,
        help='Weight for distillation loss (default: 1.0)'
    )

    # Solution 1: EWC
    v3_group.add_argument(
        '--use_ewc', action='store_true',
        help='[V3] Use Elastic Weight Consolidation (Solution 1)'
    )
    v3_group.add_argument(
        '--ewc_lambda', type=float, default=1000.0,
        help='EWC regularization strength (default: 1000)'
    )

    # Solution 1: Adaptive Unfreezing
    v3_group.add_argument(
        '--use_adaptive_unfreeze', action='store_true',
        help='[V3] Partially unfreeze base NF for Task > 0 (Solution 1)'
    )
    v3_group.add_argument(
        '--adaptive_unfreeze_ratio', type=float, default=0.4,
        help='Ratio of later blocks to unfreeze (default: 0.4)'
    )

    # Solution 4: Hybrid Routing
    v3_group.add_argument(
        '--use_hybrid_routing', action='store_true',
        help='[V3] Use Two-Stage Hybrid Router with likelihood (Solution 4)'
    )
    v3_group.add_argument(
        '--hybrid_routing_top_k', type=int, default=2,
        help='Top-K candidates for stage 2 routing (default: 2)'
    )

    # Solution 4: Regional Prototype
    v3_group.add_argument(
        '--use_regional_prototype', action='store_true',
        help='[V3] Use regional prototypes for spatial structure (Solution 4)'
    )
    v3_group.add_argument(
        '--regional_prototype_grid', type=int, default=4,
        help='Grid size for regional prototypes (default: 4 → 16 regions)'
    )

    # =========================================================================
    # V3 No-Replay Solutions: ACL + OGP
    # =========================================================================
    noreplay_group = parser.add_argument_group('V3 No-Replay Solutions (ACL/OGP)')

    # Auxiliary Coupling Layer (ACL)
    noreplay_group.add_argument(
        '--use_acl', action='store_true',
        help='[V3] Use Auxiliary Coupling Layer for nonlinear manifold adaptation'
    )
    noreplay_group.add_argument(
        '--no_acl', action='store_true',
        help='[V3] Disable Auxiliary Coupling Layer - for ablation study'
    )
    noreplay_group.add_argument(
        '--acl_n_layers', type=int, default=2,
        help='Number of coupling layers per ACL (default: 2)'
    )
    noreplay_group.add_argument(
        '--acl_hidden_ratio', type=float, default=0.5,
        help='Hidden dim ratio for ACL (default: 0.5)'
    )
    # V45: ACL gating and regularization
    noreplay_group.add_argument(
        '--per_class_acl_layers', type=str, default='',
        help='JSON dict for per-class ACL layers, e.g. \'{"grid":2,"cable":2}\' (empty=use acl_n_layers)'
    )
    noreplay_group.add_argument(
        '--acl_gate', action='store_true',
        help='Enable learnable gate α in ACL coupling layers'
    )
    noreplay_group.add_argument(
        '--acl_gate_init', type=float, default=0.0,
        help='Raw sigmoid init for ACL gate (2.0→0.88, -2.0→0.12, default: 0.0)'
    )
    noreplay_group.add_argument(
        '--acl_gate_l1_lambda', type=float, default=0.0,
        help='L1 regularization lambda on ACL gate values (0=disabled)'
    )
    noreplay_group.add_argument(
        '--acl_weight_decay', type=float, default=0.0,
        help='Extra weight decay for ACL params only (0=use default optimizer WD)'
    )
    noreplay_group.add_argument(
        '--spatial_var_lambda', type=float, default=0.0,
        help='Spatial variance preservation loss lambda (0=disabled)'
    )
    # V46: Spatial ACL
    noreplay_group.add_argument(
        '--acl_subnet_type', type=str, default='fc', choices=['fc', 'spatial'],
        help='[V46] ACL subnet type: fc (MLP) or spatial (depthwise conv)'
    )
    noreplay_group.add_argument(
        '--acl_kernel_size', type=int, default=3,
        help='[V46] Kernel size for spatial ACL (default: 3)'
    )

    # Subnet depth
    noreplay_group.add_argument(
        '--subnet_depth', type=int, default=2,
        help='Number of Linear layers in coupling subnet (2=default, 3=deeper like ACL)'
    )
    noreplay_group.add_argument(
        '--subnet_hidden_ratio', type=float, default=2.0,
        help='Subnet hidden dim ratio: hidden_dim = ratio * dims_in (default: 2.0)'
    )
    noreplay_group.add_argument(
        '--subnet_residual', action='store_true',
        help='Add residual (skip) connection in subnet'
    )
    noreplay_group.add_argument(
        '--activation_fn', type=str, default='relu', choices=['relu', 'gelu', 'silu'],
        help='Activation function for subnet (default: relu)'
    )
    noreplay_group.add_argument(
        '--use_layernorm', action='store_true',
        help='Apply LayerNorm after each activation in subnet'
    )
    noreplay_group.add_argument(
        '--focal_gamma', type=float, default=0.0,
        help='Focal NLL gamma (0=disabled, typical: 0.5~2.0)'
    )
    noreplay_group.add_argument(
        '--score_smooth_sigma', type=float, default=4.0,
        help='Gaussian smoothing sigma for score map (default: 4.0)'
    )
    noreplay_group.add_argument(
        '--interp_mode', type=str, default='bicubic',
        choices=['bicubic', 'bilinear', 'nearest'],
        help='Interpolation mode for upsampling anomaly maps (default: bicubic)'
    )
    noreplay_group.add_argument(
        '--use_multitask_loss', action='store_true',
        help='Use multi-task loss (pixel-level + image-level)'
    )
    noreplay_group.add_argument(
        '--multitask_alpha', type=float, default=0.7,
        help='Weight for pixel-level loss in multi-task learning (default: 0.7)'
    )
    noreplay_group.add_argument(
        '--multitask_beta', type=float, default=0.3,
        help='Weight for image-level loss in multi-task learning (default: 0.3)'
    )

    # Nonlinear LoRA (Sinter activation)
    noreplay_group.add_argument(
        '--use_nonlinear_lora', action='store_true',
        help='Use Sinter nonlinear activation in LoRA (x + alpha*sin(x) between A and B)'
    )
    noreplay_group.add_argument(
        '--nonlinear_lora_alpha', type=float, default=0.5,
        help='Sinter amplitude for nonlinear LoRA (default: 0.5)'
    )

    # Per-Task Latent Affine
    noreplay_group.add_argument(
        '--use_latent_affine', action='store_true',
        help='Use per-task learnable affine transform on latent z after base NF'
    )

    # Orthogonal Gradient Projection (OGP)
    noreplay_group.add_argument(
        '--use_ogp', action='store_true',
        help='[V3] Use Orthogonal Gradient Projection for gradient-based protection'
    )
    noreplay_group.add_argument(
        '--ogp_threshold', type=float, default=0.99,
        help='Cumulative variance threshold for OGP SVD (default: 0.99)'
    )
    noreplay_group.add_argument(
        '--ogp_max_rank', type=int, default=50,
        help='Maximum rank per task per parameter (default: 50)'
    )
    noreplay_group.add_argument(
        '--ogp_n_samples', type=int, default=300,
        help='Number of samples for OGP gradient collection (default: 300)'
    )

    # =========================================================================
    # V3 Task-Conditioned Multi-Scale Context (Fundamental Solution)
    # =========================================================================
    tcms_group = parser.add_argument_group('V3 Task-Conditioned MS Context')

    tcms_group.add_argument(
        '--use_task_conditioned_ms_context', action='store_true',
        help='[V3] Use TaskConditionedMSContext (fixes MSContext forgetting issue)'
    )
    tcms_group.add_argument(
        '--tc_ms_context_dilations', type=str, default='1,2,4',
        help='Dilation rates for TC-MSContext (comma-separated, default: 1,2,4)'
    )
    tcms_group.add_argument(
        '--tc_ms_context_use_regional', action='store_true', default=True,
        help='Use regional context in TC-MSContext (default: True)'
    )
    tcms_group.add_argument(
        '--tc_ms_context_no_regional', action='store_true',
        help='Disable regional context in TC-MSContext'
    )
    tcms_group.add_argument(
        '--tc_ms_context_lora_rank', type=int, default=16,
        help='LoRA rank for regional_proj in TC-MSContext (default: 16)'
    )

    # =========================================================================
    # V5 Score Aggregation (Patch → Image)
    # =========================================================================
    score_group = parser.add_argument_group('V5 Score Aggregation')

    score_group.add_argument(
        '--score_aggregation_mode', type=str, default='percentile',
        choices=['percentile', 'top_k', 'top_k_percent', 'max', 'mean', 'spatial_cluster'],
        help='[V5] Score aggregation mode (default: percentile)'
    )
    score_group.add_argument(
        '--score_aggregation_percentile', type=float, default=0.99,
        help='Percentile for percentile mode (default: 0.99)'
    )
    score_group.add_argument(
        '--score_aggregation_top_k', type=int, default=10,
        help='K value for top_k mode (default: 10)'
    )
    score_group.add_argument(
        '--score_aggregation_top_k_percent', type=float, default=0.05,
        help='Percentage for top_k_percent mode (default: 0.05 = top 5%%)'
    )

    # =========================================================================
    # V5 Structural Improvements
    # =========================================================================
    v5_group = parser.add_argument_group('V5 Structural Improvements')

    # Phase 1: Tail-Aware Loss
    v5_group.add_argument(
        '--use_tail_aware_loss', action='store_true',
        help='[V5] Use tail-aware loss for training (Phase 1)'
    )
    v5_group.add_argument(
        '--tail_weight', type=float, default=0.3,
        help='Weight for tail loss component (default: 0.3)'
    )
    v5_group.add_argument(
        '--tail_top_k_ratio', type=float, default=0.05,
        help='Ratio of top patches for tail loss (default: 0.05 = top 5%%)'
    )
    v5_group.add_argument(
        '--tail_match_eval_topk', action='store_true',
        help='If eval mode is top_k, force train tail k to match score_aggregation_top_k'
    )
    v5_group.add_argument(
        '--tail_loss_blend_mode', type=str, default='mixed',
        choices=['mixed', 'tail_only'],
        help='Tail-aware loss blend mode: mixed or tail_only (default: mixed)'
    )

    # Phase 1: Spatial Clustering Score
    v5_group.add_argument(
        '--cluster_weight', type=float, default=0.5,
        help='Bonus weight for clustered anomalies (default: 0.5)'
    )
    v5_group.add_argument(
        '--cluster_high_score_percentile', type=float, default=0.9,
        help='Percentile threshold for high-score regions (default: 0.9)'
    )

    # Phase 2: Semantic Projector
    v5_group.add_argument(
        '--use_semantic_projector', action='store_true',
        help='[V5] Use semantic projector for position-agnostic features (Phase 2)'
    )
    v5_group.add_argument(
        '--semantic_bottleneck_ratio', type=float, default=0.5,
        help='Bottleneck ratio for semantic projector (default: 0.5)'
    )

    # Phase 2: Task-Adaptive Context
    v5_group.add_argument(
        '--use_task_adaptive_context', action='store_true',
        help='[V5] Use task-adaptive context mixer (Phase 2)'
    )

    # Phase 3: Global Context
    v5_group.add_argument(
        '--use_global_context', action='store_true',
        help='[V5] Use global context module for long-range dependency (Phase 3)'
    )
    v5_group.add_argument(
        '--global_context_regions', type=int, default=4,
        help='Number of regional tokens for global context (default: 4)'
    )
    v5_group.add_argument(
        '--global_context_reduction', type=int, default=4,
        help='Channel reduction factor for global context (default: 4)'
    )

    # =========================================================================
    # V5.5: Position-Agnostic Improvements (Rotation Invariance)
    # =========================================================================
    v55_group = parser.add_argument_group('V5.5 Position-Agnostic Improvements')

    # Direction 1: Relative Position Encoding
    v55_group.add_argument(
        '--use_relative_position', action='store_true',
        help='[V5.5] Use relative position encoding instead of absolute (Direction 1)'
    )
    v55_group.add_argument(
        '--relative_position_max_dist', type=int, default=7,
        help='Max relative distance for position encoding (default: 7)'
    )
    v55_group.add_argument(
        '--relative_position_num_heads', type=int, default=4,
        help='Number of attention heads for relative position (default: 4)'
    )

    # Direction 2: Dual Branch Scoring
    v55_group.add_argument(
        '--use_dual_branch', action='store_true',
        help='[V5.5] Use dual branch scoring (pos + nopos) (Direction 2)'
    )

    # Direction 3: Local Consistency Calibration
    v55_group.add_argument(
        '--use_local_consistency', action='store_true',
        help='[V5.5] Use local consistency score calibration (Direction 3)'
    )
    v55_group.add_argument(
        '--local_consistency_kernel', type=int, default=3,
        help='Kernel size for local consistency check (default: 3)'
    )
    v55_group.add_argument(
        '--local_consistency_temperature', type=float, default=1.0,
        help='Temperature for consistency weighting (default: 1.0)'
    )

    # =========================================================================
    # V5.6: Improved Position-Agnostic Solutions
    # =========================================================================
    v56_group = parser.add_argument_group('V5.6 Improved Position-Agnostic')

    # Improved Direction 2: Anti-Collapse Dual Branch
    v56_group.add_argument(
        '--use_improved_dual_branch', action='store_true',
        help='[V5.6] Use improved dual branch with anti-collapse mechanism'
    )
    v56_group.add_argument(
        '--dual_branch_init_alpha', type=float, default=0.7,
        help='Initial alpha value for dual branch (default: 0.7, prefer pos)'
    )
    v56_group.add_argument(
        '--dual_branch_min_alpha', type=float, default=0.3,
        help='Minimum alpha to prevent collapse (default: 0.3)'
    )
    v56_group.add_argument(
        '--dual_branch_max_alpha', type=float, default=0.9,
        help='Maximum alpha (default: 0.9)'
    )

    # Alternative Direction 2: Score-Guided Branch
    v56_group.add_argument(
        '--use_score_guided_dual', action='store_true',
        help='[V5.6] Use score-guided dual branch (simpler alternative)'
    )
    v56_group.add_argument(
        '--score_guided_temperature', type=float, default=1.0,
        help='Temperature for score-guided alpha (default: 1.0)'
    )
    v56_group.add_argument(
        '--score_guided_min_alpha', type=float, default=0.2,
        help='Minimum alpha for score-guided (default: 0.2)'
    )

    # Improved Direction 3: Multi-Scale Consistency
    v56_group.add_argument(
        '--use_multiscale_consistency', action='store_true',
        help='[V5.6] Use multi-scale local consistency (3x3, 5x5, 7x7)'
    )
    v56_group.add_argument(
        '--multiscale_kernel_sizes', type=str, default='3,5,7',
        help='Comma-separated kernel sizes for multi-scale (default: 3,5,7)'
    )

    # =========================================================================
    # V5.7: Rotation-Invariant Position Encoding
    # =========================================================================
    v57_group = parser.add_argument_group('V5.7 Rotation-Invariant PE')

    # Direction C: Multi-Orientation Ensemble
    v57_group.add_argument(
        '--use_multi_orientation', action='store_true',
        help='[V5.7] Use multi-orientation ensemble (test-time rotation)'
    )
    v57_group.add_argument(
        '--multi_orientation_n', type=int, default=4,
        help='Number of orientations (default: 4 = 0,90,180,270)'
    )

    # Direction D: Content-Based PE
    v57_group.add_argument(
        '--use_content_based_pe', action='store_true',
        help='[V5.7] Use content-based positional embedding'
    )
    v57_group.add_argument(
        '--content_pe_n_prototypes', type=int, default=16,
        help='Number of semantic prototypes (default: 16)'
    )
    v57_group.add_argument(
        '--content_pe_temperature', type=float, default=0.1,
        help='Softmax temperature for prototype matching (default: 0.1)'
    )
    v57_group.add_argument(
        '--content_pe_blend_grid', action='store_true', default=True,
        help='Blend content PE with grid PE (default: True)'
    )
    v57_group.add_argument(
        '--no_content_pe_blend_grid', action='store_false', dest='content_pe_blend_grid',
        help='Do not blend content PE with grid PE'
    )

    # Direction E: Hybrid PE
    v57_group.add_argument(
        '--use_hybrid_pe', action='store_true',
        help='[V5.7] Use hybrid PE (per-patch content vs grid selection)'
    )
    v57_group.add_argument(
        '--hybrid_pe_n_prototypes', type=int, default=16,
        help='Number of prototypes for hybrid PE (default: 16)'
    )

    # V5.8 Task-Adaptive Position Encoding (TAPE)
    v58_group = parser.add_argument_group('V5.8 TAPE (Task-Adaptive PE)')
    v58_group.add_argument(
        '--use_tape', action='store_true',
        help='[V5.8] Use Task-Adaptive Position Encoding (learns optimal PE strength per task)'
    )
    v58_group.add_argument(
        '--tape_init_value', type=float, default=0.0,
        help='TAPE gate init value (sigmoid applied). 0=50%%, 2=88%%, -2=12%% (default: 0.0)'
    )

    # V6 Data Augmentation
    v6_group = parser.add_argument_group('V6 Data Augmentation')
    v6_group.add_argument(
        '--use_rotation_aug', action='store_true',
        help='[V6] Use random rotation augmentation during training'
    )
    v6_group.add_argument(
        '--rotation_degrees', type=float, default=180.0,
        help='Rotation range in degrees (default: 180 for ±180°)'
    )

    # V6.1: Spatial Transformer Network (STN)
    stn_group = parser.add_argument_group('V6.1 Spatial Transformer Network')
    stn_group.add_argument(
        '--use_stn', action='store_true',
        help='[V6.1] Use Spatial Transformer Network for automatic image alignment'
    )
    stn_group.add_argument(
        '--stn_mode', type=str, default='rotation',
        choices=['rotation', 'rotation_scale', 'affine'],
        help='STN transformation mode (default: rotation)'
    )
    stn_group.add_argument(
        '--stn_hidden_dim', type=int, default=128,
        help='Hidden dimension for STN localization network (default: 128)'
    )
    stn_group.add_argument(
        '--stn_dropout', type=float, default=0.1,
        help='Dropout rate for STN (default: 0.1)'
    )
    stn_group.add_argument(
        '--stn_rotation_reg_weight', type=float, default=0.01,
        help='Regularization weight for rotation angle (default: 0.01)'
    )
    stn_group.add_argument(
        '--stn_pretrain_epochs', type=int, default=0,
        help='Number of epochs to pretrain STN before main training (default: 0)'
    )

    # =========================================================================
    # V6 Ablation Experiments
    # =========================================================================
    v7_group = parser.add_argument_group('V6 Ablation Experiments')

    v7_group.add_argument(
        '--use_regular_linear', action='store_true',
        help='[V6-Exp1] Use regular Linear layers instead of LoRA'
    )
    v7_group.add_argument(
        '--use_task_separated', action='store_true',
        help='[V6-Exp2] Train separate NF for each task (no base sharing)'
    )
    v7_group.add_argument(
        '--use_spectral_norm', action='store_true',
        help='[V6-Exp3] Apply spectral normalization to subnet layers'
    )

    # =========================================================================
    # V7 Denoising Score Matching (DSM)
    # =========================================================================
    dsm_group = parser.add_argument_group('V7 Denoising Score Matching (DSM)')

    dsm_group.add_argument(
        '--use_dsm', action='store_true',
        help='[V7] Enable Denoising Score Matching loss (combines NF with score matching)'
    )
    dsm_group.add_argument(
        '--dsm_mode', type=str, default='hybrid',
        choices=['dsm_only', 'nll_only', 'hybrid'],
        help='[V7] DSM training mode (default: hybrid)'
    )
    dsm_group.add_argument(
        '--dsm_alpha', type=float, default=0.5,
        help='[V7] Hybrid loss weight: alpha*NLL + (1-alpha)*DSM (default: 0.5)'
    )
    dsm_group.add_argument(
        '--dsm_sigma_min', type=float, default=0.01,
        help='[V7] Minimum noise scale (default: 0.01)'
    )
    dsm_group.add_argument(
        '--dsm_sigma_max', type=float, default=1.0,
        help='[V7] Maximum noise scale (default: 1.0)'
    )
    dsm_group.add_argument(
        '--dsm_n_projections', type=int, default=1,
        help='[V7] Number of SSM projections (default: 1)'
    )
    dsm_group.add_argument(
        '--dsm_use_sliced', action='store_true', default=True,
        help='[V7] Use Sliced Score Matching for efficiency (default: True)'
    )
    dsm_group.add_argument(
        '--dsm_noise_mode', type=str, default='geometric',
        choices=['geometric', 'uniform', 'fixed'],
        help='[V7] Noise schedule mode (default: geometric)'
    )
    dsm_group.add_argument(
        '--dsm_clean_penalty', type=float, default=0.0,
        help='[V7] MULDE-style clean data penalty weight (default: 0.0)'
    )

    return parser


def parse_ablation_args(parsed_args) -> AblationConfig:
    """
    Create AblationConfig from parsed arguments.

    Args:
        parsed_args: Parsed argparse namespace

    Returns:
        AblationConfig instance
    """
    # Check for preset first
    if hasattr(parsed_args, 'ablation_preset') and parsed_args.ablation_preset:
        config = get_ablation_config(parsed_args.ablation_preset)
    else:
        config = AblationConfig()

    # Apply individual overrides
    if hasattr(parsed_args, 'no_lora') and parsed_args.no_lora:
        config.use_lora = False
    if hasattr(parsed_args, 'no_router') and parsed_args.no_router:
        config.use_router = False
    if hasattr(parsed_args, 'no_task_adapter') and parsed_args.no_task_adapter:
        config.use_task_adapter = False
    if hasattr(parsed_args, 'no_pos_embedding') and parsed_args.no_pos_embedding:
        config.use_pos_embedding = False
    if hasattr(parsed_args, 'no_task_bias') and parsed_args.no_task_bias:
        config.use_task_bias = False
    if hasattr(parsed_args, 'no_mahalanobis') and parsed_args.no_mahalanobis:
        config.use_mahalanobis = False

    # Class-level adapter management
    if hasattr(parsed_args, 'use_class_level_adapters') and parsed_args.use_class_level_adapters:
        config.use_class_level_adapters = True

    # Apply slow stage setting
    if hasattr(parsed_args, 'enable_slow_stage'):
        config.use_slow_stage = parsed_args.enable_slow_stage

    # Apply no_freeze_base setting
    if hasattr(parsed_args, 'no_freeze_base') and parsed_args.no_freeze_base:
        config.no_freeze_base = True

    # Apply LoRA settings
    if hasattr(parsed_args, 'lora_rank'):
        config.lora_rank = parsed_args.lora_rank

    # Apply adapter mode settings
    if hasattr(parsed_args, 'adapter_mode'):
        config.adapter_mode = parsed_args.adapter_mode
    if hasattr(parsed_args, 'soft_ln_init_scale'):
        config.soft_ln_init_scale = parsed_args.soft_ln_init_scale

    # Apply logdet regularization
    if hasattr(parsed_args, 'lambda_logdet'):
        config.lambda_logdet = parsed_args.lambda_logdet

    # Apply spatial context settings (enabled by default)
    if hasattr(parsed_args, 'no_spatial_context') and parsed_args.no_spatial_context:
        config.use_spatial_context = False
    if hasattr(parsed_args, 'spatial_context_mode'):
        config.spatial_context_mode = parsed_args.spatial_context_mode
    if hasattr(parsed_args, 'spatial_context_kernel'):
        config.spatial_context_kernel = parsed_args.spatial_context_kernel

    # Apply scale context settings (enabled by default)
    if hasattr(parsed_args, 'no_scale_context') and parsed_args.no_scale_context:
        config.use_scale_context = False
    if hasattr(parsed_args, 'scale_context_kernel'):
        config.scale_context_kernel = parsed_args.scale_context_kernel
    if hasattr(parsed_args, 'scale_context_init_scale'):
        config.scale_context_init_scale = parsed_args.scale_context_init_scale
    if hasattr(parsed_args, 'scale_context_max_alpha'):
        config.scale_context_max_alpha = parsed_args.scale_context_max_alpha
    if hasattr(parsed_args, 'scale_context_blocks'):
        config.scale_context_blocks = parsed_args.scale_context_blocks

    # =========================================================================
    # V3 Improvements
    # =========================================================================

    # Solution 3: TSA Adapter
    # NOTE: Must also set adapter_mode because __post_init__ was already called during config creation
    if hasattr(parsed_args, 'no_tsa') and parsed_args.no_tsa:
        config.use_tsa = False
        config.adapter_mode = "soft_ln"  # Fallback to SoftLN
    elif hasattr(parsed_args, 'use_tsa') and parsed_args.use_tsa:
        config.use_tsa = True
        config.adapter_mode = "tsa"  # Critical: __post_init__ won't run again

    # Solution 2: Multi-Scale Context
    if hasattr(parsed_args, 'use_ms_context') and parsed_args.use_ms_context:
        config.use_ms_context = True
    if hasattr(parsed_args, 'ms_context_dilations') and parsed_args.ms_context_dilations:
        # Parse comma-separated string to tuple
        dilations = tuple(int(d.strip()) for d in parsed_args.ms_context_dilations.split(','))
        config.ms_context_dilations = dilations

    # Solution 1: Feature Bank
    if hasattr(parsed_args, 'use_feature_bank') and parsed_args.use_feature_bank:
        config.use_feature_bank = True
    if hasattr(parsed_args, 'feature_bank_size'):
        config.feature_bank_size = parsed_args.feature_bank_size
    if hasattr(parsed_args, 'feature_bank_selection'):
        config.feature_bank_selection = parsed_args.feature_bank_selection

    # Solution 1: Distillation
    if hasattr(parsed_args, 'use_distillation') and parsed_args.use_distillation:
        config.use_distillation = True
    if hasattr(parsed_args, 'distillation_mode'):
        config.distillation_mode = parsed_args.distillation_mode
    if hasattr(parsed_args, 'distillation_weight'):
        config.distillation_weight = parsed_args.distillation_weight

    # Solution 1: EWC
    if hasattr(parsed_args, 'use_ewc') and parsed_args.use_ewc:
        config.use_ewc = True
    if hasattr(parsed_args, 'ewc_lambda'):
        config.ewc_lambda = parsed_args.ewc_lambda

    # Solution 1: Adaptive Unfreezing
    if hasattr(parsed_args, 'use_adaptive_unfreeze') and parsed_args.use_adaptive_unfreeze:
        config.use_adaptive_unfreeze = True
    if hasattr(parsed_args, 'adaptive_unfreeze_ratio'):
        config.adaptive_unfreeze_ratio = parsed_args.adaptive_unfreeze_ratio

    # Solution 4: Hybrid Routing
    if hasattr(parsed_args, 'use_hybrid_routing') and parsed_args.use_hybrid_routing:
        config.use_hybrid_routing = True
    if hasattr(parsed_args, 'hybrid_routing_top_k'):
        config.hybrid_routing_top_k = parsed_args.hybrid_routing_top_k

    # Solution 4: Regional Prototype
    if hasattr(parsed_args, 'use_regional_prototype') and parsed_args.use_regional_prototype:
        config.use_regional_prototype = True
    if hasattr(parsed_args, 'regional_prototype_grid'):
        config.regional_prototype_grid = parsed_args.regional_prototype_grid

    # =========================================================================
    # V3 No-Replay Solutions: ACL + OGP
    # =========================================================================

    # Auxiliary Coupling Layer (ACL)
    if hasattr(parsed_args, 'no_acl') and parsed_args.no_acl:
        config.use_acl = False
    elif hasattr(parsed_args, 'use_acl') and parsed_args.use_acl:
        config.use_acl = True
    if hasattr(parsed_args, 'acl_n_layers'):
        config.acl_n_layers = parsed_args.acl_n_layers
    if hasattr(parsed_args, 'acl_hidden_ratio'):
        config.acl_hidden_ratio = parsed_args.acl_hidden_ratio
    # V45: ACL gating and regularization
    if hasattr(parsed_args, 'per_class_acl_layers'):
        config.per_class_acl_layers = parsed_args.per_class_acl_layers
    if hasattr(parsed_args, 'acl_gate') and parsed_args.acl_gate:
        config.acl_gate = True
    if hasattr(parsed_args, 'acl_gate_init'):
        config.acl_gate_init = parsed_args.acl_gate_init
    if hasattr(parsed_args, 'acl_gate_l1_lambda'):
        config.acl_gate_l1_lambda = parsed_args.acl_gate_l1_lambda
    if hasattr(parsed_args, 'acl_weight_decay'):
        config.acl_weight_decay = parsed_args.acl_weight_decay
    if hasattr(parsed_args, 'spatial_var_lambda'):
        config.spatial_var_lambda = parsed_args.spatial_var_lambda

    # V46: Spatial ACL settings
    if hasattr(parsed_args, 'acl_subnet_type'):
        config.acl_subnet_type = parsed_args.acl_subnet_type
    if hasattr(parsed_args, 'acl_kernel_size'):
        config.acl_kernel_size = parsed_args.acl_kernel_size

    # Subnet depth
    if hasattr(parsed_args, 'subnet_depth'):
        config.subnet_depth = parsed_args.subnet_depth
    if hasattr(parsed_args, 'subnet_hidden_ratio'):
        config.subnet_hidden_ratio = parsed_args.subnet_hidden_ratio
    if hasattr(parsed_args, 'subnet_residual') and parsed_args.subnet_residual:
        config.subnet_residual = True
    if hasattr(parsed_args, 'activation_fn'):
        config.activation_fn = parsed_args.activation_fn
    if hasattr(parsed_args, 'use_layernorm') and parsed_args.use_layernorm:
        config.use_layernorm = True
    if hasattr(parsed_args, 'focal_gamma'):
        config.focal_gamma = parsed_args.focal_gamma
    if hasattr(parsed_args, 'score_smooth_sigma'):
        config.score_smooth_sigma = parsed_args.score_smooth_sigma
    if hasattr(parsed_args, 'use_multitask_loss') and parsed_args.use_multitask_loss:
        config.use_multitask_loss = True
    if hasattr(parsed_args, 'multitask_alpha'):
        config.multitask_alpha = parsed_args.multitask_alpha
    if hasattr(parsed_args, 'multitask_beta'):
        config.multitask_beta = parsed_args.multitask_beta

    # Nonlinear LoRA
    if hasattr(parsed_args, 'use_nonlinear_lora') and parsed_args.use_nonlinear_lora:
        config.use_nonlinear_lora = True
    if hasattr(parsed_args, 'nonlinear_lora_alpha'):
        config.nonlinear_lora_alpha = parsed_args.nonlinear_lora_alpha

    # Per-Task Latent Affine
    if hasattr(parsed_args, 'use_latent_affine') and parsed_args.use_latent_affine:
        config.use_latent_affine = True

    # Orthogonal Gradient Projection (OGP)
    if hasattr(parsed_args, 'use_ogp') and parsed_args.use_ogp:
        config.use_ogp = True
    if hasattr(parsed_args, 'ogp_threshold'):
        config.ogp_threshold = parsed_args.ogp_threshold
    if hasattr(parsed_args, 'ogp_max_rank'):
        config.ogp_max_rank = parsed_args.ogp_max_rank
    if hasattr(parsed_args, 'ogp_n_samples'):
        config.ogp_n_samples = parsed_args.ogp_n_samples

    # =========================================================================
    # V3 Task-Conditioned Multi-Scale Context
    # =========================================================================
    if hasattr(parsed_args, 'use_task_conditioned_ms_context') and parsed_args.use_task_conditioned_ms_context:
        config.use_task_conditioned_ms_context = True
    if hasattr(parsed_args, 'tc_ms_context_dilations') and parsed_args.tc_ms_context_dilations:
        # Parse comma-separated string to tuple
        dilations = tuple(int(d.strip()) for d in parsed_args.tc_ms_context_dilations.split(','))
        config.tc_ms_context_dilations = dilations
    if hasattr(parsed_args, 'tc_ms_context_no_regional') and parsed_args.tc_ms_context_no_regional:
        config.tc_ms_context_use_regional = False
    if hasattr(parsed_args, 'tc_ms_context_lora_rank'):
        config.tc_ms_context_lora_rank = parsed_args.tc_ms_context_lora_rank

    # =========================================================================
    # V5 Score Aggregation
    # =========================================================================
    if hasattr(parsed_args, 'score_aggregation_mode'):
        config.score_aggregation_mode = parsed_args.score_aggregation_mode
    if hasattr(parsed_args, 'score_aggregation_percentile'):
        config.score_aggregation_percentile = parsed_args.score_aggregation_percentile
    if hasattr(parsed_args, 'score_aggregation_top_k'):
        config.score_aggregation_top_k = parsed_args.score_aggregation_top_k
    if hasattr(parsed_args, 'score_aggregation_top_k_percent'):
        config.score_aggregation_top_k_percent = parsed_args.score_aggregation_top_k_percent

    # =========================================================================
    # V5 Structural Improvements
    # =========================================================================
    # Phase 1: Tail-Aware Loss
    if hasattr(parsed_args, 'use_tail_aware_loss') and parsed_args.use_tail_aware_loss:
        config.use_tail_aware_loss = True
    if hasattr(parsed_args, 'tail_weight'):
        config.tail_weight = parsed_args.tail_weight
    if hasattr(parsed_args, 'tail_top_k_ratio'):
        config.tail_top_k_ratio = parsed_args.tail_top_k_ratio
    if hasattr(parsed_args, 'tail_match_eval_topk') and parsed_args.tail_match_eval_topk:
        config.tail_match_eval_topk = True
    if hasattr(parsed_args, 'tail_loss_blend_mode'):
        config.tail_loss_blend_mode = parsed_args.tail_loss_blend_mode

    # Phase 1: Spatial Clustering Score
    if hasattr(parsed_args, 'cluster_weight'):
        config.cluster_weight = parsed_args.cluster_weight
    if hasattr(parsed_args, 'cluster_high_score_percentile'):
        config.cluster_high_score_percentile = parsed_args.cluster_high_score_percentile

    # Phase 2: Semantic Projector
    if hasattr(parsed_args, 'use_semantic_projector') and parsed_args.use_semantic_projector:
        config.use_semantic_projector = True
    if hasattr(parsed_args, 'semantic_bottleneck_ratio'):
        config.semantic_bottleneck_ratio = parsed_args.semantic_bottleneck_ratio

    # Phase 2: Task-Adaptive Context
    if hasattr(parsed_args, 'use_task_adaptive_context') and parsed_args.use_task_adaptive_context:
        config.use_task_adaptive_context = True

    # Phase 3: Global Context
    if hasattr(parsed_args, 'use_global_context') and parsed_args.use_global_context:
        config.use_global_context = True
    if hasattr(parsed_args, 'global_context_regions'):
        config.global_context_regions = parsed_args.global_context_regions
    if hasattr(parsed_args, 'global_context_reduction'):
        config.global_context_reduction = parsed_args.global_context_reduction

    # =========================================================================
    # V5.5: Position-Agnostic Improvements
    # =========================================================================
    # Direction 1: Relative Position Encoding
    if hasattr(parsed_args, 'use_relative_position') and parsed_args.use_relative_position:
        config.use_relative_position = True
    if hasattr(parsed_args, 'relative_position_max_dist'):
        config.relative_position_max_dist = parsed_args.relative_position_max_dist
    if hasattr(parsed_args, 'relative_position_num_heads'):
        config.relative_position_num_heads = parsed_args.relative_position_num_heads

    # Direction 2: Dual Branch Scoring
    if hasattr(parsed_args, 'use_dual_branch') and parsed_args.use_dual_branch:
        config.use_dual_branch = True

    # Direction 3: Local Consistency Calibration
    if hasattr(parsed_args, 'use_local_consistency') and parsed_args.use_local_consistency:
        config.use_local_consistency = True
    if hasattr(parsed_args, 'local_consistency_kernel'):
        config.local_consistency_kernel = parsed_args.local_consistency_kernel
    if hasattr(parsed_args, 'local_consistency_temperature'):
        config.local_consistency_temperature = parsed_args.local_consistency_temperature

    # =========================================================================
    # V5.6: Improved Position-Agnostic Solutions
    # =========================================================================
    # Improved Direction 2: Anti-Collapse Dual Branch
    if hasattr(parsed_args, 'use_improved_dual_branch') and parsed_args.use_improved_dual_branch:
        config.use_improved_dual_branch = True
    if hasattr(parsed_args, 'dual_branch_init_alpha'):
        config.dual_branch_init_alpha = parsed_args.dual_branch_init_alpha
    if hasattr(parsed_args, 'dual_branch_min_alpha'):
        config.dual_branch_min_alpha = parsed_args.dual_branch_min_alpha
    if hasattr(parsed_args, 'dual_branch_max_alpha'):
        config.dual_branch_max_alpha = parsed_args.dual_branch_max_alpha

    # Alternative Direction 2: Score-Guided Branch
    if hasattr(parsed_args, 'use_score_guided_dual') and parsed_args.use_score_guided_dual:
        config.use_score_guided_dual = True
    if hasattr(parsed_args, 'score_guided_temperature'):
        config.score_guided_temperature = parsed_args.score_guided_temperature
    if hasattr(parsed_args, 'score_guided_min_alpha'):
        config.score_guided_min_alpha = parsed_args.score_guided_min_alpha

    # Improved Direction 3: Multi-Scale Consistency
    if hasattr(parsed_args, 'use_multiscale_consistency') and parsed_args.use_multiscale_consistency:
        config.use_multiscale_consistency = True
    if hasattr(parsed_args, 'multiscale_kernel_sizes'):
        config.multiscale_kernel_sizes = parsed_args.multiscale_kernel_sizes

    # =========================================================================
    # V5.7: Rotation-Invariant Position Encoding
    # =========================================================================
    # Direction C: Multi-Orientation Ensemble
    if hasattr(parsed_args, 'use_multi_orientation') and parsed_args.use_multi_orientation:
        config.use_multi_orientation = True
    if hasattr(parsed_args, 'multi_orientation_n'):
        config.multi_orientation_n = parsed_args.multi_orientation_n

    # Direction D: Content-Based PE
    if hasattr(parsed_args, 'use_content_based_pe') and parsed_args.use_content_based_pe:
        config.use_content_based_pe = True
    if hasattr(parsed_args, 'content_pe_n_prototypes'):
        config.content_pe_n_prototypes = parsed_args.content_pe_n_prototypes
    if hasattr(parsed_args, 'content_pe_temperature'):
        config.content_pe_temperature = parsed_args.content_pe_temperature
    if hasattr(parsed_args, 'content_pe_blend_grid'):
        config.content_pe_blend_grid = parsed_args.content_pe_blend_grid

    # Direction E: Hybrid PE
    if hasattr(parsed_args, 'use_hybrid_pe') and parsed_args.use_hybrid_pe:
        config.use_hybrid_pe = True
    if hasattr(parsed_args, 'hybrid_pe_n_prototypes'):
        config.hybrid_pe_n_prototypes = parsed_args.hybrid_pe_n_prototypes

    # V5.8 TAPE
    if hasattr(parsed_args, 'use_tape') and parsed_args.use_tape:
        config.use_tape = True
    if hasattr(parsed_args, 'tape_init_value'):
        config.tape_init_value = parsed_args.tape_init_value

    # V6 Data Augmentation
    if hasattr(parsed_args, 'use_rotation_aug') and parsed_args.use_rotation_aug:
        config.use_rotation_aug = True
    if hasattr(parsed_args, 'rotation_degrees'):
        config.rotation_degrees = parsed_args.rotation_degrees

    # V6.1 Spatial Transformer Network (STN)
    if hasattr(parsed_args, 'use_stn') and parsed_args.use_stn:
        config.use_stn = True
    if hasattr(parsed_args, 'stn_mode'):
        config.stn_mode = parsed_args.stn_mode
    if hasattr(parsed_args, 'stn_hidden_dim'):
        config.stn_hidden_dim = parsed_args.stn_hidden_dim
    if hasattr(parsed_args, 'stn_dropout'):
        config.stn_dropout = parsed_args.stn_dropout
    if hasattr(parsed_args, 'stn_rotation_reg_weight'):
        config.stn_rotation_reg_weight = parsed_args.stn_rotation_reg_weight
    if hasattr(parsed_args, 'stn_pretrain_epochs'):
        config.stn_pretrain_epochs = parsed_args.stn_pretrain_epochs

    # =========================================================================
    # V6 Ablation Experiments
    # =========================================================================
    if hasattr(parsed_args, 'use_regular_linear') and parsed_args.use_regular_linear:
        config.use_regular_linear = True
    if hasattr(parsed_args, 'use_task_separated') and parsed_args.use_task_separated:
        config.use_task_separated = True
    if hasattr(parsed_args, 'use_spectral_norm') and parsed_args.use_spectral_norm:
        config.use_spectral_norm = True

    # =========================================================================
    # V7 Denoising Score Matching (DSM)
    # =========================================================================
    if hasattr(parsed_args, 'use_dsm') and parsed_args.use_dsm:
        config.use_dsm = True
    if hasattr(parsed_args, 'dsm_mode'):
        config.dsm_mode = parsed_args.dsm_mode
    if hasattr(parsed_args, 'dsm_alpha'):
        config.dsm_alpha = parsed_args.dsm_alpha
    if hasattr(parsed_args, 'dsm_sigma_min'):
        config.dsm_sigma_min = parsed_args.dsm_sigma_min
    if hasattr(parsed_args, 'dsm_sigma_max'):
        config.dsm_sigma_max = parsed_args.dsm_sigma_max
    if hasattr(parsed_args, 'dsm_n_projections'):
        config.dsm_n_projections = parsed_args.dsm_n_projections
    if hasattr(parsed_args, 'dsm_use_sliced'):
        config.dsm_use_sliced = parsed_args.dsm_use_sliced
    if hasattr(parsed_args, 'dsm_noise_mode'):
        config.dsm_noise_mode = parsed_args.dsm_noise_mode
    if hasattr(parsed_args, 'dsm_clean_penalty'):
        config.dsm_clean_penalty = parsed_args.dsm_clean_penalty

    # Re-run __post_init__ to apply all validation/conflict resolution logic
    # This is necessary because __post_init__ was called when config was created (with default values)
    # but we've now modified the config with parsed arguments
    config.__post_init__()

    return config
