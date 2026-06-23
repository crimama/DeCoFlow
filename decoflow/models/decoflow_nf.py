"""
DeCoFlow: Spatial-Aware Normalizing Flow with LoRA Experts.

Structural Decomposition for Normalizing Flow with task-specific adaptation.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, TYPE_CHECKING

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from decoflow.models.position_embedding import positionalencoding2d

from decoflow.models.dcl import DCLSubnet, DCLContextSubnet, LightweightMSContext, TaskConditionedMSContext, AuxiliaryCouplingLayer, PerTaskLatentAffine
from decoflow.models.adapters import (
    FeatureStatistics, TaskInputAdapter, create_task_adapter, SpatialContextMixer, TaskSpecificAlignment,
    SemanticProjector, TaskAdaptiveContextMixer, LightweightGlobalContext,  # V5 modules
    RelativePositionEmbedding, DualBranchScorer, LocalConsistencyCalibrator,  # V5.5 modules
    ImprovedDualBranchScorer, MultiScaleLocalConsistency, ScoreGuidedDualBranch,  # V5.6 modules
    MultiOrientationEnsemble, ContentBasedPositionalEmbedding, HybridRotationInvariantPE,  # V5.7 modules
    TaskAdaptivePositionEncoding  # V5.8 TAPE
)

if TYPE_CHECKING:
    from decoflow.config.ablation import AblationConfig


class DeCoFlowNF(nn.Module):
    """
    DeCoFlow: Structural Decomposition for Normalizing Flow.

    Key Features:
    - Base NF weights shared across all tasks
    - Task-specific LoRA adapters for distribution shift
    - Task-specific biases for mean shift adaptation
    - Task-specific input adapters for feature normalization
    - Feature statistics alignment for cross-task consistency
    - Zero-initialization for stable adaptation

    Ablation Support:
    - use_lora: Enable/disable LoRA adaptation
    - use_task_adapter: Enable/disable task input adapters
    - use_task_bias: Enable/disable task-specific biases
    """

    def __init__(self,
                 embed_dim: int = 512,
                 coupling_layers: int = 8,
                 clamp_alpha: float = 1.9,
                 lora_rank: int = 32,
                 lora_alpha: float = 1.0,
                 device: str = 'cuda',
                 ablation_config: 'AblationConfig' = None):
        super(DeCoFlowNF, self).__init__()

        self.embed_dim = embed_dim
        self.coupling_layers = coupling_layers
        self.clamp_alpha = clamp_alpha
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.device = device

        # Ablation configuration
        self.ablation_config = ablation_config
        if ablation_config is None:
            # Default: all components enabled
            self.use_lora = True
            self.use_task_adapter = True
            self.use_task_bias = True
            self.adapter_mode = "soft_ln"
            self.soft_ln_init_scale = 0.01
            self.use_spatial_context = True
            self.spatial_context_mode = "depthwise_residual"
            self.spatial_context_kernel = 3
            # Scale-specific context
            self.use_scale_context = True
            self.scale_context_kernel = 3
            self.scale_context_init_scale = 0.1
            self.scale_context_max_alpha = 0.2
            self.scale_context_blocks = "all"
            # V3 defaults (all disabled for backward compatibility)
            self.use_ms_context = False
            self.ms_context_dilations = (1, 2, 4)
            self.ms_context_use_regional = True
            self.use_tsa = False
            self.use_adaptive_unfreeze = False
            self.adaptive_unfreeze_ratio = 0.4
            # V3 ACL defaults
            self.use_acl = False
            self.acl_n_layers = 2
            self.acl_hidden_ratio = 0.5
            self.acl_subnet_type = 'fc'
            self.acl_kernel_size = 3
            # V3 Task-Conditioned MS Context defaults
            self.use_task_conditioned_ms_context = False
            self.tc_ms_context_dilations = (1, 2, 4)
            self.tc_ms_context_use_regional = True
            self.tc_ms_context_lora_rank = 16
            # V5 Structural Improvements defaults
            self.use_semantic_projector = False
            self.semantic_bottleneck_ratio = 0.5
            self.use_task_adaptive_context = False
            self.use_global_context = False
            self.global_context_regions = 4
            self.global_context_reduction = 4
            # V5.5 Position-Agnostic defaults
            self.use_relative_position = False
            self.relative_position_max_dist = 7
            self.relative_position_num_heads = 4
            self.use_dual_branch = False
            self.use_local_consistency = False
            self.local_consistency_kernel = 3
            self.local_consistency_temperature = 1.0
            # V5.6 Improved Position-Agnostic defaults
            self.use_improved_dual_branch = False
            self.dual_branch_init_alpha = 0.7
            self.dual_branch_min_alpha = 0.3
            self.dual_branch_max_alpha = 0.9
            self.use_score_guided_dual = False
            self.score_guided_temperature = 1.0
            self.score_guided_min_alpha = 0.2
            self.use_multiscale_consistency = False
            self.multiscale_kernel_sizes = [3, 5, 7]
            # V5.7 Rotation-Invariant PE defaults
            self.use_multi_orientation = False
            self.multi_orientation_n = 4
            self.use_content_based_pe = False
            self.content_pe_n_prototypes = 16
            self.content_pe_temperature = 0.1
            self.content_pe_blend_grid = True
            self.use_hybrid_pe = False
            self.hybrid_pe_n_prototypes = 16
            # V5.8 TAPE defaults
            self.use_tape = False
            self.tape_init_value = 0.0
            # V6 Ablation Experiments defaults
            self.use_regular_linear = False
            self.use_task_separated = False
            self.use_spectral_norm = False
            # Nonlinear LoRA (Sinter) defaults
            self.use_nonlinear_lora = False
            self.nonlinear_lora_alpha = 0.5
            self.subnet_depth = 2
            self.subnet_hidden_ratio = 2.0
            self.subnet_residual = False
            # Per-Task Latent Affine defaults
            self.use_latent_affine = False
            # Subnet architecture defaults
            self.activation_fn = 'relu'
            self.use_layernorm = False
        else:
            self.use_lora = ablation_config.use_lora
            self.use_task_adapter = ablation_config.use_task_adapter
            self.use_task_bias = ablation_config.use_task_bias
            self.adapter_mode = ablation_config.adapter_mode
            self.soft_ln_init_scale = ablation_config.soft_ln_init_scale
            self.use_spatial_context = ablation_config.use_spatial_context
            self.spatial_context_mode = ablation_config.spatial_context_mode
            self.spatial_context_kernel = ablation_config.spatial_context_kernel
            # Scale-specific context
            self.use_scale_context = ablation_config.use_scale_context
            self.scale_context_kernel = ablation_config.scale_context_kernel
            self.scale_context_init_scale = ablation_config.scale_context_init_scale
            self.scale_context_max_alpha = ablation_config.scale_context_max_alpha
            self.scale_context_blocks = getattr(ablation_config, 'scale_context_blocks', 'all')
            # V3 settings
            self.use_ms_context = ablation_config.use_ms_context
            self.ms_context_dilations = ablation_config.ms_context_dilations
            self.ms_context_use_regional = ablation_config.ms_context_use_regional
            self.use_tsa = ablation_config.use_tsa
            self.use_adaptive_unfreeze = ablation_config.use_adaptive_unfreeze
            self.adaptive_unfreeze_ratio = ablation_config.adaptive_unfreeze_ratio
            # V3 ACL settings
            self.use_acl = ablation_config.use_acl
            self.acl_n_layers = ablation_config.acl_n_layers
            self.acl_hidden_ratio = ablation_config.acl_hidden_ratio
            # V45: ACL gating and per-class ACL
            self.per_class_acl_layers = getattr(ablation_config, 'per_class_acl_layers', '')
            self.acl_gate = getattr(ablation_config, 'acl_gate', False)
            self.acl_gate_init = getattr(ablation_config, 'acl_gate_init', 0.0)
            self.spatial_var_lambda = getattr(ablation_config, 'spatial_var_lambda', 0.0)
            self._acl_spatial_var_loss = None
            # V46: Spatial ACL settings
            self.acl_subnet_type = getattr(ablation_config, 'acl_subnet_type', 'fc')
            self.acl_kernel_size = getattr(ablation_config, 'acl_kernel_size', 3)
            # V3 Task-Conditioned MS Context settings
            self.use_task_conditioned_ms_context = ablation_config.use_task_conditioned_ms_context
            self.tc_ms_context_dilations = ablation_config.tc_ms_context_dilations
            self.tc_ms_context_use_regional = ablation_config.tc_ms_context_use_regional
            self.tc_ms_context_lora_rank = ablation_config.tc_ms_context_lora_rank
            # V5 Structural Improvements settings
            self.use_semantic_projector = ablation_config.use_semantic_projector
            self.semantic_bottleneck_ratio = ablation_config.semantic_bottleneck_ratio
            self.use_task_adaptive_context = ablation_config.use_task_adaptive_context
            self.use_global_context = ablation_config.use_global_context
            self.global_context_regions = ablation_config.global_context_regions
            self.global_context_reduction = ablation_config.global_context_reduction
            # V5.5 Position-Agnostic settings
            self.use_relative_position = ablation_config.use_relative_position
            self.relative_position_max_dist = ablation_config.relative_position_max_dist
            self.relative_position_num_heads = ablation_config.relative_position_num_heads
            self.use_dual_branch = ablation_config.use_dual_branch
            self.use_local_consistency = ablation_config.use_local_consistency
            self.local_consistency_kernel = ablation_config.local_consistency_kernel
            self.local_consistency_temperature = ablation_config.local_consistency_temperature
            # V5.6 Improved Position-Agnostic settings
            self.use_improved_dual_branch = ablation_config.use_improved_dual_branch
            self.dual_branch_init_alpha = ablation_config.dual_branch_init_alpha
            self.dual_branch_min_alpha = ablation_config.dual_branch_min_alpha
            self.dual_branch_max_alpha = ablation_config.dual_branch_max_alpha
            self.use_score_guided_dual = ablation_config.use_score_guided_dual
            self.score_guided_temperature = ablation_config.score_guided_temperature
            self.score_guided_min_alpha = ablation_config.score_guided_min_alpha
            self.use_multiscale_consistency = ablation_config.use_multiscale_consistency
            # Parse kernel sizes from string
            kernel_str = ablation_config.multiscale_kernel_sizes
            self.multiscale_kernel_sizes = [int(k) for k in kernel_str.split(',')]
            # V5.7 Rotation-Invariant PE settings
            self.use_multi_orientation = ablation_config.use_multi_orientation
            self.multi_orientation_n = ablation_config.multi_orientation_n
            self.use_content_based_pe = ablation_config.use_content_based_pe
            self.content_pe_n_prototypes = ablation_config.content_pe_n_prototypes
            self.content_pe_temperature = ablation_config.content_pe_temperature
            self.content_pe_blend_grid = ablation_config.content_pe_blend_grid
            self.use_hybrid_pe = ablation_config.use_hybrid_pe
            self.hybrid_pe_n_prototypes = ablation_config.hybrid_pe_n_prototypes
            # V5.8 TAPE settings
            self.use_tape = ablation_config.use_tape
            self.tape_init_value = ablation_config.tape_init_value
            # V6 Ablation Experiments
            self.use_regular_linear = getattr(ablation_config, 'use_regular_linear', False)
            self.use_task_separated = getattr(ablation_config, 'use_task_separated', False)
            self.use_spectral_norm = getattr(ablation_config, 'use_spectral_norm', False)
            # Nonlinear LoRA (Sinter) settings
            self.use_nonlinear_lora = getattr(ablation_config, 'use_nonlinear_lora', False)
            self.nonlinear_lora_alpha = getattr(ablation_config, 'nonlinear_lora_alpha', 0.5)
            # Subnet depth and architecture
            self.subnet_depth = getattr(ablation_config, 'subnet_depth', 2)
            self.subnet_hidden_ratio = getattr(ablation_config, 'subnet_hidden_ratio', 2.0)
            self.subnet_residual = getattr(ablation_config, 'subnet_residual', False)
            self.activation_fn = getattr(ablation_config, 'activation_fn', 'relu')
            self.use_layernorm = getattr(ablation_config, 'use_layernorm', False)
            # Per-Task Latent Affine settings
            self.use_latent_affine = getattr(ablation_config, 'use_latent_affine', False)

        # Track tasks
        self.num_tasks = 0
        self.current_task_id: Optional[int] = None

        # Class-level adapter management (when use_class_level_adapters=True)
        # This allows finer-grained learning when multiple classes are in one task
        self.use_class_level = ablation_config.use_class_level_adapters if ablation_config else False
        self.num_classes = 0
        self.current_class_id: Optional[int] = None
        # Mapping: class_id -> task_id (for freeze/unfreeze logic)
        self.class_to_task: dict = {}

        # Build flow
        self.subnets: List[DCLSubnet] = []
        self.flow = self._build_flow()

        # Feature statistics from Task 0 (reference distribution)
        self.reference_stats = FeatureStatistics(device=device)

        # Task-specific input adapters for pre-conditioning
        self.input_adapters = nn.ModuleDict()

        # V3: Deep Invertible Adapters (ACL) per task
        # Applied AFTER base NF for nonlinear manifold adaptation
        self.acl_adapters = nn.ModuleDict()

        # Per-Task Latent Affine transforms
        # Lightweight post-flow per-task adaptation (alternative to ACL)
        self.latent_affines = nn.ModuleDict()

        # V6: Complete Separated Mode - independent NF for each task
        # When use_task_separated=True, each task gets its own complete NF
        # This is the upper bound experiment (no parameter sharing)
        self.task_flows = nn.ModuleDict()  # task_id -> (flow, subnets list)
        self.task_subnets = {}  # task_id -> list of subnets (not nn.Module, just reference)

        # Spatial Context Mixer (Baseline 1.5) or MS-Context (V3)
        # Priority: TaskConditionedMSContext > LightweightMSContext > SpatialContextMixer
        if self.use_task_conditioned_ms_context:
            # V3 Improved: Task-Conditioned MS Context (fundamental solution)
            # - Shared base frozen after Task 0
            # - Task-specific adaptation via LoRA
            self.spatial_mixer = TaskConditionedMSContext(
                channels=embed_dim,
                dilations=self.tc_ms_context_dilations,
                use_regional=self.tc_ms_context_use_regional,
                regional_grid=4,
                lora_rank=self.tc_ms_context_lora_rank,
                lora_alpha=1.0
            )
            print(f"✅ [V3] TaskConditionedMSContext enabled: dilations={self.tc_ms_context_dilations}, "
                  f"regional={self.tc_ms_context_use_regional}, lora_rank={self.tc_ms_context_lora_rank}")
        elif self.use_ms_context:
            # V3: Lightweight Multi-Scale Context (supersedes SpatialContextMixer)
            # WARNING: This has catastrophic forgetting issues - use TaskConditionedMSContext instead
            self.spatial_mixer = LightweightMSContext(
                channels=embed_dim,
                dilations=self.ms_context_dilations,
                use_regional=self.ms_context_use_regional,
                regional_grid=4
            )
            print(f"⚠️  [V3] LightweightMSContext enabled: dilations={self.ms_context_dilations}, "
                  f"regional={self.ms_context_use_regional} (WARNING: May cause forgetting)")
        elif self.use_spatial_context:
            self.spatial_mixer = SpatialContextMixer(
                channels=embed_dim,
                mode=self.spatial_context_mode,
                kernel_size=self.spatial_context_kernel,
                learnable=True
            )
            print(f"✅ SpatialContextMixer enabled: mode={self.spatial_context_mode}, kernel={self.spatial_context_kernel}")
        else:
            self.spatial_mixer = None

        # V5: Semantic Projector (Position-agnostic feature extraction)
        if self.use_semantic_projector:
            self.semantic_projector = SemanticProjector(
                channels=embed_dim,
                bottleneck_ratio=self.semantic_bottleneck_ratio
            )
            print(f"✅ [V5] SemanticProjector enabled: bottleneck_ratio={self.semantic_bottleneck_ratio}")
        else:
            self.semantic_projector = None

        # V5: Task-Adaptive Context Mixer (wraps frozen SpatialMixer)
        # Created in add_task when use_task_adaptive_context=True and task_id > 0
        self.task_adaptive_mixer = None  # Will be created if needed

        # V5: Global Context Module (long-range dependency)
        if self.use_global_context:
            self.global_context = LightweightGlobalContext(
                channels=embed_dim,
                num_regions=self.global_context_regions,
                reduction=self.global_context_reduction
            )
            print(f"✅ [V5] LightweightGlobalContext enabled: regions={self.global_context_regions}, reduction={self.global_context_reduction}")
        else:
            self.global_context = None

        # V5.5: Relative Position Encoding (Direction 1)
        if self.use_relative_position:
            self.relative_position = RelativePositionEmbedding(
                channels=embed_dim,
                max_relative_distance=self.relative_position_max_dist,
                num_heads=self.relative_position_num_heads
            )
            print(f"✅ [V5.5] RelativePositionEmbedding enabled: max_dist={self.relative_position_max_dist}, heads={self.relative_position_num_heads}")
        else:
            self.relative_position = None

        # V5.5: Dual Branch Scorer (Direction 2)
        if self.use_dual_branch:
            self.dual_branch_scorer = DualBranchScorer(channels=embed_dim)
            print(f"✅ [V5.5] DualBranchScorer enabled: learns α to blend pos/nopos branches")
        else:
            self.dual_branch_scorer = None

        # V5.5: Local Consistency Calibrator (Direction 3)
        if self.use_local_consistency:
            self.local_consistency = LocalConsistencyCalibrator(
                kernel_size=self.local_consistency_kernel,
                temperature=self.local_consistency_temperature
            )
            print(f"✅ [V5.5] LocalConsistencyCalibrator enabled: kernel={self.local_consistency_kernel}, temp={self.local_consistency_temperature}")
        else:
            self.local_consistency = None

        # V5.6: Improved Dual Branch Scorer (fixes α collapse)
        if self.use_improved_dual_branch:
            self.improved_dual_branch = ImprovedDualBranchScorer(
                channels=embed_dim,
                init_alpha=self.dual_branch_init_alpha,
                min_alpha=self.dual_branch_min_alpha,
                max_alpha=self.dual_branch_max_alpha
            )
            print(f"✅ [V5.6] ImprovedDualBranchScorer enabled: init_α={self.dual_branch_init_alpha}, "
                  f"α∈[{self.dual_branch_min_alpha}, {self.dual_branch_max_alpha}]")
        else:
            self.improved_dual_branch = None

        # V5.6: Score-Guided Dual Branch (alternative, simpler)
        if self.use_score_guided_dual:
            self.score_guided_dual = ScoreGuidedDualBranch(
                temperature=self.score_guided_temperature,
                min_alpha=self.score_guided_min_alpha
            )
            print(f"✅ [V5.6] ScoreGuidedDualBranch enabled: temp={self.score_guided_temperature}, min_α={self.score_guided_min_alpha}")
        else:
            self.score_guided_dual = None

        # V5.6: Multi-Scale Local Consistency
        if self.use_multiscale_consistency:
            self.multiscale_consistency = MultiScaleLocalConsistency(
                kernel_sizes=self.multiscale_kernel_sizes,
                temperature=self.local_consistency_temperature
            )
            print(f"✅ [V5.6] MultiScaleLocalConsistency enabled: kernels={self.multiscale_kernel_sizes}")
        else:
            self.multiscale_consistency = None

        # V5.7: Multi-Orientation Ensemble (Direction C)
        # Test-time rotation ensemble - no learnable parameters
        if self.use_multi_orientation:
            self.multi_orientation = MultiOrientationEnsemble(
                n_orientations=self.multi_orientation_n
            )
            print(f"✅ [V5.7] MultiOrientationEnsemble enabled: {self.multi_orientation_n} orientations (0°, 90°, 180°, 270°)")
        else:
            self.multi_orientation = None

        # V5.7: Content-Based PE (Direction D)
        # Semantic prototype-based position encoding
        if self.use_content_based_pe:
            self.content_based_pe = ContentBasedPositionalEmbedding(
                embed_dim=embed_dim,
                n_prototypes=self.content_pe_n_prototypes,
                temperature=self.content_pe_temperature,
                blend_with_grid=self.content_pe_blend_grid
            )
            print(f"✅ [V5.7] ContentBasedPE enabled: {self.content_pe_n_prototypes} prototypes, "
                  f"temp={self.content_pe_temperature}, blend_grid={self.content_pe_blend_grid}")
        else:
            self.content_based_pe = None

        # V5.7: Hybrid PE (Direction E)
        # Per-patch selection between content PE and grid PE
        if self.use_hybrid_pe:
            self.hybrid_pe = HybridRotationInvariantPE(
                embed_dim=embed_dim,
                n_prototypes=self.hybrid_pe_n_prototypes
            )
            print(f"✅ [V5.7] HybridPE enabled: {self.hybrid_pe_n_prototypes} prototypes, per-patch content/grid selection")
        else:
            self.hybrid_pe = None

        # V5.8: Task-Adaptive Position Encoding (TAPE)
        # Learns optimal PE strength per task
        if self.use_tape:
            self.tape = TaskAdaptivePositionEncoding(
                init_value=self.tape_init_value
            )
            print(f"✅ [V5.8] TAPE enabled: init_value={self.tape_init_value} "
                  f"(sigmoid={torch.sigmoid(torch.tensor(self.tape_init_value)).item():.3f})")
        else:
            self.tape = None

        self.to(device)

    def _resolve_scale_context_block_indices(self) -> set:
        """Resolve which block indices should have scale context enabled."""
        spec = self.scale_context_blocks
        n = self.coupling_layers
        if spec == "all":
            return set(range(n))
        elif spec == "first_only":
            return {0}
        elif spec == "except_last":
            return set(range(n - 1))
        elif spec == "first_half":
            return set(range(n // 2))
        elif spec == "last_half":
            return set(range(n // 2, n))
        else:
            # Comma-separated indices e.g. "0,1,2"
            return set(int(i.strip()) for i in spec.split(",") if i.strip().isdigit())

    def _build_flow(self) -> Ff.SequenceINN:
        """Build normalizing flow with DCL coupling blocks."""
        # Resolve which blocks get scale context
        sc_block_indices = self._resolve_scale_context_block_indices() if self.use_scale_context else set()

        def make_subnet_for_block(block_idx):
            """Create a subnet factory for a specific block index."""
            def make_subnet(dims_in, dims_out):
                use_ctx = block_idx in sc_block_indices
                if use_ctx:
                    subnet = DCLContextSubnet(
                        dims_in, dims_out,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        use_lora=self.use_lora,
                        use_task_bias=self.use_task_bias,
                        context_kernel=self.scale_context_kernel,
                        context_init_scale=self.scale_context_init_scale,
                        context_max_alpha=self.scale_context_max_alpha,
                        use_nonlinear_lora=self.use_nonlinear_lora,
                        nonlinear_lora_alpha=self.nonlinear_lora_alpha,
                        subnet_depth=self.subnet_depth,
                        use_regular_linear=self.use_regular_linear,
                        hidden_ratio=self.subnet_hidden_ratio,
                        activation_fn=self.activation_fn,
                        use_layernorm=self.use_layernorm
                    )
                else:
                    subnet = DCLSubnet(
                        dims_in, dims_out,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        use_lora=self.use_lora,
                        use_task_bias=self.use_task_bias,
                        use_regular_linear=self.use_regular_linear,
                        use_spectral_norm=self.use_spectral_norm,
                        use_nonlinear_lora=self.use_nonlinear_lora,
                        nonlinear_lora_alpha=self.nonlinear_lora_alpha,
                        subnet_depth=self.subnet_depth,
                        hidden_ratio=self.subnet_hidden_ratio,
                        use_residual=self.subnet_residual,
                        activation_fn=self.activation_fn,
                        use_layernorm=self.use_layernorm
                    )
                self.subnets.append(subnet)
                return subnet
            return make_subnet

        coder = Ff.SequenceINN(self.embed_dim)
        ablation_info = []
        if not self.use_lora:
            ablation_info.append("LoRA=OFF")
        if not self.use_task_adapter:
            ablation_info.append("TaskAdapter=OFF")
        if not self.use_task_bias:
            ablation_info.append("TaskBias=OFF")
        if self.use_scale_context:
            sc_spec = self.scale_context_blocks
            if sc_spec == "all":
                ablation_info.append(f"ScaleContext=ON(k={self.scale_context_kernel})")
            else:
                ablation_info.append(f"ScaleContext=ON(k={self.scale_context_kernel},blocks={sc_spec})")
        if self.subnet_depth != 2:
            ablation_info.append(f"SubnetDepth={self.subnet_depth}")
        # V6 ablation info
        if self.use_regular_linear:
            ablation_info.append("RegularLinear=ON")
        if self.use_spectral_norm:
            ablation_info.append("SpectralNorm=ON")
        if self.use_task_separated:
            ablation_info.append("TaskSeparated=ON")
        ablation_str = f" [{', '.join(ablation_info)}]" if ablation_info else ""
        print(f'DeCoFlow => Embed Dim: {self.embed_dim}, LoRA Rank: {self.lora_rank}{ablation_str}')

        for k in range(self.coupling_layers):
            coder.append(
                Fm.AllInOneBlock,
                subnet_constructor=make_subnet_for_block(k),
                affine_clamping=self.clamp_alpha,
                global_affine_type='SOFTPLUS',
                permute_soft=False  # Hard permutation (faster, no O(n²) init)
            )

        return coder

    def _build_task_flow(self, task_id: int) -> Ff.SequenceINN:
        """
        Build a completely independent NF for a specific task (Complete Separated mode).

        This creates a fresh flow with new weights, not sharing any parameters with
        the base flow or other tasks. Used when use_task_separated=True.

        Args:
            task_id: The task ID for which to create the flow

        Returns:
            A new SequenceINN flow instance
        """
        task_subnets = []
        sc_block_indices = self._resolve_scale_context_block_indices() if self.use_scale_context else set()

        def make_task_subnet_for_block(block_idx):
            def make_task_subnet(dims_in, dims_out):
                use_ctx = block_idx in sc_block_indices
                if use_ctx:
                    subnet = DCLContextSubnet(
                        dims_in, dims_out,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        use_lora=self.use_lora,
                        use_task_bias=self.use_task_bias,
                        context_kernel=self.scale_context_kernel,
                        context_init_scale=self.scale_context_init_scale,
                        context_max_alpha=self.scale_context_max_alpha,
                        use_nonlinear_lora=self.use_nonlinear_lora,
                        nonlinear_lora_alpha=self.nonlinear_lora_alpha,
                        subnet_depth=self.subnet_depth,
                        use_regular_linear=self.use_regular_linear,
                        hidden_ratio=self.subnet_hidden_ratio,
                        activation_fn=self.activation_fn,
                        use_layernorm=self.use_layernorm
                    )
                else:
                    subnet = DCLSubnet(
                        dims_in, dims_out,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        use_lora=self.use_lora,
                        use_task_bias=self.use_task_bias,
                        use_regular_linear=self.use_regular_linear,
                        use_spectral_norm=self.use_spectral_norm,
                        use_nonlinear_lora=self.use_nonlinear_lora,
                        nonlinear_lora_alpha=self.nonlinear_lora_alpha,
                        subnet_depth=self.subnet_depth,
                        hidden_ratio=self.subnet_hidden_ratio,
                        use_residual=self.subnet_residual,
                        activation_fn=self.activation_fn,
                        use_layernorm=self.use_layernorm
                    )
                task_subnets.append(subnet)
                return subnet
            return make_task_subnet

        flow = Ff.SequenceINN(self.embed_dim)
        for k in range(self.coupling_layers):
            flow.append(
                Fm.AllInOneBlock,
                subnet_constructor=make_task_subnet_for_block(k),
                affine_clamping=self.clamp_alpha,
                global_affine_type='SOFTPLUS',
                permute_soft=False
            )

        # Store subnet references
        self.task_subnets[str(task_id)] = task_subnets

        # Initialize subnets for this task (unfreeze base, add adapters)
        for subnet in task_subnets:
            subnet.unfreeze_base()
            if self.use_lora or self.use_task_bias:
                subnet.add_task_adapter(task_id)

        return flow.to(self.device)

    def _get_acl_n_layers(self, class_name=None):
        """Get ACL n_layers for a given class (V45 per-class ACL support)."""
        if self.per_class_acl_layers and class_name:
            import json
            layers_dict = json.loads(self.per_class_acl_layers)
            return layers_dict.get(class_name, 1)  # default=1 layer
        return self.acl_n_layers

    def add_task(self, task_id: int, class_name: str = None):
        """
        Add LoRA adapters and input adapter for a new task.

        NEW Design (Task 0 also uses LoRA):
        - Task 0: Train base weights + LoRA adapter (equal treatment)
        - Task > 0: Freeze base, add LoRA + task bias + input adapter

        Key Benefits:
        - Task 0 is no longer "special" - all tasks use LoRA for adaptation
        - Base NF learns general feature transformation
        - LoRA handles task-specific adaptation uniformly

        Respects ablation flags:
        - use_lora: If False, skip LoRA adapters
        - use_task_adapter: If False, skip input adapters
        - use_task_bias: If False, skip task-specific biases
        """
        task_key = str(task_id)

        if task_id == 0:
            # Task 0: Train base weights + LoRA adapter + InputAdapter (self-adaptation)
            for subnet in self.subnets:
                subnet.unfreeze_base()
                # NEW: Also add LoRA adapter for Task 0
                if self.use_lora or self.use_task_bias:
                    subnet.add_task_adapter(task_id)

            # Add InputAdapter for Task 0 as well (self-adaptation)
            # This helps Task 0 pixel-level performance by learning input feature adjustment
            if self.use_task_adapter:
                self.input_adapters[task_key] = create_task_adapter(
                    adapter_mode=self.adapter_mode,
                    channels=self.embed_dim,
                    reference_mean=None,  # No reference for Task 0 (it IS the reference)
                    reference_std=None,
                    task_id=task_id,
                    soft_ln_init_scale=self.soft_ln_init_scale
                ).to(self.device)

            # V3: Add ACL for Task 0 (all tasks get ACL for uniform treatment)
            if self.use_acl:
                acl_layers = self._get_acl_n_layers(class_name)
                self.acl_adapters[task_key] = AuxiliaryCouplingLayer(
                    channels=self.embed_dim,
                    task_id=task_id,
                    n_layers=acl_layers,
                    hidden_ratio=self.acl_hidden_ratio,
                    clamp_alpha=self.clamp_alpha,
                    use_gate=self.acl_gate,
                    gate_init=self.acl_gate_init,
                    subnet_type=self.acl_subnet_type,
                    kernel_size=self.acl_kernel_size
                ).to(self.device)

            # Per-Task Latent Affine for Task 0
            if self.use_latent_affine:
                self.latent_affines[task_key] = PerTaskLatentAffine(
                    dim=self.embed_dim
                ).to(self.device)

            components = []
            if self.use_lora:
                components.append(f"LoRA (rank={self.lora_rank})")
            if self.use_task_bias:
                components.append("Task Biases")
            if self.use_task_adapter:
                adapter_info = f"Input Adapter ({self.adapter_mode})"
                if self.adapter_mode == "soft_ln":
                    adapter_info += f" [init_scale={self.soft_ln_init_scale}]"
                components.append(adapter_info)
            if self.use_acl:
                components.append(f"ACL ({self.acl_n_layers} layers)")
            if self.use_latent_affine:
                components.append("LatentAffine")

            print(f"✅ Task {task_id}: Base weights trainable + {' + '.join(components) if components else 'no adapters'}")
            print(f"   📊 Feature statistics will be collected during training")
        else:
            # Task > 0: Freeze or partially unfreeze base, add LoRA adapters + task biases
            # V6-Exp2: Task-separated mode - each task trains independently
            if self.use_task_separated:
                # Complete Separated: Create a brand new, independent NF for this task
                # No parameter sharing with base flow or other tasks
                task_flow = self._build_task_flow(task_id)
                self.task_flows[task_key] = task_flow

                # Count parameters for this task's flow
                task_params = sum(p.numel() for p in task_flow.parameters())
                print(f"   🔀 [V6] Complete Separated: Created independent NF for Task {task_id}")
                print(f"      📊 Task {task_id} NF parameters: {task_params:,}")

            elif self.use_adaptive_unfreeze:
                # V3: Adaptive unfreezing - unfreeze later blocks
                n_blocks = self.coupling_layers
                n_frozen = int(n_blocks * (1 - self.adaptive_unfreeze_ratio))

                for i, subnet in enumerate(self.subnets):
                    block_idx = i // 2  # Each block has ~2 subnets
                    if block_idx < n_frozen:
                        subnet.freeze_base()
                    else:
                        subnet.unfreeze_base()  # Allow later blocks to adapt

                    if self.use_lora or self.use_task_bias:
                        subnet.add_task_adapter(task_id)

                print(f"   🔓 [V3] Adaptive Unfreeze: {n_frozen}/{n_blocks} blocks frozen, {n_blocks - n_frozen} unfrozen")
            else:
                # Standard: Freeze all base weights
                for subnet in self.subnets:
                    subnet.freeze_base()
                    if self.use_lora or self.use_task_bias:
                        subnet.add_task_adapter(task_id)

            # Create input adapter with reference statistics
            if self.use_task_adapter:
                if self.reference_stats.is_initialized:
                    ref_mean, ref_std = self.reference_stats.get_reference_params()
                    print(f"   📊 Reference stats: mean_norm={ref_mean.norm():.4f}, std_mean={ref_std.mean():.4f}")
                else:
                    ref_mean, ref_std = None, None
                    print(f"   ⚠️  No reference stats available, using default initialization")

                self.input_adapters[task_key] = create_task_adapter(
                    adapter_mode=self.adapter_mode,
                    channels=self.embed_dim,
                    reference_mean=ref_mean,
                    reference_std=ref_std,
                    task_id=task_id,
                    soft_ln_init_scale=self.soft_ln_init_scale
                ).to(self.device)

            # V3: Add ACL for Task > 0 (nonlinear manifold adaptation)
            if self.use_acl:
                acl_layers = self._get_acl_n_layers(class_name)
                self.acl_adapters[task_key] = AuxiliaryCouplingLayer(
                    channels=self.embed_dim,
                    task_id=task_id,
                    n_layers=acl_layers,
                    hidden_ratio=self.acl_hidden_ratio,
                    clamp_alpha=self.clamp_alpha,
                    use_gate=self.acl_gate,
                    gate_init=self.acl_gate_init,
                    subnet_type=self.acl_subnet_type,
                    kernel_size=self.acl_kernel_size
                ).to(self.device)
                print(f"   🔄 [V3] ACL added: {acl_layers} coupling layers [{self.acl_subnet_type}]"
                      + (f" (gate init={self.acl_gate_init})" if self.acl_gate else ""))

            # Per-Task Latent Affine for Task > 0
            if self.use_latent_affine:
                self.latent_affines[task_key] = PerTaskLatentAffine(
                    dim=self.embed_dim
                ).to(self.device)
                print(f"   🔄 LatentAffine added: {2 * self.embed_dim} params")

            # Print status (skip for task_separated mode - already printed above)
            if not self.use_task_separated:
                components = []
                if self.use_lora:
                    components.append(f"LoRA (rank={self.lora_rank})")
                if self.use_task_bias:
                    components.append("Task Biases")
                if self.use_task_adapter:
                    adapter_info = f"Input Adapter ({self.adapter_mode})"
                    if self.adapter_mode == "soft_ln":
                        adapter_info += f" [init_scale={self.soft_ln_init_scale}]"
                    elif self.adapter_mode == "no_ln_after_task0":
                        adapter_info += " [LN=OFF]"
                    components.append(adapter_info)
                if self.use_acl:
                    components.append(f"ACL ({self.acl_n_layers} layers)")
                if self.use_latent_affine:
                    components.append("LatentAffine")

                if components:
                    print(f"✅ Task {task_id}: Base frozen, {' + '.join(components)} added")
                else:
                    print(f"✅ Task {task_id}: Base frozen (no task-specific components - ablation mode)")

        # V3: Add task to TaskConditionedMSContext (handles its own freeze/unfreeze)
        if self.use_task_conditioned_ms_context and self.spatial_mixer is not None:
            self.spatial_mixer.add_task(task_id)

        # V5: Handle Task-Adaptive Context Mixer
        if self.use_task_adaptive_context and self.spatial_mixer is not None:
            if task_id == 0:
                # Task 0: Create TaskAdaptiveContextMixer wrapping the spatial_mixer
                # This will freeze the base mixer and allow task-specific adaptation
                self.task_adaptive_mixer = TaskAdaptiveContextMixer(
                    channels=self.embed_dim,
                    base_mixer=self.spatial_mixer
                )
                self.task_adaptive_mixer.add_task(task_id)
                print(f"   🔧 [V5] TaskAdaptiveContextMixer created (base will freeze after Task 0)")
            else:
                # Task > 0: Add task-specific adapter to existing TaskAdaptiveContextMixer
                if self.task_adaptive_mixer is not None:
                    self.task_adaptive_mixer.add_task(task_id)
                    print(f"   🔧 [V5] Task-specific context adapter added")

        # V5.8: Register task with TAPE
        if self.use_tape and self.tape is not None:
            self.tape.add_task(task_id, device=self.device)
            print(f"   📐 [V5.8] TAPE gate added for Task {task_id}")

        self.num_tasks = task_id + 1
        self.set_active_task(task_id)

    def update_reference_stats(self, features: torch.Tensor):
        """Update reference statistics (call during Task 0 training)."""
        self.reference_stats.update(features)

    def finalize_reference_stats(self):
        """Finalize reference statistics after Task 0 training."""
        self.reference_stats.finalize()

    def set_active_task(self, task_id: Optional[int]):
        """Set the currently active LoRA adapter."""
        self.current_task_id = task_id

        # Set active task for TaskConditionedMSContext
        if self.use_task_conditioned_ms_context and self.spatial_mixer is not None:
            self.spatial_mixer.set_active_task(task_id)

        # V5: Set active task for TaskAdaptiveContextMixer
        if self.use_task_adaptive_context and self.task_adaptive_mixer is not None:
            self.task_adaptive_mixer.set_active_task(task_id)

        if task_id is None:
            for subnet in self.subnets:
                subnet.set_active_task(None)
        else:
            # All tasks (including Task 0) now use LoRA
            for subnet in self.subnets:
                subnet.set_active_task(task_id)

    # =========================================================================
    # Class-Level Adapter Management (for use_class_level_adapters mode)
    # =========================================================================

    def add_class(self, class_id: int, class_name: str, task_id: int):
        """
        Add adapters for a specific class (class-level mode).

        This method is called when use_class_level_adapters=True.
        Creates separate LoRA, ACL, and InputAdapter for each class,
        allowing finer-grained learning when multiple classes are grouped
        into a single task (step).

        Args:
            class_id: Global class index (unique across all tasks)
            class_name: Class name (e.g., 'leather', 'grid')
            task_id: Task ID this class belongs to (for freeze/unfreeze logic)
        """
        class_key = str(class_id)
        self.class_to_task[class_id] = task_id

        # Determine freeze behavior based on task_id
        is_task_0 = (task_id == 0)

        if is_task_0:
            # Task 0 classes: Train base weights + LoRA adapter
            for subnet in self.subnets:
                subnet.unfreeze_base()
                if self.use_lora or self.use_task_bias:
                    subnet.add_task_adapter(class_id)  # Use class_id as adapter key

            # Add InputAdapter for this class
            if self.use_task_adapter:
                self.input_adapters[class_key] = create_task_adapter(
                    adapter_mode=self.adapter_mode,
                    channels=self.embed_dim,
                    reference_mean=None,  # No reference for Task 0 classes
                    reference_std=None,
                    task_id=task_id,  # Use task_id for initialization logic
                    soft_ln_init_scale=self.soft_ln_init_scale
                ).to(self.device)

            # Add ACL for this class
            if self.use_acl:
                self.acl_adapters[class_key] = AuxiliaryCouplingLayer(
                    channels=self.embed_dim,
                    task_id=class_id,  # Use class_id for ACL
                    n_layers=self.acl_n_layers,
                    hidden_ratio=self.acl_hidden_ratio,
                    clamp_alpha=self.clamp_alpha,
                    subnet_type=self.acl_subnet_type,
                    kernel_size=self.acl_kernel_size
                ).to(self.device)

            # Add Latent Affine for this class
            if self.use_latent_affine:
                self.latent_affines[class_key] = PerTaskLatentAffine(
                    dim=self.embed_dim
                ).to(self.device)

            components = []
            if self.use_lora:
                components.append(f"LoRA (rank={self.lora_rank})")
            if self.use_task_adapter:
                components.append(f"InputAdapter ({self.adapter_mode})")
            if self.use_acl:
                components.append(f"ACL ({self.acl_n_layers} layers [{self.acl_subnet_type}])")
            if self.use_latent_affine:
                components.append("LatentAffine")

            print(f"   ✅ Class {class_id} ({class_name}): Base trainable + {' + '.join(components) if components else 'no adapters'}")
        else:
            # Task > 0 classes: Freeze base, add LoRA + InputAdapter
            for subnet in self.subnets:
                subnet.freeze_base()
                if self.use_lora or self.use_task_bias:
                    subnet.add_task_adapter(class_id)  # Use class_id as adapter key

            # Create InputAdapter with reference statistics
            if self.use_task_adapter:
                if self.reference_stats.is_initialized:
                    ref_mean, ref_std = self.reference_stats.get_reference_params()
                else:
                    ref_mean, ref_std = None, None

                self.input_adapters[class_key] = create_task_adapter(
                    adapter_mode=self.adapter_mode,
                    channels=self.embed_dim,
                    reference_mean=ref_mean,
                    reference_std=ref_std,
                    task_id=task_id,  # Use task_id for initialization logic
                    soft_ln_init_scale=self.soft_ln_init_scale
                ).to(self.device)

            # Add ACL for this class
            if self.use_acl:
                self.acl_adapters[class_key] = AuxiliaryCouplingLayer(
                    channels=self.embed_dim,
                    task_id=class_id,
                    n_layers=self.acl_n_layers,
                    hidden_ratio=self.acl_hidden_ratio,
                    clamp_alpha=self.clamp_alpha,
                    subnet_type=self.acl_subnet_type,
                    kernel_size=self.acl_kernel_size
                ).to(self.device)

            # Add Latent Affine for this class
            if self.use_latent_affine:
                self.latent_affines[class_key] = PerTaskLatentAffine(
                    dim=self.embed_dim
                ).to(self.device)

            components = []
            if self.use_lora:
                components.append(f"LoRA (rank={self.lora_rank})")
            if self.use_task_adapter:
                components.append(f"InputAdapter ({self.adapter_mode})")
            if self.use_acl:
                components.append(f"ACL ({self.acl_n_layers} layers)")
            if self.use_latent_affine:
                components.append("LatentAffine")

            print(f"   ✅ Class {class_id} ({class_name}): Base frozen + {' + '.join(components) if components else 'no adapters'}")

        # Update TAPE if enabled
        if self.use_tape and self.tape is not None:
            self.tape.add_task(class_id, device=self.device)

        self.num_classes = max(self.num_classes, class_id + 1)
        self.set_active_class(class_id)

    def set_active_class(self, class_id: Optional[int]):
        """
        Set the currently active class adapter (class-level mode).

        Args:
            class_id: Class ID to activate (global index), or None to deactivate
        """
        self.current_class_id = class_id

        # For TaskConditionedMSContext: use task_id
        if class_id is not None:
            task_id = self.class_to_task.get(class_id, 0)
            if self.use_task_conditioned_ms_context and self.spatial_mixer is not None:
                self.spatial_mixer.set_active_task(task_id)
            if self.use_task_adaptive_context and self.task_adaptive_mixer is not None:
                self.task_adaptive_mixer.set_active_task(task_id)

        if class_id is None:
            for subnet in self.subnets:
                subnet.set_active_task(None)
        else:
            # Set subnet to use class-specific adapter
            for subnet in self.subnets:
                subnet.set_active_task(class_id)  # Use class_id as adapter key

    def get_trainable_params_for_class(self, class_id: int):
        """
        Get trainable parameters for a specific class (class-level mode).

        Args:
            class_id: Global class index

        Returns:
            List of trainable parameters for this class
        """
        params = []
        class_key = str(class_id)
        task_id = self.class_to_task.get(class_id, 0)

        if task_id == 0:
            # Task 0 classes: Base parameters + LoRA + InputAdapter + ACL
            for subnet in self.subnets:
                if self.use_regular_linear:
                    for p in subnet.parameters():
                        if p.requires_grad:
                            params.append(p)
                else:
                    layers = self._get_subnet_layers(subnet)

                    # Base parameters (shared, but trainable for Task 0)
                    for layer in layers:
                        params.extend(layer.base_linear.parameters())

                    # LoRA parameters for this class
                    for layer in layers:
                        if class_key in layer.lora_A:
                            params.append(layer.lora_A[class_key])
                            params.append(layer.lora_B[class_key])

                    # Task-specific biases
                    for layer in layers:
                        if class_key in layer.task_biases:
                            params.append(layer.task_biases[class_key])

                    # Context parameters (only for Task 0)
                    if hasattr(subnet, 'context_conv'):
                        params.extend(subnet.context_conv.parameters())
                    if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                        params.append(subnet.context_scale_param)

            # InputAdapter for this class
            if class_key in self.input_adapters:
                params.extend(self.input_adapters[class_key].parameters())

            # ACL for this class
            if class_key in self.acl_adapters:
                params.extend(self.acl_adapters[class_key].parameters())

            # Latent Affine for this class
            if class_key in self.latent_affines:
                params.extend(self.latent_affines[class_key].parameters())
        else:
            # Task > 0 classes: LoRA/full-rank + InputAdapter + ACL only
            for subnet in self.subnets:
                if self.use_regular_linear:
                    if hasattr(subnet, 'task_layers') and class_key in subnet.task_layers:
                        params.extend(subnet.task_layers[class_key].parameters())
                else:
                    layers = self._get_subnet_layers(subnet)

                    # LoRA parameters
                    for layer in layers:
                        if class_key in layer.lora_A:
                            params.append(layer.lora_A[class_key])
                            params.append(layer.lora_B[class_key])

                    # Task-specific biases
                    for layer in layers:
                        if class_key in layer.task_biases:
                            params.append(layer.task_biases[class_key])

            # InputAdapter
            if class_key in self.input_adapters:
                params.extend(self.input_adapters[class_key].parameters())

            # ACL
            if class_key in self.acl_adapters:
                params.extend(self.acl_adapters[class_key].parameters())

            # Latent Affine
            if class_key in self.latent_affines:
                params.extend(self.latent_affines[class_key].parameters())

        # TAPE parameters for this class
        if self.use_tape and self.tape is not None:
            tape_params = self.tape.get_task_params(class_id)
            if tape_params:
                params.extend(tape_params)

        return params

    def get_active_adapter_key(self) -> str:
        """
        Get the current adapter key (class_id for class-level mode, task_id otherwise).

        Returns:
            String key for accessing adapters in ModuleDict
        """
        if self.use_class_level and self.current_class_id is not None:
            return str(self.current_class_id)
        elif self.current_task_id is not None:
            return str(self.current_task_id)
        else:
            return "0"  # Default to Task 0

    def _get_subnet_layers(self, subnet):
        """
        Get all LoRALinear layers from a subnet.

        Handles both DCLSubnet (layer1, layer2) and
        DCLContextSubnet (s_layer1, s_layer2, t_layer1, t_layer2).
        """
        if hasattr(subnet, 's_layer1'):
            # DCLContextSubnet: 4 layers (s_layer1, s_layer2, t_layer1, t_layer2)
            return [subnet.s_layer1, subnet.s_layer2, subnet.t_layer1, subnet.t_layer2]
        else:
            # DCLSubnet: 2 layers (layer1, layer2)
            return [subnet.layer1, subnet.layer2]

    def get_trainable_params(self, task_id: int):
        """Get trainable parameters for a specific task."""
        params = []
        task_key = str(task_id)

        # V6: Complete Separated Mode - task > 0 has its own independent flow
        if self.use_task_separated and task_id > 0:
            # Return parameters from the task-specific flow
            if task_key in self.task_flows:
                params.extend(self.task_flows[task_key].parameters())

            # Input adapter parameters
            if task_key in self.input_adapters:
                params.extend(self.input_adapters[task_key].parameters())

            # V3: ACL parameters
            if task_key in self.acl_adapters:
                params.extend(self.acl_adapters[task_key].parameters())

            # Latent Affine parameters
            if task_key in self.latent_affines:
                params.extend(self.latent_affines[task_key].parameters())

            return params

        if task_id == 0:
            # Task 0: Base parameters + LoRA parameters + InputAdapter
            for subnet in self.subnets:
                if self.use_regular_linear:
                    # Regular linear mode: train all base nn.Linear params directly
                    for p in subnet.parameters():
                        if p.requires_grad:
                            params.append(p)
                else:
                    layers = self._get_subnet_layers(subnet)

                    # Base parameters
                    for layer in layers:
                        params.extend(layer.base_linear.parameters())

                    # LoRA parameters for Task 0
                    for layer in layers:
                        if task_key in layer.lora_A:
                            params.append(layer.lora_A[task_key])
                            params.append(layer.lora_B[task_key])

                    # Task-specific biases for Task 0
                    for layer in layers:
                        if task_key in layer.task_biases:
                            params.append(layer.task_biases[task_key])

                    # DCLContextSubnet: context_conv and context scaling parameters
                    if hasattr(subnet, 'context_conv'):
                        params.extend(subnet.context_conv.parameters())
                    if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                        params.append(subnet.context_scale_param)

            # Include InputAdapter parameters for Task 0
            if task_key in self.input_adapters:
                params.extend(self.input_adapters[task_key].parameters())

            # V3: ACL parameters for Task 0
            if task_key in self.acl_adapters:
                params.extend(self.acl_adapters[task_key].parameters())

            # Latent Affine parameters for Task 0
            if task_key in self.latent_affines:
                params.extend(self.latent_affines[task_key].parameters())
        else:
            # Task > 0: LoRA parameters + Task Biases + Input Adapter
            for subnet in self.subnets:
                if self.use_regular_linear:
                    # Regular linear mode: train the task-specific layer copies
                    if hasattr(subnet, 'task_layers') and task_key in subnet.task_layers:
                        params.extend(subnet.task_layers[task_key].parameters())
                else:
                    layers = self._get_subnet_layers(subnet)

                    # LoRA A, B matrices
                    for layer in layers:
                        if task_key in layer.lora_A:
                            params.append(layer.lora_A[task_key])
                            params.append(layer.lora_B[task_key])

                    # Task-specific biases
                    for layer in layers:
                        if task_key in layer.task_biases:
                            params.append(layer.task_biases[task_key])

                    # V4.1: DCLContextSubnet context parameters are frozen for Task > 0
                    # They are only trained in Task 0 (see the task_id == 0 block above)

            # Input adapter parameters
            if task_key in self.input_adapters:
                params.extend(self.input_adapters[task_key].parameters())

            # V3: ACL parameters for Task > 0
            if task_key in self.acl_adapters:
                params.extend(self.acl_adapters[task_key].parameters())

            # Latent Affine parameters for Task > 0
            if task_key in self.latent_affines:
                params.extend(self.latent_affines[task_key].parameters())

        # Spatial mixer parameters
        if self.spatial_mixer is not None:
            if self.use_task_conditioned_ms_context:
                # TaskConditionedMSContext: Use its own get_trainable_params
                # This properly handles shared vs task-specific params
                params.extend(self.spatial_mixer.get_trainable_params(task_id))
            elif not self.use_task_adaptive_context:
                # Other spatial mixers (SpatialContextMixer, LightweightMSContext):
                # Train all params with each task (may cause forgetting for LightweightMSContext)
                # Skip if task_adaptive_context is enabled (handled separately)
                params.extend(self.spatial_mixer.parameters())

        # V5: Semantic Projector parameters (trained with each task)
        if self.use_semantic_projector and self.semantic_projector is not None:
            params.extend(self.semantic_projector.parameters())

        # V5: Task-Adaptive Context Mixer parameters
        if self.use_task_adaptive_context and self.task_adaptive_mixer is not None:
            params.extend(self.task_adaptive_mixer.get_trainable_params(task_id))

        # V5: Global Context parameters (trained with each task)
        if self.use_global_context and self.global_context is not None:
            params.extend(self.global_context.parameters())

        # V5.5: Relative Position parameters
        if self.use_relative_position and self.relative_position is not None:
            params.extend(self.relative_position.parameters())

        # V5.5: Dual Branch Scorer parameters
        if self.use_dual_branch and self.dual_branch_scorer is not None:
            params.extend(self.dual_branch_scorer.parameters())

        # V5.5: Local Consistency parameters
        if self.use_local_consistency and self.local_consistency is not None:
            params.extend(self.local_consistency.parameters())

        # V5.6: Improved Dual Branch parameters
        if self.use_improved_dual_branch and self.improved_dual_branch is not None:
            params.extend(self.improved_dual_branch.parameters())

        # V5.6: Score-Guided Dual Branch parameters
        if self.use_score_guided_dual and self.score_guided_dual is not None:
            params.extend(self.score_guided_dual.parameters())

        # V5.6: Multi-Scale Consistency parameters
        if self.use_multiscale_consistency and self.multiscale_consistency is not None:
            params.extend(self.multiscale_consistency.parameters())

        # V5.7: Content-Based PE parameters (Direction D)
        # MultiOrientationEnsemble has NO learnable params (test-time only)
        if self.use_content_based_pe and self.content_based_pe is not None:
            params.extend(self.content_based_pe.parameters())

        # V5.7: Hybrid PE parameters (Direction E)
        if self.use_hybrid_pe and self.hybrid_pe is not None:
            params.extend(self.hybrid_pe.parameters())

        # V5.8: TAPE parameters (task-specific PE gate)
        if self.use_tape and self.tape is not None:
            params.extend(self.tape.get_trainable_params(task_id))

        return params

    def forward(self, patch_embeddings_with_pos: torch.Tensor, reverse: bool = False):
        """
        Forward or inverse transformation with task-specific input pre-conditioning.

        Key Design:
        - Task 0: No pre-conditioning (base model)
        - Task 1+: TaskInputAdapter handles all distribution alignment (if enabled)

        Respects ablation flags:
        - use_task_adapter: If False, skip input adapter

        Args:
            patch_embeddings_with_pos: (B, H, W, D) spatial patch embeddings
                - Standard mode: features + grid PE
                - V5.7 mode (ContentBasedPE/HybridPE): raw features (no PE)
            reverse: If True, generate samples from latent

        Returns:
            z: (B, H, W, D) latent or reconstructed embeddings
            logdet_patch: (B, H, W) patch-wise log determinant of Jacobian
        """
        B, H, W, D = patch_embeddings_with_pos.shape
        x = patch_embeddings_with_pos

        # V5.7/V5.8: Apply custom PE when enabled
        # When enabled, input is raw features (no PE), we generate and apply PE here
        if not reverse:
            if self.use_tape and self.tape is not None and self.current_task_id is not None:
                # V5.8: TAPE - Task-Adaptive PE strength
                grid_pe = self._generate_grid_pe(B, H, W, D)
                x = self.tape(x, grid_pe, self.current_task_id)
            elif self.use_content_based_pe and self.content_based_pe is not None:
                # V5.7: Content-Based PE (Direction D)
                grid_pe = self._generate_grid_pe(B, H, W, D)
                x = self.content_based_pe(x, grid_pe)
            elif self.use_hybrid_pe and self.hybrid_pe is not None:
                # V5.7: Hybrid PE (Direction E)
                grid_pe = self._generate_grid_pe(B, H, W, D)
                x = self.hybrid_pe(x, grid_pe)

        # V5.5: Apply Relative Position Encoding (Direction 1)
        # Blends absolute PE with relative position information
        if self.use_relative_position and self.relative_position is not None and not reverse:
            x = self.relative_position(x)

        # Apply task-specific input adapter for ALL tasks (if enabled)
        # v3: Task 0 also uses InputAdapter for self-adaptation
        if self.use_task_adapter and not reverse and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.input_adapters:
                x = self.input_adapters[task_key](x)

        # V5: Apply Semantic Projector (position-agnostic semantic context)
        if self.use_semantic_projector and self.semantic_projector is not None and not reverse:
            x = self.semantic_projector(x)

        # Apply spatial context mixing (Baseline 1.5 - Legacy or V5 Task-Adaptive)
        # This allows scale(s) to see local context: "abnormal compared to neighbors?"
        if not reverse:
            if self.use_task_adaptive_context and self.task_adaptive_mixer is not None:
                # V5: Use TaskAdaptiveContextMixer (frozen base + task-specific adaptation)
                x = self.task_adaptive_mixer(x)
            elif self.spatial_mixer is not None:
                # Standard spatial mixer
                x = self.spatial_mixer(x)

        # V5: Apply Global Context (long-range dependency via cross-attention)
        if self.use_global_context and self.global_context is not None and not reverse:
            x = self.global_context(x)

        # Flatten spatial dimensions
        x_flat = x.reshape(B * H * W, D)

        # Set spatial info for DCLContextSubnet (Baseline 1.5 Improved)
        # This allows the context-aware s-network to reshape features for 3x3 conv
        if self.use_scale_context:
            DCLContextSubnet._spatial_info = (B, H, W)

        # Select the appropriate flow based on task_separated mode
        # Complete Separated: Each task > 0 has its own independent NF
        task_key = str(self.current_task_id) if self.current_task_id is not None else "0"
        if self.use_task_separated and task_key in self.task_flows:
            # Use task-specific flow (Complete Separated mode)
            current_flow = self.task_flows[task_key]
        else:
            # Use base flow (standard mode or Task 0)
            current_flow = self.flow

        # Flow transformation
        if not reverse:
            z_flat, log_jac_det_flat = current_flow(x_flat)
        else:
            z_flat, log_jac_det_flat = current_flow(x_flat, rev=True)

        # Clear spatial info after forward pass
        if self.use_scale_context:
            DCLContextSubnet._spatial_info = None

        # FrEIA returns (N,) or (N,1) - ensure flat shape
        log_jac_det_flat = log_jac_det_flat.reshape(-1)

        # Keep patch-wise log_det: (BHW,) -> (B, H, W)
        logdet_patch = log_jac_det_flat.reshape(B, H, W)

        # Reshape back to spatial
        z = z_flat.reshape(B, H, W, D)

        # V3: Apply ACL (Auxiliary Coupling Layer) if enabled
        # ACL is applied AFTER base NF for nonlinear manifold adaptation
        if self.use_acl and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.acl_adapters:
                acl = self.acl_adapters[task_key]
                # V45/V46: Compute spatial variance before ACL
                # V46 fix: use dim=(1,2) for true spatial variance, not dim=-1 (channel var)
                if self.spatial_var_lambda > 0 and not reverse:
                    var_before = z.var(dim=(1, 2)).mean()
                z, acl_logdet = acl(z, reverse=reverse)
                logdet_patch = logdet_patch + acl_logdet
                # V45/V46: Compute spatial variance loss after ACL
                if self.spatial_var_lambda > 0 and not reverse:
                    var_after = z.var(dim=(1, 2)).mean()
                    self._acl_spatial_var_loss = torch.relu(var_before - var_after)

        # Apply Per-Task Latent Affine if enabled
        # Applied AFTER ACL (or directly after base NF if ACL is off)
        if self.use_latent_affine and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.latent_affines:
                affine = self.latent_affines[task_key]
                # z shape: (B, H, W, D) — apply element-wise over D
                z, affine_logdet = affine(z, reverse=reverse)
                logdet_patch = logdet_patch + affine_logdet

        return z, logdet_patch

    def get_acl_gate_l1_loss(self):
        """V45: Get L1 loss on ACL gate parameters (encourages smaller gates)."""
        task_key = str(self.current_task_id) if self.current_task_id is not None else None
        if task_key and task_key in self.acl_adapters:
            acl = self.acl_adapters[task_key]
            l1 = torch.tensor(0.0, device=next(acl.parameters()).device)
            for block in acl.coupling_blocks:
                if hasattr(block, 'use_gate') and block.use_gate:
                    l1 = l1 + torch.sigmoid(block.gate_raw)
            return l1
        return None

    def log_prob(self, patch_embeddings_with_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of patch embeddings (image-level).

        Args:
            patch_embeddings_with_pos: (B, H, W, D)

        Returns:
            log_prob: (B,) log probability for each image
        """
        B, H, W, D = patch_embeddings_with_pos.shape

        # Forward transformation - now returns patch-wise logdet
        z, logdet_patch = self.forward(patch_embeddings_with_pos, reverse=False)

        # Patch-wise log p(z): (B, H, W)
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Patch-wise log p(x) = log p(z) + log|det J|: (B, H, W)
        log_px_patch = log_pz_patch + logdet_patch

        # Image-level log-likelihood: sum over all patches
        log_px_image = log_px_patch.sum(dim=(1, 2))  # (B,)

        return log_px_image

    def log_prob_patch(self, patch_embeddings_with_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute patch-wise log probability (for anomaly detection).

        Args:
            patch_embeddings_with_pos: (B, H, W, D)

        Returns:
            log_prob_patch: (B, H, W) log probability for each patch
        """
        B, H, W, D = patch_embeddings_with_pos.shape

        # Forward transformation - returns patch-wise logdet
        z, logdet_patch = self.forward(patch_embeddings_with_pos, reverse=False)

        # Patch-wise log p(z): (B, H, W)
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Patch-wise log p(x) = log p(z) + log|det J|: (B, H, W)
        log_px_patch = log_pz_patch + logdet_patch

        return log_px_patch

    # =========================================================================
    # Slow-Fast Stage Methods for Stage 2 (SLOW Consolidation)
    # =========================================================================

    def unfreeze_last_k_blocks(self, k: int = 2):
        """
        Unfreeze only the last K coupling blocks for slow consolidation.

        This allows gradual generalization of the base NF while protecting
        earlier blocks that encode fundamental transformations.

        Args:
            k: Number of last coupling blocks to unfreeze (default: 2)
        """
        n_subnets = len(self.subnets)
        n_blocks = self.coupling_layers

        # Each coupling block has 2 subnets (s and t networks)
        subnets_per_block = n_subnets // n_blocks
        last_k_subnets = k * subnets_per_block

        # Freeze all first
        for subnet in self.subnets:
            subnet.freeze_base()

        # Unfreeze last k blocks
        for i in range(n_subnets - last_k_subnets, n_subnets):
            self.subnets[i].unfreeze_base()

        unfrozen_count = sum(
            1 for subnet in self.subnets
            if not self._get_subnet_layers(subnet)[0].base_frozen
        )
        print(f"   Unfroze last {k} blocks ({unfrozen_count} subnets)")

    def freeze_all_base(self):
        """Freeze all base parameters (for Stage 1 FAST)."""
        for subnet in self.subnets:
            subnet.freeze_base()

    def get_base_params_for_slow_update(self, k: int = 2) -> List:
        """
        Get base parameters from last K blocks for slow update.

        Args:
            k: Number of last coupling blocks

        Returns:
            List of base parameters to update
        """
        params = []
        n_subnets = len(self.subnets)
        n_blocks = self.coupling_layers
        subnets_per_block = n_subnets // n_blocks
        last_k_subnets = k * subnets_per_block

        for i in range(n_subnets - last_k_subnets, n_subnets):
            subnet = self.subnets[i]
            if self.use_regular_linear:
                params.extend(subnet.parameters())
            else:
                layers = self._get_subnet_layers(subnet)
                for layer in layers:
                    params.extend(layer.base_linear.parameters())

        return params

    def get_fast_params(self, task_id: int) -> List:
        """
        Get FAST adaptation parameters (LoRA + InputAdapter).

        These are trained in Stage 1 (FAST Adaptation).
        """
        params = []
        task_key = str(task_id)

        for subnet in self.subnets:
            if self.use_regular_linear:
                # Regular linear mode: train task-specific layer copies
                if hasattr(subnet, 'task_layers') and task_key in subnet.task_layers:
                    params.extend(subnet.task_layers[task_key].parameters())
            else:
                layers = self._get_subnet_layers(subnet)

                # LoRA A, B matrices
                for layer in layers:
                    if task_key in layer.lora_A:
                        params.append(layer.lora_A[task_key])
                        params.append(layer.lora_B[task_key])

                # Task-specific biases
                for layer in layers:
                    if task_key in layer.task_biases:
                        params.append(layer.task_biases[task_key])

                # V4.1: DCLContextSubnet context parameters - only trained in Task 0
                if task_id == 0:
                    if hasattr(subnet, 'context_conv'):
                        params.extend(subnet.context_conv.parameters())
                    if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                        params.append(subnet.context_scale_param)

        # Input adapter parameters
        if task_key in self.input_adapters:
            params.extend(self.input_adapters[task_key].parameters())

        # V3: ACL parameters (part of FAST stage)
        if task_key in self.acl_adapters:
            params.extend(self.acl_adapters[task_key].parameters())

        # Latent Affine parameters (part of FAST stage)
        if task_key in self.latent_affines:
            params.extend(self.latent_affines[task_key].parameters())

        # V4 Complete Separation: Spatial mixer only trained in Task 0
        if self.spatial_mixer is not None and task_id == 0:
            params.extend(self.spatial_mixer.parameters())

        return params

    def freeze_fast_params(self, task_id: int):
        """Freeze FAST parameters (for Stage 2)."""
        task_key = str(task_id)

        for subnet in self.subnets:
            if self.use_regular_linear:
                if hasattr(subnet, 'task_layers') and task_key in subnet.task_layers:
                    for param in subnet.task_layers[task_key].parameters():
                        param.requires_grad = False
            else:
                layers = self._get_subnet_layers(subnet)

                for layer in layers:
                    if task_key in layer.lora_A:
                        layer.lora_A[task_key].requires_grad = False
                        layer.lora_B[task_key].requires_grad = False
                    if task_key in layer.task_biases:
                        layer.task_biases[task_key].requires_grad = False

                # V4.1: DCLContextSubnet context parameters - only freeze for Task 0
                if task_id == 0:
                    if hasattr(subnet, 'context_conv'):
                        for param in subnet.context_conv.parameters():
                            param.requires_grad = False
                    if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                        subnet.context_scale_param.requires_grad = False

        if task_key in self.input_adapters:
            for param in self.input_adapters[task_key].parameters():
                param.requires_grad = False

        # V3: Freeze ACL parameters
        if task_key in self.acl_adapters:
            for param in self.acl_adapters[task_key].parameters():
                param.requires_grad = False

        # Freeze Latent Affine parameters
        if task_key in self.latent_affines:
            for param in self.latent_affines[task_key].parameters():
                param.requires_grad = False

        # V4: Freeze spatial mixer only for Task 0 (it's already frozen for Task 1+)
        if self.spatial_mixer is not None and task_id == 0:
            for param in self.spatial_mixer.parameters():
                param.requires_grad = False

    def unfreeze_fast_params(self, task_id: int):
        """Unfreeze FAST parameters (after Stage 2, if needed)."""
        task_key = str(task_id)

        for subnet in self.subnets:
            if self.use_regular_linear:
                if hasattr(subnet, 'task_layers') and task_key in subnet.task_layers:
                    for param in subnet.task_layers[task_key].parameters():
                        param.requires_grad = True
            else:
                layers = self._get_subnet_layers(subnet)

                for layer in layers:
                    if task_key in layer.lora_A:
                        layer.lora_A[task_key].requires_grad = True
                        layer.lora_B[task_key].requires_grad = True
                    if task_key in layer.task_biases:
                        layer.task_biases[task_key].requires_grad = True

                # V4.1: DCLContextSubnet context parameters - only unfreeze for Task 0
                if task_id == 0:
                    if hasattr(subnet, 'context_conv'):
                        for param in subnet.context_conv.parameters():
                            param.requires_grad = True
                if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                    subnet.context_scale_param.requires_grad = True

        if task_key in self.input_adapters:
            for param in self.input_adapters[task_key].parameters():
                param.requires_grad = True

        # V3: Unfreeze ACL parameters
        if task_key in self.acl_adapters:
            for param in self.acl_adapters[task_key].parameters():
                param.requires_grad = True

        # Unfreeze Latent Affine parameters
        if task_key in self.latent_affines:
            for param in self.latent_affines[task_key].parameters():
                param.requires_grad = True

        # V4: Unfreeze spatial mixer only for Task 0 (it stays frozen for Task 1+)
        if self.spatial_mixer is not None and task_id == 0:
            for param in self.spatial_mixer.parameters():
                param.requires_grad = True

    # =========================================================================
    # Context Alpha Logging Methods
    # =========================================================================

    def get_context_info(self) -> dict:
        """
        Get context alpha information for logging.

        Returns:
            dict with 'alpha_mean', 'alpha_min', 'alpha_max' values
            or empty dict if not using scale context
        """
        if not self.use_scale_context:
            return {}

        # Collect alpha from all subnets
        alphas = []
        for subnet in self.subnets:
            if hasattr(subnet, 'get_context_alpha'):
                alpha = subnet.get_context_alpha()
                if alpha is not None:
                    alphas.append(alpha)

        if alphas:
            return {
                'alpha_mean': sum(alphas) / len(alphas),
                'alpha_min': min(alphas),
                'alpha_max': max(alphas)
            }

        return {}

    def _generate_grid_pe(self, B: int, H: int, W: int, D: int) -> torch.Tensor:
        """
        Generate 2D sinusoidal positional encoding (grid PE).

        V5.7: Used by ContentBasedPE and HybridPE for blending with semantic PE.

        Args:
            B: Batch size
            H: Height of patch grid
            W: Width of patch grid
            D: Embedding dimension

        Returns:
            grid_pe: (B, H, W, D) positional encoding
        """
        # Generate positional encoding
        pos_embed = positionalencoding2d(D, H, W, device=self.device)  # (D, H, W)

        # Reshape and expand for batch
        pos_embed_expanded = pos_embed.unsqueeze(0).permute(0, 2, 3, 1)  # (1, H, W, D)
        grid_pe = pos_embed_expanded.expand(B, -1, -1, -1)  # (B, H, W, D)

        return grid_pe
