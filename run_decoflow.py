#!/usr/bin/env python
"""
DeCoFlow: Continual Anomaly Detection Runner

This is the main entry point for running DeCoFlow training and evaluation.
Uses the modular decoflow package (fully independent, no NFCAD dependency).

Ablation Support:
    --ablation_preset: Use predefined ablation configuration
    --no_lora: Disable LoRA adaptation
    --no_router: Disable router (use oracle task_id)
    --no_task_adapter: Disable task input adapter
    --no_pos_embedding: Disable positional embedding
    --no_task_bias: Disable task-specific bias
    --no_mahalanobis: Use Euclidean distance instead of Mahalanobis
"""

import argparse
import os
from datetime import datetime
from typing import List

import torch
from torch.utils.data import DataLoader


def parse_cl_scenario(scenario: str, all_classes: List[str]) -> List[List[str]]:
    """
    Parse continual learning scenario string and return task groupings.

    Args:
        scenario: Scenario string in format "base-inc"
            - "1-1": 1 class per task (default, 15 tasks for MVTec)
            - "14-1": 14 classes first, then 1 class (2 tasks)
            - "10-5": 10 classes first, then 5 classes (2 tasks)
            - "3-3": 3 classes per task (5 tasks)
            - "10-1": 10 classes first, then 1 class per step (6 tasks)
        all_classes: List of all class names

    Returns:
        List of task class groups, e.g., [['leather', 'grid', 'transistor'], ['carpet'], ...]
    """
    parts = scenario.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid scenario format: {scenario}. Expected 'base-inc' format.")

    try:
        base_size = int(parts[0])
        inc_size = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid scenario format: {scenario}. Both parts must be integers.")

    n_classes = len(all_classes)

    # Validate
    if base_size <= 0 or inc_size <= 0:
        raise ValueError(f"Both base and inc must be positive integers. Got: {base_size}, {inc_size}")

    if base_size > n_classes:
        raise ValueError(f"Base size ({base_size}) exceeds number of classes ({n_classes})")

    # Build task groups
    tasks = []

    # First task (base)
    tasks.append(all_classes[:base_size])
    remaining = all_classes[base_size:]

    # Incremental tasks
    while remaining:
        chunk = remaining[:inc_size]
        tasks.append(chunk)
        remaining = remaining[inc_size:]

    return tasks


def get_scenario_description(scenario: str, tasks: List[List[str]]) -> str:
    """Generate a human-readable description of the scenario."""
    n_tasks = len(tasks)
    task_sizes = [len(t) for t in tasks]

    # Format like "14-1 with 1 Step" or "3-3 with 4 Steps"
    parts = scenario.split('-')
    base_size, inc_size = int(parts[0]), int(parts[1])

    if base_size == inc_size:
        # Uniform tasks like "3-3"
        steps = n_tasks - 1
        step_str = "Step" if steps == 1 else "Steps"
        return f"{scenario} ({base_size} classes/task, {n_tasks} tasks, {steps} {step_str})"
    else:
        # Non-uniform like "14-1" or "10-5"
        steps = n_tasks - 1
        step_str = "Step" if steps == 1 else "Steps"
        return f"{scenario} (base={base_size}, inc={inc_size}, {n_tasks} tasks, {steps} {step_str})"

# Import everything from decoflow package (no NFCAD dependency)
from decoflow import (
    DeCoFlowNF,
    DeCoFlowContinualTrainer,
    PositionalEmbeddingGenerator,
    TrainingLogger,
    setup_training_logger,
    evaluate_all_tasks,
    evaluate_routing_performance,
    FlowDiagnostics,
    # Feature extraction
    create_feature_extractor,
    get_backbone_type,
    # Utilities
    init_seeds,
    setting_lr_parameters,
    get_config,
    # Data
    create_task_dataset,
)
from decoflow.config import (
    AblationConfig,
    add_ablation_args,
    parse_ablation_args
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeCoFlow: Continual Anomaly Detection')
    parser.add_argument('--task_classes', type=str, nargs='+',
                        default=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                        help='Classes to learn sequentially')
    parser.add_argument('--num_epochs', type=int, default=60,
                        help='Number of epochs per task')
    parser.add_argument('--lora_rank', type=int, default=64,
                        help='LoRA rank for adaptation')
    parser.add_argument('--lora_alpha', type=float, default=1.0,
                        help='LoRA alpha scaling factor (scaling = alpha/rank, default: 1.0)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--slow_lr_ratio', type=float, default=0.2,
                        help='LR ratio for slow update')
    parser.add_argument('--slow_blocks_k', type=int, default=2,
                        help='Number of last coupling blocks to unfreeze in Stage 2')
    parser.add_argument('--enable_slow_stage', action='store_true',
                        help='Enable Stage 2 (SLOW consolidation)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save log files')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment')
    parser.add_argument('--backbone_name', type=str,
                        default='wide_resnet50_2',
                        help='Backbone model name from timm (ViT: vit_*, deit_*, etc. / CNN: wide_resnet50_2, efficientnet_b7, etc.)')
    parser.add_argument('--use_high_res', action='store_true',
                        help='Use high-resolution features (56×56 from layer1 for CNNs)')
    parser.add_argument('--vit_blocks', nargs='+', type=int, default=None,
                        help='ViT blocks to extract features from (e.g., 3 6 9 11). Default: [9, 10, 11]')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--msk_size', type=int, default=256,
                        help='Mask size for evaluation (default: 256)')
    parser.add_argument('--num_coupling_layers', type=int, default=8,
                        help='Number of coupling layers in NF model (default: 8)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--train_drop_last_mode', type=str, default='auto',
                        choices=['auto', 'always', 'never'],
                        help='Train DataLoader drop_last policy (default: auto)')
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to dataset root directory')
    parser.add_argument('--dataset', type=str, default='mvtec',
                        choices=['mvtec', 'visa', 'mpdd', 'adnet', 'realiad'],
                        help='Dataset to use (mvtec, visa, mpdd, adnet, or realiad)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Embedding dimension for NF model (default: same as ViT output dim)')
    parser.add_argument('--run_diagnostics', action='store_true',
                        help='Run Flow diagnostics after training (analyze scale(s) behavior)')
    parser.add_argument('--sigma_sweep', action='store_true',
                        help='Run post-training sigma sweep (tests multiple sigma values)')
    parser.add_argument('--final_eval_only', action='store_true',
                        help='Skip after-task evaluations and evaluate all learned tasks only after training')

    # Checkpoint
    parser.add_argument('--save_checkpoint', action='store_true', default=True,
                        help='Save checkpoint after each task (default: True)')
    parser.add_argument('--no_save_checkpoint', action='store_false', dest='save_checkpoint',
                        help='Disable checkpoint saving')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint dir to load (skip training, eval only)')

    # Continual Learning Scenario
    parser.add_argument('--cl_scenario', type=str, default='1-1',
                        help='''Continual learning scenario configuration.
                        Format: "base-inc" where base=first task classes, inc=incremental classes per step.
                        Examples:
                          - "1-1": 1 class per task (15 tasks total, default)
                          - "14-1": 14 classes in first task, then 1 class (2 tasks)
                          - "10-5": 10 classes in first task, then 5 classes (2 tasks)
                          - "3-3": 3 classes per task (5 tasks)
                          - "10-1": 10 classes in first task, then 1 class per step (6 tasks)
                        ''')

    # Add ablation arguments
    add_ablation_args(parser)

    parsed_args = parser.parse_args()

    # Parse ablation configuration
    ablation_config = parse_ablation_args(parsed_args)

    # Setup configuration using decoflow's config system
    args = get_config(
        img_size=parsed_args.img_size,
        msk_size=parsed_args.msk_size,
        data_path=parsed_args.data_path,
        batch_size=parsed_args.batch_size,
        seed=parsed_args.seed,
        lr=parsed_args.lr,
    )
    # Add dataset type to args
    args.dataset = parsed_args.dataset
    args.task_classes = parsed_args.task_classes
    args.cl_scenario = parsed_args.cl_scenario

    # Initialize seeds
    init_seeds(args.seed)
    setting_lr_parameters(args)

    # Determine embed_dim early (before config saving)
    # Infer default embed_dim from backbone name
    def get_default_embed_dim(backbone_name: str) -> int:
        """Infer embedding dimension from backbone model name."""
        backbone_lower = backbone_name.lower()
        backbone_type = get_backbone_type(backbone_name)

        # ViT-based backbones
        if backbone_type == 'vit':
            if 'vit_giant' in backbone_lower or 'vit_g' in backbone_lower:
                return 1536
            elif 'vit_huge' in backbone_lower or 'vit_h' in backbone_lower:
                return 1280
            elif 'vit_large' in backbone_lower or 'vit_l' in backbone_lower:
                return 1024
            elif 'vit_base' in backbone_lower or 'vit_b' in backbone_lower:
                return 768
            elif 'vit_small' in backbone_lower or 'vit_s' in backbone_lower:
                return 384
            elif 'vit_tiny' in backbone_lower or 'vit_t' in backbone_lower:
                return 192

        # CNN-based backbones - use target dimension (will be projected)
        # For CNN, we typically want a moderate dimension that works well with NF
        if backbone_type == 'cnn':
            # Common choices for anomaly detection
            if 'efficientnet_b7' in backbone_lower or 'efficientnet_b6' in backbone_lower:
                return 1024
            elif 'efficientnet_b5' in backbone_lower or 'efficientnet_b4' in backbone_lower:
                return 768
            elif 'wide_resnet101' in backbone_lower:
                return 1024
            elif 'wide_resnet50' in backbone_lower or 'resnet101' in backbone_lower:
                return 768
            elif 'convnext_large' in backbone_lower or 'convnext_base' in backbone_lower:
                return 1024
            elif 'convnext_small' in backbone_lower or 'convnext_tiny' in backbone_lower:
                return 768
            else:
                # Default for most CNN backbones
                return 768

        # Fallback: try to get from timm
        try:
            import timm
            model = timm.create_model(backbone_name, pretrained=False)
            if hasattr(model, 'embed_dim'):
                dim = model.embed_dim
                del model
                return dim
            elif hasattr(model, 'num_features'):
                dim = model.num_features
                del model
                # Cap at reasonable size for NF
                return min(dim, 1024)
        except:
            pass
        return 768  # Default fallback

    embed_dim = parsed_args.embed_dim if parsed_args.embed_dim is not None else get_default_embed_dim(parsed_args.backbone_name)

    # Setup training logger
    experiment_name = parsed_args.experiment_name
    if experiment_name is None:
        task_str = "_".join(parsed_args.task_classes[:3])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include ablation info in experiment name
        ablation_suffix = ablation_config.get_experiment_name()
        experiment_name = f"decoflow_{task_str}_{ablation_suffix}_{timestamp}"

    logger = setup_training_logger(
        log_dir=parsed_args.log_dir,
        experiment_name=experiment_name
    )

    # Save config to log folder
    config = {
        'experiment_name': experiment_name,
        'dataset': parsed_args.dataset,
        'cl_scenario': parsed_args.cl_scenario,
        'task_classes': parsed_args.task_classes,
        'num_epochs': parsed_args.num_epochs,
        'lora_rank': parsed_args.lora_rank,
        'lora_alpha': parsed_args.lora_alpha,
        'lr': parsed_args.lr,
        'slow_lr_ratio': parsed_args.slow_lr_ratio,
        'slow_blocks_k': parsed_args.slow_blocks_k,
        'enable_slow_stage': parsed_args.enable_slow_stage,
        'backbone_name': parsed_args.backbone_name,
        'backbone_type': get_backbone_type(parsed_args.backbone_name),
        'img_size': parsed_args.img_size,
        'msk_size': parsed_args.msk_size,
        'num_coupling_layers': parsed_args.num_coupling_layers,
        'batch_size': parsed_args.batch_size,
        'train_drop_last_mode': parsed_args.train_drop_last_mode,
        'data_path': parsed_args.data_path,
        'seed': parsed_args.seed,
        'embed_dim': embed_dim,
        'use_high_res': getattr(parsed_args, 'use_high_res', False),
        # Ablation settings
        'ablation': {
            'use_lora': ablation_config.use_lora,
            'use_router': ablation_config.use_router,
            'use_task_adapter': ablation_config.use_task_adapter,
            'use_pos_embedding': ablation_config.use_pos_embedding,
            'use_task_bias': ablation_config.use_task_bias,
            'use_mahalanobis': ablation_config.use_mahalanobis,
            'adapter_mode': ablation_config.adapter_mode,
            'soft_ln_init_scale': ablation_config.soft_ln_init_scale,
            'lambda_logdet': ablation_config.lambda_logdet,
            'use_spatial_context': ablation_config.use_spatial_context,
            'spatial_context_mode': ablation_config.spatial_context_mode,
            'spatial_context_kernel': ablation_config.spatial_context_kernel,
            'use_scale_context': ablation_config.use_scale_context,
            'scale_context_kernel': ablation_config.scale_context_kernel,
            # V5 score/tail settings (critical for reproducibility)
            'score_aggregation_mode': ablation_config.score_aggregation_mode,
            'score_aggregation_percentile': ablation_config.score_aggregation_percentile,
            'score_aggregation_top_k': ablation_config.score_aggregation_top_k,
            'score_aggregation_top_k_percent': ablation_config.score_aggregation_top_k_percent,
            'use_tail_aware_loss': ablation_config.use_tail_aware_loss,
            'tail_weight': ablation_config.tail_weight,
            'tail_top_k_ratio': ablation_config.tail_top_k_ratio,
            'tail_match_eval_topk': getattr(ablation_config, 'tail_match_eval_topk', False),
            'tail_loss_blend_mode': getattr(ablation_config, 'tail_loss_blend_mode', 'mixed'),
            # V46 ACB subnet details
            'acb_subnet_type': getattr(ablation_config, 'acb_subnet_type', 'fc'),
            'acb_kernel_size': getattr(ablation_config, 'acb_kernel_size', 3),
            # Subnet/activation details
            'activation_fn': getattr(ablation_config, 'activation_fn', 'relu'),
            'use_layernorm': getattr(ablation_config, 'use_layernorm', False),
            'focal_gamma': getattr(ablation_config, 'focal_gamma', 0.0),
            # V3 Options
            'use_tsa': ablation_config.use_tsa,
            'use_ms_context': ablation_config.use_ms_context,
            'use_acb': ablation_config.use_acb,
            'acb_n_blocks': ablation_config.acb_n_blocks,
            'subnet_depth': ablation_config.subnet_depth,
            'use_nonlinear_lora': ablation_config.use_nonlinear_lora,
            'nonlinear_lora_alpha': ablation_config.nonlinear_lora_alpha,
            'use_latent_affine': ablation_config.use_latent_affine,
            'use_ogp': ablation_config.use_ogp,
            'ogp_threshold': ablation_config.ogp_threshold,
            'ogp_max_rank': ablation_config.ogp_max_rank,
            'use_feature_bank': ablation_config.use_feature_bank,
            'use_distillation': ablation_config.use_distillation,
            'use_ewc': ablation_config.use_ewc,
            'use_hybrid_routing': ablation_config.use_hybrid_routing,
            'use_regional_prototype': ablation_config.use_regional_prototype,
            # V45: ACB gating and regularization
            'per_class_acb_blocks': ablation_config.per_class_acb_blocks,
            'acb_gate': ablation_config.acb_gate,
            'acb_gate_init': ablation_config.acb_gate_init,
            'acb_gate_l1_lambda': ablation_config.acb_gate_l1_lambda,
            'acb_weight_decay': ablation_config.acb_weight_decay,
            'spatial_var_lambda': ablation_config.spatial_var_lambda,
        },
        'score_smooth_sigma': ablation_config.score_smooth_sigma,
        'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    logger.save_config(config)

    # Setup continual learning tasks based on scenario
    ALL_CLASSES = parsed_args.task_classes
    GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}

    # Parse scenario and create task groupings
    CONTINUAL_TASKS = parse_cl_scenario(parsed_args.cl_scenario, ALL_CLASSES)
    scenario_desc = get_scenario_description(parsed_args.cl_scenario, CONTINUAL_TASKS)

    print("\n" + "="*70)
    print("DeCoFlow: Continual Anomaly Detection")
    print("="*70)
    print(f"   Dataset: {parsed_args.dataset.upper()}")
    print(f"   CL Scenario: {scenario_desc}")
    print(f"   Task Structure:")
    for t_id, t_classes in enumerate(CONTINUAL_TASKS):
        print(f"      Task {t_id}: {t_classes} ({len(t_classes)} classes)")
    print(f"   Total Classes: {len(ALL_CLASSES)}")
    print(f"   Backbone: {parsed_args.backbone_name}")
    print(f"   Image Size: {parsed_args.img_size}")
    print(f"   Coupling Layers: {parsed_args.num_coupling_layers}")
    print(f"   Embedding Dim: {embed_dim}")
    print(f"   LoRA Rank: {parsed_args.lora_rank}")
    print(f"   LoRA Alpha: {parsed_args.lora_alpha} (scaling={parsed_args.lora_alpha/parsed_args.lora_rank:.6f})")
    print(f"   Epochs per Task: {parsed_args.num_epochs}")
    print(f"   Data Path: {parsed_args.data_path}")
    print(f"   Log Directory: {parsed_args.log_dir}")
    print(f"   Slow-Fast Training: {'FAST+SLOW' if parsed_args.enable_slow_stage else 'FAST only'}")
    print(f"   Score Smooth Sigma: {ablation_config.score_smooth_sigma}")
    print("-"*70)
    print(f"   {ablation_config}")
    print("="*70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize feature extractor (supports both ViT and CNN backbones)
    backbone_type = get_backbone_type(parsed_args.backbone_name)
    # Determine ViT blocks to extract
    vit_blocks = parsed_args.vit_blocks if parsed_args.vit_blocks else [9, 10, 11]

    feature_extractor = create_feature_extractor(
        backbone_name=parsed_args.backbone_name,
        input_shape=(3, parsed_args.img_size, parsed_args.img_size),
        target_embed_dimension=embed_dim,
        device=device,
        # ViT-specific
        blocks_to_extract=vit_blocks if backbone_type == 'vit' else None,
        remove_cls_token=True,
        # CNN-specific
        patch_size=3,
        patch_stride=1,
        use_high_res=parsed_args.use_high_res if backbone_type == 'cnn' else False,
    )
    print(f"\nFeature Extractor initialized: {parsed_args.backbone_name} ({backbone_type.upper()}, Frozen)")
    print(f"   Embedding Dimension: {embed_dim}")

    # Initialize positional embedding generator
    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    # Initialize DeCoFlow model with ablation config
    nf_model = DeCoFlowNF(
        embed_dim=embed_dim,
        coupling_layers=parsed_args.num_coupling_layers,
        clamp_alpha=1.9,
        lora_rank=parsed_args.lora_rank,
        lora_alpha=parsed_args.lora_alpha,
        device=device,
        ablation_config=ablation_config
    )
    print("DeCoFlow NF model initialized")

    # Pass enable_slow_stage and score_smooth_sigma to args
    args.enable_slow_stage = parsed_args.enable_slow_stage
    args.score_smooth_sigma = ablation_config.score_smooth_sigma
    args.interp_mode = getattr(parsed_args, 'interp_mode', 'bicubic')

    # Initialize continual trainer with ablation config
    trainer = DeCoFlowContinualTrainer(
        vit_extractor=feature_extractor,
        pos_embed_generator=pos_embed_generator,
        nf_model=nf_model,
        args=args,
        device=device,
        slow_lr_ratio=parsed_args.slow_lr_ratio,
        slow_blocks_k=parsed_args.slow_blocks_k,
        ablation_config=ablation_config
    )
    trainer.set_logger(logger)
    results = {}
    oracle_results = {}

    # Load checkpoint if specified (eval-only mode)
    if parsed_args.load_checkpoint:
        from decoflow.utils.checkpoint import load_checkpoint
        print(f"\n📂 Loading checkpoint from: {parsed_args.load_checkpoint}")
        # Pre-create task structure and LoRA adapters before loading state_dict
        for tid, tc in enumerate(CONTINUAL_TASKS):
            trainer.task_classes[tid] = tc
            nf_model.add_task(tid)
            nf_model.set_active_task(tid)
        load_checkpoint(
            nf_model=nf_model,
            router=trainer.router,
            checkpoint_dir=parsed_args.load_checkpoint,
            device=device,
        )
        print(f"✅ Checkpoint loaded ({len(CONTINUAL_TASKS)} tasks) - skipping training, running eval only")
        # Jump to final evaluation
        use_router = ablation_config.use_router
        results = evaluate_all_tasks(trainer, args, use_router=use_router, target_size=args.msk_size)
        if results:
            print(f"\n{'='*70}")
            print(f"Checkpoint Evaluation Results:")
            print(f"   Mean Image AUC: {results['mean_img_auc']:.4f}")
            print(f"   Mean Pixel AUC: {results['mean_pixel_auc']:.4f}")
            print(f"   Mean Image AP: {results.get('mean_img_ap', 0.0):.4f}")
            print(f"   Mean Pixel AP: {results.get('mean_pixel_ap', 0.0):.4f}")
            print(f"{'='*70}")
            # Save final results table
            logger.save_final_results_table(results, CONTINUAL_TASKS)
        if not parsed_args.sigma_sweep:
            return
        # Skip training loop, jump to sigma sweep below
        skip_training = True
    else:
        skip_training = False

    # Training loop (skipped when loading checkpoint for eval-only)
    for task_id, task_classes in enumerate(CONTINUAL_TASKS):
        if skip_training:
            break
        print(f"\n{'#'*70}")
        print(f"# Task {task_id}: {task_classes}")
        print(f"{'#'*70}")

        # Create task dataset
        args.class_to_idx = {cls: GLOBAL_CLASS_TO_IDX[cls] for cls in task_classes}
        args.n_classes = len(task_classes)

        train_dataset = create_task_dataset(
            args, task_classes, GLOBAL_CLASS_TO_IDX, train=True,
            use_rotation_aug=ablation_config.use_rotation_aug,
            rotation_degrees=ablation_config.rotation_degrees
        )
        # Auto mode avoids dropping samples for small classes (e.g., toothbrush).
        if parsed_args.train_drop_last_mode == 'always':
            train_drop_last = True
        elif parsed_args.train_drop_last_mode == 'never':
            train_drop_last = False
        else:
            train_drop_last = len(train_dataset) >= (2 * args.batch_size)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=False, drop_last=train_drop_last
        )

        # Train on this task
        trainer.train_task(
            task_id=task_id,
            task_classes=task_classes,
            train_loader=train_loader,
            num_epochs=parsed_args.num_epochs,
            lr=parsed_args.lr,
            log_interval=10,
            global_class_to_idx=GLOBAL_CLASS_TO_IDX  # For class-level adapter mode
        )

        if parsed_args.final_eval_only:
            logger.info(f"\nSkipping intermediate evaluation after Task {task_id} (--final_eval_only)")
        else:
            # Evaluate all tasks seen so far
            logger.info(f"\nEvaluation after Task {task_id}")
            try:
                # Use router only if enabled
                use_router = ablation_config.use_router
                results = evaluate_all_tasks(trainer, args, use_router=use_router, target_size=args.msk_size)

                if logger and results:
                    eval_metrics = {
                        'mean_img_auc': results.get('mean_img_auc', 0.0),
                        'mean_pixel_auc': results.get('mean_pixel_auc', 0.0),
                        'mean_img_ap': results.get('mean_img_ap', 0.0),
                        'mean_pixel_ap': results.get('mean_pixel_ap', 0.0),
                        'mean_routing_accuracy': results.get('mean_routing_accuracy', None)
                    }
                    logger.log_evaluation(task_id, eval_metrics)

                    # Save evaluation results to CSV
                    evaluated_classes = results.get('classes', [])
                    img_aucs = results.get('img_aucs', [])
                    pixel_aucs = results.get('pixel_aucs', [])
                    routing_accuracies = results.get('routing_accuracies', None)

                    # Save per-task evaluation results
                    logger.save_evaluation_results_csv(
                        task_id=task_id,
                        epoch=parsed_args.num_epochs - 1,
                        class_names=evaluated_classes,
                        img_aucs=img_aucs,
                        pixel_aucs=pixel_aucs,
                        routing_accuracies=routing_accuracies
                    )

                    # Save continual learning results (including all previous tasks)
                    logger.save_continual_results_csv(
                        task_id=task_id,
                        current_classes=task_classes,
                        all_classes=evaluated_classes,
                        img_aucs=img_aucs,
                        pixel_aucs=pixel_aucs,
                        continual_tasks=CONTINUAL_TASKS,
                        routing_accuracies=routing_accuracies
                    )

                    # Save unified evaluation results
                    logger.save_unified_evaluation_csv(
                        task_id=task_id,
                        epoch=parsed_args.num_epochs - 1,
                        all_classes=evaluated_classes,
                        img_aucs=img_aucs,
                        pixel_aucs=pixel_aucs,
                        ALL_CLASSES=ALL_CLASSES,
                        routing_accuracies=routing_accuracies
                    )

                    # Save evaluation metrics to history
                    logger.save_evaluation_metrics(results)

            except Exception as e:
                logger.warning(f"Error during evaluation: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                results = {}

        # Save checkpoint after each task
        if parsed_args.save_checkpoint:
            from decoflow.utils.checkpoint import save_checkpoint
            ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
            save_checkpoint(
                nf_model=trainer.nf_model,
                router=trainer.router,
                task_id=task_id,
                save_dir=ckpt_dir,
                config=config,
            )

    if not skip_training and parsed_args.final_eval_only:
        logger.info("\nFinal evaluation after all tasks (--final_eval_only)")
        try:
            use_router = ablation_config.use_router
            results = evaluate_all_tasks(trainer, args, use_router=use_router, target_size=args.msk_size)
            if logger and results:
                logger.save_evaluation_metrics(results)
        except Exception as e:
            logger.warning(f"Error during final evaluation: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            results = {}

    # Final Summary
    print("\n" + "="*70)
    print("Continual Learning Completed!")
    print("="*70)
    print(f"   Total Tasks: {len(CONTINUAL_TASKS)}")
    print(f"   Total Classes: {len(ALL_CLASSES)}")
    if results and 'mean_img_auc' in results:
        print(f"   Final Mean Image AUC: {results['mean_img_auc']:.4f}")
        print(f"   Final Mean Pixel AUC: {results['mean_pixel_auc']:.4f}")
        print(f"   Final Mean Image AP: {results.get('mean_img_ap', 0.0):.4f}")
        print(f"   Final Mean Pixel AP: {results.get('mean_pixel_ap', 0.0):.4f}")
    print("="*70)

    # Routing Performance Analysis (only if router is enabled)
    routing_metrics = None
    if ablation_config.use_router:
        print("\nRouting Performance Analysis")
        routing_metrics = evaluate_routing_performance(trainer, args, target_size=args.msk_size)

        # Oracle evaluation
        print("\nEvaluating with Oracle (ground truth task_id)...")
        try:
            oracle_results = evaluate_all_tasks(trainer, args, use_router=False, target_size=args.msk_size)
        except Exception as e:
            print(f"Error during oracle evaluation: {e}")
            oracle_results = {}
    else:
        print("\n[Ablation] Router disabled - using oracle task_id for all evaluations")
        oracle_results = results

    # Save final summary
    if results and 'mean_img_auc' in results:
        # Save final results table (simple format)
        logger.save_final_results_table(results, CONTINUAL_TASKS)

        # Prepare additional metrics
        additional_metrics = {}
        if results.get('mean_class_routing_accuracy') is not None:
            additional_metrics['mean_class_routing_accuracy'] = results.get('mean_class_routing_accuracy')
        if results.get('mean_task_routing_accuracy') is not None:
            additional_metrics['mean_task_routing_accuracy'] = results.get('mean_task_routing_accuracy')
        if routing_metrics is not None:
            additional_metrics['overall_routing_accuracy'] = routing_metrics.get('overall_accuracy', -1.0)
            for task_id_key, acc in routing_metrics.get('task_accuracies', {}).items():
                additional_metrics[f'task_{task_id_key}_routing_accuracy'] = acc

        if oracle_results and 'mean_img_auc' in oracle_results:
            additional_metrics['oracle_mean_img_auc'] = oracle_results['mean_img_auc']
            additional_metrics['oracle_mean_pixel_auc'] = oracle_results['mean_pixel_auc']

        # Save final summary to CSV
        logger.save_final_summary(
            strategy_name="DeCoFlow",
            all_classes=results.get('classes', ALL_CLASSES),
            img_aucs=results.get('img_aucs', []),
            pixel_aucs=results.get('pixel_aucs', []),
            num_tasks=len(CONTINUAL_TASKS),
            ablation_config=ablation_config,
            routing_accuracy=results.get('mean_routing_accuracy', None),
            additional_metrics=additional_metrics,
            img_aps=results.get('img_aps', []),
            pixel_aps=results.get('pixel_aps', [])
        )

    # Run sigma sweep if requested
    if parsed_args.sigma_sweep:
        print("\n" + "="*70)
        print("Running Sigma Sweep (post-training evaluation)")
        print("="*70)
        sigma_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        sigma_results = []
        per_class_best_sigma = {}  # class_name -> (best_sigma, best_pixel_ap)
        for sigma in sigma_values:
            print(f"\n--- Sigma = {sigma} ---")
            args.score_smooth_sigma = sigma
            try:
                use_router = ablation_config.use_router
                sweep_result = evaluate_all_tasks(trainer, args, use_router=use_router, target_size=args.msk_size)
                entry = {
                    'sigma': sigma,
                    'img_auc': sweep_result.get('mean_img_auc', 0.0),
                    'pixel_auc': sweep_result.get('mean_pixel_auc', 0.0),
                    'img_ap': sweep_result.get('mean_img_ap', 0.0),
                    'pixel_ap': sweep_result.get('mean_pixel_ap', 0.0),
                }
                sigma_results.append(entry)
                print(f"  Sigma={sigma}: I-AUC={entry['img_auc']:.4f}, P-AUC={entry['pixel_auc']:.4f}, P-AP={entry['pixel_ap']:.4f}")

                # Track per-class best sigma
                classes = sweep_result.get('classes', [])
                pixel_aps = sweep_result.get('pixel_aps', [])
                for cls_name, pap in zip(classes, pixel_aps):
                    if cls_name not in per_class_best_sigma or pap > per_class_best_sigma[cls_name][1]:
                        per_class_best_sigma[cls_name] = (sigma, pap)
            except Exception as e:
                print(f"  Error with sigma={sigma}: {e}")

        # Save sigma sweep results
        if sigma_results:
            import csv
            sweep_path = logger.log_dir / "sigma_sweep_results.csv"
            with open(sweep_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['sigma', 'img_auc', 'pixel_auc', 'img_ap', 'pixel_ap'])
                writer.writeheader()
                writer.writerows(sigma_results)
            print(f"\nSigma sweep results saved to: {sweep_path}")
            # Find best sigma
            best = max(sigma_results, key=lambda x: x['pixel_ap'])
            print(f"Best sigma for P-AP: {best['sigma']} (P-AP={best['pixel_ap']:.4f})")

            # Report per-class optimal sigma
            if per_class_best_sigma:
                print(f"\nPer-class optimal sigma:")
                adaptive_path = logger.log_dir / "per_class_optimal_sigma.csv"
                with open(adaptive_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['class_name', 'best_sigma', 'pixel_ap'])
                    for cls_name in sorted(per_class_best_sigma.keys()):
                        s, pap = per_class_best_sigma[cls_name]
                        print(f"  {cls_name}: sigma={s}, P-AP={pap:.4f}")
                        writer.writerow([cls_name, s, pap])
                print(f"Per-class sigma saved to: {adaptive_path}")

                # Run per-class adaptive sigma evaluation
                print(f"\n--- Per-Class Adaptive Sigma Evaluation ---")
                adaptive_sigma_dict = {cls: s for cls, (s, _) in per_class_best_sigma.items()}
                args.score_smooth_sigma = 0.0  # Disable global sigma (per-class overrides)
                adaptive_result = evaluate_all_tasks(
                    trainer, args, use_router=use_router,
                    target_size=args.msk_size,
                    per_class_sigma=adaptive_sigma_dict
                )
                if adaptive_result:
                    print(f"\n  Adaptive Sigma Result:")
                    print(f"    I-AUC={adaptive_result.get('mean_img_auc',0):.4f}, "
                          f"P-AUC={adaptive_result.get('mean_pixel_auc',0):.4f}, "
                          f"P-AP={adaptive_result.get('mean_pixel_ap',0):.4f}")
                    # Save adaptive results
                    adaptive_results_path = logger.log_dir / "adaptive_sigma_results.csv"
                    with open(adaptive_results_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['class_name', 'sigma', 'img_auc', 'pixel_auc', 'img_ap', 'pixel_ap'])
                        classes = adaptive_result.get('classes', [])
                        img_aucs_list = adaptive_result.get('img_aucs', [])
                        pixel_aucs_list = adaptive_result.get('pixel_aucs', [])
                        img_aps_list = adaptive_result.get('img_aps', [])
                        pixel_aps_list = adaptive_result.get('pixel_aps', [])
                        for i, cls in enumerate(classes):
                            writer.writerow([cls, adaptive_sigma_dict.get(cls, 0.0),
                                           img_aucs_list[i], pixel_aucs_list[i],
                                           img_aps_list[i], pixel_aps_list[i]])
                    print(f"  Adaptive results saved to: {adaptive_results_path}")

        # Restore original sigma
        args.score_smooth_sigma = ablation_config.score_smooth_sigma

    # Run Flow diagnostics if requested
    if parsed_args.run_diagnostics:
        print("\n" + "="*70)
        print("Running Flow Diagnostics (scale(s) analysis)")
        print("="*70)

        diagnostics_dir = logger.log_dir / "diagnostics"
        diagnostics = FlowDiagnostics(save_dir=str(diagnostics_dir), max_samples=500)

        # Collect diagnostics for each task
        for task_id, task_classes in enumerate(CONTINUAL_TASKS):
            print(f"\nCollecting diagnostics for Task {task_id}: {task_classes}")

            trainer.nf_model.set_active_task(task_id)
            trainer.nf_model.eval()
            trainer.vit_extractor.eval()

            for cls_name in task_classes:
                try:
                    # Create test dataset
                    args.class_to_idx = {cls_name: GLOBAL_CLASS_TO_IDX[cls_name]}
                    args.n_classes = 1

                    test_dataset = create_task_dataset(args, [cls_name], GLOBAL_CLASS_TO_IDX, train=False)
                    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

                    with torch.no_grad():
                        for images, labels, masks, _, _ in test_loader:
                            images = images.to(device)

                            # Extract features
                            patch_embeddings, spatial_shape = feature_extractor(images, return_spatial_shape=True)

                            if ablation_config.use_pos_embedding:
                                patch_embeddings_with_pos = pos_embed_generator(spatial_shape, patch_embeddings)
                            else:
                                B = patch_embeddings.shape[0]
                                H, W = spatial_shape
                                patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)

                            # Forward through NF
                            z, logdet_patch = nf_model.forward(patch_embeddings_with_pos, reverse=False)

                            # Collect
                            is_anomaly = (labels > 0).long()
                            diagnostics.collect(
                                z=z,
                                logdet_patch=logdet_patch,
                                is_anomaly=is_anomaly,
                                images=images,
                                masks=masks,
                                class_name=cls_name
                            )
                except Exception as e:
                    print(f"Failed to collect diagnostics for {cls_name}: {e}")
                    continue

        # Generate diagnostic plots
        diagnostics.analyze_and_save(task_id=len(CONTINUAL_TASKS)-1)
        print(f"\nDiagnostics saved to: {diagnostics_dir}")

    # Close logger
    logger.close()

    return trainer, results, oracle_results


if __name__ == '__main__':
    main()
