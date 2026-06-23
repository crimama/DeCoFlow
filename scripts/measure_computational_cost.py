#!/usr/bin/env python
"""
EXP-1.5: Computational Cost Measurement for DeCoFlow

Measures:
1. Model parameters (Base, Per-Task LoRA, Total)
2. GPU memory usage (peak, allocated)
3. Training time per task
4. Inference time (per image, per batch)

Usage:
    python scripts/measure_computational_cost.py \
        --task_classes leather grid transistor \
        --num_epochs 10 \
        --output_file ./analysis_results/computational_cost.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from decoflow import (
    DeCoFlowNF,
    DeCoFlowContinualTrainer,
    PositionalEmbeddingGenerator,
    create_feature_extractor,
    get_backbone_type,
    init_seeds,
    setting_lr_parameters,
    get_config,
    create_task_dataset,
)
from decoflow.config import AblationConfig, parse_ablation_args


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    return total_size / (1024 * 1024)


def get_subnet_lora_layers(subnet) -> List[nn.Module]:
    """Get all LoRA layers from a subnet (handles both DCLSubnet and DCLContextSubnet)."""
    layers = []
    # DCLSubnet style (layer1, layer2)
    if hasattr(subnet, 'layer1') and hasattr(subnet, 'layer2'):
        layers.extend([subnet.layer1, subnet.layer2])
    # DCLContextSubnet style (s_layer1, s_layer2, t_layer1, t_layer2)
    if hasattr(subnet, 's_layer1'):
        layers.extend([subnet.s_layer1, subnet.s_layer2, subnet.t_layer1, subnet.t_layer2])
    return layers


def count_lora_parameters_per_task(nf_model: DeCoFlowNF, task_id: int) -> Dict[str, int]:
    """Count LoRA parameters for a specific task with detailed breakdown."""
    task_key = str(task_id)
    lora_params = 0
    task_bias_params = 0
    task_adapter_params = 0
    acb_params = 0

    # Detailed LoRA breakdown
    lora_a_params = 0
    lora_b_params = 0
    num_lora_layers = 0

    # Count LoRA A and B matrices
    for subnet in nf_model.subnets:
        for layer in get_subnet_lora_layers(subnet):
            if hasattr(layer, 'lora_A') and task_key in layer.lora_A:
                lora_a_params += layer.lora_A[task_key].numel()
                num_lora_layers += 1
            if hasattr(layer, 'lora_B') and task_key in layer.lora_B:
                lora_b_params += layer.lora_B[task_key].numel()
            if hasattr(layer, 'task_biases') and task_key in layer.task_biases:
                task_bias_params += layer.task_biases[task_key].numel()

    lora_params = lora_a_params + lora_b_params

    # Count task adapter parameters (TaskSpecificAlignment)
    if task_key in nf_model.input_adapters:
        task_adapter_params = count_parameters(nf_model.input_adapters[task_key])

    # Count ACB parameters for this task
    if hasattr(nf_model, 'acb_adapters') and task_key in nf_model.acb_adapters:
        acb_params = count_parameters(nf_model.acb_adapters[task_key])

    return {
        'lora_params': lora_params,
        'lora_a_params': lora_a_params,
        'lora_b_params': lora_b_params,
        'num_lora_layers': num_lora_layers,
        'task_bias_params': task_bias_params,
        'task_adapter_params': task_adapter_params,
        'acb_params': acb_params,
        'total_per_task': lora_params + task_bias_params + task_adapter_params + acb_params,
        'total_excl_acb': lora_params + task_bias_params + task_adapter_params,
    }


def count_base_parameters(nf_model: DeCoFlowNF) -> Dict[str, int]:
    """Count base (shared) parameters."""
    base_params = 0

    # Subnet base weights
    for subnet in nf_model.subnets:
        for layer in get_subnet_lora_layers(subnet):
            if hasattr(layer, 'base_linear'):
                base_params += count_parameters(layer.base_linear)
        # Also count context conv if DCLContextSubnet
        if hasattr(subnet, 'context_conv'):
            base_params += count_parameters(subnet.context_conv)
            if hasattr(subnet, 'context_scale_param'):
                base_params += 1  # alpha param

    # Spatial context mixer
    if hasattr(nf_model, 'spatial_context') and nf_model.spatial_context is not None:
        base_params += count_parameters(nf_model.spatial_context)

    # Scale context
    if hasattr(nf_model, 'scale_contexts') and nf_model.scale_contexts is not None:
        base_params += count_parameters(nf_model.scale_contexts)

    # TSA adapter
    if hasattr(nf_model, 'task_specific_alignment') and nf_model.task_specific_alignment is not None:
        base_params += count_parameters(nf_model.task_specific_alignment)

    # ACB blocks
    if hasattr(nf_model, 'acb_blocks') and nf_model.acb_blocks is not None:
        base_params += count_parameters(nf_model.acb_blocks)

    return {
        'base_params': base_params
    }


def measure_gpu_memory() -> Dict[str, float]:
    """Measure GPU memory usage."""
    if not torch.cuda.is_available():
        return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0}

    torch.cuda.synchronize()
    return {
        'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
        'reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
        'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
    }


def measure_inference_time(
    trainer: DeCoFlowContinualTrainer,
    test_loader: DataLoader,
    device: torch.device,
    ablation_config: AblationConfig,
    num_warmup: int = 5,
    num_iterations: int = 50
) -> Dict[str, float]:
    """Measure inference time."""
    trainer.nf_model.eval()
    trainer.vit_extractor.eval()

    times = []

    # Get a batch of data
    images, _, _, _, _ = next(iter(test_loader))
    images = images.to(device)
    batch_size = images.shape[0]

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            patch_embeddings, spatial_shape = trainer.vit_extractor(images, return_spatial_shape=True)
            if ablation_config.use_pos_embedding:
                patch_embeddings_with_pos = trainer.pos_embed_generator(spatial_shape, patch_embeddings)
            else:
                B = patch_embeddings.shape[0]
                H, W = spatial_shape
                patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)
            z, logdet = trainer.nf_model.forward(patch_embeddings_with_pos, reverse=False)

        torch.cuda.synchronize()

        # Measure
        for _ in range(num_iterations):
            start_time = time.perf_counter()

            patch_embeddings, spatial_shape = trainer.vit_extractor(images, return_spatial_shape=True)
            if ablation_config.use_pos_embedding:
                patch_embeddings_with_pos = trainer.pos_embed_generator(spatial_shape, patch_embeddings)
            else:
                B = patch_embeddings.shape[0]
                H, W = spatial_shape
                patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)
            z, logdet = trainer.nf_model.forward(patch_embeddings_with_pos, reverse=False)

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            times.append(end_time - start_time)

    avg_batch_time = sum(times) / len(times)
    avg_per_image = avg_batch_time / batch_size

    return {
        'avg_batch_time_ms': avg_batch_time * 1000,
        'avg_per_image_ms': avg_per_image * 1000,
        'batch_size': batch_size,
        'std_batch_time_ms': (sum((t - avg_batch_time)**2 for t in times) / len(times)) ** 0.5 * 1000,
        'throughput_images_per_sec': batch_size / avg_batch_time
    }


def main():
    parser = argparse.ArgumentParser(description='EXP-1.5: Computational Cost Measurement')
    parser.add_argument('--task_classes', type=str, nargs='+',
                        default=['leather', 'grid', 'transistor'],
                        help='Classes to measure')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs per task for timing measurement')
    parser.add_argument('--backbone_name', type=str, default='wide_resnet50_2',
                        help='Backbone model name')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_coupling_layers', type=int, default=6, help='Number of coupling layers')
    parser.add_argument('--acb_n_blocks', type=int, default=2, help='Number of ACB blocks')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD', help='Dataset path')
    parser.add_argument('--output_file', type=str, default='./analysis_results/computational_cost.json',
                        help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-1.5: Computational Cost Measurement")
    print("=" * 70)

    # Initialize seeds
    init_seeds(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Setup ablation config (default config)
    ablation_config = AblationConfig(
        use_lora=True,
        use_router=True,
        use_task_adapter=True,
        use_pos_embedding=True,
        use_tsa=True,
        use_acb=True,
        acb_n_blocks=args.acb_n_blocks,
        use_spatial_context=True,
        use_scale_context=True,
    )

    # Setup configuration
    config = get_config(
        img_size=args.img_size,
        msk_size=256,
        data_path=args.data_path,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=2e-4,
    )
    setting_lr_parameters(config)
    config.dataset = 'mvtec'

    # Get embedding dimension
    embed_dim = 768  # Default for wide_resnet50_2

    # Initialize feature extractor
    backbone_type = get_backbone_type(args.backbone_name)
    feature_extractor = create_feature_extractor(
        backbone_name=args.backbone_name,
        input_shape=(3, args.img_size, args.img_size),
        target_embed_dimension=embed_dim,
        device=device,
        blocks_to_extract=[1, 3, 5, 11] if backbone_type == 'vit' else None,
        remove_cls_token=True,
        patch_size=3,
        patch_stride=1,
    )

    # Count backbone parameters
    backbone_params = count_parameters(feature_extractor.backbone)
    backbone_trainable = count_parameters(feature_extractor.backbone, trainable_only=True)

    print(f"\nBackbone ({args.backbone_name}):")
    print(f"  Total params: {backbone_params:,}")
    print(f"  Trainable params: {backbone_trainable:,} (frozen)")

    # Initialize positional embedding generator
    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    # Initialize DeCoFlow model
    nf_model = DeCoFlowNF(
        embed_dim=embed_dim,
        coupling_layers=args.num_coupling_layers,
        clamp_alpha=1.9,
        lora_rank=args.lora_rank,
        lora_alpha=1.0,
        device=device,
        ablation_config=ablation_config
    )

    # Memory after model initialization
    mem_after_init = measure_gpu_memory()

    # Count base parameters
    base_params = count_base_parameters(nf_model)
    total_nf_params = count_parameters(nf_model)

    print(f"\nDeCoFlow NF Model:")
    print(f"  Total params (before tasks): {total_nf_params:,}")
    print(f"  Base (shared) params: {base_params['base_params']:,}")

    # Initialize trainer
    trainer = DeCoFlowContinualTrainer(
        vit_extractor=feature_extractor,
        pos_embed_generator=pos_embed_generator,
        nf_model=nf_model,
        args=config,
        device=device,
        ablation_config=ablation_config
    )

    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'backbone': args.backbone_name,
            'img_size': args.img_size,
            'num_coupling_layers': args.num_coupling_layers,
            'acb_n_blocks': args.acb_n_blocks,
            'lora_rank': args.lora_rank,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'task_classes': args.task_classes,
        },
        'parameters': {
            'backbone_total': backbone_params,
            'backbone_trainable': backbone_trainable,
            'nf_model_total': total_nf_params,
            'nf_model_base': base_params['base_params'],
        },
        'per_task': {},
        'memory': {
            'after_init': mem_after_init,
        },
        'timing': {},
    }

    # Setup class mappings
    ALL_CLASSES = args.task_classes
    GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}

    # Train and measure each task
    task_training_times = []

    for task_id, task_class in enumerate(args.task_classes):
        print(f"\n{'='*70}")
        print(f"Task {task_id}: {task_class}")
        print(f"{'='*70}")

        # Create task dataset
        config.class_to_idx = {task_class: GLOBAL_CLASS_TO_IDX[task_class]}
        config.n_classes = 1

        train_dataset = create_task_dataset(
            config, [task_class], GLOBAL_CLASS_TO_IDX, train=True,
            use_rotation_aug=False
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=False, drop_last=True
        )

        # Reset peak memory before training
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Train task and measure time
        start_time = time.time()

        trainer.train_task(
            task_id=task_id,
            task_classes=[task_class],
            train_loader=train_loader,
            num_epochs=args.num_epochs,
            lr=2e-4,
            log_interval=50,
            global_class_to_idx=GLOBAL_CLASS_TO_IDX
        )

        training_time = time.time() - start_time
        task_training_times.append(training_time)

        # Memory after task
        mem_after_task = measure_gpu_memory()

        # Count task-specific parameters
        task_params = count_lora_parameters_per_task(nf_model, task_id)

        print(f"\n  Task {task_id} Parameters (Detailed Breakdown):")
        print(f"    LoRA A params: {task_params['lora_a_params']:,}")
        print(f"    LoRA B params: {task_params['lora_b_params']:,}")
        print(f"    LoRA total: {task_params['lora_params']:,} ({task_params['num_lora_layers']} layers)")
        print(f"    Task bias params: {task_params['task_bias_params']:,}")
        print(f"    TSA Adapter params: {task_params['task_adapter_params']:,}")
        print(f"    ACB params: {task_params['acb_params']:,}")
        print(f"    ---")
        print(f"    Total (excl. ACB): {task_params['total_excl_acb']:,}")
        print(f"    Total (incl. ACB): {task_params['total_per_task']:,}")
        print(f"    Training time: {training_time:.2f}s ({training_time/60:.2f}min)")
        print(f"    Peak GPU memory: {mem_after_task['max_allocated_mb']:.1f} MB")

        results['per_task'][f'task_{task_id}'] = {
            'class': task_class,
            'lora_a_params': task_params['lora_a_params'],
            'lora_b_params': task_params['lora_b_params'],
            'lora_params': task_params['lora_params'],
            'num_lora_layers': task_params['num_lora_layers'],
            'task_bias_params': task_params['task_bias_params'],
            'task_adapter_params': task_params['task_adapter_params'],
            'acb_params': task_params['acb_params'],
            'total_excl_acb': task_params['total_excl_acb'],
            'total_per_task': task_params['total_per_task'],
            'training_time_sec': training_time,
            'peak_memory_mb': mem_after_task['max_allocated_mb'],
        }

    # Inference time measurement
    print(f"\n{'='*70}")
    print("Inference Time Measurement")
    print(f"{'='*70}")

    # Create test dataset for inference measurement
    config.class_to_idx = {args.task_classes[0]: GLOBAL_CLASS_TO_IDX[args.task_classes[0]]}
    config.n_classes = 1

    test_dataset = create_task_dataset(
        config, [args.task_classes[0]], GLOBAL_CLASS_TO_IDX, train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    nf_model.set_active_task(0)
    inference_times = measure_inference_time(
        trainer, test_loader, device, ablation_config,
        num_warmup=10, num_iterations=100
    )

    print(f"  Batch size: {inference_times['batch_size']}")
    print(f"  Avg batch time: {inference_times['avg_batch_time_ms']:.2f} ± {inference_times['std_batch_time_ms']:.2f} ms")
    print(f"  Avg per image: {inference_times['avg_per_image_ms']:.2f} ms")
    print(f"  Throughput: {inference_times['throughput_images_per_sec']:.1f} images/sec")

    results['timing']['inference'] = inference_times
    results['timing']['training'] = {
        'total_time_sec': sum(task_training_times),
        'avg_per_task_sec': sum(task_training_times) / len(task_training_times),
        'per_task_times_sec': task_training_times,
    }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Collect detailed task parameter breakdown
    task_param_details = []
    for task_id in range(len(args.task_classes)):
        task_param_details.append(count_lora_parameters_per_task(nf_model, task_id))

    # Calculate averages (excluding Task 0 for per-task overhead)
    if len(args.task_classes) > 1:
        avg_lora = sum(tp['lora_params'] for tp in task_param_details[1:]) / len(task_param_details[1:])
        avg_bias = sum(tp['task_bias_params'] for tp in task_param_details[1:]) / len(task_param_details[1:])
        avg_wa = sum(tp['task_adapter_params'] for tp in task_param_details[1:]) / len(task_param_details[1:])
        avg_acb = sum(tp['acb_params'] for tp in task_param_details[1:]) / len(task_param_details[1:])
        avg_excl_acb = sum(tp['total_excl_acb'] for tp in task_param_details[1:]) / len(task_param_details[1:])
        avg_per_task_params = sum(tp['total_per_task'] for tp in task_param_details[1:]) / len(task_param_details[1:])
    else:
        avg_lora = task_param_details[0]['lora_params']
        avg_bias = task_param_details[0]['task_bias_params']
        avg_wa = task_param_details[0]['task_adapter_params']
        avg_acb = task_param_details[0]['acb_params']
        avg_excl_acb = task_param_details[0]['total_excl_acb']
        avg_per_task_params = task_param_details[0]['total_per_task']

    total_params = total_nf_params
    for task_id in range(1, len(args.task_classes)):
        total_params += task_param_details[task_id]['total_per_task']

    param_ratio = (avg_per_task_params / base_params['base_params']) * 100 if base_params['base_params'] > 0 else 0
    param_ratio_excl_acb = (avg_excl_acb / base_params['base_params']) * 100 if base_params['base_params'] > 0 else 0

    print(f"\nParameter Efficiency:")
    print(f"  Backbone (frozen): {backbone_params:,}")
    print(f"  NF Base params: {base_params['base_params']:,}")
    print(f"\n  Per-Task Breakdown (avg over Task 1+):")
    print(f"    LoRA: {int(avg_lora):,}")
    print(f"    TaskBias: {int(avg_bias):,}")
    print(f"    TaskSpecificAlignment: {int(avg_wa):,}")
    print(f"    ACB: {int(avg_acb):,}")
    print(f"    ---")
    print(f"    Total (excl. ACB): {int(avg_excl_acb):,} ({param_ratio_excl_acb:.2f}% of NF base)")
    print(f"    Total (incl. ACB): {int(avg_per_task_params):,} ({param_ratio:.2f}% of NF base)")
    print(f"\n  Total after {len(args.task_classes)} tasks: {total_params:,}")

    print(f"\nMemory Efficiency:")
    final_mem = measure_gpu_memory()
    print(f"  Final GPU memory: {final_mem['allocated_mb']:.1f} MB")
    print(f"  Peak GPU memory: {final_mem['max_allocated_mb']:.1f} MB")

    print(f"\nTiming:")
    print(f"  Avg training time per task: {results['timing']['training']['avg_per_task_sec']:.1f}s ({results['timing']['training']['avg_per_task_sec']/60:.1f}min)")
    print(f"  Inference per image: {inference_times['avg_per_image_ms']:.2f}ms")

    results['summary'] = {
        'backbone_params': backbone_params,
        'nf_base_params': base_params['base_params'],
        'per_task_breakdown': {
            'lora': int(avg_lora),
            'task_bias': int(avg_bias),
            'task_specific_alignment': int(avg_wa),
            'acb': int(avg_acb),
        },
        'avg_per_task_params': int(avg_per_task_params),
        'avg_per_task_excl_acb': int(avg_excl_acb),
        'per_task_ratio_percent': param_ratio,
        'per_task_ratio_excl_acb_percent': param_ratio_excl_acb,
        'total_params_after_all_tasks': total_params,
        'final_memory_mb': final_mem['allocated_mb'],
        'peak_memory_mb': final_mem['max_allocated_mb'],
        'avg_train_time_per_task_sec': results['timing']['training']['avg_per_task_sec'],
        'inference_per_image_ms': inference_times['avg_per_image_ms'],
    }

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_file}")

    return results


if __name__ == '__main__':
    main()
