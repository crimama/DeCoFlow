"""
Checkpoint utilities for saving and loading model state.

Saves after each task:
- NF model state_dict (base weights, LoRA adapters, ACL layers, input adapters)
- Router prototypes (mean, covariance, precision per task)
- Reference feature statistics (mean, std)
- Config snapshot
"""

import os
import json
import shutil
import torch
from typing import Optional, Dict, Any


def save_checkpoint(
    nf_model,
    router,
    task_id: int,
    save_dir: str,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Save full model state after a task completes.

    Args:
        nf_model: DeCoFlowNF model
        router: PrototypeRouter (or None)
        task_id: Current task ID
        save_dir: Base directory for checkpoints (e.g., logs/experiment/checkpoints)
        config: Optional config dict to save alongside
    """
    task_dir = os.path.join(save_dir, f"task_{task_id}")
    os.makedirs(task_dir, exist_ok=True)

    # 1. Save NF model state_dict (includes base, LoRA, ACL, input_adapters)
    model_path = os.path.join(task_dir, "nf_model.pth")
    torch.save(nf_model.state_dict(), model_path)

    # 2. Save reference statistics
    ref_stats = nf_model.reference_stats
    if ref_stats.is_initialized:
        stats_path = os.path.join(task_dir, "reference_stats.pth")
        torch.save({
            'mean': ref_stats.mean,
            'std': ref_stats.std,
            'n_samples': ref_stats.n_samples,
        }, stats_path)

    # 3. Save router prototypes
    if router is not None and hasattr(router, 'prototypes'):
        router_data = {}
        for tid, proto in router.prototypes.items():
            router_data[str(tid)] = {
                'task_id': proto.task_id,
                'task_classes': proto.task_classes,
                'mean': proto.mean,
                'covariance': proto.covariance,
                'precision': proto.precision,
                'n_samples': proto.n_samples,
            }
        router_path = os.path.join(task_dir, "router.pth")
        torch.save(router_data, router_path)

    # 4. Save config snapshot
    if config is not None:
        config_path = os.path.join(task_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    # Also maintain a "latest" symlink/copy for convenience
    latest_dir = os.path.join(save_dir, "latest")
    if os.path.exists(latest_dir):
        if os.path.islink(latest_dir):
            os.unlink(latest_dir)
        else:
            shutil.rmtree(latest_dir)
    os.symlink(os.path.abspath(task_dir), latest_dir)

    size_mb = sum(
        os.path.getsize(os.path.join(task_dir, f))
        for f in os.listdir(task_dir)
        if os.path.isfile(os.path.join(task_dir, f))
    ) / (1024 * 1024)
    print(f"💾 Checkpoint saved: {task_dir} ({size_mb:.1f} MB)")


def load_checkpoint(
    nf_model,
    router,
    checkpoint_dir: str,
    device: str = 'cuda',
    task_id: Optional[int] = None,
):
    """
    Load model state from checkpoint.

    Args:
        nf_model: DeCoFlowNF model to load weights into
        router: PrototypeRouter to load prototypes into (or None)
        checkpoint_dir: Base checkpoint directory (e.g., logs/experiment/checkpoints)
        device: Device to load tensors onto
        task_id: Specific task checkpoint to load. If None, loads 'latest'.

    Returns:
        config: Config dict from checkpoint (or None)
    """
    if task_id is not None:
        task_dir = os.path.join(checkpoint_dir, f"task_{task_id}")
    else:
        task_dir = os.path.join(checkpoint_dir, "latest")
        if os.path.islink(task_dir):
            task_dir = os.readlink(task_dir)

    if not os.path.exists(task_dir):
        raise FileNotFoundError(f"Checkpoint not found: {task_dir}")

    # 1. Load NF model
    model_path = os.path.join(task_dir, "nf_model.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        nf_model.load_state_dict(state_dict)
        print(f"✅ NF model loaded from {model_path}")

    # 2. Load reference statistics
    stats_path = os.path.join(task_dir, "reference_stats.pth")
    if os.path.exists(stats_path):
        stats_data = torch.load(stats_path, map_location=device, weights_only=True)
        nf_model.reference_stats.mean = stats_data['mean']
        nf_model.reference_stats.std = stats_data['std']
        nf_model.reference_stats.n_samples = stats_data['n_samples']
        nf_model.reference_stats.is_initialized = True
        print(f"✅ Reference stats loaded (n_samples={stats_data['n_samples']})")

    # 3. Load router prototypes
    if router is not None:
        router_path = os.path.join(task_dir, "router.pth")
        if os.path.exists(router_path):
            from decoflow.models.routing import TaskPrototype
            router_data = torch.load(router_path, map_location=device, weights_only=False)
            router.prototypes.clear()
            for tid_str, pdata in router_data.items():
                tid = int(tid_str)
                proto = TaskPrototype(
                    task_id=pdata['task_id'],
                    task_classes=pdata['task_classes'],
                    device=device,
                )
                proto.mean = pdata['mean'].to(device)
                proto.covariance = pdata['covariance'].to(device)
                proto.precision = pdata['precision'].to(device)
                proto.n_samples = pdata['n_samples']
                router.prototypes[tid] = proto
            print(f"✅ Router loaded ({len(router.prototypes)} prototypes)")

    # 4. Load config
    config = None
    config_path = os.path.join(task_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    return config


def cleanup_checkpoint(save_dir: str):
    """
    Delete entire checkpoint directory to free space.

    Args:
        save_dir: Base checkpoint directory to remove
    """
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"🗑️ Checkpoint deleted: {save_dir}")


def get_checkpoint_size_mb(save_dir: str) -> float:
    """Get total size of checkpoint directory in MB."""
    total = 0
    if os.path.exists(save_dir):
        for dirpath, _, filenames in os.walk(save_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    return total / (1024 * 1024)
