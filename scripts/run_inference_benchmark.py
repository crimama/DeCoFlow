#!/usr/bin/env python
"""
DeCoFlow Inference Time Benchmark.

Measures per-component and end-to-end inference latency (ms/image) for the
DeCoFlow continual anomaly detection pipeline. Results are formatted for
direct inclusion in a paper table.

Pipeline stages measured:
  1. Backbone  -- Feature extraction (CNN or ViT, frozen)
  2. Router    -- Prototype-based task routing (Mahalanobis distance)
  3. NF        -- Normalizing Flow forward pass (PE + InputAdapter + base + LoRA + ACL)
  4. Score Map -- Anomaly score computation + bicubic upsampling to msk_size

Note on Router timing:
  The actual trainer.inference() calls get_image_level_features() which runs
  the backbone a second time. This benchmark measures that faithfully --
  the "Router" column includes the second backbone pass + distance computation,
  matching the real end-to-end latency. An optimized pipeline could reuse
  features and save ~50% of the Router time.

Usage:
    python scripts/run_inference_benchmark.py
    python scripts/run_inference_benchmark.py --gpu_id 0 --num_test 200
    python scripts/run_inference_benchmark.py --checkpoint_path /path/to/checkpoints/latest
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from decoflow import (
    DeCoFlowNF,
    DeCoFlowContinualTrainer,
    PositionalEmbeddingGenerator,
    create_feature_extractor,
    get_backbone_type,
    get_config,
    init_seeds,
)
from decoflow.config.ablation import AblationConfig
from decoflow.utils.checkpoint import load_checkpoint
from decoflow.data import get_dataset_class


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeCoFlow Inference Time Benchmark"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/Volume/DeCoFlow/logs/V48_01_H04_highres_clean/checkpoints/latest",
        help="Path to checkpoint directory (must contain config.json, nf_model.pth, router.pth)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=6)
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--data_path", type=str, default="/Volume/MVTecAD")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1 for per-image timing)",
    )
    return parser.parse_args()


class InferenceTimer:
    """Accumulates GPU-synchronised elapsed times for named stages."""

    def __init__(self, device: torch.device):
        self.device = device
        self.records = {}  # stage_name -> list of elapsed_ms

    def start(self):
        """Synchronise GPU and record a start timestamp."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()

    def stop(self, stage_name: str):
        """Synchronise GPU, compute elapsed time, and store it."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self.records.setdefault(stage_name, []).append(elapsed_ms)


def build_ablation_config_from_dict(ablation_dict: dict) -> AblationConfig:
    """Build an AblationConfig from the ablation sub-dict in config.json."""
    cfg = AblationConfig()
    for key, value in ablation_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def parse_cl_scenario(scenario: str, all_classes: list) -> list:
    """Parse CL scenario string into task groupings."""
    parts = scenario.split("-")
    base_size, inc_size = int(parts[0]), int(parts[1])
    tasks = [all_classes[:base_size]]
    remaining = all_classes[base_size:]
    while remaining:
        tasks.append(remaining[:inc_size])
        remaining = remaining[inc_size:]
    return tasks


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # GPU setup
    # ------------------------------------------------------------------
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Load checkpoint config
    # ------------------------------------------------------------------
    config_path = os.path.join(args.checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        ckpt_config = json.load(f)

    task_classes_list = ckpt_config["task_classes"]
    backbone_name = ckpt_config["backbone_name"]
    embed_dim = ckpt_config["embed_dim"]
    num_coupling_layers = ckpt_config["num_coupling_layers"]
    lora_rank = ckpt_config["lora_rank"]
    lora_alpha = ckpt_config.get("lora_alpha", 1.0)
    img_size = ckpt_config["img_size"]
    msk_size = ckpt_config.get("msk_size", 256)
    use_high_res = ckpt_config.get("use_high_res", False)
    cl_scenario = ckpt_config.get("cl_scenario", "1-1")
    seed = ckpt_config.get("seed", 0)
    ablation_dict = ckpt_config.get("ablation", {})
    score_smooth_sigma = ckpt_config.get("score_smooth_sigma", 0.0)

    ablation_config = build_ablation_config_from_dict(ablation_dict)
    init_seeds(seed)

    continual_tasks = parse_cl_scenario(cl_scenario, task_classes_list)
    num_tasks = len(continual_tasks)
    acl_n_layers = ablation_dict.get("acl_n_layers", 0)

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------
    print("=" * 70)
    print("DeCoFlow Inference Benchmark")
    print("=" * 70)
    print(f"  Checkpoint  : {args.checkpoint_path}")
    print(f"  Backbone    : {backbone_name}")
    print(f"  Embed Dim   : {embed_dim}")
    print(f"  DCL + ACL   : {num_coupling_layers} + {acl_n_layers}")
    print(f"  LoRA Rank   : {lora_rank}")
    print(f"  Image Size  : {img_size}x{img_size}")
    print(f"  Mask Size   : {msk_size}x{msk_size}")
    print(f"  High-Res    : {use_high_res}")
    print(f"  Tasks       : {num_tasks} ({cl_scenario})")
    print(f"  Device      : {device} (GPU {args.gpu_id})")
    print(f"  Warmup      : {args.num_warmup} iterations")
    print(f"  Test        : {args.num_test} iterations")
    print(f"  Batch Size  : {args.batch_size}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Build model components
    # ------------------------------------------------------------------
    backbone_type = get_backbone_type(backbone_name)

    feature_extractor = create_feature_extractor(
        backbone_name=backbone_name,
        input_shape=(3, img_size, img_size),
        target_embed_dimension=embed_dim,
        device=device,
        blocks_to_extract=[9, 10, 11] if backbone_type == "vit" else None,
        remove_cls_token=True,
        patch_size=3,
        patch_stride=1,
        use_high_res=use_high_res if backbone_type == "cnn" else False,
    )

    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    nf_model = DeCoFlowNF(
        embed_dim=embed_dim,
        coupling_layers=num_coupling_layers,
        clamp_alpha=1.9,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=device,
        ablation_config=ablation_config,
    )

    cfg = get_config(
        img_size=img_size,
        msk_size=msk_size,
        data_path=args.data_path,
        batch_size=args.batch_size,
        seed=seed,
    )
    cfg.dataset = args.dataset
    cfg.enable_slow_stage = False
    cfg.score_smooth_sigma = score_smooth_sigma
    cfg.interp_mode = "bicubic"

    trainer = DeCoFlowContinualTrainer(
        vit_extractor=feature_extractor,
        pos_embed_generator=pos_embed_generator,
        nf_model=nf_model,
        args=cfg,
        device=device,
        ablation_config=ablation_config,
    )

    # Pre-create task structure and LoRA adapters before loading state_dict
    for tid, tc in enumerate(continual_tasks):
        trainer.task_classes[tid] = tc
        nf_model.add_task(tid)
        nf_model.set_active_task(tid)

    print("\nLoading checkpoint...")
    # load_checkpoint expects the base checkpoints/ dir and appends /latest internally
    ckpt_base = args.checkpoint_path
    if ckpt_base.endswith("/latest") or ckpt_base.endswith("/latest/"):
        ckpt_base = os.path.dirname(ckpt_base.rstrip("/"))
    load_checkpoint(
        nf_model=nf_model,
        router=trainer.router,
        checkpoint_dir=ckpt_base,
        device=device,
    )
    print(f"Checkpoint loaded ({num_tasks} tasks).\n")

    nf_model.eval()
    feature_extractor.eval()

    # ------------------------------------------------------------------
    # Prepare test data
    # ------------------------------------------------------------------
    test_class = task_classes_list[0]
    DatasetClass = get_dataset_class(args.dataset)
    test_dataset = DatasetClass(
        args.data_path,
        class_name=test_class,
        train=False,
        img_size=img_size,
        crp_size=img_size,
        msk_size=msk_size,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    total_iters = args.num_warmup + args.num_test
    print(f"Test class: {test_class} ({len(test_dataset)} images)")
    print(f"Iterations needed: {total_iters} (will cycle dataset if necessary)")
    print()

    use_router = (
        ablation_config.use_router
        and trainer.router is not None
        and len(trainer.router.prototypes) > 0
    )

    # ------------------------------------------------------------------
    # Benchmark loop
    # ------------------------------------------------------------------
    timer = InferenceTimer(device)
    iters_done = 0

    with torch.no_grad():
        while iters_done < total_iters:
            for images, labels, masks, _, _ in test_loader:
                if iters_done >= total_iters:
                    break

                images = images.to(device, non_blocking=True)
                B = images.shape[0]

                # ============ Stage 1: Backbone ============
                timer.start()
                patch_embeddings, spatial_shape = feature_extractor(
                    images, return_spatial_shape=True
                )
                timer.stop("backbone")

                # ============ Stage 2: Router ============
                # Faithfully reproduces trainer.inference() behaviour:
                # calls get_image_level_features() which runs backbone again.
                timer.start()
                if use_router:
                    image_features = feature_extractor.get_image_level_features(
                        images
                    )
                    predicted_tasks = trainer.router.route(image_features)
                else:
                    predicted_tasks = torch.zeros(
                        B, dtype=torch.long, device=device
                    )
                timer.stop("router")

                # ============ Stage 3: NF Forward ============
                timer.start()
                H, W = spatial_shape

                # Positional embedding
                if ablation_config.use_pos_embedding:
                    pe_input = pos_embed_generator(spatial_shape, patch_embeddings)
                else:
                    pe_input = patch_embeddings.reshape(B, H, W, -1)

                # Per-task NF forward (matches trainer.inference logic)
                unique_tasks = predicted_tasks.unique()
                anomaly_scores = torch.zeros(B, H, W, device=device)
                image_scores = torch.zeros(B, device=device)

                for t_id in unique_tasks:
                    tmask = predicted_tasks == t_id
                    task_pe = pe_input[tmask]
                    if task_pe.shape[0] == 0:
                        continue

                    nf_model.set_active_task(t_id.item())
                    z, logdet_patch = nf_model.forward(task_pe, reverse=False)
                    D = z.shape[-1]

                    # Anomaly score = -log p(x) = -(log p(z) + log|det J|)
                    log_pz = (
                        -0.5 * (z ** 2).sum(dim=-1)
                        - 0.5 * D * math.log(2 * math.pi)
                    )
                    task_anomaly = -(log_pz + logdet_patch)

                    # Image-level aggregation (top_k, matching trainer default)
                    flat = task_anomaly.reshape(task_pe.shape[0], -1)
                    k = min(
                        getattr(ablation_config, "score_aggregation_top_k", 3),
                        flat.shape[1],
                    )
                    top_k_vals, _ = torch.topk(flat, k, dim=1)
                    task_img_scores = top_k_vals.mean(dim=1)

                    anomaly_scores[tmask] = task_anomaly
                    image_scores[tmask] = task_img_scores

                timer.stop("nf_forward")

                # ============ Stage 4: Score Map (Upsampling) ============
                timer.start()
                score_map = F.interpolate(
                    anomaly_scores.unsqueeze(1),
                    size=(msk_size, msk_size),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)
                timer.stop("score_map")

                iters_done += 1

                if iters_done % 20 == 0 or iters_done == total_iters:
                    print(
                        f"\r  Progress: {iters_done}/{total_iters}",
                        end="",
                        flush=True,
                    )

    print("\n")

    # ------------------------------------------------------------------
    # Report results
    # ------------------------------------------------------------------
    n_skip = args.num_warmup
    backbone_ms = np.array(timer.records["backbone"][n_skip:])
    router_ms = np.array(timer.records["router"][n_skip:])
    nf_ms = np.array(timer.records["nf_forward"][n_skip:])
    score_ms = np.array(timer.records["score_map"][n_skip:])
    total_ms = backbone_ms + router_ms + nf_ms + score_ms

    bs = args.batch_size
    backbone_per = backbone_ms / bs
    router_per = router_ms / bs
    nf_per = nf_ms / bs
    score_per = score_ms / bs
    total_per = total_ms / bs

    n_measured = len(backbone_per)

    print("=" * 70)
    print("INFERENCE BENCHMARK RESULTS")
    print("=" * 70)
    print(
        f"  Model   : {backbone_name}, {embed_dim}d, "
        f"DCL{num_coupling_layers}+ACL{acl_n_layers}, LoRA r={lora_rank}"
    )
    print(f"  Input   : {img_size}x{img_size} -> Score Map: {msk_size}x{msk_size}")
    if use_high_res:
        print(f"  Features: 56x56 (high-res from layer1+layer2)")
    print(f"  BS={bs}, {n_measured} iters measured ({n_skip} warmup skipped)")
    print()
    print(f"  {'Component':<20s}  {'Mean':>8s}  {'Std':>8s}  {'Unit':<6s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*6}")
    print(f"  {'Backbone':<20s}  {backbone_per.mean():>8.2f}  {backbone_per.std():>8.2f}  ms/img")
    print(f"  {'Router':<20s}  {router_per.mean():>8.2f}  {router_per.std():>8.2f}  ms/img")
    print(f"  {'NF Forward':<20s}  {nf_per.mean():>8.2f}  {nf_per.std():>8.2f}  ms/img")
    print(f"  {'Score Map':<20s}  {score_per.mean():>8.2f}  {score_per.std():>8.2f}  ms/img")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*6}")
    print(f"  {'TOTAL':<20s}  {total_per.mean():>8.2f}  {total_per.std():>8.2f}  ms/img")
    print()

    throughput = 1000.0 / total_per.mean()
    batch_fps = 1000.0 / total_ms.mean() * bs
    print(f"  Throughput      : {throughput:.1f} img/s  (single-image latency)")
    if bs > 1:
        print(f"  Batch Throughput: {batch_fps:.1f} img/s  (batch={bs})")
    print()

    # Percentage breakdown
    total_mean = total_per.mean()
    print(f"  Breakdown:")
    print(f"    Backbone  : {backbone_per.mean()/total_mean*100:5.1f}%")
    print(f"    Router    : {router_per.mean()/total_mean*100:5.1f}%")
    print(f"    NF Forward: {nf_per.mean()/total_mean*100:5.1f}%")
    print(f"    Score Map : {score_per.mean()/total_mean*100:5.1f}%")
    print()

    # Paper-ready compact summary
    print("-" * 50)
    print("Paper-ready (copy-paste):")
    print("-" * 50)
    print(f"  Backbone   : {backbone_per.mean():.1f} ms")
    print(f"  Router     : {router_per.mean():.1f} ms")
    print(f"  NF Forward : {nf_per.mean():.1f} ms")
    print(f"  Score Map  : {score_per.mean():.1f} ms")
    print(f"  Total      : {total_per.mean():.1f} ms/image ({throughput:.0f} FPS)")
    print("-" * 50)

    # LaTeX table row
    print()
    print("LaTeX table row:")
    print(
        f"  DeCoFlow & {backbone_per.mean():.1f} & {router_per.mean():.1f} "
        f"& {nf_per.mean():.1f} & {score_per.mean():.1f} "
        f"& {total_per.mean():.1f} & {throughput:.0f} \\\\"
    )

    # Hardware and model info
    print()
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"  GPU              : {gpu_name}")
        print(f"  Peak GPU Memory  : {gpu_mem_mb:.0f} MB")

    nf_params = sum(p.numel() for p in nf_model.parameters())
    bb_params = sum(p.numel() for p in feature_extractor.parameters())
    print(f"  NF Parameters    : {nf_params:,} ({nf_params/1e6:.2f}M)")
    print(f"  Backbone Params  : {bb_params:,} ({bb_params/1e6:.2f}M)")
    print("=" * 70)


if __name__ == "__main__":
    main()
