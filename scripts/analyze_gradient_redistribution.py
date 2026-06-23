#!/usr/bin/env python3
"""Gradient Redistribution Analysis for Table 9."""
import os, sys, argparse, csv, json, math
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from decoflow.extractors import create_feature_extractor, get_backbone_type
from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.models.position_embedding import PositionalEmbeddingGenerator
from decoflow.config.ablation import AblationConfig
from decoflow.data.mvtec import MVTEC
from decoflow.utils.checkpoint import load_checkpoint
from torch.utils.data import DataLoader

MVTEC_CLASSES = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"
]

def compute_per_patch_gradients(nf_model, features, task_id, tail_weight, tail_ratio=0.02):
    nf_model.train()
    nf_model.zero_grad()
    features_input = features.clone().detach().requires_grad_(True)
    z, logdet_patch = nf_model(features_input)
    B, H, W, D = z.shape
    num_patches = H * W
    log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
    nll_patch = -(log_pz + logdet_patch)
    flat_nll = nll_patch.reshape(B, -1)
    mean_loss = flat_nll.mean()
    k = max(1, int(num_patches * tail_ratio))
    top_k_nll, top_k_idx = torch.topk(flat_nll, k, dim=1)
    tail_loss = top_k_nll.mean()
    if tail_weight == 0.0:
        loss = mean_loss
    else:
        loss = (1 - tail_weight) * mean_loss + tail_weight * tail_loss
    loss.backward()
    grad_feat = features_input.grad
    grad_mag = grad_feat.norm(dim=-1)
    flat_grad = grad_mag.reshape(B, -1)
    gt_list, gn_list = [], []
    for b in range(B):
        tail_mask = torch.zeros(num_patches, dtype=torch.bool, device=features.device)
        tail_mask[top_k_idx[b]] = True
        gt_list.append(flat_grad[b, tail_mask].mean().item())
        gn_list.append(flat_grad[b, ~tail_mask].mean().item())
    return {"grad_at_tail": np.mean(gt_list), "grad_at_nontail": np.mean(gn_list),
            "ratio": np.mean(gt_list) / (np.mean(gn_list) + 1e-10)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_path", default="/Volume/MVTecAD")
    parser.add_argument("--output_dir", default="logs/5_Analysis/Gradient")
    parser.add_argument("--num_batches", type=int, default=30)
    parser.add_argument("--tail_ratio", type=float, default=0.02)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Gradient Redistribution Analysis (Table 9)")
    print("=" * 70)

    with open(os.path.join(args.checkpoint_dir, "task_0", "config.json")) as f:
        cfg = json.load(f)
    backbone_name = cfg.get("backbone_name", "wide_resnet50_2")
    use_high_res = cfg.get("use_high_res", True)
    lora_rank = cfg.get("lora_rank", 64)
    ncl = cfg.get("num_coupling_layers", 6)
    img_size = 256 if use_high_res else 224

    backbone_type = get_backbone_type(backbone_name)
    feature_extractor = create_feature_extractor(
        backbone_name=backbone_name, input_shape=(3, img_size, img_size),
        target_embed_dimension=768, device=device,
        use_high_res=use_high_res if backbone_type == "cnn" else False)

    with torch.no_grad():
        dummy_feat = feature_extractor(torch.randn(1, 3, img_size, img_size).to(device))
        _, fH, fW, embed_dim = dummy_feat.shape
    print("Feature shape: ({}, {}, {})".format(fH, fW, embed_dim))

    pos_embed_gen = PositionalEmbeddingGenerator(device=device)
    ablation_config = AblationConfig(use_lora=True, use_tsa=True, use_acl=True, scale_context_kernel=cfg.get("ablation", {}).get("scale_context_kernel", 5), acl_n_layers=cfg.get("ablation", {}).get("acl_n_layers", 2), use_tail_aware_loss=True, tail_weight=0.85, tail_top_k_ratio=0.02)
    nf_model = DeCoFlowNF(embed_dim=embed_dim, coupling_layers=ncl, clamp_alpha=1.9,
        lora_rank=lora_rank, lora_alpha=1.0, device=device, ablation_config=ablation_config)

    for tid in range(len(MVTEC_CLASSES)):
        nf_model.add_task(tid)
        nf_model.set_active_task(tid)
    load_checkpoint(nf_model, router=None, checkpoint_dir=args.checkpoint_dir, device=device)

    task_id = 0
    nf_model.set_active_task(task_id)
    print("Analyzing task {} ({})".format(task_id, MVTEC_CLASSES[task_id]))

    dataset = MVTEC(root=args.data_path, class_name=MVTEC_CLASSES[task_id], train=True,
                    img_size=img_size, crp_size=img_size, msk_size=img_size)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)

    configs = [("Mean-only", 0.0), ("Tail-Aware", 0.85)]
    all_results = {}
    for name, tw in configs:
        print("--- {} (tw={}) ---".format(name, tw))
        brs = []
        for bi, batch in enumerate(loader):
            if bi >= args.num_batches:
                break
            imgs = batch[0].to(device)
            with torch.no_grad():
                feats = feature_extractor(imgs)
                B_f, H_f, W_f, D_f = feats.shape; feats = pos_embed_gen((H_f, W_f), feats)
            brs.append(compute_per_patch_gradients(nf_model, feats, task_id, tw, args.tail_ratio))
            if (bi + 1) % 10 == 0:
                print("  Batch {}/{}".format(bi+1, args.num_batches))
        avg = {k: np.mean([r[k] for r in brs]) for k in ["grad_at_tail","grad_at_nontail","ratio"]}
        all_results[name] = avg
        print("  Grad@Tail={:.6f}, Grad@NonTail={:.6f}, Ratio={:.1f}x".format(
            avg["grad_at_tail"], avg["grad_at_nontail"], avg["ratio"]))

    mr, tr = all_results["Mean-only"], all_results["Tail-Aware"]
    amp_t = tr["grad_at_tail"] / (mr["grad_at_tail"] + 1e-10)
    amp_n = tr["grad_at_nontail"] / (mr["grad_at_nontail"] + 1e-10)
    amp_r = tr["ratio"] / (mr["ratio"] + 1e-10)

    print("")
    print("=" * 70)
    print("Table 9: TAL Gradient Redistribution")
    print("=" * 70)
    print("{:<15} {:<15} {:<15} {:<10}".format("Config", "Grad@Tail", "Grad@NonTail", "Ratio"))
    print("-" * 55)
    print("{:<15} {:<15.4f} {:<15.4f} {:.2f}x".format("Mean-only", mr["grad_at_tail"], mr["grad_at_nontail"], mr["ratio"]))
    print("{:<15} {:<15.4f} {:<15.4f} {:.2f}x".format("Tail-Aware", tr["grad_at_tail"], tr["grad_at_nontail"], tr["ratio"]))
    print("{:<15} {:<15.1f}x {:<15.2f}x {:.1f}x".format("Amplification", amp_t, amp_n, amp_r))

    csv_path = os.path.join(args.output_dir, "gradient_redistribution_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config","grad_at_tail","grad_at_nontail","ratio"])
        w.writerow(["Mean-only", "{:.6f}".format(mr["grad_at_tail"]),
                    "{:.6f}".format(mr["grad_at_nontail"]), "{:.2f}".format(mr["ratio"])])
        w.writerow(["Tail-Aware", "{:.6f}".format(tr["grad_at_tail"]),
                    "{:.6f}".format(tr["grad_at_nontail"]), "{:.2f}".format(tr["ratio"])])
        w.writerow(["Amplification", "{:.2f}".format(amp_t),
                    "{:.4f}".format(amp_n), "{:.2f}".format(amp_r)])
    print("Saved to {}".format(csv_path))
    print("Done!")

if __name__ == "__main__":
    main()
