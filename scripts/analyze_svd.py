#!/usr/bin/env python3
"""SVD Analysis for Table 8 (tab:svd).
Trains full fine-tuning for 1 task, extracts delta_W, performs SVD.
"""
import os, sys, argparse, csv, json, math
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from decoflow.extractors import create_feature_extractor, get_backbone_type
from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.models.position_embedding import PositionalEmbeddingGenerator
from decoflow.config.ablation import AblationConfig
from decoflow.data.mvtec import MVTEC
from torch.utils.data import DataLoader

MVTEC_CLASSES = [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"
]

def extract_subnet_weights(nf_model):
    """Extract all subnet linear layer weights as a dict."""
    weights = {}
    for name, param in nf_model.named_parameters():
        if "subnet" in name and "weight" in name and param.dim() == 2:
            weights[name] = param.detach().cpu().clone()
    return weights

def compute_svd_analysis(w_init, w_trained):
    """Compute SVD of weight delta and return energy spectrum."""
    results = []
    for name in w_init:
        if name not in w_trained:
            continue
        delta = w_trained[name] - w_init[name]
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        total_energy = (S ** 2).sum().item()
        if total_energy < 1e-12:
            continue
        cumulative = torch.cumsum(S ** 2, dim=0) / total_energy
        eff_rank = (cumulative < 0.99).sum().item() + 1
        results.append({
            "name": name,
            "shape": list(delta.shape),
            "effective_rank": eff_rank,
            "total_rank": len(S),
            "top_16_energy": cumulative[min(15, len(cumulative)-1)].item() if len(S) > 0 else 0,
            "top_64_energy": cumulative[min(63, len(cumulative)-1)].item() if len(S) > 63 else cumulative[-1].item(),
            "singular_values": S.numpy().tolist()[:100],
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_path", default="/Volume/MVTecAD")
    parser.add_argument("--output_dir", default="logs/5_Analysis/SVD_HR")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SVD Analysis (Table 8)")
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

    # === Step 1: Create fresh model and save initial weights ===
    print("")
    print("--- Step 1: Capturing initial weights ---")
    ablation_no_lora = AblationConfig(use_lora=False, use_tsa=False, use_acb=False, scale_context_kernel=cfg.get("ablation", {}).get("scale_context_kernel", 5))
    model_init = DeCoFlowNF(embed_dim=embed_dim, coupling_layers=ncl, clamp_alpha=1.9,
        lora_rank=lora_rank, lora_alpha=1.0, device=device, ablation_config=ablation_no_lora)
    w_init = extract_subnet_weights(model_init)
    print("  Captured {} subnet weight tensors".format(len(w_init)))

    # === Step 2: Train full fine-tuning for Task 0 (bottle) ===
    print("")
    print("--- Step 2: Training full FT on Task 0 (bottle) ---")
    pos_embed_gen = PositionalEmbeddingGenerator(device=device)
    model_init.train()
    optimizer = torch.optim.AdamW(model_init.parameters(), lr=3e-4, weight_decay=1e-4)

    dataset = MVTEC(root=args.data_path, class_name="bottle", train=True,
                    img_size=img_size, crp_size=img_size, msk_size=img_size)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)

    num_epochs = 60
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            imgs = batch[0].to(device)
            with torch.no_grad():
                feats = feature_extractor(imgs)
                B_f, H_f, W_f, D_f = feats.shape; feats = pos_embed_gen((H_f, W_f), feats)
            z, logdet = model_init(feats)
            B, H, W, D = z.shape
            log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            nll = -(log_pz + logdet)
            loss = nll.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print("  Epoch {}/{}: Loss={:.2f}".format(epoch+1, num_epochs, total_loss/len(loader)))

    # === Step 3: Extract trained weights and compute SVD ===
    print("")
    print("--- Step 3: SVD analysis ---")
    w_trained = extract_subnet_weights(model_init)
    svd_results = compute_svd_analysis(w_init, w_trained)

    all_eff_ranks = [r["effective_rank"] for r in svd_results]
    all_total_ranks = [r["total_rank"] for r in svd_results]
    all_top16 = [r["top_16_energy"] for r in svd_results]
    all_top64 = [r["top_64_energy"] for r in svd_results]

    mean_eff_rank = np.mean(all_eff_ranks)
    mean_total_rank = np.mean(all_total_ranks)
    mean_top16 = np.mean(all_top16) * 100
    mean_top64 = np.mean(all_top64) * 100

    print("")
    print("=" * 70)
    print("Table 8: SVD Analysis Results")
    print("=" * 70)
    print("Mean Effective Rank (99%% energy): {:.0f}".format(mean_eff_rank))
    print("Mean Total Rank: {:.0f}".format(mean_total_rank))
    print("Energy captured by rank-16: {:.1f}%%".format(mean_top16))
    print("Energy captured by rank-64: {:.1f}%%".format(mean_top64))
    print("Number of layers analyzed: {}".format(len(svd_results)))

    csv_path = os.path.join(args.output_dir, "svd_analysis_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer","shape","effective_rank","total_rank","top16_energy","top64_energy"])
        for r in svd_results:
            w.writerow([r["name"], str(r["shape"]), r["effective_rank"],
                        r["total_rank"], "{:.4f}".format(r["top_16_energy"]),
                        "{:.4f}".format(r["top_64_energy"])])
        w.writerow(["MEAN", "", "{:.0f}".format(mean_eff_rank),
                    "{:.0f}".format(mean_total_rank),
                    "{:.4f}".format(mean_top16/100), "{:.4f}".format(mean_top64/100)])
    print("Saved to {}".format(csv_path))

    summary_path = os.path.join(args.output_dir, "svd_summary.json")
    summary = {
        "mean_effective_rank": float(mean_eff_rank),
        "mean_total_rank": float(mean_total_rank),
        "energy_at_rank16_pct": float(mean_top16),
        "energy_at_rank64_pct": float(mean_top64),
        "num_layers": len(svd_results),
        "per_layer": [{"name": r["name"], "eff_rank": r["effective_rank"],
                       "top16": r["top_16_energy"], "top64": r["top_64_energy"]}
                      for r in svd_results]
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to {}".format(summary_path))
    print("Done!")

if __name__ == "__main__":
    main()
