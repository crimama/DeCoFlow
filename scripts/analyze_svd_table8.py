#!/usr/bin/env python3
"""
SVD Analysis for Table 8 (tab:svd) — 15-class HR SOTA configuration.

Analyzes the weight delta between Task 0 initial weights and trained weights
to determine the effective rank and energy captured at various LoRA ranks.

Uses V48_01 checkpoint (task_0 vs freshly initialized model).
"""
import sys, os, json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / "logs" / "V48_01_H04_highres_clean" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "5_Analysis" / "SVD_HR"


def get_fresh_model_weights(device='cpu'):
    """Initialize a fresh model to get initial (random) weights."""
    from decoflow.models.decoflow_nf import DeCoFlowNF
    from decoflow.config.ablation import AblationConfig

    ablation = AblationConfig(
        use_lora=True, use_acb=True, acb_n_blocks=2,
        use_tsa=True,
        scale_context_kernel=5,
    )
    model = DeCoFlowNF(
        embed_dim=768, coupling_layers=6, clamp_alpha=1.9,
        lora_rank=64, lora_alpha=1.0, device=device,
        ablation_config=ablation,
    )
    model.add_task(0)
    model.set_active_task(0)
    return {k: v.clone() for k, v in model.state_dict().items()}


def analyze_weight_deltas(init_weights, trained_weights):
    """Compute SVD of weight deltas for all base_linear layers."""
    results = []

    base_keys = [k for k in sorted(trained_weights.keys())
                 if 'base_linear.weight' in k and 'subnet' in k]

    for key in base_keys:
        W_init = init_weights[key].float()
        W_trained = trained_weights[key].float()
        delta = W_trained - W_init

        # SVD
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

        # Energy spectrum
        total_energy = (S ** 2).sum().item()
        if total_energy < 1e-12:
            continue

        cumulative = (S ** 2).cumsum(0) / total_energy

        # Effective rank at thresholds
        eff_rank_90 = (cumulative < 0.90).sum().item() + 1
        eff_rank_95 = (cumulative < 0.95).sum().item() + 1
        eff_rank_99 = (cumulative < 0.99).sum().item() + 1

        # Energy at specific ranks
        energy_at = {}
        for r in [8, 16, 32, 64, 128]:
            if r <= len(S):
                energy_at[r] = (S[:r] ** 2).sum().item() / total_energy * 100
            else:
                energy_at[r] = 100.0

        layer_name = key.replace('.base_linear.weight', '')
        results.append({
            'layer': layer_name,
            'shape': list(delta.shape),
            'delta_norm': delta.norm().item(),
            'base_norm': W_init.norm().item(),
            'relative_change': delta.norm().item() / (W_init.norm().item() + 1e-12),
            'eff_rank_90': eff_rank_90,
            'eff_rank_95': eff_rank_95,
            'eff_rank_99': eff_rank_99,
            'energy_at_rank': energy_at,
            'top_singular_values': S[:10].tolist(),
        })

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cpu'

    print("=" * 60)
    print("SVD Analysis for Table 8 — HR 15-class SOTA")
    print("=" * 60)

    # 1. Get fresh (randomly initialized) weights
    print("\n[1/3] Initializing fresh model weights...")
    torch.manual_seed(0)
    init_weights = get_fresh_model_weights(device)

    # 2. Load trained Task 0 weights
    print("[2/3] Loading V48_01 Task 0 checkpoint...")
    trained_weights = torch.load(
        CHECKPOINT_DIR / "task_0" / "nf_model.pth",
        map_location=device, weights_only=False
    )

    # 3. Analyze
    print("[3/3] Computing SVD of weight deltas...\n")
    results = analyze_weight_deltas(init_weights, trained_weights)

    if not results:
        print("ERROR: No weight changes detected!")
        return

    # Aggregate
    all_eff_rank_95 = [r['eff_rank_95'] for r in results]
    all_energy_16 = [r['energy_at_rank'][16] for r in results]
    all_energy_64 = [r['energy_at_rank'][64] for r in results]
    all_energy_128 = [r['energy_at_rank'][128] for r in results]

    print(f"{'Layer':<50} {'Shape':>12} {'EffRank95':>10} {'E@16':>8} {'E@64':>8} {'E@128':>8}")
    print("-" * 100)
    for r in results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        print(f"{r['layer']:<50} {shape_str:>12} {r['eff_rank_95']:>10} "
              f"{r['energy_at_rank'][16]:>7.1f}% {r['energy_at_rank'][64]:>7.1f}% "
              f"{r['energy_at_rank'][128]:>7.1f}%")

    print("-" * 100)
    print(f"\n=== SUMMARY (for Table 8) ===")
    print(f"Effective Rank (95% energy): {np.mean(all_eff_rank_95):.1f} +/- {np.std(all_eff_rank_95):.1f}")
    print(f"Energy captured at rank 16:  {np.mean(all_energy_16):.1f}% +/- {np.std(all_energy_16):.1f}%")
    print(f"Energy captured at rank 64:  {np.mean(all_energy_64):.1f}% +/- {np.std(all_energy_64):.1f}%")
    print(f"Energy captured at rank 128: {np.mean(all_energy_128):.1f}% +/- {np.std(all_energy_128):.1f}%")

    # Save
    summary = {
        'num_layers': len(results),
        'eff_rank_95_mean': float(np.mean(all_eff_rank_95)),
        'eff_rank_95_std': float(np.std(all_eff_rank_95)),
        'energy_at_rank_16_mean': float(np.mean(all_energy_16)),
        'energy_at_rank_64_mean': float(np.mean(all_energy_64)),
        'energy_at_rank_128_mean': float(np.mean(all_energy_128)),
        'per_layer': results,
    }
    with open(OUTPUT_DIR / 'svd_analysis_hr.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR / 'svd_analysis_hr.json'}")


if __name__ == '__main__':
    main()
