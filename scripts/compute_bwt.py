#!/usr/bin/env python3
"""
Compute Backward Transfer (BWT) from continual learning experiment results.

BWT Definition:
  BWT = (1/(T-1)) * sum_{i=0}^{T-2}(R_{T,i} - R_{i,i})

Where:
  - R_{T,i} = performance on task i after learning all T tasks
  - R_{i,i} = performance on task i immediately after learning task i
  - Negative BWT = forgetting
  - BWT = 0 = zero forgetting (perfect)
  - Positive BWT = positive transfer (rare)

Usage:
  python scripts/compute_bwt.py --exp_dir logs_organized/1_Main_Results/MVTec_Main/MVTec-WRN50-60ep-lr2e4-dia4
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_continual_results(exp_dir: Path) -> Dict[int, pd.DataFrame]:
    """
    Load all continual_results_after_task_*.csv files.

    Returns:
        Dict mapping task_id -> DataFrame with results after that task
    """
    results = {}

    for f in sorted(exp_dir.glob("continual_results_after_task_*.csv")):
        task_id = int(f.stem.split("_")[-1])
        df = pd.read_csv(f)
        results[task_id] = df

    return results


def compute_bwt(results: Dict[int, pd.DataFrame], metric: str = "img_auc") -> Tuple[float, Dict[int, float]]:
    """
    Compute Backward Transfer (BWT).

    Args:
        results: Dict of task_id -> DataFrame with continual results
        metric: Which metric to use ('img_auc' or 'pixel_auc')

    Returns:
        (overall_bwt, per_task_bwt) tuple
    """
    num_tasks = len(results)

    if num_tasks < 2:
        return 0.0, {}

    per_task_bwt = {}
    total_bwt = 0.0

    # For each task i (except the last one)
    for task_i in range(num_tasks - 1):
        # R_{i,i} = performance on task i after learning task i
        df_after_i = results[task_i]
        # Filter to get only task i's result (exclude Average rows)
        task_i_rows = df_after_i[df_after_i['task_id'] == task_i]
        if len(task_i_rows) == 0:
            # Try filtering by is_current_task
            task_i_rows = df_after_i[df_after_i['is_current_task'] == True]

        if len(task_i_rows) == 0:
            continue

        R_ii = task_i_rows[metric].values[0]

        # R_{T,i} = performance on task i after learning all tasks
        final_task = num_tasks - 1
        df_after_final = results[final_task]
        task_i_final = df_after_final[df_after_final['task_id'] == task_i]

        if len(task_i_final) == 0:
            continue

        R_Ti = task_i_final[metric].values[0]

        # BWT_i = R_{T,i} - R_{i,i}
        bwt_i = R_Ti - R_ii
        per_task_bwt[task_i] = bwt_i
        total_bwt += bwt_i

    # Average BWT
    if len(per_task_bwt) > 0:
        overall_bwt = total_bwt / len(per_task_bwt)
    else:
        overall_bwt = 0.0

    return overall_bwt, per_task_bwt


def compute_forgetting_measure(results: Dict[int, pd.DataFrame], metric: str = "img_auc") -> Tuple[float, Dict[int, float]]:
    """
    Compute Forgetting Measure (FM) - alternative metric.

    FM_i = max_{j in [i, T-1]} R_{j,i} - R_{T,i}

    This measures the maximum drop from peak performance.
    """
    num_tasks = len(results)
    per_task_fm = {}

    for task_i in range(num_tasks - 1):
        # Track performance on task i across all subsequent tasks
        peak_perf = 0.0
        final_perf = 0.0

        for task_j in range(task_i, num_tasks):
            df = results[task_j]
            task_i_rows = df[df['task_id'] == task_i]

            if len(task_i_rows) == 0:
                continue

            perf = task_i_rows[metric].values[0]

            if task_j == task_i:
                peak_perf = perf
            else:
                peak_perf = max(peak_perf, perf)

            if task_j == num_tasks - 1:
                final_perf = perf

        fm_i = peak_perf - final_perf
        per_task_fm[task_i] = fm_i

    avg_fm = np.mean(list(per_task_fm.values())) if per_task_fm else 0.0

    return avg_fm, per_task_fm


def print_results(exp_name: str, results: Dict[int, pd.DataFrame]):
    """Print BWT and FM analysis."""
    print(f"\n{'='*70}")
    print(f"BWT Analysis: {exp_name}")
    print(f"{'='*70}")

    # Image AUC BWT
    bwt_iauc, per_task_bwt_iauc = compute_bwt(results, "img_auc")
    fm_iauc, per_task_fm_iauc = compute_forgetting_measure(results, "img_auc")

    print(f"\n[Image AUC]")
    print(f"{'Task':<10} {'Initial':<12} {'Final':<12} {'BWT':<12} {'FM':<12}")
    print("-" * 58)

    num_tasks = len(results)
    final_results = results[num_tasks - 1]

    for task_i in range(num_tasks - 1):
        # Initial
        init_df = results[task_i]
        init_row = init_df[init_df['task_id'] == task_i]
        init_val = init_row['img_auc'].values[0] if len(init_row) > 0 else 0

        # Final
        final_row = final_results[final_results['task_id'] == task_i]
        final_val = final_row['img_auc'].values[0] if len(final_row) > 0 else 0

        bwt_i = per_task_bwt_iauc.get(task_i, 0)
        fm_i = per_task_fm_iauc.get(task_i, 0)

        print(f"Task {task_i:<5} {init_val:<12.4f} {final_val:<12.4f} {bwt_i:<+12.4f} {fm_i:<12.4f}")

    print("-" * 58)
    print(f"{'Average':<10} {'':<12} {'':<12} {bwt_iauc:<+12.4f} {fm_iauc:<12.4f}")

    # Summary
    print(f"\n[Summary]")
    print(f"  BWT (Image AUC): {bwt_iauc:+.4f}")
    print(f"  FM (Image AUC):  {fm_iauc:.4f}")

    # Pixel AUC if available
    if 'pixel_auc' in results[0].columns:
        bwt_pauc, _ = compute_bwt(results, "pixel_auc")
        fm_pauc, _ = compute_forgetting_measure(results, "pixel_auc")
        print(f"  BWT (Pixel AUC): {bwt_pauc:+.4f}")
        print(f"  FM (Pixel AUC):  {fm_pauc:.4f}")

    return {
        'bwt_iauc': bwt_iauc,
        'fm_iauc': fm_iauc,
        'per_task_bwt': per_task_bwt_iauc,
        'per_task_fm': per_task_fm_iauc
    }


def main():
    parser = argparse.ArgumentParser(description="Compute BWT from experiment results")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--metric", type=str, default="img_auc", choices=["img_auc", "pixel_auc"])
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    if not exp_dir.exists():
        print(f"Error: {exp_dir} does not exist")
        sys.exit(1)

    results = load_continual_results(exp_dir)

    if not results:
        print(f"Error: No continual_results_after_task_*.csv files found in {exp_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} task results")

    exp_name = exp_dir.name
    metrics = print_results(exp_name, results)

    return metrics


if __name__ == "__main__":
    main()
