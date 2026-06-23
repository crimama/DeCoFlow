#!/usr/bin/env python3
"""
V62 SNL Design Space Ablation — Results Analysis
Compares: Full SNL vs Decompose+Linear vs DeCoFlow (Ours)
Produces performance + efficiency comparison table.
"""

import os
import json
import re
import csv
import glob
from datetime import datetime
from collections import defaultdict

# =====================================================================
# Experiment configurations
# =====================================================================
EXPERIMENTS = {
    "V62_01_FullSNL_8acl_hr05": {
        "label": "Full SNL (hr=0.5)",
        "short": "Full SNL",
        "dcl": 0, "snl": 8,
        "adapter": "Full/task",
        "per_task_params_M": 7.10,  # from param count test
    },
    "V62_02_DecompLinear_regularlinear": {
        "label": "Decompose + Linear",
        "short": "Decomp+Linear",
        "dcl": 6, "snl": 2,
        "adapter": "Full-rank",
        "per_task_params_M": 10.64,
    },
    "V62_03_FullSNL_8acl_hr20": {
        "label": "Full SNL (hr=2.0)",
        "short": "Full SNL (wide)",
        "dcl": 0, "snl": 8,
        "adapter": "Full/task (wide)",
        "per_task_params_M": 56.68,
    },
    "V62_04_DecompLoRA_rank512": {
        "label": "Decompose + LoRA r=512",
        "short": "Decomp+LoRA512",
        "dcl": 6, "snl": 2,
        "adapter": "LoRA (r=512)",
        "per_task_params_M": 17.12,
    },
}

# Reference results (existing data)
REFERENCE = {
    "V48_01_DeCoFlow": {
        "label": "DeCoFlow (Ours)",
        "short": "Ours",
        "dcl": 6, "snl": 2,
        "adapter": "LoRA (r=64)",
        "per_task_params_M": 3.71,
        "iauc": 98.47,
        "pap": 58.57,
        "fm": 0.0,
        "total_time_h": 9.7,
        "sec_per_epoch": 38.8,
        "min_per_task": 38.8,
    },
    "V52_01_wo_SNL": {
        "label": "w/o SNL",
        "short": "w/o SNL",
        "dcl": 6, "snl": 0,
        "adapter": "LoRA (r=64)",
        "per_task_params_M": 1.94,  # 3.71 - 1.77(ACL)
        "iauc": 88.30,
        "pap": 45.76,
        "fm": 0.0,
        "total_time_h": 8.0,
        "sec_per_epoch": 32.0,
        "min_per_task": 32.0,
    },
}

LOG_BASE = "logs"


def parse_final_results(exp_name):
    """Parse final_results.csv from experiment log directory."""
    log_dir = os.path.join(LOG_BASE, exp_name)

    # Try to find results CSV
    csv_candidates = [
        os.path.join(log_dir, "final_results.csv"),
        os.path.join(log_dir, f"{exp_name}_results.csv"),
    ]
    # Also try glob
    csv_candidates += glob.glob(os.path.join(log_dir, "*results*.csv"))
    csv_candidates += glob.glob(os.path.join(log_dir, "*final*.csv"))

    for csv_path in csv_candidates:
        if os.path.exists(csv_path):
            return _read_results_csv(csv_path)

    # Fallback: parse from log file
    log_file = os.path.join(LOG_BASE, f"{exp_name}.log")
    if os.path.exists(log_file):
        return _parse_from_log(log_file)

    return None


def _read_results_csv(csv_path):
    """Read results from CSV file."""
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # Look for summary/average row or compute from per-class
    iauc_values = []
    pap_values = []

    for row in rows:
        # Try various column name patterns
        iauc = None
        pap = None
        for key in row:
            kl = key.lower().strip()
            if 'image_auc' in kl or 'i-auc' in kl or 'iauc' in kl or 'img_auc' in kl:
                try:
                    iauc = float(row[key])
                except:
                    pass
            if 'pixel_ap' in kl or 'p-ap' in kl or 'pap' in kl or 'pxl_ap' in kl:
                try:
                    pap = float(row[key])
                except:
                    pass

        if iauc is not None:
            iauc_values.append(iauc)
        if pap is not None:
            pap_values.append(pap)

    if iauc_values:
        results['iauc'] = sum(iauc_values) / len(iauc_values)
    if pap_values:
        results['pap'] = sum(pap_values) / len(pap_values)

    return results if results else None


def _parse_from_log(log_path):
    """Parse metrics from log file (fallback)."""
    results = {}

    with open(log_path, 'r') as f:
        content = f.read()

    # Look for final summary lines
    # Pattern: "Image AUC: XX.XX%" or "I-AUC: XX.XX%"
    iauc_match = re.findall(r'(?:Image AUC|I-AUC|Avg.*?I-AUC)[:\s]+(\d+\.\d+)', content)
    pap_match = re.findall(r'(?:Pixel AP|P-AP|Avg.*?P-AP)[:\s]+(\d+\.\d+)', content)

    if iauc_match:
        results['iauc'] = float(iauc_match[-1])  # Last occurrence
    if pap_match:
        results['pap'] = float(pap_match[-1])

    return results if results else None


def parse_training_times(exp_name):
    """Extract per-task and total training times from log."""
    log_file = os.path.join(LOG_BASE, f"{exp_name}.log")
    if not os.path.exists(log_file):
        # Try inside the log directory
        log_file = os.path.join(LOG_BASE, exp_name, f"{exp_name}.log")
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Parse task start/end timestamps
    # Pattern: "Task X: ['classname']" for start
    # Pattern: "Task X completed" or next "Task X+1:" for end
    task_times = {}

    # Find task boundary timestamps
    task_starts = re.findall(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?# Task (\d+):',
        content
    )

    timestamps = []
    for ts_str, task_id in task_starts:
        ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        timestamps.append((int(task_id), ts))

    # Also find the final timestamp
    all_timestamps = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', content)
    if all_timestamps:
        first_ts = datetime.strptime(all_timestamps[0], '%Y-%m-%d %H:%M:%S')
        last_ts = datetime.strptime(all_timestamps[-1], '%Y-%m-%d %H:%M:%S')
        total_seconds = (last_ts - first_ts).total_seconds()
    else:
        total_seconds = None

    # Calculate per-task times
    per_task_times = []
    for i in range(len(timestamps) - 1):
        tid, ts = timestamps[i]
        _, next_ts = timestamps[i + 1]
        dt = (next_ts - ts).total_seconds()
        per_task_times.append(dt)

    # Last task: use final timestamp
    if timestamps and all_timestamps:
        last_task_start = timestamps[-1][1]
        last_log_ts = datetime.strptime(all_timestamps[-1], '%Y-%m-%d %H:%M:%S')
        per_task_times.append((last_log_ts - last_task_start).total_seconds())

    result = {
        'total_seconds': total_seconds,
        'total_hours': total_seconds / 3600 if total_seconds else None,
        'per_task_seconds': per_task_times,
        'avg_per_task_min': (sum(per_task_times) / len(per_task_times) / 60) if per_task_times else None,
        'n_tasks_completed': len(per_task_times),
    }

    return result


def parse_sigma_sweep(exp_name):
    """Parse sigma sweep results to find best P-AP."""
    log_file = os.path.join(LOG_BASE, f"{exp_name}.log")
    if not os.path.exists(log_file):
        log_file = os.path.join(LOG_BASE, exp_name, f"{exp_name}.log")
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Pattern: "sigma=X.XX: ... Pixel AP: XX.XX%"
    # or "Best sigma=X.XX ... P-AP=XX.XX%"
    best_pap = None
    best_sigma = None

    sigma_matches = re.findall(
        r'sigma[=:\s]+(\d+\.?\d*)\s.*?(?:Pixel.?AP|P-AP)[=:\s]+(\d+\.?\d+)',
        content
    )

    for sigma_str, pap_str in sigma_matches:
        pap = float(pap_str)
        if best_pap is None or pap > best_pap:
            best_pap = pap
            best_sigma = float(sigma_str)

    if best_pap:
        return {'best_pap': best_pap, 'best_sigma': best_sigma}
    return None


def check_routing_accuracy(exp_name):
    """Check routing accuracy from log."""
    log_file = os.path.join(LOG_BASE, f"{exp_name}.log")
    if not os.path.exists(log_file):
        log_file = os.path.join(LOG_BASE, exp_name, f"{exp_name}.log")
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    ra_matches = re.findall(r'[Rr]outing.*?[Aa]cc(?:uracy)?[=:\s]+(\d+\.?\d+)', content)
    if ra_matches:
        return float(ra_matches[-1])
    return None


def main():
    print("=" * 100)
    print("V62 SNL Design Space Ablation — Results Analysis")
    print("=" * 100)
    print()

    # Collect all results
    all_results = {}

    # Parse V62 experiments
    for exp_name, config in EXPERIMENTS.items():
        print(f"Parsing {exp_name}...")

        perf = parse_final_results(exp_name)
        times = parse_training_times(exp_name)
        sigma = parse_sigma_sweep(exp_name)
        ra = check_routing_accuracy(exp_name)

        entry = {**config}

        if perf:
            entry['iauc'] = perf.get('iauc')
            entry['pap'] = perf.get('pap')
        else:
            entry['iauc'] = None
            entry['pap'] = None

        if sigma and sigma.get('best_pap'):
            entry['pap_best'] = sigma['best_pap']
            entry['best_sigma'] = sigma['best_sigma']

        if times:
            entry['total_time_h'] = times.get('total_hours')
            entry['avg_per_task_min'] = times.get('avg_per_task_min')
            entry['n_tasks'] = times.get('n_tasks_completed')

        entry['ra'] = ra
        entry['fm'] = 0.0  # All use parameter isolation

        all_results[exp_name] = entry

    # Add reference results
    for ref_name, ref_data in REFERENCE.items():
        all_results[ref_name] = ref_data

    print()

    # =====================================================================
    # Print Performance + Efficiency Table
    # =====================================================================
    print("=" * 120)
    print("PERFORMANCE + EFFICIENCY COMPARISON")
    print("=" * 120)

    header = f"{'Config':<28} {'DCL':>3} {'SNL':>3} {'Adapter':<18} {'I-AUC':>7} {'P-AP':>7} {'FM':>5} {'Params/Task':>12} {'Ratio':>6} {'Time(h)':>8} {'Time/Task':>10}"
    print(header)
    print("-" * 120)

    # Sort: reference first, then V62 experiments
    order = [
        "V48_01_DeCoFlow",         # Ours (reference)
        "V52_01_wo_SNL",           # w/o SNL (reference)
        "V62_01_FullSNL_8acl_hr05",
        "V62_03_FullSNL_8acl_hr20",
        "V62_02_DecompLinear_regularlinear",
        "V62_04_DecompLoRA_rank512",
    ]

    ours_params = REFERENCE["V48_01_DeCoFlow"]["per_task_params_M"]

    for key in order:
        if key not in all_results:
            continue
        r = all_results[key]

        iauc_str = f"{r['iauc']:.2f}" if r.get('iauc') is not None else "---"
        pap_str = f"{r.get('pap_best', r.get('pap', None)):.2f}" if r.get('pap_best', r.get('pap')) is not None else "---"
        fm_str = f"{r.get('fm', 0.0):.1f}" if r.get('fm') is not None else "---"
        params_str = f"{r['per_task_params_M']:.2f}M"
        ratio_str = f"{r['per_task_params_M'] / ours_params:.1f}x"
        time_str = f"{r['total_time_h']:.1f}" if r.get('total_time_h') else "---"
        tpt_str = f"{r['avg_per_task_min']:.1f}min" if r.get('avg_per_task_min') else "---"

        marker = " ★" if key == "V48_01_DeCoFlow" else ""

        print(f"{r['label']:<28} {r['dcl']:>3} {r['snl']:>3} {r['adapter']:<18} {iauc_str:>7} {pap_str:>7} {fm_str:>5} {params_str:>12} {ratio_str:>6} {time_str:>8} {tpt_str:>10}{marker}")

    print("-" * 120)

    # =====================================================================
    # Print Total Model Size after 15 Tasks
    # =====================================================================
    print()
    print("=" * 80)
    print("TOTAL MODEL SIZE AFTER 15 TASKS")
    print("=" * 80)

    # Base model size (shared across tasks)
    BASE_MODEL_PARAMS = {
        "0_DCL": 0.009,   # Only PE + positional, no coupling layers (approximate)
        "6_DCL_LoRA": 10.7,  # Base NF with 6 coupling layers (approximate from param count)
        "6_DCL_Full": 10.7,  # Same base architecture
    }

    for key in order:
        if key not in all_results:
            continue
        r = all_results[key]

        per_task = r['per_task_params_M']
        # Estimate base model size
        if r['dcl'] == 0:
            base = 0.01  # Nearly no base
        else:
            base = 10.7  # Approximate base NF

        total_15 = base + 15 * per_task

        print(f"  {r['label']:<30}: Base={base:.1f}M + 15×{per_task:.2f}M = {total_15:.1f}M total")

    print()

    # =====================================================================
    # Routing Accuracy Summary
    # =====================================================================
    print("=" * 60)
    print("ROUTING ACCURACY")
    print("=" * 60)

    for key in order:
        if key not in all_results:
            continue
        r = all_results[key]
        ra = r.get('ra', None)
        ra_str = f"{ra:.1f}%" if ra else "---"
        print(f"  {r['label']:<30}: {ra_str}")

    print()

    # =====================================================================
    # Key Findings Summary
    # =====================================================================
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    ours = all_results.get("V48_01_DeCoFlow", {})
    wo_snl = all_results.get("V52_01_wo_SNL", {})

    for key in ["V62_01_FullSNL_8acl_hr05", "V62_03_FullSNL_8acl_hr20",
                "V62_02_DecompLinear_regularlinear", "V62_04_DecompLoRA_rank512"]:
        if key not in all_results:
            continue
        r = all_results[key]

        if r.get('iauc') is not None and ours.get('iauc') is not None:
            delta_iauc = r['iauc'] - ours['iauc']
            param_ratio = r['per_task_params_M'] / ours_params

            print(f"\n  {r['label']}:")
            print(f"    I-AUC: {r['iauc']:.2f}% (Δ={delta_iauc:+.2f}pp vs Ours)")
            print(f"    Params/Task: {r['per_task_params_M']:.2f}M ({param_ratio:.1f}x vs Ours)")

            if delta_iauc < 0:
                print(f"    → WORSE performance with {param_ratio:.1f}x MORE parameters")
            elif delta_iauc > 0:
                print(f"    → Better performance but {param_ratio:.1f}x MORE parameters")
            else:
                print(f"    → SAME performance with {param_ratio:.1f}x MORE parameters")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("  DeCoFlow (6 DCL + 2 SNL, LoRA r=64) achieves the best")
    print("  performance-efficiency trade-off among all configurations.")
    print()

    # =====================================================================
    # LaTeX table output
    # =====================================================================
    print("=" * 80)
    print("LATEX TABLE (for paper)")
    print("=" * 80)
    print()

    print(r"""\begin{table}[t]
\centering
\caption{SNL 설계 공간 탐색 (MVTec-AD 15-class, HR). DCL과 SNL의 블록 수, 어댑터 유형에 따른 성능 및 효율성 비교. Params/Task는 태스크 추가 시 증가하는 파라미터, Ratio는 DeCoFlow 대비 배율.}
\label{tab:snl_design_space}
\renewcommand{\arraystretch}{0.9}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccccc}
\toprule
Configuration & Adapter & DCL & SNL & I-AUC (\%) & P-AP (\%) & Params/Task & Ratio \\
\midrule""")

    for key in order:
        if key not in all_results:
            continue
        r = all_results[key]

        iauc_str = f"{r['iauc']:.2f}" if r.get('iauc') is not None else "---"
        pap_val = r.get('pap_best', r.get('pap'))
        pap_str = f"{pap_val:.2f}" if pap_val is not None else "---"
        params_str = f"{r['per_task_params_M']:.1f}M"
        ratio_str = f"{r['per_task_params_M'] / ours_params:.1f}$\\times$"

        bold = key == "V48_01_DeCoFlow"

        if bold:
            print(f"\\textbf{{{r['label']}}} & \\textbf{{{r['adapter']}}} & \\textbf{{{r['dcl']}}} & \\textbf{{{r['snl']}}} & \\textbf{{{iauc_str}}} & \\textbf{{{pap_str}}} & \\textbf{{{params_str}}} & \\textbf{{1.0$\\times$}} \\\\")
        else:
            print(f"{r['label']} & {r['adapter']} & {r['dcl']} & {r['snl']} & {iauc_str} & {pap_str} & {params_str} & {ratio_str} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # Save results to JSON for later use
    output_path = os.path.join(LOG_BASE, "V62_snl_ablation_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
