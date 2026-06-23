#!/usr/bin/env python3
"""
Generate summary CSV files for organized log directories.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

def parse_final_results(results_path: str) -> Optional[Dict]:
    """Parse final_results.csv file."""
    if not os.path.exists(results_path):
        return None

    try:
        with open(results_path, 'r') as f:
            lines = f.readlines()

        # Find the Mean/Overall row
        for line in lines:
            if 'Mean' in line or 'Overall' in line:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    return {
                        'routing_acc': parts[2].replace('%', '').replace('*', ''),
                        'image_auc': parts[3].replace('*', ''),
                        'pixel_auc': parts[4].replace('*', ''),
                        'pixel_ap': parts[5].replace('*', '') if len(parts) > 5 else 'N/A'
                    }
    except Exception as e:
        print(f"Error parsing {results_path}: {e}")
    return None


def parse_config(config_path: str) -> Optional[Dict]:
    """Parse config.json file."""
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        return {
            'backbone': config.get('backbone_name', 'N/A'),
            'num_epochs': config.get('num_epochs', 'N/A'),
            'lr': config.get('lr', 'N/A'),
            'lora_rank': config.get('lora_rank', 'N/A'),
            'num_coupling_layers': config.get('num_coupling_layers', 'N/A'),
            'acl_n_layers': config.get('acl_n_layers', 'N/A'),
            'tail_weight': config.get('tail_weight', 'N/A'),
            'lambda_logdet': config.get('lambda_logdet', 'N/A'),
            'num_tasks': len(config.get('task_classes', [])),
        }
    except Exception as e:
        print(f"Error parsing {config_path}: {e}")
    return None


def generate_summary_for_directory(base_dir: str, output_path: str):
    """Generate summary CSV for a directory of experiments."""
    experiments = []

    for exp_name in sorted(os.listdir(base_dir)):
        exp_path = os.path.join(base_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue

        # Parse results
        results_path = os.path.join(exp_path, 'final_results.csv')
        config_path = os.path.join(exp_path, 'config.json')

        results = parse_final_results(results_path)
        config = parse_config(config_path)

        exp_data = {'experiment_name': exp_name}

        if results:
            exp_data.update(results)
        else:
            exp_data.update({
                'routing_acc': 'N/A',
                'image_auc': 'N/A',
                'pixel_auc': 'N/A',
                'pixel_ap': 'N/A'
            })

        if config:
            exp_data.update(config)
        else:
            exp_data.update({
                'backbone': 'N/A',
                'num_epochs': 'N/A',
                'lr': 'N/A',
                'lora_rank': 'N/A',
                'num_coupling_layers': 'N/A',
                'acl_n_layers': 'N/A',
                'tail_weight': 'N/A',
                'lambda_logdet': 'N/A',
                'num_tasks': 'N/A'
            })

        experiments.append(exp_data)

    # Write CSV
    if experiments:
        fieldnames = [
            'experiment_name', 'image_auc', 'pixel_auc', 'pixel_ap', 'routing_acc',
            'backbone', 'num_epochs', 'lr', 'lora_rank', 'num_coupling_layers',
            'acl_n_layers', 'tail_weight', 'lambda_logdet', 'num_tasks'
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(experiments)

        print(f"Generated: {output_path} ({len(experiments)} experiments)")
    else:
        print(f"No experiments found in {base_dir}")


def main():
    base_path = "/Volume/DeCoFlow/logs_organized"

    # Define directories to process
    directories = [
        ("1_Main_Results/MVTec_Main", "1_Main_Results/MVTec_Main_summary.csv"),
        ("1_Main_Results/VisA_Main", "1_Main_Results/VisA_Main_summary.csv"),
        ("1_Main_Results/MPDD_Main", "1_Main_Results/MPDD_Main_summary.csv"),
        ("2_Ablation/Component", "2_Ablation/Component_summary.csv"),
        ("2_Ablation/Hyperparameter", "2_Ablation/Hyperparameter_summary.csv"),
        ("2_Ablation/LoRA_Rank", "2_Ablation/LoRA_Rank_summary.csv"),
        ("2_Ablation/Architecture_Depth", "2_Ablation/Architecture_Depth_summary.csv"),
        ("2_Ablation/Loss_Function", "2_Ablation/Loss_Function_summary.csv"),
        ("3_Interaction_Effect/3class", "3_Interaction_Effect/3class_summary.csv"),
        ("3_Interaction_Effect/15class", "3_Interaction_Effect/15class_summary.csv"),
        ("4_CL_Scenarios/1x1", "4_CL_Scenarios/1x1_summary.csv"),
        ("4_CL_Scenarios/3x3", "4_CL_Scenarios/3x3_summary.csv"),
        ("4_CL_Scenarios/5x5", "4_CL_Scenarios/5x5_summary.csv"),
        ("4_CL_Scenarios/Others", "4_CL_Scenarios/Others_summary.csv"),
        ("6_Dataset_Specific/VisA_Optimization", "6_Dataset_Specific/VisA_Optimization_summary.csv"),
    ]

    for input_dir, output_file in directories:
        input_path = os.path.join(base_path, input_dir)
        output_path = os.path.join(base_path, output_file)

        if os.path.exists(input_path):
            generate_summary_for_directory(input_path, output_path)
        else:
            print(f"Directory not found: {input_path}")


if __name__ == "__main__":
    main()
