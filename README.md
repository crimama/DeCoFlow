# DeCoFlow

Structural Decomposition of Normalizing Flows for Continual Anomaly Detection.

This repository contains the code-only release of DeCoFlow. The public naming follows the paper:

- `DeCoFlow`: overall framework
- `DCL`: Decomposed Coupling Layer with frozen base weights and task-specific LoRA adapters
- `TSA`: Task-Specific Alignment
- `TAL`: Tail-Aware Loss
- `ACB`: Auxiliary Coupling Blocks

## Contents

- `decoflow/`: core Python package
- `run_decoflow.py`: main continual anomaly detection training/evaluation entry point
- `scripts/`: reusable experiment and analysis scripts
- `requirements.txt`: Python dependencies

Large experiment artifacts, logs, paper sources, baselines, worktree metadata, notebooks, checkpoints, and generated results are intentionally excluded.

## Setup

```bash
pip install -r requirements.txt
```

## Example

```bash
python run_decoflow.py \
  --dataset mvtec \
  --data_path /path/to/MVTecAD \
  --cl_scenario 1-1 \
  --num_epochs 60 \
  --num_coupling_layers 6 \
  --use_tsa \
  --use_acb \
  --acb_n_blocks 2 \
  --use_tail_aware_loss
```

For VisA, set `--dataset visa` and point `--data_path` to the VisA root.

## Notes

The code was exported from a larger research workspace into a clean repository. Historical internal names were normalized to match the paper terminology in this release.
