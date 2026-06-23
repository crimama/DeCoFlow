#!/bin/bash

# =============================================================================
# DeCoFlow: Default Training Script (MAIN Configuration)
# =============================================================================
# Configuration: DCL6 + ACB2 (Total 8 blocks)
# Expected Performance: Img AUC 98.05%, Pix AUC 97.81%, Pix AP 55.80%
# =============================================================================

echo "============================================================="
echo "DeCoFlow: MAIN Configuration (DCL6 + ACB2)"
echo "============================================================="
echo "Timestamp: $(date)"
echo "============================================================="

# Default GPU (can be overridden with GPU=X ./run.sh)
GPU=${GPU:-0}

# Experiment name (can be overridden with EXP_NAME=xxx ./run.sh)
EXP_NAME=${EXP_NAME:-"MVTec-DCL6-ACB2"}

CUDA_VISIBLE_DEVICES=$GPU python run_decoflow.py \
    --dataset mvtec \
    --data_path /Data/MVTecAD \
    --task_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 \
    --lr 3e-4 \
    --lora_rank 64 \
    --num_coupling_layers 6 \
    --acb_n_blocks 2 \
    --batch_size 16 \
    --use_tsa \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --experiment_name "$EXP_NAME"

echo "============================================================="
echo "Training completed!"
echo "============================================================="
