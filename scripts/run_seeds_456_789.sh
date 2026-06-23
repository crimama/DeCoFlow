#!/bin/bash
# =============================================================================
# Multi-Seed Experiments (Seeds 456, 789) - MAIN Configuration
# =============================================================================
# DCL6 + ACB2 configuration (matching current MAIN)
# =============================================================================

set -e

CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
EPOCHS=60
LR=3e-4
BATCH_SIZE=16
NCL=6  # DCL blocks
ACB_BLOCKS=2
LORA_RANK=64
DATA_PATH="/Data/MVTecAD"
LOG_DIR="./logs/Final"

COMMON_ARGS="--task_classes ${CLASSES} \
    --num_epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --num_coupling_layers ${NCL} \
    --lora_rank ${LORA_RANK} \
    --data_path ${DATA_PATH} \
    --log_dir ${LOG_DIR} \
    --use_tsa \
    --use_acb \
    --acb_n_blocks ${ACB_BLOCKS} \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5"

echo "=============================================="
echo "Multi-Seed Experiments (MAIN Config: DCL6+ACB2)"
echo "=============================================="
echo "Seeds: 456, 789"
echo "GPUs: 0, 1"
echo "=============================================="

# GPU 0: Seed 456
run_seed456() {
    echo "[GPU 0] Running Seed 456..."
    CUDA_VISIBLE_DEVICES=0 python run_decoflow.py ${COMMON_ARGS} \
        --seed 456 \
        --experiment_name "MVTec-MAIN-Seed456" \
        > ${LOG_DIR}/seed456.log 2>&1
    echo "[GPU 0] Seed 456 Complete!"
}

# GPU 1: Seed 789
run_seed789() {
    echo "[GPU 1] Running Seed 789..."
    CUDA_VISIBLE_DEVICES=1 python run_decoflow.py ${COMMON_ARGS} \
        --seed 789 \
        --experiment_name "MVTec-MAIN-Seed789" \
        > ${LOG_DIR}/seed789.log 2>&1
    echo "[GPU 1] Seed 789 Complete!"
}

# Run in parallel
run_seed456 &
PID1=$!

run_seed789 &
PID2=$!

echo "Running Seed 456 (PID: $PID1) and Seed 789 (PID: $PID2) in parallel..."
wait $PID1 $PID2

echo ""
echo "=============================================="
echo "Both seed experiments completed!"
echo "=============================================="
echo "Results:"
echo "  - ${LOG_DIR}/MVTec-MAIN-Seed456/"
echo "  - ${LOG_DIR}/MVTec-MAIN-Seed789/"
