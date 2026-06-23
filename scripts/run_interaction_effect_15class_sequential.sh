#!/bin/bash
# =============================================================================
# Interaction Effect Experiment (15-class) - Sequential per GPU
# =============================================================================
# GPU 0: 4 experiments (sequential)
# GPU 1: 4 experiments (sequential)
# Both GPUs run in parallel → 2 experiments at any time
# =============================================================================

set -e

# Configuration
CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
EPOCHS=60
LR=3e-4
BATCH_SIZE=16
NCL=6
ACL_LAYERS=2
LORA_RANK=64
DATA_PATH="/Data/MVTecAD"
LOG_DIR="./logs/InteractionEffect_15class"

mkdir -p ${LOG_DIR}

COMMON_ARGS="--task_classes ${CLASSES} \
    --num_epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --num_coupling_layers ${NCL} \
    --lora_rank ${LORA_RANK} \
    --data_path ${DATA_PATH} \
    --log_dir ${LOG_DIR} \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3"

TAL_ARGS="--use_tail_aware_loss --tail_weight 1.0 --tail_top_k_ratio 0.02"

echo "=============================================="
echo "Interaction Effect (15-class) - Sequential per GPU"
echo "=============================================="
echo "GPU 0: Trainable Baseline → WA → TAL → ACL"
echo "GPU 1: Frozen Baseline → WA → TAL → ACL"
echo "=============================================="

# -----------------------------------------------------------------------------
# GPU 0: Trainable experiments (sequential)
# -----------------------------------------------------------------------------
run_gpu0() {
    echo "[GPU 0] Starting Trainable experiments..."

    echo "[GPU 0] [1/4] Trainable Baseline"
    CUDA_VISIBLE_DEVICES=0 python run_decoflow.py ${COMMON_ARGS} \
        --no_freeze_base \
        --no_lora \
        --no_tsa \
        --no_acl \
        --experiment_name "IE15_Trainable_Baseline" \
        > ${LOG_DIR}/trainable_baseline.log 2>&1
    echo "[GPU 0] [1/4] Complete!"

    echo "[GPU 0] [2/4] Trainable + WA"
    CUDA_VISIBLE_DEVICES=0 python run_decoflow.py ${COMMON_ARGS} \
        --no_freeze_base \
        --no_lora \
        --use_tsa \
        --no_acl \
        --experiment_name "IE15_Trainable_WA" \
        > ${LOG_DIR}/trainable_wa.log 2>&1
    echo "[GPU 0] [2/4] Complete!"

    echo "[GPU 0] [3/4] Trainable + TAL"
    CUDA_VISIBLE_DEVICES=0 python run_decoflow.py ${COMMON_ARGS} \
        --no_freeze_base \
        --no_lora \
        --no_tsa \
        --no_acl \
        ${TAL_ARGS} \
        --experiment_name "IE15_Trainable_TAL" \
        > ${LOG_DIR}/trainable_tal.log 2>&1
    echo "[GPU 0] [3/4] Complete!"

    echo "[GPU 0] [4/4] Trainable + ACL"
    CUDA_VISIBLE_DEVICES=0 python run_decoflow.py ${COMMON_ARGS} \
        --no_freeze_base \
        --no_lora \
        --no_tsa \
        --use_acl \
        --acl_n_layers ${ACL_LAYERS} \
        --experiment_name "IE15_Trainable_DIA" \
        > ${LOG_DIR}/trainable_acl.log 2>&1
    echo "[GPU 0] [4/4] Complete!"

    echo "[GPU 0] All Trainable experiments done!"
}

# -----------------------------------------------------------------------------
# GPU 1: Frozen experiments (sequential)
# -----------------------------------------------------------------------------
run_gpu1() {
    echo "[GPU 1] Starting Frozen experiments..."

    echo "[GPU 1] [1/4] Frozen Baseline"
    CUDA_VISIBLE_DEVICES=1 python run_decoflow.py ${COMMON_ARGS} \
        --no_tsa \
        --no_acl \
        --experiment_name "IE15_Frozen_Baseline" \
        > ${LOG_DIR}/frozen_baseline.log 2>&1
    echo "[GPU 1] [1/4] Complete!"

    echo "[GPU 1] [2/4] Frozen + WA"
    CUDA_VISIBLE_DEVICES=1 python run_decoflow.py ${COMMON_ARGS} \
        --use_tsa \
        --no_acl \
        --experiment_name "IE15_Frozen_WA" \
        > ${LOG_DIR}/frozen_wa.log 2>&1
    echo "[GPU 1] [2/4] Complete!"

    echo "[GPU 1] [3/4] Frozen + TAL"
    CUDA_VISIBLE_DEVICES=1 python run_decoflow.py ${COMMON_ARGS} \
        --no_tsa \
        --no_acl \
        ${TAL_ARGS} \
        --experiment_name "IE15_Frozen_TAL" \
        > ${LOG_DIR}/frozen_tal.log 2>&1
    echo "[GPU 1] [3/4] Complete!"

    echo "[GPU 1] [4/4] Frozen + ACL"
    CUDA_VISIBLE_DEVICES=1 python run_decoflow.py ${COMMON_ARGS} \
        --no_tsa \
        --use_acl \
        --acl_n_layers ${ACL_LAYERS} \
        --experiment_name "IE15_Frozen_DIA" \
        > ${LOG_DIR}/frozen_acl.log 2>&1
    echo "[GPU 1] [4/4] Complete!"

    echo "[GPU 1] All Frozen experiments done!"
}

# -----------------------------------------------------------------------------
# Run both GPUs in parallel
# -----------------------------------------------------------------------------
run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

echo "Running GPU 0 (PID: $PID_GPU0) and GPU 1 (PID: $PID_GPU1) in parallel..."
wait $PID_GPU0 $PID_GPU1

echo ""
echo "=============================================="
echo "All 8 experiments completed!"
echo "=============================================="
echo "Results: ${LOG_DIR}/"
echo "Analyze: python scripts/analyze_interaction_effect.py --log_dir ${LOG_DIR}"
