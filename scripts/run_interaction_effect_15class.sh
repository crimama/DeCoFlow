#!/bin/bash
# =============================================================================
# Interaction Effect Experiment (15-class): Full MVTec Validation
# =============================================================================
#
# Goal: Validate the 5-class findings on the full 15-class MVTec dataset
#       to resolve WA inconsistency and confirm integral component hypothesis
#
# Hypothesis:
#   - If these modules are "generic boosters" → similar effect in both settings
#   - If these modules are "Base Freeze compensators" → large effect only when frozen
#
# Experiment Design:
#   Group 1: Base Trainable (no freeze, no LoRA)
#   Group 2: Base Frozen (with LoRA - our approach)
# =============================================================================

set -e

# Configuration - Full 15 MVTec classes
CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
EPOCHS=60
LR=3e-4
BATCH_SIZE=16
NCL=6  # DCL blocks
ACL_LAYERS=2
LORA_RANK=64
DATA_PATH="/Data/MVTecAD"
LOG_DIR="./logs/InteractionEffect_15class"

# Create log directory
mkdir -p ${LOG_DIR}

# Common args (MAIN baseline settings)
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

# TAL common settings (when enabled)
TAL_ARGS="--use_tail_aware_loss --tail_weight 1.0 --tail_top_k_ratio 0.02"

echo "=============================================="
echo "Interaction Effect Experiment (15-class)"
echo "=============================================="
echo "Classes: ${CLASSES}"
echo "Epochs: ${EPOCHS}"
echo "GPUs: 0, 1, 4, 5"
echo "=============================================="

# -----------------------------------------------------------------------------
# Group 1: Base Trainable (no freeze, no LoRA)
# The base NF adapts directly to each task without parameter isolation
# -----------------------------------------------------------------------------

# 1. Trainable Baseline: No modules
echo "[1/8] Starting: Trainable Baseline (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python run_decoflow.py ${COMMON_ARGS} \
    --no_freeze_base \
    --no_lora \
    --no_tsa \
    --no_acl \
    --experiment_name "IE15_Trainable_Baseline" \
    > ${LOG_DIR}/trainable_baseline.log 2>&1 &
PID1=$!

# 2. Trainable + WA: Only TaskSpecificAlignment added
echo "[2/8] Starting: Trainable + WA (GPU 1)"
CUDA_VISIBLE_DEVICES=1 python run_decoflow.py ${COMMON_ARGS} \
    --no_freeze_base \
    --no_lora \
    --use_tsa \
    --no_acl \
    --experiment_name "IE15_Trainable_WA" \
    > ${LOG_DIR}/trainable_wa.log 2>&1 &
PID2=$!

# 3. Trainable + TAL: Only Tail-Aware Loss added
echo "[3/8] Starting: Trainable + TAL (GPU 4)"
CUDA_VISIBLE_DEVICES=4 python run_decoflow.py ${COMMON_ARGS} \
    --no_freeze_base \
    --no_lora \
    --no_tsa \
    --no_acl \
    ${TAL_ARGS} \
    --experiment_name "IE15_Trainable_TAL" \
    > ${LOG_DIR}/trainable_tal.log 2>&1 &
PID3=$!

# 4. Trainable + ACL: Only ACL added
echo "[4/8] Starting: Trainable + ACL (GPU 5)"
CUDA_VISIBLE_DEVICES=5 python run_decoflow.py ${COMMON_ARGS} \
    --no_freeze_base \
    --no_lora \
    --no_tsa \
    --use_acl \
    --acl_n_layers ${ACL_LAYERS} \
    --experiment_name "IE15_Trainable_DIA" \
    > ${LOG_DIR}/trainable_acl.log 2>&1 &
PID4=$!

echo "Waiting for Group 1 (Trainable) experiments to complete..."
wait $PID1 $PID2 $PID3 $PID4
echo "Group 1 complete!"

# -----------------------------------------------------------------------------
# Group 2: Base Frozen (with LoRA - our approach)
# The base NF is frozen after Task 0, only LoRA adapts
# -----------------------------------------------------------------------------

# 5. Frozen Baseline: No modules (only LoRA)
echo "[5/8] Starting: Frozen Baseline (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python run_decoflow.py ${COMMON_ARGS} \
    --no_tsa \
    --no_acl \
    --experiment_name "IE15_Frozen_Baseline" \
    > ${LOG_DIR}/frozen_baseline.log 2>&1 &
PID5=$!

# 6. Frozen + WA: TaskSpecificAlignment added
echo "[6/8] Starting: Frozen + WA (GPU 1)"
CUDA_VISIBLE_DEVICES=1 python run_decoflow.py ${COMMON_ARGS} \
    --use_tsa \
    --no_acl \
    --experiment_name "IE15_Frozen_WA" \
    > ${LOG_DIR}/frozen_wa.log 2>&1 &
PID6=$!

# 7. Frozen + TAL: Tail-Aware Loss added
echo "[7/8] Starting: Frozen + TAL (GPU 4)"
CUDA_VISIBLE_DEVICES=4 python run_decoflow.py ${COMMON_ARGS} \
    --no_tsa \
    --no_acl \
    ${TAL_ARGS} \
    --experiment_name "IE15_Frozen_TAL" \
    > ${LOG_DIR}/frozen_tal.log 2>&1 &
PID7=$!

# 8. Frozen + ACL: ACL added
echo "[8/8] Starting: Frozen + ACL (GPU 5)"
CUDA_VISIBLE_DEVICES=5 python run_decoflow.py ${COMMON_ARGS} \
    --no_tsa \
    --use_acl \
    --acl_n_layers ${ACL_LAYERS} \
    --experiment_name "IE15_Frozen_DIA" \
    > ${LOG_DIR}/frozen_acl.log 2>&1 &
PID8=$!

echo "Waiting for Group 2 (Frozen) experiments to complete..."
wait $PID5 $PID6 $PID7 $PID8
echo "Group 2 complete!"

echo ""
echo "=============================================="
echo "All 15-class experiments completed!"
echo "=============================================="
echo ""
echo "Results are in: ${LOG_DIR}/"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_interaction_effect.py --log_dir ${LOG_DIR}"
