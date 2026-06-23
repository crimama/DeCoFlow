#!/bin/bash
# =============================================================================
# DeCoFlow Ablation Study Execution Script (v2)
# =============================================================================
#
# MAIN Experiment: MVTec-WRN50-TailW0.7-TopK3-TailTopK2-ScaleK5-lr3e-4-MAIN
# - Image AUC: 98.29%
# - Pixel AUC: 97.82%
# - Routing Accuracy: 100%
#
# NOTE: 기존 ablation 실험들은 다른 하이퍼파라미터로 진행되어 재실험 필요
# - 기존: lr=2e-4, scale_ctx_k=3, lambda_logdet=1e-5
# - MAIN: lr=3e-4, scale_ctx_k=5, lambda_logdet=1e-4
#
# =============================================================================

# Common configuration (same as MAIN experiment)
DATASET="mvtec"
DATA_PATH="/Data/MVTecAD"
BACKBONE="wide_resnet50_2"
EPOCHS=60
LR="3e-4"
LORA_RANK=64
COUPLING_LAYERS=8
ACB_BLOCKS=4
BATCH_SIZE=16
LOG_DIR="./logs/Ablation"

# All 15 MVTec classes in order
CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

# Base command with MAIN experiment settings
BASE_CMD="python run_decoflow.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --task_classes ${CLASSES} \
    --backbone_name ${BACKBONE} \
    --num_epochs ${EPOCHS} \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --num_coupling_layers ${COUPLING_LAYERS} \
    --batch_size ${BATCH_SIZE} \
    --use_acb \
    --acb_n_blocks ${ACB_BLOCKS} \
    --use_tsa \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --log_dir ${LOG_DIR}"

# =============================================================================
# Section 1: Core Component Ablation (v2 - MAIN 설정 기준)
# =============================================================================

run_core_ablation() {
    echo "========================================"
    echo "Core Component Ablation Experiments (v2)"
    echo "========================================"

    # 1.1 wo_ACB: Disable Auxiliary Coupling Blocks
    echo "[1.1] Running wo_ACB-v2..."
    $BASE_CMD --no_acb --experiment_name "MVTec-Main-Ablation-wo_ACB"

    # 1.2 wo_Adapter: Disable TSA Adapter
    echo "[1.2] Running wo_Adapter-v2..."
    $BASE_CMD --no_task_adapter --experiment_name "MVTec-Main-Ablation-wo_Adapter"

    # 1.3 wo_LoRA: Disable LoRA adaptation
    echo "[1.3] Running wo_LoRA-v2..."
    $BASE_CMD --no_lora --experiment_name "MVTec-Main-Ablation-wo_LoRA"

    # 1.4 wo_Router: Use Oracle task ID
    echo "[1.4] Running wo_Router-v2..."
    $BASE_CMD --no_router --experiment_name "MVTec-Main-Ablation-wo_Router"
}

# =============================================================================
# Section 2: Context Module Ablation (v2 - MAIN 설정 기준)
# =============================================================================

run_context_ablation() {
    echo "========================================"
    echo "Context Module Ablation Experiments (v2)"
    echo "========================================"

    # 2.1 wo_PosEmbed: Disable Positional Embedding
    echo "[2.1] Running wo_PosEmbed-v2..."
    $BASE_CMD --no_pos_embedding --experiment_name "MVTec-Main-Ablation-wo_PosEmbed"

    # 2.2 wo_SpatialCtx: Disable Spatial Context Mixing
    echo "[2.2] Running wo_SpatialCtx-v2..."
    $BASE_CMD --no_spatial_context --experiment_name "MVTec-Main-Ablation-wo_SpatialCtx"

    # 2.3 wo_ScaleCtx: Disable Scale Context Injection
    echo "[2.3] Running wo_ScaleCtx-v2..."
    $BASE_CMD --no_scale_context --experiment_name "MVTec-Main-Ablation-wo_ScaleCtx"
}

# =============================================================================
# Section 3: Design Choice Ablation
# =============================================================================

run_design_ablation() {
    echo "========================================"
    echo "Design Choice Ablation Experiments"
    echo "========================================"

    # 3.1 Regular Linear (instead of LoRA)
    echo "[3.1] Running Regular Linear..."
    $BASE_CMD \
        --use_regular_linear \
        --experiment_name "MVTec-Main-Ablation-Design-RegularLinear"

    # 3.2 Task-Separated Training (Upper Bound)
    echo "[3.2] Running Task-Separated..."
    $BASE_CMD \
        --use_task_separated \
        --experiment_name "MVTec-Main-Ablation-Design-TaskSeparated"

    # 3.3 All-Shared (Lower Bound)
    echo "[3.3] Running All-Shared..."
    python run_decoflow.py \
        --dataset ${DATASET} \
        --data_path ${DATA_PATH} \
        --task_classes ${CLASSES} \
        --backbone_name ${BACKBONE} \
        --num_epochs ${EPOCHS} \
        --lr ${LR} \
        --num_coupling_layers ${COUPLING_LAYERS} \
        --batch_size ${BATCH_SIZE} \
        --no_lora \
        --no_acb \
        --no_task_adapter \
        --use_tail_aware_loss \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --log_dir ${LOG_DIR} \
        --experiment_name "MVTec-Main-Ablation-Design-AllShared"
}

# =============================================================================
# Section 4: Module Combination Ablation
# =============================================================================

run_combination_ablation() {
    echo "========================================"
    echo "Module Combination Ablation Experiments"
    echo "========================================"

    # 4.1 wo_ACB + wo_Adapter
    echo "[4.1] Running wo_ACB + wo_Adapter..."
    $BASE_CMD \
        --no_acb \
        --no_task_adapter \
        --experiment_name "MVTec-Main-Ablation-wo_ACB_Adapter"

    # 4.2 LoRA Only
    echo "[4.2] Running LoRA Only..."
    python run_decoflow.py \
        --dataset ${DATASET} \
        --data_path ${DATA_PATH} \
        --task_classes ${CLASSES} \
        --backbone_name ${BACKBONE} \
        --num_epochs ${EPOCHS} \
        --lr ${LR} \
        --lora_rank ${LORA_RANK} \
        --num_coupling_layers ${COUPLING_LAYERS} \
        --batch_size ${BATCH_SIZE} \
        --no_acb \
        --no_task_adapter \
        --no_spatial_context \
        --no_scale_context \
        --use_tail_aware_loss \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --log_dir ${LOG_DIR} \
        --experiment_name "MVTec-Main-Ablation-LoRA-Only"
}

# =============================================================================
# Main Execution
# =============================================================================

echo "=============================================="
echo "DeCoFlow Ablation Study (v2)"
echo "=============================================="
echo "MAIN: MVTec-Main-TailW0.7-TopK3-TailTopK2-ScaleK5-lr3e-4-MAIN"
echo "  Image AUC: 98.29%"
echo "  Pixel AUC: 97.82%"
echo ""
echo "Settings: lr=3e-4, scale_ctx_k=5, lambda_logdet=1e-4"
echo "=============================================="

case "$1" in
    "core")
        run_core_ablation
        ;;
    "context")
        run_context_ablation
        ;;
    "design")
        run_design_ablation
        ;;
    "combination")
        run_combination_ablation
        ;;
    "all")
        run_core_ablation
        run_context_ablation
        run_design_ablation
        run_combination_ablation
        ;;
    *)
        echo "Usage: $0 {core|context|design|combination|all}"
        echo ""
        echo "Sections:"
        echo "  core        - Core component ablation (wo_ACB, wo_Adapter, wo_LoRA, wo_Router)"
        echo "  context     - Context module ablation (wo_PosEmbed, wo_SpatialCtx, wo_ScaleCtx)"
        echo "  design      - Design choice ablation (RegularLinear, TaskSeparated, AllShared)"
        echo "  combination - Module combination ablation (wo_ACB_Adapter, LoRA-Only)"
        echo "  all         - Run ALL experiments"
        echo ""
        echo "Estimated time: ~4 hours per experiment (60 epochs, 15 classes)"
        echo "Total: 12 experiments = ~48 hours (single GPU)"
        exit 1
        ;;
esac

echo "Done!"
