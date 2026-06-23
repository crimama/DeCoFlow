#!/bin/bash
# =============================================================================
# DeCoFlow Core Component Ablation Experiments
# =============================================================================
# GPU 0, 1, 4, 5에서 6개 실험 병렬 실행
#
# MAIN Experiment 설정 기준:
#   - Image AUC: 98.29%, Pixel AUC: 97.82%
#   - lr=3e-4, scale_ctx_k=5, lambda_logdet=1e-4
# =============================================================================

cd /Volume/DeCoFlow

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
LOG_DIR="./logs/Ablation-Core"

# All 15 MVTec classes
CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

echo "=============================================="
echo "DeCoFlow Core Component Ablation Experiments"
echo "=============================================="
echo "Starting 6 experiments on GPUs 0, 1, 4, 5"
echo ""

# Create log directory
mkdir -p ${LOG_DIR}

# =============================================================================
# GPU 0: Exp 1 (w/o SpatialContextMixer) → Exp 5 (w/o DCL subnet)
# =============================================================================
echo "[GPU 0] Launching experiments 1 and 5..."
(
    CUDA_VISIBLE_DEVICES=0 python run_decoflow.py \
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
        --no_spatial_context \
        --log_dir ${LOG_DIR} \
        --experiment_name "MVTec-Ablation-wo_SpatialContextMixer"

    CUDA_VISIBLE_DEVICES=0 python run_decoflow.py \
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
        --no_lora \
        --log_dir ${LOG_DIR} \
        --experiment_name "MVTec-Ablation-wo_DCLSubnet"
) > ${LOG_DIR}/gpu0.log 2>&1 &
PID0=$!

# =============================================================================
# GPU 1: Exp 2 (w/o TaskSpecificAlignment) → Exp 6 (w/o Scale Context)
# =============================================================================
echo "[GPU 1] Launching experiments 2 and 6..."
(
    CUDA_VISIBLE_DEVICES=1 python run_decoflow.py \
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
        --no_tsa \
        --use_tail_aware_loss \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --log_dir ${LOG_DIR} \
        --experiment_name "MVTec-Ablation-wo_TSA"

    CUDA_VISIBLE_DEVICES=1 python run_decoflow.py \
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
        --no_scale_context \
        --log_dir ${LOG_DIR} \
        --experiment_name "MVTec-Ablation-wo_ScaleContext"
) > ${LOG_DIR}/gpu1.log 2>&1 &
PID1=$!

# =============================================================================
# GPU 4: Exp 3 (w/o Tail Aware Loss)
# =============================================================================
echo "[GPU 4] Launching experiment 3..."
CUDA_VISIBLE_DEVICES=4 python run_decoflow.py \
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
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --log_dir ${LOG_DIR} \
    --experiment_name "MVTec-Ablation-wo_TailAwareLoss" \
    > ${LOG_DIR}/gpu4.log 2>&1 &
PID4=$!

# =============================================================================
# GPU 5: Exp 4 (w/o LogDet Regularization)
# =============================================================================
echo "[GPU 5] Launching experiment 4..."
CUDA_VISIBLE_DEVICES=5 python run_decoflow.py \
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
    --lambda_logdet 0 \
    --scale_context_kernel 5 \
    --log_dir ${LOG_DIR} \
    --experiment_name "MVTec-Ablation-wo_LogDetReg" \
    > ${LOG_DIR}/gpu5.log 2>&1 &
PID5=$!

echo ""
echo "=============================================="
echo "All experiments launched!"
echo "=============================================="
echo ""
echo "Process IDs:"
echo "  GPU 0 (Exp 1→5): PID $PID0"
echo "  GPU 1 (Exp 2→6): PID $PID1"
echo "  GPU 4 (Exp 3):   PID $PID4"
echo "  GPU 5 (Exp 4):   PID $PID5"
echo ""
echo "Monitor logs:"
echo "  tail -f ${LOG_DIR}/gpu0.log"
echo "  tail -f ${LOG_DIR}/gpu1.log"
echo "  tail -f ${LOG_DIR}/gpu4.log"
echo "  tail -f ${LOG_DIR}/gpu5.log"
echo ""
echo "Monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Experiment Summary:"
echo "  1. w/o SpatialContextMixer (GPU 0, first)"
echo "  2. w/o TaskSpecificAlignment (GPU 1, first)"
echo "  3. w/o Tail Aware Loss (GPU 4)"
echo "  4. w/o LogDet Regularization (GPU 5)"
echo "  5. w/o DCL subnet / ACB only (GPU 0, second)"
echo "  6. w/o Scale Context (GPU 1, second)"
echo ""
echo "Estimated time: ~4 hours per experiment"
echo "  GPU 0, 1: ~8 hours (2 experiments each)"
echo "  GPU 4, 5: ~4 hours (1 experiment each)"
