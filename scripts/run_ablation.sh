#!/bin/bash

# =============================================================================
# DeCoFlow: TBD Hyperparameter Ablation Study
# =============================================================================
# Base: DCL6+ACL2 (MAIN Configuration)
# Each experiment changes ONLY ONE hyperparameter from the default
# =============================================================================

echo "============================================================="
echo "DeCoFlow: TBD Hyperparameter Ablation Study"
echo "============================================================="
echo "Timestamp: $(date)"
echo "============================================================="

LOG_DIR="/Volume/DeCoFlow/logs/Ablation/Hyperparameter"
mkdir -p $LOG_DIR

# =============================================================================
# MAIN Default Configuration (DO NOT MODIFY)
# =============================================================================
# backbone_name = wide_resnet50_2
# num_coupling_layers = 6
# acl_n_layers = 2
# lora_rank = 64
# lr = 3e-4
# num_epochs = 60
# batch_size = 16
# lambda_logdet = 1e-4
# tail_weight = 0.7
# tail_top_k_ratio = 0.02
# score_aggregation_top_k = 3
# scale_context_kernel = 5
# spatial_context_kernel = 3 (default)
# =============================================================================

BASE_CMD="python run_decoflow.py \
    --dataset mvtec --data_path /Data/MVTecAD \
    --task_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 --lr 3e-4 \
    --lora_rank 64 \
    --num_coupling_layers 6 --acl_n_layers 2 \
    --batch_size 16 \
    --use_tsa \
    --log_dir $LOG_DIR"

# =============================================================================
# TBD Experiments List (18 total)
# =============================================================================
# 1. Complete Separated (Training Strategy)
# 2-3. lambda_logdet: 1e-6, 1e-5
# 4-5. scale_context_kernel: 3, 7
# 6-7. spatial_context_kernel: 5, 7
# 8-11. tail_weight: 0.1, 0.3, 0.5, 0.8
# 12-15. tail_top_k_ratio: 0.01, 0.03, 0.05, 0.10
# 16-18. score_aggregation_top_k: 5, 7, 10
# =============================================================================

# =============================================================================
# Batch 1: GPU 0 - Complete Separated + lambda_logdet experiments
# =============================================================================
run_gpu0() {
    echo "[GPU 0] Starting experiments..."
    # 2. lambda_logdet = 1e-6
    echo "[GPU 0-2] lambda_logdet=1e-6"
    CUDA_VISIBLE_DEVICES=0 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-6 \
        --scale_context_kernel 5 \
        --experiment_name "LogDet-1e6"

    # 3. lambda_logdet = 1e-5
    echo "[GPU 0-3] lambda_logdet=1e-5"
    CUDA_VISIBLE_DEVICES=0 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-5 \
        --scale_context_kernel 5 \
        --experiment_name "LogDet-1e5"

    # 4. scale_context_kernel = 3
    echo "[GPU 0-4] scale_context_kernel=3"
    CUDA_VISIBLE_DEVICES=0 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 3 \
        --experiment_name "ScaleCtxK-3"

    echo "[GPU 0] All experiments completed!"
}

# =============================================================================
# Batch 2: GPU 1 - spatial_context_kernel + tail_weight experiments
# =============================================================================
run_gpu1() {
    echo "[GPU 1] Starting experiments..."

    # 5. scale_context_kernel = 7
    echo "[GPU 0-5] scale_context_kernel=7"
    CUDA_VISIBLE_DEVICES=1 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 7 \
        --experiment_name "ScaleCtxK-7"
    
    # 8. tail_weight = 0.1
    echo "[GPU 1-3] tail_weight=0.1"
    CUDA_VISIBLE_DEVICES=1 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.1 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --experiment_name "TailW-0.1"

    # 9. tail_weight = 0.3
    echo "[GPU 1-4] tail_weight=0.3"
    CUDA_VISIBLE_DEVICES=1 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.3 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --experiment_name "TailW-0.3"
    

    echo "[GPU 1] All experiments completed!"
}

# =============================================================================
# Batch 3: GPU 4 - tail_weight + tail_top_k_ratio experiments
# =============================================================================
run_gpu4() {
    echo "[GPU 4] Starting experiments..."

    # 10. tail_weight = 0.5
    echo "[GPU 1-5] tail_weight=0.5"
    CUDA_VISIBLE_DEVICES=4 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.5 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --experiment_name "TailW-0.5"
    
    # 13. tail_top_k_ratio = 0.03
    echo "[GPU 4-3] tail_top_k_ratio=0.03"
    CUDA_VISIBLE_DEVICES=4 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.7 \
        --tail_top_k_ratio 0.03 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --experiment_name "TailTopK-0.03"

    # 14. tail_top_k_ratio = 0.05
    echo "[GPU 4-4] tail_top_k_ratio=0.05"
    CUDA_VISIBLE_DEVICES=4 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.7 \
        --tail_top_k_ratio 0.05 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --experiment_name "TailTopK-0.05"

    echo "[GPU 4] All experiments completed!"
}

# =============================================================================
# Batch 4: GPU 5 - tail_top_k_ratio + score_aggregation_top_k experiments
# =============================================================================
run_gpu5() {
    echo "[GPU 5] Starting experiments..."

    echo "[GPU 4-4] tail_weight=1.0"
    CUDA_VISIBLE_DEVICES=5 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 1.0 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --experiment_name "TailW-1.0-TopK2"
    
    # 17. score_aggregation_top_k = 7
    echo "[GPU 5-3] score_aggregation_top_k=7"
    CUDA_VISIBLE_DEVICES=5 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 7 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --experiment_name "ScoreTopK-7"

    # 18. score_aggregation_top_k = 10
    echo "[GPU 5-4] score_aggregation_top_k=10"
    CUDA_VISIBLE_DEVICES=5 $BASE_CMD \
        --use_tail_aware_loss --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k --score_aggregation_top_k 10 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --experiment_name "ScoreTopK-10"

    echo "[GPU 5] All experiments completed!"
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    echo ""
    echo "============================================================="
    echo "TBD Hyperparameter Ablation Study (DCL6+ACL2 Base)"
    echo "============================================================="
    echo "GPU 0 (5 experiments):"
    echo "  - CompleteSeparated"
    echo "  - LogDet: 1e-6, 1e-5"
    echo "  - ScaleCtxK: 3, 7"
    echo ""
    echo "GPU 1 (5 experiments):"
    echo "  - SpatialCtxK: 5, 7"
    echo "  - TailW: 0.1, 0.3, 0.5"
    echo ""
    echo "GPU 4 (4 experiments):"
    echo "  - TailW: 0.8"
    echo "  - TailTopK: 0.01, 0.03, 0.05"
    echo ""
    echo "GPU 5 (4 experiments):"
    echo "  - TailTopK: 0.10"
    echo "  - ScoreTopK: 5, 7, 10"
    echo ""
    echo "Total: 18 experiments"
    echo "Log directory: $LOG_DIR"
    echo "============================================================="
}

# =============================================================================
# Main Execution
# =============================================================================
case "${1:-all}" in
    "gpu0")
        run_gpu0
        ;;
    "gpu1")
        run_gpu1
        ;;
    "gpu4")
        run_gpu4
        ;;
    "gpu5")
        run_gpu5
        ;;
    "all")
        print_summary
        echo ""
        echo "Starting all experiments in parallel on GPUs 0, 1, 4, 5..."
        echo ""

        nohup bash -c "cd /Volume/DeCoFlow && LOG_DIR=$LOG_DIR; BASE_CMD=\"$BASE_CMD\"; $(declare -f run_gpu0); run_gpu0" > $LOG_DIR/gpu0.log 2>&1 &
        PID_GPU0=$!

        nohup bash -c "cd /Volume/DeCoFlow && LOG_DIR=$LOG_DIR; BASE_CMD=\"$BASE_CMD\"; $(declare -f run_gpu1); run_gpu1" > $LOG_DIR/gpu1.log 2>&1 &
        PID_GPU1=$!

        nohup bash -c "cd /Volume/DeCoFlow && LOG_DIR=$LOG_DIR; BASE_CMD=\"$BASE_CMD\"; $(declare -f run_gpu4); run_gpu4" > $LOG_DIR/gpu4.log 2>&1 &
        PID_GPU4=$!

        nohup bash -c "cd /Volume/DeCoFlow && LOG_DIR=$LOG_DIR; BASE_CMD=\"$BASE_CMD\"; $(declare -f run_gpu5); run_gpu5" > $LOG_DIR/gpu5.log 2>&1 &
        PID_GPU5=$!

        echo "Process IDs:"
        echo "  GPU 0: $PID_GPU0"
        echo "  GPU 1: $PID_GPU1"
        echo "  GPU 4: $PID_GPU4"
        echo "  GPU 5: $PID_GPU5"
        echo ""
        echo "Monitor: tail -f $LOG_DIR/gpu*.log"
        ;;
    "summary")
        print_summary
        ;;
    *)
        echo "Usage: $0 {gpu0|gpu1|gpu4|gpu5|all|summary}"
        print_summary
        ;;
esac
