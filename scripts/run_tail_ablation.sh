#!/bin/bash
# Tail Loss Hyperparameter Combination Experiments
# Combinations: tail_top_k_ratio x tail_weight
# GPU 0: ratio=0.01, weight=1.0
# GPU 1: ratio=0.01, weight=1.2
# GPU 4: ratio=0.02, weight=1.0
# GPU 5: ratio=0.02, weight=1.2

LOG_DIR="/Volume/DeCoFlow/logs/Ablation/TailCombination"
mkdir -p $LOG_DIR

# Base command (DCL6+ACL2 configuration)
BASE_CMD="python run_decoflow.py \
    --dataset mvtec \
    --data_path /Data/MVTecAD \
    --task_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 \
    --lr 3e-4 \
    --lora_rank 64 \
    --num_coupling_layers 6 \
    --acl_n_layers 2 \
    --batch_size 16 \
    --use_tsa \
    --use_tail_aware_loss \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5"

# GPU 0: ratio=0.01, weight=1.0
echo "[GPU 0] TailTopK-0.01-TailW-1.0"
CUDA_VISIBLE_DEVICES=0 $BASE_CMD \
    --tail_top_k_ratio 0.01 \
    --tail_weight 1.0 \
    --experiment_name "TailTopK-0.01-TailW-1.0" \
    --log_dir "$LOG_DIR" &

# GPU 1: ratio=0.01, weight=1.2
echo "[GPU 1] TailTopK-0.01-TailW-1.2"
CUDA_VISIBLE_DEVICES=1 $BASE_CMD \
    --tail_top_k_ratio 0.01 \
    --tail_weight 1.2 \
    --experiment_name "TailTopK-0.01-TailW-1.2" \
    --log_dir "$LOG_DIR" &

# GPU 4: ratio=0.02, weight=1.0
echo "[GPU 4] TailTopK-0.02-TailW-1.0"
CUDA_VISIBLE_DEVICES=4 $BASE_CMD \
    --tail_top_k_ratio 0.02 \
    --tail_weight 1.0 \
    --experiment_name "TailTopK-0.02-TailW-1.0" \
    --log_dir "$LOG_DIR" &

# GPU 5: ratio=0.02, weight=1.2
echo "[GPU 5] TailTopK-0.02-TailW-1.2"
CUDA_VISIBLE_DEVICES=5 $BASE_CMD \
    --tail_top_k_ratio 0.02 \
    --tail_weight 1.2 \
    --experiment_name "TailTopK-0.02-TailW-1.2" \
    --log_dir "$LOG_DIR" &

echo "All 4 experiments started!"
wait
echo "All experiments completed!"
