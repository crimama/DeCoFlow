#!/bin/bash
# Core Component Ablation Study (DCL6+ACL2 기준)
# MAIN: NCL=6, ACL=2, lr=3e-4, logdet=1e-4, scale_k=5, epochs=60
# GPU: 0, 1, 4, 5

TASK_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
DATA_PATH="/Data/MVTecAD"
LOG_BASE="/Volume/DeCoFlow/logs/Ablation/DCL6-ACL2-Component"

mkdir -p $LOG_BASE

# MAIN settings (DCL6+ACL2)
COMMON_ARGS="--dataset mvtec \
    --data_path $DATA_PATH \
    --task_classes $TASK_CLASSES \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 \
    --lr 3e-4 \
    --lora_rank 64 \
    --batch_size 16 \
    --num_coupling_layers 6 \
    --acl_n_layers 2 \
    --use_tsa \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5"

run_experiment() {
    local gpu=$1
    local name=$2
    local extra_args=$3
    local logfile="$LOG_BASE/gpu${gpu}_${name}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: Starting $name"
    CUDA_VISIBLE_DEVICES=$gpu python run_decoflow.py \
        $COMMON_ARGS \
        $extra_args \
        --experiment_name "$name" \
        --log_dir $LOG_BASE \
        > $logfile 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: Completed $name"
}

# ============================================================
# Batch 1: w/o LoRA, w/o ScaleCtx, w/o SpatialCtx, w/o TSA
# ============================================================
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Batch 1"
echo "============================================================"

run_experiment 0 "woLoRA" "--no_lora" &
PID0=$!
run_experiment 1 "woScaleCtx" "--no_scale_context" &
PID1=$!
run_experiment 4 "woSpatialCtx" "--no_spatial_context" &
PID4=$!
run_experiment 5 "woWhitening" "--no_tsa" &
PID5=$!

echo "Batch 1 PIDs: GPU0=$PID0, GPU1=$PID1, GPU4=$PID4, GPU5=$PID5"
wait $PID0 $PID1 $PID4 $PID5

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch 1 completed!"
echo ""

# ============================================================
# Batch 2: w/o TailLoss, w/o LogDet, w/o PosEmbed, Sequential
# ============================================================
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Batch 2"
echo "============================================================"

run_experiment 0 "woTailLoss" "--tail_weight 0" &
PID0=$!
run_experiment 1 "woLogDet" "--lambda_logdet 0" &
PID1=$!
run_experiment 4 "woPosEmbed" "--no_pos_embedding" &
PID4=$!
run_experiment 5 "Sequential" "--no_freeze_base" &
PID5=$!

echo "Batch 2 PIDs: GPU0=$PID0, GPU1=$PID1, GPU4=$PID4, GPU5=$PID5"
wait $PID0 $PID1 $PID4 $PID5

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch 2 completed!"
echo ""

# ============================================================
# Batch 3: Complete Separated (single experiment)
# ============================================================
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Batch 3"
echo "============================================================"

run_experiment 0 "CompleteSep" "--use_task_separated" &
PID0=$!

wait $PID0

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments completed!"
echo "============================================================"

# Print summary
echo ""
echo "=== Results Summary ==="
for name in woLoRA woScaleCtx woSpatialCtx woWhitening woTailLoss woLogDet woPosEmbed Sequential CompleteSep; do
    result_file="$LOG_BASE/$name/final_results.csv"
    if [ -f "$result_file" ]; then
        echo "--- $name ---"
        tail -1 "$result_file"
    else
        echo "--- $name: No results found ---"
    fi
done
