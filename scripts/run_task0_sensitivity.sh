#!/bin/bash
# Task 0 Sensitivity Experiment
# 5 different Task 0 configurations to analyze the impact of first task selection

# Base configuration (same as main experiment)
BASE_CONFIG="--dataset mvtec \
    --data_path /Data/MVTecAD \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 \
    --lr 3e-4 \
    --lora_rank 64 \
    --num_coupling_layers 6 \
    --acl_n_layers 2 \
    --batch_size 16 \
    --use_tsa \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5"

# All 15 MVTec classes
ALL_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

# GPU selection (use GPU 1)
export CUDA_VISIBLE_DEVICES=1

# Task 0 candidates (excluding bottle which is default)
TASK0_CANDIDATES=("wood" "transistor" "carpet" "metal_nut")

echo "Starting Task 0 Sensitivity Experiments..."
echo "GPU: $CUDA_VISIBLE_DEVICES"

for task0 in "${TASK0_CANDIDATES[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running with Task 0 = $task0"
    echo "=============================================="

    # Reorder classes: put task0 first, then rest in original order
    REORDERED_CLASSES="$task0"
    for cls in $ALL_CLASSES; do
        if [ "$cls" != "$task0" ]; then
            REORDERED_CLASSES="$REORDERED_CLASSES $cls"
        fi
    done

    EXP_NAME="Task0Sensitivity-${task0}"

    echo "Class order: $REORDERED_CLASSES"
    echo "Experiment name: $EXP_NAME"

    python run_decoflow.py \
        $BASE_CONFIG \
        --task_classes $REORDERED_CLASSES \
        --experiment_name "$EXP_NAME" \
        --log_dir logs/Task0_Sensitivity

    echo "Completed: $task0"
done

echo ""
echo "All Task 0 Sensitivity experiments completed!"
