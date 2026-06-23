#!/bin/bash
# V60: Seed Experiments for MVTec SOTA (V48_01 H04+HR config)
# 7 seeds × 7 GPUs (0,1,2,3,4,6,7) — GPU 5 excluded
# Base config: NCL6+ACB2, rank64, WRN50, high_res, 60ep, lr=3e-4, tw=0.85

SEEDS=(42 123 256 512 1024 2024 7777)
GPUS=(0 1 2 3 4 6 7)

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU=${GPUS[$i]}
    EXP_NAME="V60_seed${SEED}"

    echo "[GPU ${GPU}] Launching ${EXP_NAME} (seed=${SEED})..."

    CUDA_VISIBLE_DEVICES=${GPU} python run_decoflow.py \
        --dataset mvtec \
        --data_path /Volume/MVTecAD \
        --task_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
        --backbone_name wide_resnet50_2 \
        --use_high_res \
        --num_epochs 60 \
        --lr 3e-4 \
        --lora_rank 64 \
        --num_coupling_layers 6 \
        --acb_n_blocks 2 \
        --batch_size 16 \
        --use_tsa \
        --use_tail_aware_loss \
        --tail_weight 0.85 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --seed ${SEED} \
        --sigma_sweep \
        --experiment_name "${EXP_NAME}" \
        > logs/${EXP_NAME}.log 2>&1 &

    echo "  PID: $!"
done

echo ""
echo "All 7 seed experiments launched."
echo "Monitor with: tail -f logs/V60_seed*.log"
echo "Check GPUs:   nvidia-smi"
