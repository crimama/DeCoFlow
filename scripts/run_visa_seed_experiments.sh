#!/bin/bash
# V61: VisA SOTA Seed Experiments (V59_04 config)
# 5 seeds × GPU 0-4
# Config: NCL10+ACL6, ViT224, vit_blocks [1,2,3,5], rank64, 100ep, lr=2e-4, tw=0.8

SEEDS=(314 999 2024 42 1024)
GPUS=(0 1 2 3 4)

cd /Volume/DeCoFlow

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU=${GPUS[$i]}
    EXP_NAME="V61_VisA_seed${SEED}"

    echo "[GPU ${GPU}] Launching ${EXP_NAME} (seed=${SEED})..."

    CUDA_VISIBLE_DEVICES=${GPU} python run_decoflow.py \
        --dataset visa \
        --data_path /Volume/VisA \
        --task_classes candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --vit_blocks 1 2 3 5 \
        --img_size 224 \
        --num_epochs 100 \
        --lr 2e-4 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --acl_n_layers 6 \
        --batch_size 16 \
        --use_tsa \
        --use_tail_aware_loss \
        --tail_weight 0.8 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 3 \
        --spatial_context_kernel 3 \
        --seed "${SEED}" \
        --experiment_name "${EXP_NAME}" \
        > "logs/${EXP_NAME}.log" 2>&1 &

    echo "  PID: $!"
done

echo ""
echo "All 5 VisA seed experiments launched."
echo "Monitor: tail -f logs/V61_VisA_seed*.log"
