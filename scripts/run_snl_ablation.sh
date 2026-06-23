#!/bin/bash
# =================================================================
# V62: SNL Design Space Ablation
# Purpose: Compare Full SNL vs Decompose+Linear vs DeCoFlow (Ours)
# Base config: V48_01 (H04+HR, seed=0)
# =================================================================

# Common base config (matching V48_01 exactly)
COMMON="--dataset mvtec --data_path /Volume/MVTecAD \
    --task_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
    --backbone_name wide_resnet50_2 --use_high_res \
    --num_epochs 60 --lr 3e-4 --batch_size 16 --lora_rank 64 \
    --use_tsa --use_tail_aware_loss --tail_weight 0.85 --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k --score_aggregation_top_k 3 --lambda_logdet 1e-4 \
    --scale_context_kernel 5 --score_smooth_sigma 0.0"

# -----------------------------------------------------------------
# GPU 0: Full SNL (0 DCL + 8 ACB, default hidden_ratio=0.5)
# All blocks are task-specific ACB. No shared base at all.
# -----------------------------------------------------------------
echo "=== [GPU 0] V62_01: Full SNL (0 DCL + 8 ACB, hr=0.5) ==="
CUDA_VISIBLE_DEVICES=0 nohup python run_decoflow.py $COMMON \
    --num_coupling_layers 0 \
    --acb_n_blocks 8 \
    --acb_hidden_ratio 0.5 \
    --experiment_name "V62_01_FullSNL_8acb_hr05" \
    > logs/V62_01_FullSNL_8acb_hr05.log 2>&1 &
PID0=$!
echo "  PID=$PID0"

# -----------------------------------------------------------------
# GPU 1: Decompose + Linear (6 DCL with full independent subnets + 2 ACB)
# Uses use_regular_linear: each task gets independent full-rank subnets
# No shared base in DCL subnets (base exists as template, not used)
# -----------------------------------------------------------------
echo "=== [GPU 1] V62_02: Decompose + Linear (use_regular_linear) ==="
CUDA_VISIBLE_DEVICES=1 nohup python run_decoflow.py $COMMON \
    --num_coupling_layers 6 \
    --acb_n_blocks 2 \
    --use_regular_linear \
    --experiment_name "V62_02_DecompLinear_regularlinear" \
    > logs/V62_02_DecompLinear_regularlinear.log 2>&1 &
PID1=$!
echo "  PID=$PID1"

# -----------------------------------------------------------------
# GPU 2: Full SNL capacity-matched (0 DCL + 8 ACB, hidden_ratio=2.0)
# ACB blocks with same capacity as DCL subnets for fair comparison
# -----------------------------------------------------------------
echo "=== [GPU 2] V62_03: Full SNL capacity-matched (hr=2.0) ==="
CUDA_VISIBLE_DEVICES=2 nohup python run_decoflow.py $COMMON \
    --num_coupling_layers 0 \
    --acb_n_blocks 8 \
    --acb_hidden_ratio 2.0 \
    --experiment_name "V62_03_FullSNL_8acb_hr20" \
    > logs/V62_03_FullSNL_8acb_hr20.log 2>&1 &
PID2=$!
echo "  PID=$PID2"

# -----------------------------------------------------------------
# GPU 3: Decompose + LoRA high-rank (rank=512, 6 DCL + 2 ACB)
# Base frozen + near-full-rank adapter
# Tests if low-rank LoRA is truly sufficient
# -----------------------------------------------------------------
echo "=== [GPU 3] V62_04: Decompose + LoRA rank=512 ==="
CUDA_VISIBLE_DEVICES=3 nohup python run_decoflow.py $COMMON \
    --num_coupling_layers 6 \
    --acb_n_blocks 2 \
    --lora_rank 512 \
    --experiment_name "V62_04_DecompLoRA_rank512" \
    > logs/V62_04_DecompLoRA_rank512.log 2>&1 &
PID3=$!
echo "  PID=$PID3"

echo ""
echo "All experiments launched!"
echo "  GPU 0 (Full SNL hr=0.5):     PID=$PID0"
echo "  GPU 1 (Decompose+Linear):    PID=$PID1"
echo "  GPU 2 (Full SNL hr=2.0):     PID=$PID2"
echo "  GPU 3 (Decompose+LoRA r512): PID=$PID3"
echo ""
echo "Monitor with:"
echo "  tail -f logs/V62_01_FullSNL_8acb_hr05.log"
echo "  tail -f logs/V62_02_DecompLinear_regularlinear.log"
echo "  tail -f logs/V62_03_FullSNL_8acb_hr20.log"
echo "  tail -f logs/V62_04_DecompLoRA_rank512.log"
