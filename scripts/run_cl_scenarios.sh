#!/bin/bash
# run_cl_scenarios.sh
# Continual Learning 시나리오 실험 스크립트
# 기준 설정: MVTec-WRN50-TailW0.7-TopK3-TailTopK2-ScaleK5-lr3e-4-MAIN

set -e

# 기본 하이퍼파라미터 (MAIN 설정 기준)
BASE_ARGS="--dataset mvtec \
    --data_path /Data/MVTecAD \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 \
    --lr 3e-4 \
    --lora_rank 64 \
    --num_coupling_layers 8 \
    --acl_n_layers 4 \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --log_dir ./logs/Final"

# 기본 클래스 순서 (알파벳 순)
DEFAULT_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

# Texture-first 순서
TEXTURE_FIRST_CLASSES="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"

# Hard-first 순서 (난이도 높은 순)
HARD_FIRST_CLASSES="screw toothbrush capsule cable pill grid zipper transistor carpet wood hazelnut metal_nut leather tile bottle"

# Easy-first 순서 (성능 높은 순)
EASY_FIRST_CLASSES="bottle leather metal_nut tile hazelnut carpet grid cable pill zipper transistor wood capsule toothbrush screw"

usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  all          Run all 6 experiments in parallel (requires 6 GPUs)"
    echo "  cl-3-3       Run CL 3-3 scenario (5 tasks, 3 classes each)"
    echo "  cl-5-5       Run CL 5-5 scenario (3 tasks, 5 classes each)"
    echo "  cl-14-1      Run CL 14-1 scenario (2 tasks, 14+1 classes)"
    echo "  texture      Run Texture-First ordering"
    echo "  hard         Run Hard-First ordering"
    echo "  easy         Run Easy-First ordering"
    echo "  sequential   Run all experiments sequentially on single GPU"
    echo ""
    echo "Examples:"
    echo "  $0 all                # Run all in parallel"
    echo "  $0 cl-3-3             # Run only CL 3-3 scenario"
    echo "  CUDA_VISIBLE_DEVICES=0 $0 sequential  # Run all on GPU 0"
}

run_cl_3_3() {
    echo "Running CL 3-3 scenario (5 tasks)..."
    python run_decoflow.py $BASE_ARGS \
        --task_classes $DEFAULT_CLASSES \
        --cl_scenario 3-3 \
        --experiment_name "MVTec-CL-3-3-MAIN"
}

run_cl_5_5() {
    echo "Running CL 5-5 scenario (3 tasks)..."
    python run_decoflow.py $BASE_ARGS \
        --task_classes $DEFAULT_CLASSES \
        --cl_scenario 5-5 \
        --experiment_name "MVTec-CL-5-5-MAIN"
}

run_cl_14_1() {
    echo "Running CL 14-1 scenario (2 tasks)..."
    python run_decoflow.py $BASE_ARGS \
        --task_classes $DEFAULT_CLASSES \
        --cl_scenario 14-1 \
        --experiment_name "MVTec-CL-14-1-MAIN"
}

run_texture_first() {
    echo "Running Texture-First ordering (1-1 scenario)..."
    python run_decoflow.py $BASE_ARGS \
        --task_classes $TEXTURE_FIRST_CLASSES \
        --cl_scenario 1-1 \
        --experiment_name "MVTec-CL-1-1-TextureFirst-MAIN"
}

run_hard_first() {
    echo "Running Hard-First ordering (1-1 scenario)..."
    python run_decoflow.py $BASE_ARGS \
        --task_classes $HARD_FIRST_CLASSES \
        --cl_scenario 1-1 \
        --experiment_name "MVTec-CL-1-1-HardFirst-MAIN"
}

run_easy_first() {
    echo "Running Easy-First ordering (1-1 scenario)..."
    python run_decoflow.py $BASE_ARGS \
        --task_classes $EASY_FIRST_CLASSES \
        --cl_scenario 1-1 \
        --experiment_name "MVTec-CL-1-1-EasyFirst-MAIN"
}

case "$1" in
    all)
        echo "Starting all 6 CL scenario experiments in parallel..."
        echo "This requires 6 available GPUs (0-5)"

        CUDA_VISIBLE_DEVICES=0 python run_decoflow.py $BASE_ARGS \
            --task_classes $DEFAULT_CLASSES \
            --cl_scenario 3-3 \
            --experiment_name "MVTec-CL-3-3-MAIN" &

        CUDA_VISIBLE_DEVICES=1 python run_decoflow.py $BASE_ARGS \
            --task_classes $DEFAULT_CLASSES \
            --cl_scenario 5-5 \
            --experiment_name "MVTec-CL-5-5-MAIN" &

        CUDA_VISIBLE_DEVICES=2 python run_decoflow.py $BASE_ARGS \
            --task_classes $DEFAULT_CLASSES \
            --cl_scenario 14-1 \
            --experiment_name "MVTec-CL-14-1-MAIN" &

        CUDA_VISIBLE_DEVICES=3 python run_decoflow.py $BASE_ARGS \
            --task_classes $TEXTURE_FIRST_CLASSES \
            --cl_scenario 1-1 \
            --experiment_name "MVTec-CL-1-1-TextureFirst-MAIN" &

        CUDA_VISIBLE_DEVICES=4 python run_decoflow.py $BASE_ARGS \
            --task_classes $HARD_FIRST_CLASSES \
            --cl_scenario 1-1 \
            --experiment_name "MVTec-CL-1-1-HardFirst-MAIN" &

        CUDA_VISIBLE_DEVICES=5 python run_decoflow.py $BASE_ARGS \
            --task_classes $EASY_FIRST_CLASSES \
            --cl_scenario 1-1 \
            --experiment_name "MVTec-CL-1-1-EasyFirst-MAIN" &

        wait
        echo "All CL scenario experiments completed!"
        ;;

    cl-3-3)
        run_cl_3_3
        ;;

    cl-5-5)
        run_cl_5_5
        ;;

    cl-14-1)
        run_cl_14_1
        ;;

    texture)
        run_texture_first
        ;;

    hard)
        run_hard_first
        ;;

    easy)
        run_easy_first
        ;;

    sequential)
        echo "Running all experiments sequentially..."
        run_cl_3_3
        run_cl_5_5
        run_cl_14_1
        run_texture_first
        run_hard_first
        run_easy_first
        echo "All experiments completed!"
        ;;

    *)
        usage
        exit 1
        ;;
esac
