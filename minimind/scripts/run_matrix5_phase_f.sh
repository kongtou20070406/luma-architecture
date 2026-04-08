#!/bin/bash
# Matrix 5 Phase F: ExitController 深度循环实验
# 基线: E9 全 winner 配置 (MoR=0.7 + WJ=0.5 + MHC=3 + Exit=0.8)
# 目标: 让模型跑更深的循环，释放 MoR 多级路由潜力
# 4 experiments × 2000 steps
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix5_hyperopt_20260406"
DATA="../dataset/pretrain_h_python.jsonl"

mkdir -p "$OUT_DIR"

ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --reason_shared_depth 2 \
  --mamba_chunk_size 32"

TRAIN="--iters 428 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 20 --dod_interval 100 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 2 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

# E9 winner 配置作为基线参数
E9_ARGS="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --world_jepa_weight 0.5 --mhc_streams 3 --exit_score_threshold 0.8"

run_experiment() {
    local name=$1; shift
    local extra_args="$@"
    echo ""
    echo "================================================================"
    echo "  $name — $(date)"
    echo "================================================================"
    if [ -f "$OUT_DIR/${name}.log" ] && grep -q "^Done\." "$OUT_DIR/${name}.log" 2>/dev/null; then
        echo "  $name SKIPPED (already completed)"
        return 0
    fi
    cd "$TRAINER_DIR"
    $PYTHON $SCRIPT $ARCH $TRAIN $E9_ARGS $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — $(grep -oP 'loss_lm=[\d.]+' "$OUT_DIR/${name}.log" | tail -1)"
    fi
}

get_final_loss() {
    grep -oP 'loss_lm=[\d.]+' "$1" | tail -1 | grep -oP '[\d.]+'
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase F: ExitController 深度循环 (5 experiments × 428)    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "428 steps = 2 epoch, 带 c_t frozen injection fix"
echo ""

# FB: E9 配置 baseline (428 steps, 带 c_t fix)
run_experiment "FB_baseline_428"

# F0: min_loops=4, 强制至少 4 轮循环
run_experiment "F0_min_loops_4" --exit_min_loops 4

# F1: bias_init=-2.0, 初始倾向不退出
run_experiment "F1_bias_neg2" --exit_bias_init -2.0

# F2: warmup_steps=100, 前100步跑满循环 (428步总共，300太多)
run_experiment "F2_warmup_100" --exit_warmup_steps 100

# F3: 三者组合
run_experiment "F3_all_combined" --exit_min_loops 4 --exit_bias_init -2.0 --exit_warmup_steps 100

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase F RESULTS                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"

FB_LOSS=$(get_final_loss "$OUT_DIR/FB_baseline_428.log")
F0_LOSS=$(get_final_loss "$OUT_DIR/F0_min_loops_4.log")
F1_LOSS=$(get_final_loss "$OUT_DIR/F1_bias_neg2.log")
F2_LOSS=$(get_final_loss "$OUT_DIR/F2_warmup_100.log")
F3_LOSS=$(get_final_loss "$OUT_DIR/F3_all_combined.log")

echo "  FB baseline 428: ${FB_LOSS:-FAILED}"
echo "  F0 min_loops=4:  ${F0_LOSS:-FAILED}"
echo "  F1 bias=-2.0:    ${F1_LOSS:-FAILED}"
echo "  F2 warmup=100:   ${F2_LOSS:-FAILED}"
echo "  F3 all combined: ${F3_LOSS:-FAILED}"
echo ""
echo "Done. — $(date)"
