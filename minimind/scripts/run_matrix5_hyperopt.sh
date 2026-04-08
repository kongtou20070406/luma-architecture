#!/bin/bash
# Matrix 5: 基线微调搜索 — 两阶段自适应设计
# Phase 1: 9 单因子实验 → Phase 2: 自动选 winner 组合 3 实验
# 基线: A1 (482M) + AR1 + GL1 + MH4 + EX5 全部胜出配置
# 12 experiments × 2000 steps, seq=2048, bs=1, accum=2
# 数据: pretrain_h_python.jsonl (6300 samples, ~3.2 epoch)
# 预计总时间: ~8 小时
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

TRAIN="--iters 2000 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 50 --dod_interval 200 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 2 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

run_experiment() {
    local name=$1; shift
    local extra_args="$@"
    echo ""
    echo "================================================================"
    echo "  $name — $(date)"
    echo "================================================================"
    # 跳过已完成的实验
    if [ -f "$OUT_DIR/${name}.log" ] && grep -q "^Done\." "$OUT_DIR/${name}.log" 2>/dev/null; then
        echo "  $name SKIPPED (already completed)"
        return 0
    fi
    cd "$TRAINER_DIR"
    $PYTHON $SCRIPT $ARCH $TRAIN $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — Metrics: $(grep -oP 'Metrics: \K.*' "$OUT_DIR/${name}.log" | tail -1)"
    fi
}

# 从 log 文件提取最终 loss
get_final_loss() {
    grep -oP 'loss_lm=[\d.]+' "$1" | tail -1 | grep -oP '[\d.]+'
}

# 比较两个 loss，返回较小的那个对应的标签
pick_winner() {
    local label_a=$1 loss_a=$2 label_b=$3 loss_b=$4
    if [ -z "$loss_a" ] || [ -z "$loss_b" ]; then
        echo "$label_a"  # fallback
        return
    fi
    local result=$(python3 -c "print('$label_a' if float('$loss_a') <= float('$loss_b') else '$label_b')")
    echo "$result"
}

echo "Matrix 5: Two-Phase Hyperopt (A1 482M) — $(date)"
echo "Output: $OUT_DIR"

# ================================================================
#  PHASE 1: 单因子实验 (9 experiments)
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: 单因子微调 (9 experiments × 2000 steps)          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# E0: Baseline
run_experiment "E0_baseline"

# Rho-1 维度
run_experiment "E1_rho1_070" --selective_loss_ratio 0.7
run_experiment "E2_rho1_050" --selective_loss_ratio 0.5

# True MoR 维度
run_experiment "E3_mor_cr07" \
    --enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01
run_experiment "E4_mor_cr05" \
    --enable_token_depth_routing 1 --mor_target_continue_ratio 0.5 --mor_balance_weight 0.01

# World-JEPA weight 维度
run_experiment "E5_wj_05" --world_jepa_weight 0.5
run_experiment "E6_wj_20" --world_jepa_weight 2.0

# MHC streams 维度
run_experiment "E7_mhc3" --mhc_streams 3

# Exit threshold 维度
run_experiment "E8_exit_08" --exit_score_threshold 0.8

# ================================================================
#  PHASE 1 结果分析 → 选出每维 winner
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1 RESULTS — 选择每维 winner                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

LOSS_E0=$(get_final_loss "$OUT_DIR/E0_baseline.log")
LOSS_E1=$(get_final_loss "$OUT_DIR/E1_rho1_070.log")
LOSS_E2=$(get_final_loss "$OUT_DIR/E2_rho1_050.log")
LOSS_E3=$(get_final_loss "$OUT_DIR/E3_mor_cr07.log")
LOSS_E4=$(get_final_loss "$OUT_DIR/E4_mor_cr05.log")
LOSS_E5=$(get_final_loss "$OUT_DIR/E5_wj_05.log")
LOSS_E6=$(get_final_loss "$OUT_DIR/E6_wj_20.log")
LOSS_E7=$(get_final_loss "$OUT_DIR/E7_mhc3.log")
LOSS_E8=$(get_final_loss "$OUT_DIR/E8_exit_08.log")

echo "  E0 baseline:   $LOSS_E0"
echo "  E1 rho1=0.7:   $LOSS_E1"
echo "  E2 rho1=0.5:   $LOSS_E2"
echo "  E3 mor cr=0.7: $LOSS_E3"
echo "  E4 mor cr=0.5: $LOSS_E4"
echo "  E5 wj=0.5:     $LOSS_E5"
echo "  E6 wj=2.0:     $LOSS_E6"
echo "  E7 mhc=3:      $LOSS_E7"
echo "  E8 exit=0.8:   $LOSS_E8"

# 每维选 winner（包括 baseline=不改）
# Rho-1: E0 vs E1 vs E2 → 三选一
BEST_RHO1=$($PYTHON -c "
losses = [('off', ${LOSS_E0:-999}), ('0.7', ${LOSS_E1:-999}), ('0.5', ${LOSS_E2:-999})]
best = min(losses, key=lambda x: x[1])
print(best[0])
")
# MoR: E0 vs E3 vs E4
BEST_MOR=$($PYTHON -c "
losses = [('off', ${LOSS_E0:-999}), ('0.7', ${LOSS_E3:-999}), ('0.5', ${LOSS_E4:-999})]
best = min(losses, key=lambda x: x[1])
print(best[0])
")
# World-JEPA: E5 vs E0 vs E6
BEST_WJ=$($PYTHON -c "
losses = [('0.5', ${LOSS_E5:-999}), ('1.0', ${LOSS_E0:-999}), ('2.0', ${LOSS_E6:-999})]
best = min(losses, key=lambda x: x[1])
print(best[0])
")
# MHC: E0(2) vs E7(3)
BEST_MHC=$($PYTHON -c "
losses = [('2', ${LOSS_E0:-999}), ('3', ${LOSS_E7:-999})]
best = min(losses, key=lambda x: x[1])
print(best[0])
")
# Exit: E0(default) vs E8(0.8)
BEST_EXIT=$($PYTHON -c "
losses = [('default', ${LOSS_E0:-999}), ('0.8', ${LOSS_E8:-999})]
best = min(losses, key=lambda x: x[1])
print(best[0])
")

echo ""
echo "  Winners:"
echo "    Rho-1:      $BEST_RHO1"
echo "    MoR:        $BEST_MOR"
echo "    World-JEPA: $BEST_WJ"
echo "    MHC:        $BEST_MHC"
echo "    Exit:       $BEST_EXIT"

# ================================================================
#  PHASE 2: Winner 组合实验 (3 experiments)
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Winner 组合 (3 experiments × 2000 steps)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# 构建每维 winner 的 CLI 参数
build_rho1_args() {
    case "$BEST_RHO1" in
        "0.7") echo "--selective_loss_ratio 0.7" ;;
        "0.5") echo "--selective_loss_ratio 0.5" ;;
        *) echo "" ;;
    esac
}
build_mor_args() {
    case "$BEST_MOR" in
        "0.7") echo "--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01" ;;
        "0.5") echo "--enable_token_depth_routing 1 --mor_target_continue_ratio 0.5 --mor_balance_weight 0.01" ;;
        *) echo "" ;;
    esac
}
build_wj_args() {
    case "$BEST_WJ" in
        "0.5") echo "--world_jepa_weight 0.5" ;;
        "2.0") echo "--world_jepa_weight 2.0" ;;
        *) echo "" ;;
    esac
}
build_mhc_args() {
    case "$BEST_MHC" in
        "3") echo "--mhc_streams 3" ;;
        *) echo "" ;;
    esac
}
build_exit_args() {
    case "$BEST_EXIT" in
        "0.8") echo "--exit_score_threshold 0.8" ;;
        *) echo "" ;;
    esac
}

RHO1_ARGS=$(build_rho1_args)
MOR_ARGS=$(build_mor_args)
WJ_ARGS=$(build_wj_args)
MHC_ARGS=$(build_mhc_args)
EXIT_ARGS=$(build_exit_args)

# E9: 全部 winner 组合
ALL_WINNER_ARGS="$RHO1_ARGS $MOR_ARGS $WJ_ARGS $MHC_ARGS $EXIT_ARGS"
echo "  E9 全部 winner 组合: $ALL_WINNER_ARGS"
run_experiment "E9_all_winners" $ALL_WINNER_ARGS

# E10: winner 去掉 MoR（测试 MoR 在组合中是否真的有用）
NO_MOR_ARGS="$RHO1_ARGS $WJ_ARGS $MHC_ARGS $EXIT_ARGS"
echo "  E10 winner 去 MoR: $NO_MOR_ARGS"
run_experiment "E10_winners_no_mor" $NO_MOR_ARGS

# E11: winner 去掉 Rho-1（测试 Rho-1 在组合中是否真的有用）
NO_RHO1_ARGS="$MOR_ARGS $WJ_ARGS $MHC_ARGS $EXIT_ARGS"
echo "  E11 winner 去 Rho-1: $NO_RHO1_ARGS"
run_experiment "E11_winners_no_rho1" $NO_RHO1_ARGS

# ================================================================
#  FINAL SUMMARY
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETE — $(date)  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "=== PHASE 1: 单因子 ==="
printf "%-25s %12s\n" "Experiment" "Final_Loss"
echo "--------------------------------------"
for f in "$OUT_DIR"/E[0-8]_*.log; do
    name=$(basename "$f" .log)
    loss=$(get_final_loss "$f")
    printf "%-25s %12s\n" "$name" "${loss:-FAILED}"
done

echo ""
echo "=== PHASE 2: Winner 组合 ==="
printf "%-25s %12s\n" "Experiment" "Final_Loss"
echo "--------------------------------------"
for f in "$OUT_DIR"/E{9,10,11}_*.log; do
    name=$(basename "$f" .log)
    loss=$(get_final_loss "$f")
    printf "%-25s %12s\n" "$name" "${loss:-FAILED}"
done

echo ""
echo "Per-dimension winners: Rho1=$BEST_RHO1  MoR=$BEST_MOR  WJ=$BEST_WJ  MHC=$BEST_MHC  Exit=$BEST_EXIT"

# DOD summary
echo ""
echo "=== DOD Summary ==="
for f in "$OUT_DIR"/E*.log; do
    name=$(basename "$f" .log)
    last_dod=$(grep 'DOD/DMD' "$f" | tail -1)
    if [ -n "$last_dod" ]; then
        printf "%-25s %s\n" "$name" "$last_dod"
    fi
done

# ================================================================
#  PHASE 3: V4 大数据集验证 (0.1 epoch, ~3100 steps)
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: V4 数据集 0.1 epoch (~3100 steps)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

DATA_V4="../dataset/pretrain_v4.jsonl"
TRAIN_V4="--iters 3100 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 50 --dod_interval 500 --save_interval 0 \
  --data_path $DATA_V4 \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 2 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

run_v4_experiment() {
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
    $PYTHON $SCRIPT $ARCH $TRAIN_V4 $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — Metrics: $(grep -oP 'Metrics: \K.*' "$OUT_DIR/${name}.log" | tail -1)"
    fi
}

# V0: V4 baseline (0.1 epoch)
run_v4_experiment "V0_v4_baseline"

# V1: V4 + Rho-1=0.7
run_v4_experiment "V1_v4_rho1_070" --selective_loss_ratio 0.7

# V2: V4 + Phase 1/2 best combo
run_v4_experiment "V2_v4_best_combo" $ALL_WINNER_ARGS

# V3: V4 + best combo + Rho-1=0.7
run_v4_experiment "V3_v4_best_combo_rho1" $ALL_WINNER_ARGS --selective_loss_ratio 0.7

echo ""
echo "=== PHASE 3: V4 Results ==="
printf "%-25s %12s\n" "Experiment" "Final_Loss"
echo "--------------------------------------"
for f in "$OUT_DIR"/V*.log; do
    name=$(basename "$f" .log)
    loss=$(get_final_loss "$f")
    printf "%-25s %12s\n" "$name" "${loss:-FAILED}"
done
echo ""
echo "ALL DONE — $(date)"
