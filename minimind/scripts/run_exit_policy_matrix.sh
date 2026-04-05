#!/bin/bash
# 4.1 Exit Policy Experiment Matrix
# Tests: second-order delta exit + varying min_loops + reason_active_loops
# Base: C5 config (660M), Phase 4, seq=512, bs=4 (fast iteration)
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/exit_policy_20260405"
DATA="../dataset/pretrain_h_python.jsonl"

mkdir -p "$OUT_DIR"

# C5 architecture (660M)
ARCH="--hidden_size 1024 --intermediate_size 4096 \
  --compression_layers 32 --num_attention_heads 16 --num_key_value_heads 4 \
  --c_t_dim 128 --meta_dim 192 --mamba_d_state 256 --factorized_vocab_dim 256 \
  --reason_shared_depth 2"

# Fast iteration: seq=512, bs=4 (same as C4/C5 long runs for comparability)
TRAIN="--iters 2500 --batch_size 4 --reason_loops 12 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 50 --dod_interval 500 --save_interval 500 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1"

run_experiment() {
    local name=$1; shift
    local extra_args="$@"
    echo ""
    echo "================================================================"
    echo "  $name — $(date)"
    echo "================================================================"
    cd "$TRAINER_DIR"
    $PYTHON $SCRIPT $ARCH $TRAIN $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  $name FAILED (rc=$rc) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — $(tail -1 "$OUT_DIR/${name}.log")"
    fi
}

echo "4.1 Exit Policy Experiment Matrix — $(date)"
echo "Output: $OUT_DIR"
echo ""

# ── E0: Baseline — 当前 exit policy, loops=12 ──────────────────────────
# 对照组：和 C5-long 相同配置但只跑 2500 步
run_experiment "E0_baseline_loops12" \
    --phase 4

# ── E1: 二阶差分 exit, weight=0.3 ──────────────────────────────────────
# Phase 1 核心实验：二阶差分信号帮助 exit controller 识别真正收敛
run_experiment "E1_second_order_w03" \
    --phase 4 \
    --exit_second_order_delta_weight 0.3

# ── E2: 二阶差分 exit, weight=0.5 (更强) ──────────────────────────────
run_experiment "E2_second_order_w05" \
    --phase 4 \
    --exit_second_order_delta_weight 0.5

# ── E3: 更多 loops (20), 测试 exit 能否有效利用 ─────────────────────────
# 关键问题：10x20 ≈ 10x15? 如果 exit 有效，20 loops 应该比 12 好
run_experiment "E3_loops20_no_2nd" \
    --phase 4 \
    --reason_loops 20

# ── E4: 20 loops + 二阶差分 (核心对比) ──────────────────────────────────
# 如果 E4 > E3 且 E4 > E0，说明二阶差分让模型真正利用了更多 loops
run_experiment "E4_loops20_second_order_w03" \
    --phase 4 \
    --reason_loops 20 \
    --exit_second_order_delta_weight 0.3

# ── E5: 20 loops + 二阶差分 + 降低 min_loops (让 exit 更自由) ──────────
run_experiment "E5_loops20_2nd_minloops1" \
    --phase 4 \
    --reason_loops 20 \
    --exit_second_order_delta_weight 0.3 \
    --exit_train_use_sampling 1

echo ""
echo "================================================================"
echo "  ALL EXIT POLICY EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""

# Summary
for f in "$OUT_DIR"/E*.log; do
    name=$(basename "$f" .log)
    last_loss=$(grep -oP 'loss_lm=[\d.]+' "$f" | tail -1)
    last_dod=$(grep "DOD/DMD" "$f" | tail -1)
    peak=$(grep "Peak VRAM" "$f")
    echo "$name: $last_loss | $last_dod | $peak"
done
