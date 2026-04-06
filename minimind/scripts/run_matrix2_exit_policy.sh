#!/bin/bash
# Matrix 2: Exit Policy — 在预训练中培养自适应退出能力
# 基线: AR1 + GL1 + MH4 (compress paper + accum=2 + streams=2)
# 6 experiments × 1500 steps, seq=2048, bs=1, accum=2
# Architecture: A1 (768h, L44, 12/3 heads, shared_depth=2)
#
# 核心问题: exit_ctrl 在所有之前实验中 dead
# 原因: exit_aux_weight=0.0 → ExitController 从未收到梯度
# 策略: 打开 exit_aux_weight + second_order_delta + 探索更多 loops
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix2_exit_20260406"
DATA="../dataset/pretrain_h_python.jsonl"

mkdir -p "$OUT_DIR"

# A1 winner architecture (482M)
ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --reason_shared_depth 2 \
  --mamba_chunk_size 32"

# AR1 + GL1 + MH4 winner config
TRAIN="--iters 1500 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 50 --dod_interval 200 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 2 --mhc_alpha_init 0.01"

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
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — $(tail -1 "$OUT_DIR/${name}.log")"
    fi
}

echo "Matrix 2: Exit Policy (A1 arch, 482M, AR1+GL1+MH4) — $(date)"
echo "Output: $OUT_DIR"
echo ""

# EX0: Baseline — exit_aux=0, loops=12 (exit_ctrl will be dead, same as before)
run_experiment "EX0_baseline_noaux" \
    --reason_loops 12 --exit_aux_weight 0.0

# EX1: 打开 exit_aux=0.01, loops=12 — 最小改动，让 exit_ctrl 有梯度
run_experiment "EX1_aux001_loops12" \
    --reason_loops 12 --exit_aux_weight 0.01

# EX2: exit_aux=0.05, loops=12 — 更强 exit 信号
run_experiment "EX2_aux005_loops12" \
    --reason_loops 12 --exit_aux_weight 0.05

# EX3: exit_aux=0.01 + second_order=0.3, loops=12 — 二阶差分帮助识别收敛
run_experiment "EX3_aux001_2nd03_loops12" \
    --reason_loops 12 --exit_aux_weight 0.01 \
    --exit_second_order_delta_weight 0.3

# EX4: exit_aux=0.01, loops=20 — 给更多预算，看 exit 能否学会提前退出
run_experiment "EX4_aux001_loops20" \
    --reason_loops 20 --exit_aux_weight 0.01

# EX5: exit_aux=0.01 + second_order=0.3, loops=20 — 核心实验
# 如果 EX5 loss ≈ EX4 但平均 loop 更低 → exit policy 有效
run_experiment "EX5_aux001_2nd03_loops20" \
    --reason_loops 20 --exit_aux_weight 0.01 \
    --exit_second_order_delta_weight 0.3

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""

# Summary
echo "=== RESULTS SUMMARY ==="
for f in "$OUT_DIR"/EX*.log; do
    name=$(basename "$f" .log)
    last_loss=$(grep -oP 'loss_lm=[\d.]+' "$f" | tail -1)
    peak=$(grep "Peak VRAM" "$f" | tail -1)
    dead=$(grep "DOD/DMD" "$f" | tail -1 | grep -oP "dead=\[.*?\]")
    echo "$name: $last_loss | $peak | $dead"
done
