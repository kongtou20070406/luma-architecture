#!/bin/bash
# Matrix 10: MHC 门控调优 — 救活 Multi-Head Compression 梯度流
# 基线: AR1 + GL1 (compress paper + reason legacy, accum=2)
# 6 experiments × 1500 steps, seq=2048, bs=1, accum=2, reason_loops=12
# Architecture: A1 (768h, L44, 12/3 heads, shared_depth=2)
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix10_mhc_20260406"
DATA="../dataset/pretrain_h_python.jsonl"

mkdir -p "$OUT_DIR"

# A1 winner architecture (482M)
ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --reason_shared_depth 2 \
  --mamba_chunk_size 32"

# AR1 + GL1 winner config
TRAIN="--iters 1500 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 --reason_loops 12 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 50 --dod_interval 200 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy"

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

echo "Matrix 10: MHC Gate Tuning (A1 arch, 482M, AR1+GL1 baseline) — $(date)"
echo "Output: $OUT_DIR"
echo ""

# MH0: Baseline — current alpha=0.01 (MHC typically dies)
run_experiment "MH0_alpha001" \
    --mhc_alpha_init 0.01

# MH1: alpha=0.03 — 中间值，比 0.01 更强但比 0.05 更保守
run_experiment "MH1_alpha003" \
    --mhc_alpha_init 0.03

# MH2: alpha=0.02 — 细粒度搜索
run_experiment "MH2_alpha002" \
    --mhc_alpha_init 0.02

# MH3: alpha=0.10 — 激进高温，让 routing 差异化更快
run_experiment "MH3_alpha010" \
    --mhc_alpha_init 0.10

# MH4: n_streams=2 — 减少 stream 数量，降低 routing 难度
run_experiment "MH4_streams2" \
    --mhc_alpha_init 0.01 --mhc_streams 2

# MH5: n_streams=8 — 更多 stream，看 routing 是否更难学
run_experiment "MH5_streams8" \
    --mhc_alpha_init 0.01 --mhc_streams 8

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""

# Summary
echo "=== RESULTS SUMMARY ==="
for f in "$OUT_DIR"/MH*.log; do
    name=$(basename "$f" .log)
    last_loss=$(grep -oP 'loss_lm=[\d.]+' "$f" | tail -1)
    peak=$(grep "Peak VRAM" "$f" | tail -1)
    dead=$(grep "DOD/DMD" "$f" | tail -1 | grep -oP "dead=\[.*?\]")
    echo "$name: $last_loss | $peak | $dead"
done
