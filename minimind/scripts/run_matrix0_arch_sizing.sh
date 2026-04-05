#!/bin/bash
# Matrix 0: Architecture Sizing �� find ~450M config on RTX 5090 (32GB)
# 4 configs × 200 steps each, seq=2048, bs=1, reason_loops=12
# Key metric: step-200 loss + peak VRAM
# A0 已跑完 (389M, loss=5.94, VRAM=9.15GB)，跳过
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix0_arch_20260405"
DATA="../dataset/pretrain_h_python.jsonl"

mkdir -p "$OUT_DIR"

# Shared training params — bs=1 固定 (bs=2 在 32GB VRAM 下 OOM)
TRAIN="--iters 200 --batch_size 1 --max_seq_len 2048 --reason_loops 12 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 25 --dod_interval 100 --save_interval 9999 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 4 \
  --reason_shared_depth 2 --factorized_vocab_dim 256 \
  --mamba_d_state 192 --mamba_chunk_size 32"

run_experiment() {
    local name=$1; shift
    local arch_args="$@"
    echo ""
    echo "================================================================"
    echo "  $name — $(date)"
    echo "================================================================"
    cd "$TRAINER_DIR"
    $PYTHON $SCRIPT $arch_args $TRAIN \
        > "$OUT_DIR/${name}_bs1.log" 2>&1
    echo "  $name DONE — $(tail -1 "$OUT_DIR/${name}_bs1.log")"
}

echo "Matrix 0: Architecture Sizing (bs=1) — $(date)"
echo "Output: $OUT_DIR"
echo ""

# A0 已完成: 389M, loss=5.94, VRAM=9.15GB — 跳过

# A1: Deep — 768h, L44 (456M) — narrow but deep, more Mamba layers
run_experiment "A1_deep_768h_L44" \
    --hidden_size 768 --intermediate_size 3072 \
    --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3

# A2: Wide — 832h, L36 (447M) — wider per-layer capacity
run_experiment "A2_wide_832h_L36" \
    --hidden_size 832 --intermediate_size 3328 \
    --compression_layers 36 --num_attention_heads 13 --num_key_value_heads 4

# A3: Balanced — 800h, L40 (453M) �� compromise
run_experiment "A3_bal_800h_L40" \
    --hidden_size 800 --intermediate_size 3200 \
    --compression_layers 40 --num_attention_heads 12 --num_key_value_heads 4

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""

# Summary (including A0 from previous run)
echo "=== RESULTS SUMMARY ==="
echo "A0_base_768h_L36: loss_lm=5.9355 | Peak VRAM: 9367 MB (previous run)"
for f in "$OUT_DIR"/A{1,2,3}*_bs1.log; do
    name=$(basename "$f" _bs1.log)
    last_loss=$(grep -oP 'loss_lm=[\d.]+' "$f" | tail -1)
    last_dod=$(grep "DOD/DMD" "$f" | tail -1)
    peak=$(grep "Peak VRAM" "$f")
    echo "$name: $last_loss | $peak"
done
