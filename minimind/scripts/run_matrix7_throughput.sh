#!/bin/bash
# Matrix 7: Training Throughput — gradient accumulation + activation offload
# Goal: achieve effective bs=2 without OOM on RTX 5090 (32GB)
# Architecture: A1 (768h, L44, 12/3 heads, shared_depth=2, 482M)
# Baseline: B2' config (world_sigreg=0.10, mask=0.25)
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix7_throughput_20260405"
DATA="../dataset/pretrain_h_python.jsonl"

mkdir -p "$OUT_DIR"

ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --reason_shared_depth 2 \
  --mamba_chunk_size 32"

# Base training config (B2' winner)
BASE="--iters 1000 --max_seq_len 2048 --reason_loops 12 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 50 --dod_interval 200 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --mhc_alpha_init 0.05"

run_experiment() {
    local name=$1; shift
    local extra_args="$@"
    echo ""
    echo "================================================================"
    echo "  $name — $(date)"
    echo "================================================================"
    cd "$TRAINER_DIR"
    $PYTHON $SCRIPT $ARCH $BASE $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — $(tail -1 "$OUT_DIR/${name}.log")"
    fi
}

echo "Matrix 7: Training Throughput (A1 arch, 482M) — $(date)"
echo "Output: $OUT_DIR"
echo ""

# GL0: Baseline — bs=1, no accumulation (1000 steps for speed comparison)
run_experiment "GL0_baseline_bs1" \
    --batch_size 1 --accumulation_steps 1

# GL1: Gradient accumulation — bs=1, accum=2 (effective bs=2, zero VRAM overhead)
run_experiment "GL1_accum2" \
    --batch_size 1 --accumulation_steps 2

# GL2: Activation offload + accum=2
run_experiment "GL2_offload_accum2" \
    --batch_size 1 --accumulation_steps 2 \
    --activation_offload_compress 1

# GL3: Real bs=2 + activation offload (may OOM!)
run_experiment "GL3_bs2_offload" \
    --batch_size 2 --accumulation_steps 1 \
    --activation_offload_compress 1 || true

# GL4: Real bs=2 + offload + aggressive CUDA memory config
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
run_experiment "GL4_bs2_offload_aggressive" \
    --batch_size 2 --accumulation_steps 1 \
    --activation_offload_compress 1 || true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""

# Summary
echo "=== RESULTS SUMMARY ==="
for f in "$OUT_DIR"/GL*.log; do
    name=$(basename "$f" .log)
    last_loss=$(grep -oP 'loss_lm=[\d.]+' "$f" | tail -1)
    peak=$(grep "Peak VRAM" "$f" | tail -1)
    eta=$(grep -oP 'eta=[\d.]+min' "$f" | head -1)
    oom=$(grep -c "OutOfMemoryError\|CUDA out of memory" "$f")
    if [ "$oom" -gt 0 ]; then
        echo "$name: OOM! | $peak"
    else
        echo "$name: $last_loss | $peak | $eta"
    fi
done
