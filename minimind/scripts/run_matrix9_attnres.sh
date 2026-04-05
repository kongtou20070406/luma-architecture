#!/bin/bash
# Matrix 9: AttnRes 改造 — Kimi Block AttnRes (arxiv 2603.15031) vs current lerp
# 6 experiments × 2100 steps, seq=2048, bs=1, reason_loops=12
# Architecture: A1 (768h, L44, 12/3 heads, shared_depth=2)
# Baseline: B2' config (world_sigreg=0.10, mask=0.25) + mhc_alpha_init from B5' result
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix9_attnres_20260405"
DATA="../dataset/pretrain_h_python.jsonl"

mkdir -p "$OUT_DIR"

# A1 winner architecture (482M)
ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --reason_shared_depth 2 \
  --mamba_chunk_size 32"

# B2' winning config as baseline
TRAIN="--iters 2100 --batch_size 1 --max_seq_len 2048 --reason_loops 12 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 50 --dod_interval 200 --save_interval 500 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25"

# Use B5' mhc_alpha if it won, otherwise B2' default
# B2' won (MHC died in B5' with 0.05), using B2' default 0.01
MHC_ALPHA="--mhc_alpha_init 0.01"

run_experiment() {
    local name=$1; shift
    local extra_args="$@"
    echo ""
    echo "================================================================"
    echo "  $name — $(date)"
    echo "================================================================"
    cd "$TRAINER_DIR"
    $PYTHON $SCRIPT $ARCH $TRAIN $MHC_ALPHA $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    echo "  $name DONE — $(tail -1 "$OUT_DIR/${name}.log")"
}

echo "Matrix 9: AttnRes Refactor (A1 arch, 482M) — $(date)"
echo "Output: $OUT_DIR"
echo ""

# AR0: Baseline — current lerp AttnRes (same as B2' + mhc_alpha)
run_experiment "AR0_baseline" \
    --attnres_mode legacy

# AR1: Paper AttnRes in CompressionZone only, legacy in ReasoningLoop
run_experiment "AR1_compress_paper" \
    --attnres_compress_mode paper \
    --attnres_reason_mode legacy

# AR2: Legacy in CompressionZone, Paper AttnRes in ReasoningLoop only
run_experiment "AR2_reason_paper" \
    --attnres_compress_mode legacy \
    --attnres_reason_mode paper

# AR3: Full paper Block AttnRes (both zones)
run_experiment "AR3_full_paper" \
    --attnres_mode paper

# AR4: Full paper + input-dependent query (W_q @ h)
# NOTE: not implemented yet — skipping for now, will add if AR3 shows promise
# run_experiment "AR4_input_dep_query" \
#     --attnres_mode paper_input_dep

# AR5: Paper output (direct replace, V raw) but global pseudo_query
run_experiment "AR5_paper_global_q" \
    --attnres_mode paper_global_q

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE �� $(date)"
echo "================================================================"
echo ""

# Summary
echo "=== RESULTS SUMMARY ==="
for f in "$OUT_DIR"/AR*.log; do
    name=$(basename "$f" .log)
    last_loss=$(grep -oP 'loss_lm=[\d.]+' "$f" | tail -1)
    last_dod=$(grep "DOD/DMD" "$f" | tail -1)
    peak=$(grep "Peak VRAM" "$f")
    dead=$(echo "$last_dod" | grep -oP "dead=\[.*?\]")
    echo "$name: $last_loss | $peak | $dead"
done
