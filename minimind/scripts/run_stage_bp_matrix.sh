#!/bin/bash
# Stage B' Experiment Matrix: seq=2048 + World-JEPA variations (MIMO upgrade)
# Winner: C5 config (compress=32, shared=2, 660M) + MIMO (rank=2, chunk=32)
# All runs: bs=2, seq=2048, 2500 steps, save_interval=500
# MIMO 后 VRAM ~7.6GB (seq=2048), 远低于 32GB 上限
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/stage_bp_20260405"
DATA="../dataset/pretrain_h_python.jsonl"

mkdir -p "$OUT_DIR"

# C5 winner architecture params (660M) + MIMO
ARCH="--hidden_size 1024 --intermediate_size 4096 \
  --compression_layers 32 --num_attention_heads 16 --num_key_value_heads 4 \
  --c_t_dim 128 --meta_dim 192 --mamba_d_state 256 --factorized_vocab_dim 256 \
  --reason_shared_depth 2 \
  --mamba_chunk_size 32"

# Shared training params — seq=2048, NO fp8_activation_compress (NaN with seq>=1024)
# 427 packs / bs=1 = 427 steps/epoch, 5 epoch ≈ 2135 steps
TRAIN="--iters 2100 --batch_size 1 --max_seq_len 2048 --reason_loops 12 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 50 --dod_interval 200 --save_interval 500 \
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
    echo "  $name DONE — $(tail -1 "$OUT_DIR/${name}.log")"
}

echo "Stage B' Experiment Matrix (seq=2048 + MIMO) — $(date)"
echo "Output: $OUT_DIR"
echo ""

# B0': Baseline — Phase 4, seq=2048, no World-JEPA
run_experiment "B0p_baseline" \
    --phase 4

# B1': LeWM (full), SIGreg=0.05, mask=0.25
run_experiment "B1p_lewm_sig005_mask025" \
    --phase 6 \
    --world_jepa_mode full \
    --world_jepa_weight 1.0 \
    --world_sigreg_weight 0.05 \
    --world_mask_ratio 0.25

# B2': LeWM (full), SIGreg=0.10, mask=0.25 (stronger regularization)
run_experiment "B2p_lewm_sig010_mask025" \
    --phase 6 \
    --world_jepa_mode full \
    --world_jepa_weight 1.0 \
    --world_sigreg_weight 0.10 \
    --world_mask_ratio 0.25

# B3': EMA (scaffold), SIGreg=0.05, mask=0.25, decay=0.996
run_experiment "B3p_ema_sig005_mask025" \
    --phase 6 \
    --world_jepa_mode scaffold \
    --world_jepa_weight 1.0 \
    --world_sigreg_weight 0.05 \
    --world_mask_ratio 0.25 \
    --world_ema_decay 0.996

# B4': LeWM (full), SIGreg=0.05, mask=0.50 (aggressive masking)
run_experiment "B4p_lewm_sig005_mask050" \
    --phase 6 \
    --world_jepa_mode full \
    --world_jepa_weight 1.0 \
    --world_sigreg_weight 0.05 \
    --world_mask_ratio 0.50

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""

# Summary
for f in "$OUT_DIR"/B*.log; do
    name=$(basename "$f" .log)
    last_loss=$(grep -oP 'loss_lm=[\d.]+' "$f" | tail -1)
    last_dod=$(grep "DOD/DMD" "$f" | tail -1)
    peak=$(grep "Peak VRAM" "$f")
    echo "$name: $last_loss | $last_dod | $peak"
done
