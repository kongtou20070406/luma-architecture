#!/bin/bash
# NM+ES 实验矩阵 — 神经调制c_t写入 + 增强退出信号
# 基线: IS9 最优 (Time Cond + LoRA32 + Memory K=4 + CMDA)
# NM: Neuromodulated c_t writer (surprise-gated + Hebbian)
# ES: Enhanced exit signals (entropy/sensitivity/curvature/confidence)
# 数据: openr1_math_hard_4k.jsonl (v4), 500 steps
set -e
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT_PY="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix7_neuromod_20260408"
DATA="/home/kt/ai/luma-architecture/luma_dataset/synthetic/openr1_math_hard_4k.jsonl"
mkdir -p "$OUT_DIR"

ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 16 --reason_shared_depth 4 \
  --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --mamba_chunk_size 32"

TRAIN="--iters 500 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 --cpu_offload_optimizer 1 \
  --log_interval 20 --dod_interval 200 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 --world_jepa_mode full --world_jepa_weight 0.5 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 3 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

IS9="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --exit_score_threshold 0.8 --enable_sigreg_ct 1 --sigreg_ct_weight 0.05 \
  --enable_time_conditioning 1 --loop_lora_rank 32 \
  --introspection_input_mode memory --introspection_memory_tokens 4 --introspection_inject_mode cmda"

cd "$TRAINER_DIR"

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
    local exit_code=0
    $PYTHON $SCRIPT_PY $ARCH $TRAIN $IS9 $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1 || exit_code=$?
    local final_loss=$(grep -oP 'loss_lm=[\d.]+' "$OUT_DIR/${name}.log" | tail -1)
    local avg_loops=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
    local max_loops=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | grep -oP '\d+' | sort -n | tail -1)
    local params=$(grep -oP 'Params: [\d.]+M' "$OUT_DIR/${name}.log" | head -1)
    if [ $exit_code -ne 0 ]; then
        local err_line=$(grep -i "error\|exception\|traceback" "$OUT_DIR/${name}.log" | tail -1)
        echo "  $name FAILED (exit $exit_code) — $err_line"
    else
        echo "  $name DONE — ${final_loss}, avg_loops=${avg_loops}, max=${max_loops}, ${params}"
    fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  NM+ES 实验矩阵 (v4 data, 500 steps)                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ═══════════════════════════════════════════════
# NM: Neuromodulated c_t writer
# ═══════════════════════════════════════════════

# NM1: Surprise-gated write (no Hebbian, just gain modulation)
# neuromod_hebb_rank=0 不行 (linear size=0), 用 rank=1 最小化赫布项
run_experiment "NM1_surprise_gate" --enable_neuromod_ct 1

# NM2: + Hebbian rank=8 (default)
run_experiment "NM2_hebb8" --enable_neuromod_ct 1 --neuromod_hebb_rank 8

# NM3: + Delta Rule (减少赫布干扰)
run_experiment "NM3_delta_rule" --enable_neuromod_ct 1 --neuromod_hebb_rank 8 --neuromod_use_delta_rule 1

# NM4: Backpropamine (自学习调制信号)
run_experiment "NM4_learned" --enable_neuromod_ct 1 --neuromod_mode learned --neuromod_hebb_rank 8

# NM5: Hebbian rank=4 (消融)
run_experiment "NM5_hebb4" --enable_neuromod_ct 1 --neuromod_hebb_rank 4

# NM6: Hebbian rank=16 (消融)
run_experiment "NM6_hebb16" --enable_neuromod_ct 1 --neuromod_hebb_rank 16

# ═══════════════════════════════════════════════
# ES: Enhanced exit signals
# ═══════════════════════════════════════════════

# ES1: Entropy + confidence probe
run_experiment "ES1_entropy_conf" --enable_exit_entropy_signal 1 --enable_exit_confidence_gap 1

# ES2: Per-token sensitivity
run_experiment "ES2_token_sens" --enable_exit_token_sensitivity 1

# ES3: c_t curvature
run_experiment "ES3_ct_curv" --enable_exit_ct_curvature 1

# ES4: All ES signals
run_experiment "ES4_all_signals" --enable_exit_entropy_signal 1 --enable_exit_confidence_gap 1 \
  --enable_exit_token_sensitivity 1 --enable_exit_ct_curvature 1

# ═══════════════════════════════════════════════
# Combo: NM + ES
# ═══════════════════════════════════════════════

# C1: NM2 (Hebb8) + ES4 (all signals)
run_experiment "C1_hebb8_all_es" --enable_neuromod_ct 1 --neuromod_hebb_rank 8 \
  --enable_exit_entropy_signal 1 --enable_exit_confidence_gap 1 \
  --enable_exit_token_sensitivity 1 --enable_exit_ct_curvature 1

# C2: NM3 (Delta Rule) + ES4
run_experiment "C2_delta_all_es" --enable_neuromod_ct 1 --neuromod_hebb_rank 8 --neuromod_use_delta_rule 1 \
  --enable_exit_entropy_signal 1 --enable_exit_confidence_gap 1 \
  --enable_exit_token_sensitivity 1 --enable_exit_ct_curvature 1

# C3: NM4 (learned) + ES4
run_experiment "C3_learned_all_es" --enable_neuromod_ct 1 --neuromod_mode learned --neuromod_hebb_rank 8 \
  --enable_exit_entropy_signal 1 --enable_exit_confidence_gap 1 \
  --enable_exit_token_sensitivity 1 --enable_exit_ct_curvature 1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  NM+ES Results                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for name in NM1_surprise_gate NM2_hebb8 NM3_delta_rule NM4_learned NM5_hebb4 NM6_hebb16 \
            ES1_entropy_conf ES2_token_sens ES3_ct_curv ES4_all_signals \
            C1_hebb8_all_es C2_delta_all_es C3_learned_all_es; do
    log="$OUT_DIR/${name}.log"
    if [ -f "$log" ]; then
        loss=$(grep -oP 'loss_lm=[\d.]+' "$log" | tail -1)
        avg=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
        max=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | sort -n | tail -1)
        params=$(grep -oP 'Params: [\d.]+M' "$log" | head -1)
        echo "  $name: ${loss:-FAILED}  avg=${avg}  max=${max:-?}  ${params}"
    fi
done
echo ""
echo "Done. — $(date)"
