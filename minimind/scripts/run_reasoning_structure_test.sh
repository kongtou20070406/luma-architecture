#!/bin/bash
# RS (Reasoning Structure) 实验矩阵 — 推理区结构优化
# 基线: CR5 (c16_d4, 286M) + SJ1 (SigReg ct) + E9 + Time Conditioning (DP2 最优)
# A 组: Minimum Loops 强制深度
# B 组: Loop LoRA (per-loop low-rank adaptation)
# C 组: Loop FFN Gating (loop-dependent gate)
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix5_hyperopt_20260406"
DATA="/home/kt/ai/luma-architecture/luma_dataset/synthetic/openr1_math_hard_2k.jsonl"

# CR5 架构: c16_d4
ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 16 --reason_shared_depth 4 \
  --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --mamba_chunk_size 32"

TRAIN="--iters 350 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 20 --dod_interval 120 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 2 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

# E9 + SJ1 最优 + Time Conditioning (DP2 最优)
BASE="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --world_jepa_weight 0.5 --mhc_streams 3 --exit_score_threshold 0.8 \
  --enable_sigreg_ct 1 --sigreg_ct_weight 0.05 \
  --enable_time_conditioning 1"

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
    local exit_code=0
    $PYTHON $SCRIPT $ARCH $TRAIN $BASE $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1 || exit_code=$?
    local final_loss=$(grep -oP 'loss_lm=[\d.]+' "$OUT_DIR/${name}.log" | tail -1)
    local avg_loops=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
    local params=$(grep -oP 'Params: [\d.]+M' "$OUT_DIR/${name}.log" | head -1)
    if [ $exit_code -ne 0 ]; then
        local err_line=$(grep -i "error\|exception\|traceback" "$OUT_DIR/${name}.log" | tail -1)
        echo "  $name FAILED (exit $exit_code) — $err_line"
    else
        echo "  $name DONE — ${final_loss}, avg_loops=${avg_loops}, ${params}"
    fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RS 实验矩阵 (9 experiments × 350 steps)                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "基线: CR5 + SJ1 + E9 + Time Conditioning"
echo "目标: 找到能推深循环的推理区结构"
echo ""

# ═══════════════════════════════════════════════
# A 组: Minimum Loops 强制深度
# ═══════════════════════════════════════════════

# RS0: 对照 (min_loops=2, time_cond=on, 复现 DP2)
run_experiment "RS0_baseline" --exit_min_loops 2

# RS1: 强制 5 轮
run_experiment "RS1_min5" --exit_min_loops 5

# RS2: 强制 8 轮
run_experiment "RS2_min8" --exit_min_loops 8

# RS3: min_loops=5, 不开 time conditioning (隔离 min_loops 效果)
run_experiment "RS3_min5_notime" --exit_min_loops 5 --enable_time_conditioning 0

# ═══════════════════════════════════════════════
# B 组: Loop LoRA (per-loop low-rank adaptation)
# ═══════════════════════════════════════════════

# RS4: LoRA rank=16
run_experiment "RS4_lora16" --loop_lora_rank 16

# RS5: LoRA rank=32
run_experiment "RS5_lora32" --loop_lora_rank 32

# RS6: LoRA rank=16 + min_loops=5
run_experiment "RS6_lora16_min5" --loop_lora_rank 16 --exit_min_loops 5

# ═══════════════════════════════════════════════
# C 组: Loop FFN Gating
# ═══════════════════════════════════════════════

# RS7: Loop FFN Gate
run_experiment "RS7_ffngate" --enable_loop_ffn_gate 1

# RS8: Loop FFN Gate + min_loops=5
run_experiment "RS8_ffngate_min5" --enable_loop_ffn_gate 1 --exit_min_loops 5

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RS Results                                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for name in RS0_baseline RS1_min5 RS2_min8 RS3_min5_notime RS4_lora16 RS5_lora32 RS6_lora16_min5 RS7_ffngate RS8_ffngate_min5; do
    log="$OUT_DIR/${name}.log"
    if [ -f "$log" ]; then
        loss=$(grep -oP 'loss_lm=[\d.]+' "$log" | tail -1)
        avg=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
        max=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | sort -n | tail -1)
        params=$(grep -oP 'Params: [\d.]+M' "$log" | head -1)
        echo "  $name: ${loss:-FAILED}  avg_loops=${avg}  max_loops=${max:-?}  ${params}"
    fi
done
echo ""
echo "Done. — $(date)"
