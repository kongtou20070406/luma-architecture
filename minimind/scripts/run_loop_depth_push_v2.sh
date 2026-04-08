#!/bin/bash
# LD (Loop Depth Push v2) 实验矩阵 — 让循环真正有用
# 基线: IS9 最优 (Time Cond + LoRA32 + Memory K=4 + CMDA)
# 方向 1: Exit bias — 负偏置让模型倾向于晚退出
# 方向 2: Curriculum on depth — warmup 期间强制深循环
# 方向 3: Per-loop loss (RLTT) — 每轮独立学习信号
# 方向 4: 组合
# 数据: openr1_math_hard_4k.jsonl (v4), 每实验限时 30 分钟
set -e
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT_PY="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix6_loop_depth_20260408"
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

# IS9 最优基线
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
echo "║  LD 实验矩阵 — Loop Depth Push v2 (v4 data, 30min/exp)     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "基线: IS9 (Time Cond + LoRA32 + Memory K=4 + CMDA)"
echo ""

# LD0: 对照 (IS9 on v4 data, 500 steps)
run_experiment "LD0_baseline"

# ═══════════════════════════════════════════════
# 方向 1: Exit Bias — 负偏置让 exit_score 降低
# ═══════════════════════════════════════════════
# bias=-1 → sigmoid(-1+signals) 更难达到 threshold
run_experiment "LD1_bias_neg1" --exit_bias_init -1.0

# bias=-2 → 更强的不退出倾向
run_experiment "LD2_bias_neg2" --exit_bias_init -2.0

# ═══════════════════════════════════════════════
# 方向 2: Curriculum — warmup 期间强制不退出
# ═══════════════════════════════════════════════
# 前 100 步强制跑满（不退出），之后自由
run_experiment "LD3_warmup100" --exit_warmup_steps 100

# 前 200 步强制跑满
run_experiment "LD4_warmup200" --exit_warmup_steps 200

# ═══════════════════════════════════════════════
# 方向 3: Per-loop loss (RLTT) — 每轮独立梯度
# ═══════════════════════════════════════════════
# RLTT weight=0.1
run_experiment "LD5_rltt01" --loop_lm_loss_weight 0.1

# RLTT weight=0.3
run_experiment "LD6_rltt03" --loop_lm_loss_weight 0.3

# ═══════════════════════════════════════════════
# 方向 4: 组合
# ═══════════════════════════════════════════════
# bias=-1 + warmup100 + RLTT 0.1
run_experiment "LD7_combo1" --exit_bias_init -1.0 --exit_warmup_steps 100 --loop_lm_loss_weight 0.1

# bias=-2 + warmup200 + RLTT 0.3
run_experiment "LD8_combo2" --exit_bias_init -2.0 --exit_warmup_steps 200 --loop_lm_loss_weight 0.3

# warmup100 + RLTT 0.1 (no bias)
run_experiment "LD9_warmup_rltt" --exit_warmup_steps 100 --loop_lm_loss_weight 0.1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LD Results                                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for name in LD0_baseline LD1_bias_neg1 LD2_bias_neg2 LD3_warmup100 LD4_warmup200 LD5_rltt01 LD6_rltt03 LD7_combo1 LD8_combo2 LD9_warmup_rltt; do
    log="$OUT_DIR/${name}.log"
    if [ -f "$log" ]; then
        loss=$(grep -oP 'loss_lm=[\d.]+' "$log" | tail -1)
        avg=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
        max=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | sort -n | tail -1)
        params=$(grep -oP 'Params: [\d.]+M' "$log" | head -1)
        steps=$(grep -oP 'Done\. \d+ steps' "$log" | grep -oP '\d+')
        echo "  $name: ${loss:-FAILED}  avg=${avg}  max=${max:-?}  steps=${steps:-?}  ${params}"
    fi
done
echo ""
echo "Done. — $(date)"
