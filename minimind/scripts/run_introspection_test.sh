#!/bin/bash
# IS (Introspection Stream) 实验矩阵 — 自省流结构优化
# 基线: CR5 (c16_d4, 286M) + SJ1 + E9 + Time Conditioning + LoRA32 (RS5 最优)
# A 组: 自省流输入升级 (memory token / chunked pooling)
# B 组: 自省流容量升级 (meta_dim / c_t_dim)
# C 组: c_t 注入方式升级 (token_aware / bixt / cmda)
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

# E9 + SJ1 + Time Cond + LoRA32 (RS5 最优)
BASE="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --world_jepa_weight 0.5 --mhc_streams 3 --exit_score_threshold 0.8 \
  --enable_sigreg_ct 1 --sigreg_ct_weight 0.05 \
  --enable_time_conditioning 1 --loop_lora_rank 32"

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
echo "║  IS 实验矩阵 (10 experiments × 350 steps)                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "基线: CR5 + SJ1 + E9 + Time Cond + LoRA32"
echo "目标: 升级自省流，打破 c_t 信息瓶颈"
echo ""

# ═══════════════════════════════════════════════
# A 组: 自省流输入升级
# ═══════════════════════════════════════════════

# IS0: 对照 (mean pooling + broadcast injection)
run_experiment "IS0_baseline"

# IS1: Memory token K=4
run_experiment "IS1_memory4" --introspection_input_mode memory --introspection_memory_tokens 4

# IS2: Memory token K=8
run_experiment "IS2_memory8" --introspection_input_mode memory --introspection_memory_tokens 8

# IS3: Chunked pooling (8 chunks)
run_experiment "IS3_chunked" --introspection_input_mode chunked

# ═══════════════════════════════════════════════
# B 组: 自省流容量升级
# ═══════════════════════════════════════════════

# IS4: meta_dim 96→192
run_experiment "IS4_meta192" --meta_dim 192

# IS5: c_t_dim 64→128
run_experiment "IS5_ct128" --c_t_dim 128

# ═══════════════════════════════════════════════
# C 组: c_t 注入方式升级
# ═══════════════════════════════════════════════

# IS6: Token-aware c_t injection
run_experiment "IS6_tokenaware" --introspection_inject_mode token_aware

# IS7: BiXT 双向 cross-attention (memory↔主流)
run_experiment "IS7_bixt" --introspection_inject_mode bixt --introspection_memory_tokens 4

# IS8: CMDA 双向通道调制 (SlowFast style)
run_experiment "IS8_cmda" --introspection_inject_mode cmda

# IS9: Memory K=4 + CMDA 调制注入
run_experiment "IS9_memory_cmda" --introspection_input_mode memory --introspection_memory_tokens 4 --introspection_inject_mode cmda

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IS Results                                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for name in IS0_baseline IS1_memory4 IS2_memory8 IS3_chunked IS4_meta192 IS5_ct128 IS6_tokenaware IS7_bixt IS8_cmda IS9_memory_cmda; do
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
