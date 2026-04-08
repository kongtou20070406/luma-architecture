#!/bin/bash
# Loop Depth 假设验证: 难数据 vs 简单数据，loop 深度会不会自然增加？
# 用 E9 winner 配置，只换数据源
# 2 experiments × 428 steps (2 epoch equivalent)
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix5_hyperopt_20260406"

ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --reason_shared_depth 2 \
  --mamba_chunk_size 32"

TRAIN_BASE="--batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 20 --dod_interval 100 --save_interval 0 \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 2 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

# E9 winner
E9_ARGS="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --world_jepa_weight 0.5 --mhc_streams 3 --exit_score_threshold 0.8"

run_experiment() {
    local name=$1; shift
    local data_path=$1; shift
    local iters=$1; shift
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
    $PYTHON $SCRIPT $ARCH $TRAIN_BASE --data_path "$data_path" --iters "$iters" $E9_ARGS $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    local exit_code=$?
    local final_loss=$(grep -oP 'loss_lm=[\d.]+' "$OUT_DIR/${name}.log" | tail -1)
    local avg_loops=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | tail -5 | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — ${final_loss}, avg_loops=${avg_loops}"
    fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Loop Depth 假设验证: 难数据 vs 简单数据                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# LD0: 简单数据 (原 V3, 同 FB baseline)
run_experiment "LD0_easy_data" "../dataset/pretrain_h_python.jsonl" 428

# LD1: 难数据 (MATH competition Level 4-5)
run_experiment "LD1_hard_math" "/home/kt/ai/luma-architecture/luma_dataset/synthetic/math_competition_hard.jsonl" 428

# LD2: 长推理链 (OpenR1-Math, 筛选 <=2048tok, avg 1475 tok/sample)
run_experiment "LD2_long_cot" "/home/kt/ai/luma-architecture/luma_dataset/synthetic/openr1_math_hard_2k.jsonl" 428

# LD3: OpenR1 筛选 <=4096tok + seq_len=4096 (无截断，测试长链 + 深循环)
run_experiment "LD3_long_cot_4k" "/home/kt/ai/luma-architecture/luma_dataset/synthetic/openr1_math_hard_4k.jsonl" 428 --max_seq_len 4096

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Loop Depth Results                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for name in LD0_easy_data LD1_hard_math LD2_long_cot LD3_long_cot_4k; do
    log="$OUT_DIR/${name}.log"
    if [ -f "$log" ]; then
        loss=$(grep -oP 'loss_lm=[\d.]+' "$log" | tail -1)
        # 提取所有 loops=N，计算平均
        avg=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
        max=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | sort -n | tail -1)
        echo "  $name: ${loss:-?}  avg_loops=${avg}  max_loops=${max:-?}"
    fi
done
echo ""
echo "Done. — $(date)"
