#!/bin/bash
# Identity-Biased Recurrence 实验
# 基线: CR5 (c16_d4, 286M) + SJ1 (SigReg ct)
# 测试不同 alpha 值对循环深度的影响
# alpha=0 → 无 identity skip (对照)
# alpha=0.8 → h = 0.8*h_new + 0.2*h_old (论文推荐)
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

# E9 + SJ1 最优
E9_SJ1="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --world_jepa_weight 0.5 --mhc_streams 3 --exit_score_threshold 0.8 \
  --enable_sigreg_ct 1 --sigreg_ct_weight 0.05"

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
    $PYTHON $SCRIPT $ARCH $TRAIN $E9_SJ1 $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    local exit_code=$?
    local final_loss=$(grep -oP 'loss_lm=[\d.]+' "$OUT_DIR/${name}.log" | tail -1)
    local avg_loops=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
    local params=$(grep -oP 'Params: [\d.]+M' "$OUT_DIR/${name}.log" | head -1)
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — ${final_loss}, avg_loops=${avg_loops}, ${params}"
    fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Identity Recurrence 实验 (5 experiments × 350 steps)      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "基线: CR5 (c16_d4, 286M) + SJ1 (SigReg ct)"
echo ""

# IR0: 对照 (alpha=0, 无 identity skip)
run_experiment "IR0_alpha0" --identity_recurrence_alpha 0.0

# IR1: alpha=0.5 (50% new + 50% old)
run_experiment "IR1_alpha05" --identity_recurrence_alpha 0.5

# IR2: alpha=0.7 (70% new + 30% old)
run_experiment "IR2_alpha07" --identity_recurrence_alpha 0.7

# IR3: alpha=0.8 (论文推荐: 80% new + 20% old)
run_experiment "IR3_alpha08" --identity_recurrence_alpha 0.8

# IR4: alpha=0.9 (90% new + 10% old, 微量 identity)
run_experiment "IR4_alpha09" --identity_recurrence_alpha 0.9

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Identity Recurrence Results                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for name in IR0_alpha0 IR1_alpha05 IR2_alpha07 IR3_alpha08 IR4_alpha09; do
    log="$OUT_DIR/${name}.log"
    if [ -f "$log" ]; then
        loss=$(grep -oP 'loss_lm=[\d.]+' "$log" | tail -1)
        avg=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
        max=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | sort -n | tail -1)
        echo "  $name: ${loss:-FAILED}  avg_loops=${avg}  max_loops=${max:-?}"
    fi
done
echo ""
echo "Done. — $(date)"
