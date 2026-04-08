#!/bin/bash
# Depth Push (DP) 实验矩阵 — 推深循环的论文方案对比
# 基线: CR5 (c16_d4, 286M) + SJ1 (SigReg ct)
# 方案:
#   3: Exit entropy regularization (Ouro, NeurIPS 2025)
#   6: Time/step-size conditioning (LoopFormer, ICLR 2026)
#   4: RLTT dense LM loss at intermediate loops (Princeton 2026)
#   1: Shortcut-consistency training (LoopFormer, ICLR 2026)
#   5: Coconut continuous thought re-injection (Meta, ICLR 2025) — 长期方向
# 组合: 3+6 / 3+4 / 3+6+4 / 1+6
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

TRAIN="--iters 500 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 20 --dod_interval 200 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 2 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

# E9 + SJ1 最优
BASE="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
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
echo "║  Depth Push 实验 (8 experiments × 500 steps)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "基线: CR5 (c16_d4, 286M) + SJ1 (SigReg ct)"
echo "关键指标: avg_loops 是否突破 3"
echo ""

# DP0: 对照 (CR5+SJ1, 无新方案)
run_experiment "DP0_baseline"

# DP1: 方案 3 — Exit entropy (Ouro)
run_experiment "DP1_entropy" --exit_entropy_weight 0.1

# DP2: 方案 6 — Time conditioning (LoopFormer)
run_experiment "DP2_time" --enable_time_conditioning 1

# DP3: 方案 3+6 — Entropy + Time
run_experiment "DP3_entropy_time" --exit_entropy_weight 0.1 --enable_time_conditioning 1

# DP4: 方案 4 — RLTT dense LM loss
run_experiment "DP4_rltt" --loop_lm_loss_weight 0.05

# DP5: 方案 3+4 — Entropy + RLTT
run_experiment "DP5_entropy_rltt" --exit_entropy_weight 0.1 --loop_lm_loss_weight 0.05

# DP6: 方案 1 — Shortcut consistency (LoopFormer)
run_experiment "DP6_shortcut" --shortcut_consistency_weight 0.1 --loop_lm_loss_weight 0.01

# DP7: 方案 3+6+4 — Full combo (Entropy + Time + RLTT)
run_experiment "DP7_full" --exit_entropy_weight 0.1 --enable_time_conditioning 1 --loop_lm_loss_weight 0.05

# DP8: 方案 5 — Coconut (c_t → thought token → re-inject, 1 round)
run_experiment "DP8_coconut" --enable_coconut 1 --coconut_rounds 1

# DP9: Coconut + Entropy (最侵入式组合)
run_experiment "DP9_coconut_entropy" --enable_coconut 1 --coconut_rounds 1 --exit_entropy_weight 0.1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Depth Push Results                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for name in DP0_baseline DP1_entropy DP2_time DP3_entropy_time DP4_rltt DP5_entropy_rltt DP6_shortcut DP7_full DP8_coconut DP9_coconut_entropy; do
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
