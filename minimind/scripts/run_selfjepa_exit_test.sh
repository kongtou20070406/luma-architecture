#!/bin/bash
# Self-JEPA 激活实验: SigReg 防坍缩 + c_t drift 参与退出决策
# 数据: OpenR1-Math (筛选 <=2048tok), 0.5 epoch ≈ 240 steps
# 基线: E9 winner (MoR + WJ + MHC + Exit)
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix5_hyperopt_20260406"
DATA="/home/kt/ai/luma-architecture/luma_dataset/synthetic/openr1_math_hard_2k.jsonl"

ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --reason_shared_depth 2 \
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

E9_ARGS="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --world_jepa_weight 0.5 --mhc_streams 3 --exit_score_threshold 0.8"

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
    $PYTHON $SCRIPT $ARCH $TRAIN $E9_ARGS $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    local exit_code=$?
    local final_loss=$(grep -oP 'loss_lm=[\d.]+' "$OUT_DIR/${name}.log" | tail -1)
    local avg_loops=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — ${final_loss}, avg_loops=${avg_loops}"
    fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Self-JEPA 激活实验 (6 experiments × 240 steps)            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "数据: OpenR1-Math 3k (2305条, avg 1916 tok, seq=3072)"
echo ""

# SJ0: 基线 (E9 winner on OpenR1 data, no changes)
run_experiment "SJ0_baseline"

# SJ1: + SigReg ct (防止 c_t 表示坍缩)
run_experiment "SJ1_sigreg_ct" --enable_sigreg_ct 1 --sigreg_ct_weight 0.05

# SJ2: + SigReg ct + rollout (全面防坍缩)
run_experiment "SJ2_sigreg_all" --enable_sigreg_ct 1 --sigreg_ct_weight 0.05 --enable_sigreg_rollout 1

# SJ3: + c_t drift 参与退出 (c_t 还在变 → 别退)
run_experiment "SJ3_ct_drift" --exit_ct_drift_weight 0.5

# SJ4: + c_t drift 强力版
run_experiment "SJ4_ct_drift_strong" --exit_ct_drift_weight 1.5

# SJ5: SigReg ct + c_t drift (组合)
run_experiment "SJ5_sigreg_drift" --enable_sigreg_ct 1 --sigreg_ct_weight 0.05 --exit_ct_drift_weight 0.5

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Self-JEPA Results                                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for name in SJ0_baseline SJ1_sigreg_ct SJ2_sigreg_all SJ3_ct_drift SJ4_ct_drift_strong SJ5_sigreg_drift; do
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
