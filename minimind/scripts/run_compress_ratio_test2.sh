#!/bin/bash
# 压缩区/推理区比例实验 第二轮
# CR3 (c24_d4, 354M, loss=10.28) 是第一轮最优
# CR3b: 隔离变量 — 砍压缩但不加depth
# CR4-CR6: 在 CR3 基础上继续砍压缩区
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix5_hyperopt_20260406"
DATA="/home/kt/ai/luma-architecture/luma_dataset/synthetic/openr1_math_hard_2k.jsonl"

ARCH_BASE="--hidden_size 768 --intermediate_size 3072 \
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

SJ_BEST_ARGS="--enable_sigreg_ct 1 --sigreg_ct_weight 0.05"

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
    $PYTHON $SCRIPT $ARCH_BASE $TRAIN $E9_ARGS $SJ_BEST_ARGS $extra_args \
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
echo "║  CR 第二轮: 继续砍压缩区 (4 experiments × 350 steps)      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# CR3b: 24层压缩 + depth=2 (隔离: 砍压缩的收益 vs 加depth的收益)
run_experiment "CR3b_c24_d2" --compression_layers 24 --reason_shared_depth 2

# CR4: 20层压缩 + depth=4 (在CR3基础上继续砍)
run_experiment "CR4_c20_d4" --compression_layers 20 --reason_shared_depth 4

# CR5: 16层压缩 + depth=4 (接近极限)
run_experiment "CR5_c16_d4" --compression_layers 16 --reason_shared_depth 4

# CR6: 12层压缩 + depth=4 (极端: 压缩区极薄)
run_experiment "CR6_c12_d4" --compression_layers 12 --reason_shared_depth 4

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CR Round 2 Results                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for name in CR3b_c24_d2 CR4_c20_d4 CR5_c16_d4 CR6_c12_d4; do
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
