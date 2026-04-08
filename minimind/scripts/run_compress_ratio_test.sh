#!/bin/bash
# 压缩区/推理区比例实验: 压缩区是否太重导致推理循环无事可做？
# 核心假设: 44层压缩区本身已是完整LLM，推理循环只做边际精修
# 验证方法: 砍压缩层 + 增加推理depth，观察 avg_loops 是否自然增加
# 数据: OpenR1-Math (筛选 <=2048tok)
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix5_hyperopt_20260406"
DATA="/home/kt/ai/luma-architecture/luma_dataset/synthetic/openr1_math_hard_2k.jsonl"

# compression_layers 和 reason_shared_depth 每个实验单独指定
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

# SJ最优 = SJ1: SigReg ct (-5.9% vs baseline)
SJ_BEST_ARGS="--enable_sigreg_ct 1 --sigreg_ct_weight 0.05"

# E9 基础参数
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
echo "║  压缩区/推理区比例实验 (4 experiments × 350 steps)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "假设: 44层压缩区太重，推理循环无事可做 → loops=2"
echo ""

# CR0: 当前 baseline (44层压缩 + 2层推理)
run_experiment "CR0_c44_d2" --compression_layers 44 --reason_shared_depth 2

# CR1: 砍压缩区到36层，推理不变 (验证: 纯砍压缩能否驱动更深循环)
run_experiment "CR1_c36_d2" --compression_layers 36 --reason_shared_depth 2

# CR2: 32层压缩 + 3层推理 (参数量接近CR1，但推理更强)
run_experiment "CR2_c32_d3" --compression_layers 32 --reason_shared_depth 3

# CR3: 激进平衡 24层压缩 + 4层推理 (大幅削弱压缩，强化推理)
run_experiment "CR3_c24_d4" --compression_layers 24 --reason_shared_depth 4

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Compress Ratio Results                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  预期: 压缩层越少 → avg_loops 越高 (推理区需要更多循环补偿)"
echo ""

for name in CR0_c44_d2 CR1_c36_d2 CR2_c32_d3 CR3_c24_d4; do
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
