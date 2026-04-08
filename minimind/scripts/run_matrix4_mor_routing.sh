#!/bin/bash
# Matrix 4: MoR (Mixture-of-Reasoning) Routing
# 基线: A1 (482M) + AR1 + GL1 + MH4 + EX5 全部胜出配置
# 6 experiments × 1500 steps, seq=2048, bs=1, accum=2
# Architecture: A1 (768h, L44, 12/3 heads, shared_depth=2)
#
# 核心问题: 不同 reasoning loop 是否应该有不同的"思考风格"？
# MoR: loop-conditioned LoRA experts, 每个 loop 选择 top-k experts
# Expert: down(768→96) + SiLU + up(96→768), zero-init up → residual safe
#
# 实验设计:
#   D0: Baseline — MoR 关闭（现有最优配置）
#   D1: 4 experts, topk=2 — 默认配置
#   D2: 8 experts, topk=2 — 更多专家，更细粒度路由
#   D3: 4 experts, topk=1 — 更稀疏路由
#   D4: Rho-1 only (无 MoR, selective=0.6) — 隔离 Rho-1 效果
#   D5: 4 experts, topk=2 + selective_loss=0.6 — MoR + Rho-1 联合
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
SCRIPT="train_luma_refactor.py"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix4_mor_$(date +%Y%m%d)"
DATA="../dataset/pretrain_h_python.jsonl"

mkdir -p "$OUT_DIR"

# A1 winner architecture (482M)
ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --reason_shared_depth 2 \
  --mamba_chunk_size 32"

# 全部 M1+M9+M7+M10+M2 胜出配置
TRAIN="--iters 1500 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --log_interval 50 --dod_interval 200 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 \
  --world_jepa_mode full --world_jepa_weight 1.0 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 2 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

run_experiment() {
    local name=$1; shift
    local extra_args="$@"
    echo ""
    echo "================================================================"
    echo "  $name — $(date)"
    echo "================================================================"
    cd "$TRAINER_DIR"
    $PYTHON $SCRIPT $ARCH $TRAIN $extra_args \
        > "$OUT_DIR/${name}.log" 2>&1
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code) — $(tail -1 "$OUT_DIR/${name}.log")"
    else
        echo "  $name DONE — Metrics: $(grep -oP 'Metrics: \K.*' "$OUT_DIR/${name}.log" | tail -1)"
    fi
}

echo "Matrix 4: MoR Routing (A1 arch, 482M, full winner config) — $(date)"
echo "Output: $OUT_DIR"
echo ""

# D0: Baseline — MoR 关闭
run_experiment "D0_baseline_no_mor" \
    --reason_mor_routing 0

# D1: 4 experts, topk=2 (默认配置)
run_experiment "D1_4exp_topk2" \
    --reason_mor_routing 1 --reason_mor_num_experts 4 --reason_mor_topk 2

# D2: 8 experts, topk=2 (更多专家)
run_experiment "D2_8exp_topk2" \
    --reason_mor_routing 1 --reason_mor_num_experts 8 --reason_mor_topk 2

# D3: 4 experts, topk=1 (稀疏路由 — 每 loop 只选 1 个 expert)
run_experiment "D3_4exp_topk1" \
    --reason_mor_routing 1 --reason_mor_num_experts 4 --reason_mor_topk 1

# D4: Rho-1 only baseline (无 MoR, selective_loss=0.6) — 隔离 Rho-1 效果
run_experiment "D4_rho1_only" \
    --reason_mor_routing 0 \
    --selective_loss_ratio 0.6

# D5: 4 experts, topk=2 + Rho-1 selective loss 60% (联合实验)
run_experiment "D5_4exp_topk2_sel06" \
    --reason_mor_routing 1 --reason_mor_num_experts 4 --reason_mor_topk 2 \
    --selective_loss_ratio 0.6

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""

# Summary
echo "=== RESULTS SUMMARY ==="
for f in "$OUT_DIR"/D*.log; do
    name=$(basename "$f" .log)
    last_loss=$(grep -oP 'loss_lm=[\d.]+' "$f" | tail -1)
    peak=$(grep "Peak VRAM" "$f" | tail -1)
    dead=$(grep "DOD/DMD" "$f" | tail -1 | grep -oP "dead=\[.*?\]")
    loops=$(grep -oP 'loops=\d+/\d+' "$f" | tail -1)
    echo "$name: $last_loss | $peak | $dead | $loops"
done
