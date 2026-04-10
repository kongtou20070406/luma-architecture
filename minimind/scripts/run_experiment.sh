#!/bin/bash
# Luma 实验启动脚本 — 所有实验必须通过此脚本启动
# 用法: ./run_experiment.sh <实验名> [额外参数...]
# 例子: ./run_experiment.sh M2_ct_per_layer --ct_per_layer_inject 1
#       ./run_experiment.sh G0_baseline
set -e

if [ -z "$1" ]; then
    echo "用法: $0 <实验名> [额外参数...]"
    echo "例子: $0 M2_ct_per_layer --ct_per_layer_inject 1"
    exit 1
fi

EXP_NAME="$1"; shift

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
ARTIFACTS="/home/kt/ai/luma-architecture/minimind/artifacts"
DATA="/home/kt/ai/luma-architecture/luma_dataset/mixes/v5_pretrain.jsonl"

# ── 架构参数（293M，不要改） ──────────────────────────────────────────
ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 16 --reason_shared_depth 4 \
  --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --mamba_chunk_size 32"

# ── 训练参数（VRAM安全配置） ──────────────────────────────────────────
TRAIN="--iters 2001 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 --cpu_offload_optimizer 1 \
  --data_path $DATA \
  --phase 6 --world_jepa_mode full --world_jepa_weight 0.5 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 3 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3 \
  --cosine_total_steps 3500"

# ── IS9 基线（赫布+CMDA+MoR+LoRA32+Memory） ─────────────────────────
IS9="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --exit_score_threshold 0.8 --enable_sigreg_ct 1 --sigreg_ct_weight 0.05 \
  --enable_time_conditioning 1 --loop_lora_rank 32 \
  --introspection_input_mode memory --introspection_memory_tokens 4 --introspection_inject_mode cmda \
  --enable_neuromod_ct 1 --neuromod_mode jepa_surprise --neuromod_hebb_rank 32"

# ── G0 默认（slow_k=1, 无cos_sigreg） ────────────────────────────────
G0="--slow_k 1 --cos_sigreg_weight 0.0"

# ── 检查 GPU ──────────────────────────────────────────────────────────
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: GPU 上已有 $GPU_PROCS 个进程在跑！"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
    read -p "继续启动？(y/N) " confirm
    [ "$confirm" != "y" ] && exit 1
fi

LOG="$ARTIFACTS/${EXP_NAME}.log"
echo "================================================================"
echo "  实验: $EXP_NAME"
echo "  日志: $LOG"
echo "  额外参数: $@"
echo "  时间: $(date)"
echo "================================================================"

cd "$TRAINER_DIR"
PYTHONUNBUFFERED=1 nohup $PYTHON train_luma_refactor.py \
  $ARCH $TRAIN $IS9 $G0 "$@" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "查看日志: tail -f $LOG"

# 等几秒确认没有立即崩溃
sleep 5
if ! kill -0 $PID 2>/dev/null; then
    echo "ERROR: 进程已退出！查看日志:"
    tail -20 "$LOG"
    exit 1
fi

echo "启动成功，VRAM:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
