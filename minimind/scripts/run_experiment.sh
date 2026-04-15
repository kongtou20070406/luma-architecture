#!/bin/bash
# Luma 实验启动脚本 — 默认 Stellarator (Hero v19 生产形态, 215M, seq=2048, 1 epoch)
# 4.15 v19 扶正：主干 F_main(h) 不看 c_t + low-rank modulator(c_t) + sigmoid gated fusion
# 实测 fp_proxy L=0.062, sig_raw=0, ema=7.52 @ step 26250
# 用法: ./run_experiment.sh <实验名> [额外参数...]
# 例子: ./run_experiment.sh baseline                        # 默认 v19 Stellarator
#       ./run_experiment.sh legacy_v13 --stellarator_mode 0 --loop_lora_rank 32  # 回退 v13
set -e

if [ -z "$1" ]; then
    echo "用法: $0 <实验名> [额外参数...]"
    echo "例子: $0 hero_v7_baseline"
    exit 1
fi

EXP_NAME="$1"; shift

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
TRAINER_DIR="/home/kt/ai/luma-architecture/minimind/trainer"
ARTIFACTS="/home/kt/ai/luma-architecture/minimind/artifacts"
LOG_DIR="$ARTIFACTS/phase_e"
DATA="/home/kt/ai/luma-architecture/luma_dataset/mixes/v5_pretrain.jsonl"

# ── 架构参数（216M hero v7，Phase E damped 生产形态） ─────────────────
# hero v6/v7 实测最稳定规模：compression=12, reason_depth=2, reason_loops=4
# 扩到 reason_depth=4 会触发 body Lipschitz > 1，damped 收缩失败
ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 12 --reason_shared_depth 2 \
  --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --mamba_chunk_size 32 --reason_loops 4"

# ── 训练参数 ────────────────────────────────────────────────────────
# 1 epoch v5 = 80444 iters (bs=1, accum=2, 160889 packs)
# fp8=0, grad_ckpt=0: Phase E 二阶导需求 + Mamba3 reentrant 冲突
# activation_offload_compress 补偿 VRAM
TRAIN="--iters 80444 --cosine_total_steps 80444 \
  --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --data_path $DATA \
  --fp8 0 --use_gradient_checkpointing 0 \
  --activation_offload_compress 1 --cpu_offload_optimizer 1"

# ── Phase E damped（生产形态：一阶近似 h ← (1-η)h + η·F(h)） ──────────
PHASE_E="--enable_energy_reason_core 1 \
  --phase_e_K_max 3 --phase_e_eta 0.5 \
  --phase_e_damped_mode 1 --phase_e_k_backprop 1"

# ── Phase 6: scaffold World-JEPA (用户红线：双流 JEPA) ────────────────
# scaffold mode + block mask + LeWM Cramér-Wold SIGReg
# 4.14 v17 调参：降 JEPA 难度
# - mask_ratio 0.6→0.4: 减少 mask 区域，cosine 预测更容易
# - sigreg_weight 0.05→0.02: v16 sigreg_raw 爆到 80，惩罚项贡献 loss_w≈2（太大）
# - world_jepa_weight 0.5→0.3: 整体降 backward 贡献
PHASE6="--phase 6 --world_jepa_mode scaffold --world_jepa_weight 0.3 \
  --world_sigreg_weight 0.02 --world_mask_ratio 0.4 \
  --world_mask_scheme block --world_mask_block_mean 32 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 3 --mhc_alpha_init 0.01 \
  --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

# ── IS9 记忆栈：introspection memory + CMDA + MoR + 赫布（Stellarator 下 LoRA=0）────
# 4.14 v18: 关 self-JEPA 所有 SIGReg (ct/delta/rollout/loop/cos)。
#   LeJEPA paper SIGReg 只用于 World-JEPA 防 encoder 坍缩，
#   c_t 是 64 维人格向量，有 RMSNorm + ct_norm clamp，不需要额外正则。
# 4.15 v19: loop_lora_rank=0（Stellarator 主干不需要 LoRA 制造 per-loop 差异）。
IS9="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --exit_score_threshold 0.8 --enable_sigreg_ct 0 --sigreg_ct_weight 0.0 \
  --sigreg_delta_weight 0.0 \
  --enable_time_conditioning 1 --loop_lora_rank 0 \
  --introspection_input_mode memory --introspection_memory_tokens 4 --introspection_inject_mode cmda \
  --enable_neuromod_ct 1 --neuromod_mode jepa_surprise --neuromod_hebb_rank 32"

# ── Stellarator 仿星器架构 (v19 生产形态) ─────────────────────────────
# 主干 F_main(h) 不看 c_t + low-rank modulator(c_t, rank=8) + sigmoid gated fusion
# 实测 fp_proxy L=0.062 (严格收缩), ema=7.52 @ step 26250 (碾压 v16 同期 ~17)
# 理论: Lip(h_next, h) ≤ 1 + g·α ≈ 1.05
STELLARATOR="--stellarator_mode 1 --stellarator_mod_rank 8"

# ── 检查 GPU ──────────────────────────────────────────────────────────
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: GPU 上已有 $GPU_PROCS 个进程在跑！"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
    read -p "继续启动？(y/N) " confirm
    [ "$confirm" != "y" ] && exit 1
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${EXP_NAME}.log"
echo "================================================================"
echo "  实验: $EXP_NAME (Phase E damped, 216M, 1 epoch)"
echo "  日志: $LOG"
echo "  额外参数: $@"
echo "  时间: $(date)"
echo "================================================================"

cd "$TRAINER_DIR"
PYTHONUNBUFFERED=1 nohup $PYTHON train_luma_refactor.py \
  $ARCH $TRAIN $PHASE_E $PHASE6 $IS9 $STELLARATOR "$@" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "查看日志: tail -f $LOG"

sleep 5
if ! kill -0 $PID 2>/dev/null; then
    echo "ERROR: 进程已退出！查看日志:"
    tail -20 "$LOG"
    exit 1
fi

echo "启动成功，VRAM:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
