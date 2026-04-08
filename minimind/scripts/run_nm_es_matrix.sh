#!/bin/bash
# NM (Neuromodulation) + ES (Exit Signal) 实验矩阵
# 基线: IS9 (Time Cond + LoRA32 + Memory K=4 + CMDA)
# 数据: v4 (4k), 500步
set -e
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

PYTHON="/home/kt/ai/.venvs/luma-global/bin/python"
OUT_DIR="/home/kt/ai/luma-architecture/minimind/artifacts/matrix7_nm_es_20260408"
DATA="/home/kt/ai/luma-architecture/luma_dataset/synthetic/openr1_math_hard_4k.jsonl"
mkdir -p "$OUT_DIR"

ARCH="--hidden_size 768 --intermediate_size 3072 \
  --compression_layers 16 --reason_shared_depth 4 \
  --num_attention_heads 12 --num_key_value_heads 3 \
  --c_t_dim 64 --meta_dim 96 --mamba_d_state 192 --factorized_vocab_dim 256 \
  --mamba_chunk_size 32"

TRAIN="--iters 500 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 --cpu_offload_optimizer 1 \
  --log_interval 20 --dod_interval 200 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 --world_jepa_mode full --world_jepa_weight 0.5 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 3 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

# IS9 最优基线
IS9="--enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 --mor_balance_weight 0.01 \
  --exit_score_threshold 0.8 --enable_sigreg_ct 1 --sigreg_ct_weight 0.05 \
  --enable_time_conditioning 1 --loop_lora_rank 32 \
  --introspection_input_mode memory --introspection_memory_tokens 4 --introspection_inject_mode cmda"

cd /home/kt/ai/luma-architecture/minimind/trainer

run_exp() {
    local name=$1; shift
    echo ""
    echo "================================================================"
    echo "  $name — $(date)"
    echo "================================================================"
    if [ -f "$OUT_DIR/${name}.log" ] && grep -q "^Done\." "$OUT_DIR/${name}.log" 2>/dev/null; then
        echo "  $name SKIPPED (already done)"
        return 0
    fi
    local exit_code=0
    $PYTHON train_luma_refactor.py $ARCH $TRAIN $IS9 "$@" \
      > "$OUT_DIR/${name}.log" 2>&1 || exit_code=$?
    local loss=$(grep -oP 'loss_lm=[\d.]+' "$OUT_DIR/${name}.log" | tail -1)
    local avg=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
    local max=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | grep -oP '\d+' | sort -n | tail -1)
    local params=$(grep -oP 'Params: [\d.]+M' "$OUT_DIR/${name}.log" | head -1)
    if [ $exit_code -ne 0 ]; then
        local err=$(grep -i "error\|exception\|traceback" "$OUT_DIR/${name}.log" | tail -1)
        echo "  $name FAILED (exit $exit_code) — $err"
    else
        echo "  $name DONE — ${loss}, avg=${avg}, max=${max}, ${params}"
    fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  NM+ES 实验矩阵 (v4, 500步)                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# NM0: 对照 (= LD0)
run_exp "NM0_baseline"

# ═══ NM: Neuromodulation 方向 ═══

# NM1: Surprise-Gated Write (最简版)
run_exp "NM1_surprise" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 0

# NM2: Surprise + Hebbian rank=8
run_exp "NM2_surprise_hebb8" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 8

# NM3: Surprise + Hebbian rank=8 + Delta Rule
run_exp "NM3_surprise_hebb_delta" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 8 --neuromod_use_delta_rule 1

# NM4: Backpropamine 自学习 M(t)
run_exp "NM4_learned" --enable_neuromod_ct 1 --neuromod_mode learned --neuromod_hebb_rank 8

# NM5: Hebbian rank=4 消融
run_exp "NM5_hebb4" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 4

# NM6: Hebbian rank=16 消融
run_exp "NM6_hebb16" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32

# ═══ ES: Exit Signal 方向 ═══

# ES1: Entropy probe
run_exp "ES1_entropy" --enable_exit_entropy_signal 1

# ES2: Token sensitivity
run_exp "ES2_token_sens" --enable_exit_token_sensitivity 1

# ES3: c_t curvature
run_exp "ES3_ct_curv" --enable_exit_ct_curvature 1

# ES4: Confidence gap
run_exp "ES4_confidence" --enable_exit_confidence_gap 1

# ES5: All ES signals combined
run_exp "ES5_all_signals" --enable_exit_entropy_signal 1 --enable_exit_token_sensitivity 1 --enable_exit_ct_curvature 1 --enable_exit_confidence_gap 1

# NMES1/NMES2: 已移除 — hebb8 效果差，用 FUSE 系列 (hebb16) 替代

# ═══ Rank 消融追加 ═══

# NM7: hebb rank=24
run_exp "NM7_hebb24" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 24

# NM8: hebb rank=32
run_exp "NM8_hebb32" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32

# NM9: hebb rank=20 — 16和24之间的甜蜜点？
run_exp "NM9_hebb20" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 20

# NM10: hebb rank=16 + jepa_surprise (短实验验证 jepa vs self_check)
run_exp "NM10_jepa_hebb16" --enable_neuromod_ct 1 --neuromod_mode jepa_surprise --neuromod_hebb_rank 32

# ═══ 三合: hebb16 + 两个 ES ═══

# FUSE_B1: hebb16 + confidence + ct_curv
run_exp "FUSE_B1_hebb16_conf_curv" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32 --enable_exit_confidence_gap 1 --enable_exit_ct_curvature 1

# FUSE_B2: hebb16 + confidence + entropy
run_exp "FUSE_B2_hebb16_conf_entr" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32 --enable_exit_confidence_gap 1 --enable_exit_entropy_signal 1

# FUSE_B3: hebb16 + ct_curv + entropy
run_exp "FUSE_B3_hebb16_curv_entr" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32 --enable_exit_ct_curvature 1 --enable_exit_entropy_signal 1

# ═══ 四合 + 变体 ═══

# FUSE1: hebb16 + confidence + ct_curv + entropy (四强)
run_exp "FUSE1_hebb16_3es" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32 --enable_exit_confidence_gap 1 --enable_exit_ct_curvature 1 --enable_exit_entropy_signal 1

# FUSE2: FUSE1 + jepa_surprise
run_exp "FUSE2_jepa_hebb16_3es" --enable_neuromod_ct 1 --neuromod_mode jepa_surprise --neuromod_hebb_rank 32 --enable_exit_confidence_gap 1 --enable_exit_ct_curvature 1 --enable_exit_entropy_signal 1

# FUSE3: FUSE1 + bias=-1
run_exp "FUSE3_hebb16_3es_bias" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32 --enable_exit_confidence_gap 1 --enable_exit_ct_curvature 1 --enable_exit_entropy_signal 1 --exit_bias_init -1.0

# ═══ PC: Predictive Coding 方向 ═══

# PC1: PC alone (alpha=0.1)
run_exp "PC1_alpha01" --enable_pc_correction 1 --pc_alpha 0.1

# PC2: PC alpha=0.05 (更温和)
run_exp "PC2_alpha005" --enable_pc_correction 1 --pc_alpha 0.05

# PC3: PC + hebb16
run_exp "PC3_hebb16" --enable_pc_correction 1 --pc_alpha 0.1 --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32

# PC4: PC + hebb16 + confidence + entropy (PC + 最强组合)
run_exp "PC4_hebb16_conf_entr" --enable_pc_correction 1 --pc_alpha 0.1 --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32 --enable_exit_confidence_gap 1 --enable_exit_entropy_signal 1

# PC5: PC + hebb16 + jepa_surprise
run_exp "PC5_jepa_hebb16" --enable_pc_correction 1 --pc_alpha 0.1 --enable_neuromod_ct 1 --neuromod_mode jepa_surprise --neuromod_hebb_rank 32

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  NM+ES+PC Results                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
for name in NM0_baseline NM1_surprise NM2_surprise_hebb8 NM3_surprise_hebb_delta NM4_learned NM5_hebb4 NM6_hebb16 NM7_hebb24 NM8_hebb32 NM9_jepa_hebb16 ES1_entropy ES2_token_sens ES3_ct_curv ES4_confidence ES5_all_signals FUSE_B1_hebb16_conf_curv FUSE_B2_hebb16_conf_entr FUSE_B3_hebb16_curv_entr FUSE1_hebb16_3es FUSE2_jepa_hebb16_3es FUSE3_hebb16_3es_bias PC1_alpha01 PC2_alpha005 PC3_hebb16 PC4_hebb16_conf_entr PC5_jepa_hebb16; do
    log="$OUT_DIR/${name}.log"
    if [ -f "$log" ]; then
        loss=$(grep -oP 'loss_lm=[\d.]+' "$log" | tail -1)
        avg=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
        max=$(grep -oP 'loops=\d+' "$log" | grep -oP '\d+' | sort -n | tail -1)
        params=$(grep -oP 'Params: [\d.]+M' "$log" | head -1)
        echo "  $name: ${loss:-FAILED}  avg=${avg}  max=${max:-?}  ${params}"
    fi
done
echo "Done. — $(date)"

# ═══ 追加: NM + warmup 长训练 ═══
# 1000 步, warmup=200, 不开 RLTT (避免 OOM)

TRAIN_LONG="--iters 1000 --batch_size 1 --accumulation_steps 2 --max_seq_len 2048 \
  --fp8 1 --use_gradient_checkpointing 1 --cpu_offload_optimizer 1 \
  --log_interval 20 --dod_interval 300 --save_interval 0 \
  --data_path $DATA \
  --self_progress_shape_weight 0.05 --self_rollout_weight 0.1 \
  --phase 6 --world_jepa_mode full --world_jepa_weight 0.5 \
  --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --attnres_compress_mode paper --attnres_reason_mode legacy \
  --mhc_streams 3 --mhc_alpha_init 0.01 \
  --reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3"

run_exp_long() {
    local name=$1; shift
    echo ""
    echo "================================================================"
    echo "  $name (1000 steps) — $(date)"
    echo "================================================================"
    if [ -f "$OUT_DIR/${name}.log" ] && grep -q "^Done\." "$OUT_DIR/${name}.log" 2>/dev/null; then
        echo "  $name SKIPPED (already done)"
        return 0
    fi
    local exit_code=0
    $PYTHON train_luma_refactor.py $ARCH $TRAIN_LONG $IS9 "$@" \
      > "$OUT_DIR/${name}.log" 2>&1 || exit_code=$?
    local loss=$(grep -oP 'loss_lm=[\d.]+' "$OUT_DIR/${name}.log" | tail -1)
    local avg=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | grep -oP '\d+' | awk '{s+=$1;n++}END{if(n>0)printf "%.1f",s/n; else print "?"}')
    local max=$(grep -oP 'loops=\d+' "$OUT_DIR/${name}.log" | grep -oP '\d+' | sort -n | tail -1)
    local params=$(grep -oP 'Params: [\d.]+M' "$OUT_DIR/${name}.log" | head -1)
    if [ $exit_code -ne 0 ]; then
        echo "  $name FAILED (exit $exit_code)"
    else
        echo "  $name DONE — ${loss}, avg=${avg}, max=${max}, ${params}"
    fi
}

# ═══ 补充: rank=48 消融 ═══
run_exp "NM11_hebb48" --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 48

# ═══ Phase 6: 长训练 — 自动选短实验前三 ═══
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 6: 等短实验全部完成后手动选前三跑 1000 步             ║"
echo "║  请运行: run_long_top3.sh <name1> <name2> <name3>           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "短实验排名:"
# 自动排名
for name in NM0_baseline NM6_hebb16 NM7_hebb24 NM8_hebb32 NM9_jepa_hebb16 ES1_entropy ES3_ct_curv ES4_confidence ES5_all_signals FUSE_A1_hebb16_conf FUSE_A2_hebb16_curv FUSE_A3_hebb16_entr FUSE_B1_hebb16_conf_curv FUSE_B2_hebb16_conf_entr FUSE_B3_hebb16_curv_entr FUSE1_hebb16_3es FUSE2_jepa_hebb16_3es FUSE3_hebb16_3es_bias PC1_alpha01 PC2_alpha005 PC3_hebb16 PC4_hebb16_conf_entr PC5_jepa_hebb16; do
    log="$OUT_DIR/${name}.log"
    if [ -f "$log" ] && grep -q "^Done\." "$log" 2>/dev/null; then
        loss=$(grep -oP 'loss_lm=[\d.]+' "$log" | tail -1 | grep -oP '[\d.]+')
        echo "  $loss $name"
    fi
done | sort -n
