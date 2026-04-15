"""
Luma Training Refactor Trainer — Phase 0 / Phase 1+
====================================================
按照 docs/agent/Luma_Training_Refactor_Plan.md 的分阶段方案执行。

Phase 0 默认配置：
  - 只保留 loss_lm（所有 aux loss 权重归零）
  - 固定学习率（不用余弦退火）
  - 每 --grad_log_interval 步记录各模块梯度范数
  - 每步记录 loss_lm 到 artifacts/refactor_metrics.jsonl

用法：
  cd /home/kt/ai/luma-architecture/minimind
  source /home/kt/ai/.venvs/luma-global/bin/activate
  python trainer/train_luma_refactor.py \\
      --data_path dataset/pretrain_diag.jsonl \\
      --max_seq_len 512 \\
      --iters 1500 \\
      --phase 0
"""

import os
import sys
import json
import time
import math
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset
from dataset.packed_dataset import PackedPretrainDataset
from luma_stage0.dynamics_analysis import (
    CtStateTracker, GradTrajectoryTracker, LayerGradTracker, ExitDepthTracker,
    render_dynamics_report, render_markdown, save_report,
)
from luma_stage0.optimizers import LumaCosineScheduler, LumaMuonAdamWOptimizer, LumaOptimizerConfig
from model.model_minimind import LumaConfig, LumaForCausalLM
from trainer.trainer_utils import Logger, setup_seed

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "refactor"
CKPT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "checkpoints"


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    step: int,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    compress_probe: torch.nn.Module | None = None,
) -> None:
    """Save training checkpoint. Handles torch.compile wrapped models."""
    raw_model = getattr(model, "_orig_mod", model)
    ckpt = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }
    if compress_probe is not None:
        ckpt["compress_probe"] = compress_probe.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"Checkpoint saved: {path} (step {step})", file=sys.stderr)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    compress_probe: torch.nn.Module | None = None,
) -> int:
    """Load checkpoint, return the step to resume from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if compress_probe is not None and "compress_probe" in ckpt:
        compress_probe.load_state_dict(ckpt["compress_probe"])
    step = ckpt["step"]
    print(f"Checkpoint loaded: {path} (resuming from step {step})", file=sys.stderr)
    return step


def cleanup_checkpoints(ckpt_dir: Path, keep: int = 2, phase: int | None = None) -> None:
    """Keep only the most recent `keep` checkpoints (filtered by phase if given)."""
    pattern = f"phase{phase}_*.pt" if phase is not None else "*.pt"
    ckpts = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep]:
        old.unlink()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _base_arch_kwargs(args: argparse.Namespace) -> dict:
    """所有阶段共用的模型架构参数。"""
    return dict(
        vocab_size=args.vocab_size,
        factorized_vocab_dim=args.factorized_vocab_dim,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        reason_intermediate_size=args.reason_intermediate_size or args.intermediate_size,
        reason_shared_depth=args.reason_shared_depth,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        compression_layers=args.compression_layers,
        compression_active_layers=args.compression_active_layers or args.compression_layers,
        reason_loops=args.reason_loops,
        reason_loops_max=args.reason_loops_max,
        reason_active_loops=args.reason_active_loops or args.reason_loops,
        slow_k=args.slow_k,
        c_t_dim=args.c_t_dim,
        meta_dim=args.meta_dim,
        meta_state=args.meta_state,
        mamba_d_state=args.mamba_d_state,
        mamba_expand=args.mamba_expand,
        mamba_headdim=args.mamba_headdim,
        mamba_chunk_size=args.mamba_chunk_size,
        max_position_embeddings=args.max_seq_len,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        world_jepa_mode=args.world_jepa_mode,
        world_mask_ratio=args.world_mask_ratio,
        h_mask_ratio=getattr(args, "h_mask_ratio", 0.0),
        h_mask_surprise_weight=getattr(args, "h_mask_surprise_weight", 0.3),
        h_mask_loss_mode=getattr(args, "h_mask_loss_mode", "mse"),
        h_mask_loss_weight=getattr(args, "h_mask_loss_weight", 0.1),
        world_ema_decay=args.world_ema_decay,
        ct_world_reg_mode=getattr(args, "ct_world_reg_mode", "none"),
        ct_world_var_weight=getattr(args, "ct_world_var_weight", 1.0),
        ct_world_cov_weight=getattr(args, "ct_world_cov_weight", 0.04),
        self_rollout_steps=args.self_rollout_steps,
        self_check_dim=args.self_check_dim,
        self_check_k=args.self_check_k,
        exit_train_use_sampling=bool(args.exit_train_use_sampling),
        exit_eval_use_sampling=bool(args.exit_eval_use_sampling),
        exit_sampling_temperature=args.exit_sampling_temperature,
        exit_score_threshold=getattr(args, "exit_score_threshold", 0.85),
        exit_second_order_delta_weight=getattr(args, "exit_second_order_delta_weight", 0.0),
        exit_min_loops=getattr(args, "exit_min_loops", 2),
        exit_bias_init=getattr(args, "exit_bias_init", 0.0),
        exit_warmup_steps=getattr(args, "exit_warmup_steps", 0),
        exit_progressive_warmup=getattr(args, "exit_progressive_warmup", 0),
        exit_ct_drift_weight=getattr(args, "exit_ct_drift_weight", 0.0),
        identity_recurrence_alpha=getattr(args, "identity_recurrence_alpha", 0.0),
        exit_entropy_weight=getattr(args, "exit_entropy_weight", 0.0),
        loop_lm_loss_weight=getattr(args, "loop_lm_loss_weight", 0.0),
        rltt_stride=getattr(args, "rltt_stride", 2),
        shortcut_consistency_weight=getattr(args, "shortcut_consistency_weight", 0.0),
        enable_time_conditioning=bool(getattr(args, "enable_time_conditioning", 0)),
        enable_coconut=bool(getattr(args, "enable_coconut", 0)),
        coconut_rounds=getattr(args, "coconut_rounds", 1),
        loop_lora_rank=getattr(args, "loop_lora_rank", 0),
        enable_loop_ffn_gate=bool(getattr(args, "enable_loop_ffn_gate", 0)),
        introspection_input_mode=getattr(args, "introspection_input_mode", "mean"),
        introspection_memory_tokens=getattr(args, "introspection_memory_tokens", 4),
        introspection_inject_mode=getattr(args, "introspection_inject_mode", "broadcast"),
        enable_introspection_swa=bool(getattr(args, "enable_introspection_swa", 0)),
        neuromod_fox_decay=bool(getattr(args, "neuromod_fox_decay", 0)),
        enable_neuromod_ct=bool(getattr(args, "enable_neuromod_ct", 0)),
        neuromod_hebb_rank=getattr(args, "neuromod_hebb_rank", 8),
        neuromod_use_delta_rule=bool(getattr(args, "neuromod_use_delta_rule", 0)),
        neuromod_mode=getattr(args, "neuromod_mode", "surprise"),
        enable_pc_correction=bool(getattr(args, "enable_pc_correction", 0)),
        pc_alpha=getattr(args, "pc_alpha", 0.1),
        enable_exit_entropy_signal=bool(getattr(args, "enable_exit_entropy_signal", 0)),
        enable_exit_token_sensitivity=bool(getattr(args, "enable_exit_token_sensitivity", 0)),
        enable_exit_ct_curvature=bool(getattr(args, "enable_exit_ct_curvature", 0)),
        enable_exit_confidence_gap=bool(getattr(args, "enable_exit_confidence_gap", 0)),
        use_gradient_checkpointing=bool(args.use_gradient_checkpointing),
        activation_offload_compress=bool(getattr(args, "activation_offload_compress", 0)),
        mamba_fp8_activation_cache=bool(getattr(args, "mamba_fp8_activation_cache", 0)),
        mamba_fp8_act_block_size=getattr(args, "mamba_fp8_act_block_size", 128),
        reason_num_phases=getattr(args, "reason_num_phases", 0),
        reason_head_partition=bool(getattr(args, "reason_head_partition", 0)),
        reason_mor_routing=bool(getattr(args, "reason_mor_routing", 0)),
        reason_mor_num_experts=getattr(args, "reason_mor_num_experts", 4),
        reason_mor_topk=getattr(args, "reason_mor_topk", 2),
        enable_token_depth_routing=bool(getattr(args, "enable_token_depth_routing", 0)),
        mor_target_continue_ratio=getattr(args, "mor_target_continue_ratio", 0.6),
        mor_balance_weight=getattr(args, "mor_balance_weight", 0.01),
        mhc_alpha_init=getattr(args, "mhc_alpha_init", 0.01),
        mhc_streams=getattr(args, "mhc_streams", 4),
        attnres_mode=getattr(args, "attnres_mode", "legacy"),
        attnres_compress_mode=getattr(args, "attnres_compress_mode", ""),
        attnres_reason_mode=getattr(args, "attnres_reason_mode", ""),
        # ── Phase E 能量梯度流推理（主 backbone 集成）──────────────────
        enable_energy_reason_core=bool(getattr(args, "enable_energy_reason_core", 0)),
        phase_e_K_max=getattr(args, "phase_e_K_max", 3),
        phase_e_eta=getattr(args, "phase_e_eta", 0.1),
        phase_e_k_backprop=getattr(args, "phase_e_k_backprop", 1),
        phase_e_temperature=getattr(args, "phase_e_temperature", 0.0),
        phase_e_grad_stop_eps=getattr(args, "phase_e_grad_stop_eps", 0.0),
        phase_e_damped_mode=bool(getattr(args, "phase_e_damped_mode", 1)),
        # Stellarator mode (v19+): 主干/调制/融合三层推理核心
        stellarator_mode=bool(getattr(args, "stellarator_mode", 0)),
        stellarator_mod_rank=getattr(args, "stellarator_mod_rank", 8),
        # World JEPA 难度 / 防崩升级（2026-04-13）
        world_mask_scheme=getattr(args, "world_mask_scheme", "block"),
        world_mask_block_mean=getattr(args, "world_mask_block_mean", 32),
        world_mask_use_mask_token=bool(getattr(args, "world_mask_use_mask_token", 1)),
    )


def build_phase0_config(args: argparse.Namespace) -> LumaConfig:
    """Phase 0/1/2: 只留 lm_loss，所有辅助 loss 权重归零，禁用可关闭的辅助模块。"""
    return LumaConfig(
        **_base_arch_kwargs(args),
        # ── 所有辅助 loss 关闭 ──────────────────────────────────────────────
        enable_world_jepa=False,
        world_jepa_weight=0.0,
        disable_self_jepa=True,
        self_jepa_weight=0.0,
        self_rollout_weight=0.0,
        self_jepa_residual_reg=0.0,
        exit_aux_weight=0.0,
        rollout_zone_weight=0.0,
        routing_tier_entropy_weight=0.0,
        routing_min_local_share_weight=0.0,
        trajectory_vitality_weight=0.0,
        compression_dynamics_weight=0.0,
        enable_self_check_ring=False,
    )


def build_phase3_config(args: argparse.Namespace) -> LumaConfig:
    """Phase 3: 加入 self-JEPA loss（c_t.detach() stop-gradient，不污染主干）。
    compress_probe 继续保留（在 trainer 里，Phase 3 时 compress_weight > 0）。
    """
    return LumaConfig(
        **_base_arch_kwargs(args),
        # ── 加 self-JEPA，其余辅助 loss 仍关闭 ─────────────────────────────
        enable_world_jepa=False,
        world_jepa_weight=0.0,
        disable_self_jepa=False,          # 关键：开启 JEPA
        self_jepa_weight=args.self_jepa_weight,
        self_rollout_weight=0.0,          # rollout 暂不启用
        self_jepa_residual_reg=0.01,
        exit_aux_weight=0.0,
        rollout_zone_weight=0.0,
        routing_tier_entropy_weight=0.0,
        routing_min_local_share_weight=0.0,
        trajectory_vitality_weight=0.0,
        compression_dynamics_weight=0.0,
        enable_self_check_ring=False,
        # SIGreg 关闭（Phase 3.5 才开）
        enable_sigreg_delta=False,
        enable_sigreg_rollout=False,
    )


def build_phase35_config(args: argparse.Namespace) -> LumaConfig:
    """Phase 3.5: Phase 3 基础上加 SIGreg（对 self-JEPA 的 pred_delta_c 做正则）。
    SIGreg 来自 LeJEPA/LeWorldModel 论文（arXiv:2511.08544 / 2603.19312）：
      - 衡量 pred_delta_c 的分布与各向同性高斯 N(0,I) 的距离（Cramér-Wold 统计检验）
      - 防止 c_t predictor 输出坍缩到同一点
      - 不需要 stop-gradient（论文结论）
    注意：self_jepa_residual_reg 设为 0，避免与 SIGreg 方向冲突
    （residual_reg 收缩范数 vs SIGreg 扩张分布，两者同时开会在后期不稳定）。
    """
    return LumaConfig(
        **_base_arch_kwargs(args),
        enable_world_jepa=False,
        world_jepa_weight=0.0,
        disable_self_jepa=False,
        self_jepa_weight=args.self_jepa_weight,
        self_rollout_weight=0.0,
        self_jepa_residual_reg=0.0,   # 关闭：与 SIGreg 冲突
        exit_aux_weight=0.0,
        rollout_zone_weight=0.0,
        routing_tier_entropy_weight=0.0,
        routing_min_local_share_weight=0.0,
        trajectory_vitality_weight=0.0,
        compression_dynamics_weight=0.0,
        enable_self_check_ring=False,
        # ── SIGreg：加在 self-JEPA pred_delta_c 上 ──────────────────────────
        enable_sigreg_delta=True,         # 对 pred_delta_c 做 Cramér-Wold 正则
        enable_sigreg_rollout=False,      # rollout 还没启用，不需要
        sigreg_delta_weight=args.sigreg_delta_weight,
        # c_t SIGreg（方案 2，默认关闭）
        enable_sigreg_ct=args.enable_sigreg_ct,
        sigreg_ct_weight=args.sigreg_ct_weight,
        # SIGreg 超参（按论文推荐值）
        sigreg_num_slices=128,
        sigreg_t_min=0.2,
        sigreg_t_max=4.0,
        sigreg_num_points=17,
        sigreg_lambda=1.0,
        sigreg_eps=1e-6,
    )


def build_phase4_config(args: argparse.Namespace) -> LumaConfig:
    """Phase 4: Phase 3.5 基础上加 self_check_ring（stop-gradient，训练 ring 预测推理改善方向）。
    self_check_ring 输入：c_t.detach() + delta_h（已 detach）+ know_gap.detach()
    self_check_loss：BCE(ring_score, sigmoid(improve_scalar * 5))
    DOD 目标：rank ≥ 3，能量分布不退步。
    """
    return LumaConfig(
        **_base_arch_kwargs(args),
        enable_world_jepa=False,
        world_jepa_weight=0.0,
        disable_self_jepa=False,
        self_jepa_weight=args.self_jepa_weight,
        self_rollout_weight=getattr(args, "self_rollout_weight", 0.0),
        self_rollout_weighting_mode=getattr(args, "self_rollout_weighting_mode", "legacy"),
        self_jepa_residual_reg=0.0,
        exit_aux_weight=0.0,
        rollout_zone_weight=getattr(args, "rollout_zone_weight", 0.0),
        routing_tier_entropy_weight=0.0,
        routing_min_local_share_weight=0.0,
        trajectory_vitality_weight=getattr(args, "trajectory_vitality_weight", 0.0),
        compression_dynamics_weight=0.0,
        # progress-shape
        self_progress_shape_weight=getattr(args, "self_progress_shape_weight", 0.0),
        enable_progress_exit_readout=bool(getattr(args, "enable_progress_exit_readout", 0)),
        enable_backtrack_aware_progress=bool(getattr(args, "enable_backtrack_aware_progress", 0)),
        # 局部几何
        self_local_delta_consistency_weight=getattr(args, "self_local_delta_consistency_weight", 0.0),
        self_local_curvature_weight=getattr(args, "self_local_curvature_weight", 0.0),
        # self_check_ring 启用
        enable_self_check_ring=True,
        self_check_loss_weight=args.self_check_loss_weight,
        # SIGreg（继承 Phase 3.5 的成功配置）
        enable_sigreg_delta=True,
        enable_sigreg_rollout=False,
        enable_sigreg_ct=bool(getattr(args, "enable_sigreg_ct", 0)),
        sigreg_ct_weight=getattr(args, "sigreg_ct_weight", 0.05),
        loop_sigreg_weight=getattr(args, "loop_sigreg_weight", 0.0),
        ct_injection_mode=getattr(args, "ct_injection_mode", "add"),
        jepa_predictor_dropout=getattr(args, "jepa_predictor_dropout", 0.0),
        cmda_token_wish=bool(getattr(args, "cmda_token_wish", 0)),
        ct_gated_attn=bool(getattr(args, "ct_gated_attn", 0)),
        ct_conditioned_lora=bool(getattr(args, "ct_conditioned_lora", 0)),
        ct_delta_inject=bool(getattr(args, "ct_delta_inject", 0)),
        ct_inject_scale=getattr(args, "ct_inject_scale", 1.0),
        ct_per_layer_inject=bool(getattr(args, "ct_per_layer_inject", 0)),
        delta_h_scale=getattr(args, "delta_h_scale", 0.0),
        delta_h_normalize=bool(getattr(args, "delta_h_normalize", 0)),
        cos_sigreg_weight=getattr(args, "cos_sigreg_weight", 0.0),
        ct_momentum=getattr(args, "ct_momentum", 0.0),
        freeze_ct_during_reason=bool(getattr(args, "freeze_ct_during_reason", 0)),
        sigreg_delta_weight=args.sigreg_delta_weight,
        sigreg_num_slices=128,
        sigreg_t_min=0.2,
        sigreg_t_max=4.0,
        sigreg_num_points=17,
        sigreg_lambda=1.0,
        sigreg_eps=1e-6,
    )


def build_phase5_config(args: argparse.Namespace) -> LumaConfig:
    """Phase 5: Phase 4 基础上加 c_t 双向梯度缩放（GradScale）。
    ct_grad_scale=0.2: c_t→backbone 和 c_t→辅助loss 的反向梯度缩放为 20%，
    防止辅助目标过度干扰主干、也防止主干梯度淹没自省信号。
    DOD 目标：rank ≥ 3，能量分布进一步均匀化。
    """
    return LumaConfig(
        **_base_arch_kwargs(args),
        enable_world_jepa=False,
        world_jepa_weight=0.0,
        disable_self_jepa=False,
        self_jepa_weight=args.self_jepa_weight,
        self_rollout_weight=0.0,
        self_jepa_residual_reg=0.0,
        exit_aux_weight=0.0,
        rollout_zone_weight=0.0,
        routing_tier_entropy_weight=0.0,
        routing_min_local_share_weight=0.0,
        trajectory_vitality_weight=0.0,
        compression_dynamics_weight=0.0,
        # self_check_ring（继承 Phase 4）
        enable_self_check_ring=True,
        self_check_loss_weight=args.self_check_loss_weight,
        # SIGreg（继承 Phase 3.5）
        enable_sigreg_delta=True,
        enable_sigreg_rollout=False,
        enable_sigreg_ct=False,
        sigreg_delta_weight=args.sigreg_delta_weight,
        sigreg_num_slices=128,
        sigreg_t_min=0.2,
        sigreg_t_max=4.0,
        sigreg_num_points=17,
        sigreg_lambda=1.0,
        sigreg_eps=1e-6,
        # Phase 5：c_t 控制
        ct_grad_scale=args.ct_grad_scale,
        ct_grad_scale_aux=args.ct_grad_scale_aux,
        ct_norm_penalty_weight=args.ct_norm_penalty_weight,
    )


def build_phase6_config(args: argparse.Namespace) -> LumaConfig:
    """Phase 6: Phase 4 + World-JEPA (LeWM or EMA)。
    世界模型预测：通过 masked latent prediction 学习时序结构。
    LeWM 模式（world_jepa_mode='full'）: 单编码器 + SIGreg 防坍缩，省 VRAM。
    EMA 模式（world_jepa_mode='scaffold'）: 双编码器 + EMA target，更稳定。
    """
    return LumaConfig(
        **_base_arch_kwargs(args),
        # ── World-JEPA 启用 ──────────────────────────────────────────────────
        enable_world_jepa=True,
        world_jepa_weight=args.world_jepa_weight,
        world_sigreg_weight=args.world_sigreg_weight,
        world_jepa_reason_only=bool(args.world_jepa_reason_only),
        enable_ct_world_jepa=bool(args.enable_ct_world_jepa),
        ct_world_jepa_weight=args.ct_world_jepa_weight,
        # ── self-JEPA（继承 Phase 4）──────────────────────────────────────────
        disable_self_jepa=False,
        self_jepa_weight=args.self_jepa_weight,
        self_rollout_weight=args.self_rollout_weight,
        self_progress_shape_weight=getattr(args, "self_progress_shape_weight", 0.0),
        # ── self_check_ring（继承 Phase 4）────────────────────────────────────
        enable_self_check_ring=True,
        self_check_loss_weight=args.self_check_loss_weight,
        # ── SIGreg（继承 Phase 3.5 + 新增 ct/rollout 可选）────────────────────
        enable_sigreg_delta=True,
        enable_sigreg_rollout=bool(getattr(args, "enable_sigreg_rollout", 0)),
        enable_sigreg_ct=bool(getattr(args, "enable_sigreg_ct", 0)),
        sigreg_delta_weight=args.sigreg_delta_weight,
        sigreg_ct_weight=getattr(args, "sigreg_ct_weight", 0.05),
        loop_sigreg_weight=getattr(args, "loop_sigreg_weight", 0.0),
        ct_injection_mode=getattr(args, "ct_injection_mode", "add"),
        jepa_predictor_dropout=getattr(args, "jepa_predictor_dropout", 0.0),
        cmda_token_wish=bool(getattr(args, "cmda_token_wish", 0)),
        ct_gated_attn=bool(getattr(args, "ct_gated_attn", 0)),
        ct_conditioned_lora=bool(getattr(args, "ct_conditioned_lora", 0)),
        ct_delta_inject=bool(getattr(args, "ct_delta_inject", 0)),
        ct_inject_scale=getattr(args, "ct_inject_scale", 1.0),
        ct_per_layer_inject=bool(getattr(args, "ct_per_layer_inject", 0)),
        delta_h_scale=getattr(args, "delta_h_scale", 0.0),
        delta_h_normalize=bool(getattr(args, "delta_h_normalize", 0)),
        cos_sigreg_weight=getattr(args, "cos_sigreg_weight", 0.0),
        ct_momentum=getattr(args, "ct_momentum", 0.0),
        freeze_ct_during_reason=bool(getattr(args, "freeze_ct_during_reason", 0)),
        sigreg_rollout_weight=getattr(args, "sigreg_rollout_weight", 0.05),
        sigreg_num_slices=128,
        sigreg_t_min=0.2,
        sigreg_t_max=4.0,
        sigreg_num_points=17,
        sigreg_lambda=1.0,
        sigreg_eps=1e-6,
        # ── 其他辅助 loss 关闭 ────────────────────────────────────────────────
        self_jepa_residual_reg=0.0,
        exit_aux_weight=getattr(args, "exit_aux_weight", 0.0),
        rollout_zone_weight=0.0,
        routing_tier_entropy_weight=0.0,
        routing_min_local_share_weight=0.0,
        trajectory_vitality_weight=0.0,
        compression_dynamics_weight=0.0,
    )


# ---------------------------------------------------------------------------
# Gradient norm monitoring
# ---------------------------------------------------------------------------

def _module_grad_norm(params) -> float:
    """计算一组参数梯度的 L2 范数，强制转 float32 避免 bf16 下溢。"""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total)


def compute_grad_metrics(model: LumaForCausalLM) -> dict:
    """
    三组参数的梯度范数 + v2 逐层范数。

    v1 zone-level: compress / shared / reasoning（向后兼容）
    v2 per-layer: 每层一个范数，用于高维 POD 分析
    """
    backbone = model.model  # LumaBackbone

    # ── v1: zone-level ──
    compress_params = list(backbone.embedding.parameters()) + list(backbone.compression.parameters())
    shared_params = list(backbone.reason_core.parameters())
    reasoning_params = (
        list(backbone.reason_memory.parameters())
        + list(backbone.mhc.parameters())
        + list(backbone.unified_attnres.parameters())
        + list(backbone.introspection_state_stream.parameters())
        + list(backbone.self_jepa_residual_predictor.parameters())
        + list(backbone.world_latent_jepa.parameters())
        + list(backbone.exit_controller.parameters())
        + list(backbone.loop_norm.parameters())
        + list(backbone.final_norm.parameters())
    )

    n_compress = _module_grad_norm(compress_params)
    n_shared = _module_grad_norm(shared_params)
    n_reasoning = _module_grad_norm(reasoning_params)

    norms = [v for v in [n_compress, n_shared, n_reasoning] if v > 0]
    ratio = max(norms) / min(norms) if len(norms) >= 2 else float("inf")

    # ── v2: per-layer ──
    layer_norms: dict[str, float] = {}
    layer_norms["embedding"] = _module_grad_norm(list(backbone.embedding.parameters()))
    for i, layer in enumerate(backbone.compression.layers):
        layer_norms[f"compress_{i:02d}"] = _module_grad_norm(list(layer.parameters()))
    for i, layer in enumerate(backbone.reason_core.shared_layers):
        layer_norms[f"reason_shared_{i}"] = _module_grad_norm(list(layer.parameters()))
    layer_norms["mhc"] = _module_grad_norm(list(backbone.mhc.parameters()))
    layer_norms["introspection"] = _module_grad_norm(list(backbone.introspection_state_stream.parameters()))
    layer_norms["self_jepa"] = _module_grad_norm(list(backbone.self_jepa_residual_predictor.parameters()))
    layer_norms["world_jepa"] = _module_grad_norm(list(backbone.world_latent_jepa.parameters()))
    layer_norms["exit_ctrl"] = _module_grad_norm(list(backbone.exit_controller.parameters()))

    return {
        "grad_norm_compress": n_compress,
        "grad_norm_shared": n_shared,
        "grad_norm_reasoning": n_reasoning,
        "grad_ratio": ratio,
        "layer_grad_norms": layer_norms,
    }


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------

def log_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _to_python_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        if value.numel() == 1:
            return float(value.detach().float().item())
        return value.detach().float().mean().item()
    if isinstance(value, (int, float, bool)):
        return value
    return float(value)


def _mean_or_default(values, default: float = 0.0) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def _find_named_param(named_params: dict[str, torch.nn.Parameter], suffix: str) -> torch.nn.Parameter | None:
    for name, param in named_params.items():
        if name.endswith(suffix):
            return param
    return None


@torch.no_grad()
def compute_wc_spectrum(raw_model: LumaForCausalLM) -> dict[str, float | list[float]]:
    named_params = dict(raw_model.named_parameters())
    wc = _find_named_param(named_params, "reason_core.ct_injection.proj.weight")
    if wc is None:
        return {"wc_sv_top3": [], "wc_sv_top1": 0.0, "wc_cond": 0.0}
    sv = torch.linalg.svdvals(wc.detach().float())
    top3 = sv[:3].tolist()
    cond = float((sv[0] / sv[-1].clamp(min=1e-8)).item()) if sv.numel() > 1 else 1.0
    return {
        "wc_sv_top3": [float(x) for x in top3],
        "wc_sv_top1": float(top3[0]) if top3 else 0.0,
        "wc_cond": cond,
    }


def compute_gradient_source_split(
    raw_model: LumaForCausalLM,
    lm_loss: torch.Tensor,
    aux: dict,
    enabled: bool,
    allow_multi_backward: bool = False,
) -> dict[str, float]:
    """梯度源分解 probe。

    注意：torch.autograd.grad(inputs=...) 和 use_reentrant=True 的 gradient checkpointing
    不兼容。训练主流程开启 checkpointing 时，allow_multi_backward 必须为 False，
    此时 probe 返回 0.0 占位，实际分解只能在单独的理论诊断运行里做
    （关闭 checkpointing 或改用 use_reentrant=False）。
    """
    if not enabled or not allow_multi_backward:
        return {
            "grad_lm_to_wc": 0.0,
            "grad_hmask_to_ct_head": 0.0,
            "grad_selfjepa_to_hebb": 0.0,
            "grad_probe_enabled": 0.0,
        }
    named_params = dict(raw_model.named_parameters())
    wc = _find_named_param(named_params, "reason_core.ct_injection.proj.weight")
    c_t_head = _find_named_param(named_params, "introspection_state_stream.c_t_head.weight")
    hebb = _find_named_param(named_params, "neuromod_ct_writer.hebb_out.weight")

    def _grad_norm(loss_term: torch.Tensor | None, param: torch.nn.Parameter | None) -> float:
        if loss_term is None or param is None:
            return 0.0
        if not isinstance(loss_term, torch.Tensor) or not loss_term.requires_grad:
            return 0.0
        grad = torch.autograd.grad(loss_term, param, retain_graph=True, allow_unused=True)[0]
        return float(grad.detach().float().norm().item()) if grad is not None else 0.0

    return {
        "grad_lm_to_wc": _grad_norm(lm_loss, wc),
        "grad_hmask_to_ct_head": _grad_norm(aux.get("h_mask_loss"), c_t_head),
        "grad_selfjepa_to_hebb": _grad_norm(aux.get("self_jepa_loss"), hebb),
        "grad_probe_enabled": 1.0,
    }


@torch.no_grad()
def compute_param_grad_totals(raw_model: LumaForCausalLM) -> dict[str, float]:
    """Backward 后直接从 .grad 读关键参数的总梯度范数，不触发额外 autograd.grad。
    这是 gradient source split 在 checkpointing 开启时的降级版本——只给出 "lm 主路径" 的总梯度，
    不能分解出每个 loss 的贡献，但足够用于监控增长趋势。
    """
    named_params = dict(raw_model.named_parameters())
    targets = {
        "grad_total_wc": _find_named_param(named_params, "reason_core.ct_injection.proj.weight"),
        "grad_total_c_t_head": _find_named_param(named_params, "introspection_state_stream.c_t_head.weight"),
        "grad_total_hebb_out": _find_named_param(named_params, "neuromod_ct_writer.hebb_out.weight"),
    }
    out: dict[str, float] = {}
    for key, param in targets.items():
        if param is None or param.grad is None:
            out[key] = 0.0
        else:
            out[key] = float(param.grad.detach().float().norm().item())
    return out


def should_log_dynamics_detail(step: int, args, burst_remaining: int) -> bool:
    if burst_remaining > 0:
        return True
    if step <= 256:
        return step % args.dynamics_log_dense_interval == 0
    if step <= 2048:
        return step % args.dynamics_log_mid_interval == 0
    return step % args.dynamics_log_sparse_interval == 0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _infinite_loader(loader: DataLoader):
    """无限循环 DataLoader，确保可以跑任意步数。"""
    while True:
        yield from loader


def train(args, luma_config: LumaConfig, model: LumaForCausalLM,
          loader: DataLoader, optimizer, scheduler, scaler,
          autocast_ctx, metrics_path: Path,
          compress_probe: torch.nn.Module = None,
          compress_weight: float = 0.0,
          start_step: int = 0):

    model.train()
    start_time = time.time()
    step = start_step
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    save_interval = getattr(args, "save_interval", 0)
    ckpt_keep = getattr(args, "ckpt_keep", 3)

    # ── 动力学分析器 ─────────────────────────────────────────────────────
    grad_tracker = GradTrajectoryTracker(window=min(args.dod_interval, 200))
    layer_grad_tracker = LayerGradTracker(window=min(args.dod_interval, 200))
    ct_tracker = CtStateTracker(window=100, proj_dim=64)
    exit_depth_tracker = ExitDepthTracker(window=min(args.dod_interval, 200),
                                          max_loops=getattr(args, "reason_loops", 15))
    grad_snapshots = []   # 每 dod_interval 步存一份 analyze() 结果
    ct_snapshots = []
    layer_grad_snapshots = []
    exit_depth_snapshots = []
    dyn_report_path = metrics_path.parent / metrics_path.name.replace("_metrics.jsonl", "_dynamics.json")
    dyn_md_path = dyn_report_path.with_suffix(".md")
    dynamics_path = ARTIFACTS_DIR.parent / "dynamics" / f"{args.save_weight}_phase{args.phase}.jsonl"
    burst_remaining = 0
    prev_wc_cond = None
    prev_grad_probe = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── 动力学监控：区间峰值 loops、loss EMA、步计时 ──
    interval_max_loops = 0       # 改进 #1: log_interval 内的最大循环数
    loss_ema = None              # 改进 #2: loss 滑动平均
    step_start_time = time.time()  # 改进 #6: 每步计时

    for input_ids, labels in _infinite_loader(loader):
        step += 1
        if step > args.iters:
            break
        step_start_time = time.time()  # 改进 #6: 每步开始计时
        model.model.exit_controller._global_step = step

        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        detail_requested = should_log_dynamics_detail(step, args, burst_remaining)
        theory_probe_requested = detail_requested or (args.theory_probe_interval > 0 and step % args.theory_probe_interval == 0)

        with autocast_ctx:
            # theory_probe_requested 时让 model 在 loop_idx==0 跑一次 rho_h_frozen/rho_c_drift/eta_moving_fp 测量
            res = model(input_ids, labels=labels, measure_theory_probes=theory_probe_requested)
            raw_model = getattr(model, "_orig_mod", model)
            aux_snapshot = getattr(raw_model, "last_aux", {})
            # Phase 3+: res.loss 已包含 self_jepa_term（模型内部加权）
            # res.aux_loss = self_jepa_weight * self_jepa_loss（+ 其他目前为 0 的项）
            loss_jepa_val = res.aux_loss.item() if res.aux_loss is not None else 0.0
            aux_loss = res.aux_loss if res.aux_loss is not None else res.loss.new_zeros(())

            # ── Rho-1 Selective Loss: 只对高信息量 token 计算 loss ──────────
            _sel_ratio = getattr(args, "selective_loss_ratio", 1.0)
            if _sel_ratio < 1.0:
                # 从 logits 重算 per-token loss，取 top-k% 最高 loss token
                # logits 可能比 labels 长（reason_memory token），截取对齐
                _raw_logits = res.logits
                _label_len = labels.size(-1)
                if _raw_logits.size(-2) > _label_len:
                    _raw_logits = _raw_logits[:, -_label_len:, :]
                _logits = _raw_logits[..., :-1, :].contiguous()
                _labels = labels[..., 1:].contiguous()
                _per_tok = F.cross_entropy(
                    _logits.view(-1, _logits.size(-1)),
                    _labels.view(-1),
                    ignore_index=-100, reduction="none",
                )
                _valid = (_labels.view(-1) != -100)
                _valid_losses = _per_tok[_valid]
                _k = max(1, int(_valid_losses.numel() * _sel_ratio))
                _topk, _ = _valid_losses.topk(_k)
                loss_lm = _topk.mean()
                # 替换 total loss = selective_lm + aux
                _total_lm = loss_lm + aux_loss
            else:
                loss_lm = res.loss - aux_loss
                _total_lm = res.loss

            # ── Phase 2+: 压缩区辅助 loss ──────────────────────────────────
            loss_compress_val = 0.0
            if compress_probe is not None and compress_weight > 0:
                raw = getattr(model, "_orig_mod", model)
                h_compress = raw.last_aux["compression_h"]
                logits_c = compress_probe(h_compress)
                x_c = logits_c[:, :-1, :].contiguous()
                y_c = labels[:, 1:].contiguous()
                loss_compress = F.cross_entropy(
                    x_c.view(-1, x_c.size(-1)), y_c.view(-1), ignore_index=-100
                )
                loss = (_total_lm + compress_weight * loss_compress) / args.accumulation_steps
                loss_compress_val = loss_compress.item()
            else:
                loss = _total_lm / args.accumulation_steps

        wc_spectrum = compute_wc_spectrum(raw_model) if theory_probe_requested else {"wc_sv_top3": [], "wc_sv_top1": 0.0, "wc_cond": 0.0}
        # grad_source_split 需要多次 backward，与 gradient checkpointing (use_reentrant=True) 不兼容。
        # 开启 checkpointing 时降级为占位 0.0，只能在专门的理论诊断 run 里关闭 checkpointing 才能激活。
        _allow_multi_backward = theory_probe_requested and not bool(getattr(args, "use_gradient_checkpointing", 0))
        grad_source_split = compute_gradient_source_split(
            raw_model, loss_lm, aux_snapshot, theory_probe_requested,
            allow_multi_backward=_allow_multi_backward,
        )
        scaler.scale(loss).backward()

        # NaN 探针：检查梯度是否有 NaN
        for _pname, _pparam in model.named_parameters():
            if _pparam.grad is not None and not torch.isfinite(_pparam.grad).all():
                _nan_g = (~torch.isfinite(_pparam.grad)).sum().item()
                print(f"[NaN PROBE grad] step={step} {_pname} has {_nan_g} non-finite grads, grad_norm={_pparam.grad.norm().item()}", flush=True)
                break  # 只报第一个

        # ── 梯度监控（backward 后、step 前）────────────────────────────────
        do_grad_log = (step % args.grad_log_interval == 0)
        # Always unscale so grad_metrics can be computed for tracker every step
        scaler.unscale_(optimizer.matrix_optimizer)
        scaler.unscale_(optimizer.scalar_optimizer)
        grad_metrics = compute_grad_metrics(raw_model)
        # grad_totals: backward 后直接读 .grad 的总范数，不受 checkpointing 限制
        if theory_probe_requested:
            grad_totals = compute_param_grad_totals(raw_model)
        else:
            grad_totals = {"grad_total_wc": 0.0, "grad_total_c_t_head": 0.0, "grad_total_hebb_out": 0.0}

        if step % args.accumulation_steps == 0:
            all_params = list(model.parameters())
            if compress_probe is not None:
                all_params += list(compress_probe.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            scaler.update()
            # World-JEPA EMA target update (no-op for LeWM mode)
            if hasattr(raw_model, "update_world_target"):
                raw_model.update_world_target()
            optimizer.zero_grad(set_to_none=True)
            if compress_probe is not None:
                compress_probe.zero_grad(set_to_none=True)
            # Release fragmented CUDA cache to prevent OOM on long sequences
            torch.cuda.empty_cache()

        # ── 日志 ────────────────────────────────────────────────────────────
        current_loss = loss.item() * args.accumulation_steps
        scalar_lr = next(g["lr"] for g in optimizer.param_groups if g.get("optim_family") == "adamw")
        matrix_lr = next(g["lr"] for g in optimizer.param_groups if g.get("optim_family") == "muon")

        # 改进 #5: NaN watchdog — loss 异常时立即停止训练
        if not math.isfinite(current_loss):
            Logger(f"⚠ NaN/Inf detected at step {step}, loss={current_loss}, stopping training")
            break

        # 改进 #2: loss 滑动平均 (EMA, alpha=0.01)
        loss_ema = 0.99 * loss_ema + 0.01 * current_loss if loss_ema is not None else current_loss

        # ── Exit depth tracking ──────────────────────────────────────────
        _last_aux = getattr(getattr(model, "_orig_mod", model), "last_aux", {})
        _exit_loops = _last_aux.get("executed_loops", 0)

        # 改进 #1: 区间峰值 loops
        interval_max_loops = max(interval_max_loops, _exit_loops)

        # 改进 #6: 训练速度（tok/s）
        _step_elapsed = max(time.time() - step_start_time, 1e-6)
        _tok_per_sec = int(args.batch_size * args.accumulation_steps * args.max_seq_len / _step_elapsed)

        record = {
            "step": step,
            "loss_lm": loss_lm.item(),
            "loss_compress": loss_compress_val,
            "loss_jepa": loss_jepa_val,
            "loss_total": current_loss,
            "scalar_lr": scalar_lr,
            "matrix_lr": matrix_lr,
            "exit_loops": _exit_loops,
            "selective_ratio": _sel_ratio,
            "elapsed_s": round(time.time() - start_time, 1),
            **grad_metrics,
        }
        log_jsonl(metrics_path, record)

        dynamics_trigger = False
        alpha_true = _to_python_scalar(aux_snapshot.get("ct_inject_ratio", 0.0)) or 0.0
        if alpha_true >= 0.04:  # α_crit ≈ 0.045，预警线
            dynamics_trigger = True
        if prev_wc_cond is not None and wc_spectrum["wc_cond"] > max(1.0, prev_wc_cond * 1.5):
            dynamics_trigger = True
        # burst trigger 用 grad_totals（backward 后真实读出的总梯度），
        # 不受 gradient checkpointing 限制；grad_source_split 只在专门诊断 run 里有值。
        current_grad_probe = (
            grad_totals.get("grad_total_wc", 0.0),
            grad_totals.get("grad_total_c_t_head", 0.0),
            grad_totals.get("grad_total_hebb_out", 0.0),
        )
        if prev_grad_probe is not None:
            for now, prev in zip(current_grad_probe, prev_grad_probe):
                if prev > 0 and now > 2.0 * prev:
                    dynamics_trigger = True
                    break
        if dynamics_trigger:
            burst_remaining = max(burst_remaining, args.dynamics_burst_len)

        if args.dynamics_jsonl and detail_requested:
            dynamics_mod_summary = aux_snapshot.get("dynamics_modulation_summary", {})
            layer_grad_norms = grad_metrics.get("layer_grad_norms", {})
            detail_record = {
                "step": step,
                "seed": 42,
                "phase": args.phase,
                "loop_idx": int(aux_snapshot.get("executed_loops", 0)) - 1,
                "lm_loss": loss_lm.item(),
                "aux_loss": _to_python_scalar(aux_loss),
                "h_mask_loss": _to_python_scalar(aux_snapshot.get("h_mask_loss")),
                "world_jepa_loss": _to_python_scalar(aux_snapshot.get("world_jepa_loss")),
                "self_jepa_loss": _to_python_scalar(aux_snapshot.get("self_jepa_loss")),
                # ct_inj_pre: 注入前的相对强度 = ||proj(c_t)|| / ||h||。
                "ct_inj_pre": _to_python_scalar(aux_snapshot.get("ct_inject_ratio_pre")),
                # alpha_true: 实际注入到 h 的相对强度 = ||applied_ct_bias|| / ||h||。
                "alpha_true": alpha_true,
                # rho_h_frozen: 冻结 c_t 和 loop_idx，扰动 h 测 F_k 局部雅可比谱半径
                # rho_c_drift: 冻结 h 和 loop_idx，测 F_k 对 c_t 的敏感度
                # eta_moving_fp: c_t 变化对 F 的贡献 / h 扰动对 F 的贡献（moving fixed point 强度）
                "rho_h_frozen": (aux_snapshot.get("theory_probes") or {}).get("rho_h_frozen"),
                "rho_c_drift": (aux_snapshot.get("theory_probes") or {}).get("rho_c_drift"),
                "eta_moving_fp": (aux_snapshot.get("theory_probes") or {}).get("eta_moving_fp"),
                # probe_delta_*_norm: probe 时实际用的扰动 L2 范数，调试用
                "probe_delta_h_norm": (aux_snapshot.get("theory_probes") or {}).get("probe_delta_h_norm"),
                "probe_delta_c_norm": (aux_snapshot.get("theory_probes") or {}).get("probe_delta_c_norm"),
                "ct_norm_raw": _mean_or_default(aux_snapshot.get("ct_norm_raw_history", [])),
                "ct_norm_after_writer": _mean_or_default(aux_snapshot.get("ct_norm_after_writer_history", [])),
                "meta_last_norm": _mean_or_default(aux_snapshot.get("meta_last_norm_history", [])),
                "c_t_head_out_norm": _mean_or_default(aux_snapshot.get("c_t_head_out_norm_history", [])),
                "wc_sv_top1": wc_spectrum["wc_sv_top1"],
                "wc_sv_top3": wc_spectrum["wc_sv_top3"],
                "wc_cond": wc_spectrum["wc_cond"],
                "hebb_write_norm": _mean_or_default(aux_snapshot.get("nm_hebb_write_history", [])),
                "surprise_mean": _mean_or_default(aux_snapshot.get("nm_surprise_history", [])),
                # grad_probe_enabled: 0=降级占位（checkpointing 开）, 1=真实分解
                "grad_probe_enabled": grad_source_split.get("grad_probe_enabled", 0.0),
                "grad_lm_to_wc": grad_source_split["grad_lm_to_wc"],
                "grad_hmask_to_ct_head": grad_source_split["grad_hmask_to_ct_head"],
                "grad_selfjepa_to_hebb": grad_source_split["grad_selfjepa_to_hebb"],
                # grad_total_*: backward 后直接读 .grad 的总范数，checkpointing 兼容的降级版
                "grad_total_wc": grad_totals.get("grad_total_wc", 0.0),
                "grad_total_c_t_head": grad_totals.get("grad_total_c_t_head", 0.0),
                "grad_total_hebb_out": grad_totals.get("grad_total_hebb_out", 0.0),
                "grad_shared_0": layer_grad_norms.get("reason_shared_0", 0.0),
                "grad_shared_last": layer_grad_norms.get(f"reason_shared_{max(args.reason_shared_depth - 1, 0)}", 0.0),
                "loop_lora_delta_ratio_mean": _to_python_scalar(dynamics_mod_summary.get("loop_lora_delta_ratio_mean", 0.0)),
                "loop_lora_delta_norm_mean": _to_python_scalar(dynamics_mod_summary.get("loop_lora_delta_norm_mean", 0.0)),
                "ct_perp": _to_python_scalar((aux_snapshot.get("ct_delta_perp") or [0.0])[0]),
                "l_est": (
                    aux_snapshot["per_loop_delta_h"][1] / max(aux_snapshot["per_loop_delta_h"][0], 1e-8)
                    if len(aux_snapshot.get("per_loop_delta_h", [])) >= 2 else 0.0
                ),
            }
            log_jsonl(dynamics_path, detail_record)

        prev_wc_cond = wc_spectrum["wc_cond"] if wc_spectrum["wc_cond"] else prev_wc_cond
        prev_grad_probe = current_grad_probe
        if burst_remaining > 0:
            burst_remaining -= 1

        if step % args.log_interval == 0 or step == args.iters:
            spend = time.time() - start_time
            eta = spend / step * (args.iters - step) / 60
            grad_line = (
                f"  grad: compress={grad_metrics.get('grad_norm_compress', 'n/a'):.3e}"
                f"  shared={grad_metrics.get('grad_norm_shared', 'n/a'):.3e}"
                f"  reasoning={grad_metrics.get('grad_norm_reasoning', 'n/a'):.3e}"
                f"  ratio={grad_metrics.get('grad_ratio', 'n/a'):.2f}"
                if grad_metrics else ""
            )
            # ── Loss 拆细（effective = 已加权；raw = 未加权组件）────────────
            _compress_s = f"  loss_c={loss_compress_val:.3f}" if compress_weight > 0 else ""
            # World-JEPA 拆解：cosine + sigreg
            _world_jepa_eff = _last_aux.get("world_jepa_loss_effective")
            _world_jepa_val = _world_jepa_eff.item() if _world_jepa_eff is not None and hasattr(_world_jepa_eff, "item") else (float(_world_jepa_eff) if _world_jepa_eff is not None else 0.0)
            _world_cos_raw = _last_aux.get("world_jepa_cosine", torch.tensor(0.0))
            _world_cos_val = _world_cos_raw.item() if hasattr(_world_cos_raw, "item") else float(_world_cos_raw)
            _world_sigreg_raw = _last_aux.get("world_sigreg_raw", torch.tensor(0.0))
            _world_sigreg_val = _world_sigreg_raw.item() if hasattr(_world_sigreg_raw, "item") else float(_world_sigreg_raw)
            # self-JEPA 已加权
            _self_jepa_eff = _last_aux.get("self_jepa_loss_effective")
            _self_jepa_val = _self_jepa_eff.item() if _self_jepa_eff is not None and hasattr(_self_jepa_eff, "item") else (float(_self_jepa_eff) if _self_jepa_eff is not None else 0.0)
            _h_mask_val = _last_aux.get("h_mask_loss", torch.tensor(0.0))
            _h_mask_val = _h_mask_val.item() if hasattr(_h_mask_val, "item") else float(_h_mask_val)
            _self_check_eff = _last_aux.get("self_check_loss_effective")
            _self_check_val = _self_check_eff.item() if _self_check_eff is not None and hasattr(_self_check_eff, "item") else (float(_self_check_eff) if _self_check_eff is not None else 0.0)
            # aux_loss 合计（loss_j 的原始含义）
            _aux_total = f"  loss_aux={loss_jepa_val:.3f}" if loss_jepa_val > 0 else ""

            exit_line = f"  loops={_exit_loops}/{args.reason_loops}  peak={interval_max_loops}" if _exit_loops > 0 else ""
            ema_line = f"  ema={loss_ema:.3f}" if loss_ema is not None else ""
            speed_line = f"  tok/s={_tok_per_sec}"
            # 多行输出，方便人工 review
            Logger(f"[{step}/{args.iters}] loss_lm={loss_lm.item():.3f}{_compress_s}{_aux_total}  scalar_lr={scalar_lr:.2e}  eta={eta:.1f}min{exit_line}{ema_line}{speed_line}")
            # loss 拆细子行：world-JEPA 和 self-JEPA
            _loss_parts = []
            if _world_jepa_val > 0:
                _loss_parts.append(f"w={_world_jepa_val:.3f}(cos={_world_cos_val:.3f} sig_raw={_world_sigreg_val:.1f})")
            if _self_jepa_val > 0:
                _loss_parts.append(f"sj={_self_jepa_val:.4f}")
            if _self_check_val > 0:
                _loss_parts.append(f"sc={_self_check_val:.4f}")
            if _h_mask_val > 0:
                _loss_parts.append(f"hm={_h_mask_val:.4f}")
            if _loss_parts:
                Logger(f"  losses: {'  '.join(_loss_parts)}")
            # 梯度子行
            if grad_metrics:
                Logger(grad_line)
            # 改进 #1: 每次打印后重置 interval_max_loops
            interval_max_loops = 0
            # ── 动力学监控摘要 ──
            _aux = _last_aux
            if _aux.get("per_loop_delta_h"):
                _dh_raw = _aux["per_loop_delta_h"]
                _dh = [f"{x:.3f}" for x in _dh_raw]
                _ct = [f"{x:.3f}" for x in _aux.get("per_loop_ct_change", [])]
                _je = [f"{x:.3f}" for x in _aux.get("per_loop_jepa_err", [])]
                # L_est: 旧口径 proxy，时变系统里只表示相邻两步 δh 的相对变化，不等于真实收缩率。
                _L_est = _dh_raw[1] / max(_dh_raw[0], 1e-8) if len(_dh_raw) >= 2 else 0.0
                # ct_perp: c_t 变化方向和 c_t 自身的垂直度 (1=垂直/新方向, 0=平行/同方向累积)
                _ct_perp = _aux.get("ct_delta_perp", [0.0])[0] if _aux.get("ct_delta_perp") else 0.0
                _ct_inj_pre = _aux.get("ct_inject_ratio_pre", 0.0)
                _ct_inj_r = _aux.get("ct_inject_ratio", 0.0)
                _lora_ratio = _to_python_scalar(_aux.get("dynamics_modulation_summary", {}).get("loop_lora_delta_ratio_mean", 0.0)) or 0.0
                Logger(f"  loops detail: dh={_dh}  ct={_ct}  jepa={_je}  L_est={_L_est:.3f}  ct_perp={_ct_perp:.3f}  ct_inj_pre={_ct_inj_pre:.3f}  alpha={_ct_inj_r:.3f}  lora_ratio={_lora_ratio:.3f}")
            if _aux.get("nm_gain_history"):
                _g = [f"{x:.2f}" for x in _aux["nm_gain_history"]]
                _h = [f"{x:.4f}" for x in _aux.get("nm_hebb_norm_history", [])]
                _w = [f"{x:.4f}" for x in _aux.get("nm_hebb_write_history", [])]
                Logger(f"  hebb: gain={_g}  norm={_h}  write={_w}")
            if _aux.get("ct_cosine_trajectory"):
                _cos = [f"{x:.3f}" for x in _aux["ct_cosine_trajectory"]]
                Logger(f"  ct_traj: cos={_cos}")
            if _aux.get("per_loop_dt_inject") and any(x > 0 for x in _aux["per_loop_dt_inject"]):
                _di = [f"{x:.4f}" for x in _aux["per_loop_dt_inject"]]
                Logger(f"  dt_ratio: {_di}")  # <0.01 无效, 0.01-0.1 轻度, >0.1 强, >1.0 危险
            _fp = _aux.get("fixed_point_analysis", {})
            if _fp:
                _Lg = _fp.get("L_global", -1)
                _Ld = _fp.get("L_per_dir_top4", [])
                _sd = _fp.get("slow_directions", 0)
                _dd = _fp.get("dead_directions", 0)
                Logger(f"  fp_proxy: L={_Lg:.3f}  dirs={_Ld}  slow={_sd}  dead={_dd}")
            _snorms = _aux.get("step_norms", [])
            _sangles = _aux.get("step_angles", [])
            _accels = _aux.get("accel_norms", [])
            if _snorms and len(_snorms) >= 2:
                Logger(f"  dynamics: step_norm={_snorms[0]:.3f}→{_snorms[-1]:.3f} decay={_snorms[-1]/_snorms[0]:.3f}"
                       f"  angle={'→'.join(f'{a:.2f}' for a in _sangles[:3])}"
                       f"  accel={'→'.join(f'{a:.2f}' for a in _accels[:3])}")
            if _aux.get("loss_head") is not None:
                _lh = _aux["loss_head"].item() if hasattr(_aux["loss_head"], "item") else _aux["loss_head"]
                _lm = _aux["loss_mid"].item() if hasattr(_aux.get("loss_mid", 0), "item") else _aux.get("loss_mid", 0)
                _lt = _aux["loss_tail"].item() if hasattr(_aux.get("loss_tail", 0), "item") else _aux.get("loss_tail", 0)
                Logger(f"  loss_pos: head={_lh:.4f}  mid={_lm:.4f}  tail={_lt:.4f}")

        # ── 梯度方向诊断 ────────────────────────────────────────────────────
        if do_grad_log and grad_metrics:
            compress_norm = grad_metrics["grad_norm_compress"]
            if compress_norm < 1e-7:
                Logger(
                    f"WARNING [step {step}] grad_norm_compress={compress_norm:.2e} ~= 0"
                    "  -> 梯度无法传到压缩区，需执行预案 S1（跳连捷径）"
                )

        # ── Checkpoint 保存 ─────────────────────────────────────────────────
        if save_interval > 0 and (step % save_interval == 0 or step == args.iters):
            ckpt_path = ckpt_dir / f"phase{args.phase}_step{step}.pt"
            save_checkpoint(ckpt_path, step, model, optimizer, scheduler, args,
                            compress_probe=compress_probe)
            cleanup_checkpoints(ckpt_dir, keep=ckpt_keep, phase=args.phase)

        # ── 动力学追踪（每步更新，每 dod_interval 步分析）──────────────────
        grad_tracker.update(step, grad_metrics)
        layer_norms = grad_metrics.get("layer_grad_norms")
        if layer_norms:
            layer_grad_tracker.update(step, layer_norms)
        raw_m = getattr(model, "_orig_mod", model)
        if hasattr(raw_m, "last_aux") and raw_m.last_aux:
            ct_tensor = raw_m.last_aux.get("c_t")
            if ct_tensor is not None and ct_tensor.dim() >= 2:
                ct_tracker.update(step, ct_tensor)
            exit_loops = raw_m.last_aux.get("exit_loops")
            if exit_loops is not None:
                exit_depth_tracker.update(step, exit_loops)

        if args.dod_interval > 0 and (step % args.dod_interval == 0 or step == args.iters):
            snap_grad = grad_tracker.analyze()
            snap_ct = ct_tracker.analyze()
            snap_layer = layer_grad_tracker.analyze()
            snap_exit = exit_depth_tracker.analyze()
            if snap_grad:
                grad_snapshots.append(snap_grad)
                ct_snapshots.append(snap_ct)
                if snap_layer:
                    layer_grad_snapshots.append(snap_layer)
                if snap_exit:
                    exit_depth_snapshots.append(snap_exit)
                # v1 日志行（兼容）
                dod_rank = snap_grad.get("dod_rank", -1)
                e1 = snap_grad.get("energy_mode1_pct", 100.0)
                radius = snap_grad.get("dmd_spectral_radius", float("nan"))
                dmd_str = f"{radius:.4f}" if math.isfinite(radius) else "nan"
                # v2 逐层信息
                v2_rank = snap_layer.get("dod_rank", -1) if snap_layer else -1
                v2_e1 = snap_layer.get("energy_mode1_pct", 100.0) if snap_layer else 100.0
                v2_dims = snap_layer.get("n_layers", 0) if snap_layer else 0
                dead = snap_layer.get("dead_layers", []) if snap_layer else []
                # 退出深度
                exit_info = ""
                if snap_exit:
                    exit_info = (f"  exit: mean={snap_exit['mean_depth']:.1f}"
                                 f" std={snap_exit['std_depth']:.2f}"
                                 f" entropy={snap_exit['depth_entropy']:.3f}")
                Logger(
                    f"[DOD step {step}]"
                    f"  rank={v2_rank}/{v2_dims} mode1={v2_e1:.1f}%"
                    f"  dmd_radius={dmd_str}"
                    + (f"  dead={dead}" if dead else "")
                    + exit_info
                )
                # h_diversity: 跨循环 h 的多样性（DOD 时打印）
                _hd = _last_aux.get("h_diversity_across_loops", 0) if _last_aux else 0
                Logger(f"  h_diversity={_hd:.4f}")
                _md = _last_aux.get("mamba_diag", {}) if _last_aux else {}
                if _md.get("layer1_cos"):
                    Logger(f"  mamba_diag: L1_cos={[f'{x:.3f}' for x in _md['layer1_cos']]}  L2_cos={[f'{x:.3f}' for x in _md['layer2_cos']]}")
                # 快照写入 jsonl
                snapshot_record = {"step": step, "dynamics_snapshot": snap_grad, "ct_snapshot": snap_ct}
                if snap_layer:
                    # 只存摘要，不存完整 layer_stats（避免 jsonl 膨胀）
                    snapshot_record["v2_layer_snapshot"] = {
                        k: v for k, v in snap_layer.items() if k != "layer_stats"
                    }
                if snap_exit:
                    snapshot_record["exit_depth_snapshot"] = snap_exit
                log_jsonl(metrics_path, snapshot_record)

    # ── Peak VRAM 报告 ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
        print(f"Peak VRAM: {peak_mb:.0f} MB ({peak_mb/1024:.2f} GB)  reserved: {reserved_mb:.0f} MB ({reserved_mb/1024:.2f} GB)", file=sys.stderr)

    # ── 训练结束：生成完整动力学报告 ────────────────────────────────────────
    final_snap = grad_tracker.analyze()
    if final_snap:
        grad_snapshots.append(final_snap)
        ct_snapshots.append(ct_tracker.analyze())
        final_layer = layer_grad_tracker.analyze()
        if final_layer:
            layer_grad_snapshots.append(final_layer)
        final_exit = exit_depth_tracker.analyze()
        if final_exit:
            exit_depth_snapshots.append(final_exit)
    report = render_dynamics_report(
        grad_snapshots, ct_snapshots, args.phase, step,
        layer_grad_history=layer_grad_snapshots or None,
        exit_depth_history=exit_depth_snapshots or None,
    )
    if report:
        save_report(report, dyn_report_path)
        md = render_markdown(report)
        dyn_md_path.write_text(md, encoding="utf-8")
        Logger(f"\n{'='*60}")
        Logger(f"[Dynamics Report] → {dyn_report_path}")
        Logger(md)
        Logger('='*60)

    Logger(f"Done. {step} steps, loss_lm={current_loss:.4f}")
    Logger(f"Metrics: {metrics_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Luma Refactor Trainer (Phase 0+)")
    # ── data / io ──────────────────────────────────────────────────────────
    parser.add_argument("--data_path", type=str,
                        default="../dataset/pretrain_diag.jsonl")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--tokenizer_path", type=str,
                        default="../model/qwen3_5_tokenizer")
    parser.add_argument("--save_weight", default="luma_refactor", type=str)
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3, 35, 4, 5, 6],
                        help="当前阶段。35=Phase 3.5, 4=+self_check, 5=+ct_grad_scale(废弃), 6=+world_jepa")
    # ── training ───────────────────────────────────────────────────────────
    parser.add_argument("--iters", type=int, default=642,
                        help="总训练步数（非 epoch）— 默认 3 epoch (214 步/epoch × 3)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--grad_log_interval", type=int, default=50,
                        help="每隔多少步记录一次梯度范数")
    parser.add_argument("--dod_interval", type=int, default=200,
                        help="每隔多少步做一次 DOD/DMD 动力学分析快照")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="每隔多少步保存 checkpoint (0=不保存)")
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help="Checkpoint 保存目录 (默认: artifacts/checkpoints)")
    parser.add_argument("--resume", type=str, default="",
                        help="从指定 checkpoint 恢复训练 (路径)")
    parser.add_argument("--ckpt_keep", type=int, default=3,
                        help="保留最近几个 checkpoint (默认 3)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # ── lr (固定，不用余弦退火) ────────────────────────────────────────────
    parser.add_argument("--matrix_lr", type=float, default=0.008,
                        help="Muon lr（只管 48M FFN/attn，正交化后步长稳定）")
    parser.add_argument("--scalar_lr", type=float, default=6e-4,
                        help="AdamW lr（管 143M 包括 98M Mamba，匹配 Mamba 标准 lr）")
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--betas", nargs=2, default=(0.9, 0.95), type=float)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--muon_clip_factor", type=float, default=1.0)
    parser.add_argument("--modular_norm_power", type=float, default=0.5)
    parser.add_argument("--use_8bit_muon", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_8bit_adamw", type=int, default=1, choices=[0, 1])
    parser.add_argument("--cpu_offload_optimizer", type=int, default=0, choices=[0, 1],
                        help="Offload optimizer states (momentum/m/v) to CPU pinned memory. Saves ~0.7GB (8bit) / ~1.3GB (fp32).")
    parser.add_argument("--activation_offload_compress", type=int, default=0, choices=[0, 1],
                        help="Offload compress zone activations to CPU during forward. Saves ~1-3GB, costs ~20-50ms/step.")
    parser.add_argument("--fp8", type=int, default=0, choices=[0, 1],
                        help="启用 FP8 forward (Linear 层用 FP8 tensor core, backward 仍 bf16)")
    # ── model arch ─────────────────────────────────────────────────────────
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--reason_intermediate_size", type=int, default=0)
    parser.add_argument("--reason_shared_depth", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--num_key_value_heads", type=int, default=3)
    parser.add_argument("--compression_layers", type=int, default=24)
    parser.add_argument("--compression_active_layers", type=int, default=0)
    parser.add_argument("--reason_loops", type=int, default=15)
    parser.add_argument("--reason_loops_max", type=int, default=20)
    parser.add_argument("--reason_active_loops", type=int, default=0)
    parser.add_argument("--slow_k", type=int, default=1)
    parser.add_argument("--c_t_dim", type=int, default=64)
    parser.add_argument("--meta_dim", type=int, default=96)
    parser.add_argument("--meta_state", type=int, default=32)
    parser.add_argument("--mamba_d_state", type=int, default=192)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_headdim", type=int, default=64)
    parser.add_argument("--mamba_chunk_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=151936)
    parser.add_argument("--factorized_vocab_dim", type=int, default=192)
    parser.add_argument("--bos_token_id", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=2)
    # ── aux params（只保留结构需要的，loss 全部归零）──────────────────────
    parser.add_argument("--self_rollout_steps", type=int, default=10)
    parser.add_argument("--mhc_alpha_init", type=float, default=0.01,
                        help="MHC 残差流的初始 alpha（默认 0.01，建议 0.05 避免梯度饥饿）")
    parser.add_argument("--mhc_streams", type=int, default=4,
                        help="MHC 残差流数量（默认 4）")
    parser.add_argument("--attnres_mode", type=str, default="legacy",
                        choices=["legacy", "paper", "paper_global_q"],
                        help="AttnRes 模式: legacy=当前lerp, paper=Kimi Block AttnRes, paper_global_q=论文输出+全局query")
    parser.add_argument("--attnres_compress_mode", type=str, default="",
                        help="覆盖压缩区 AttnRes 模式（空=跟随 attnres_mode）")
    parser.add_argument("--attnres_reason_mode", type=str, default="",
                        help="覆盖推理区 AttnRes 模式（空=跟随 attnres_mode）")
    parser.add_argument("--world_jepa_mode", type=str, default="full")
    # ── Phase E 能量梯度流推理（主 backbone 集成）──────────────────
    # 启用时 LumaReasonCore 会把 shared_layers stack 从 "一次 forward" 换成 "K 步能量梯度下降"
    # E(h) = 0.5 ||h - body(h, c_t)||²，h ← h - η ∇_h E
    parser.add_argument("--enable_energy_reason_core", type=int, default=0, choices=[0, 1],
                        help="Phase E: 主 backbone 启用能量梯度流推理（替换 shared_layers 单次 forward）")
    parser.add_argument("--phase_e_K_max", type=int, default=3,
                        help="Phase E: 每个 outer loop 内的能量梯度步数（3 是稳定值）")
    parser.add_argument("--phase_e_eta", type=float, default=0.1,
                        help="Phase E: 梯度步长 η，η·λ_max(∇²E)<2 保证稳定")
    parser.add_argument("--phase_e_k_backprop", type=int, default=1,
                        help="Phase E: truncated backprop 最后 N 步保留 create_graph（显存与梯度权衡）")
    parser.add_argument("--phase_e_temperature", type=float, default=0.0,
                        help="Phase E: Langevin 噪声温度 T（0=确定性，Step 3 探索性）")
    parser.add_argument("--phase_e_grad_stop_eps", type=float, default=0.0,
                        help="Phase E: 梯度范数早停阈值（0=跑满 K_max）")
    parser.add_argument("--phase_e_damped_mode", type=int, default=1, choices=[0, 1],
                        help="Phase E: 1=damped fixed-point (默认, 稳定), 0=grad mode (完整∇E, 需 double backward)")
    # Stellarator mode (v19+): 结构性仿星器改造
    parser.add_argument("--stellarator_mode", type=int, default=0, choices=[0, 1],
                        help="Stellarator: 主干 F_main(h) 不看 c_t + 低秩 modulator(c_t) + sigmoid gated fusion")
    parser.add_argument("--stellarator_mod_rank", type=int, default=8,
                        help="Stellarator: c_t modulator 的低秩维度 (默认 8)")
    # Mamba3 FP8 activation cache (saved_tensors_hooks per-block 量化, 不改 kernel)
    parser.add_argument("--mamba_fp8_activation_cache", type=int, default=0, choices=[0, 1],
                        help="Mamba3 激活缓存 FP8 (per-block 量化, 30-68% 内存节省, bf16 compute 不变)")
    parser.add_argument("--mamba_fp8_act_block_size", type=int, default=128,
                        help="FP8 每 block scale 的 block 大小 (默认 128)")
    # ── World JEPA 难度升级（scaffold 模式防崩 + V-JEPA 风格）──
    parser.add_argument("--world_mask_scheme", type=str, default="block", choices=["random", "block"],
                        help="scaffold JEPA mask 策略: random=旧版单 token 随机, block=V-JEPA 风格 span")
    parser.add_argument("--world_mask_block_mean", type=int, default=32,
                        help="scaffold block mask 的几何分布平均 span 长度（token 数），32≈1 子句；LeWM 用 48")
    parser.add_argument("--world_mask_use_mask_token", type=int, default=1, choices=[0, 1],
                        help="scaffold: 用 learned mask token 替换被遮挡位置的 hidden（防止 predictor 泄漏）")
    # SIGReg（Cramér-Wold）单正则项防崩，复用已有 --world_sigreg_weight 等 flag
    parser.add_argument("--world_jepa_weight", type=float, default=1.0,
                        help="Phase 6+: world JEPA loss 权重")
    parser.add_argument("--world_sigreg_weight", type=float, default=0.05,
                        help="Phase 6+: world JEPA SIGreg 权重（LeWM 防坍缩）")
    parser.add_argument("--world_jepa_reason_only", type=int, default=1,
                        help="Phase 6+: World JEPA 梯度只走 reasoning 区（1=开启）")
    parser.add_argument("--enable_ct_world_jepa", type=int, default=0,
                        help="Phase 6+: 启用 c_t 轨迹 World JEPA（替代 h-space）")
    parser.add_argument("--ct_world_jepa_weight", type=float, default=0.3,
                        help="Phase 6+: c_t World JEPA loss 权重")
    parser.add_argument("--world_mask_ratio", type=float, default=0.25)
    parser.add_argument("--h_mask_ratio", type=float, default=0.0,
                        help="masked h prediction ratio，error 混入赫布 surprise (0=off)")
    parser.add_argument("--h_mask_surprise_weight", type=float, default=0.3,
                        help="masked h error 在 surprise 中的混合权重")
    parser.add_argument("--h_mask_loss_mode", type=str, default="cosine", choices=["mse", "cosine", "surprise_only", "off"],
                        help="h_mask loss 模式: mse=数值回传（长训练会爆炸）, cosine=方向回传（有界 [0,2]，推荐）, surprise_only=不反传只做 surprise, off=完全关闭")
    parser.add_argument("--h_mask_loss_weight", type=float, default=0.1,
                        help="h_mask_term 在总 loss 中的权重（仅 mse 模式下生效）")
    parser.add_argument("--ct_world_reg_mode", type=str, default="none", choices=["none", "vicreg"],
                        help="c_t World JEPA 正则: none/vicreg")
    parser.add_argument("--ct_world_var_weight", type=float, default=1.0,
                        help="c_t World JEPA VICReg variance 权重")
    parser.add_argument("--ct_world_cov_weight", type=float, default=0.04,
                        help="c_t World JEPA VICReg covariance 权重")
    parser.add_argument("--world_ema_decay", type=float, default=0.99)
    parser.add_argument("--self_check_dim", type=int, default=16)
    parser.add_argument("--self_check_k", type=int, default=2)
    parser.add_argument("--exit_train_use_sampling", type=int, default=1)
    parser.add_argument("--exit_eval_use_sampling", type=int, default=0)
    parser.add_argument("--exit_sampling_temperature", type=float, default=1.0)
    parser.add_argument("--exit_score_threshold", type=float, default=0.85,
                        help="Exit score threshold for non-sampling mode (higher=more conservative)")
    parser.add_argument("--exit_second_order_delta_weight", type=float, default=0.0,
                        help="Exit policy: weight on second-order delta_h convergence signal (0=disabled)")
    parser.add_argument("--exit_aux_weight", type=float, default=0.0,
                        help="Exit auxiliary loss weight (0=disabled, 推荐 0.01-0.05)")
    parser.add_argument("--exit_min_loops", type=int, default=2,
                        help="Minimum loops before exit is allowed")
    parser.add_argument("--exit_bias_init", type=float, default=0.0,
                        help="Initial bias for exit logit (negative = stay longer)")
    parser.add_argument("--exit_warmup_steps", type=int, default=0,
                        help="Training steps during which exit is disabled (force full loops)")
    parser.add_argument("--exit_progressive_warmup", type=int, default=0,
                        help="Progressive loop warmup: first N steps cycle through depths 1→max→1→max...")
    parser.add_argument("--exit_ct_drift_weight", type=float, default=0.0,
                        help="Exit: weight on c_t drift signal (higher = stay longer when c_t changing)")
    parser.add_argument("--identity_recurrence_alpha", type=float, default=0.0,
                        help="Identity-biased recurrence: 0=off, 0.8=blend 80%% new + 20%% old h")
    parser.add_argument("--exit_entropy_weight", type=float, default=0.0,
                        help="Ouro: maximize exit score entropy to prevent exit collapse")
    parser.add_argument("--loop_lm_loss_weight", type=float, default=0.0,
                        help="RLTT: dense LM loss at each intermediate loop step")
    parser.add_argument("--rltt_stride", type=int, default=2,
                        help="RLTT: save h every N loops for dense loss (higher = less VRAM)")
    parser.add_argument("--shortcut_consistency_weight", type=float, default=0.0,
                        help="LoopFormer: KL-div consistency between short and full loop paths")
    parser.add_argument("--enable_time_conditioning", type=int, default=0, choices=[0, 1],
                        help="LoopFormer: inject normalized time t and step-size dt into reasoning loop")
    parser.add_argument("--enable_coconut", type=int, default=0, choices=[0, 1],
                        help="Coconut: c_t → thought token → re-inject into reasoning loop")
    parser.add_argument("--coconut_rounds", type=int, default=1,
                        help="Coconut: number of continuous thought re-injection rounds")
    parser.add_argument("--loop_lora_rank", type=int, default=0,
                        help="RS: per-loop LoRA rank on FFN (0=off)")
    parser.add_argument("--enable_loop_ffn_gate", type=int, default=0, choices=[0, 1],
                        help="RS: loop-dependent FFN gating")
    parser.add_argument("--introspection_input_mode", type=str, default="mean",
                        choices=["mean", "memory", "chunked", "chunked_memory"],
                        help="IS: introspection input mode (mean/memory/chunked/chunked_memory)")
    parser.add_argument("--introspection_memory_tokens", type=int, default=4,
                        help="IS: number of memory tokens for memory input mode")
    parser.add_argument("--introspection_inject_mode", type=str, default="broadcast",
                        choices=["broadcast", "token_aware", "bixt", "cmda", "bixt_cmda"],
                        help="IS: c_t injection mode (broadcast/token_aware/bixt/cmda/bixt_cmda)")
    # NM: Neuromodulated c_t writer
    parser.add_argument("--enable_introspection_swa", type=int, default=0, choices=[0, 1],
                        help="IS: sliding window attention between Mamba layers in introspection")
    parser.add_argument("--neuromod_fox_decay", type=int, default=0, choices=[0, 1],
                        help="NM: FoX-style learned forget gate on prev_c_t (prevent hebb noise accumulation)")
    parser.add_argument("--enable_neuromod_ct", type=int, default=0, choices=[0, 1],
                        help="NM: enable neuromodulated c_t writer")
    parser.add_argument("--neuromod_hebb_rank", type=int, default=8,
                        help="NM: rank for Hebbian outer product approximation")
    parser.add_argument("--hebb_init_scale", type=float, default=0.0,
                        help="hebb_out 权重初始化缩放 (0=默认zeros, >0 用 normal(std=scale) 模拟长训练)")
    parser.add_argument("--neuromod_use_delta_rule", type=int, default=0, choices=[0, 1],
                        help="NM: use Delta Rule to reduce Hebbian interference")
    parser.add_argument("--neuromod_mode", type=str, default="surprise",
                        choices=["surprise", "learned", "ponder", "multi", "jepa_surprise"],
                        help="NM: modulation mode (surprise/learned/ponder/multi/jepa_surprise)")
    # ES: Enhanced exit signals
    parser.add_argument("--enable_pc_correction", type=int, default=0, choices=[0, 1],
                        help="PC: predictive coding error correction in reasoning loop")
    parser.add_argument("--pc_alpha", type=float, default=0.1,
                        help="PC: error correction strength (0.1=gentle)")
    parser.add_argument("--cosine_total_steps", type=int, default=0,
                        help="Cosine decay 总步数 (0=用 iters，设大于 iters 可截掉尾段)")
    parser.add_argument("--enable_cosine_decay", type=int, default=0, choices=[0, 1],
                        help="LR: cosine decay to min_lr over total iters (default: fixed LR)")
    parser.add_argument("--enable_exit_entropy_signal", type=int, default=0, choices=[0, 1],
                        help="ES: entropy proxy for exit decision")
    parser.add_argument("--enable_exit_token_sensitivity", type=int, default=0, choices=[0, 1],
                        help="ES: per-token delta sensitivity for exit")
    parser.add_argument("--enable_exit_ct_curvature", type=int, default=0, choices=[0, 1],
                        help="ES: c_t trajectory curvature for exit")
    parser.add_argument("--enable_exit_confidence_gap", type=int, default=0, choices=[0, 1],
                        help="ES: confidence gap proxy for exit")
    parser.add_argument("--enable_sigreg_rollout", type=int, default=0, choices=[0, 1],
                        help="Enable SigReg on self-JEPA rollout predictions")
    # Reasoning partitioning
    parser.add_argument("--reason_num_phases", type=int, default=0,
                        help="Phase embedding: 0=disabled, >0=number of distinct phase embeddings for reasoning loops")
    parser.add_argument("--reason_head_partition", type=int, default=0, choices=[0, 1],
                        help="Head partition: each loop activates a rotating subset of diff_attn heads")
    parser.add_argument("--reason_mor_routing", type=int, default=0, choices=[0, 1],
                        help="MoR: loop-conditioned expert routing after shared layers")
    parser.add_argument("--reason_mor_num_experts", type=int, default=4)
    parser.add_argument("--reason_mor_topk", type=int, default=2)
    # True MoR: token-level depth routing (arxiv 2507.10524)
    parser.add_argument("--enable_token_depth_routing", type=int, default=0, choices=[0, 1],
                        help="True MoR: per-token depth routing, simple tokens exit early")
    parser.add_argument("--mor_target_continue_ratio", type=float, default=0.6,
                        help="Target fraction of tokens continuing per loop (balance loss target)")
    parser.add_argument("--mor_balance_weight", type=float, default=0.01)
    # Rho-1 Selective Loss
    parser.add_argument("--selective_loss_ratio", type=float, default=1.0,
                        help="Rho-1 风格：只对 top-k%% 高 loss token 训练 (1.0=全部, 0.6=top 60%%)")
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_gradient_checkpointing", type=int, default=0, choices=[0, 1],
                        help="1=对 reason_core 每次循环做 gradient checkpointing，省~70%激活VRAM，速度约慢30%")

    # 新增参数
    parser.add_argument("--compress_weight", type=float, default=0.2,
                        help="Phase 2+: 压缩区辅助 loss 权重，0=关闭")
    parser.add_argument("--self_jepa_weight", type=float, default=0.1,
                        help="Phase 3+: self-JEPA loss 权重（模型内部加权，stop-gradient 已在模型里实现）")
    parser.add_argument("--sigreg_delta_weight", type=float, default=0.05,
                        help="Phase 3.5+: self-JEPA pred_delta_c 的 SIGreg 正则权重（LeJEPA 论文推荐 0.05-0.1）")
    parser.add_argument("--sigreg_ct_weight", type=float, default=0.05,
                        help="Phase 3.5-b+: c_t 本身的 SIGreg 正则权重，防止认知状态空间坍缩")
    parser.add_argument("--loop_sigreg_weight", type=float, default=0.0,
                        help="Loop SigReg: c_t 跨循环多样性正则，惩罚 loop 间 c_t 过于相似")
    parser.add_argument("--ct_injection_mode", type=str, default="add", choices=["add", "film"],
                        help="c_t 注入方式: add(加法) / film(FiLM调制，c_t微变→h大变)")
    parser.add_argument("--jepa_predictor_dropout", type=float, default=0.0,
                        help="JEPA predictor dropout，弱化预测器防止完美预测")
    parser.add_argument("--cmda_token_wish", type=int, default=0, choices=[0, 1],
                        help="CMDA per-token wish gate: 每个 token 决定被调制程度")
    parser.add_argument("--ct_gated_attn", type=int, default=0, choices=[0, 1],
                        help="c_t 条件门控: 自省流调制 diff_attn 输出")
    parser.add_argument("--ct_conditioned_lora", type=int, default=0, choices=[0, 1],
                        help="c_t 条件 LoRA: c_t 控制 FFN 的 LoRA 权重，改变 F 的不动点")
    parser.add_argument("--ct_delta_inject", type=int, default=0, choices=[0, 1],
                        help="c_t 调制 Mamba SSM dt: 直接改变状态转移特征值")
    parser.add_argument("--ct_inject_scale", type=float, default=1.0,
                        help="c_t 注入主流的权重缩放 (1.0=默认, 2.0=翻倍)")
    parser.add_argument("--ct_per_layer_inject", type=int, default=0, choices=[0, 1],
                        help="c_t 每层独立注入 (4 个独立控制通道)")
    parser.add_argument("--delta_h_scale", type=float, default=0.0,
                        help="δh 注入 introspection 的缩放 (0=off, 0.1=轻度)")
    parser.add_argument("--delta_h_normalize", type=int, default=0, choices=[0, 1],
                        help="是否归一化 δh 方向 (0=原始, 1=单位向量)")
    parser.add_argument("--cos_sigreg_weight", type=float, default=0.0,
                        help="Cos SigReg: 惩罚相邻 loop c_t 方向相似度，直接推动方向多样性")
    parser.add_argument("--ct_momentum", type=float, default=0.0,
                        help="自省流更新频率: 每 slow_k 轮更新一次 (1=每轮, 2=隔轮)")
    parser.add_argument("--freeze_ct_during_reason", type=int, default=0, choices=[0, 1],
                        help="理论诊断: 推理环内冻结 c_t 梯度，区分 moving fixed point 与主循环不稳定")
    parser.add_argument("--enable_sigreg_ct", type=int, default=0, choices=[0, 1],
                        help="1=对 c_t 加 SIGreg（方案 2），0=只对 pred_delta_c 加（方案 1）")
    parser.add_argument("--self_check_loss_weight", type=float, default=0.1,
                        help="Phase 4+: self_check_ring loss 权重")
    parser.add_argument("--ct_grad_scale", type=float, default=0.2,
                        help="Phase 5: c_t→backbone 梯度缩放因子（0.2 = 反向梯度缩为 20%%）")
    parser.add_argument("--ct_grad_scale_aux", type=float, default=None,
                        help="Phase 5: aux loss→c_t 梯度缩放因子（None = 跟 ct_grad_scale 相同）")
    parser.add_argument("--ct_norm_penalty_weight", type=float, default=0.0,
                        help="Phase 5: c_t L2 范数正则权重（0.01-0.1），间接控制 c_t 影响力")
    parser.add_argument("--use_packing", type=int, default=1, choices=[0, 1],
                        help="1=PackedPretrainDataset（推荐），0=原始 padding 数据集")
    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"],
                        help="模型参数精度。bfloat16 显著节省 VRAM（推荐），float32 仅用于精度对比实验")

    # 实验矩阵参数（Phase 4+ 可调杠杆）
    parser.add_argument("--self_rollout_weight", type=float, default=0.0,
                        help="Self-JEPA 多步 rollout 验证权重（0=关闭）")
    parser.add_argument("--self_rollout_weighting_mode", type=str, default="legacy",
                        choices=["legacy", "near3"],
                        help="rollout 加权模式：near3=(h2=1.0, h3=0.5, h4=0.2)")
    parser.add_argument("--rollout_zone_weight", type=float, default=0.0,
                        help="rollout 活跃度守卫权重")
    parser.add_argument("--trajectory_vitality_weight", type=float, default=0.0,
                        help="轨迹防冻权重（防 c_t/world 停滞）")
    parser.add_argument("--self_progress_shape_weight", type=float, default=0.0,
                        help="progress-shape (next/trend/plateau) 辅助 loss 权重")
    parser.add_argument("--enable_progress_exit_readout", type=int, default=0,
                        help="1=把 progress 信号接入退出决策")
    parser.add_argument("--enable_backtrack_aware_progress", type=int, default=0,
                        help="1=回退感知 progress head")
    parser.add_argument("--self_local_delta_consistency_weight", type=float, default=0.0,
                        help="c_t 增量方向局部一致性权重")
    parser.add_argument("--self_local_curvature_weight", type=float, default=0.0,
                        help="c_t 轨迹曲率正则权重")
    parser.add_argument("--theory_probe_interval", type=int, default=100,
                        help="每隔多少步计算一次高开销理论 probe（LoRA 扰动、梯度源分解、谱诊断）")
    parser.add_argument("--dynamics_log_dense_interval", type=int, default=10,
                        help="step 1-256 的详细动力学日志间隔")
    parser.add_argument("--dynamics_log_mid_interval", type=int, default=50,
                        help="step 257-2048 的详细动力学日志间隔")
    parser.add_argument("--dynamics_log_sparse_interval", type=int, default=200,
                        help="step 2049+ 的详细动力学日志间隔")
    parser.add_argument("--dynamics_burst_len", type=int, default=64,
                        help="触发 burst 后连续逐步记录的长度")
    parser.add_argument("--dynamics_jsonl", type=int, default=1, choices=[0, 1],
                        help="1=写出结构化 dynamics JSONL 到 artifacts/dynamics/")

    args = parser.parse_args()
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    # autocast 精度：仅在 model_dtype=float32 时开 autocast（bf16 参数本身已够）
    autocast_ctx = nullcontext()
    if device_type == "cuda" and args.model_dtype == "float32":
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)

    if args.phase == 6:
        luma_config = build_phase6_config(args)
    elif args.phase == 5:
        luma_config = build_phase5_config(args)
    elif args.phase == 4:
        luma_config = build_phase4_config(args)
    elif args.phase == 35:
        luma_config = build_phase35_config(args)
    elif args.phase >= 3:
        luma_config = build_phase3_config(args)
    else:
        luma_config = build_phase0_config(args)

    # ── tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    if luma_config.vocab_size < len(tokenizer):
        luma_config.vocab_size = len(tokenizer)

    # ── model：参数直接存 bfloat16，节省约 50% VRAM ───────────────────────
    # 关键 fp32 操作（softmax、cross_entropy、RMSNorm 内部计算）已在模型内
    # 部显式转 float() 处理，不受参数精度影响。
    param_dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32
    model = LumaForCausalLM(luma_config).to(args.device, dtype=param_dtype)
    # ── FP8 混精度 (forward 用 FP8 tensor core, backward 用 bf16) ──────
    if args.fp8:
        from model.fp8_linear import convert_to_fp8
        fp8_count = convert_to_fp8(model, min_size=4096)
        Logger(f"FP8: converted {fp8_count} Linear layers to FP8 forward")
    # hebb_init_scale: 模拟长训练后的权重状态（验证用）
    if getattr(args, "hebb_init_scale", 0) > 0:
        _scale = args.hebb_init_scale
        for name, param in model.named_parameters():
            if "hebb_out.weight" in name:
                torch.nn.init.normal_(param, std=_scale)
                Logger(f"hebb_init_scale: {name} → normal(std={_scale})")
            elif "_ct_out_norm.scale" in name:
                param.data.fill_(_scale)  # RMSNorm scale 放大 → ct 范数放大
                Logger(f"hebb_init_scale: {name} → fill({_scale})")
            elif "ct_injection" in name and "proj.weight" in name:
                torch.nn.init.normal_(param, std=_scale * 0.1)  # W_c 也放大
                Logger(f"hebb_init_scale: {name} → normal(std={_scale*0.1})")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    vram_est_gb = total_params * (2 if param_dtype == torch.bfloat16 else 4) / 1024
    fp8_tag = " [FP8]" if args.fp8 else ""
    Logger(f"Luma Refactor Params: {total_params:.3f}M  param_dtype={param_dtype}  ~{vram_est_gb:.2f}GB param VRAM{fp8_tag}")
    # 模型结构摘要：每个子模块的名称和参数量
    Logger("Model structure:")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters()) / 1e6
        Logger(f"  {name}: {n_params:.3f}M  {type(module).__name__}")
        for sub_name, sub_module in module.named_children():
            sub_params = sum(p.numel() for p in sub_module.parameters()) / 1e6
            if sub_params > 0.01:  # 只打印 >10K 参数的子模块
                Logger(f"    {sub_name}: {sub_params:.3f}M  {type(sub_module).__name__}")
    if args.phase == 6:
        Logger(f"Phase 6: World-JEPA (mode={args.world_jepa_mode}, weight={args.world_jepa_weight}, "
               f"sigreg={args.world_sigreg_weight}, mask={args.world_mask_ratio}) "
               f"+ self-JEPA ({args.self_jepa_weight}) + SIGreg delta ({args.sigreg_delta_weight}) "
               f"+ self_check_ring ({args.self_check_loss_weight}), compress_probe weight={args.compress_weight}")
    elif args.phase == 5:
        ct_str = f" + SIGreg c_t (weight={args.sigreg_ct_weight})" if args.enable_sigreg_ct else ""
        norm_str = f", ct_norm_penalty={args.ct_norm_penalty_weight}" if args.ct_norm_penalty_weight > 0 else ""
        Logger(f"Phase 5: Phase 4 + ct_grad_scale={args.ct_grad_scale}{norm_str}, self-JEPA (weight={args.self_jepa_weight}) + SIGreg delta (weight={args.sigreg_delta_weight}){ct_str} + self_check_ring (weight={args.self_check_loss_weight}), compress_probe weight={args.compress_weight}")
    elif args.phase == 4:
        ct_str = f" + SIGreg c_t (weight={args.sigreg_ct_weight})" if args.enable_sigreg_ct else ""
        Logger(f"Phase 4: self-JEPA (weight={args.self_jepa_weight}) + SIGreg delta (weight={args.sigreg_delta_weight}){ct_str} + self_check_ring (weight={args.self_check_loss_weight}), compress_probe weight={args.compress_weight}")
    elif args.phase == 35:
        ct_str = f" + SIGreg c_t (weight={args.sigreg_ct_weight})" if args.enable_sigreg_ct else ""
        Logger(f"Phase 3.5: self-JEPA (weight={args.self_jepa_weight}) + SIGreg delta (weight={args.sigreg_delta_weight}){ct_str}, residual_reg=0, compress_probe weight={args.compress_weight}")
    elif args.phase >= 3:
        Logger(f"Phase {args.phase}: self-JEPA enabled (weight={args.self_jepa_weight}, stop-grad), compress_probe weight={args.compress_weight}")
    else:
        Logger(f"Phase {args.phase}: all aux losses disabled, fixed LR mode")

    # ── optimizer (fixed LR: total_steps → ∞ so cosine factor ≈ 1.0) ────
    optimizer_config = LumaOptimizerConfig(
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        muon_momentum=args.muon_momentum,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
        muon_clip_factor=args.muon_clip_factor,
        modular_norm_power=args.modular_norm_power,
        use_8bit_muon=bool(args.use_8bit_muon),
        use_8bit_adamw=bool(args.use_8bit_adamw),
    )
    optimizer = LumaMuonAdamWOptimizer(model, optimizer_config)
    if getattr(args, "cpu_offload_optimizer", 0):
        optimizer.enable_cpu_offload()
        Logger("Optimizer CPU offload enabled — states will live on CPU pinned memory")
    routing_summary = getattr(optimizer, "routing_summary", {})
    targeted_routes = []
    # mamba 统计（不逐个打印，太多）
    mamba_counts = {"muon": 0, "adamw": 0}
    for family in ("adamw", "muon"):
        for name in routing_summary.get(family, []):
            if (
                "ct_injection.proj.weight" in name
                or "c_t_head.weight" in name
                or "h_mask_predictor.weight" in name
            ):
                targeted_routes.append(f"{name} -> {family}")
            if "mamba" in name.lower():
                mamba_counts[family] += 1
    if targeted_routes or mamba_counts["adamw"] + mamba_counts["muon"] > 0:
        Logger("Optimizer routing:")
        for line in targeted_routes:
            Logger(f"  {line}")
        Logger(f"  mamba.*: {mamba_counts['adamw']} → adamw, {mamba_counts['muon']} → muon")
    # LR schedule: cosine decay 到 min_lr (enable_cosine_decay=1) 或固定 LR (默认)
    _lr_total = getattr(args, "cosine_total_steps", 0) or args.iters
    if not getattr(args, "enable_cosine_decay", 0):
        _lr_total = 10 ** 9
    scheduler = LumaCosineScheduler(
        optimizer,
        total_steps=_lr_total,
        matrix_base_lr=args.matrix_lr,
        scalar_base_lr=args.scalar_lr,
        min_lr_ratio=0.1,
    )
    # bf16 参数不需要 GradScaler（动态范围足够）
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # ── data ──────────────────────────────────────────────────────────────
    if args.use_packing:
        train_ds = PackedPretrainDataset(
            args.data_path, tokenizer,
            max_length=args.max_seq_len,
            shuffle_docs=True,
            min_doc_tokens=4,
        )
        Logger(f"PackedDataset: {len(train_ds)} packs × seq {args.max_seq_len}  "
               f"(padding=0, 文档边界跨预测已屏蔽)")
    else:
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        Logger(f"PaddingDataset: {len(train_ds)} samples × seq {args.max_seq_len}")
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.use_compile:
        model = torch.compile(model)

    metrics_path = ARTIFACTS_DIR / f"phase{args.phase}_metrics.jsonl"
    Logger(f"Metrics → {metrics_path}")
    if args.dynamics_jsonl:
        Logger(f"Dynamics JSONL → {ARTIFACTS_DIR.parent / 'dynamics' / f'{args.save_weight}_phase{args.phase}.jsonl'}")

    # ── Phase 2+: 压缩区辅助 probe（训练完可丢弃，不影响推理结构）────────
    compress_probe = None
    compress_weight = args.compress_weight
    if args.phase >= 2 and compress_weight > 0:
        compress_probe = torch.nn.Linear(
            luma_config.hidden_size, luma_config.vocab_size, bias=False,
        ).to(args.device, dtype=param_dtype)
        # compress_probe 用独立 AdamW 优化（不走 Muon）
        compress_probe_optim = torch.optim.AdamW(
            compress_probe.parameters(), lr=args.scalar_lr, weight_decay=0.01,
        )
        # 把 compress_probe 的 param_groups 挂到主 optimizer 上，让 scheduler 能统一管理
        optimizer.param_groups.extend(compress_probe_optim.param_groups)
        Logger(f"Phase 2: compress_probe enabled, weight={compress_weight}")

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_step = load_checkpoint(resume_path, model, optimizer, scheduler,
                                         compress_probe=compress_probe)
        else:
            print(f"WARNING: checkpoint not found: {resume_path}, starting from scratch", file=sys.stderr)

    train(args, luma_config, model, loader, optimizer, scheduler, scaler,
          autocast_ctx, metrics_path,
          compress_probe=compress_probe, compress_weight=compress_weight,
          start_step=start_step)
