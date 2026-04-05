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
        world_ema_decay=args.world_ema_decay,
        self_rollout_steps=args.self_rollout_steps,
        self_check_dim=args.self_check_dim,
        self_check_k=args.self_check_k,
        exit_train_use_sampling=bool(args.exit_train_use_sampling),
        exit_eval_use_sampling=bool(args.exit_eval_use_sampling),
        exit_sampling_temperature=args.exit_sampling_temperature,
        exit_second_order_delta_weight=getattr(args, "exit_second_order_delta_weight", 0.0),
        use_gradient_checkpointing=bool(args.use_gradient_checkpointing),
        activation_offload_compress=bool(getattr(args, "activation_offload_compress", 0)),
        reason_num_phases=getattr(args, "reason_num_phases", 0),
        reason_head_partition=bool(getattr(args, "reason_head_partition", 0)),
        reason_mor_routing=bool(getattr(args, "reason_mor_routing", 0)),
        reason_mor_num_experts=getattr(args, "reason_mor_num_experts", 4),
        reason_mor_topk=getattr(args, "reason_mor_topk", 2),
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
        # ── SIGreg delta（继承 Phase 3.5）─────────────────────────────────────
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
        # ── 其他辅助 loss 关闭 ────────────────────────────────────────────────
        self_jepa_residual_reg=0.0,
        exit_aux_weight=0.0,
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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for input_ids, labels in _infinite_loader(loader):
        step += 1
        if step > args.iters:
            break

        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # Phase 3+: res.loss 已包含 self_jepa_term（模型内部加权）
            # res.aux_loss = self_jepa_weight * self_jepa_loss（+ 其他目前为 0 的项）
            loss_jepa_val = res.aux_loss.item() if res.aux_loss is not None else 0.0
            loss_lm = res.loss - (res.aux_loss if res.aux_loss is not None else res.loss.new_zeros(()))

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
                loss = (res.loss + compress_weight * loss_compress) / args.accumulation_steps
                loss_compress_val = loss_compress.item()
            else:
                loss = res.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        # ── 梯度监控（backward 后、step 前）────────────────────────────────
        do_grad_log = (step % args.grad_log_interval == 0)
        # Always unscale so grad_metrics can be computed for tracker every step
        scaler.unscale_(optimizer.matrix_optimizer)
        scaler.unscale_(optimizer.scalar_optimizer)
        raw_model = getattr(model, "_orig_mod", model)
        grad_metrics = compute_grad_metrics(raw_model)

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

        record = {
            "step": step,
            "loss_lm": loss_lm.item(),
            "loss_compress": loss_compress_val,
            "loss_jepa": loss_jepa_val,
            "loss_total": current_loss,
            "scalar_lr": scalar_lr,
            "matrix_lr": matrix_lr,
            "elapsed_s": round(time.time() - start_time, 1),
            **grad_metrics,
        }
        log_jsonl(metrics_path, record)

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
            compress_line = f"  loss_c={loss_compress_val:.4f}" if compress_weight > 0 else ""
            jepa_line = f"  loss_j={loss_jepa_val:.4f}" if loss_jepa_val > 0 else ""
            Logger(
                f"[{step}/{args.iters}] loss_lm={loss_lm.item():.4f}{compress_line}{jepa_line}"
                f"  scalar_lr={scalar_lr:.2e}  eta={eta:.1f}min"
                + grad_line
            )

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

        if step % args.dod_interval == 0 or step == args.iters:
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
                    f"[DOD/DMD step {step}]  dod_rank={dod_rank}"
                    f"  mode1_energy={e1:.1f}%"
                    f"  grad_dmd_radius={dmd_str}"
                    f"  v2_rank={v2_rank}/{v2_dims} v2_mode1={v2_e1:.1f}%"
                    + (f"  dead={dead}" if dead else "")
                    + exit_info
                )
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
    parser.add_argument("--iters", type=int, default=1500,
                        help="总训练步数（非 epoch）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
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
                        help="Muon lr（建议取正式训练 matrix_lr 的 30-50%）")
    parser.add_argument("--scalar_lr", type=float, default=1e-4,
                        help="AdamW lr（建议取正式训练 scalar_lr 的 30-50%）")
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--betas", nargs=2, default=(0.9, 0.95), type=float)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
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
    parser.add_argument("--world_jepa_mode", type=str, default="full")
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
    parser.add_argument("--world_ema_decay", type=float, default=0.99)
    parser.add_argument("--self_check_dim", type=int, default=16)
    parser.add_argument("--self_check_k", type=int, default=2)
    parser.add_argument("--exit_train_use_sampling", type=int, default=1)
    parser.add_argument("--exit_eval_use_sampling", type=int, default=0)
    parser.add_argument("--exit_sampling_temperature", type=float, default=1.0)
    parser.add_argument("--exit_second_order_delta_weight", type=float, default=0.0,
                        help="Exit policy: weight on second-order delta_h convergence signal (0=disabled)")
    # Reasoning partitioning
    parser.add_argument("--reason_num_phases", type=int, default=0,
                        help="Phase embedding: 0=disabled, >0=number of distinct phase embeddings for reasoning loops")
    parser.add_argument("--reason_head_partition", type=int, default=0, choices=[0, 1],
                        help="Head partition: each loop activates a rotating subset of diff_attn heads")
    parser.add_argument("--reason_mor_routing", type=int, default=0, choices=[0, 1],
                        help="MoR: loop-conditioned expert routing after shared layers")
    parser.add_argument("--reason_mor_num_experts", type=int, default=4)
    parser.add_argument("--reason_mor_topk", type=int, default=2)
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
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    vram_est_gb = total_params * (2 if param_dtype == torch.bfloat16 else 4) / 1024
    fp8_tag = " [FP8]" if args.fp8 else ""
    Logger(f"Luma Refactor Params: {total_params:.3f}M  param_dtype={param_dtype}  ~{vram_est_gb:.2f}GB param VRAM{fp8_tag}")
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
    # 固定 lr：把 total_steps 设成极大值，余弦因子始终 ≈ 1.0
    scheduler = LumaCosineScheduler(
        optimizer,
        total_steps=10 ** 9,
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
