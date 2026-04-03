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
from luma_stage0.optimizers import LumaCosineScheduler, LumaMuonAdamWOptimizer, LumaOptimizerConfig
from model.model_minimind import LumaConfig, LumaForCausalLM
from trainer.trainer_utils import Logger, setup_seed

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "refactor"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def build_phase0_config(args: argparse.Namespace) -> LumaConfig:
    """Phase 0: 只留 lm_loss，所有辅助 loss 权重归零，禁用可关闭的辅助模块。"""
    return LumaConfig(
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
        # ── 所有辅助 loss 关闭 ──────────────────────────────────────────────
        enable_world_jepa=False,
        world_jepa_weight=0.0,
        world_jepa_mode=args.world_jepa_mode,
        world_mask_ratio=args.world_mask_ratio,
        world_ema_decay=args.world_ema_decay,
        disable_self_jepa=True,
        self_jepa_weight=0.0,
        self_rollout_weight=0.0,
        self_rollout_steps=args.self_rollout_steps,
        self_jepa_residual_reg=0.0,
        exit_aux_weight=0.0,
        rollout_zone_weight=0.0,
        routing_tier_entropy_weight=0.0,
        routing_min_local_share_weight=0.0,
        trajectory_vitality_weight=0.0,
        compression_dynamics_weight=0.0,
        enable_self_check_ring=False,
        self_check_dim=args.self_check_dim,
        self_check_k=args.self_check_k,
        # ── exit controller 保留结构但权重归零 ─────────────────────────────
        exit_train_use_sampling=bool(args.exit_train_use_sampling),
        exit_eval_use_sampling=bool(args.exit_eval_use_sampling),
        exit_sampling_temperature=args.exit_sampling_temperature,
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
    三组参数的梯度范数：
      compress  = embedding + compression zone
      shared    = reason_core (shared reasoning block)
      reasoning = 其余推理侧参数（mhc, introspection, c_t, exit_controller 等）

    grad_ratio = max / min，越接近 1 越好，> 10 说明严重失衡。
    """
    backbone = model.model  # LumaBackbone

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

    return {
        "grad_norm_compress": n_compress,
        "grad_norm_shared": n_shared,
        "grad_norm_reasoning": n_reasoning,
        "grad_ratio": ratio,
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

def train(args, luma_config: LumaConfig, model: LumaForCausalLM,
          loader: DataLoader, optimizer, scheduler, scaler,
          autocast_ctx, metrics_path: Path):

    model.train()
    start_time = time.time()
    step = 0

    for input_ids, labels in loader:
        step += 1
        if step > args.iters:
            break

        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss_lm = res.loss          # aux losses are all 0 in Phase 0
            loss = loss_lm / args.accumulation_steps

        scaler.scale(loss).backward()

        # ── 梯度监控（backward 后、step 前）────────────────────────────────
        do_grad_log = (step % args.grad_log_interval == 0)
        grad_metrics = {}
        if do_grad_log:
            # unscale first so grad norms are in the original scale
            scaler.unscale_(optimizer.matrix_optimizer)
            scaler.unscale_(optimizer.scalar_optimizer)
            raw_model = getattr(model, "_orig_mod", model)
            grad_metrics = compute_grad_metrics(raw_model)

        if step % args.accumulation_steps == 0:
            if not do_grad_log:
                scaler.unscale_(optimizer.matrix_optimizer)
                scaler.unscale_(optimizer.scalar_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ── 日志 ────────────────────────────────────────────────────────────
        current_loss = loss.item() * args.accumulation_steps
        scalar_lr = next(g["lr"] for g in optimizer.param_groups if g.get("optim_family") == "adamw")
        matrix_lr = next(g["lr"] for g in optimizer.param_groups if g.get("optim_family") == "muon")

        record = {
            "step": step,
            "loss_lm": current_loss,
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
            Logger(
                f"[{step}/{args.iters}] loss_lm={current_loss:.4f}"
                f"  scalar_lr={scalar_lr:.2e}  eta={eta:.1f}min"
                + grad_line
            )

        # ── Phase 1 early stop hint ─────────────────────────────────────────
        if do_grad_log and grad_metrics:
            compress_norm = grad_metrics["grad_norm_compress"]
            if compress_norm < 1e-7:
                Logger(
                    f"⚠️  [step {step}] grad_norm_compress={compress_norm:.2e} ≈ 0  "
                    "→ 梯度无法传到压缩区，需执行预案 S1（跳连捷径）"
                )
            elif step >= 500:
                ratio = grad_metrics["grad_ratio"]
                Logger(
                    f"✓  [step {step}] grad_norm_compress={compress_norm:.2e}  ratio={ratio:.2f}"
                )

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
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="当前阶段（用于 metrics 文件命名）")
    # ── training ───────────────────────────────────────────────────────────
    parser.add_argument("--iters", type=int, default=1500,
                        help="总训练步数（非 epoch）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--grad_log_interval", type=int, default=50,
                        help="每隔多少步记录一次梯度范数")
    parser.add_argument("--save_interval", type=int, default=500)
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
    parser.add_argument("--mamba_chunk_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=151936)
    parser.add_argument("--factorized_vocab_dim", type=int, default=192)
    parser.add_argument("--bos_token_id", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=2)
    # ── aux params（只保留结构需要的，loss 全部归零）──────────────────────
    parser.add_argument("--self_rollout_steps", type=int, default=10)
    parser.add_argument("--world_jepa_mode", type=str, default="full")
    parser.add_argument("--world_mask_ratio", type=float, default=0.25)
    parser.add_argument("--world_ema_decay", type=float, default=0.99)
    parser.add_argument("--self_check_dim", type=int, default=16)
    parser.add_argument("--self_check_k", type=int, default=2)
    parser.add_argument("--exit_train_use_sampling", type=int, default=1)
    parser.add_argument("--exit_eval_use_sampling", type=int, default=0)
    parser.add_argument("--exit_sampling_temperature", type=float, default=1.0)
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])

    # 新增参数
    parser.add_argument("--use_packing", type=int, default=1, choices=[0, 1],
                        help="1=PackedPretrainDataset（推荐），0=原始 padding 数据集")
    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"],
                        help="模型参数精度。bfloat16 显著节省 VRAM（推荐），float32 仅用于精度对比实验")

    args = parser.parse_args()
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    # autocast 精度：仅在 model_dtype=float32 时开 autocast（bf16 参数本身已够）
    autocast_ctx = nullcontext()
    if device_type == "cuda" and args.model_dtype == "float32":
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)

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
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    vram_est_gb = total_params * (2 if param_dtype == torch.bfloat16 else 4) / 1024
    Logger(f"Luma Refactor Params: {total_params:.3f}M  param_dtype={param_dtype}  ~{vram_est_gb:.2f}GB param VRAM")
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

    train(args, luma_config, model, loader, optimizer, scheduler, scaler,
          autocast_ctx, metrics_path)
