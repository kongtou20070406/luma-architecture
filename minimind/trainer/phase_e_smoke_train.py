"""Phase E Step 1 端到端 smoke training — 独立最小训练脚本。

不依赖 LumaForCausalLM / LumaBackbone 的复杂 forward 路径，直接手工组装:
    embedding → compression → EnergyReasonCore → pre_lm_norm → lm_head

用途:
    验证 Phase E 的能量梯度流能在真实 LM CE loss 信号下驱动 loss 下降。
    这是 Phase E 的"第一次真实心跳"——下一步才谈整合进生产训练器。

使用:
    cd minimind/trainer
    python phase_e_smoke_train.py --iters 200 --batch_size 1 --max_seq_len 128 \\
        --data_path ../../luma_dataset/mixes/v5_pretrain.jsonl

理论背景: docs/reports/Luma_PhaseE_Theory_Seed_20260412.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# 允许从 minimind/ 根目录导入
_THIS_DIR = Path(__file__).resolve().parent
_MINIMIND_ROOT = _THIS_DIR.parent
if str(_MINIMIND_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIMIND_ROOT))

from model.model_minimind import (
    LumaConfig,
    FactorizedEmbedding,
    CompressionZone,
    EnergyReasonCore,
    LumaZCRMSNorm,
    FactorizedLMHead,
    LeWorldModelStyleJEPA,
)


# ═══════════════════════════════════════════════════════════
# PhaseEMinimalLM — 最小装配
# ═══════════════════════════════════════════════════════════
class PhaseEMinimalLM(nn.Module):
    """Phase E 端到端最小模型: embedding + compression + EnergyReasonCore + lm_head.

    不包含: self-jepa, world-jepa, exit_ctrl, introspection, MHC,
    token_depth_routing. 这些都是 Step 3-5 的后续任务。

    Step 进度跟随 `phase_e_step` 配置：
    - step=1: c_t 固定为 0 向量
    - step=2: c_t 由 sequence-level 池化 + head 生成（token 轴, loop 内冻结）
    - step=3: + Langevin 温度 (通过 config.phase_e_temperature)
    - step=4: + world_jepa 作为辅助 loss（内部世界模型回归）
    - step=5: + gradient-norm early stopping (通过 config.phase_e_grad_stop_eps)
    - step=6: + 拆除托卡马克补丁（需手动 config 配合）
    - step=7: + Hessian 谱约束 + 把 world_jepa 吸入能量函数
    """

    def __init__(self, config: LumaConfig, phase_e_step: int = 1):
        super().__init__()
        self.config = config
        self.phase_e_step = phase_e_step
        self.embedding = FactorizedEmbedding(config)
        self.compression = CompressionZone(config)
        # Phase 4 诊断修复：在 energy loop 之前加 RMSNorm
        # 让 energy 循环的输入始终规范化，防止 ||h|| 在训练中慢慢漂移到无穷
        # （LM loss 通过 pre_lm_norm 洗掉了 ||h||，对 magnitude drift 无压力，
        # 所以 h 可以自由爆炸而不惩罚训练目标 — pre_reason_norm 把这个问题堵在入口）
        self.pre_reason_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.reason_core = EnergyReasonCore(config)
        self.pre_lm_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = FactorizedLMHead(config, self.embedding)
        # ═══ Step 2: c_t token 轴生成头 ═══
        # 把 compressed 的 sequence-level 池化通过小 MLP 映射到 c_t 空间
        # 关键设计: 不在 loop 轴演化，一次 forward 生成一次；loop 内 c_t 冻结
        # 这让 h 的快动力学（loop）和 c_t 的慢动力学（token/sequence）
        # 时间尺度彻底解耦，不受 loop 数变化影响
        if phase_e_step >= 2:
            self.c_t_init_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.c_t_dim * 2, bias=False),
                nn.SiLU(),
                nn.Linear(config.c_t_dim * 2, config.c_t_dim, bias=False),
            )
            # 零初始化末层 → 初始 c_t ≈ 0，等价于 Step 1 起点
            nn.init.zeros_(self.c_t_init_head[-1].weight)
            # Phase 4 长训发现：无界 c_t_init_head 输出在长训里会爆发（某 batch c_t_norm 从
            # ~1 跳到 3+，触发 h 爆炸 → ρ→∞）。c_t 的 scale 参数：构造性 squash 替代 clamp。
            # 把 c_t 约束在 [-c_t_scale, c_t_scale] 范围内（通过 tanh），这是架构级约束，
            # 不是外部 clamp — 梯度可以穿过 tanh 正常流动，但输出 magnitude 有硬上界。
            self.c_t_scale = float(getattr(config, "phase_e_c_t_scale", 1.0))
        # ═══ Step 4: world_jepa 作为辅助 loss ═══
        # Luma 的内部世界模型 — LeWorldModelStyleJEPA 预测被 mask 的 hidden 表征。
        # 理论动机：从能量视角看，world_jepa 的预测残差 ||pred - target||² 本身
        # 就是一个能量项，和方案 A 的 ||h - body(h)||² 形式同构。
        # Step 4 先作为外层 aux loss 接入，Step 7 再把它吸入 E_total。
        if phase_e_step >= 4:
            self.world_latent_jepa = LeWorldModelStyleJEPA(config)

    def _compute_c_t(self, compressed: torch.Tensor) -> torch.Tensor:
        """根据 phase_e_step 决定 c_t 的生成方式.

        Step 1: c_t 恒为 0
        Step 2+: c_t 来自 compressed.mean(dim=1) 的 MLP 映射

        注意: compressed 不 detach — 让 compression 也能通过 c_t 收到 "c_t 语义需求" 的反馈
        """
        B = compressed.shape[0]
        if self.phase_e_step < 2:
            return compressed.new_zeros(B, self.config.c_t_dim)
        # Step 2+: sequence-level 池化 → c_t_init_head → c_t
        # c_t_init_head 是 bf16，所以池化后保持 bf16
        h_pool = compressed.mean(dim=1)  # [B, D] bf16
        c_t_raw = self.c_t_init_head(h_pool)  # [B, c_t_dim] bf16
        # 【Phase 4 诊断修复 2026-04-12】:
        # 无界 c_t 在长训中会有 magnitude 爆发 (phase_e_phase4_2000 观察到 step 650+)。
        # 用 tanh squash 限制 c_t 每维 ∈ [-c_t_scale, +c_t_scale]，是构造性上界不是 clamp。
        # 梯度穿过 tanh 正常流动 (除了接近饱和时 sigmoid 导数小)，上界由架构保证。
        c_t = torch.tanh(c_t_raw) * self.c_t_scale
        return c_t

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, dict]:
        T_in = input_ids.shape[1]
        h = self.embedding(input_ids)
        # compression zone forward (gradient checkpointing 不兼容 Phase E 的 autograd.grad，保持关闭)
        out = self.compression(h)
        if isinstance(out, tuple):
            compressed = out[0]  # CompressionZone 返回 (x, block_reprs, aux_dict)
        else:
            compressed = out
        # compression 在输入前 cat 了 memory tokens，截掉只保留真实 token 那段
        compressed = compressed[:, -T_in:, :]
        # Phase 4 诊断修复: energy 循环前先规范化 h 的 magnitude
        # 防止 ||h|| drift to infinity，因为 LM loss 通过 pre_lm_norm 不惩罚 magnitude
        compressed = self.pre_reason_norm(compressed)
        # c_t 生成（step 1=零, step 2+=sequence pool head）
        c_t = self._compute_c_t(compressed)
        # Step 7: world_jepa 吸入能量函数内部，通过 extra_energy_fn 参与梯度下降
        # 理论含义: 现在 ∇_h E_total = ∇_h(0.5||h-body||² + β·||pred_world(h)-target||²)
        # 两条保守梯度场联合塑造 h 的演化轨迹
        extra_fn = None
        if self.phase_e_step >= 7 and hasattr(self, "world_latent_jepa"):
            def _world_energy(h_inner: torch.Tensor, c_t_inner: torch.Tensor) -> torch.Tensor:
                # 注意: world_latent_jepa.forward 返回 dict，其中 world_jepa_loss
                # 已经是 cosine_loss + sigreg_weight*SIGreg 的总和
                w_aux = self.world_latent_jepa(h_inner)
                return self.config.world_jepa_weight * w_aux["world_jepa_loss"]
            extra_fn = _world_energy
        # 必须用 math SDP backend 才能做 create_graph=True 的二阶梯度
        with sdpa_kernel([SDPBackend.MATH]):
            h_final, reason_aux = self.reason_core(compressed, c_t, extra_energy_fn=extra_fn)
        h_final = self.pre_lm_norm(h_final)
        logits = self.lm_head(h_final)
        loss = None
        world_jepa_loss_val = 0.0
        if labels is not None:
            x_shift = logits[..., :-1, :].contiguous()
            y_shift = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                x_shift.reshape(-1, x_shift.size(-1)),
                y_shift.reshape(-1),
                ignore_index=-100,
            )
            # Step 4-6: world_jepa 作为外层 aux loss
            # Step 7: world_jepa 已被吸入能量函数，不能重复加到外层 loss（会双计数）
            if 4 <= self.phase_e_step <= 6 and hasattr(self, "world_latent_jepa"):
                world_aux = self.world_latent_jepa(h_final)
                w_loss = world_aux["world_jepa_loss"]
                loss = loss + self.config.world_jepa_weight * w_loss
                world_jepa_loss_val = float(w_loss.detach().item())
            elif self.phase_e_step >= 7 and hasattr(self, "world_latent_jepa"):
                # Step 7: world_jepa 梯度完全走内层能量路径，外层只记录 loss 值不反传
                with torch.no_grad():
                    world_aux = self.world_latent_jepa(h_final)
                    world_jepa_loss_val = float(world_aux["world_jepa_loss"].item())
        # 诊断：记录 c_t 的范数演化（Step 2 看得到）
        reason_aux["phase_e_c_t_norm"] = float(c_t.float().norm().item()) if c_t.numel() > 0 else 0.0
        reason_aux["phase_e_world_jepa_loss"] = world_jepa_loss_val
        return loss, logits, reason_aux


# ═══════════════════════════════════════════════════════════
# 简易数据加载 — 复用现有 v5 pretrain 数据
# ═══════════════════════════════════════════════════════════
class SimpleJsonlDataset(torch.utils.data.Dataset):
    """从 jsonl 文件加载 token 序列，最简实现。"""

    def __init__(self, path: str, max_seq_len: int, vocab_size: int, max_samples: int = 10000):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.records: List[List[int]] = []
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= max_samples:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # 兼容几种常见字段名
                tokens = None
                for key in ("input_ids", "tokens", "ids"):
                    if key in obj:
                        tokens = obj[key]
                        break
                if tokens is None and "text" in obj:
                    # fallback: 字符级 hash 制造伪 token 序列（仅供 smoke test）
                    text = obj["text"][:max_seq_len * 4]
                    tokens = [hash(c) % vocab_size for c in text][:max_seq_len]
                if tokens is None:
                    continue
                if len(tokens) < 2:
                    continue
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                # 所有 token 必须在 vocab_size 范围内
                tokens = [min(max(0, int(t)), vocab_size - 1) for t in tokens]
                self.records.append(tokens)
                count += 1
        print(f"[data] loaded {len(self.records)} records from {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        tokens = self.records[idx]
        # pad 到 max_seq_len
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


def _collate(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


# ═══════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_path", type=str, default="../../luma_dataset/mixes/v5_pretrain.jsonl")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_weight", type=str, default="phase_e_smoke")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--reason_shared_depth", type=int, default=2)
    parser.add_argument("--compression_active_layers", type=int, default=4)
    parser.add_argument("--phase_e_K_max", type=int, default=5)
    parser.add_argument("--phase_e_eta", type=float, default=0.1)
    parser.add_argument("--phase_e_temperature", type=float, default=0.0, help="Step 3: Langevin 温度")
    parser.add_argument("--phase_e_grad_stop_eps", type=float, default=0.0, help="Step 4: gradient-norm early stop")
    parser.add_argument("--phase_e_step", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="Phase E Step 进度：1=c_t 零, 2=c_t 从序列池化, 3=+Langevin, 4=+world_jepa aux, 5=+K=7 早停, 6=拆补丁, 7=world_jepa 进入能量, 8=Hessian 约束")
    parser.add_argument("--probe_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--phase_e_c_t_scale", type=float, default=1.0, help="c_t tanh squash 尺度")
    parser.add_argument("--swa_window", type=int, default=1024, help="sliding window attention 窗口大小（chunked 实现）")
    parser.add_argument("--mamba_d_state", type=int, default=192, help="Mamba SSM 状态维度")
    parser.add_argument("--phase_e_k_backprop", type=int, default=0,
                        help="Truncated K-loop backprop: 只对最后 N 步建 autograd 图。0=full graph。"
                             "显存 = N/K_max × 原内存。seq=2048 通关关键。")
    parser.add_argument("--optimizer_8bit", type=int, default=0, choices=[0, 1],
                        help="bitsandbytes AdamW8bit，优化器状态 8-bit 存储，省 ~4x 显存")
    parser.add_argument("--phase_e_custom_checkpoint", type=int, default=0, choices=[0, 1],
                        help="循环内自定义 checkpoint: body activations forward 后立即释放，backward re-compute。"
                             "Phase E 兼容 Mamba triton kernel 的唯一 checkpoint 方案。")
    # 显存优化（默认关）
    parser.add_argument("--cpu_offload_optimizer", type=int, default=0, choices=[0, 1],
                        help="把优化器状态卸到 CPU pinned memory（仅在纯 AdamW 下安全）")
    parser.add_argument("--activation_offload_compress", type=int, default=0, choices=[0, 1],
                        help="compression zone 激活卸载到 CPU（默认 0，速度代价 ~2.3x）")
    parser.add_argument("--use_gradient_checkpointing", type=int, default=0, choices=[0, 1],
                        help="对 compression 的 Mamba 块启用内置 reentrant gradient checkpointing (mamba3_module.py)。不影响 Phase E 能量循环的 autograd.grad 调用。")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} iters={args.iters} batch={args.batch_size} seq={args.max_seq_len}")

    # Minimal LumaConfig — Step 1 只需要基本的 CR-loop 参数
    cfg = LumaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=12,
        num_key_value_heads=3,
        compression_layers=args.compression_active_layers,
        compression_active_layers=args.compression_active_layers,
        reason_shared_depth=args.reason_shared_depth,
        reason_intermediate_size=args.hidden_size * 2,
        c_t_dim=64,
        mamba_d_state=args.mamba_d_state,
        mamba_chunk_size=32,
        factorized_vocab_dim=256,
        ct_injection_mode="add",
        # 托卡马克补丁分类（Phase 4 长训后更新）：
        # - ct_inj_max 是运行时 clamp（条件判断，非构造性）→ 拆
        # - enable_wc_row_norm 是权重级 Lipschitz 归一化（和 tanh 同类构造性约束）→ 保留
        # 原 Step 6 的 "row_norm off" 是错误分类，长训验证它其实是仿星器式架构约束
        ct_inj_max=(0.0 if args.phase_e_step >= 6 else 0.05),
        enable_wc_row_norm=True,  # 保留：W_c 的构造性 Lipschitz 上界
        ct_modulation_mode="none",
        swa_window=args.swa_window,
        reason_head_partition=False,
        enable_time_conditioning=False,
        loop_lora_rank=0,
        enable_loop_ffn_gate=False,
        ct_per_layer_inject=False,
        enable_reasoning_state_ring=False,
        enable_energy_reason_core=True,
        phase_e_K_max=args.phase_e_K_max,
        phase_e_eta=args.phase_e_eta,
        phase_e_temperature=args.phase_e_temperature,
        phase_e_grad_stop_eps=args.phase_e_grad_stop_eps,
        phase_e_k_backprop=args.phase_e_k_backprop,
        phase_e_custom_checkpoint=bool(args.phase_e_custom_checkpoint),
        phase_e_c_t_scale=args.phase_e_c_t_scale,
        use_gradient_checkpointing=bool(args.use_gradient_checkpointing),
        activation_offload_compress=bool(args.activation_offload_compress),
        disable_self_jepa=True,
        world_jepa_weight=(0.5 if args.phase_e_step >= 4 else 0.0),
        world_jepa_mode=("full" if args.phase_e_step >= 4 else "off"),
        world_mask_ratio=0.25,
        world_sigreg_weight=0.1,
        enable_token_depth_routing=False,
        enable_compression_mhc=False,
        enable_math_adapter_lane=False,
    )

    print(f"[setup] building model (phase_e_step={args.phase_e_step}, ckpt={bool(args.use_gradient_checkpointing)})...")
    model = PhaseEMinimalLM(cfg, phase_e_step=args.phase_e_step).to(device).to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] params: {n_params/1e6:.2f}M")

    print(f"[setup] loading data from {args.data_path}")
    data_path_abs = (Path(args.data_path) if os.path.isabs(args.data_path) else _THIS_DIR / args.data_path).resolve()
    if not data_path_abs.exists():
        print(f"[warn] data file not found: {data_path_abs}")
        print(f"[warn] falling back to pure random token smoke mode")
        dataset = None
    else:
        dataset = SimpleJsonlDataset(
            str(data_path_abs),
            max_seq_len=args.max_seq_len,
            vocab_size=args.vocab_size,
            max_samples=args.iters * args.batch_size * 2,
        )
        if len(dataset) == 0:
            print("[warn] dataset empty after parsing, falling back to random tokens")
            dataset = None

    loader = None
    if dataset is not None:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate,
            drop_last=True,
        )

    # Optimizer 选择：
    # - optimizer_8bit=1: bitsandbytes AdamW8bit (优化器状态 8-bit, 省 ~1.2 GB on 205M)
    # - cpu_offload_optimizer=1: LumaMuonAdamWOptimizer (注意 Muon 破坏 Phase E ρ，不推荐)
    # - 默认: plain AdamW fp32
    if args.optimizer_8bit:
        try:
            from bitsandbytes.optim import AdamW8bit
            optim = AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
            print(f"[setup] using bitsandbytes AdamW8bit (~4x smaller optimizer state)")
        except Exception as _e:
            print(f"[warn] AdamW8bit failed ({_e}), fallback to plain AdamW")
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    elif args.cpu_offload_optimizer:
        try:
            from luma_stage0.optimizers import LumaMuonAdamWOptimizer, LumaOptimizerConfig
            opt_cfg = LumaOptimizerConfig(
                matrix_lr=args.lr * 3,
                scalar_lr=args.lr,
                weight_decay=0.01,
                betas=(0.9, 0.95),
            )
            optim = LumaMuonAdamWOptimizer(model, opt_cfg)
            optim.enable_cpu_offload()
            print(f"[setup] using LumaMuonAdamWOptimizer + cpu_offload (warning: Muon 破坏 Phase E ρ)")
        except Exception as _e:
            print(f"[warn] LumaMuonAdamWOptimizer failed ({_e}), fallback to plain AdamW")
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()

    # 训练循环
    losses: List[float] = []
    energy_end_traces: List[float] = []
    rho_history: List[float] = []
    skipped_steps: List[int] = []  # spike 检测跳过的 step 列表
    start_time = time.time()
    iter_idx = 0

    def get_batch():
        """数据来源: loader 或随机 token fallback."""
        nonlocal loader
        if loader is None:
            ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.max_seq_len), device=device, dtype=torch.long)
            return ids
        try:
            batch = next(data_iter)
        except (StopIteration, NameError):
            pass
        return None

    data_iter = iter(loader) if loader is not None else None

    while iter_idx < args.iters:
        try:
            if data_iter is not None:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                input_ids = batch.to(device)
            else:
                input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.max_seq_len), device=device, dtype=torch.long)
            labels = input_ids.clone()

            loss, logits, reason_aux = model(input_ids, labels)
            optim.zero_grad()
            loss.backward()
            # 简单梯度裁剪避免 AdamW 前期爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Phase 4 长训诊断修复 (v7+)：能量异常检测 → 跳过 optim.step
            # 避免 spike batch 的坏梯度被 AdamW variance buffer 累积，污染模型参数
            # 阈值：energy > 100x rolling median 的 step 被识别为 spike，不更新参数
            _energy_end_val = reason_aux.get("phase_e_energy_trace", [0])[-1]
            _skip_this_step = False
            if len(energy_end_traces) >= 50:
                # 最近 50 步 energy_end 的中位数作为 "healthy" baseline
                _sorted_recent = sorted(energy_end_traces[-50:])
                _median_recent = _sorted_recent[25]
                if _energy_end_val > 100 * _median_recent:
                    _skip_this_step = True
            if not _skip_this_step:
                optim.step()
            else:
                skipped_steps.append(iter_idx)

            losses.append(float(loss.item()))
            energy_end_traces.append(float(reason_aux["phase_e_energy_trace"][-1]))

            # 每 probe_interval 步测一次 rho_h_full
            # 用刚刚训练步的真实 h（从 compression 出来）而不是随机 h，
            # 这样 probe 反映的是实际动力学而非假想初始分布
            if iter_idx % args.probe_interval == 0 and iter_idx > 0:
                with torch.no_grad():
                    _h_probe_input = model.embedding(input_ids)
                    _out = model.compression(_h_probe_input)
                    _h_probe = _out[0] if isinstance(_out, tuple) else _out
                    _T_in = input_ids.shape[1]
                    _h_probe = _h_probe[:, -_T_in:, :].detach()
                    # Step 2+ 用真实 c_t（从 h_probe 生成），Step 1 用 0
                    _c_t_probe = model._compute_c_t(_h_probe).detach()
                with sdpa_kernel([SDPBackend.MATH]):
                    probes = model.reason_core.measure_phase_e_probes(
                        _h_probe, _c_t_probe, rel_eps=0.05, n_hutchinson=2,
                    )
                rho_history.append(float(probes.get("rho_h_full") or -1))

            if iter_idx % args.log_interval == 0:
                elapsed = time.time() - start_time
                tok_s = (iter_idx + 1) * args.batch_size * args.max_seq_len / elapsed
                last_rho = rho_history[-1] if rho_history else float("nan")
                c_t_norm = reason_aux.get("phase_e_c_t_norm", 0.0)
                wj = reason_aux.get("phase_e_world_jepa_loss", 0.0)
                wj_str = f" wj={wj:.3f}" if wj > 0 else ""
                print(
                    f"[{iter_idx:4d}/{args.iters}] loss={loss.item():.4f} "
                    f"energy_end={energy_end_traces[-1]:.1f} "
                    f"rho_h_full={last_rho:.4f} "
                    f"c_t_norm={c_t_norm:.3f}{wj_str} "
                    f"tok/s={tok_s:.0f} "
                    f"eta={(args.iters - iter_idx) * elapsed / max(iter_idx+1,1) / 60:.1f}min",
                    flush=True,
                )
            iter_idx += 1
        except KeyboardInterrupt:
            print("\n[interrupt] exiting early")
            break

    # 总结
    print("\n" + "=" * 60)
    print(f"[final] iters done: {iter_idx}")
    if len(losses) >= 10:
        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        print(f"[final] avg loss first 10 steps: {first_10:.4f}")
        print(f"[final] avg loss last 10 steps:  {last_10:.4f}")
        print(f"[final] delta: {last_10 - first_10:+.4f}")
        if last_10 < first_10 * 0.95:
            print("[final] ✅✅ loss dropped >5% — Phase E Step 1 END-TO-END validated")
        elif last_10 < first_10:
            print("[final] ✅ loss dropped, but small — Phase E works, needs longer run")
        else:
            print("[final] ⚠️ loss did not drop — investigate")
    if rho_history:
        print(f"[final] rho_h_full trajectory: {[f'{r:.3f}' for r in rho_history[-5:]]}")
    print(f"[final] wall time: {(time.time()-start_time)/60:.1f}min")

    # 落盘 metrics
    out_dir = _MINIMIND_ROOT / "artifacts" / "phase_e"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{args.save_weight}.json"
    print(f"[final] skipped_steps: {len(skipped_steps)} (at: {skipped_steps[:10]}{'...' if len(skipped_steps)>10 else ''})")
    with open(metrics_path, "w") as f:
        json.dump({
            "iters": iter_idx,
            "losses": losses,
            "energy_end_traces": energy_end_traces,
            "rho_history": rho_history,
            "skipped_steps": skipped_steps,
            "config": {
                "K_max": cfg.phase_e_K_max,
                "eta": cfg.phase_e_eta,
                "hidden_size": cfg.hidden_size,
                "reason_shared_depth": cfg.reason_shared_depth,
                "batch_size": args.batch_size,
                "max_seq_len": args.max_seq_len,
                "lr": args.lr,
            },
        }, f, indent=2)
    print(f"[final] metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
