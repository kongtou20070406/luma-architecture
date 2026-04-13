"""Luma separates bold matrix motion from delicate control updates, and now she can optionally store both kinds of optimizer memory in 8-bit form.

Luma 会把大胆前进的大矩阵更新和精细控制参数更新分开处理，现在也可以选择把两类优化器状态都压成 8-bit 形式保存与运行。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

try:
    from bitsandbytes.optim import AdamW8bit
except Exception:  # pragma: no cover
    AdamW8bit = None

try:
    from muon import SingleDeviceMuon, SingleDeviceMuonWithAuxAdam, adam_update, muon_update
except Exception:  # pragma: no cover
    SingleDeviceMuon = None
    SingleDeviceMuonWithAuxAdam = None
    adam_update = None
    muon_update = None


CONTROL_TENSOR_NAME_PATTERNS = (
    "norm",
    "bias",
    "scale",
    "embedding",
    "embed_table",
    "lm_head",
    "hebb",           # hebb_proj_h/c + hebb_out: AdamW 控制增长，surprise>0 保证有梯度
    # 用行范数归一化控制增长，不用 weight decay
)

FORCE_ADAMW_PARAM_SUBSTRINGS = (
    "ct_injection.proj.weight",   # W_c: 显式走 AdamW，避免 Muon 对时变/定向梯度的放大。
    "c_t_head.weight",            # c_t_head: 自省投影层，显式走 AdamW 便于控制 c_t 范数增长。
    "h_mask_predictor.weight",    # h_mask_predictor: 直接驱动 c_t 的辅助头，显式走 AdamW。
    # Loop LoRA 参数：Muon 正交化让 wd 失效 → LoRA 权重在长训练中单调增长 →
    # F_k 的 Jacobian 扰动线性放大 → rho_h_frozen 后期越过 1。
    # V5b (rank=0) ablation 证实 LoRA 是 rho_h 恶化的主因（rho_h_frozen p95 从 1.62 降到 0.90）。
    "lora_A.weight",              # per-loop LoRA A 矩阵 (nn.Embedding.weight)
    "lora_B.weight",              # per-loop LoRA B 矩阵
    "lora_coeff_proj.weight",     # ct-conditioned LoRA 系数投影
    "lora_shared_A",              # 共享 LoRA A 参数（Hypernet lite）
    "lora_shared_B",              # 共享 LoRA B 参数
)


@dataclass
class LumaOptimizerConfig:
    """Luma writes optimizer choices down explicitly, so later agents can resume the exact same training temperament.

    Luma 会把优化器选择明确写进配置，方便后续 agent 在恢复训练时延续同一种“更新气质”。
    """

    matrix_lr: float = 0.02
    scalar_lr: float = 3e-4
    muon_momentum: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    muon_clip_factor: float = 1.0
    modular_norm_power: float = 0.5
    use_8bit_muon: bool = False
    use_8bit_adamw: bool = False


def _modular_norm_scale(name: str, param: torch.nn.Parameter, power: float) -> float:
    """Luma uses a lightweight Modular-Norm-style scale so width/depth changes do not wildly distort the step size.

    Luma 用一个轻量的 Modular-Norm 风格缩放来平衡不同模块的学习率，减少宽度或深度变化造成的步长失真。
    """

    if param.ndim < 2:
        return 1.0
    fan_out, fan_in = param.shape[0], param.shape[1]
    shape_scale = max(fan_in, fan_out) ** (-power)
    if "reason_core" in name:
        return shape_scale * 1.10
    if "world_latent_jepa" in name or "introspection" in name or "self_jepa" in name:
        return shape_scale * 0.90
    return shape_scale


def _apply_muon_clip_(parameters: Iterable[torch.nn.Parameter], clip_factor: float) -> None:
    """MuonClip here means clipping each gradient by a scale proportional to its own RMS, not a global hard ceiling.

    这里的 MuonClip 指的是按每个梯度自身 RMS 比例裁剪，而不是给所有参数套一个统一硬上限。
    """

    if clip_factor <= 0:
        return
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad
        rms = grad.float().pow(2).mean().sqrt()
        clip_value = (rms * clip_factor).clamp_min(1e-8)
        grad.clamp_(min=-clip_value, max=clip_value)


def _quantize_symmetric_8bit(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Luma keeps 8-bit state simple: symmetric per-tensor quantization with an explicit scale.

    Luma 把 8-bit 状态管理做成最朴素可靠的版本：按张量对称量化，并显式保存 scale。
    """

    source = tensor.detach().float()
    max_abs = source.abs().max().clamp_min(1e-8)
    scale = (max_abs / 127.0).to(source.device)
    quantized = torch.clamp(torch.round(source / scale), -127, 127).to(torch.int8)
    return quantized, scale


def _dequantize_symmetric_8bit(quantized: torch.Tensor, scale: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Luma restores a float working buffer only for the duration of a step, then compresses it again.

    Luma 只在更新步里临时还原浮点工作缓冲区，完成后就再次压回 8-bit。
    """

    return quantized.to(dtype=torch.float32) * scale.to(device=like.device, dtype=torch.float32)


class Luma8BitMuon(torch.optim.Optimizer):
    """This is an experimental but real 8-bit Muon runtime: the momentum buffer itself lives in quantized form between steps.

    这是一个实验性的、但运行时真实存在的 8-bit Muon：它的 momentum buffer 会在两次更新之间以量化形式常驻。
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.1, momentum: float = 0.95):
        if muon_update is None:
            raise RuntimeError("muon package is unavailable; install `muon-optimizer` first.")
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    grad = torch.zeros_like(param)
                state = self.state[param]
                if len(state) == 0:
                    momentum = torch.zeros_like(param, dtype=torch.float32)
                else:
                    momentum = _dequantize_symmetric_8bit(state["momentum_q"], state["momentum_scale"], param)
                update = muon_update(grad.float(), momentum, beta=group["momentum"])
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update.to(dtype=param.dtype), alpha=-group["lr"])
                momentum_q, momentum_scale = _quantize_symmetric_8bit(momentum)
                state["momentum_q"] = momentum_q
                state["momentum_scale"] = momentum_scale


class LumaOptimizerBundle:
    """Luma keeps Muon and Adam as a single training handle, even when their internal state formats differ.

    Luma 把 Muon 和 Adam 继续包装成一个统一训练句柄，即使它们的内部状态格式并不相同。
    """

    def __init__(self, model: torch.nn.Module, config: LumaOptimizerConfig | None = None):
        self.config = config or LumaOptimizerConfig()
        matrix_params, scalar_groups = self._split_params(model)
        self.matrix_optimizer = self._build_matrix_optimizer(matrix_params)
        self.scalar_optimizer = self._build_scalar_optimizer(scalar_groups)
        self.param_groups = self.matrix_optimizer.param_groups + self.scalar_optimizer.param_groups
        for group in self.matrix_optimizer.param_groups:
            group["optim_family"] = "muon"
        for group in self.scalar_optimizer.param_groups:
            group["optim_family"] = "adamw"
        self.routing_summary = getattr(self, "routing_summary", {"muon": [], "adamw": []})

    def _split_params(self, model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[dict]]:
        matrix_params: list[torch.nn.Parameter] = []
        scalar_groups: list[dict] = []
        routing_summary = {"muon": [], "adamw": []}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            scale = _modular_norm_scale(name, param, self.config.modular_norm_power)
            force_adamw = any(pattern in name for pattern in FORCE_ADAMW_PARAM_SUBSTRINGS)
            if param.ndim >= 2 and not force_adamw and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
                matrix_params.append(param)
                routing_summary["muon"].append(name)
            else:
                # 低 wd 白名单：zero-init 或梯度偏小的参数（如 hebb/lora/h_mask_predictor），
                # 默认 wd=0.1 会把零初始化的 B 矩阵压成 0，导致模块彻底失效。
                _low_wd_hit = ("hebb" in name) or ("lora" in name) or ("h_mask_predictor" in name)
                wd = 0.01 if _low_wd_hit else self.config.weight_decay
                scalar_groups.append(
                    dict(
                        params=[param],
                        lr=self.config.scalar_lr * scale,
                        betas=self.config.betas,
                        eps=self.config.eps,
                        weight_decay=wd,
                    )
                )
                routing_summary["adamw"].append(name)
        self.routing_summary = routing_summary
        return matrix_params, scalar_groups

    def _build_matrix_optimizer(self, matrix_params: list[torch.nn.Parameter]):
        if self.config.use_8bit_muon:
            return Luma8BitMuon(
                matrix_params,
                lr=self.config.matrix_lr,
                weight_decay=self.config.weight_decay,
                momentum=self.config.muon_momentum,
            )
        if SingleDeviceMuon is None:
            raise RuntimeError("muon package is unavailable; install `muon-optimizer` first.")
        return SingleDeviceMuon(
            matrix_params,
            lr=self.config.matrix_lr,
            weight_decay=self.config.weight_decay,
            momentum=self.config.muon_momentum,
        )

    def _build_scalar_optimizer(self, scalar_groups: list[dict]):
        if self.config.use_8bit_adamw:
            if AdamW8bit is None:
                raise RuntimeError("bitsandbytes is unavailable; install `bitsandbytes` first.")
            return AdamW8bit(scalar_groups)
        return torch.optim.AdamW(scalar_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.matrix_optimizer.zero_grad(set_to_none=set_to_none)
        self.scalar_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        matrix_params = [p for group in self.matrix_optimizer.param_groups for p in group["params"]]
        scalar_params = [p for group in self.scalar_optimizer.param_groups for p in group["params"]]
        for param in matrix_params:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
        _apply_muon_clip_(matrix_params + scalar_params, self.config.muon_clip_factor)
        if self._cpu_offload:
            self._states_to_gpu()
        self.matrix_optimizer.step()
        self.scalar_optimizer.step()
        if self._cpu_offload:
            self._states_to_cpu()

    # ── CPU Offload ──────────────────────────────────────────────────────
    _cpu_offload: bool = False

    def enable_cpu_offload(self) -> None:
        """将优化器状态（momentum / exp_avg / exp_avg_sq）卸载到 CPU pinned memory。

        首次 step() 后优化器才会初始化状态，所以 offload 延迟到第一次 step 完成后生效。
        预计节省 ~2.4GB VRAM（660M 模型），代价 ~10-15ms/step（PCIe 5.0）。
        """
        self._cpu_offload = True

    @torch.no_grad()
    def _states_to_cpu(self) -> None:
        for opt in (self.matrix_optimizer, self.scalar_optimizer):
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        state[k] = v.to("cpu", non_blocking=True).pin_memory()
        torch.cuda.synchronize()

    @torch.no_grad()
    def _states_to_gpu(self) -> None:
        device = None
        for opt in (self.matrix_optimizer, self.scalar_optimizer):
            for param_key, state in opt.state.items():
                if device is None:
                    # 从参数推断 GPU device
                    if isinstance(param_key, torch.Tensor):
                        device = param_key.device
                    else:
                        for g in opt.param_groups:
                            for p in g["params"]:
                                if p.is_cuda:
                                    device = p.device
                                    break
                            if device is not None:
                                break
                if device is None:
                    device = torch.device("cuda")
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and not v.is_cuda:
                        state[k] = v.to(device, non_blocking=True)
        torch.cuda.synchronize()

    def state_dict(self) -> dict:
        # 保存前确保状态在 CPU（已经在 CPU 则是 no-op）
        if self._cpu_offload:
            # state 已在 CPU，直接保存
            pass
        return {
            "config": asdict(self.config),
            "matrix_optimizer": self.matrix_optimizer.state_dict(),
            "scalar_optimizer": self.scalar_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.matrix_optimizer.load_state_dict(state_dict["matrix_optimizer"])
        self.scalar_optimizer.load_state_dict(state_dict["scalar_optimizer"])


class LumaMuonAdamWOptimizer(LumaOptimizerBundle):
    """This name stays for compatibility with existing stage scripts, but it now delegates to the more explicit optimizer bundle.

    这个旧名字为了兼容已有脚本而保留，但内部已经委托给更明确的优化器 bundle。
    """

    pass


class LumaCosineScheduler:
    """Luma keeps the scheduler explicit and serializable so checkpoint/resume can recover the exact same pulse.

    Luma 把 scheduler 写成显式且可序列化的对象，方便 checkpoint/resume 后恢复完全一致的节奏。
    """

    def __init__(
        self,
        optimizer: LumaOptimizerBundle,
        total_steps: int,
        matrix_base_lr: float,
        scalar_base_lr: float,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.matrix_base_lr = matrix_base_lr
        self.scalar_base_lr = scalar_base_lr
        self.min_lr_ratio = min_lr_ratio
        self.step_count = 0
        self._refresh_lrs()

    def _cosine_factor(self) -> float:
        progress = min(self.step_count / self.total_steps, 1.0)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    def _refresh_lrs(self) -> None:
        factor = self._cosine_factor()
        for group in self.optimizer.param_groups:
            if group.get("optim_family") == "muon":
                group["lr"] = self.matrix_base_lr * factor
            else:
                base_lr = group.get("base_lr", group["lr"])
                if "base_lr" not in group:
                    group["base_lr"] = base_lr
                # Scalar groups already include Modular-Norm scaling inside their base lr.
                # 标量组的 base lr 已经内含 Modular-Norm 缩放，这里只叠加余弦因子。
                group["lr"] = group["base_lr"] * factor

    def step(self) -> None:
        self.step_count += 1
        self._refresh_lrs()

    def state_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "matrix_base_lr": self.matrix_base_lr,
            "scalar_base_lr": self.scalar_base_lr,
            "min_lr_ratio": self.min_lr_ratio,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.total_steps = state_dict["total_steps"]
        self.matrix_base_lr = state_dict["matrix_base_lr"]
        self.scalar_base_lr = state_dict["scalar_base_lr"]
        self.min_lr_ratio = state_dict["min_lr_ratio"]
        self.step_count = state_dict["step_count"]
        self._refresh_lrs()
