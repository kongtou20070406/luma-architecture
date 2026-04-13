import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from dataclasses import dataclass
import warnings
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAMBA_LOCAL = _REPO_ROOT / "third_party" / "mamba-official"
if _MAMBA_LOCAL.exists() and str(_MAMBA_LOCAL) not in sys.path:
    sys.path.insert(0, str(_MAMBA_LOCAL))
from mamba_ssm.modules.mamba3 import Mamba3

# FP8 activation cache 支持（saved_tensors_hooks + per-block 量化）
# 不改 triton kernel，只压缩保存给 backward 的激活
_FP8_SCRIPT = _REPO_ROOT / "minimind" / "scripts" / "fp8_mamba3"
if _FP8_SCRIPT.exists() and str(_FP8_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_FP8_SCRIPT))
try:
    from fp8_saved_tensors import Fp8ActivationContext  # noqa: E402
    _FP8_HOOK_AVAILABLE = True
except Exception:
    _FP8_HOOK_AVAILABLE = False


class ZCRMSNorm(nn.Module):
    """Zero-centered RMSNorm: y = (1 + scale) * RMSNorm(x)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y * (1.0 + self.scale)


@dataclass
class Mamba3Config:
    d_model: int = 768
    d_state: int = 192
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 0.5
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    is_outproj_norm: bool = False
    is_mimo: bool = False
    mimo_rank: int = 2  # 4→2: smem = (chunk_size*rank)² must fit RTX 5090's 101KB limit
    chunk_size: int = 32  # rank=2 × chunk=32 = 64 (at design limit)
    dropout: float = 0.0
    # Workaround for current tilelang backward smem limit on RTX 5090.
    # With rank=2 chunk=16, MIMO should fit. Keep fallback as safety net.
    train_use_siso_fallback: bool = False  # MIMO should work with rank=2 chunk=32; keep SISO fallback via auto_fallback
    # If True: when MIMO path hits runtime kernel limits, auto fallback to SISO.
    auto_fallback_on_mimo_error: bool = True
    # Gradient checkpointing for Mamba3: recompute forward during backward to save VRAM.
    # Critical for seq>=2048 where backward buffers accumulate to >3GB.
    use_gradient_checkpointing: bool = False
    # FP8 activation cache: pack saved_tensors 到 fp8+scale，backward 前 dequantize 回 bf16。
    # 不改 triton kernel (compute 仍 bf16)，只压缩激活存储。
    # ~48% Mamba activation 内存节省，~2% 相对数值误差。
    use_fp8_activation_cache: bool = False
    fp8_act_block_size: int = 128


class Mamba3Block(nn.Module):
    """
    Thin wrapper around official Mamba-3 implementation from state-spaces/mamba (mamba3-release).
    This path uses official fused kernels and update rules (including exp-trapezoid discretization and
    complex-state machinery inside the upstream module).
    """

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        self.pre_norm = ZCRMSNorm(cfg.d_model)
        self.mamba = Mamba3(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            expand=cfg.expand,
            headdim=cfg.headdim,
            ngroups=cfg.ngroups,
            rope_fraction=cfg.rope_fraction,
            dt_min=cfg.dt_min,
            dt_max=cfg.dt_max,
            dt_init_floor=cfg.dt_init_floor,
            A_floor=cfg.A_floor,
            is_outproj_norm=cfg.is_outproj_norm,
            is_mimo=cfg.is_mimo,
            mimo_rank=cfg.mimo_rank,
            chunk_size=cfg.chunk_size,
            dropout=cfg.dropout,
        )
        self.mamba_siso = None
        if cfg.is_mimo and (cfg.train_use_siso_fallback or cfg.auto_fallback_on_mimo_error):
            self.mamba_siso = Mamba3(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                expand=cfg.expand,
                headdim=cfg.headdim,
                ngroups=cfg.ngroups,
                rope_fraction=cfg.rope_fraction,
                dt_min=cfg.dt_min,
                dt_max=cfg.dt_max,
                dt_init_floor=cfg.dt_init_floor,
                A_floor=cfg.A_floor,
                is_outproj_norm=cfg.is_outproj_norm,
                is_mimo=False,
                mimo_rank=1,
                chunk_size=cfg.chunk_size,
                dropout=cfg.dropout,
            )
        self.post_norm = ZCRMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self._mimo_disabled_runtime = False

    def _run_mamba(self, x: torch.Tensor, dt_external_bias=None) -> torch.Tensor:
        """Run the appropriate Mamba path (SISO or MIMO with fallback)."""
        if self.training and self.mamba_siso is not None:
            return self.mamba_siso(x, dt_external_bias=dt_external_bias)
        elif self._mimo_disabled_runtime and self.mamba_siso is not None:
            return self.mamba_siso(x, dt_external_bias=dt_external_bias)
        else:
            try:
                return self.mamba(x, dt_external_bias=dt_external_bias)
            except Exception as err:
                is_mimo_error = (
                    self.cfg.auto_fallback_on_mimo_error
                    and self.mamba_siso is not None
                    and (
                        "dynamic shared memory size" in str(err)
                        or "TileLang" in str(err)
                        or "tvm.error.InternalError" in str(err)
                    )
                )
                if not is_mimo_error:
                    raise
                self._mimo_disabled_runtime = True
                msg = (
                    f"[CRITICAL] MIMO→SISO FALLBACK TRIGGERED! "
                    f"Error: {err}. "
                    f"Config: rank={self.cfg.mimo_rank}, chunk={self.cfg.chunk_size}, "
                    f"product={self.cfg.mimo_rank * self.cfg.chunk_size}. "
                    f"All subsequent forward passes will use SISO. "
                    f"Fix: reduce mimo_rank or chunk_size so rank*chunk<=64."
                )
                warnings.warn(msg, RuntimeWarning)
                print(f"\n{'='*60}\n{msg}\n{'='*60}\n", flush=True)
                return self.mamba_siso(x)

    def _run_mamba_maybe_fp8(self, x: torch.Tensor, dt_external_bias=None) -> torch.Tensor:
        """Run _run_mamba, optionally wrapping its saved activations in FP8."""
        if (
            self.cfg.use_fp8_activation_cache
            and self.training
            and _FP8_HOOK_AVAILABLE
        ):
            # saved_tensors_hooks 只作用于当前 context 内的 save_for_backward 调用
            with Fp8ActivationContext(block_size=self.cfg.fp8_act_block_size):
                return self._run_mamba(x, dt_external_bias=dt_external_bias)
        return self._run_mamba(x, dt_external_bias=dt_external_bias)

    def forward(self, x: torch.Tensor, dt_external_bias=None) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        if self.cfg.use_gradient_checkpointing and self.training:
            if dt_external_bias is not None:
                # dt_external_bias 需要 detach 后重新 requires_grad，避免 reentrant checkpoint 冲突
                x = self._run_mamba_maybe_fp8(x, dt_external_bias=dt_external_bias)
            else:
                # 根本限制：Mamba triton kernel 的 ctx.saved_tensors 在 backward 中会被
                # 多次访问，而新 pytorch non-reentrant ckpt 的 saved_tensors_hooks 只允许
                # 单次 unpack（会触发 CheckpointError: already unpacked once）。
                # 因此 compression zone 要使用 Mamba + gradient checkpointing 时只能用
                # use_reentrant=True 路径；但这和 Phase E 主循环里任何 torch.autograd.grad
                # 调用全局冲突（所以 Phase E 训练必须关 use_gradient_checkpointing=0，
                # 用 activation_offload_compress 作为替代内存策略）。
                x = torch_checkpoint(self._run_mamba_maybe_fp8, x, use_reentrant=True)
        else:
            x = self._run_mamba_maybe_fp8(x, dt_external_bias=dt_external_bias)
        x = self.post_norm(x)
        x = self.dropout(x)
        return residual + x


class Mamba3Stack(nn.Module):
    def __init__(self, cfg: Mamba3Config, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([Mamba3Block(cfg) for _ in range(num_layers)])
        self.final_norm = ZCRMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
