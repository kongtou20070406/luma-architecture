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

    def _run_mamba(self, x: torch.Tensor) -> torch.Tensor:
        """Run the appropriate Mamba path (SISO or MIMO with fallback)."""
        if self.training and self.mamba_siso is not None:
            return self.mamba_siso(x)
        elif self._mimo_disabled_runtime and self.mamba_siso is not None:
            return self.mamba_siso(x)
        else:
            try:
                return self.mamba(x)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        # Gradient checkpointing: recompute Mamba forward during backward to save VRAM.
        # Without this, SISO backward buffers accumulate to >3GB at seq>=2048.
        # Must use use_reentrant=True because Mamba3 backward unpacks saved_tensors multiple times.
        if self.cfg.use_gradient_checkpointing and self.training:
            x = torch_checkpoint(self._run_mamba, x, use_reentrant=True)
        else:
            x = self._run_mamba(x)
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
