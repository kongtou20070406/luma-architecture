import torch
from torch import nn
from dataclasses import dataclass
import warnings

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
    mimo_rank: int = 4
    chunk_size: int = 64
    dropout: float = 0.0
    # Workaround for current tilelang backward smem limit on RTX 5090.
    # If True: training uses SISO path, eval/inference keeps MIMO path.
    train_use_siso_fallback: bool = True
    # If True: when MIMO path hits runtime kernel limits, auto fallback to SISO.
    auto_fallback_on_mimo_error: bool = True


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
        if cfg.is_mimo and cfg.train_use_siso_fallback:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        # Current tilelang MIMO kernels can exceed dynamic shared-memory limit on 5090.
        # Keep train path stable with SISO and auto-fallback when runtime kernel limits are hit.
        if self.training and self.mamba_siso is not None:
            x = self.mamba_siso(x)
        elif self._mimo_disabled_runtime and self.mamba_siso is not None:
            x = self.mamba_siso(x)
        else:
            try:
                x = self.mamba(x)
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
                warnings.warn(
                    "MIMO kernel failed at runtime; falling back to SISO path for stability.",
                    RuntimeWarning,
                )
                x = self.mamba_siso(x)
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
