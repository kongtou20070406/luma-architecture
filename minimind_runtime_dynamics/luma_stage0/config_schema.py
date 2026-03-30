"""Luma keeps her early contracts here so later agents can build without guessing.

This schema freezes the shapes and defaults of stage-0 components before the
full architecture is wired together.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class CompressionZoneConfig:
    d_model: int = 768
    d_state: int = 192
    groups: int = 4
    layers_per_group: int = 6
    mamba_layers_per_group: int = 5
    retrieval_layers_per_group: int = 1
    ffn_dim: int = 3072
    block_repr_every_n_layers: int = 3
    block_repr_count: int = 8
    swa_window: int = 1024
    kda_no_fox: bool = True
    mamba_mimo_enabled: bool = True
    mamba_complex_state_enabled: bool = True
    mamba_post_gate_rmsnorm_required: bool = True


@dataclass
class ReasonLoopConfig:
    loops_min: int = 1
    loops_max: int = 8
    slow_k: int = 2
    meta_dim: int = 96
    c_t_dim: int = 64
    reason_ffn_dim: int = 3072
    swa_window: int = 1024
    exit_threshold: float = 0.85
    know_gap_threshold: float = 0.7
    router_dim: int = 256


@dataclass
class MHCConfig:
    n_streams: int = 4
    apply_zone: str = "reason_loop_only"
    sinkhorn_iters: int = 20
    alpha_init: float = 0.01
    state_layout: str = "B_T_N_C"
    dynamic_maps_from: str = "current_stream_state"
    residual_constraint: str = "birkhoff_doubly_stochastic"


@dataclass
class LossConfig:
    lm_weight: float = 1.0
    world_weight: float = 0.1
    self_weight: float = 0.03
    rollout_weight: float = 0.01
    residual_reg_weight: float = 0.001
    rollout_steps: int = 2
    self_target: str = "delta_c_t"
    world_target: str = "masked_latent_prediction"


@dataclass
class GateConfig:
    stagnation_k: int = 3
    delta_h_threshold: float = 0.015
    c_t_cos_low: float = 0.70
    c_t_cos_high: float = 0.95


@dataclass
class Stage0HarnessConfig:
    seed: int = 42
    batch_size: int = 1
    seq_len: int = 128
    device: str = "cuda"
    dtype: str = "bfloat16"
    out_metrics_file: str = "artifacts/stage0_metrics.jsonl"


def build_default_config() -> Dict[str, Any]:
    return {
        "compression_zone": asdict(CompressionZoneConfig()),
        "reason_loop": asdict(ReasonLoopConfig()),
        "mhc": asdict(MHCConfig()),
        "loss": asdict(LossConfig()),
        "gate": asdict(GateConfig()),
        "stage0_harness": asdict(Stage0HarnessConfig()),
    }
