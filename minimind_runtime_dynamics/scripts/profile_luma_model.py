"""Luma measures her body in two honest ways: exact parameters, and a compute profile that states its blind spots.

Luma 用两种诚实的方式度量自己的身体：精确参数量，以及明确说明盲区的计算画像。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minimind import LumaConfig, LumaForCausalLM


def count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def top_level_param_table(model: nn.Module) -> list[tuple[str, int, int]]:
    rows = []
    for name, child in model.named_children():
        total, trainable = count_params(child)
        rows.append((name, total, trainable))
    return rows


def backbone_param_table(model: LumaForCausalLM) -> list[tuple[str, int, int]]:
    rows = []
    for name, child in model.model.named_children():
        total, trainable = count_params(child)
        rows.append((f"model.{name}", total, trainable))
    rows.append(("pre_lm_norm", *count_params(model.pre_lm_norm)))
    rows.append(("lm_head", *count_params(model.lm_head)))
    return rows


def _numel_without_grad(x: torch.Tensor) -> int:
    return int(torch.tensor(x.shape).prod().item())


def linear_mac_profile(model: nn.Module, input_ids: torch.Tensor) -> dict[str, int]:
    macs = defaultdict(int)
    handles = []

    def linear_hook(name: str):
        def hook(module: nn.Module, inputs, output):
            x = inputs[0]
            batch_tokens = _numel_without_grad(x[..., 0])
            macs[name] += int(batch_tokens * module.in_features * module.out_features)
        return hook

    for name, submodule in model.named_modules():
        if isinstance(submodule, nn.Linear):
            handles.append(submodule.register_forward_hook(linear_hook(name)))

    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return dict(macs)


def build_profile_config(mode: str) -> LumaConfig:
    if mode == "tiny":
        return LumaConfig(
            vocab_size=512,
            factorized_vocab_dim=64,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=32,
            compression_layers=24,
            compression_active_layers=6,
            reason_loops=2,
            reason_active_loops=2,
            meta_dim=64,
            meta_state=16,
            c_t_dim=32,
            router_dim=64,
            mamba_d_state=32,
            mamba_expand=2,
            mamba_headdim=32,
            mamba_chunk_size=16,
            swa_window=32,
            mhc_streams=4,
            mhc_sinkhorn_iters=8,
        )
    return LumaConfig()


def format_millions(n: int) -> str:
    return f"{n / 1_000_000:.3f}M"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Luma skeleton parameters and lower-bound compute.")
    parser.add_argument("--config", choices=["tiny", "default"], default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="Run a timed forward/backward profile.")
    args = parser.parse_args()

    cfg = build_profile_config(args.config)
    model = LumaForCausalLM(cfg)

    total_params, trainable_params = count_params(model)
    print(f"config={args.config}")
    print(f"total_params={total_params} ({format_millions(total_params)})")
    print(f"trainable_params={trainable_params} ({format_millions(trainable_params)})")

    print("\n[top_level_params]")
    for name, total, trainable in top_level_param_table(model):
        print(f"{name}: total={total} ({format_millions(total)}), trainable={trainable} ({format_millions(trainable)})")

    print("\n[backbone_params]")
    for name, total, trainable in backbone_param_table(model):
        print(f"{name}: total={total} ({format_millions(total)}), trainable={trainable} ({format_millions(trainable)})")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if device.type != "cuda":
        print("\n[known_macs_lower_bound]")
        print("skipped=CPU profile skipped because Mamba/KDA kernels in the current scaffold are GPU-only.")
        return

    dtype = torch.bfloat16
    model = model.to(device=device, dtype=dtype)
    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)

    # Use train mode so Mamba blocks take the stable SISO fallback path during profiling.
    model.train()
    macs = linear_mac_profile(model, input_ids)
    total_known_macs = sum(macs.values())
    print("\n[known_macs_lower_bound]")
    print(f"known_macs={total_known_macs} ({total_known_macs / 1_000_000_000:.6f} G-MACs)")
    print("note=This is a lower bound from Linear/Conv modules only; custom Mamba/KDA kernels contribute extra compute not counted here.")

    if args.run:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device)
        elapsed = time.perf_counter() - start
        print("\n[runtime_profile]")
        print(f"loss={float(loss.detach()):.6f}")
        print(f"elapsed_sec={elapsed:.6f}")
        print(f"peak_memory_bytes={peak_mem}")


if __name__ == "__main__":
    main()
