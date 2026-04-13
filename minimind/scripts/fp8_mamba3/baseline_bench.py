"""Phase 0.2 + 0.3: Mamba3 MIMO bf16 基准测试

记录当前 bf16 Mamba3Block 的：
1. 前向 output 统计（mean/std/max/min）
2. 反向梯度 L2 范数
3. Peak activation memory

作为后续 FP8 改造的对照组。
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from model.mamba3_module import Mamba3Block, Mamba3Config

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "fp8_mamba3"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def run_bench(d_model: int, seq_len: int, batch: int, *, is_mimo: bool,
              name: str) -> dict:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = Mamba3Config(
        d_model=d_model,
        d_state=192,
        expand=2,
        headdim=64,
        chunk_size=32,
        is_mimo=is_mimo,
        mimo_rank=2,
        dropout=0.0,
        use_gradient_checkpointing=False,
    )
    block = Mamba3Block(cfg).cuda().to(torch.bfloat16)
    block.train()

    x = torch.randn(batch, seq_len, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    baseline_mem = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = block(x)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000

    peak_after_fwd = torch.cuda.max_memory_allocated()

    loss = y.float().pow(2).sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - t0) * 1000

    peak_after_bwd = torch.cuda.max_memory_allocated()

    # 统计前向输出
    y_f = y.detach().float()
    # 梯度范数
    grad_l2 = {}
    for pname, p in block.named_parameters():
        if p.grad is not None:
            grad_l2[pname[:50]] = float(p.grad.norm().item())
    x_grad_norm = float(x.grad.norm().item()) if x.grad is not None else None

    result = {
        "name": name,
        "config": {
            "d_model": d_model,
            "d_state": cfg.d_state,
            "headdim": cfg.headdim,
            "expand": cfg.expand,
            "mimo_rank": cfg.mimo_rank if is_mimo else 1,
            "is_mimo": is_mimo,
            "chunk_size": cfg.chunk_size,
            "seq_len": seq_len,
            "batch": batch,
            "dtype": "bf16",
        },
        "output_stats": {
            "shape": list(y_f.shape),
            "mean": float(y_f.mean().item()),
            "std": float(y_f.std().item()),
            "max": float(y_f.max().item()),
            "min": float(y_f.min().item()),
            "abs_mean": float(y_f.abs().mean().item()),
            "abs_p99": float(y_f.abs().flatten().kthvalue(int(y_f.numel() * 0.99)).values.item()),
        },
        "grad_norms": {
            "x_grad": x_grad_norm,
            "param_grad_sum": float(sum(grad_l2.values())),
            "param_grad_max": float(max(grad_l2.values())) if grad_l2 else 0.0,
            "top3": sorted(grad_l2.items(), key=lambda kv: -kv[1])[:3],
        },
        "memory_mb": {
            "baseline": baseline_mem / 1e6,
            "peak_after_fwd": peak_after_fwd / 1e6,
            "peak_after_bwd": peak_after_bwd / 1e6,
            "activation_delta": (peak_after_fwd - baseline_mem) / 1e6,
        },
        "timing_ms": {
            "fwd": fwd_ms,
            "bwd": bwd_ms,
        },
    }
    return result


def main() -> None:
    results = []
    configs = [
        dict(d_model=768, seq_len=1024, batch=1, is_mimo=True, name="mimo_seq1024"),
        dict(d_model=768, seq_len=2048, batch=1, is_mimo=True, name="mimo_seq2048"),
        dict(d_model=768, seq_len=1024, batch=1, is_mimo=False, name="siso_seq1024"),
    ]
    for cfg in configs:
        try:
            torch.cuda.empty_cache()
            r = run_bench(**cfg)
            print(f"✅ {r['name']}: fwd={r['timing_ms']['fwd']:.1f}ms bwd={r['timing_ms']['bwd']:.1f}ms "
                  f"activation={r['memory_mb']['activation_delta']:.1f}MB "
                  f"out_std={r['output_stats']['std']:.4f} abs_p99={r['output_stats']['abs_p99']:.3f}")
            results.append(r)
        except Exception as e:
            print(f"❌ {cfg['name']}: {type(e).__name__}: {e}")
            results.append({"name": cfg["name"], "error": str(e)})

    out_path = ARTIFACTS / "bf16_baseline_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n📝 Saved to {out_path}")


if __name__ == "__main__":
    main()
