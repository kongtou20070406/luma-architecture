"""Phase 1.3 smoke: Mamba3Block bf16 vs FP8 activation cache 对比。

测量:
- forward 输出数值差（max abs diff）
- backward grad 差
- Peak activation memory 节省
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from model.mamba3_module import Mamba3Block, Mamba3Config

def run(fp8: bool, *, d_model=768, seq_len=2048, batch=1, is_mimo=True, seed=42,
        grad_ckpt: bool = False) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = Mamba3Config(
        d_model=d_model, d_state=192, expand=2, headdim=64,
        chunk_size=32, is_mimo=is_mimo, mimo_rank=2, dropout=0.0,
        use_gradient_checkpointing=grad_ckpt,
        use_fp8_activation_cache=fp8, fp8_act_block_size=128,
    )
    block = Mamba3Block(cfg).cuda().to(torch.bfloat16)
    block.train()

    # 固定输入（同 seed 保证两次对比一致）
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    mem_before = torch.cuda.memory_allocated()
    y = block(x)
    torch.cuda.synchronize()
    peak_fwd = torch.cuda.max_memory_allocated()

    loss = y.float().pow(2).sum()
    loss.backward()
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated()

    return {
        "fp8": fp8,
        "y_sample": y.detach().float().abs().mean().item(),
        "y_std": y.detach().float().std().item(),
        "loss": float(loss.detach().item()),
        "x_grad_norm": float(x.grad.norm().item()),
        "mem_peak_fwd_MB": (peak_fwd - mem_before) / 1e6,
        "mem_peak_bwd_MB": (peak_bwd - mem_before) / 1e6,
    }


def main() -> None:
    print("=" * 60)
    print("Phase 1.3 smoke: Mamba3Block bf16 vs FP8 activation cache")
    print("=" * 60)
    print("\n--- BF16 baseline (no grad_ckpt) ---")
    bf16 = run(fp8=False)
    for k, v in bf16.items():
        print(f"  {k}: {v}")

    print("\n--- FP8 activation cache (no grad_ckpt) ---")
    fp8 = run(fp8=True)
    for k, v in fp8.items():
        print(f"  {k}: {v}")

    print("\n--- FP8 + grad_ckpt COMBO ---")
    try:
        combo = run(fp8=True, grad_ckpt=True)
        for k, v in combo.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        combo = None

    print("\n--- Diff ---")
    print(f"  y abs_mean diff: {abs(bf16['y_sample'] - fp8['y_sample']):.6f}")
    print(f"  loss diff: {abs(bf16['loss'] - fp8['loss']):.2f} "
          f"(rel {abs(bf16['loss'] - fp8['loss']) / max(abs(bf16['loss']), 1e-6) * 100:.2f}%)")
    print(f"  x_grad diff: {abs(bf16['x_grad_norm'] - fp8['x_grad_norm']):.4f} "
          f"(rel {abs(bf16['x_grad_norm'] - fp8['x_grad_norm']) / max(abs(bf16['x_grad_norm']), 1e-6) * 100:.2f}%)")
    print(f"  mem fwd: bf16={bf16['mem_peak_fwd_MB']:.1f} MB  fp8={fp8['mem_peak_fwd_MB']:.1f} MB  "
          f"saved={(bf16['mem_peak_fwd_MB'] - fp8['mem_peak_fwd_MB']):.1f} MB "
          f"({(1 - fp8['mem_peak_fwd_MB'] / max(bf16['mem_peak_fwd_MB'], 1e-6)) * 100:.1f}%)")
    print(f"  mem bwd: bf16={bf16['mem_peak_bwd_MB']:.1f} MB  fp8={fp8['mem_peak_bwd_MB']:.1f} MB")


if __name__ == "__main__":
    main()
