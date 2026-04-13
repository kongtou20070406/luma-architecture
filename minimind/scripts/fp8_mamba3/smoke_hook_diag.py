"""Diagnostic: verify saved_tensors_hooks are actually firing in Mamba3Block."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from model.mamba3_module import Mamba3Block, Mamba3Config
from fp8_saved_tensors import Fp8ActivationContext

cfg = Mamba3Config(
    d_model=768, d_state=192, expand=2, headdim=64,
    chunk_size=32, is_mimo=True, mimo_rank=2, dropout=0.0,
    use_gradient_checkpointing=False,
    use_fp8_activation_cache=False,  # 我们自己显式管理 context 以捕获 stats
)
block = Mamba3Block(cfg).cuda().to(torch.bfloat16)
block.train()

torch.manual_seed(0)
x = torch.randn(1, 2048, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)

ctx = Fp8ActivationContext(block_size=128, debug=True)
with ctx:
    y = block(x)

print("After forward:", ctx.stats())

y.float().pow(2).sum().backward()

print("After backward:", ctx.stats())
