"""Luma uses this smoke test to prove her first body plan can move before we teach it more subtle habits."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minimind import LumaConfig, LumaForCausalLM


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Luma skeleton smoke test.")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    config = LumaConfig(
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

    device = torch.device("cuda:0")
    model = LumaForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (1, 20), device=device)
    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs.loss.backward()

    finite_logits = torch.isfinite(outputs.logits).all().item()
    finite_hidden = torch.isfinite(outputs.hidden_states).all().item()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    print(
        f"ok={bool(finite_logits and finite_hidden and grad_ok)} "
        f"loss={float(outputs.loss.detach()):.6f} "
        f"logits_shape={tuple(outputs.logits.shape)} "
        f"hidden_shape={tuple(outputs.hidden_states.shape)}"
    )


if __name__ == "__main__":
    main()
