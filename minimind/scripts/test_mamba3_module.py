import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.mamba3_module import Mamba3Config, Mamba3Stack


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    cfg = Mamba3Config(
        d_model=768,
        d_state=192,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        dt_min=1e-3,
        dt_max=1e-1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        chunk_size=256,
        dropout=0.0,
    )
    model = Mamba3Stack(cfg, num_layers=2).to(device=device, dtype=torch.bfloat16)
    model.train()

    bsz, seqlen, d_model = 2, 256, cfg.d_model
    x = torch.randn(bsz, seqlen, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x)
        loss = (y.float().pow(2).mean())

    loss.backward()

    grad_ok = x.grad is not None and torch.isfinite(x.grad).all()
    y_ok = torch.isfinite(y).all()
    print(f"forward_ok={bool(y_ok)} backward_ok={bool(grad_ok)}")
    print(f"loss={loss.item():.6f} y_shape={tuple(y.shape)} dtype={y.dtype}")


if __name__ == "__main__":
    main()
