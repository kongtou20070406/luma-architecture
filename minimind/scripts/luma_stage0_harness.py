import argparse
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luma_stage0.config_schema import build_default_config
from luma_stage0.metrics_schema import Stage0MetricRecord, write_metric_jsonl, now
from model.mamba3_module import Mamba3Config, Mamba3Stack


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Luma Stage-0 short-run harness")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(ROOT / "artifacts" / "stage0_metrics.jsonl"))
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)
    _set_seed(args.seed)

    cfg = build_default_config()
    comp = cfg["compression_zone"]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Stage-0 short-run uses stable SISO path to ensure deterministic scaffolding.
    m_cfg = Mamba3Config(
        d_model=comp["d_model"],
        d_state=comp["d_state"],
        expand=2,
        headdim=64,
        ngroups=1,
        is_mimo=False,
        mimo_rank=4,
        chunk_size=64,
    )
    model = Mamba3Stack(m_cfg, num_layers=1).to(device=device, dtype=dtype)
    model.train()

    x = torch.randn(args.batch_size, args.seq_len, comp["d_model"], device=device, dtype=dtype, requires_grad=True)
    y = model(x)
    loss = y.float().pow(2).mean()
    loss.backward()

    grad_ok = x.grad is not None and torch.isfinite(x.grad).all().item()
    y_ok = torch.isfinite(y).all().item()
    rec = Stage0MetricRecord(
        event="stage0_harness_forward_backward",
        ok=bool(grad_ok and y_ok),
        value=float(loss.detach().item()),
        note=f"shape={tuple(y.shape)} dtype={str(y.dtype)}",
        timestamp=now(),
    )
    write_metric_jsonl(args.out, rec)
    print(f"ok={rec.ok} loss={rec.value:.6f} {rec.note}")


if __name__ == "__main__":
    main()

