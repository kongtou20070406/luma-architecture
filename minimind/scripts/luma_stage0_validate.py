import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luma_stage0.config_schema import build_default_config


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Luma stage-0 config contracts")
    parser.add_argument("--out", type=str, default=str(ROOT / "artifacts" / "stage0_config_snapshot.json"))
    args = parser.parse_args()

    cfg = build_default_config()

    comp = cfg["compression_zone"]
    reason = cfg["reason_loop"]
    mhc = cfg["mhc"]
    loss = cfg["loss"]
    gate = cfg["gate"]

    # Compression zone contracts
    _assert(comp["groups"] == 4, "groups must be 4")
    _assert(comp["layers_per_group"] == 6, "layers_per_group must be 6")
    _assert(comp["mamba_layers_per_group"] == 5, "mamba_layers_per_group must be 5")
    _assert(comp["retrieval_layers_per_group"] == 1, "retrieval_layers_per_group must be 1")
    _assert(comp["block_repr_count"] == 8, "block_repr_count must be 8")

    # Reason loop contracts
    _assert(reason["slow_k"] == 2, "slow_k must be 2")
    _assert(reason["meta_dim"] == 96, "meta_dim must be 96")
    _assert(reason["c_t_dim"] == 64, "c_t_dim must be 64")
    _assert(reason["loops_max"] == 8, "loops_max must be 8")

    # mHC contracts
    _assert(mhc["n_streams"] == 4, "mHC n_streams must be 4")
    _assert(mhc["apply_zone"] == "reason_loop_only", "mHC must start in reason loop only")
    _assert(mhc["sinkhorn_iters"] >= 10, "mHC sinkhorn_iters must be >= 10")
    _assert(mhc["residual_constraint"] == "birkhoff_doubly_stochastic", "mHC residual constraint invalid")

    # Loss contracts
    _assert(loss["self_target"] == "delta_c_t", "self target must be delta_c_t")
    _assert(loss["world_target"] == "masked_latent_prediction", "world target must be masked latent")
    _assert(loss["rollout_steps"] == 2, "rollout steps must be 2")

    # Gate contracts
    _assert(0.0 < gate["delta_h_threshold"] < 1.0, "delta_h_threshold out of range")
    _assert(gate["c_t_cos_low"] < gate["c_t_cos_high"], "c_t cosine bounds invalid")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("stage0 config validation OK")


if __name__ == "__main__":
    main()
