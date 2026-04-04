#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from analyze_luma_dynamics_layer2 import run_layer2_analysis

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRAM = ROOT / "luma_stage0" / "dynamics_autoresearch_program.json"
RUN_STAGE12 = ROOT / "scripts" / "run_luma_stage12.py"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_flag(cmd: list[str], key: str, value) -> None:
    flag = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _resolve_candidate_args(program: dict, candidate: str | None) -> dict:
    fixed = dict(program["fixed_eval_args"])
    if candidate is None:
        return fixed
    overrides = program.get("candidate_overrides", {})
    if candidate not in overrides:
        raise KeyError(f"unknown candidate: {candidate}")
    fixed.update(overrides[candidate])
    return fixed


def _score_report(report: dict, program: dict) -> tuple[float, dict]:
    per_task = report["per_task"]
    stage1 = report["stage1"]
    stage2 = report["stage2"]
    weights = program["score_formula"]["weights"]
    metrics = {}
    for metric_name in weights:
        if metric_name.endswith("_self_tail"):
            bucket = metric_name[: -len("_self_tail")]
            if bucket not in per_task:
                raise KeyError(f"score metric '{metric_name}' requires bucket '{bucket}', but it is missing from per_task")
            metrics[metric_name] = float(per_task[bucket]["stage2"]["self_loss_tail"])
        else:
            raise KeyError(f"unsupported score metric key: {metric_name}")
    score = sum(weights[name] * metrics[name] for name in weights)

    guards = program["guard_rules"]
    rollout_watch_buckets = ["math", "python_code", "mixed", "persona_seed", "arc_agi"]
    rollout_candidates = []
    for bucket in rollout_watch_buckets:
        if bucket in per_task:
            rollout_candidates.append(float(per_task[bucket]["stage2"]["rollout_nonzero_ratio"]))
    if not rollout_candidates:
        raise KeyError("no rollout_nonzero_ratio buckets available for guard checks")
    rollout_nonzero = max(rollout_candidates)
    guard_status = {
        "c_t_var_ok": float(stage1["c_t_var"]) >= float(guards["min_c_t_var"]),
        "rollout_nonzero_ok": rollout_nonzero >= float(guards["min_any_rollout_nonzero_ratio"]),
        "dialogue_ok": float(per_task["dialogue"]["stage2"]["self_loss_tail"]) <= float(guards["max_dialogue_self_tail"]),
        "emotion_ok": float(per_task["emotion"]["stage2"]["self_loss_tail"]) <= float(guards["max_emotion_self_tail"]),
        "nonfinite_ok": stage2.get("first_nonfinite_step") is None,
    }
    if "min_sigreg_source_std" in guards:
        guard_status["sigreg_source_std_ok"] = float(stage2.get("world_sigreg_source_std", 0.0)) >= float(guards["min_sigreg_source_std"])
    guard_status["all_ok"] = all(guard_status.values())

    def _tail_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        width = max(1, min(2, len(values)))
        return float(sum(values[-width:]) / width)

    bucket_scores = {}
    for bucket, bucket_payload in per_task.items():
        stage2 = bucket_payload.get("stage2", {})
        bucket_scores[bucket] = {
            "self_loss_tail": float(stage2.get("self_loss_tail", 0.0)),
            "self_rollout_tail": float(stage2.get("self_rollout_tail", 0.0)),
            "rollout_nonzero_ratio": float(stage2.get("rollout_nonzero_ratio", 0.0)),
            "rollout_active_ratio": float(stage2.get("rollout_active_ratio", 0.0)),
            "mean_loss": _tail_mean(stage2.get("losses", [])),
        }

    diagnostics = {
        "candidate": report.get("candidate_name"),
        "score": score,
        "metrics": metrics,
        "bucket_scores": bucket_scores,
        "rollout_nonzero_max": rollout_nonzero,
        "c_t_var": float(stage1["c_t_var"]),
        "hard_loop_var": float(stage1["hard_loop_var"]),
        "guard": guard_status,
        "checkpoint_lineage": report.get("checkpoint_lineage", []),
        "load_checkpoint": report.get("load_checkpoint"),
        "save_checkpoint": report.get("save_checkpoint"),
        "sigreg_source_mean": float(stage2.get("world_sigreg_source_mean", stage1.get("sigreg_source_mean", 0.0))),
        "sigreg_source_std": float(stage2.get("world_sigreg_source_std", stage1.get("sigreg_source_std", 0.0))),
        "first_nonfinite_step": stage2.get("first_nonfinite_step"),
        "world_sigreg_loss_head": float(stage2.get("world_sigreg_loss_head", 0.0)),
        "world_sigreg_loss_tail": float(stage2.get("world_sigreg_loss_tail", 0.0)),
        "world_sigreg_loss_max": float(stage2.get("world_sigreg_loss_max", 0.0)),
        "world_sigreg_loss_step": int(stage2.get("world_sigreg_loss_step", -1)),
        "grad_norm_total_tail": float(stage2.get("grad_norm_total_tail", 0.0)),
        "grad_norm_world_encoder_tail": float(stage2.get("grad_norm_world_encoder_tail", 0.0)),
    }
    return score, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one fixed Luma dynamics candidate eval and emit a mechanical score.")
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--candidate", type=str, default=None)
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--layer2-json-out", type=Path, default=None)
    parser.add_argument("--layer2-md-out", type=Path, default=None)
    parser.add_argument("--layer2-csv-prefix", type=Path, default=None)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra raw CLI args forwarded to run_luma_stage12.py")
    args = parser.parse_args()

    program = _load_json(args.program)
    fixed = _resolve_candidate_args(program, args.candidate)
    if args.stage2_steps is not None:
        fixed["stage2_steps"] = args.stage2_steps
    else:
        fixed["stage2_steps"] = int(program["stages"]["short_prescreen_steps"])

    cmd = [sys.executable, str(RUN_STAGE12)]
    for key, value in fixed.items():
        _append_flag(cmd, key, value)
    if args.candidate:
        _append_flag(cmd, "candidate_name", args.candidate)
    _append_flag(cmd, "json_out", args.json_out)
    _append_flag(cmd, "metrics_out", args.metrics_out)
    _append_flag(cmd, "load_checkpoint", args.load_checkpoint)
    _append_flag(cmd, "save_checkpoint", args.save_checkpoint)
    for raw in args.extra_arg:
        cmd.extend(shlex.split(raw))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    if args.layer2_json_out is not None:
        args.layer2_json_out.parent.mkdir(parents=True, exist_ok=True)
    if args.layer2_md_out is not None:
        args.layer2_md_out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode

    report = _load_json(args.json_out)
    score, diagnostics = _score_report(report, program)
    layer2_json_out = args.layer2_json_out or args.json_out.with_suffix(".layer2.json")
    layer2_md_out = args.layer2_md_out or args.json_out.with_suffix(".layer2.md")
    layer2_csv_prefix = args.layer2_csv_prefix or args.json_out.with_suffix(".layer2")
    layer2 = run_layer2_analysis(
        report_path=args.json_out,
        metrics_path=args.metrics_out,
        json_out=layer2_json_out,
        md_out=layer2_md_out,
        csv_prefix=layer2_csv_prefix,
    )
    report["layer2_analysis"] = layer2
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostics["layer2"] = {
        "pod_effective_rank": int(layer2["pod"]["effective_rank"]),
        "pod_top1_energy_ratio": float(layer2["pod"]["modes"][0]["energy_ratio"]) if layer2["pod"]["modes"] else 0.0,
        "dmd_spectral_radius": float(layer2["dmd"]["spectral_radius"]),
        "forcing_top_abs_corr": float(layer2["forcing_response"]["top_abs_corr"]),
        "paths": {
            "json": str(layer2_json_out),
            "md": str(layer2_md_out),
            "csv_prefix": str(layer2_csv_prefix),
        },
    }
    if args.summary_out is not None:
        args.summary_out.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
