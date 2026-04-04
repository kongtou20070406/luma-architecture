#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MINIMIND_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/home/kt/ai/.venvs/luma-global/bin/python")
EVAL_SCRIPT = MINIMIND_ROOT / "scripts" / "run_dynamics_candidate_eval.py"
PROGRAM_PATH = MINIMIND_ROOT / "luma_stage0" / "dynamics_autoresearch_program.json"
INIT_HELPER = Path("/home/kt/.codex/skills/codex-autoresearch/scripts/autoresearch_init_run.py")
RECORD_HELPER = Path("/home/kt/.codex/skills/codex-autoresearch/scripts/autoresearch_record_iteration.py")


@dataclass
class EvalResult:
    ok: bool
    score: float
    rank: int | None
    rollout_nonzero: float | None
    first_nonfinite_step: int | None
    summary_path: Path
    report_path: Path
    metrics_path: Path
    ckpt_path: Path
    log_path: Path
    error: str = ""


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _run(cmd: list[str], log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(cmd, check=False, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, check=False, text=True, stdout=f, stderr=subprocess.STDOUT)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_rollout_nonzero(report: dict[str, Any]) -> float:
    stage2 = report.get("stage2", {}) or {}
    value = stage2.get("rollout_nonzero_ratio", None)
    if value is not None:
        return float(value)
    per_task = report.get("per_task", {}) or {}
    candidates = []
    for bucket in ("math", "python_code", "mixed", "persona_seed", "arc_agi"):
        if bucket in per_task:
            candidates.append(float((per_task[bucket].get("stage2", {}) or {}).get("rollout_nonzero_ratio", 0.0)))
    return max(candidates) if candidates else 0.0


def run_eval(
    *,
    out_dir: Path,
    tag: str,
    candidate: str,
    stage_steps: int,
    extra_args: list[str],
    load_checkpoint: Path | None = None,
) -> EvalResult:
    reports = out_dir / "reports"
    metrics = out_dir / "metrics"
    summaries = out_dir / "summaries"
    checkpoints = out_dir / "checkpoints"
    logs = out_dir / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    summaries.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    report_path = reports / f"{tag}.json"
    summary_path = summaries / f"{tag}.summary.json"
    metrics_path = metrics / f"{tag}.jsonl"
    ckpt_path = checkpoints / f"{tag}.pt"
    log_path = logs / f"{tag}.log"

    cmd = [
        str(PYTHON_BIN),
        str(EVAL_SCRIPT),
        "--program",
        str(PROGRAM_PATH),
        "--candidate",
        candidate,
        "--stage2-steps",
        str(stage_steps),
        "--json-out",
        str(report_path),
        "--metrics-out",
        str(metrics_path),
        "--summary-out",
        str(summary_path),
        "--save-checkpoint",
        str(ckpt_path),
    ]
    if load_checkpoint is not None:
        cmd.extend(["--load-checkpoint", str(load_checkpoint)])
    for raw in extra_args:
        # Use --extra-arg=<raw> so raw values that start with "--" are treated as payload.
        cmd.append(f"--extra-arg={raw}")

    completed = _run(cmd, log_path=log_path)
    if completed.returncode != 0:
        return EvalResult(
            ok=False,
            score=1e6,
            rank=None,
            rollout_nonzero=None,
            first_nonfinite_step=None,
            summary_path=summary_path,
            report_path=report_path,
            metrics_path=metrics_path,
            ckpt_path=ckpt_path,
            log_path=log_path,
            error=f"returncode={completed.returncode}",
        )

    try:
        summary = _json(summary_path)
        report = _json(report_path)
        raw_score = summary.get("score", None)
        score = float(raw_score)
        if not math.isfinite(score):
            return EvalResult(
                ok=False,
                score=1e6,
                rank=None,
                rollout_nonzero=None,
                first_nonfinite_step=None,
                summary_path=summary_path,
                report_path=report_path,
                metrics_path=metrics_path,
                ckpt_path=ckpt_path,
                log_path=log_path,
                error=f"non_finite_score={raw_score}",
            )
        rank = int(_finite_float((summary.get("layer2", {}) or {}).get("pod_effective_rank", 0), 0.0))
        rollout_nonzero = _finite_float(_extract_rollout_nonzero(report), 0.0)
        first_nonfinite_step = (report.get("stage2", {}) or {}).get("first_nonfinite_step", None)
        if first_nonfinite_step is not None:
            return EvalResult(
                ok=False,
                score=1e6,
                rank=None,
                rollout_nonzero=None,
                first_nonfinite_step=first_nonfinite_step,
                summary_path=summary_path,
                report_path=report_path,
                metrics_path=metrics_path,
                ckpt_path=ckpt_path,
                log_path=log_path,
                error=f"non_finite_step={first_nonfinite_step}",
            )
        return EvalResult(
            ok=True,
            score=score,
            rank=rank,
            rollout_nonzero=rollout_nonzero,
            first_nonfinite_step=first_nonfinite_step,
            summary_path=summary_path,
            report_path=report_path,
            metrics_path=metrics_path,
            ckpt_path=ckpt_path,
            log_path=log_path,
        )
    except Exception as exc:  # pragma: no cover - diagnostics path
        return EvalResult(
            ok=False,
            score=1e6,
            rank=None,
            rollout_nonzero=None,
            first_nonfinite_step=None,
            summary_path=summary_path,
            report_path=report_path,
            metrics_path=metrics_path,
            ckpt_path=ckpt_path,
            log_path=log_path,
            error=f"parse_error={exc}",
        )


def make_extra_args(rng: random.Random, idx: int) -> list[str]:
    # Iteration families:
    # 0: balanced
    # 1: rank-first (lighter self, stronger world stabilization)
    # 2: score-safe (lighter regularizers)
    family = idx % 3
    if family == 0:
        world_sigreg = rng.choice([0.010, 0.015, 0.020, 0.030])
        warmup = rng.choice([256, 512, 1024])
        self_jepa = rng.choice([0.7, 0.8, 1.0])
        self_rollout = rng.choice([0.2, 0.35, 0.5])
        exit_aux = rng.choice([0.003, 0.006, 0.010])
    elif family == 1:
        world_sigreg = rng.choice([0.015, 0.020, 0.030])
        warmup = rng.choice([512, 1024, 1536])
        self_jepa = rng.choice([0.4, 0.6, 0.8])
        self_rollout = rng.choice([0.1, 0.2, 0.35])
        exit_aux = rng.choice([0.0, 0.003, 0.006])
    else:
        world_sigreg = rng.choice([0.008, 0.010, 0.015])
        warmup = rng.choice([256, 512, 1024])
        self_jepa = rng.choice([0.8, 1.0])
        self_rollout = rng.choice([0.2, 0.35])
        exit_aux = rng.choice([0.0, 0.003])

    rollout_zone = rng.choice([0.0, 0.002, 0.005, 0.010, 0.020])
    route_entropy = rng.choice([0.0, 0.002, 0.005, 0.010])
    route_local_share = rng.choice([0.0, 0.002, 0.005, 0.010])
    vitality = rng.choice([0.0, 0.002, 0.005, 0.010])
    compression_w = rng.choice([0.0, 0.001, 0.003, 0.005])
    compression_drift_floor = rng.choice([0.01, 0.02, 0.03])
    compression_var_floor = rng.choice([0.001, 0.003, 0.005])
    world_delta_weight = rng.choice([0.05, 0.1, 0.2])

    args = [
        "--seq-len 1024",
        "--sigreg-world-source sigreg_on_encoder_latent",
        "--sigreg-world-fp32-only",
        "--force-fp32",
        f"--world-sigreg-weight {world_sigreg}",
        f"--sigreg-world-warmup-steps {warmup}",
        f"--world-delta-weight {world_delta_weight}",
        f"--self-jepa-weight {self_jepa}",
        f"--self-rollout-weight {self_rollout}",
        f"--exit-aux-weight {exit_aux}",
        f"--rollout-zone-weight {rollout_zone}",
        f"--routing-tier-entropy-weight {route_entropy}",
        f"--routing-min-local-share-weight {route_local_share}",
        f"--trajectory-vitality-weight {vitality}",
        f"--compression-dynamics-weight {compression_w}",
        f"--compression-block-drift-floor {compression_drift_floor}",
        f"--compression-block-var-floor {compression_var_floor}",
    ]
    return args


def objective(
    *,
    score: float,
    rank_4096: int,
    rank_10240: int,
    rollout_4096: float,
    rollout_10240: float,
    first_nonfinite_4096: int | None,
    first_nonfinite_10240: int | None,
    score_budget: float,
    min_rollout: float,
) -> float:
    value = score
    if rank_4096 < 3:
        value += 0.30 * float(3 - rank_4096)
    if rank_10240 < 3:
        value += 1.00 * float(3 - rank_10240)
    if rollout_4096 < min_rollout:
        value += 0.50 * (min_rollout - rollout_4096) / max(min_rollout, 1e-6)
    if rollout_10240 < min_rollout:
        value += 1.50 * (min_rollout - rollout_10240) / max(min_rollout, 1e-6)
    if first_nonfinite_4096 is not None:
        value += 5.0
    if first_nonfinite_10240 is not None:
        value += 5.0
    if score > score_budget:
        value += 1.0 * ((score / max(score_budget, 1e-12)) - 1.0)
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Foreground 100-iteration dynamics-constraint search on 4096/10240.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--candidate", type=str, default="A2-progress_shape_v1-h3+progress_exit_readout+m1_full_regularizers_from_a4e_sigreg_low")
    parser.add_argument("--score-tolerance", type=float, default=0.15)
    parser.add_argument("--min-rollout-nonzero", type=float, default=0.03)
    args = parser.parse_args()

    if not PYTHON_BIN.exists():
        raise SystemExit(f"missing python env: {PYTHON_BIN}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_tag = f"luma_dynamics_constraints_fg_{stamp}"
    out_dir = MINIMIND_ROOT / "artifacts" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / "run-journal.jsonl"

    results_path = REPO_ROOT / "research-results.tsv"
    state_path = REPO_ROOT / "autoresearch-state.json"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()

    # Baseline before run init.
    baseline_args = [
        "--seq-len 1024",
        "--sigreg-world-source sigreg_on_encoder_latent",
        "--sigreg-world-fp32-only",
        "--force-fp32",
    ]
    base_4096 = run_eval(
        out_dir=out_dir,
        tag="baseline_4096",
        candidate=args.candidate,
        stage_steps=4096,
        extra_args=baseline_args,
        load_checkpoint=None,
    )
    if not base_4096.ok:
        raise SystemExit(f"baseline_4096 failed: {base_4096.error}; log={base_4096.log_path}")
    base_10240 = run_eval(
        out_dir=out_dir,
        tag="baseline_10240",
        candidate=args.candidate,
        stage_steps=10240,
        extra_args=baseline_args,
        load_checkpoint=base_4096.ckpt_path,
    )
    if not base_10240.ok:
        raise SystemExit(f"baseline_10240 failed: {base_10240.error}; log={base_10240.log_path}")

    baseline_score_budget = float(base_10240.score) * (1.0 + args.score_tolerance)
    baseline_objective = objective(
        score=float(base_10240.score),
        rank_4096=int(base_4096.rank),
        rank_10240=int(base_10240.rank),
        rollout_4096=float(base_4096.rollout_nonzero),
        rollout_10240=float(base_10240.rollout_nonzero),
        first_nonfinite_4096=base_4096.first_nonfinite_step,
        first_nonfinite_10240=base_10240.first_nonfinite_step,
        score_budget=baseline_score_budget,
        min_rollout=args.min_rollout_nonzero,
    )

    init_cmd = [
        "python3",
        str(INIT_HELPER),
        "--results-path",
        str(results_path),
        "--state-path",
        str(state_path),
        "--mode",
        "loop",
        "--session-mode",
        "foreground",
        "--goal",
        "Tune unified dynamics constraints to keep scores within +15% while achieving stable POD rank>=3 and non-flat rollout across 4096 and 10240.",
        "--scope",
        "minimind/model/model_minimind.py, minimind/scripts/run_luma_stage12.py, minimind/luma_stage0/dynamics_autoresearch_program.json",
        "--metric-name",
        "dynamics_objective_4096_10240",
        "--direction",
        "lower",
        "--verify",
        f"{PYTHON_BIN} {EVAL_SCRIPT} --program {PROGRAM_PATH} --candidate {args.candidate} --stage2-steps 4096/10240",
        "--guard",
        "pod_rank>=3 on 4096+10240 and rollout_nonzero_ratio>=threshold and score<=budget",
        "--run-tag",
        run_tag,
        "--iterations",
        str(args.iterations),
        "--stop-condition",
        "iteration_cap",
        "--rollback-policy",
        "revert",
        "--parallel-mode",
        "serial",
        "--web-search",
        "disabled",
        "--environment-summary",
        "cpu=32 ram=95366MB gpu=RTX5090(32GB) python=luma-global",
        "--baseline-metric",
        str(baseline_objective),
        "--baseline-commit",
        commit,
        "--baseline-description",
        f"baseline candidate={args.candidate} objective={baseline_objective:.6f} score10240={base_10240.score:.6f} rank4096={base_4096.rank} rank10240={base_10240.rank}",
        "--force",
    ]
    init_run = _run(init_cmd)
    if init_run.returncode != 0:
        raise SystemExit(f"init failed: {init_run.stderr}")

    retained_metric = baseline_objective
    consecutive_hard_pass = 0
    rng = random.Random(args.seed)

    for i in range(1, args.iterations + 1):
        iter_tag = f"iter_{i:03d}"
        extra_args = make_extra_args(rng, i)

        e4096 = run_eval(
            out_dir=out_dir,
            tag=f"{iter_tag}_4096",
            candidate=args.candidate,
            stage_steps=4096,
            extra_args=extra_args,
            load_checkpoint=None,
        )
        if not e4096.ok:
            desc = f"{iter_tag} failed@4096 error={e4096.error} log={e4096.log_path}"
            _run(
                [
                    "python3",
                    str(RECORD_HELPER),
                    "--results-path",
                    str(results_path),
                    "--state-path",
                    str(state_path),
                    "--status",
                    "crash",
                    "--metric",
                    str(retained_metric),
                    "--commit",
                    commit,
                    "--guard",
                    "fail",
                    "--description",
                    desc,
                ]
            )
            continue

        e10240 = run_eval(
            out_dir=out_dir,
            tag=f"{iter_tag}_10240",
            candidate=args.candidate,
            stage_steps=10240,
            extra_args=extra_args,
            load_checkpoint=e4096.ckpt_path,
        )
        if not e10240.ok:
            desc = f"{iter_tag} failed@10240 error={e10240.error} log={e10240.log_path}"
            _run(
                [
                    "python3",
                    str(RECORD_HELPER),
                    "--results-path",
                    str(results_path),
                    "--state-path",
                    str(state_path),
                    "--status",
                    "crash",
                    "--metric",
                    str(retained_metric),
                    "--commit",
                    commit,
                    "--guard",
                    "fail",
                    "--description",
                    desc,
                ]
            )
            continue

        obj = objective(
            score=float(e10240.score),
            rank_4096=int(e4096.rank),
            rank_10240=int(e10240.rank),
            rollout_4096=float(e4096.rollout_nonzero),
            rollout_10240=float(e10240.rollout_nonzero),
            first_nonfinite_4096=e4096.first_nonfinite_step,
            first_nonfinite_10240=e10240.first_nonfinite_step,
            score_budget=baseline_score_budget,
            min_rollout=args.min_rollout_nonzero,
        )

        hard_pass = (
            int(e4096.rank) >= 3
            and int(e10240.rank) >= 3
            and float(e4096.rollout_nonzero) >= args.min_rollout_nonzero
            and float(e10240.rollout_nonzero) >= args.min_rollout_nonzero
            and float(e10240.score) <= baseline_score_budget
            and e4096.first_nonfinite_step is None
            and e10240.first_nonfinite_step is None
        )
        if hard_pass:
            consecutive_hard_pass += 1
        else:
            consecutive_hard_pass = 0

        keep = hard_pass and (obj < retained_metric)
        status = "keep" if keep else "discard"
        guard = "pass" if hard_pass else "fail"
        labels: list[str] = []
        if int(e4096.rank) >= 3 and int(e10240.rank) >= 3:
            labels.append("rank3-stable")
        if float(e4096.rollout_nonzero) >= args.min_rollout_nonzero and float(e10240.rollout_nonzero) >= args.min_rollout_nonzero:
            labels.append("rollout-alive")
        if float(e10240.score) <= baseline_score_budget:
            labels.append("score-budget-pass")
        if hard_pass:
            labels.append("hard-pass")

        desc = (
            f"{iter_tag} obj={obj:.6f} score10240={e10240.score:.6f} "
            f"rank4096={e4096.rank} rank10240={e10240.rank} "
            f"roll4096={e4096.rollout_nonzero:.6f} roll10240={e10240.rollout_nonzero:.6f} "
            f"budget={baseline_score_budget:.6f} args={extra_args} "
            f"r4096={e4096.report_path.name} r10240={e10240.report_path.name}"
        )

        rec_cmd = [
            "python3",
            str(RECORD_HELPER),
            "--results-path",
            str(results_path),
            "--state-path",
            str(state_path),
            "--status",
            status,
            "--metric",
            str(obj),
            "--commit",
            commit,
            "--guard",
            guard,
            "--description",
            desc,
        ]
        for lb in labels:
            rec_cmd.extend(["--label", lb])
        rec = _run(rec_cmd)
        if rec.returncode != 0:
            raise SystemExit(f"record iteration failed at {iter_tag}: {rec.stderr}")

        if keep:
            retained_metric = obj

        with journal_path.open("a", encoding="utf-8") as jf:
            jf.write(
                json.dumps(
                    {
                        "iter": i,
                        "status": status,
                        "hard_pass": hard_pass,
                        "objective": obj,
                        "retained_metric": retained_metric,
                        "score_budget": baseline_score_budget,
                        "score_10240": e10240.score,
                        "rank_4096": e4096.rank,
                        "rank_10240": e10240.rank,
                        "rollout_4096": e4096.rollout_nonzero,
                        "rollout_10240": e10240.rollout_nonzero,
                        "labels": labels,
                        "extra_args": extra_args,
                        "report_4096": str(e4096.report_path),
                        "report_10240": str(e10240.report_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        # Early stop when we already have stable passes twice in a row.
        if consecutive_hard_pass >= 2:
            break

    print(
        json.dumps(
            {
                "run_tag": run_tag,
                "out_dir": str(out_dir),
                "results_path": str(results_path),
                "state_path": str(state_path),
                "baseline_objective": baseline_objective,
                "baseline_score_budget": baseline_score_budget,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
