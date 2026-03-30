#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRAM = ROOT / "luma_stage0" / "dynamics_autoresearch_program.json"
EVAL_SCRIPT = ROOT / "scripts" / "run_dynamics_candidate_eval.py"


@dataclass
class CandidateResult:
    stage_name: str
    stage_steps: int
    candidate: str
    score: float | None
    guard_all_ok: bool
    status: str
    summary_path: Path
    report_path: Path
    metrics_path: Path
    log_path: Path
    checkpoint_path: Path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize(name: str) -> str:
    return name.replace("+", "__").replace("/", "_").replace(" ", "_")


def stage_timeout(program: dict[str, Any], steps: int) -> int:
    return int(program.get("timeouts_sec", {}).get(str(steps), 21600))


def append_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "stage_name",
        "stage_steps",
        "candidate",
        "status",
        "score",
        "guard_all_ok",
        "summary_path",
        "report_path",
        "metrics_path",
        "log_path",
        "checkpoint_path",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def rank_candidates(results: list[CandidateResult]) -> list[CandidateResult]:
    eligible = [r for r in results if r.status == "ok" and r.guard_all_ok and r.score is not None]
    return sorted(eligible, key=lambda r: (r.score, r.candidate))


def prior_stage_name(stage_name: str) -> str | None:
    order = {
        "mid_rescreen": "short_prescreen",
        "long_round1": "mid_rescreen",
        "long_confirm": "long_round1",
    }
    return order.get(stage_name)


def infer_prior_checkpoints(output_dir: Path, stage_name: str, candidates: list[str]) -> dict[str, Path]:
    previous_stage = prior_stage_name(stage_name)
    if previous_stage is None:
        return {}
    inferred: dict[str, Path] = {}
    for candidate in candidates:
        checkpoint_path = output_dir / "checkpoints" / f"{previous_stage}__{sanitize(candidate)}.pt"
        if checkpoint_path.exists():
            inferred[candidate] = checkpoint_path
    return inferred


def run_one_candidate(
    *,
    program_path: Path,
    program: dict[str, Any],
    output_dir: Path,
    runtime_path: Path,
    heartbeat_path: Path,
    stop_path: Path,
    stage_name: str,
    stage_steps: int,
    candidate: str,
    load_checkpoint: Path | None = None,
) -> CandidateResult:
    stem = f"{stage_name}__{sanitize(candidate)}"
    report_path = output_dir / "reports" / f"{stem}.json"
    metrics_path = output_dir / "metrics" / f"{stem}.jsonl"
    summary_path = output_dir / "summaries" / f"{stem}.json"
    log_path = output_dir / "logs" / f"{stem}.log"
    checkpoint_path = output_dir / "checkpoints" / f"{stem}.pt"
    timeout_sec = stage_timeout(program, stage_steps)

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--program",
        str(program_path),
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
        str(checkpoint_path),
    ]
    if load_checkpoint is not None:
        cmd.extend(["--load-checkpoint", str(load_checkpoint)])

    runtime = {
        "status": "running",
        "stage_name": stage_name,
        "stage_steps": stage_steps,
        "candidate": candidate,
        "started_at": time.time(),
        "timeout_sec": timeout_sec,
        "cmd": cmd,
        "load_checkpoint": str(load_checkpoint) if load_checkpoint is not None else None,
        "save_checkpoint": str(checkpoint_path),
    }
    dump_json(runtime_path, runtime)
    dump_json(heartbeat_path, {"status": "starting", "candidate": candidate, "stage_name": stage_name, "ts": time.time()})

    log_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("CMD: " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        start = time.time()
        status = "failed"
        while True:
            if stop_path.exists():
                os.killpg(proc.pid, signal.SIGTERM)
                status = "stopped"
                break
            ret = proc.poll()
            now = time.time()
            dump_json(
                heartbeat_path,
                {
                    "status": "running",
                    "candidate": candidate,
                    "stage_name": stage_name,
                    "stage_steps": stage_steps,
                    "pid": proc.pid,
                    "elapsed_sec": now - start,
                    "timeout_sec": timeout_sec,
                    "ts": now,
                },
            )
            if ret is not None:
                status = "ok" if ret == 0 else f"failed:{ret}"
                break
            if now - start > timeout_sec:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(5)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                status = "timeout"
                break
            time.sleep(30)

        if proc.poll() is None:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass

    score = None
    guard_all_ok = False
    if summary_path.exists():
        summary = load_json(summary_path)
        score = summary.get("score")
        guard_all_ok = bool(summary.get("guard", {}).get("all_ok", False))

    dump_json(
        runtime_path,
        {
            "status": "idle",
            "last_completed_candidate": candidate,
            "last_completed_stage": stage_name,
            "last_status": status,
            "last_score": score,
            "ts": time.time(),
        },
    )
    return CandidateResult(
        stage_name=stage_name,
        stage_steps=stage_steps,
        candidate=candidate,
        score=score,
        guard_all_ok=guard_all_ok,
        status=status,
        summary_path=summary_path,
        report_path=report_path,
        metrics_path=metrics_path,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
    )


def run_stage(
    *,
    program_path: Path,
    program: dict[str, Any],
    output_dir: Path,
    runtime_path: Path,
    heartbeat_path: Path,
    stop_path: Path,
    results_tsv: Path,
    stage_name: str,
    stage_steps: int,
    candidates: list[str],
    load_checkpoints: dict[str, Path] | None = None,
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    load_checkpoints = load_checkpoints or {}
    for candidate in candidates:
        if stop_path.exists():
            break
        result = run_one_candidate(
            program_path=program_path,
            program=program,
            output_dir=output_dir,
            runtime_path=runtime_path,
            heartbeat_path=heartbeat_path,
            stop_path=stop_path,
            stage_name=stage_name,
            stage_steps=stage_steps,
            candidate=candidate,
            load_checkpoint=load_checkpoints.get(candidate),
        )
        results.append(result)
        append_tsv(
            results_tsv,
            {
                "timestamp": int(time.time()),
                "stage_name": result.stage_name,
                "stage_steps": result.stage_steps,
                "candidate": result.candidate,
                "status": result.status,
                "score": "" if result.score is None else result.score,
                "guard_all_ok": result.guard_all_ok,
                "summary_path": str(result.summary_path),
                "report_path": str(result.report_path),
                "metrics_path": str(result.metrics_path),
                "log_path": str(result.log_path),
                "checkpoint_path": str(result.checkpoint_path),
            },
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed Luma dynamics screening program with a pure local watchdog.")
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "autoresearch_dynamics_local")
    parser.add_argument(
        "--start-stage",
        choices=["short_prescreen", "mid_rescreen", "long_round1", "long_confirm"],
        default="short_prescreen",
        help="Stage to begin from. For non-short starts, provide --candidates.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="",
        help="Comma-separated candidate list for runs that start from mid/long stages.",
    )
    parser.add_argument(
        "--stop-after-stage",
        choices=["short_prescreen", "mid_rescreen", "long_round1", "long_confirm"],
        default=None,
        help="Stop cleanly after completing the named stage.",
    )
    args = parser.parse_args()

    program = load_json(args.program)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_tsv = output_dir / "research-results.tsv"
    runtime_path = output_dir / "autoresearch-runtime.json"
    heartbeat_path = output_dir / "watchdog-heartbeat.json"
    state_path = output_dir / "autoresearch-state.json"
    stop_path = output_dir / "STOP"

    dump_json(runtime_path, {"status": "booting", "ts": time.time()})
    dump_json(state_path, {"status": "starting", "program": program["program_name"], "ts": time.time()})

    stages = program["stages"]
    promotions = program.get("promotions", {})
    candidate_priority = list(program["candidate_priority"])
    manual_candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]

    if args.start_stage != "short_prescreen" and not manual_candidates:
        raise SystemExit("--candidates is required when --start-stage is not short_prescreen")

    short_ranked: list[CandidateResult] = []
    mid_ranked: list[CandidateResult] = []
    long_ranked: list[CandidateResult] = []
    confirm_ranked: list[CandidateResult] = []
    checkpoint_lineage: dict[str, Path] = {}

    if args.start_stage == "short_prescreen":
        short_results = run_stage(
            program_path=args.program,
            program=program,
            output_dir=output_dir,
            runtime_path=runtime_path,
            heartbeat_path=heartbeat_path,
            stop_path=stop_path,
            results_tsv=results_tsv,
            stage_name="short_prescreen",
            stage_steps=int(stages["short_prescreen_steps"]),
            candidates=candidate_priority,
        )
        checkpoint_lineage.update({result.candidate: result.checkpoint_path for result in short_results if result.status == "ok"})
        short_ranked = rank_candidates(short_results)
        dump_json(
            state_path,
            {
                "status": "short_complete",
                "ranked": [r.candidate for r in short_ranked],
                "scores": {r.candidate: r.score for r in short_ranked},
                "ts": time.time(),
            },
        )
        if args.stop_after_stage == "short_prescreen":
            dump_json(runtime_path, {"status": "stopped_after_stage", "stage": "short_prescreen", "ts": time.time()})
            dump_json(heartbeat_path, {"status": "stopped_after_stage", "stage": "short_prescreen", "ts": time.time()})
            return 0
        mid_candidates = [r.candidate for r in short_ranked[: int(promotions.get("to_mid_top_k", 4))]]
    else:
        mid_candidates = manual_candidates
        checkpoint_lineage.update(infer_prior_checkpoints(output_dir, "mid_rescreen", mid_candidates))

    if args.start_stage in {"short_prescreen", "mid_rescreen"}:
        mid_results = run_stage(
            program_path=args.program,
            program=program,
            output_dir=output_dir,
            runtime_path=runtime_path,
            heartbeat_path=heartbeat_path,
            stop_path=stop_path,
            results_tsv=results_tsv,
            stage_name="mid_rescreen",
            stage_steps=int(stages["mid_rescreen_steps"]),
            candidates=mid_candidates,
            load_checkpoints=checkpoint_lineage,
        ) if mid_candidates and not stop_path.exists() else []
        checkpoint_lineage.update({result.candidate: result.checkpoint_path for result in mid_results if result.status == "ok"})
        mid_ranked = rank_candidates(mid_results)
        dump_json(
            state_path,
            {
                "status": "mid_complete",
                "ranked": [r.candidate for r in mid_ranked],
                "scores": {r.candidate: r.score for r in mid_ranked},
                "ts": time.time(),
            },
        )
        if args.stop_after_stage == "mid_rescreen":
            dump_json(runtime_path, {"status": "stopped_after_stage", "stage": "mid_rescreen", "ts": time.time()})
            dump_json(heartbeat_path, {"status": "stopped_after_stage", "stage": "mid_rescreen", "ts": time.time()})
            return 0
        long_candidates = [r.candidate for r in mid_ranked[: int(promotions.get("to_long_top_k", 3))]]
    elif args.start_stage == "long_round1":
        long_candidates = manual_candidates
        checkpoint_lineage.update(infer_prior_checkpoints(output_dir, "long_round1", long_candidates))
    else:
        long_candidates = []

    if args.start_stage in {"short_prescreen", "mid_rescreen", "long_round1"}:
        long_results = run_stage(
            program_path=args.program,
            program=program,
            output_dir=output_dir,
            runtime_path=runtime_path,
            heartbeat_path=heartbeat_path,
            stop_path=stop_path,
            results_tsv=results_tsv,
            stage_name="long_round1",
            stage_steps=int(stages["long_round1_steps"]),
            candidates=long_candidates,
            load_checkpoints=checkpoint_lineage,
        ) if long_candidates and not stop_path.exists() else []
        checkpoint_lineage.update({result.candidate: result.checkpoint_path for result in long_results if result.status == "ok"})
        long_ranked = rank_candidates(long_results)
        dump_json(
            state_path,
            {
                "status": "long_round1_complete",
                "ranked": [r.candidate for r in long_ranked],
                "scores": {r.candidate: r.score for r in long_ranked},
                "ts": time.time(),
            },
        )
        if args.stop_after_stage == "long_round1":
            dump_json(runtime_path, {"status": "stopped_after_stage", "stage": "long_round1", "ts": time.time()})
            dump_json(heartbeat_path, {"status": "stopped_after_stage", "stage": "long_round1", "ts": time.time()})
            return 0
        confirm_candidates = [r.candidate for r in long_ranked[: int(promotions.get("to_confirm_top_k", 2))]]
    else:
        confirm_candidates = manual_candidates
        checkpoint_lineage.update(infer_prior_checkpoints(output_dir, "long_confirm", confirm_candidates))

    confirm_results = run_stage(
        program_path=args.program,
        program=program,
        output_dir=output_dir,
        runtime_path=runtime_path,
        heartbeat_path=heartbeat_path,
        stop_path=stop_path,
        results_tsv=results_tsv,
        stage_name="long_confirm",
        stage_steps=int(stages["long_confirm_steps"]),
        candidates=confirm_candidates,
        load_checkpoints=checkpoint_lineage,
    ) if confirm_candidates and not stop_path.exists() else []
    confirm_ranked = rank_candidates(confirm_results)

    dump_json(
        state_path,
        {
            "status": "complete" if not stop_path.exists() else "stopped",
            "short_ranked": [r.candidate for r in short_ranked],
            "mid_ranked": [r.candidate for r in mid_ranked],
            "long_ranked": [r.candidate for r in long_ranked],
            "confirm_ranked": [r.candidate for r in confirm_ranked],
            "winner": confirm_ranked[0].candidate if confirm_ranked else (long_ranked[0].candidate if long_ranked else None),
            "ts": time.time(),
        },
    )
    dump_json(runtime_path, {"status": "complete" if not stop_path.exists() else "stopped", "ts": time.time()})
    dump_json(heartbeat_path, {"status": "complete" if not stop_path.exists() else "stopped", "ts": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
