#!/usr/bin/env python3
"""Luma runs second-layer dynamics analysis after each experiment so we can judge structure quality, not only tail loss.

Luma 会在每次实验后做二层动力学分析，这样我们看的不只是尾损失，还能看到动力学结构是否健康。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FORCING_KEYS = [
    "progress_next_mean",
    "progress_trend_mean",
    "progress_plateau_mean",
    "world_surprise_mean",
    "c_t_var",
]

RESPONSE_KEYS = [
    "self_loss_tail",
    "self_rollout_tail",
    "rollout_nonzero_ratio",
    "rollout_active_ratio",
    "mean_delta_norm",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.allclose(a, a.mean()) or np.allclose(b, b.mean()):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _pod_analysis(x: np.ndarray) -> dict[str, Any]:
    if x.shape[0] < 2:
        return {"modes": [], "total_energy": 0.0, "effective_rank": 0}
    x_centered = x - x.mean(axis=0, keepdims=True)
    cov = (x_centered.T @ x_centered) / max(1, x_centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total = float(eigvals.sum())
    if total <= 0:
        return {"modes": [], "total_energy": 0.0, "effective_rank": 0}
    ratios = eigvals / total
    cumulative = np.cumsum(ratios)
    modes = []
    for i in range(len(eigvals)):
        modes.append(
            {
                "mode": i + 1,
                "eigenvalue": float(eigvals[i]),
                "energy_ratio": float(ratios[i]),
                "cumulative_energy": float(cumulative[i]),
                "vector": [float(v) for v in eigvecs[:, i].tolist()],
            }
        )
    effective_rank = int(np.sum(ratios > 1e-3))
    return {"modes": modes, "total_energy": total, "effective_rank": effective_rank}


def _dmd_analysis(x: np.ndarray) -> dict[str, Any]:
    if x.shape[0] < 3:
        return {"modes": [], "spectral_radius": 0.0, "stable": True}
    x_centered = x - x.mean(axis=0, keepdims=True)
    x0 = x_centered[:-1].T
    x1 = x_centered[1:].T
    if np.allclose(x0, 0.0) or np.allclose(x1, 0.0):
        return {"modes": [], "spectral_radius": 0.0, "stable": True}
    a = x1 @ np.linalg.pinv(x0)
    eigvals = np.linalg.eigvals(a)
    modes = []
    abs_vals = np.abs(eigvals)
    for idx, eig in enumerate(eigvals):
        modes.append(
            {
                "mode": idx + 1,
                "real": float(np.real(eig)),
                "imag": float(np.imag(eig)),
                "magnitude": float(np.abs(eig)),
                "angle": float(np.angle(eig)),
            }
        )
    spectral_radius = float(abs_vals.max()) if abs_vals.size else 0.0
    return {
        "modes": modes,
        "spectral_radius": spectral_radius,
        "stable": bool(spectral_radius < 1.0 + 1e-3),
    }


def _forcing_response_analysis(report: dict[str, Any]) -> dict[str, Any]:
    per_task = report.get("per_task", {})
    rows = []
    for bucket, payload in per_task.items():
        stage1 = payload.get("stage1", {})
        stage2 = payload.get("stage2", {})
        row = {"bucket": bucket}
        for key in FORCING_KEYS:
            row[key] = float(stage1.get(key, 0.0))
        for key in RESPONSE_KEYS:
            row[key] = float(stage2.get(key, 0.0))
        rows.append(row)

    if len(rows) < 2:
        return {"bucket_rows": rows, "correlations": [], "top_abs_corr": 0.0}

    correlations = []
    for fk in FORCING_KEYS:
        f_vec = np.asarray([row[fk] for row in rows], dtype=np.float64)
        for rk in RESPONSE_KEYS:
            r_vec = np.asarray([row[rk] for row in rows], dtype=np.float64)
            corr = _safe_corr(f_vec, r_vec)
            correlations.append(
                {
                    "forcing": fk,
                    "response": rk,
                    "corr": corr,
                    "abs_corr": abs(corr),
                }
            )
    correlations.sort(key=lambda item: item["abs_corr"], reverse=True)
    top_abs = correlations[0]["abs_corr"] if correlations else 0.0
    return {"bucket_rows": rows, "correlations": correlations, "top_abs_corr": float(top_abs)}


def _build_mixed_trajectory(report: dict[str, Any]) -> np.ndarray:
    stage2 = report.get("stage2", {})
    losses = np.asarray(stage2.get("losses", []), dtype=np.float64)
    self_losses = np.asarray(stage2.get("self_losses", []), dtype=np.float64)
    rollout_losses = np.asarray(stage2.get("rollout_losses", []), dtype=np.float64)
    t = min(len(losses), len(self_losses), len(rollout_losses))
    if t <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    x = np.stack([losses[:t], self_losses[:t], rollout_losses[:t]], axis=1)
    keep = np.isfinite(x).all(axis=1)
    return x[keep]


def _metrics_digest(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, float] = {}
    failures = 0
    for record in metrics:
        event = str(record.get("event", ""))
        value = float(record.get("value", 0.0))
        latest[event] = value
        if bool(record.get("ok", True)) is False:
            failures += 1
    return {"num_records": len(metrics), "num_failed_checks": failures, "latest_event_values": latest}


def _write_bucket_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = ["bucket"] + FORCING_KEYS + RESPONSE_KEYS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_modes_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(analysis: dict[str, Any]) -> str:
    pod_modes = analysis["pod"]["modes"]
    dmd_modes = analysis["dmd"]["modes"]
    corr_rows = analysis["forcing_response"]["correlations"][:10]
    lines = []
    lines.append("# Luma Layer-2 Dynamics Analysis")
    lines.append("")
    lines.append("## POD")
    lines.append(f"- effective_rank: {analysis['pod']['effective_rank']}")
    lines.append(f"- total_energy: {analysis['pod']['total_energy']:.6f}")
    lines.append("")
    lines.append("| mode | eigenvalue | energy_ratio | cumulative_energy |")
    lines.append("|---:|---:|---:|---:|")
    for row in pod_modes[:5]:
        lines.append(
            f"| {row['mode']} | {row['eigenvalue']:.6f} | {row['energy_ratio']:.6f} | {row['cumulative_energy']:.6f} |"
        )
    lines.append("")
    lines.append("## DMD")
    lines.append(f"- spectral_radius: {analysis['dmd']['spectral_radius']:.6f}")
    lines.append(f"- stable: {analysis['dmd']['stable']}")
    lines.append("")
    lines.append("| mode | real | imag | magnitude | angle |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in dmd_modes[:5]:
        lines.append(
            f"| {row['mode']} | {row['real']:.6f} | {row['imag']:.6f} | {row['magnitude']:.6f} | {row['angle']:.6f} |"
        )
    lines.append("")
    lines.append("## Forcing-Response (Top |corr|)")
    lines.append("| forcing | response | corr | abs_corr |")
    lines.append("|---|---|---:|---:|")
    for row in corr_rows:
        lines.append(f"| {row['forcing']} | {row['response']} | {row['corr']:.6f} | {row['abs_corr']:.6f} |")
    lines.append("")
    lines.append("## Metrics Digest")
    lines.append(f"- num_records: {analysis['metrics_digest']['num_records']}")
    lines.append(f"- num_failed_checks: {analysis['metrics_digest']['num_failed_checks']}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_layer2_analysis(
    *,
    report_path: Path,
    metrics_path: Path,
    json_out: Path,
    md_out: Path | None = None,
    csv_prefix: Path | None = None,
) -> dict[str, Any]:
    report = _load_json(report_path)
    metrics = _load_metrics(metrics_path)
    x = _build_mixed_trajectory(report)
    pod = _pod_analysis(x)
    dmd = _dmd_analysis(x)
    forcing_response = _forcing_response_analysis(report)
    digest = _metrics_digest(metrics)
    analysis = {
        "candidate_name": report.get("candidate_name"),
        "trajectory_rows": int(x.shape[0]),
        "trajectory_dims": int(x.shape[1]) if x.ndim == 2 else 0,
        "pod": pod,
        "dmd": dmd,
        "forcing_response": forcing_response,
        "metrics_digest": digest,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(_render_markdown(analysis), encoding="utf-8")

    if csv_prefix is not None:
        _write_bucket_csv(csv_prefix.with_name(csv_prefix.name + ".bucket_table.csv"), forcing_response["bucket_rows"])
        _write_modes_csv(csv_prefix.with_name(csv_prefix.name + ".pod_modes.csv"), pod["modes"])
        _write_modes_csv(csv_prefix.with_name(csv_prefix.name + ".dmd_modes.csv"), dmd["modes"])
        _write_modes_csv(csv_prefix.with_name(csv_prefix.name + ".forcing_corr.csv"), forcing_response["correlations"])

    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline layer-2 (POD/DMD/forcing-response) analysis for one Luma run.")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--csv-prefix", type=Path, default=None)
    args = parser.parse_args()

    analysis = run_layer2_analysis(
        report_path=args.report_json,
        metrics_path=args.metrics_jsonl,
        json_out=args.json_out,
        md_out=args.md_out,
        csv_prefix=args.csv_prefix,
    )
    print(json.dumps({"layer2": {"pod_rank": analysis["pod"]["effective_rank"], "dmd_radius": analysis["dmd"]["spectral_radius"]}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
