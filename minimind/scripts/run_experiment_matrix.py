#!/usr/bin/env python3
"""
Phase 4 Extensions 实验矩阵自动 runner
串行跑 24 个实验，每个 1500 步，提取关键指标，输出到统一 results.jsonl
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "/home/kt/ai/.venvs/luma-global/bin/python"
TRAINER = str(Path(__file__).resolve().parent.parent / "trainer" / "train_luma_refactor.py")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "experiment_matrix_20260403"

# Phase 4 基线参数（所有实验共享）
BASE_ARGS = [
    "--phase", "4",
    "--iters", "1500",
    "--use_gradient_checkpointing", "1",
    "--log_interval", "50",
    "--dod_interval", "200",
    "--batch_size", "4",
]

# ─── 实验定义 ─────────────────────────────────────────────

EXPERIMENTS = {
    # A组：Rollout Loss 回归
    "A1": {
        "desc": "rollout basic",
        "args": ["--self_rollout_weight", "0.1"],
    },
    "A2": {
        "desc": "rollout near3",
        "args": ["--self_rollout_weight", "0.1", "--self_rollout_weighting_mode", "near3"],
    },
    "A3": {
        "desc": "rollout near3 strong",
        "args": ["--self_rollout_weight", "0.2", "--self_rollout_weighting_mode", "near3"],
    },
    "A4": {
        "desc": "rollout + zone guard",
        "args": ["--self_rollout_weight", "0.1", "--rollout_zone_weight", "0.01"],
    },
    "A5": {
        "desc": "rollout + vitality",
        "args": ["--self_rollout_weight", "0.1", "--trajectory_vitality_weight", "0.01"],
    },
    "A6": {
        "desc": "rollout full suite",
        "args": [
            "--self_rollout_weight", "0.1", "--self_rollout_weighting_mode", "near3",
            "--rollout_zone_weight", "0.01", "--trajectory_vitality_weight", "0.01",
        ],
    },

    # B组：Progress-Shape 与退出决策
    "B1": {
        "desc": "progress-shape",
        "args": ["--self_progress_shape_weight", "0.05"],
    },
    "B2": {
        "desc": "progress + exit readout",
        "args": ["--self_progress_shape_weight", "0.05", "--enable_progress_exit_readout", "1"],
    },
    "B3": {
        "desc": "progress + exit + backtrack",
        "args": [
            "--self_progress_shape_weight", "0.05",
            "--enable_progress_exit_readout", "1",
            "--enable_backtrack_aware_progress", "1",
        ],
    },
    "B4": {
        "desc": "progress + rollout near3",
        "args": [
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1", "--self_rollout_weighting_mode", "near3",
        ],
    },
    "B5": {
        "desc": "progress + exit + rollout near3",
        "args": [
            "--self_progress_shape_weight", "0.05", "--enable_progress_exit_readout", "1",
            "--self_rollout_weight", "0.1", "--self_rollout_weighting_mode", "near3",
        ],
    },

    # C组：状态几何与正则
    "C1": {
        "desc": "local delta consistency",
        "args": ["--self_local_delta_consistency_weight", "0.01"],
    },
    "C2": {
        "desc": "curvature reg",
        "args": ["--self_local_curvature_weight", "0.005"],
    },
    "C3": {
        "desc": "sigreg on c_t",
        "args": ["--enable_sigreg_ct", "1", "--sigreg_ct_weight", "0.03"],
    },
    "C4": {
        "desc": "full geometry suite",
        "args": [
            "--self_local_delta_consistency_weight", "0.01",
            "--self_local_curvature_weight", "0.005",
            "--enable_sigreg_ct", "1", "--sigreg_ct_weight", "0.03",
        ],
    },

    # D组：数据集多样性（需要对应数据文件存在）
    "D1": {
        "desc": "diag + math 50:50",
        "args": ["--data_path", "../dataset/pretrain_diag_math.jsonl"],
    },
    "D2": {
        "desc": "diag + emotion + persona 40:30:30",
        "args": ["--data_path", "../dataset/pretrain_diag_emo_persona.jsonl"],
    },
    "D3": {
        "desc": "full mix 25:25:25:25",
        "args": ["--data_path", "../dataset/pretrain_full_mix.jsonl"],
    },
    "D4": {
        "desc": "full mix + rollout legacy",
        "args": [
            "--data_path", "../dataset/pretrain_full_mix.jsonl",
            "--self_rollout_weight", "0.1",
        ],
    },
    "D5": {
        "desc": "full mix + progress-shape",
        "args": [
            "--data_path", "../dataset/pretrain_full_mix.jsonl",
            "--self_progress_shape_weight", "0.05",
        ],
    },

    # E组：根据 A-C 结果的最优组合
    "E1": {
        "desc": "progress-shape only (B1 repeat for data baseline)",
        "args": [
            "--self_progress_shape_weight", "0.05",
        ],
    },
    "E2": {
        "desc": "progress + local consistency (B1+C1)",
        "args": [
            "--self_progress_shape_weight", "0.05",
            "--self_local_delta_consistency_weight", "0.01",
        ],
    },
    "E3": {
        "desc": "progress + rollout legacy (B1+A1)",
        "args": [
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
    "E4": {
        "desc": "progress + local consistency + full mix data",
        "args": [
            "--data_path", "../dataset/pretrain_full_mix.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_local_delta_consistency_weight", "0.01",
        ],
    },

    # F组：根据 A-C 结果的最优组合验证
    "F1": {
        "desc": "progress + local consistency (B1+C1)",
        "args": [
            "--self_progress_shape_weight", "0.05",
            "--self_local_delta_consistency_weight", "0.01",
        ],
    },
    "F2": {
        "desc": "progress + rollout legacy (B1+A1)",
        "args": [
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
    "F3": {
        "desc": "progress + local consistency + full mix data (B1+C1+D3)",
        "args": [
            "--data_path", "../dataset/pretrain_full_mix.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_local_delta_consistency_weight", "0.01",
        ],
    },
    "F4": {
        "desc": "progress + rollout + local consistency (B1+A1+C1)",
        "args": [
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
            "--self_local_delta_consistency_weight", "0.01",
        ],
    },
    "F5": {
        "desc": "progress + rollout + local consistency + full mix (all best)",
        "args": [
            "--data_path", "../dataset/pretrain_full_mix.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
            "--self_local_delta_consistency_weight", "0.01",
        ],
    },
    "F6": {
        "desc": "progress + sigreg_ct (B1+C3)",
        "args": [
            "--self_progress_shape_weight", "0.05",
            "--enable_sigreg_ct", "1", "--sigreg_ct_weight", "0.03",
        ],
    },

    # G组：新数据源验证（使用 F2 最优配置 progress+rollout 作为 baseline）
    "G1": {
        "desc": "F2 + full_mix_v2 (7类均匀)",
        "args": [
            "--data_path", "../dataset/pretrain_full_mix.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
    "G2": {
        "desc": "F2 + reasoning_mix (math+arc+python为主)",
        "args": [
            "--data_path", "../dataset/pretrain_reasoning_mix.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
    "G3": {
        "desc": "F2 + chinese_heavy (中文为主)",
        "args": [
            "--data_path", "../dataset/pretrain_chinese_heavy.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
    "G4": {
        "desc": "F2 + full_mix_large (全量不上采样)",
        "args": [
            "--data_path", "../dataset/pretrain_full_mix_large.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
    "G5": {
        "desc": "F2 + diag_math (persona+math, 旧配比对照)",
        "args": [
            "--data_path", "../dataset/pretrain_diag_math.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
    # H组：G5 基础 + 第三类数据 5%，验证 312M 3类数据上限
    "H1": {
        "desc": "G5 + scifi 5%",
        "args": [
            "--data_path", "../dataset/pretrain_h_scifi.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
    "H2": {
        "desc": "G5 + python 5%",
        "args": [
            "--data_path", "../dataset/pretrain_h_python.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
    "H3": {
        "desc": "G5 + ARC 5%",
        "args": [
            "--data_path", "../dataset/pretrain_h_arc.jsonl",
            "--self_progress_shape_weight", "0.05",
            "--self_rollout_weight", "0.1",
        ],
    },
}


def parse_log(log_path: str) -> dict:
    """从训练 log 提取关键指标。"""
    metrics = {
        "final_loss_lm": None,
        "dod_snapshots": [],
        "final_dod_rank": None,
        "final_mode1": None,
        "final_dmd_radius": None,
        "final_ratio": None,
        "loss_trajectory": [],
    }
    with open(log_path) as f:
        for line in f:
            # DOD/DMD 行
            m = re.search(r'\[DOD/DMD step (\d+)\]\s+dod_rank=(\d+)\s+mode1_energy=([\d.]+)%\s+grad_dmd_radius=([\d.]+)', line)
            if m:
                metrics["dod_snapshots"].append({
                    "step": int(m.group(1)),
                    "rank": int(m.group(2)),
                    "mode1": float(m.group(3)),
                    "radius": float(m.group(4)),
                })
            # loss 行
            m = re.search(r'\[(\d+)/\d+\] loss_lm=([\d.]+).*ratio=([\d.]+)', line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                ratio = float(m.group(3))
                metrics["loss_trajectory"].append({"step": step, "loss": loss, "ratio": ratio})
            # Done 行
            m = re.search(r'Done\.\s+\d+ steps, loss_lm=([\d.]+)', line)
            if m:
                metrics["final_loss_lm"] = float(m.group(1))

    if metrics["dod_snapshots"]:
        last = metrics["dod_snapshots"][-1]
        metrics["final_dod_rank"] = last["rank"]
        metrics["final_mode1"] = last["mode1"]
        metrics["final_dmd_radius"] = last["radius"]
    if metrics["loss_trajectory"]:
        metrics["final_ratio"] = metrics["loss_trajectory"][-1]["ratio"]

    return metrics


def judge(m: dict, exp_id: str = "") -> str:
    rank = m.get("final_dod_rank")
    mode1 = m.get("final_mode1")
    loss = m.get("final_loss_lm")
    if rank is None:
        return "NO_DATA"
    # rank < 3 始终是坍缩
    if rank < 3:
        return "FAIL"
    # mode1 > 95 始终是坍缩
    if mode1 is not None and mode1 > 95:
        return "FAIL"
    # 多类型数据实验（D/G 组）允许更高的 loss，因为数据更难
    uses_mixed_data = exp_id.startswith(("D", "G"))
    loss_fail = 8.0 if uses_mixed_data else 1.40
    loss_warn = 5.0 if uses_mixed_data else 1.30
    mode1_warn = 85 if uses_mixed_data else 80
    if loss is not None and loss > loss_fail:
        return "FAIL"
    if (mode1 is not None and mode1 > mode1_warn) or (loss is not None and loss > loss_warn):
        return "WARN"
    return "PASS"


def run_experiment(exp_id: str, exp: dict, results_dir: Path) -> dict:
    log_path = results_dir / f"{exp_id}.log"
    print(f"\n{'='*60}")
    print(f"[{exp_id}] {exp['desc']}")
    print(f"{'='*60}")

    # 检查 D 组数据文件是否存在
    for i, arg in enumerate(exp["args"]):
        if arg == "--data_path":
            data_file = Path(__file__).resolve().parent.parent / "trainer" / exp["args"][i + 1]
            if not data_file.exists():
                # 尝试绝对路径
                data_file2 = Path(exp["args"][i + 1].replace("..", str(Path(__file__).resolve().parent.parent)))
                if not data_file2.exists():
                    print(f"  SKIP: data file not found: {exp['args'][i + 1]}")
                    return {"id": exp_id, "desc": exp["desc"], "status": "SKIP_NO_DATA"}

    # 构建 Phase 4 的参数（不用 phase 6 框架了，直接用 phase 4）
    cmd = [PYTHON, TRAINER] + ["--phase", "4"] + BASE_ARGS[2:] + exp["args"]

    print(f"  CMD: {' '.join(cmd[-len(exp['args']):])} ")
    t0 = time.time()

    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(Path(TRAINER).parent))

    dt = time.time() - t0
    print(f"  Completed in {dt/60:.1f} min (exit code {proc.returncode})")

    metrics = parse_log(str(log_path))
    verdict = judge(metrics, exp_id)
    status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "NO_DATA": "??"}.get(verdict, "??")

    result = {
        "id": exp_id,
        "desc": exp["desc"],
        "verdict": verdict,
        "final_loss_lm": metrics["final_loss_lm"],
        "final_dod_rank": metrics["final_dod_rank"],
        "final_mode1": metrics["final_mode1"],
        "final_dmd_radius": metrics["final_dmd_radius"],
        "final_ratio": metrics["final_ratio"],
        "dod_snapshots": metrics["dod_snapshots"],
        "time_min": round(dt / 60, 1),
    }

    print(f"  {status_icon} {verdict}: loss={metrics['final_loss_lm']}, rank={metrics['final_dod_rank']}, mode1={metrics['final_mode1']}%, radius={metrics['final_dmd_radius']}, ratio={metrics['final_ratio']}")
    return result


def main():
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.jsonl"

    # 支持从特定实验开始（断点续跑）
    start_from = sys.argv[1] if len(sys.argv) > 1 else None
    skip_groups = set(sys.argv[2:]) if len(sys.argv) > 2 else set()

    # 加载已完成的实验
    done = set()
    if results_file.exists():
        for line in open(results_file):
            r = json.loads(line)
            done.add(r["id"])

    started = start_from is None
    all_results = []

    for exp_id in EXPERIMENTS:
        if not started:
            if exp_id == start_from:
                started = True
            else:
                continue

        if exp_id in done:
            print(f"[{exp_id}] Already done, skipping")
            continue

        group = exp_id[0]
        if group in skip_groups:
            print(f"[{exp_id}] Group {group} skipped")
            continue

        result = run_experiment(exp_id, EXPERIMENTS[exp_id], results_dir)
        all_results.append(result)

        with open(results_file, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 打印汇总
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'ID':<5} {'Verdict':<8} {'Loss':<8} {'Rank':<5} {'Mode1':<8} {'Ratio':<8} {'Desc'}")
    print("-" * 70)
    for r in all_results:
        loss = f"{r['final_loss_lm']:.2f}" if r['final_loss_lm'] else "N/A"
        rank = str(r.get('final_dod_rank', '?'))
        mode1 = f"{r['final_mode1']:.1f}%" if r.get('final_mode1') is not None else "N/A"
        ratio = f"{r['final_ratio']:.1f}" if r.get('final_ratio') is not None else "N/A"
        print(f"{r['id']:<5} {r['verdict']:<8} {loss:<8} {rank:<5} {mode1:<8} {ratio:<8} {r['desc']}")


if __name__ == "__main__":
    main()
