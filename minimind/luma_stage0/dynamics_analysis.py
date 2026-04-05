"""
Luma 训练动力学在线分析 v2
========================
v1 对三维梯度范数做 POD，rank 上限永远是 3，区分度不足。
v2 改为逐层梯度范数（dim ≈ 30-40），POD rank 真正反映独立更新方向数。
新增 c_t batch 方差追踪和退出深度分布追踪。

POD rank（= DOD rank）：
  衡量参数更新在多少个独立方向上运动。
  - rank=1：所有梯度在同一方向上（单一 loss 主导，坏）
  - rank ≈ dim/2+：各层梯度较独立（健康）

DMD：
  分析轨迹的动力学模态。
  spectral_radius < 1.0 → 稳定收敛
  spectral_radius ≈ 1.0 → 振荡
  spectral_radius > 1.0 → 发散风险

所有数值计算在 CPU (numpy) 上完成，不占 GPU。
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 核心分析函数
# ---------------------------------------------------------------------------

def _pod_analysis(x: np.ndarray) -> Dict[str, Any]:
    """
    Proper Orthogonal Decomposition.
    x: shape (n_steps, n_features)
    """
    if x.shape[0] < 3 or x.shape[1] < 1:
        return {"effective_rank": -1, "energy_top5_pct": [], "sv_top5": [],
                "energy_mode1_pct": 100.0, "total_energy": 0.0}
    x_c = x - x.mean(axis=0, keepdims=True)
    cov = (x_c.T @ x_c) / max(1, x_c.shape[0] - 1)
    try:
        eigvals, _ = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return {"effective_rank": -1, "energy_top5_pct": [], "sv_top5": [],
                "energy_mode1_pct": 100.0, "total_energy": 0.0}
    eigvals = np.maximum(eigvals[::-1], 0.0)
    total = float(eigvals.sum())
    if total <= 0:
        return {"effective_rank": 0, "energy_top5_pct": [], "sv_top5": [],
                "energy_mode1_pct": 100.0, "total_energy": 0.0}
    ratios = eigvals / total
    effective_rank = int((ratios > 1e-3).sum())
    energy_top5 = (ratios[:5] * 100).tolist()
    sv_top5 = np.sqrt(np.maximum(eigvals[:5], 0.0)).tolist()
    energy_mode1_pct = float(ratios[0] * 100) if len(ratios) > 0 else 100.0
    return {
        "effective_rank": effective_rank,
        "energy_top5_pct": [round(v, 2) for v in energy_top5],
        "sv_top5": [round(v, 4) for v in sv_top5],
        "energy_mode1_pct": round(energy_mode1_pct, 2),
        "total_energy": round(total, 6),
    }


def _dmd_analysis(x: np.ndarray) -> Dict[str, Any]:
    """
    Dynamic Mode Decomposition.
    x: shape (n_steps, n_features)
    """
    if x.shape[0] < 4 or x.shape[1] < 1:
        return {"spectral_radius": float("nan"), "stable": None, "eig_mags_top5": []}
    x_c = x - x.mean(axis=0, keepdims=True)
    x0 = x_c[:-1].T
    x1 = x_c[1:].T
    if np.allclose(x0, 0.0) or np.allclose(x1, 0.0):
        return {"spectral_radius": 0.0, "stable": True, "eig_mags_top5": []}
    try:
        U, s, Vt = np.linalg.svd(x0, full_matrices=False)
        threshold = 1e-10 * s[0] if s[0] > 0 else 1e-10
        r = int((s > threshold).sum())
        r = min(r, min(x0.shape) - 1, 10)
        if r < 1:
            return {"spectral_radius": 0.0, "stable": True, "eig_mags_top5": []}
        U_r, s_r, Vt_r = U[:, :r], s[:r], Vt[:r, :]
        A_tilde = U_r.T @ x1 @ Vt_r.T @ np.diag(1.0 / s_r)
        eigvals = np.linalg.eigvals(A_tilde)
        mags = np.sort(np.abs(eigvals))[::-1]
        spectral_radius = float(mags[0]) if len(mags) > 0 else 0.0
    except np.linalg.LinAlgError:
        return {"spectral_radius": float("nan"), "stable": None, "eig_mags_top5": []}
    return {
        "spectral_radius": round(spectral_radius, 6),
        "stable": bool(spectral_radius < 1.0 + 1e-3),
        "eig_mags_top5": [round(float(m), 4) for m in mags[:5]],
    }


# ---------------------------------------------------------------------------
# v1 兼容：三维梯度范数追踪器（保留以防回退）
# ---------------------------------------------------------------------------

class GradTrajectoryTracker:
    """
    v1 追踪器：每步收集 (compress, shared, reasoning) 三维梯度范数。
    POD 在 3 维数据上最大 rank=3，区分度有限。
    保留用于向后兼容和快速日志输出。
    """

    def __init__(self, window: int = 200):
        self.window = window
        self._buf: deque = deque(maxlen=window)
        self._steps: deque = deque(maxlen=window)

    def update(self, step: int, grad_metrics: Dict[str, float]) -> None:
        nc = grad_metrics.get("grad_norm_compress", 0.0)
        ns = grad_metrics.get("grad_norm_shared", 0.0)
        nr = grad_metrics.get("grad_norm_reasoning", 0.0)
        if math.isfinite(nc) and math.isfinite(ns) and math.isfinite(nr):
            self._buf.append([nc, ns, nr])
            self._steps.append(step)

    def analyze(self) -> Dict[str, Any]:
        if len(self._buf) < 3:
            return {}
        x = np.array(list(self._buf), dtype=np.float64)
        pod = _pod_analysis(x)
        dmd = _dmd_analysis(x)
        return {
            "window_steps": len(self._buf),
            "first_step": self._steps[0],
            "last_step": self._steps[-1],
            "pod": pod,
            "dmd": dmd,
            "dod_rank": pod["effective_rank"],
            "energy_mode1_pct": pod.get("energy_mode1_pct", 100.0),
            "dmd_spectral_radius": dmd.get("spectral_radius", float("nan")),
        }


# ---------------------------------------------------------------------------
# v2: 逐层梯度范数追踪器
# ---------------------------------------------------------------------------

class LayerGradTracker:
    """
    v2 追踪器：每步收集逐层梯度范数向量（dim ≈ 30-40）。
    POD rank 在高维空间上真正有区分度。

    用法：
        tracker.update(step, {"compress_0": 0.12, "compress_1": 0.08, ..., "reason_shared_0": 0.05, ...})
    """

    def __init__(self, window: int = 200):
        self.window = window
        self._buf: deque = deque(maxlen=window)
        self._steps: deque = deque(maxlen=window)
        self._layer_names: Optional[List[str]] = None

    def update(self, step: int, layer_norms: Dict[str, float]) -> None:
        if not layer_norms:
            return
        if self._layer_names is None:
            self._layer_names = sorted(layer_norms.keys())
        vec = [layer_norms.get(n, 0.0) for n in self._layer_names]
        if all(math.isfinite(v) for v in vec):
            self._buf.append(vec)
            self._steps.append(step)

    def analyze(self) -> Dict[str, Any]:
        if len(self._buf) < 3:
            return {}
        x = np.array(list(self._buf), dtype=np.float64)
        pod = _pod_analysis(x)
        dmd = _dmd_analysis(x)

        # 逐层统计
        layer_stats = {}
        if self._layer_names and x.shape[0] >= 2:
            means = x.mean(axis=0)
            stds = x.std(axis=0)
            for i, name in enumerate(self._layer_names):
                layer_stats[name] = {
                    "mean_norm": round(float(means[i]), 6),
                    "std_norm": round(float(stds[i]), 6),
                }
            # 检测死层：均值梯度 < 全局均值的 1%
            global_mean = means.mean()
            dead_layers = [n for i, n in enumerate(self._layer_names)
                           if global_mean > 0 and means[i] < global_mean * 0.01]
        else:
            dead_layers = []

        return {
            "window_steps": len(self._buf),
            "first_step": self._steps[0],
            "last_step": self._steps[-1],
            "pod": pod,
            "dmd": dmd,
            "dod_rank": pod["effective_rank"],
            "energy_mode1_pct": pod.get("energy_mode1_pct", 100.0),
            "dmd_spectral_radius": dmd.get("spectral_radius", float("nan")),
            "n_layers": len(self._layer_names) if self._layer_names else 0,
            "dead_layers": dead_layers,
            "layer_stats": layer_stats,
        }


# ---------------------------------------------------------------------------
# v2: c_t 状态追踪器（修复 batch 平均问题）
# ---------------------------------------------------------------------------

class CtStateTracker:
    """
    v2: 不再只追踪 batch 平均 c_t。
    同时记录 batch 内 c_t 范数的方差——这是 adaptive computation 健康度的核心指标。
    方差趋近 0 → 所有 token 的 c_t 趋同 → adaptive depth 失效。
    """

    def __init__(self, window: int = 100, proj_dim: int = 64):
        self.window = window
        self.proj_dim = proj_dim
        self._buf: deque = deque(maxlen=window)
        self._batch_stats: deque = deque(maxlen=window)
        self._steps: deque = deque(maxlen=window)
        self._proj: Optional[np.ndarray] = None

    def update(self, step: int, c_t: torch.Tensor) -> None:
        """c_t: (batch, c_t_dim) or (batch, seq, c_t_dim)"""
        ct_f = c_t.detach().float()
        if ct_f.dim() == 3:
            ct_f = ct_f.mean(dim=1)  # (batch, c_t_dim)

        # batch 内 c_t 范数统计
        ct_norms = ct_f.norm(dim=-1)  # (batch,)
        batch_var = ct_norms.var().item() if ct_norms.numel() > 1 else 0.0
        batch_mean_norm = ct_norms.mean().item()
        batch_max_norm = ct_norms.max().item()
        batch_min_norm = ct_norms.min().item()

        # 投影均值用于 DMD 轨迹分析
        v = ct_f.mean(dim=0).cpu().numpy()
        if self._proj is None:
            rng = np.random.default_rng(42)
            self._proj = rng.standard_normal((v.shape[0], self.proj_dim)) / math.sqrt(self.proj_dim)
        v_proj = v @ self._proj

        if np.isfinite(v_proj).all() and math.isfinite(batch_var):
            self._buf.append(v_proj)
            self._batch_stats.append({
                "var": batch_var,
                "mean_norm": batch_mean_norm,
                "max_norm": batch_max_norm,
                "min_norm": batch_min_norm,
            })
            self._steps.append(step)

    def analyze(self) -> Dict[str, Any]:
        if len(self._buf) < 5:
            return {}
        x = np.array(list(self._buf), dtype=np.float64)
        dmd = _dmd_analysis(x)

        stats = list(self._batch_stats)
        batch_vars = [s["var"] for s in stats]
        batch_norms = [s["mean_norm"] for s in stats]

        # 方差趋势：正 = 方差增长（健康），负 = 方差萎缩（adaptive depth 退化）
        if len(batch_vars) >= 40:
            var_trend = float(np.mean(batch_vars[-20:]) - np.mean(batch_vars[:20]))
        else:
            var_trend = 0.0

        return {
            "window_steps": len(self._buf),
            "first_step": self._steps[0],
            "last_step": self._steps[-1],
            "ct_dmd": dmd,
            "ct_batch_var_mean": round(float(np.mean(batch_vars)), 6),
            "ct_batch_var_trend": round(var_trend, 6),
            "ct_mean_norm": round(float(np.mean(batch_norms)), 4),
            "ct_norm_spread": round(
                float(np.mean([s["max_norm"] - s["min_norm"] for s in stats])), 4
            ),
        }


# ---------------------------------------------------------------------------
# v2: 退出深度分布追踪器
# ---------------------------------------------------------------------------

class ExitDepthTracker:
    """
    追踪每步的退出循环深度分布。
    健康信号：entropy 高、std > 0、直方图有多峰结构。
    不健康信号：所有 token 在同一深度退出（entropy ≈ 0）。
    """

    def __init__(self, window: int = 200, max_loops: int = 20):
        self.window = window
        self.max_loops = max_loops
        self._buf: deque = deque(maxlen=window)
        self._steps: deque = deque(maxlen=window)

    def update(self, step: int, exit_loops: torch.Tensor) -> None:
        """exit_loops: (batch,) or (batch, seq) int tensor of per-token exit loop index."""
        loops = exit_loops.detach().cpu().flatten().float().numpy()
        self._buf.append(loops)
        self._steps.append(step)

    def analyze(self) -> Dict[str, Any]:
        if len(self._buf) < 5:
            return {}
        all_loops = np.concatenate(list(self._buf))
        bins = np.arange(self.max_loops + 2) - 0.5
        hist, _ = np.histogram(all_loops, bins=bins)
        total = max(hist.sum(), 1)
        hist_norm = hist / total

        mean_depth = float(all_loops.mean())
        std_depth = float(all_loops.std())

        # Shannon entropy（越高 = 退出分布越多样 = 越健康）
        nonzero = hist_norm[hist_norm > 0]
        entropy = -float((nonzero * np.log(nonzero + 1e-12)).sum())

        # 峰值数量检测
        is_peak = np.zeros(len(hist_norm), dtype=bool)
        for i in range(1, len(hist_norm) - 1):
            if hist_norm[i] > hist_norm[i - 1] and hist_norm[i] > hist_norm[i + 1]:
                is_peak[i] = True
        if hist_norm[0] > hist_norm[1]:
            is_peak[0] = True
        n_peaks = int(is_peak.sum())

        return {
            "window_steps": len(self._buf),
            "first_step": self._steps[0],
            "last_step": self._steps[-1],
            "mean_depth": round(mean_depth, 2),
            "std_depth": round(std_depth, 2),
            "depth_entropy": round(entropy, 4),
            "n_peaks": n_peaks,
            "depth_histogram": [round(float(h), 4) for h in hist_norm[:self.max_loops + 1]],
        }


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def render_dynamics_report(
    grad_history: List[Dict[str, Any]],
    ct_history: List[Dict[str, Any]],
    phase: int,
    total_steps: int,
    layer_grad_history: Optional[List[Dict[str, Any]]] = None,
    exit_depth_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not grad_history:
        return {}

    dod_ranks = [r["dod_rank"] for r in grad_history if "dod_rank" in r]
    energy_m1 = [r["energy_mode1_pct"] for r in grad_history if "energy_mode1_pct" in r]
    dmd_radii = [r["dmd_spectral_radius"] for r in grad_history
                 if "dmd_spectral_radius" in r and math.isfinite(r["dmd_spectral_radius"])]

    ct_radii = [r["ct_dmd"]["spectral_radius"] for r in ct_history
                if "ct_dmd" in r and math.isfinite(r["ct_dmd"].get("spectral_radius", float("nan")))]

    final = grad_history[-1] if grad_history else {}

    report = {
        "phase": phase,
        "total_steps": total_steps,
        "final_dod_rank": final.get("dod_rank", -1),
        "final_energy_mode1_pct": final.get("energy_mode1_pct", 100.0),
        "final_dmd_spectral_radius": final.get("dmd_spectral_radius", float("nan")),
        "dod_rank_trajectory": dod_ranks,
        "energy_mode1_pct_trajectory": [round(v, 1) for v in energy_m1],
        "dmd_spectral_radius_trajectory": [round(v, 4) for v in dmd_radii],
        "ct_dmd_spectral_radius_trajectory": [round(v, 4) for v in ct_radii],
        "verdict": _verdict(final.get("dod_rank", -1), final.get("energy_mode1_pct", 100.0), phase),
        "full_snapshots": grad_history,
    }

    # v2 逐层分析
    if layer_grad_history:
        lg_final = layer_grad_history[-1] if layer_grad_history else {}
        report["v2_layer_dod_rank"] = lg_final.get("dod_rank", -1)
        report["v2_layer_energy_mode1_pct"] = lg_final.get("energy_mode1_pct", 100.0)
        report["v2_layer_n_dims"] = lg_final.get("n_layers", 0)
        report["v2_layer_dead_layers"] = lg_final.get("dead_layers", [])
        report["v2_layer_dod_rank_trajectory"] = [
            r.get("dod_rank", -1) for r in layer_grad_history
        ]
        report["v2_layer_energy_mode1_trajectory"] = [
            round(r.get("energy_mode1_pct", 100.0), 1) for r in layer_grad_history
        ]
        report["v2_verdict"] = _verdict_v2(lg_final, phase)

    # v2 c_t batch 方差
    if ct_history:
        last_ct = ct_history[-1] if ct_history else {}
        report["ct_batch_var_mean"] = last_ct.get("ct_batch_var_mean", 0.0)
        report["ct_batch_var_trend"] = last_ct.get("ct_batch_var_trend", 0.0)
        report["ct_norm_spread"] = last_ct.get("ct_norm_spread", 0.0)

    # v2 退出深度
    if exit_depth_history:
        last_exit = exit_depth_history[-1] if exit_depth_history else {}
        report["exit_mean_depth"] = last_exit.get("mean_depth", 0.0)
        report["exit_std_depth"] = last_exit.get("std_depth", 0.0)
        report["exit_entropy"] = last_exit.get("depth_entropy", 0.0)
        report["exit_n_peaks"] = last_exit.get("n_peaks", 0)

    return report


def _verdict(dod_rank: int, energy_m1_pct: float, phase: int) -> str:
    """v1 判定（向后兼容，基于三维 POD）。"""
    if dod_rank < 0:
        return "INSUFFICIENT_DATA"
    effective_phase = 3 if phase == 35 else (4 if phase in (5, 6) else phase)
    target_rank = effective_phase + 1
    if dod_rank >= target_rank:
        if energy_m1_pct < 80:
            return f"PASS: dod_rank={dod_rank} (target>={target_rank}), energy均匀 (mode1={energy_m1_pct:.1f}%)"
        else:
            return f"WARN: dod_rank={dod_rank} ok但能量仍集中 mode1={energy_m1_pct:.1f}% (目标<80%)"
    else:
        return f"FAIL: dod_rank={dod_rank} < target {target_rank}, mode1={energy_m1_pct:.1f}%"


def _verdict_v2(layer_snap: Dict[str, Any], phase: int) -> str:
    """v2 判定：基于逐层 POD，更有区分度。"""
    rank = layer_snap.get("dod_rank", -1)
    n_layers = layer_snap.get("n_layers", 0)
    energy_m1 = layer_snap.get("energy_mode1_pct", 100.0)
    dead = layer_snap.get("dead_layers", [])

    if rank < 0:
        return "INSUFFICIENT_DATA"

    parts = []
    # rank 判定：rank 至少应达到 dim 的 30%
    rank_ratio = rank / max(n_layers, 1)
    if rank_ratio >= 0.5:
        parts.append(f"RANK_GOOD: {rank}/{n_layers} ({rank_ratio:.0%})")
    elif rank_ratio >= 0.3:
        parts.append(f"RANK_OK: {rank}/{n_layers} ({rank_ratio:.0%})")
    else:
        parts.append(f"RANK_LOW: {rank}/{n_layers} ({rank_ratio:.0%})")

    # mode1 判定
    if energy_m1 < 50:
        parts.append(f"mode1={energy_m1:.1f}% 均匀")
    elif energy_m1 < 70:
        parts.append(f"mode1={energy_m1:.1f}% 可接受")
    else:
        parts.append(f"mode1={energy_m1:.1f}% 集中")

    # 死层
    if dead:
        parts.append(f"DEAD_LAYERS: {dead}")

    return " | ".join(parts)


def save_report(report: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_markdown(report: Dict[str, Any]) -> str:
    p = report.get("phase", "?")
    steps = report.get("total_steps", "?")
    verdict = report.get("verdict", "?")
    rank_traj = report.get("dod_rank_trajectory", [])
    e1_traj = report.get("energy_mode1_pct_trajectory", [])

    lines = [
        f"# Luma Phase {p} Dynamics Report",
        f"*{steps} training steps*",
        "",
    ]

    # v2 逐层分析（优先展示）
    if "v2_layer_dod_rank" in report:
        v2_rank = report["v2_layer_dod_rank"]
        v2_e1 = report["v2_layer_energy_mode1_pct"]
        v2_dims = report["v2_layer_n_dims"]
        v2_verdict = report.get("v2_verdict", "?")
        v2_rank_traj = report.get("v2_layer_dod_rank_trajectory", [])
        v2_e1_traj = report.get("v2_layer_energy_mode1_trajectory", [])
        dead = report.get("v2_layer_dead_layers", [])

        lines.extend([
            "## v2 逐层分析 (推荐)",
            f"| 指标 | 值 | 说明 |",
            f"|---|---|---|",
            f"| 逐层 DOD rank | **{v2_rank}** / {v2_dims} | 梯度独立方向数 / 总层数 |",
            f"| 第一模态能量 | {v2_e1}% | <50% 优秀, <70% 可接受 |",
            f"| 死层 | {dead if dead else '无'} | 梯度 <1% 均值 |",
            "",
            f"**v2 判定**: {v2_verdict}",
            "",
            f"### Rank 轨迹: `{v2_rank_traj}`",
            f"### Mode1% 轨迹: `{v2_e1_traj}`",
            "",
        ])

    # c_t batch 方差
    if "ct_batch_var_mean" in report:
        lines.extend([
            "## c_t Batch 方差",
            f"| 指标 | 值 | 说明 |",
            f"|---|---|---|",
            f"| batch 方差均值 | {report['ct_batch_var_mean']:.6f} | >0 = adaptive depth 有效 |",
            f"| 方差趋势 | {report['ct_batch_var_trend']:.6f} | 正=增长(好), 负=萎缩(坏) |",
            f"| norm 散布 | {report['ct_norm_spread']:.4f} | max-min 范数差 |",
            "",
        ])

    # 退出深度
    if "exit_entropy" in report:
        lines.extend([
            "## 退出深度分布",
            f"| 指标 | 值 | 说明 |",
            f"|---|---|---|",
            f"| 均值深度 | {report['exit_mean_depth']:.2f} | |",
            f"| 标准差 | {report['exit_std_depth']:.2f} | >0 = 变深度 |",
            f"| entropy | {report['exit_entropy']:.4f} | 越高越多样 |",
            f"| 峰值数 | {report['exit_n_peaks']} | >1 = 多峰(好) |",
            "",
        ])

    # v1 兼容
    lines.extend([
        "## v1 三维分析 (兼容)",
        f"判定: {verdict}",
        f"### DOD Rank 轨迹: `{rank_traj}`",
        f"### Mode1% 轨��: `{e1_traj}`",
        "",
    ])

    return "\n".join(lines)
