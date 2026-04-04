"""
Luma 训练动力学在线分析
========================
在训练循环中实时收集梯度轨迹和 c_t 状态轨迹，
定期运行 POD（本文档 = DOD/DOD-rank）和 DMD 分析，
训练结束后自动生成报告。

POD rank（= 计划中的 DOD rank）：
  衡量参数更新在多少个独立方向上运动。
  - rank=1：所有梯度在同一方向上（单一 loss 主导，坏）
  - rank=3+：各区域梯度独立（健康）

DMD：
  分析 c_t 状态轨迹的动力学模态。
  spectral_radius < 1.0 → 稳定收敛
  spectral_radius ≈ 1.0 → 振荡
  spectral_radius > 1.0 → 发散风险
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
# 复用 analyze_luma_dynamics_layer2.py 的核心分析函数（不依赖那个文件，独立实现）
# ---------------------------------------------------------------------------

def _pod_analysis(x: np.ndarray) -> Dict[str, Any]:
    """
    Proper Orthogonal Decomposition（即 DOD）。
    x: shape (n_steps, n_features)
    返回 effective_rank, energy_distribution 等。
    """
    if x.shape[0] < 3 or x.shape[1] < 1:
        return {"effective_rank": -1, "energy_top5_pct": [], "sv_top5": [], "total_energy": 0.0}
    x_c = x - x.mean(axis=0, keepdims=True)
    cov = (x_c.T @ x_c) / max(1, x_c.shape[0] - 1)
    try:
        eigvals, _ = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return {"effective_rank": -1, "energy_top5_pct": [], "sv_top5": [], "total_energy": 0.0}
    eigvals = np.maximum(eigvals[::-1], 0.0)   # 降序，去负
    total = float(eigvals.sum())
    if total <= 0:
        return {"effective_rank": 0, "energy_top5_pct": [], "sv_top5": [], "total_energy": 0.0}
    ratios = eigvals / total
    effective_rank = int((ratios > 1e-3).sum())
    energy_top5 = (ratios[:5] * 100).tolist()
    sv_top5 = np.sqrt(np.maximum(eigvals[:5], 0.0)).tolist()
    # 能量比：第一模态占比（越高说明越塌缩）
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
    Dynamic Mode Decomposition。
    x: shape (n_steps, n_features)
    返回 spectral_radius（主导模态幅值），stable 标志，top-5 特征值。
    """
    if x.shape[0] < 4 or x.shape[1] < 1:
        return {"spectral_radius": float("nan"), "stable": None, "eig_mags_top5": []}
    x_c = x - x.mean(axis=0, keepdims=True)
    x0 = x_c[:-1].T   # (features, n-1)
    x1 = x_c[1:].T    # (features, n-1)
    if np.allclose(x0, 0.0) or np.allclose(x1, 0.0):
        return {"spectral_radius": 0.0, "stable": True, "eig_mags_top5": []}
    try:
        # SVD-based DMD（数值稳定版本）
        U, s, Vt = np.linalg.svd(x0, full_matrices=False)
        # 截断小奇异值
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
# 在线轨迹收集器
# ---------------------------------------------------------------------------

class GradTrajectoryTracker:
    """
    每步收集梯度范数三元组（compress, shared, reasoning），
    在窗口内做 POD/DMD 分析，追踪训练动力学。

    选择梯度范数三元组而非原始梯度的原因：
    - 直接存储梯度需要 313M × float32 = 1.2GB/步，不可行
    - 三个区域的梯度范数是一个低维的"动力学指纹"
    - POD rank=1 → 三区梯度完全同步（只有一条通路）
    - POD rank=3 → 三区梯度独立驱动（健康）
    """

    def __init__(self, window: int = 200):
        self.window = window
        # 每步一条记录：[grad_norm_compress, grad_norm_shared, grad_norm_reasoning]
        self._buf: deque = deque(maxlen=window)
        self._steps: deque = deque(maxlen=window)

    def update(self, step: int, grad_metrics: Dict[str, float]) -> None:
        """grad_metrics 来自 compute_grad_metrics()，每步调用。"""
        nc = grad_metrics.get("grad_norm_compress", 0.0)
        ns = grad_metrics.get("grad_norm_shared", 0.0)
        nr = grad_metrics.get("grad_norm_reasoning", 0.0)
        if math.isfinite(nc) and math.isfinite(ns) and math.isfinite(nr):
            self._buf.append([nc, ns, nr])
            self._steps.append(step)

    def analyze(self) -> Dict[str, Any]:
        """返回 POD + DMD 分析结果，不足 3 步时返回空字典。"""
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
            # 快速摘要（用于训练日志单行展示）
            "dod_rank": pod["effective_rank"],
            "energy_mode1_pct": pod.get("energy_mode1_pct", 100.0),
            "dmd_spectral_radius": dmd.get("spectral_radius", float("nan")),
        }


class CtStateTracker:
    """
    追踪 c_t 认知状态向量的轨迹，用于 DMD 分析。
    c_t 轨迹能显示认知状态是否在做有意义的循环更新。
    spectral_radius ≈ 1 → 健康的振荡/探索
    spectral_radius > 1 → c_t 在发散（危险）
    spectral_radius << 1 → c_t 收缩到零（无信息）
    """

    def __init__(self, window: int = 100, proj_dim: int = 64):
        self.window = window
        self.proj_dim = proj_dim
        self._buf: deque = deque(maxlen=window)
        self._steps: deque = deque(maxlen=window)
        self._proj: Optional[np.ndarray] = None

    def update(self, step: int, c_t: torch.Tensor) -> None:
        """c_t: (batch, c_t_dim) tensor from model.last_aux['c_t']"""
        v = c_t.detach().float().mean(dim=0).cpu().numpy()  # (c_t_dim,)
        if self._proj is None:
            rng = np.random.default_rng(42)
            self._proj = rng.standard_normal((v.shape[0], self.proj_dim)) / math.sqrt(self.proj_dim)
        v_proj = v @ self._proj  # (proj_dim,)
        if np.isfinite(v_proj).all():
            self._buf.append(v_proj)
            self._steps.append(step)

    def analyze(self) -> Dict[str, Any]:
        if len(self._buf) < 5:
            return {}
        x = np.array(list(self._buf), dtype=np.float64)
        dmd = _dmd_analysis(x)
        return {
            "window_steps": len(self._buf),
            "first_step": self._steps[0],
            "last_step": self._steps[-1],
            "ct_dmd": dmd,
        }


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def render_dynamics_report(
    grad_history: List[Dict[str, Any]],
    ct_history: List[Dict[str, Any]],
    phase: int,
    total_steps: int,
) -> Dict[str, Any]:
    """
    从训练全程的分析快照列表，生成结构化报告。
    grad_history: list of analyze() results from GradTrajectoryTracker
    ct_history: list of analyze() results from CtStateTracker
    """
    if not grad_history:
        return {}

    # DOD rank 轨迹
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
    return report


def _verdict(dod_rank: int, energy_m1_pct: float, phase: int) -> str:
    """给出当前阶段的诊断建议。"""
    if dod_rank < 0:
        return "INSUFFICIENT_DATA"
    # Phase 35 = Phase 3.5，Phase 5 继承 Phase 4 的目标
    effective_phase = 3 if phase == 35 else (4 if phase in (5, 6) else phase)
    target_rank = effective_phase + 1  # Phase 1 → rank≥2, Phase 2 → rank≥3, etc.
    if dod_rank >= target_rank:
        if energy_m1_pct < 80:
            return f"PASS: dod_rank={dod_rank} (target>={target_rank}), energy均匀 (mode1={energy_m1_pct:.1f}%)"
        else:
            return f"WARN: dod_rank={dod_rank} ok但能量仍集中 mode1={energy_m1_pct:.1f}% (目标<80%)"
    else:
        return f"FAIL: dod_rank={dod_rank} < target {target_rank}, mode1={energy_m1_pct:.1f}%"


def save_report(report: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_markdown(report: Dict[str, Any]) -> str:
    """生成易读的 Markdown 摘要。"""
    p = report.get("phase", "?")
    steps = report.get("total_steps", "?")
    dod_rank = report.get("final_dod_rank", "?")
    e1 = report.get("final_energy_mode1_pct", "?")
    radius = report.get("final_dmd_spectral_radius", "?")
    verdict = report.get("verdict", "?")
    rank_traj = report.get("dod_rank_trajectory", [])
    e1_traj = report.get("energy_mode1_pct_trajectory", [])

    lines = [
        f"# Luma Phase {p} Dynamics Report",
        f"*{steps} training steps*",
        "",
        "## 最终状态",
        f"| 指标 | 值 | 说明 |",
        f"|---|---|---|",
        f"| DOD rank (POD有效秩) | **{dod_rank}** | 梯度更新的独立方向数，目标≥{(3 if int(p) == 35 else (4 if int(p) in (5,6) else int(p))) + 1} |",
        f"| 第一模态能量占比 | {e1}% | <80% 为健康，越高越塌缩 |",
        f"| DMD spectral radius | {radius:.4f} | <1 稳定，>1 发散风险 |" if isinstance(radius, float) else f"| DMD spectral radius | {radius} | |",
        "",
        f"## 判断: {verdict}",
        "",
        "## DOD Rank 轨迹",
        f"`{rank_traj}`",
        "",
        "## 第一模态能量% 轨迹",
        f"`{e1_traj}`",
        "",
    ]
    return "\n".join(lines)
