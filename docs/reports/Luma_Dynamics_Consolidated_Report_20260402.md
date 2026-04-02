# Luma Dynamics Consolidated Report (2026-04-02)

## 1) Scope
本报告合并了 2026-03-29 到 2026-04-02 的 Dynamics 主线报告，统一给出“可用于当前决策”的结论与边界。

合并来源：
- `Luma_Dynamics_2048_Prescreen_Report.md`
- `Luma_Dynamics_MidLong_Summary_20260329.md`
- `Luma_Dynamics_Matrix12_Nightly_Report_20260330.md`
- `Luma_Dynamics_Matrix13_ARCAGI_Report_20260330.md`
- `Luma_Iter2_Iter5_Predictor_2048_Report.md`
- `Luma_LoopAware_2048_Report.md`
- `Luma_ProgressShape_LocalConsistency_2048_Report.md`
- `Luma_Rollout_ABC_Matrix_2048_Report.md`
- `Luma_Longrun_10240_Plan.md`、`Luma_Longrun_10240_Report.md`
- `Luma_Artifacts_Cleanup_20260402.md`

## 2) Timeline Summary

| 时间 | 关键阶段 | 当前可用结论 |
|---|---|---|
| 2026-03-29 | `2048 -> 4096 -> 10240` 动力学筛选 | `A2-progress_shape_v1-h3+progress_exit_readout` 被保留为最稳增强主候选 |
| 2026-03-30 (Matrix12) | 12 候选夜跑 | 仅 `memory_tiered_routing_v1` 曾晋级中程；hier/double-p 当时受实现 bug 污染 |
| 2026-03-30 (Matrix13+ARC-AGI) | rescue13 + ARC 桶接入 | `summary_chunk_film_v2_progress+s_local_floor` 在 `10240` 可过 guard，但 `20480` 仍出现失稳边界 |
| 2026-04-02 | sigreg 8-cell factorial 开始执行 | 当前运行中，采用 `15M -> 80M` 两阶段预算与分桶统计 |

## 3) Current Mainline Decision

### 3.1 主线锚点
- 结构主线仍是：`A2-progress_shape_v1-h3+progress_exit_readout`
- 当前正在此基线上跑 `sigreg` 8 组合矩阵（world/rollout/delta 单独与组合）。

### 3.2 候选状态
- Provisional keep：
  - `summary_chunk_film_v2_progress+s_local_floor`（需要继续看长程数值健康）
- Observe：
  - `memory_tiered_routing_v1`（重点防 tier collapse）
- Cannot judge（当次实现污染）：
  - `hier_block_token_*`、`double_p_coarse_to_fine_v1`（历史 run 存在接口 bug 污染边界）

## 4) Health-First Guard Policy (Retained)
- 先过健康，再看任务分：
  - `rollout_nonzero_ratio`
  - `future_delta_var`
  - `predicted_gain_std`
  - `exit_score_var`
- 分桶输出必须保留：
  - `math / python_code / mixed / dialogue / emotion / persona_seed / arc_agi`

## 5) Invalid Evidence Boundary (Must Keep)
- 以下情况不作为结构失败证据：
  - stale runtime（仅 runtime/heartbeat 存在，但无活进程）
  - 实现 bug 污染（shape mismatch / 缺失 import / 非结构性崩溃）
  - NaN 未隔离的排序分数（可用于“失稳警告”，不可用于“优劣排名定案”）

## 6) Artifact Anchor (Current)
- 活跃运行目录：
  - `/home/kt/ai/luma-architecture/minimind/artifacts/autoresearch_sigreg8_15m80m_20260402`
- 清理与补表记录：
  - `/home/kt/ai/luma-architecture/docs/reports/Luma_Artifacts_Cleanup_20260402.md`

