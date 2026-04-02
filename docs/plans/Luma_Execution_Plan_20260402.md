# Luma Execution Plan (2026-04-02)

## 1) Current Canonical Workspace
- 主项目唯一入口：`/home/kt/ai/luma-architecture/minimind`
- `minimind_runtime_dynamics`：已移除，不再使用
- `parameter-golf`：仅可选参考，默认实验链不依赖

## 2) Current Baseline (Frozen for screening)
- Baseline: `A2-progress_shape_v1-h3+progress_exit_readout`
- Core:
  - `world_jepa_mode=full`
  - `reason_shared_depth=2`
  - `enable_self_check_ring=true`
  - `self_check_k=2`
  - `rollout_steps=10`
  - `reason_loops=15`
- Continuation supervision:
  - `one-step main`
  - `light two-step auxiliary`

## 3) Active Matrix Policy
- 当前执行矩阵：`sigreg` 8 组合（world / rollout / delta 全析因）
- 两阶段预算：
  - Stage1: `15M tokens`
  - Stage2: `80M tokens`
- 上下文长度策略：
  - 主战场：`seq_len=1024`
  - 冒烟：`seq_len=256`
  - 确认：`seq_len=2048`
- 分桶输出必须包含：
  - `math / python_code / mixed / dialogue / emotion / persona_seed / arc_agi`

## 4) Guard and Validity Rules
- 先看健康，再看任务分：
  - `rollout_nonzero_ratio`
  - `future_delta_var`
  - `predicted_gain_std`
  - `exit_score_var`
- 证据边界：
  - stale runtime 不能当有效失败
  - 实现 bug 污染不能直接判结构失败
  - NaN score 不能用于排名定案

## 5) Report Entry
- 报告总入口：`/home/kt/ai/luma-architecture/docs/reports/README.md`
- 当前主报告：
  - `Luma_Stage12_Consolidated_Report_20260402.md`
  - `Luma_Dynamics_Consolidated_Report_20260402.md`
  - `Luma_Dynamics_Matrix13_ARCAGI_Report_20260330.md`
  - `Luma_Artifacts_Cleanup_20260402.md`

