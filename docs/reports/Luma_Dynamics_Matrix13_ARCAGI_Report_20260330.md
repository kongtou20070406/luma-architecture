# Luma Dynamics Matrix13 + Chollet ARC-AGI Report (2026-03-30)

## 1) Run Snapshot
- Run directory: `/home/kt/ai/minimind/artifacts/autoresearch_dynamics_rescue13_arcagi_20260330_083823`
- Program: `/home/kt/ai/minimind/luma_stage0/dynamics_autoresearch_program.json`
- ARC setting: `enable_arc_agi=true`（Chollet ARC-AGI text-linearized bucket）
- Final runtime state: `complete`
- Stage coverage:
  - `short_prescreen(4096)`: `13` candidates
  - `mid_rescreen(10240)`: `2` candidates
  - `long_round1(20480)`: `1` candidate
  - `long_confirm(20480)`: `0` candidates

## 2) Validity Boundary
- **有效结论（结构层）**
  - `short_prescreen(4096)` 全部 `13` 条候选均有 `ok` 结果，可用于短程结构筛选。
  - `mid_rescreen(10240)` 中仅 `summary_chunk_film_v2_progress+s_local_floor` 通过 guard。
- **部分有效（需谨慎）**
  - `mid/long` 阶段 summary 中出现 `score=NaN`，并伴随 `math_self_tail` / `mixed_self_tail` 等指标 NaN。
  - 因此 `mid/long` 结果可作为“是否守住 guard/是否数值失稳”的证据，但不宜用 `score` 排名做最终 keep/kill。

## 3) Stage Outcome
- 数据表（本轮晋级与关键有效性）：

| Stage | Candidate | Score | Guard | 备注 |
|---|---|---:|---:|---|
| short_prescreen(4096) | `...memory_tiered_routing_v1+m1_lite+zone_loss` | `0.0573` | ✅ | guard 通过，晋级 10240 |
| short_prescreen(4096) | `...summary_chunk_film_v2_progress+s_local_floor` | `0.0601` | ✅ | guard 通过，晋级 10240 |
| mid_rescreen(10240) | `...memory_tiered_routing_v1+m1_lite+zone_loss` | `NaN` | ❌ | `rollout_nonzero` 失守 |
| mid_rescreen(10240) | `...summary_chunk_film_v2_progress+s_local_floor` | `NaN` | ✅ | 仅这条继续晋级 20480 |
| long_round1(20480) | `...summary_chunk_film_v2_progress+s_local_floor` | `NaN` | ❌ | `c_t_var` 与 dialogue 项失稳 |

- `short_prescreen(4096)`:
  - guard 通过候选只有 `2` 条：
    - `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1+m1_lite+zone_loss`
    - `A2-progress_shape_v1-h3+progress_exit_readout+summary_chunk_film_v2_progress+s_local_floor`
  - 其余 11 条虽有较低分数者，但 guard 未过（主要是 rollout_nonzero 或相关健康项不达标）。
- `mid_rescreen(10240)`:
  - `memory_tiered...+m1_lite+zone_loss`: guard 未过（rollout_nonzero 失守）。
  - `summary_chunk_film_v2_progress+s_local_floor`: guard 通过。
- `long_round1(20480)`:
  - 仅 `summary_chunk_film_v2_progress+s_local_floor` 晋级，但 guard 未过（`c_t_var` 与 dialogue 相关项失稳）。

## 4) ARC-AGI Reading
- `arc_agi_self_tail` 已稳定出现在 summary 指标中，说明 ARC-AGI 桶已进入本轮统一评估口径。
- 但本轮中后程存在 NaN 污染，当前更适合把 ARC-AGI 视为“新增观察维度已接通”，而不是“已经可用于长程排名决策”的稳定主指标。

## 5) Current Keep/Kill Boundary
- **Provisional keep（本轮最稳晋级链）**
  - `A2-progress_shape_v1-h3+progress_exit_readout+summary_chunk_film_v2_progress+s_local_floor`
  - 原因：4096 与 10240 均可晋级，且 10240 仍能过 guard。
- **Observe**
  - `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1+m1_lite+zone_loss`
  - 原因：4096 可过 guard，但 10240 guard 失守。
- **Not promoted this cycle**
  - 其余候选在 4096 即未过 guard。

## 6) Next Actions
1. 先修复 `mid/long` NaN 评分路径（至少把 NaN 作为显式 fail-score，而非参与排序）。
2. 在 `summary_chunk_film_v2_progress+s_local_floor` 上做一次 10240 复验，确认 NaN 是否可复现。
3. 在 ARC-AGI 桶上新增单独数值健康阈值（例如 `arc_agi_self_tail` finite-rate）后再用于长程排名。
