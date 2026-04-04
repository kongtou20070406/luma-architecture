# Luma Dynamics FP32 严格链路报告（2026-04-03）

## 1. 结论先行
- 本轮 **没有完整跑完 100 次矩阵迭代**。
- 本轮只完成了严格链路的 baseline 两段评估：
  - `baseline_4096`：完成且有限值
  - `baseline_10240`：出现非有限值污染（严格模式下视为失败）
- 因为本轮启用了“非有限值即失败（不容错）”，主循环在 baseline 阶段直接中止，未进入 `iter_001...iter_100`。

## 2. 运行口径与产物位置
- 运行目录：
  - `/home/kt/ai/luma-architecture/minimind/artifacts/luma_dynamics_constraints_fg_20260403_115036`
- 候选基线：
  - `A2-progress_shape_v1-h3+progress_exit_readout+m1_full_regularizers_from_a4e_sigreg_low`
- 精度口径：
  - 强制 `--force-fp32`
- 严格性口径：
  - score 非有限值 / `first_nonfinite_step` 非空即判失败，不容错继续。

## 3. Baseline 结果摘要

### 3.1 `baseline_4096`（有效）
- score: `1.4535038113594054`
- Layer2:
  - POD effective rank: `1`
  - POD top1 energy ratio: `0.9997478333941875`
  - DMD spectral radius: `0.7747791954935912`
  - forcing top |corr|: `0.9238457761145215`
- 数值健康:
  - `first_nonfinite_step = None`
  - `world_sigreg_loss_max = 75.13307189941406`
  - `grad_norm_total_tail = 235.34089737017229`

分桶（4096）：

| bucket | self_loss_tail | rollout_nonzero_ratio | mean_loss |
|---|---:|---:|---:|
| math | 1.4631337523460388 | 0.0 | 104.64535522460938 |
| python_code | 1.5316027402877808 | 0.0 | 75.92313194274902 |
| mixed | 1.4499488472938538 | 0.0 | 84.74923706054688 |
| dialogue | 1.5172414779663086 | 0.0 | 60.70167827606201 |
| emotion | 1.5104944705963135 | 0.0 | 78.56272315979004 |
| persona_seed | 1.539442241191864 | 0.0 | 36.12698936462402 |
| arc_agi | 1.1547977328300476 | 0.0 | 121.79536056518555 |

### 3.2 `baseline_10240`（无效，严格模式失败）
- score: `NaN`
- Layer2:
  - POD effective rank: `1`
  - POD top1 energy ratio: `0.9999257313474416`
  - DMD spectral radius: `0.8467580616282209`
  - forcing top |corr|: `NaN`
- 数值健康:
  - `first_nonfinite_step = 3641`
  - `world_sigreg_loss_max = 87.20095825195312`
  - `grad_norm_total_tail = 761.6999602852917`

分桶（10240）：

| bucket | self_loss_tail | rollout_nonzero_ratio | mean_loss |
|---|---:|---:|---:|
| math | NaN | 0.0 | NaN |
| python_code | NaN | 0.75 | NaN |
| mixed | NaN | 0.5 | NaN |
| dialogue | 1.0391185879707336 | 0.5 | 21.769761085510254 |
| emotion | NaN | 0.75 | NaN |
| persona_seed | NaN | 0.25 | NaN |
| arc_agi | NaN | 0.875 | NaN |

## 4. 状态判定边界（这轮哪些有效）
- 有效:
  - `baseline_4096` 的指标可作为短程数值状态参考。
- 无效:
  - `baseline_10240` 不可用于结构优劣判断（已发生 non-finite 污染）。
  - 本轮无任何 `iter_xxx` 结果，因此不存在 keep/kill 结论。

## 5. 直接建议（用于下一轮重启）
- 先把本次作为“严格链路验证轮”，结论是：**10240 仍存在非有限风险，当前配置不能直接进入批量迭代筛选**。
- 下一轮建议优先做：
  1. 先固定单候选 10240 稳定性（避免直接进 100 iter）。
  2. 在通过 strict baseline 后再开启迭代矩阵。
  3. 保持 `FP32 + no-NaN-tolerance` 不变，防止假稳定。

