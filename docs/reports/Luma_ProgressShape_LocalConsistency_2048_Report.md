# Luma Progress-Shape vs Local-Consistency Report (2048-step)

## 1. 这次实验在比较什么

这轮实验沿着 `A2-predictor_progress` 这条高潜强化线，继续测试两组 Self-JEPA 强化方向：

1. `progress-shape self JEPA`
2. `local self consistency`

共同底座：

- `A2-predictor_progress`
- `full + depth2 + self_check`
- `self_check_k = 2`
- `one-step main + light two-step auxiliary`
- `rollout_steps = 10`
- `reason_loops = 15`
- `stage2_steps = 2048`
- buckets: `math / dialogue / emotion / persona_seed / python_code / mixed`

对比组：

- `baseline_predictor`
  - 只保留 `predictor_progress`
- `progress_improve`
  - 只加 `next improvement` 预测
- `progress_full`
  - 加 `next improvement + trend + plateau`
- `local_smooth`
  - 加相邻 `pred_delta_c` 的局部平滑约束
- `local_curvature`
  - 加平滑 + 短窗 curvature 约束

## 1.5 每个实验项具体怎么实现

这一节专门把这轮出现的名字翻译成“代码里到底做了什么”，方便后面 review 不再靠记忆猜。

### `A2-predictor_progress`

这是这轮所有组共享的底座强化项。

具体实现：

- `self_loop_awareness_mode = predictor_progress`
- 不把 loop index 直接写进 `c_t`
- 而是只让 `SelfJEPAResidualPredictor` 在预测 `pred_delta_c` / rollout state 时拿到：
  - `loop_progress`
  - `loop_index`

对应直觉：

- 让 predictor 知道“自己现在是第几轮”
- 但尽量保持 `c_t` 主状态本体干净

### `baseline_predictor`

这是最小 loop-aware 基线。

它只包含：

- `A2-predictor_progress`

不额外加：

- progress-shape head
- local consistency regularizer

### `progress_improve`

这是最轻的 `progress-shape self JEPA` 版本。

具体实现：

- `self_progress_shape_weight = 0.10`
- `self_progress_trend_weight = 0.0`
- `self_progress_plateau_weight = 0.0`

也就是：

- 只训练一个轻量 head 去预测“下一步 improvement”
- 不预测 trend
- 不预测 plateau

### `progress_full`

这是当前保留下来的 `A2-progress_shape_v1`。

具体实现：

- `self_progress_shape_weight = 0.10`
- `self_progress_trend_weight = 0.05`
- `self_progress_plateau_weight = 0.02`

它在 `Self-JEPA` 主线旁边又学三件事：

- `next improvement`
  - 下一步相对当前 self error 改善多少
- `trend`
  - 最近两步 improvement 的变化趋势
- `plateau`
  - 当前是否进入平台期

训练方式：

- `next improvement / trend`
  - 用轻量回归项去拟合成熟后的真实 improvement 信号
- `plateau`
  - 用轻量二分类项拟合“当前 improvement 是否已经小到接近平”

### `local_smooth`

这是最轻的 `local self consistency`。

具体实现：

- `self_local_delta_consistency_weight = 0.05`
- `self_local_curvature_weight = 0.0`

它约束：

- 当前 `pred_delta_c(t)` 和上一步 `pred_delta_c(t-1)` 的方向不要乱跳

### `local_curvature`

这是更重一点的局部几何约束。

具体实现：

- `self_local_delta_consistency_weight = 0.05`
- `self_local_curvature_weight = 0.02`

它同时约束：

- 相邻 `pred_delta_c` 的平滑性
- 短窗二阶变化，也就是 trajectory curvature 不要太暴躁

## 2. 顶层结果

| 配置 | mixed self_tail | math self_tail | dialogue self_tail | emotion self_tail | persona_seed self_tail | python_code self_tail |
|---|---:|---:|---:|---:|---:|---:|
| baseline_predictor | 0.0347 | 0.0281 | 0.7148 | 0.3550 | 0.6660 | 0.0813 |
| progress_improve | 0.0143 | 0.0233 | 1.8320 | 0.1145 | 1.5117 | 0.0510 |
| progress_full | 0.0225 | 0.0208 | 0.3223 | 0.1082 | 0.7900 | 0.0510 |
| local_smooth | 0.0386 | 0.0338 | 1.0547 | 0.5532 | 1.0430 | 0.0894 |
| local_curvature | 0.0228 | 0.0209 | 1.1641 | 0.1628 | 1.1211 | 0.0497 |

## 3. 关键观察

### 3.1 `progress_full` 是这轮最值得保留的版本

它不是把某一个桶拉爆，而是比较平衡地改善：

- `mixed`: `0.0347 -> 0.0225`
- `math`: `0.0281 -> 0.0208`
- `dialogue`: `0.7148 -> 0.3223`
- `emotion`: `0.3550 -> 0.1082`
- `python_code`: `0.0813 -> 0.0510`

代价是：

- `persona_seed`: `0.6660 -> 0.7900`

所以它的形状非常像：

- 明显提高了“我现在是不是还在推进，以及推进趋势如何”的表达能力
- 但仍然要付出一点 persona 风格代价

### 3.2 `progress_improve` 太单薄，容易偏科

它虽然把：

- `mixed`
- `math`
- `emotion`
- `python_code`

都拉得很好，尤其 `mixed` 最低：`0.0143`

但它把：

- `dialogue`: `0.7148 -> 1.8320`
- `persona_seed`: `0.6660 -> 1.5117`

直接拉坏了。

这说明：

- 只学“下一步 improvement”不够
- 它会把系统推向过强的单向优化，而不是更健康的推进感知

### 3.3 `local self consistency` 这轮不如 `progress-shape`

`local_smooth` 和 `local_curvature` 都没有给出平衡收益：

- `local_smooth` 在几乎所有桶上都更差
- `local_curvature` 虽然 `mixed / math / python_code` 看起来不错
- 但 `dialogue / persona_seed` 明显更糟

这说明至少在当前口径里：

- 直接给 `pred_delta_c` 加几何平滑约束
- 还不如先让 Self-JEPA 学会表达“推进形状”

## 4. rollout 怎么看

这轮仍然保留了：

- `rollout_active_ratio`
- `rollout_nonzero_ratio`

结果显示：

- 所有组基本都有 `rollout_active_ratio = 1.0`
- 但很多 bucket 的 `rollout_nonzero_ratio` 仍然不高

所以这轮主判断仍然以：

- `self_loss_tail`
- 分桶平衡性

为主。

## 5. 当前结论

### 结论 A：先保留 `progress_full`

在这轮里，它是最像“真的让 Self-JEPA 更会表达推进节奏”的版本。

### 结论 B：不要优先走 `local consistency`

至少在当前实现和当前权重下，这条线没有比 `progress-shape` 更好。

### 结论 C：如果继续沿 `A2-predictor_progress` 推进，下一阶段最合理的候选就是：

- `A2-progress_shape_v1`
  - 对应这轮的 `progress_full`

## 6. 一句话总结

比起直接给 `pred_delta_c` 加局部几何约束，让 Self-JEPA 学会“下一步 improvement + trend + plateau”更像当前正确方向；`progress_full` 是这轮最值得继续追的候选。
