# Luma Rollout Depth Experiment Report

## 1. 这次实验在比较什么

这次实验比较的是：

- 在更难的“长推理数学 + 对话”混合数据上
- `Self JEPA` 的 rollout 深度取 `2-step / 3-step / 4-step`
- 哪一种更有利于慢环动力学学习与动态退出

数据模式：

- `fixture_mode = hard_math_dialogue`
- 数学来源：`EleutherAI/hendrycks_math`
- 对话来源：`ConvLab/dailydialog`

报告文件：

- `2-step`: [stage12_report_hard_math_rollout2_matured.json](/home/kt/ai/minimind/artifacts/stage12_report_hard_math_rollout2_matured.json)
- `3-step`: [stage12_report_hard_math_rollout3_matured.json](/home/kt/ai/minimind/artifacts/stage12_report_hard_math_rollout3_matured.json)
- `4-step`: [stage12_report_hard_math_rollout4_matured.json](/home/kt/ai/minimind/artifacts/stage12_report_hard_math_rollout4_matured.json)

---

## 2. 先说明一个实验修正

第一次尝试 `3-step / 4-step` 时，结果几乎和 `2-step` 一样。

原因不是 rollout 深度没区别，而是：

- tiny 骨架的 `reason_loops` 太短
- 更长 horizon 还没成熟对齐就结束了

所以这次修正为：

- `reason_loops = max(4, rollout_steps * 2)`

这样：

- `3-step` 和 `4-step` 都有足够的慢环更新次数
- 结果才真正有比较意义

---

## 3. 核心结果表

| Rollout steps | hard_loop_var | soft_loop_var | c_t_var | two_step_improvement_mean | self_rollout_tail | self_loss_tail |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.0 | 0.1389 | 0.7828 | -0.0179 | 1.0156 | 0.9395 |
| 3 | 0.0 | 0.1389 | 0.6366 | 0.0089 | 0.9043 | 0.9453 |
| 4 | 0.0 | 0.1389 | 0.6449 | 0.0565 | 0.8145 | 0.9043 |

补充：

| Rollout steps | mean_kl | mean_hidden_delta | exit_score_var | sampled_exit_score_var | mean_delta_norm |
|---|---:|---:|---:|---:|---:|
| 2 | 0.9551 | 3.1406 | 1.19e-05 | 6.51e-05 | 6.3490 |
| 3 | 1.4460 | 4.0313 | 2.39e-05 | 1.06e-04 | 3.0260 |
| 4 | 1.8185 | 4.6563 | 2.86e-05 | 1.40e-04 | 2.2995 |

---

## 4. 每一列怎么理解

### `hard_loop_var`

- 硬退出下的循环步数方差
- 仍然是 `0.0`
- 说明硬退出仍然把深度分布压平

### `soft_loop_var`

- 采样退出下的循环步数方差
- 三组都为非零
- 说明训练侧软退出已经能释放深度分布

### `c_t_var`

- `c_t` 在不同样本上的方差
- 三组都明显非零
- 说明慢环没有塌缩

### `two_step_improvement_mean`

这里虽然名字历史上还叫 `two_step`，但在当前实现里它表示：

- “更长 rollout 相对一步预测，是否仍然带来收益”的平均改善量

解释：

- `2-step` 为负：说明在 harder data 上，当前 2-step rollout 仍偏短，额外展开没有形成明显收益
- `3-step` 转正：说明更长 rollout 开始带来真实收益
- `4-step` 更正：说明更长 horizon 的动力学约束在这组任务上更有效

### `self_rollout_tail`

- rollout loss 在短程训练结束时的值
- 越低越好

结果：

- `2-step = 1.0156`
- `3-step = 0.9043`
- `4-step = 0.8145`

这说明：

- rollout 深度加长以后，动力学一致性在 harder math 上是变好的

---

## 5. 最重要的结论

### 结论 1：hard math 上，`3-step / 4-step` 比 `2-step` 更有价值

证据：

- `two_step_improvement_mean`
  - `2-step`: 负值
  - `3-step`: 转正
  - `4-step`: 更明显转正
- `self_rollout_tail`
  - 随 rollout 深度增加持续降低

这说明：

- 在更难、更长推理的数学样本上
- 只用 `2-step` 可能偏短
- 更长 horizon 的动力学监督更能提供真实收益

### 结论 2：`4-step` 当前是最强的短程实验结果

从这次短程实验看：

- `4-step` 在 rollout 指标上是最好的
- `mean_kl` 和 `mean_hidden_delta` 也更强

但这不自动等于：

- 最终预训练就应该直接固定 `4-step`

因为还没验证：

- 更长训练下是否会出现误差累积反噬
- 更大模型下是否仍保持相同排序

### 结论 3：硬退出仍是主要瓶颈

不管 rollout 深度怎么变：

- `hard_loop_var` 仍然是 `0.0`

这说明：

- rollout 深度已经在改善动力学学习
- 但离散退出行为仍然被硬退出规则压平

所以 rollout 深度优化和退出策略优化，是两条都要继续推进的线。

---

## 6. 当前最合理的工程判断

### 如果目标是“先把系统做稳”

建议：

- 训练：`soft exit`
- 推理：`hard exit`
- rollout 深度：继续保留 `2-step` 作为保守默认

理由：

- `2-step` 更便宜
- 已经能工作
- 适合作为稳定下界

### 如果目标是“为 harder reasoning 提前做准备”

建议：

- 把 `3-step` 设为下一阶段重点候选
- 把 `4-step` 设为强化实验候选

理由：

- `3-step` 已经明显优于 `2-step`
- `4-step` 目前更强，但还需要更长训练确认是否稳

我自己的倾向是：

- `2-step` 不该被立刻删掉
- 但在 harder math 场景里，`3-step/4-step` 已经值得认真进入候选集

---

## 7. 我对下一步的建议

### 建议 A：主规划里保留 `2-step` 为正式默认

原因：

- 更稳
- 成本更低
- 已经被验证可运行

### 建议 B：把 `3-step` 标记为“长推理数学专项候选”

原因：

- 它已经比 `2-step` 更好
- 风险比 `4-step` 更可控

### 建议 C：继续保留 `4-step` 为专项实验路径

原因：

- 这次它是最强的
- 但还需要更长训练和更多样本证明不会因误差累积而反噬

---

## 8. 最终一句话

在 harder math + dialogue 的短程实验里，`2-step rollout` 已经不再明显占优；
`3-step` 和尤其 `4-step` 显示出更强的动力学监督价值。

所以：

- `2-step` 仍适合作为正式默认
- `3-step / 4-step` 已经值得进入下一阶段重点实验名单

