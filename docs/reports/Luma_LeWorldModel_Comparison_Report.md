# Luma LeWorldModel-Style Comparison Report

## 1. 这次比较了什么

这次实验比较四个版本：

- `scaffold`
  - 当前的丐版 `WorldLatentJEPA`
- `scaffold + self_check`
  - 丐版 world JEPA
  - 再加一个极简慢环自检流 `TinySlowSelfCheckRing`
- `full`
  - 更完整的 `LeWorldModel-style` world JEPA
- `full + self_check`
  - 更完整的 `LeWorldModel-style` world JEPA
  - 再加极简慢环自检流

实验配置统一为：

- 数据：`hard_math_dialogue`
- 数学：`EleutherAI/hendrycks_math`
- 对话：`ConvLab/dailydialog`
- `slow_k=1`
- `rollout_steps=4`
- `reason_loops=8`
- `stage2_steps=8`

对应结果文件：

- [stage12_compare_scaffold.json](/home/kt/ai/minimind/artifacts/stage12_compare_scaffold.json)
- [stage12_compare_scaffold_selfcheck.json](/home/kt/ai/minimind/artifacts/stage12_compare_scaffold_selfcheck.json)
- [stage12_compare_full.json](/home/kt/ai/minimind/artifacts/stage12_compare_full.json)
- [stage12_compare_full_selfcheck.json](/home/kt/ai/minimind/artifacts/stage12_compare_full_selfcheck.json)

---

## 2. 这次实现了什么

### `scaffold` 版本

当前丐版 `world JEPA` 仍然是：

- 轻量 observer
- EMA target
- masked latent prediction
- 较轻的 predictor

它的优点是：

- 简单
- 稳
- 参数小
- 比较适合阶段0快速验证

### `LeWorldModel-style` 版本

这次补进去的完整版是“`LeWorldModel-style` 工程迁移版”，关键补强点有三类：

1. 结构化遮挡
- 从随机点 mask 改成连续 span mask
- `probe_error` 也改成确定性的中段 block mask

2. 更强的条件预测
- online / target encoder 变成两层编码器
- predictor 不再只看轻量 summary
- 会联合使用：
  - masked online latent
  - visible summary
  - hidden summary
  - delta-style refinement

3. latent 分布正则
- 新增简单的 latent variance regularization
- 避免 world latent 过快塌缩

要诚实说明的是：

- 这不是逐项“论文一比一复刻”
- 这是把 `LeWorldModel` 的核心气质迁移到当前 LLM hidden-state scaffold 上
- 所以更准确的名字是：
  - `LeWorldModel-style fuller implementation`

### 极简慢环自检流

这次还额外加了：

- `TinySlowSelfCheckRing`

它不是替代主自省流，而是一个更便宜的小旁路：

- 输入：`c_t + delta_h + know_gap + prev_self_check_state`
- 输出：
  - `next self_check_state`
  - `self_check_score`

它现在的作用是：

- 作为退出控制器的额外信号
- 观察“更便宜的自检旁路”会不会帮助循环深度分布拉开

---

## 3. 关键结果表

| Variant | hard_loop_var | soft_loop_var | mean_kl | c_t_var | improvement | self_check_mean | self_loss_tail | self_rollout_tail | mean_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scaffold | 0.4375 | 0.1094 | 1.6359 | 0.7738 | 0.2006 | 0.5000 | 1.0508 | 0.8281 | 24.4375 |
| scaffold + self_check | 0.7344 | 0.1094 | 1.5951 | 0.7592 | 0.2130 | 0.4798 | 0.9980 | 0.7090 | 23.4375 |
| full | 0.4375 | 0.1094 | 1.6501 | 0.7682 | 0.1978 | 0.5000 | 1.0664 | 0.7656 | 23.7436 |
| full + self_check | 0.7500 | 0.1094 | 1.4754 | 0.7763 | 0.1822 | 0.5420 | 1.0449 | 0.7539 | 24.3856 |

补充观察：

- `self_rollout_tail` 越低越好
- `self_loss_tail` 越低越好
- `hard_loop_var` 非零且更高，说明硬退出下的推理深度分布更容易被拉开
- `mean_kl` 越大，说明 `c_t` 注入对主流影响越明显

---

## 4. 怎么看这四组结果

### 结论 1：`LeWorldModel-style` 版本在 rollout 上更强，但不是全维度碾压

对比 `scaffold` 和 `full`：

- `self_rollout_tail`
  - `0.8281 -> 0.7656`
- `mean_loss`
  - `24.4375 -> 23.7436`

这说明：

- fuller world JEPA 的确在“动力学一致性”这条线上更强
- 它也让整体短程训练 loss 更低

但它没有在所有指标上都更强：

- `self_loss_tail`
  - `1.0508 -> 1.0664`
- `hard_loop_var`
  - 持平

所以更准确的说法是：

- `LeWorldModel-style` 完整版更像是在“world-side 动力学约束”上变强了
- 但它还没有强到在所有短程指标上全面压过丐版

### 结论 2：极简慢环自检流在两条 world 分支上都能明显拉高 `hard_loop_var`

对比：

- `scaffold`
  - `hard_loop_var = 0.4375`
- `scaffold + self_check`
  - `hard_loop_var = 0.7344`

- `full`
  - `hard_loop_var = 0.4375`
- `full + self_check`
  - `hard_loop_var = 0.7500`

这说明：

- 这个极简自检旁路不是噪声件
- 它确实在帮助硬退出分布松开

而且在丐版 world JEPA 上，它还带来了更明显的训练收益：

- `self_loss_tail`
  - `1.0508 -> 0.9980`
- `self_rollout_tail`
  - `0.8281 -> 0.7090`
- `mean_loss`
  - `24.4375 -> 23.4375`

### 结论 3：当前短程实验里，最强综合点其实是 `scaffold + self_check`

如果看“综合平衡”：

- `scaffold + self_check`
  - `hard_loop_var` 高
  - `self_rollout_tail` 最低
  - `self_loss_tail` 也更低
  - `mean_loss` 最低

这说明一件很有意思的事：

- 当前 fuller world JEPA 方向是对的
- 但在 tiny scaffold + 短程训练里，它还没完全吃到自己的结构红利
- 反而“丐版 world JEPA + 极简慢环自检流”目前是最划算的版本

### 结论 4：`full + self_check` 不是失败，但更像“有潜力、还没调顺”

`full + self_check` 的特征是：

- `hard_loop_var = 0.7500`
- `mean_kl = 1.4754`
- `c_t_var = 0.7763`
- `self_rollout_tail = 0.7539`

这说明：

- 它的状态表达是活的
- 硬退出分布也被拉开了
- rollout 也不差

但目前整体 `mean_loss` 没赢：

- `24.3856`

所以更像是：

- 结构方向没错
- 只是短程 tiny 训练下还没完全收敛到最优工作点

---

## 5. 当前最合理的工程判断

### 如果目标是“现在就选一个更稳、更强的短程版本”

优先推荐：

- `scaffold + self_check`

理由：

- 指标最均衡
- rollout 最好
- hard exit 分布明显更开
- 平均 loss 也最低

### 如果目标是“保留更强 world-model 路线继续打磨”

保留：

- `full`
- `full + self_check`

理由：

- fuller world JEPA 已经在 rollout 与总体 loss 上显示出潜力
- 只是当前还没压过 `scaffold + self_check`

---

## 6. 我对下一步的建议

### 建议 A：主干默认先切到 `scaffold + self_check`

原因：

- 这是当前短程实验里性价比最高的版本
- 更适合作为阶段1/2继续推进的稳定底座

### 建议 B：把 `LeWorldModel-style` 继续保留成专项分支

原因：

- 它已经显示出 world-side 动力学优势
- 但可能需要：
  - 更长训练
  - 更长 rollout
  - 更难样本
  才能完全吃到结构收益

### 建议 C：下一轮重点测试 `full + self_check + longer rollout`

我会优先建议：

- `world_jepa_mode=full`
- `enable_self_check_ring=True`
- `rollout_steps=5`
- `reason_loops=10`

因为这更接近它可能真正占优的场景。

---

## 7. 最后一句话

这次实验的结论不是“完整版一定更强”，而是更细一点：

- `LeWorldModel-style` 完整 world JEPA 已经显示出更强的动力学潜力
- 但在当前 tiny + 短程验证里，最强综合版本仍然是：
  - `scaffold + tiny self-check ring`

所以现在最稳妥的路线是：

- 主干继续用更稳的 `scaffold + self_check`
- 把 `LeWorldModel-style` 作为下一阶段重点强化分支继续推进
