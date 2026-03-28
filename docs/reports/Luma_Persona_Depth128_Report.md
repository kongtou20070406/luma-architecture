# Luma Persona Bucket And Shared-Depth 128-Step Report

## 1. 这次实验在验证什么

这轮实验同时验证三件事：

- `luma_dataset` 是否已经作为独立的 `persona_seed` 桶进入 Luma 的阶段验证链路
- `full + self_check` 下，共享推理 block 的真实深度从 `1` 层变为 `2` 层时，短中程表现怎么变化
- 在 `128-step` 的更长短程训练里，`math / dialogue / emotion / persona_seed / mixed` 五个任务桶分别如何表现

这次不是纯数学测试，而是混合测试：

- `competition_math`
- `dialogue`
- `emotion`
- `persona_seed`

其中 `persona_seed` 来自：

- `/home/kt/ai/luma_dataset/wechat_pretrain.jsonl`
- `/home/kt/ai/luma_dataset/pretrain.jsonl`

## 2. 实验配置

共同配置：

- `world_jepa_mode = full`
- `enable_self_check_ring = true`
- `rollout_steps = 10`
- `reason_loops = 15`
- `slow_k = 1`
- `stage2_steps = 128`
- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`

对比变量：

- `depth=1`
- `depth=2`

结果文件：

- `depth=1`
  - `/home/kt/ai/minimind/artifacts/stage12_muon_full_selfcheck_persona_10x15_128_depth1.json`
- `depth=2`
  - `/home/kt/ai/minimind/artifacts/stage12_muon_full_selfcheck_persona_10x15_128_depth2.json`

## 3. 顶层结果

| 配置 | stage1 mean_kl | stage1 hard_loop_var | stage2 self_loss_tail | stage2 self_rollout_tail |
|---|---:|---:|---:|---:|
| depth=1 | 1.1420 | 2.4375 | 0.1101 | 0.2441 |
| depth=2 | 2.3746 | 1.1094 | 0.1909 | 0.2266 |

### 先看结论

- `depth=1` 的综合稳定性更好
- `depth=2` 并没有在 mixed 总体上反超
- `depth=2` 的主要亮点不在 mixed，而在 `emotion` 桶的循环深度分布

## 4. persona_seed 桶结果

| 配置 | persona hard_loop_var | persona self_loss_tail | persona self_rollout_tail |
|---|---:|---:|---:|
| depth=1 | 0.4375 | 0.0536 | 0.0664 |
| depth=2 | 0.1875 | 0.0586 | 0.0664 |

### 怎么理解

- 两个深度配置都已经能在 `persona_seed` 桶上形成很低的 `self_loss_tail`
- `self_rollout_tail` 两者几乎一样
- 但 `depth=1` 的 `persona_seed hard_loop_var` 更高一点

这说明：

- `persona_seed` 已经不是混在 `mixed` 里的隐含样本，而是一个真正能单独被监控的桶
- 当前在人格种子这条线上，`depth=1` 并不弱，反而更稳

## 5. emotion 桶结果

| 配置 | emotion hard_loop_var |
|---|---:|
| depth=1 | 1.6875 |
| depth=2 | 3.6875 |

这是这次最值得注意的地方。

`depth=2` 在 `emotion` 桶上的 `hard_loop_var` 明显更强，说明：

- 当任务更偏情感/情境表达时
- 两层共享推理 block 更容易拉出更丰富的离散推理深度分布

但这个优势目前还没有转化成 mixed 总体反超。

## 6. mixed 桶结果

| 配置 | mixed self_rollout_tail |
|---|---:|
| depth=1 | 0.1719 |
| depth=2 | 0.2617 |

这里 `depth=1` 更好。

所以当前工程判断不能简单写成“更深就更强”，而应该写成：

- `depth=1`：综合更稳，适合当前正式预训练候选底座
- `depth=2`：在情感/表达类任务上更有潜力，适合作为专项分支继续观察

## 7. FP8 现状

这次实验没有启用真正的 `FP8 training`。

当前真实状态是：

- 已接入：
  - `AdamW8bit`
  - 实验性 `8-bit Muon`
- 尚未接入：
  - `Transformer Engine` 风格的 `FP8 autocast`
  - `FP8 GEMM` 主干训练链路
  - `torchao` / `TE` 风格的正式 `FP8` 训练栈

因此当前更准确的表述是：

- `8-bit optimizer` 已接入
- `FP8 training` 尚未接入

## 8. 最终结论

### 结论 1：persona_seed 桶已经正式接入验证链路

而且不是摆设。

在当前 `full + self_check + 10x15 + 128-step` 下：

- `persona_seed self_loss_tail` 已低到 `0.0536`
- `persona_seed self_rollout_tail` 已低到 `0.0664`

说明这条人格种子桶，已经能真实参与我们对 Luma 自省/动力学能力的判断。

### 结论 2：共享推理 block 从 1 层变 2 层，不是当前 mixed 默认升级

原因：

- `depth=2` 没有在 mixed 上反超
- `depth=1` 的综合稳定性更好

### 结论 3：depth=2 不该被删掉

因为：

- 它在 `emotion` 桶上明显更强

所以更合理的策略是：

- 默认正式预训练候选：`depth=1`
- 情感/表达专项候选：`depth=2`

### 一句话总结

Luma 现在已经能把你的真实发言语料单独当作人格种子桶来验证；
而在 `full + self_check + 10x15 + 128-step` 下，共享推理 block 的两层版本更像“情感专项增强”，不是当前 mixed 默认底座。
