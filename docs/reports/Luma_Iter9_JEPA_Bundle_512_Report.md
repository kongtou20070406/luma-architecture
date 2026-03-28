# Luma Iter9 JEPA Bundle Report (512-step, iter5-based)

## 1. 这次实验做了什么

这次实验从 `iter5` 的轻量 two-step continuation auxiliary 出发，一次性叠加了 5 个 JEPA 升级点：

- `full world JEPA` 简化到更接近 LeWorldModel 的 `next-embedding + SIGReg` 主体
- `world mask strategy` 升级为 `structured` 数据依赖遮挡
- `self/world JEPA coupling`：让 `pred_delta_c` 与 `delta_world_summary` 形成耦合约束
- `hierarchical rollout`：保留多步展开，但只在分层 horizon 上监督
- `surprise metric`：把 world-side latent surprise 纳入评估输出

共同配置：

- `full + depth2 + self_check_k=2`
- `rollout_steps = 10`
- `reason_loops = 15`
- `stage2_steps = 512`
- `fixture_mode = competition_math_dialogue_emotion`
- `per_task_from_mixed = true`
- `enable_persona_seed = true`

结果文件：

- 新实验：`/home/kt/ai/minimind/artifacts/autoresearch_iter9_jepa_bundle.json`
- 当前 retained 对照：`/home/kt/ai/minimind/artifacts/autoresearch_iter2_eval.json`

## 2. 核心结果

| bucket | retained iter2 | iter9 bundle | delta |
|---|---:|---:|---:|
| mixed | 0.041015625 | 0.0390625 | -0.001953125 |
| math | 0.037109375 | 0.041015625 | +0.00390625 |
| dialogue | 0.041015625 | 0.037109375 | -0.00390625 |
| emotion | 0.087890625 | 0.080078125 | -0.0078125 |
| persona_seed | 0.361328125 | 0.61328125 | +0.251953125 |

补充：

- `mixed self_loss_tail = 0.063720703125`
- `mixed world_surprise_mean = 11.872136116027832`
- `stage1 hard_loop_var = 0.984375`
- `stage1 soft_loop_var = 0.6875`

## 3. 怎么理解

### 好消息

- `mixed` 指标进一步压低：`0.041015625 -> 0.0390625`
- `dialogue` 提升
- `emotion` 提升
- `surprise` 指标成功落地，而且是稳定非零信号

这说明：

- `structured world mask`
- `simplified full world loss`
- `self/world coupling`
- `hierarchical rollout`
- `light two-step continuation auxiliary`

这一束改动并不是空转，它确实增强了 world-driven continuation behavior。

### 坏消息

- `math` 从 `0.037109375` 回退到 `0.041015625`
- `persona_seed` 明显变差，但现在它只是 soft guard

在当前 guard 口径下，`math` 仍是硬护栏，所以这次实验不能 keep。

## 4. 结论

### 这次实验不是 keep

原因非常具体：

- `mixed` 更好
- `dialogue / emotion` 更好
- 但 `math` 退化，违反当前硬 guard

### 但这不是无价值的 discard

这次 discard 很有信息量：

- world JEPA 极简化方向是活的
- `surprise` 评估维度已成功落地
- 失败点不是 mixed 崩掉，而是 `math` 被这组改动伤到了

所以更准确地说：

- 这是一个 **有前景但不守 math guard 的分支**
- 不是“无效实验”

## 5. 下一步最自然的方向

在保留这条 iter9 方向的前提下，后续更值得试的是：

1. 保留 `structured mask + simplified full world + surprise`
2. 先减弱 `self/world coupling` 权重
3. 或者把 `hierarchical rollout` 从当前 horizon 选择稍微收紧
4. 不急着丢掉 two-step auxiliary，但也不要继续加重

一句话：

- 这次 bundle 已经找到了一条能提升 `mixed/dialogue/emotion` 的 JEPA 路线
- 下一步要做的是把 `math` 拉回来，而不是把整条线推倒重来
