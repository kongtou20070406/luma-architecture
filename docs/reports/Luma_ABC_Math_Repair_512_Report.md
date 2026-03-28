# Luma A/B/C Math Repair 512 Report

## 1. 这次实验在比较什么

这组实验是对 `iter9 bundle` 的第一轮廉价修补，目标只有一个：

- 在不拖垮 `mixed` 的前提下，优先把 `math rollout_tail` 修回来

共同底座：

- `full + depth2 + self_check_k=2`
- `reason_loops = 15`
- `rollout_steps = 10`
- `world_mask_strategy = structured`
- `world_full_simplify_loss = true`
- `self_rollout_hierarchical = true`
- `exit_two_step_aux_weight = 0.25`
- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `stage2_steps = 512`

对比组：

- `iter2 retained baseline`
- `iter9 bundle`
- `Exp A`: 降低 `self/world coupling` 权重到 `0.02`
- `Exp B`: 仅对 `math probe` 提高 rollout / loops 密度（`15 / 20`）
- `Exp C`: 仅对 `math probe` 采用更保守退出（`threshold=0.92, min_loops=4`）

## 2. 核心结果表

| 配置 | mixed rollout_tail | math rollout_tail | dialogue rollout_tail | emotion rollout_tail | persona rollout_tail |
|---|---:|---:|---:|---:|---:|
| iter2 | 0.0410 | 0.0371 | 0.0410 | 0.0879 | 0.3613 |
| iter9 | 0.0391 | 0.0410 | 0.0371 | 0.0801 | 0.6133 |
| Exp A | 0.0430 | 0.0449 | 0.0391 | 0.1309 | 0.5127 |
| Exp B | 0.1016 | 0.1074 | 0.0977 | 0.1387 | 0.6016 |
| Exp C | 0.0820 | 0.1675 | 0.0527 | 0.1348 | 0.3477 |

## 3. 结构诊断指标

| 配置 | mixed state_var | mixed c_t drift | mixed world drift |
|---|---:|---:|---:|
| Exp A | 0.00873 | 107.19 | 7.58 |
| Exp B | 0.00524 | 118.77 | 7.95 |
| Exp C | 0.00722 | 123.83 | 7.72 |

## 4. 结论

### Exp A：便宜，但不对症

- 降低 coupling 权重没有把 `math` 拉回到 `iter2` 或 `iter9` 之上
- `emotion` 明显恶化
- mixed 也退回到了比 `iter9` 更差的水平

判断：

- 不值得继续沿这条线做细碎调参

### Exp B：dense rollout 不是局部补药

- 这是三组里最差的一组
- 不仅 `math` 没修好，`mixed / dialogue / emotion` 也一起恶化

判断：

- “只在 math probe 上加更 dense rollout” 不是当前问题的正确 leverage point

### Exp C：math exit 更保守，结果更糟

- `math rollout_tail` 从 `0.0410` 直接恶化到 `0.1675`
- 这说明当前 `math` 不是被“退出太早”单独卡住
- 更保守的退出反而把错误动态累积放大了

判断：

- 当前不应继续沿“只让 math 桶更晚退出”这条线走

## 5. 工程判断

第一组已经给出很清楚的信号：

- `iter9 bundle` 的问题不是一个便宜的局部超参补丁就能修掉
- 真正值得继续的方向，应转向更轻结构、但更直接作用于 math 表征路径的增强

因此下一步更合理的是：

- `Exp D`: compression 前后加轻量 `math adapter lane`
- `Exp E`: compressed summary -> self lane 融合加 `math-aware gate`
- 若仍不足，再试 `Exp F`: compression / fusion block 内引入 `MHC`

## 6. 一句话总结

A/B/C 三组都没有把 `math` 修回来。

所以第二组和第三组不应建立在 A/B/C 的修补结果上，而应直接建立在 `iter9 bundle` 本身之上，再用更有针对性的轻结构增强去修 math。
