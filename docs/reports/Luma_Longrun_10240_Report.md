# Luma 10240-Step Midcourse Report

## 1. 这轮在测什么

这轮是第一次真正的中程验证，而不是 128/512 step 的短程 probe。

共同设置：
- `stage2_steps = 10240`
- `full + depth2 + self_check`
- `reason_shared_depth = 2`
- `rollout_steps = 10`
- `reason_loops = 15`
- 数据桶统一来自 mixed 主线，再做分桶 probe：
  - `math`
  - `dialogue`
  - `emotion`
  - `persona_seed`
  - `python_code`
  - `mixed`

对比组：
- `iter2_10240`
- `iter9_10240`
- `iter9_crystal_10240`
- `expd_iter9_math_adapter_10240`

## 2. 核心结果表

| name | mixed_self | mixed_roll | math_self | math_roll | dialogue_self | dialogue_roll | emotion_self | emotion_roll | persona_seed_self | persona_seed_roll | python_code_self | python_code_roll |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iter2_10240 | 0.0071 | 0.0000 | 0.0073 | 0.0000 | 1.0195 | 0.6875 | 0.4497 | 0.2305 | 0.8613 | 0.6328 | 0.0485 | 0.0000 |
| iter9_10240 | 0.0107 | 0.0000 | 0.0109 | 0.0000 | 0.6914 | 0.0000 | 0.3118 | 0.0000 | 0.7695 | 0.0000 | 0.0758 | 0.0000 |
| iter9_crystal_10240 | 0.0198 | 0.0000 | 0.0157 | 0.0000 | 0.7197 | 0.0000 | 0.2759 | 0.0000 | 0.8262 | 0.4258 | 0.1384 | 0.0000 |
| expd_iter9_math_adapter_10240 | 0.0106 | 0.0000 | 0.0105 | 0.0000 | 1.0664 | 0.3389 | 0.1572 | 0.0000 | 0.9414 | 0.5586 | 0.0615 | 0.0000 |

补充：`stage1 hard_loop_var`

| name | mixed | math | dialogue | emotion | persona_seed | python_code |
|---|---:|---:|---:|---:|---:|---:|
| iter2_10240 | 0.0000 | 0.0000 | 0.2500 | 0.6875 | 0.1094 | 1.2344 |
| iter9_10240 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2344 | 0.0000 |
| iter9_crystal_10240 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4844 | 0.0000 |
| expd_iter9_math_adapter_10240 | 0.0000 | 0.0000 | 1.6875 | 0.0000 | 1.3594 | 0.9844 |

## 3. 最重要的观察

### 观察 1：长程下，`iter2` 重新成为最稳主线

如果只看 `mixed_self` 和 `math_self`：
- `iter2_10240` 最低
- 比 `iter9`、`iter9 + crystal`、`ExpD` 都更好

这说明：
- 短程里 `iter9` 那种“更激进、更 LeWorldModel 化”的 bundle 有吸引力
- 但一拉到 `10240 step`，`iter2` 的稳态优势反而更明显

一句话说：
- `iter2` 更像真正能扛长训的骨架

### 观察 2：`iter9` 系列在长程里更像“把 rollout 压平了”

这轮最醒目的现象是：
- `iter9`
- `iter9 + crystal`

在几乎所有桶上 `rollout_tail = 0.0`。

这个结果不能直接当成“完美”。
它更像我们之前遇到过的那类情况：
- rollout 不是自然学好了
- 而是被压平了、失去分辨度了

尤其是结合：
- `stage1 hard_loop_var` 大面积为 `0.0`
- `dialogue/emotion/python_code` 也一起归零

这更接近：
- 退出/动力学信号被压缩得过于保守
- 而不是模型真的把多步 rollout 全学会了

### 观察 3：`iter9 + crystal` 没有在长程里兑现额外收益

相比 `iter9`：
- `iter9_crystal` 的 `mixed_self` 更差
- `math_self` 更差
- `python_code_self` 明显更差

它唯一保住一点的地方是：
- `persona_seed_rollout` 没像 `iter9` 一样被压成 `0`

但综合来看：
- `crystal` 这条线在 `10240 step` 下没有扶正理由

### 观察 4：`ExpD` 没把 math 修回来，但保住了一部分 bucket 动态性

`ExpD` 的特点很鲜明：
- `dialogue_rollout = 0.3389`
- `persona_seed_rollout = 0.5586`
- `stage1 hard_loop_var` 在 `dialogue/persona_seed/python_code` 也明显非零

这说明：
- 它确实没有像 `iter9` 那样把整个动态退出系统压扁

但问题也同样明确：
- `math_self` 没赢过 `iter2`
- `mixed_self` 也没赢过 `iter2`
- `persona_seed_self` 反而更差

所以：
- `ExpD` 是一个“保动态性”的修复线
- 但还不是主线替代品

### 观察 5：`python_code` 桶是有区分度的，而且 `iter2` 最稳

`python_code_self`：
- `iter2 = 0.0485`
- `ExpD = 0.0615`
- `iter9 = 0.0758`
- `iter9_crystal = 0.1384`

这很说明问题：
- 代码桶没有和别的桶一起完全失真
- 它能区分出谁更稳
- 而且当前最稳的还是 `iter2`

## 4. 当前工程结论

### 主线结论

当前长程结果支持：
- 把 `iter2` 重新确立为正式中程主线基线

也就是：
- `full + depth2 + self_check_k=2`
- `one-step continuation gain` 主监督
- `light two-step auxiliary` 辅助

### 对 `iter9` 的结论

`iter9` 不能删，因为它提供了很重要的研究方向：
- 更接近 LeWorldModel 的 world branch
- structured mask
- simplified full world JEPA

但在 `10240 step` 下，它现在更像：
- 一个会把 rollout 压平的研究分支
- 还不是当前正式主线

### 对 `crystal` 的结论

`crystal` 暂时不值得进主线：
- 短程没有稳定兑现
- 长程也没有反超 `iter2`

### 对 `ExpD` 的结论

`ExpD` 仍然值得保留：
- 它说明“轻结构修 math / 保动态性”是有空间的
- 但它现在还没强到能替换 `iter2`

## 5. 下一步建议

### 建议 A：主线先回到 `iter2`

这轮之后最稳的做法是：
- 把 `iter2` 作为后续中程/正式 trainer 的默认骨架

### 建议 B：如果继续推 `iter9`，优先修“rollout 被压平”这个问题

重点不再是继续堆：
- crystal
- coupling
- 更强 hierarchical rollout

而是要想办法恢复：
- rollout 的区分度
- exit 的动态性

### 建议 C：`ExpD` 保留为第二修复线

因为它至少证明了：
- 在不把整个系统做大的前提下
- 轻结构 lane 确实有机会修局部任务

但目前它还没超过主线。

## 6. 一句话总结

这轮 `10240-step` 的结果很清楚：
- **`iter2` 是当前最稳、最适合继续往正式长训推进的基线**
- `iter9` 和 `iter9 + crystal` 更像研究分支，当前长程里会把 rollout 压得过平
- `ExpD` 是有价值的修复线，但还没有赢过 `iter2`
