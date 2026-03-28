# Luma r_t Local Router 512 Report

## 1. 这次实验在比较什么

这次实验回答的是两个问题：

1. 把 `r_t` 重新定义成偏局部递推状态以后，它能不能在 `512-step` 口径下稳定工作。
2. 用一个动态 `switch gate` 在 `c_t` 和 `r_t` 之间做路由，是否比当前 `iter2 / Exp D` 更有价值。

共同底座：

- `full + depth2 + self_check`
- `self_check_k = 2`
- `rollout_steps = 10`
- `reason_loops = 15`
- `world_mask_strategy = structured`
- `world_full_simplify_loss = true`
- `self_world_coupling_weight = 0.05`
- `self_rollout_hierarchical = true`
- `enable_math_adapter_lane = true`
- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `stage2_steps = 512`

对比组：

- `predictor`：`r_t` 不直接进入主流，只进入 Self-JEPA / exit 侧
- `parallel`：`c_t` 主注入保留，`r_t` 作为并联偏置进入主流
- 对照：`iter2`
- 对照：`Exp D`

## 2. 新的 gate 到底是不是动态的

是的，而且不是假动态。

### `predictor`

- `r_t_drift_mean = 2.4532`
- `r_t_trust_mean = 0.5028`
- `r_t_switch_mean = 0.4643`

### `parallel`

- `r_t_drift_mean = 2.5132`
- `r_t_trust_mean = 0.5051`
- `r_t_switch_mean = 0.4663`

这说明：

- `r_t` 自身在循环中确实在变
- `trust` 没塌到 `0` 或 `1`
- `switch gate` 也没有钉死在单一路径

所以从“机制有没有活”这个角度看，这条线是成立的。

## 3. 核心结果表

| 配置 | mixed self_tail | mixed rollout_tail | math self_tail | math rollout_tail | dialogue self_tail | emotion self_tail | persona rollout_tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| iter2 | 0.0587 | 0.0410 | 0.0563 | 0.0371 | 0.0513 | 0.0704 | 0.3613 |
| Exp D | 0.0480 | 0.0313 | 0.0516 | 0.0410 | 0.0488 | 0.0645 | 0.2988 |
| predictor | 0.1021 | 0.0000 | 0.0715 | 0.0000 | 0.0896 | 0.1182 | 0.3965 |
| parallel | 0.0540 | 0.0000 | 0.0906 | 0.0000 | 0.0483 | 0.1211 | 0.0000 |

## 4. 怎么理解这些结果

### 结论 1：动态 local router 不是空壳

这一点可以确认。

`r_t` 和 `switch gate` 都形成了非零、非饱和的动态信号。

所以这条线不是“实现失败”，而是“实现成功但当前收益不够好”。

### 结论 2：`predictor` 比 `parallel` 更符合原始直觉

从设计哲学上看，`predictor` 更像我们想要的：

- `c_t` 继续做主元认知状态
- `r_t` 更偏局部递推辅状态
- 不直接抢主流注入权

结果上它也比 `parallel` 更像“可继续研究”的版本：

- `math self_tail` 更好
- `persona rollout_tail` 还保留了非零结构

但它的 mixed 和 dialogue/emotion 明显不如当前主线。

### 结论 3：`parallel` 在 mixed self_tail 上更好，但代价更大

`parallel` 的 mixed `self_tail = 0.0540`，比 `predictor` 的 `0.1021` 好很多。

但它的问题是：

- `math self_tail` 明显变差
- `persona rollout_tail` 直接掉到 `0.0`
- 它更像在把 `r_t` 当成一条新的强注入路径，而不是安全辅路

所以它不适合作为当前默认基线。

### 结论 4：两条 `r_t` 线都没有超过 `iter2 / Exp D`

这是这轮最重要的工程结论。

- `iter2` 仍然是 math 最优：`0.0371`
- `Exp D` 仍然是当前 mixed 最优：`0.0313`
- 新 `r_t` 两个版本都没有形成可替代优势

所以当前不能因为“gate 很聪明”就误判成“这条线已经值得扶正”。

## 5. 为什么 rollout_tail 会出现大面积 0

这不是报告抄错了，而是当前 probe 口径下的真实结果。

更准确地说：

- `r_t` 线在 mixed-trained probe 上，很多 bucket 的 rollout 分量没有成熟出稳定的正损失窗口
- 所以 `self_tail` 比 `rollout_tail` 更有解释力
- 这也从侧面说明：当前 `r_t` 线还没有把“更好的局部递推动力学”稳定兑现出来

## 6. 当前最合理的工程判断

### 保留主线

仍然建议保留：

- `full + depth2 + self_check_k=2`
- 当前结构修复优先沿 `Exp D` 继续

### 关于 `r_t`

当前不建议直接上 `512-step` 主线继续烧更多版本。

更合理的定位是：

- `r_t` 的“局部递推 + 动态路由”概念验证已经完成
- 机制是活的
- 但当前收益不够，暂不进入主线

### 如果以后还要继续追 `r_t`

优先顺序建议：

1. `predictor`
2. `parallel`
3. 不再回去追旧的 `blend`

因为：

- `predictor` 更符合“安全接入”的原始目标
- `parallel` 虽然 mixed self 更漂亮，但副作用更大

## 7. 重新确立：自省流当前到底在做什么

当前 Luma 的自省流更准确地说承担四件事：

1. `c_t`：慢环主状态
   - 负责记录“当前认知状态如何变化”
   - 偏全局、偏元认知
   - 不是 identity，不是 world latent

2. Self-JEPA anchor
   - 提供 `Δc_t` 的监督目标
   - 让模型学“认知状态如何推进”

3. exit-side evidence
   - 自省流本身不直接决定退出
   - 但会通过 `self_error / rollout / self_check` 影响 continuation 价值判断

4. know-gap source
   - 它仍然是未来 Router / Tape 检索触发的主要候选信号来源

也就是说：

- `c_t` 不是局部 scratchpad
- `r_t` 如果以后保留，才更像局部递推 scratchpad
- 目前主线仍然是“强 `c_t` + 轻 `self_check`”，不是“双状态并列主导”

## 8. 一句话总结

- 新的 sliding-window dynamic switch gate 是活的
- 但当前 `r_t` 两个 `512-step` 版本都没超过 `iter2 / Exp D`
- 所以这条线暂时不进入主线
- 当前更值得继续的是：保留 `Exp D` 作为结构修复主线，`r_t` 暂存为概念验证分支
