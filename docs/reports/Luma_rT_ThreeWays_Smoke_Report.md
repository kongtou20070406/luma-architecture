# Luma r_t Three-Way Smoke Report

## 1. 这次实验在比较什么

这轮实验不是正式 512-step 大验证，而是先做一个诚实的 smoke test，确认 `r_t` 是否真的接入主干，并初步比较三种接法：

- `blend`
  - `c_t` 与 `r_t` 按 trust 混合注入主流
- `parallel`
  - 主流继续吃 `c_t`，同时额外叠加一个受 trust 控制的 `r_t` 偏置
- `predictor`
  - `r_t` 不直接注入主流，只进入 Self-JEPA / exit 侧预测

共同底座：

- `full + depth2 + self_check`
- `self_check_k = 2`
- `enable_math_adapter_lane = true`
- `world_mask_strategy = structured`
- `world_full_simplify_loss = true`
- `self_world_coupling_weight = 0.05`
- `self_rollout_hierarchical = true`
- `exit_two_step_aux_weight = 0.25`

smoke 配置：

- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `stage2_steps = 8`
- `samples = 2`

结果文件：

- `blend`: [/home/kt/ai/minimind/artifacts/rt_smoke_blend.json](/home/kt/ai/minimind/artifacts/rt_smoke_blend.json)
- `parallel`: [/home/kt/ai/minimind/artifacts/rt_smoke_parallel.json](/home/kt/ai/minimind/artifacts/rt_smoke_parallel.json)
- `predictor`: [/home/kt/ai/minimind/artifacts/rt_smoke_predictor.json](/home/kt/ai/minimind/artifacts/rt_smoke_predictor.json)

## 2. 先回答一个关键问题：r_t 真的活了吗

活了。

本轮在修完观测层 bug 后，三个方案的 `r_t_drift_mean` 和 `r_t_trust_mean` 都变成了非零：

| mode | r_t_drift_mean | r_t_trust_mean |
|---|---:|---:|
| blend | 2.1309 | 0.5207 |
| parallel | 2.0014 | 0.5251 |
| predictor | 1.9685 | 0.5115 |

这说明：

- `r_t` 不再只是“结构接上但没有被统计到”
- 它确实在 loop 之间变化
- trust 也确实形成了非零门控

## 3. smoke 顶层结果

| mode | mixed self_tail | mixed rollout_tail | math rollout_tail |
|---|---:|---:|---:|
| blend | 0.5430 | 0.5195 | 0.7012 |
| parallel | 0.5879 | 0.5234 | 0.6250 |
| predictor | 0.5137 | 0.5840 | 0.5938 |

## 4. 和当前 D 基线相比怎么样

当前 `Exp D` 的正式 512-step 结果是：

- `mixed rollout_tail = 0.03125`
- `math rollout_tail = 0.041015625`

而这次 `r_t` 三组 smoke：

- mixed 全都在 `0.52 ~ 0.58`
- math 全都在 `0.59 ~ 0.70`

也就是说：

- 三种 `r_t` 接法都已经明显落后于 `D`
- 而且不是小落后，是 smoke 级别就已经拉开了一个数量级

## 5. 三种方案各自的味道

### 5.1 blend

优点：

- mixed 三组里 rollout 最低
- 说明“让 `c_t` / `r_t` 竞争混合”是最自然的一种接法

缺点：

- math 最差
- 更像把局部 reasoning state 混进主流以后，数学递推反而更乱了

### 5.2 parallel

优点：

- math 比 `blend` 稍好
- 结构直觉也比较干净

缺点：

- mixed 没好到值得继续烧 512-step
- self tail 三组里最差

### 5.3 predictor

优点：

- 不直接污染主流
- mixed self tail 三组里最好

缺点：

- rollout 最差
- 更像 `r_t` 只是被 predictor 消化了，但没真正转成更好的多步动力学

## 6. 当前判断

### 结论 1：这个方向不是“没接上”，而是“接上后暂时不划算”

这点很重要。

现在不是：

- `r_t` 完全没生效

而是：

- `r_t` 已经生效
- 但在当前 D 底座上，它带来的扰动远大于收益

### 结论 2：现在不值得直接上 512-step 正式比对

原因很简单：

- 8-step smoke 已经远差于 D
- 这时直接烧 512-step 更像是在为一个明显偏离基线的分支付费

### 结论 3：如果以后还要追 r_t，优先保留 `parallel` 或 `predictor`

如果以后一定要继续追这条线，我会优先保留：

- `parallel`
  - 因为 math 相对没那么坏
- `predictor`
  - 因为它最少直接污染主流

而不会优先保留 `blend`。

## 7. 顺手回答：D 现在比 iter2 的 math 好吗

还没有。

- `iter2 math rollout_tail = 0.037109375`
- `D math rollout_tail = 0.041015625`

所以：

- `D` 是当前最好的“iter9 修复候选”
- 但它仍然没有真正超过 `iter2` 的 math

## 8. 当前最合理的下一步

如果继续修 math / mixed，而不是为了概念完整性强追 `r_t`，更值得的还是：

1. 继续沿 `D` 线微调
2. 或者回到已验证的 JEPA / exit / adapter 小改动
3. 暂时不把 `r_t` 作为高优先级正式实验线

一句话总结：

- `r_t` 现在已经真的活了
- 但这三种接法在当前 D 底座上都明显不如现有基线
- 所以这轮更像“完成概念验毒”，而不是“发现下一条主线”
