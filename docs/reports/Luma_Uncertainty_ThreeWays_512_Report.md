# Luma Uncertainty Three-Ways Report (512-step)

## 1. 这次实验在比较什么

在当前默认基线：

- `one-step continuation gain` 主监督
- `light two-step continuation auxiliary` 默认辅助

之上，我们继续测试三种更克制的 uncertainty 接法：

1. `clipped uncertainty weighting`
- uncertainty 只允许影响 `10%~20%` 的 two-step 权重

2. `uncertainty-as-gate`
- 不直接放大 two-step loss
- 只在高 uncertainty 时允许 two-step 生效

3. `crystal + uncertainty` 低火力版
- uncertainty 和 crystal 都只做小幅辅助
- 都不允许大权重主导

统一配置：

- `full + depth2 + self_check`
- `self_check_k = 2`
- `rollout_steps = 10`
- `reason_loops = 15`
- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `enable_math_adapter_lane = true`
- `stage2_steps = 512`

基线对照：

- `baseline = two_step_aux`

---

## 2. 基线

| bucket | self_tail | rollout_tail |
|---|---:|---:|
| math | 0.0571 | 0.0625 |
| dialogue | 0.0591 | 0.0586 |
| emotion | 0.0896 | 0.1094 |
| persona_seed | 0.5527 | 0.2461 |
| mixed | 0.0544 | 0.0527 |

这是当前要守住的健康形态：

- rollout 非零
- 各桶有区分度
- 没有被统一压成极端常数

---

## 3. 三组 uncertainty 实验结果

### 3.1 `clipped uncertainty weighting`

配置：

- `exit_uncertainty_two_step_mode = clipped`
- `exit_uncertainty_two_step_weight = 0.2`
- `exit_uncertainty_two_step_cap = 0.2`

结果：

| bucket | self_tail | rollout_tail | uncertainty_mean |
|---|---:|---:|---:|
| math | 0.0869 | 0.0 | 1.0 |
| dialogue | 0.0781 | 0.0 | 1.0 |
| emotion | 0.1650 | 0.0 | 0.9980 |
| persona_seed | 1.1758 | 0.6094 | 0.9985 |
| mixed | 0.0781 | 0.0 | 1.0 |

判断：

- 没救回来
- uncertainty 仍然饱和
- rollout 继续被压扁
- persona_seed 明显更差

### 3.2 `uncertainty-as-gate`

配置：

- `exit_uncertainty_two_step_mode = gate`
- `exit_uncertainty_gate_threshold = 0.85`
- `exit_uncertainty_two_step_weight = 0.2`

结果：

| bucket | self_tail | rollout_tail | uncertainty_mean |
|---|---:|---:|---:|
| math | 0.0746 | 0.0 | 0.9961 |
| dialogue | 0.0967 | 0.0 | 1.0 |
| emotion | 0.1758 | 0.0 | 0.9941 |
| persona_seed | 0.5957 | 0.0 | 0.9769 |
| mixed | 0.1042 | 0.0 | 0.9980 |

判断：

- 也没有救回来
- gate 没能恢复 rollout 的健康区分度
- 因为 uncertainty 本身已经几乎总是高位
- 所以 gate 更像“几乎总开”

### 3.3 `crystal + uncertainty` 低火力版

配置：

- `enable_exit_jepa_crystal = true`
- `exit_crystal_two_step_weight = 0.1`
- `exit_crystal_two_step_cap = 0.1`
- `exit_uncertainty_two_step_mode = clipped`
- `exit_uncertainty_two_step_weight = 0.1`
- `exit_uncertainty_two_step_cap = 0.1`

结果：

| bucket | self_tail | rollout_tail | uncertainty_mean |
|---|---:|---:|---:|
| math | 0.0872 | 0.0 | 0.8994 |
| dialogue | 0.0330 | 0.0 | 0.1059 |
| emotion | 0.0725 | 0.0 | 0.4568 |
| persona_seed | 0.5107 | 0.0 | 0.0080 |
| mixed | 0.0430 | 0.0 | 0.5026 |

补充：

- `stage1 jepa_crystal_mean = 0.0793`
- `stage1 uncertainty_mean = 0.5814`

判断：

- 这是三组里 mixed `self_tail` 最好的一组
- 但 rollout 仍然全桶归零
- 所以它也不能判为健康改进

---

## 4. 总结判断

### 结论 1：问题不只是“weight 太大”

因为：

- `raw` 会压扁
- `clipped` 也压扁
- `gate` 也压扁
- `crystal + uncertainty` 低火力版还是压扁

说明现在的问题更深一层：

- uncertainty 头在当前接法下太容易饱和
- 一旦把它接进 two-step 辅助，就很容易把 rollout supervision 的区分能力一起带坏

### 结论 2：当前 uncertainty 不适合继续直接绑在 two-step 权重上

这条线现在最诚实的判断是：

- 有信号
- 但接法不对
- 不该继续在“直接调 two-step loss 权重”这条支路上深挖

### 结论 3：当前默认基线不变

仍然保持：

- `one-step main + light two-step auxiliary`

不把 uncertainty 接进默认主线。

---

## 5. 下一步建议

如果以后还想继续保留 uncertainty，这条线更值得改成：

1. `uncertainty 只进 exit feature，不碰 two-step loss 权重`
2. `uncertainty 先做 ranking / diagnostics，不做 supervision scaling`
3. `先修 uncertainty 头本身的饱和问题，再谈接入 two-step`

---

## 6. 一句话总结

- 三种更克制的 uncertainty 方案都没有把 rollout 从“被压平”状态救回来
- 所以 uncertainty 值得保留成研究项
- 但当前不应该继续用“直接调 two-step 辅助权重”的方式推进
