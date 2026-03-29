# Luma One-Step vs Light Two-Step Continuation Report (512-step)

## 1. 这次实验在比较什么

这次实验专门回答一个很实际的问题：

- 当前 exit learning 到底应该保持“纯 one-step continuation gain”
- 还是升级成“one-step 主监督 + light two-step auxiliary”

共同配置：

- `full + depth2 + self_check`
- `self_check_k = 2`
- `rollout_steps = 10`
- `reason_loops = 15`
- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `stage2_steps = 512`
- `enable_math_adapter_lane = true`

对比组：

- `one_step`
  - `exit_two_step_aux_weight = 0.0`
- `two_step_aux`
  - `exit_two_step_aux_weight = 0.25`

结果文件：

- `one_step`: `/home/kt/ai/minimind/artifacts/ab_onestep_512.json`
- `two_step_aux`: `/home/kt/ai/minimind/artifacts/ab_twostepaux_512.json`

---

## 2. 核心结论

这轮结果支持：

- 当前默认不该再是“纯一步”
- 更合理的是：
  - `one-step continuation gain` 做主监督
  - `light two-step continuation auxiliary` 做轻量辅助

也就是说：

- “一步和两步全都要”是对的
- 但形式必须是：
  - 一步主
  - 两步辅
  - 不是两步直接接管 exit policy

---

## 3. 顶层结果

| 配置 | mixed self_tail | mixed rollout_tail |
|---|---:|---:|
| one_step | 0.169921875 | 0.068359375 |
| two_step_aux | 0.054443359375 | 0.052734375 |

解读：

- `two_step_aux` 在 mixed 上明显更好
- 这不是轻微波动，而是很明确的改善

---

## 4. 分桶结果

| bucket | one_step self | one_step rollout | two_step_aux self | two_step_aux rollout |
|---|---:|---:|---:|---:|
| math | 0.19140625 | 0.07421875 | 0.05712890625 | 0.0625 |
| dialogue | 0.22021484375 | 0.15625 | 0.05908203125 | 0.05859375 |
| emotion | 0.19677734375 | 0.10546875 | 0.089599609375 | 0.109375 |
| persona_seed | 0.748046875 | 0.3896484375 | 0.552734375 | 0.24609375 |
| mixed | 0.169921875 | 0.068359375 | 0.054443359375 | 0.052734375 |

### 4.1 `math`

- `math rollout_tail`
  - `0.07421875 -> 0.0625`
- `math self_tail`
  - `0.19140625 -> 0.05712890625`

说明：

- 轻量两步辅助没有伤 math
- 反而把 math 拉得更稳

### 4.2 `dialogue`

- `dialogue rollout_tail`
  - `0.15625 -> 0.05859375`

这是这轮最漂亮的改善之一。

### 4.3 `emotion`

- `emotion self_tail`
  - `0.19677734375 -> 0.089599609375`
- `emotion rollout_tail`
  - `0.10546875 -> 0.109375`

说明：

- emotion 上不是全面退化
- 只是 rollout 有一个很小的回退
- 这个幅度目前看不像结构性失败

### 4.4 `persona_seed`

- `persona_seed rollout_tail`
  - `0.3896484375 -> 0.24609375`

说明：

- 这轮不是 persona 代价换 mixed
- 相反，persona 也一起回收了

---

## 5. Stage1 退出分布观察

mixed stage1：

| 配置 | hard_loop_var | soft_loop_var |
|---|---:|---:|
| one_step | 0.25 | 1.1875 |
| two_step_aux | 0.0 | 0.5 |

这说明：

- `two_step_aux` 把 loss 压低了
- 但退出分布反而更收紧了一些

所以它的当前意义更像：

- 更好的学习信号
- 还不是更成熟的退出策略本身

也就是说，下一步仍然值得继续优化 exit policy，而不是就此停下。

---

## 6. 工程判断

当前最合理的默认基线应该更新为：

- `one-step continuation gain` 作为主监督
- `light two-step continuation auxiliary` 作为默认辅助

不再建议默认使用：

- `one-step only`

### 更准确的口径

- `iter2`
  - 代表 stable one-step continuation-gain skeleton
- `iter5`
  - 代表 one-step main + light two-step auxiliary upgrade line

所以如果要一句话概括现在的主线：

- 一步做主
- 两步做辅
- 共同服务 continuation learning

---

## 7. 下一步建议

沿这条新基线继续优化 exit policy，最值得试的三条是：

1. `gain confidence gate`
- 让 one-step 和 two-step 的组合权重取决于当前 gain 不确定性
- 不确定时多参考 two-step
- 稳定时仍由 one-step 主导

2. `emotion-safe two-step weighting`
- 对 `emotion-like` 状态降低 two-step 辅助权重
- 避免在情感任务上把退出信号过度理性化

3. `crystal as auxiliary, not baseline`
- `JEPA crystal` 可以作为次级辅助特征重新试
- 但不建议直接扶正进默认主线
- 更适合做“在 one-step + light two-step 上再加一个轻量排序信号”

---

## 8. 一句话总结

当前 `512-step` 结果表明：

- 纯 one-step 已经不是最佳默认
- `one-step main + light two-step auxiliary` 是更好的当前基线
- 它让 `math / dialogue / persona_seed / mixed` 同时改善
- `emotion` 只出现了很小的 rollout 回退
