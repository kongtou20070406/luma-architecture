# Luma Introspection-Uncertainty Two-Step Report (512-step)

## 1. 这次实验在比较什么

这次实验在当前默认基线之上，加了一条新的轻量信号：

- `introspection uncertainty`

它来自自省流，而不是来自 world branch 或外部 probe。

目标不是直接改写 exit logit，而是：

- 让自省流给出“当前到底有多疑惑 / 多不确定”
- 再用这个 uncertainty 去调节 `light two-step continuation auxiliary` 的权重

对比组：

- `two_step_aux`：当前默认基线
- `two_step_aux_uncert`：在基线上增加 uncertainty-aware two-step weighting

共同配置：

- `full + depth2 + self_check`
- `self_check_k = 2`
- `rollout_steps = 10`
- `reason_loops = 15`
- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `enable_math_adapter_lane = true`
- `stage2_steps = 512`

---

## 2. 结果摘要

### 基线：`two_step_aux`

| bucket | self_tail | rollout_tail |
|---|---:|---:|
| math | 0.0571 | 0.0625 |
| dialogue | 0.0591 | 0.0586 |
| emotion | 0.0896 | 0.1094 |
| persona_seed | 0.5527 | 0.2461 |
| mixed | 0.0544 | 0.0527 |

### 新版：`two_step_aux_uncert`

| bucket | self_tail | rollout_tail | uncertainty_mean |
|---|---:|---:|---:|
| math | 0.0884 | 0.0 | 0.9717 |
| dialogue | 0.0840 | 0.0 | 0.9639 |
| emotion | 0.2578 | 0.0 | 0.9961 |
| persona_seed | 0.8398 | 0.0 | 1.0 |
| mixed | 0.0779 | 0.0 | 0.9678 |

stage1:

- `uncertainty_mean = 0.5795`

说明：

- uncertainty 头不是空的
- 它在 stage1 已经形成了明确非零信号

---

## 3. 怎么理解这轮结果

### 结论 1：疑惑度信号是活的

这不是“没学起来”。

相反：

- stage1 已经有非零 uncertainty
- stage2 的 bucket probe 中，uncertainty 几乎都顶到 `0.96 ~ 1.0`

所以当前问题不是：

- 这个头没作用

而是：

- 它现在太强、太早、太容易饱和

### 结论 2：当前接法不可直接扶正

因为这次出现了一个非常明显的坏信号：

- 各桶 `rollout_tail` 全部变成 `0.0`

这不是“特别好”，更像是：

- two-step 辅助被 uncertainty 放大/压缩到失去辨别结构
- rollout supervision 被不健康地压扁了

也就是说：

- 当前 uncertainty 接法不适合直接进默认基线

### 结论 3：它值得保留为研究备选，但要降火力

更合理的后续方向是：

1. 只在高 uncertainty 区间轻量抬高 two-step 权重
2. 对 uncertainty 做温度或 clipping，避免饱和到接近 `1.0`
3. 不让 uncertainty 直接决定 rollout loss 的强弱，而只做小幅调制

---

## 4. 当前工程判断

- `introspection uncertainty` 这个想法本身有价值
- 但第一轮实现方式太猛，不适合直接扶正
- 它更适合作为：
  - `one-step main + light two-step auxiliary` 之上的轻量修饰项
  - 而不是新的主导信号

---

## 5. 一句话总结

- 疑惑度头是活的
- 但这版 uncertainty-aware two-step weighting 太强，已经把 rollout 行为压坏了
- 所以它值得保留成备选研究项，但当前不应该进默认基线
