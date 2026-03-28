# Luma Self-Check Cadence And JEPA Crystal Report

## 1. 这次实验在比较什么

这次实验专门回答两个问题：

1. `self_check_k` 到底是不是 `2` 更合适，还是 `1 / 3` 更好
2. 在退出信号里加入 `JEPA-guided entropy crystallization` 以后，是否真的更有利于动态退出与整体表现

共同配置：

- `full + depth2 + self_check`
- `rollout_steps = 10`
- `reason_loops = 15`
- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `stage2_steps = 128`

对比组：

- `k1`
- `k2`
- `k3`
- `k2 + crystal`
- `k3 + crystal`
- `k4 + crystal`
- `k3 + crystal + 10x20`

## 2. 顶层结果

| 配置 | hard_loop_var | self_tail | rollout_tail | crystal_mean |
|---|---:|---:|---:|
| k1 | 1.1094 | 0.1909 | 0.2266 | - |
| k2 | 0.9375 | 0.1631 | 0.1582 | - |
| k3 | 0.9375 | 0.1519 | 0.1406 | 0.0000 |
| k2 + crystal | 0.9375 | 0.1072 | 0.0879 | 0.1739 |
| k3 + crystal | 0.9844 | 0.1035 | 0.1348 | 0.1626 |
| k4 + crystal | 0.9375 | 0.1606 | 0.1445 | 0.1739 |
| k3 + crystal + 10x20 | 0.9844 | 0.1035 | 0.1348 | 0.1626 |

### 先看总体

- `k1` 不是最优
- `k2` 已经明显优于 `k1`
- `k3` 在总体 `self/rollout` 上继续变好
- `k2 + crystal` 在顶层 `self_tail / rollout_tail` 上最好
- `k3 + crystal` 在顶层 `self_tail` 上继续很强，但 mixed rollout 没有同步变好
- `k4 + crystal` 没有继续提升
- `k3 + crystal + 10x20` 与 `10x15` 几乎重合，说明额外 loops 预算仍未被真正利用

但这还不是最终结论，因为 Luma 不能只看一个 mixed 顶层数字。

## 3. 分桶结果

### 3.1 `k1`

| bucket | hard_loop_var | self_tail | rollout_tail |
|---|---:|---:|---:|
| math | 1.25 | 0.1208 | 0.1367 |
| dialogue | 0.6875 | 0.3027 | 0.2168 |
| emotion | 3.6875 | 0.0957 | 0.1133 |
| persona_seed | 0.1875 | 0.0586 | 0.0664 |
| mixed | 1.1094 | 0.2734 | 0.2617 |

### 3.2 `k2`

| bucket | hard_loop_var | self_tail | rollout_tail |
|---|---:|---:|---:|
| math | 1.25 | 0.1052 | 0.1426 |
| dialogue | 0.5 | 0.2432 | 0.2676 |
| emotion | 3.6875 | 0.1069 | 0.1289 |
| persona_seed | 0.1875 | 0.0573 | 0.0840 |
| mixed | 0.9375 | 0.2607 | 0.2168 |

### 3.3 `k3`

| bucket | hard_loop_var | self_tail | rollout_tail |
|---|---:|---:|---:|
| math | 1.25 | 0.1133 | 0.1836 |
| dialogue | 0.5 | 0.1162 | 0.1211 |
| emotion | 3.6875 | 0.1187 | 0.1152 |
| persona_seed | 0.1094 | 0.0367 | 0.0586 |
| mixed | 0.9375 | 0.2529 | 0.2559 |

### 3.4 `k2 + crystal`

| bucket | hard_loop_var | self_tail | rollout_tail |
|---|---:|---:|---:|
| math | 1.1875 | 0.1099 | 0.1172 |
| dialogue | 0.6875 | 0.1812 | 0.1953 |
| emotion | 3.1875 | 0.0886 | 0.1094 |
| persona_seed | 0.0 | 0.0337 | 0.0527 |
| mixed | 0.9375 | 0.2549 | 0.2949 |

补充：

- `k2 + crystal` 的 `jepa_crystal_mean = 0.1739`
- 说明这个信号不是空的，确实形成了非零结晶度

### 3.5 `k3 + crystal`

| bucket | hard_loop_var | self_tail | rollout_tail |
|---|---:|---:|---:|
| math | 1.1875 | 0.1206 | 0.1426 |
| dialogue | 0.6875 | 0.1465 | 0.1719 |
| emotion | 3.6875 | 0.1206 | 0.1426 |
| persona_seed | 0.0 | 0.0419 | 0.0488 |
| mixed | 0.9844 | 0.1367 | 0.2598 |

补充：

- `k3 + crystal` 的 `jepa_crystal_mean = 0.1626`
- 它在 persona 上的 `hard_loop_var` 也掉到了 `0.0`

### 3.6 `k4 + crystal`

| bucket | hard_loop_var | self_tail | rollout_tail |
|---|---:|---:|---:|
| math | 1.1875 | 0.1191 | 0.1250 |
| dialogue | 0.6875 | 0.2158 | 0.1641 |
| emotion | 3.1875 | 0.0908 | 0.1328 |
| persona_seed | 0.0 | 0.0361 | 0.0430 |
| mixed | 0.9375 | 0.2085 | 0.2578 |

### 3.7 `k3 + crystal + 10x20`

这个组合在当前 128-step 短程实验里，与 `k3 + crystal + 10x15` 几乎完全一致。

这说明当前问题仍然更像是：

- exit policy / continuation value 先把样本截断
- 而不是 `reason_loops` 上限不够

## 4. 怎么理解这些结果

### 结论 1：`k=2` 不是偶然点

它确实比 `k=1` 更好，这一点比较明确：

- `k1 rollout_tail = 0.2266`
- `k2 rollout_tail = 0.1582`

所以“慢一点的 self_check”是有效方向，不是噪声。

### 结论 2：`k=3` 很有意思，但不等于直接扶正

`k=3` 的特点是：

- 顶层 `self/rollout` 继续变好
- `dialogue` 和 `persona_seed` 很强

但它的问题是：

- `math rollout_tail` 变差
- mixed 的 rollout 反而比 `k=2` 更差

所以它更像：

- 对话/人格友好型 self_check 节奏

而不是当前最平衡的默认值。

### 结论 3：`k2 + crystal` 很强，但 mixed 上不稳

它的优点非常明显：

- 顶层 `self_tail / rollout_tail` 最好
- `emotion` 和 `persona_seed` 也很好

但代价也很明显：

- mixed `rollout_tail` 反而退回到 `0.2949`
- persona 的 `hard_loop_var` 直接掉到 `0.0`

这说明：

- `JEPA crystal` 不是无效信号
- 但当前它更像在“局部收紧退出判据”
- 还没有自然转化成更好的 mixed 总体动态利用

### 结论 4：`k3 + crystal` 比 `k4 + crystal` 更值得保留

因为：

- `k3 + crystal` 的顶层 `self_tail` 更好
- `k4 + crystal` 没有继续放大 crystal 的优势
- `k4 + crystal` 更像“更新过慢，退出信号变钝”

所以如果 crystal 继续保留专项研究，应该优先保留：

- `k=3 + crystal`

### 结论 5：`10x20` 仍未兑现额外预算

`k3 + crystal + 10x20` 与 `10x15` 基本重合，说明：

- 问题不是没有更长预算
- 问题仍然是退出策略没有把更长预算转成真实收益

## 5. 当前最合理的工程判断

如果目标是：

### “当前最稳默认”

仍然建议：

- `full + depth2 + self_check`
- `self_check_k = 2`

原因：

- 它不是所有桶最强
- 但它是当前最平衡、最不偏科的选择

### “更偏对话 / persona / 聊天伙伴感”

可以继续跟踪：

- `self_check_k = 3`

### “更激进的退出信号研究”

可以保留专项分支：

- `self_check_k = 2 + JEPA crystal`
- `self_check_k = 3 + JEPA crystal`

但现在还不建议扶正为默认，因为 mixed 总体还没有兑现出来。

## 6. 关于“当前 JEPA predictor 是不是官方实现”

不是。

当前状态更准确地说是：

- `SelfJEPAResidualPredictor`：自研工程实现
- `WorldLatentJEPA`：自研 scaffold
- `LeWorldModelStyleJEPA`：论文风格工程迁移版

所以现在是：

- 方向对齐论文
- 但 predictor 本体不是官方原仓直接实现

### 一句话总结

- `self_check_k = 2` 仍然是当前最稳默认
- `k=3` 值得作为“聊天伙伴感更强”的专项候选
- `k=3 + crystal` 比 `k=4 + crystal` 更值得继续研究
- `JEPA crystal` 值得继续研究，但现在还不适合直接扶正
- `10x20` 仍未被真正利用起来，下一步应继续攻 exit policy，而不是只加 loops
