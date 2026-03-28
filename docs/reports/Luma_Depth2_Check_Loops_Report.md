# Luma Depth2 Loops And Self-Check Report

## 1. 这次实验在比较什么

这轮实验专门围绕新的基础假设展开：

- 基底使用 `full + depth2 + self_check`
- 目标是思考：如何在**尽量不掉 emotion** 的情况下，提高推理能力

因此这次比较了四组：

1. `baseline_d2_10x15`
2. `depth2_10x20`
3. `depth2_slowcheck2`
4. `depth2_heavier`

共同设置：

- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `stage2_steps = 128`
- `world_jepa_mode = full`
- `reason_shared_depth = 2`

其中：

- `depth2_10x20`
  - 把 `reason_loops` 从 `15` 提到 `20`
- `depth2_slowcheck2`
  - 把 `self_check_k` 从 `1` 改到 `2`
- `depth2_heavier`
  - 轻量增重自省/自检：
    - `meta_dim = 80`
    - `meta_state = 24`
    - `c_t_dim = 40`
    - `self_check_dim = 24`

## 2. 核心结果表

| 配置 | top hard_loop_var | top self_tail | top rollout_tail |
|---|---:|---:|---:|
| baseline_d2_10x15 | 1.1094 | 0.1909 | 0.2266 |
| depth2_10x20 | 1.1094 | 0.1909 | 0.2266 |
| depth2_slowcheck2 | 0.9375 | 0.1631 | 0.1582 |
| depth2_heavier | 9.7344 | 0.1182 | 0.2207 |

先看总体结论：

- 单纯增加 `reason_loops` 到 `20`，在当前条件下**没有带来可见收益**
- 把 `self_check` 放慢到 `k=2`，能明显改善总体 `self/rollout` 收敛
- 轻量增重自省/自检会把循环深度分布拉得很开，但开始出现任务偏置

## 3. 各任务桶结果

### 3.1 baseline_d2_10x15

| bucket | hard_loop_var | self_tail | rollout_tail |
|---|---:|---:|---:|
| math | 1.25 | 0.1208 | 0.1367 |
| dialogue | 0.6875 | 0.3027 | 0.2168 |
| emotion | 3.6875 | 0.0957 | 0.1133 |
| persona_seed | 0.1875 | 0.0586 | 0.0664 |
| mixed | 1.1094 | 0.2734 | 0.2617 |

### 3.2 depth2_10x20

结果与 `baseline_d2_10x15` 几乎完全一致。

这说明：

- 当前不是最大 loop 上限不够
- 而是更长 loops 没有被真正利用

也就是：瓶颈还是 `exit policy / continuation value`，而不是简单预算不足

### 3.3 depth2_slowcheck2

| bucket | hard_loop_var | self_tail | rollout_tail |
|---|---:|---:|---:|
| math | 1.25 | 0.1052 | 0.1426 |
| dialogue | 0.5 | 0.2432 | 0.2676 |
| emotion | 3.6875 | 0.1069 | 0.1289 |
| persona_seed | 0.1875 | 0.0573 | 0.0840 |
| mixed | 0.9375 | 0.2607 | 0.2168 |

怎么理解：

- mixed 总体更好了
- math `self_tail` 更好
- emotion 略有回退，但不是灾难性回退
- persona_seed 的 rollout 也略差一些

这更像：

- 放慢 `self_check` 让它少一点噪声
- 帮 mixed 总体和推理收敛
- 但也让情感/人格相关桶失去了一点即时读数优势

### 3.4 depth2_heavier

| bucket | hard_loop_var | self_tail | rollout_tail |
|---|---:|---:|---:|
| math | 0.6875 | 0.1626 | 0.2461 |
| dialogue | 7.5 | 0.0815 | 0.0996 |
| emotion | 7.6875 | 0.1353 | 0.2285 |
| persona_seed | 0.0 | 0.0426 | 0.0352 |
| mixed | 9.7344 | 0.2129 | 0.1992 |

这组很有戏剧性。

优点：

- mixed 的循环深度分布被大幅拉开
- dialogue 提升很大
- persona_seed 提升也很明显

代价：

- math 明显变差
- emotion 的 self/rollout 也明显变差

所以它不是“全面升级”，而更像：

- 更强的表达/对话/人格驱动配置
- 但开始牺牲任务均衡性

## 4. 最重要的判断

### 结论 1：`10x20` 当前没必要直接扶正

因为：

- 在 `depth2` 基底上，`10x20` 和 `10x15` 基本一样

这说明当前更应该投资的是：

- exit policy
- continuation value 学习

而不是继续盲目加 loops

### 结论 2：慢 self_check 是值得继续保留的候选

虽然它不是全桶都更好，但它有一个很好的特征：

- mixed 总体明显改善
- emotion 没有崩

所以如果目标是：

- 在不明显掉 emotion 的情况下提升推理

那 `self_check_k = 2` 是个很值得继续跟踪的方向

### 结论 3：轻量增重很强，但当前太偏科

它已经证明了一件事：

- Luma 不是已经到容量上限了
- 自省/自检容量稍微加一点，系统是会明显响应的

但当前响应方向太偏：

- 更像推高 `dialogue / persona_seed`
- 同时压了 `math / emotion`

所以这组更适合作为专项分支，而不是现在就扶正成默认

## 5. 当前最合理的下一步

如果目标是：

### “尽量不掉 emotion，同时继续提推理”

我建议按这个顺序走：

1. 基底保持：
   - `full + depth2 + self_check`
2. loops 保持：
   - `10x15`
3. 优先继续验证：
   - `self_check_k = 2`
4. 暂不扶正：
   - `10x20`
   - `heavier introspection/self_check`

### 一句话总结

在 `full + depth2` 基底上，真正值得继续追的不是“更多 loops”，而是“更干净的 self_check 调度”。

`self_check_k = 2` 目前是最像“提高推理、但不明显伤 emotion”的候选。
