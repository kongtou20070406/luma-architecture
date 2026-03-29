# Luma Iter2 vs Iter5 vs Predictor-Progress Report (2048-step)

## 1. 这次实验在比较什么

这轮实验专门回答两个问题：

1. `iter5` 到底要不要扶正成新基线
2. `predictor_progress` 应该加在 `iter2` 线上，还是 `iter5` 线上

共同配置：

- `full + depth2 + self_check`
- `self_check_k = 2`
- `rollout_steps = 10`
- `reason_loops = 15`
- `stage2_steps = 2048`
- buckets: `math / dialogue / emotion / persona_seed / python_code / mixed`

这次我先把历史命名对齐清楚：

- `iter2`
  - 当前正式长程基线
  - `one-step main + light two-step auxiliary`
  - 不开 `math_adapter_lane`
- `iter5`
  - 这次按“最接近 512-step A/B 两步辅助升级线”的可复现定义来跑
  - 也就是：`iter2 + math_adapter_lane`
- `iter2 + predictor_progress`
  - 在 `iter2` 上加 `Self-JEPA predictor` 的 loop progress awareness
- `iter5 + predictor_progress`
  - 在 `iter5` 上再加 `predictor_progress`

## 2. 先回答一个直接问题

### `iter5` 有没有做过 10240 中程测试？

没有单独做过。

长程 `10240-step` 目前正式跑过的是：

- `iter2`
- `iter9`
- `iter9 + crystal`
- `ExpD`

所以这次 `iter5` 的判断，必须建立在新的 `2048-step` 对比上，而不是假设它已经被长程验证过。

## 3. 顶层结果

| 配置 | stage1 ct_kl | hard_loop_var | c_t_var | mixed self_tail | math self_tail | python_code self_tail |
|---|---:|---:|---:|---:|---:|---:|
| iter2 | 1.7689 | 0.8594 | 0.5618 | 0.0263 | 0.0199 | 0.0378 |
| iter5 | 1.7550 | 1.7344 | 0.4894 | 0.0325 | 0.0260 | 0.0524 |
| iter2 + predictor_progress | 1.7700 | 0.3594 | 0.5669 | 0.0131 | 0.0151 | 0.0557 |
| iter5 + predictor_progress | 1.8330 | 0.9844 | 0.5511 | 0.0255 | 0.0267 | 0.0546 |

## 4. 分桶结果

| 配置 | mixed | math | dialogue | emotion | persona_seed | python_code |
|---|---:|---:|---:|---:|---:|---:|
| iter2 | 0.0263 | 0.0199 | 0.4258 | 0.1530 | 0.7090 | 0.0378 |
| iter5 | 0.0325 | 0.0260 | 0.8086 | 0.3062 | 0.9844 | 0.0524 |
| iter2 + predictor_progress | 0.0131 | 0.0151 | 0.4893 | 0.1992 | 0.6055 | 0.0557 |
| iter5 + predictor_progress | 0.0255 | 0.0267 | 0.7520 | 0.2769 | 0.8027 | 0.0546 |

## 5. rollout 这次怎么看

这轮我已经把新指标也一起纳入了：

- `rollout_active_ratio`
- `rollout_nonzero_ratio`

结果很明确：

- 四组基本都是 `rollout_active_ratio = 1.0`
- 但大多数 bucket 的 `rollout_nonzero_ratio = 0.0`

这说明：

- 不是 rollout 逻辑完全没走到
- 而是这轮里 rollout supervision 虽然被“触发”了，但大多数时候没有形成有区分度的非零误差

所以这轮比较胜负，仍然主要看：

- `stage2 self_loss_tail`
- `stage1 c_t_var / hard_loop_var`

## 6. 关键观察

### 6.1 `iter5` 不值得扶正

这是这轮最明确的结论。

`iter5` 相比 `iter2`：

- `mixed` 更差：`0.0325 > 0.0263`
- `math` 更差：`0.0260 > 0.0199`
- `python_code` 更差：`0.0524 > 0.0378`
- `dialogue / emotion / persona_seed` 全部更差

也就是说：

- `iter5` 不是“更稳的那个”
- 至少在这次可复现定义里，它是全面退步

### 6.2 `iter2 + predictor_progress` 是最强候选，但不是无代价碾压

它的优点非常明显：

- `mixed` 最好：`0.0131`
- `math` 最好：`0.0151`
- `persona_seed` 也优于 `iter2`

但它也不是无代价：

- `dialogue` 变差：`0.4893 > 0.4258`
- `emotion` 变差：`0.1992 > 0.1530`
- `python_code` 也略差：`0.0557 > 0.0378`

所以它更像：

- 强推理 / 强 mixed 候选
- 但还不是“全桶都一起更好”的新基线

### 6.3 `iter5 + predictor_progress` 没把 `iter5` 救回来

它确实比裸 `iter5` 好一点：

- `mixed`: `0.0325 -> 0.0255`
- `persona_seed`: `0.9844 -> 0.8027`

但仍然没有超过：

- `iter2`
- 更不用说 `iter2 + predictor_progress`

所以这条线也不值得扶正。

## 7. 当前结论

### 结论 A：不要扶正 `iter5`

这轮证据已经很够了。

### 结论 B：如果要继续推进 loop-aware Self-JEPA，优先保留 `iter2 + predictor_progress`

因为它至少证明了一件很重要的事：

- 让 Self-JEPA predictor 知道 loop progress
- 在 `iter2` 这条稳基线之上，确实能继续压低 `mixed / math`

### 结论 C：但现在还不能直接把 `iter2 + predictor_progress` 扶正成总基线

原因是：

- `dialogue`
- `emotion`
- `python_code`

还没有一起守住。

也就是说，它现在更适合当：

- 下一阶段 Self-JEPA 强化候选线
- 而不是立刻替换正式长程基线

## 8. 建议

1. 保持正式长程基线仍为 `iter2`
2. 不扶正 `iter5`
3. 如果要继续推 Self-JEPA，下一步沿 `iter2 + predictor_progress` 做：
   - `progress-shape self JEPA`
   - `local self consistency`

## 9. 一句话总结

`iter5` 这次没有站住；真正值得继续追的是 `iter2 + predictor_progress`，但它现在还属于“高潜强化线”，还不是可以直接替换长程主线的版本。
