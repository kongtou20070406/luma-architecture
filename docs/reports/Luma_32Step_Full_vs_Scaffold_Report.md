# Luma 32-Step Full vs Scaffold Report

## 1. 这次实验比较什么

这次实验把短程训练时长从此前常用的 `16` 步进一步拉长到 `32` 步，目标是检查：

- 在更长的短程训练里，`full world JEPA` 会不会开始兑现它的动力学优势
- `scaffold + self_check` 是否仍然保持综合更稳
- 在更复杂的 `competition_math + dialogue + emotion` 混合任务上，两者排序是否发生变化

统一配置：

- `rollout_steps = 10`
- `reason_loops = 15`
- `slow_k = 1`
- `enable_self_check_ring = True`
- `stage2_steps = 32`
- `fixture_mode = competition_math_dialogue_emotion`

结果文件：

- `scaffold`:
  [stage12_competition_emotion_scaffold_selfcheck_10x15_futurevalue_32.json](/home/kt/ai/minimind/artifacts/stage12_competition_emotion_scaffold_selfcheck_10x15_futurevalue_32.json)
- `full`:
  [stage12_competition_emotion_full_selfcheck_10x15_futurevalue_32.json](/home/kt/ai/minimind/artifacts/stage12_competition_emotion_full_selfcheck_10x15_futurevalue_32.json)

---

## 2. 总结结论

结论先说：

- 在这组 `32-step` 更长短程实验里，`scaffold + self_check` 仍然是综合更稳的版本。
- `full + self_check` 没有在整体上反超。
- 但 `full` 在 `emotion` 桶上出现了更明显的 rollout 改善信号，说明它的优势更可能在“更长训练 + 更细粒度世界态约束”下慢慢体现，而不是在所有桶里立刻压过 `scaffold`。

---

## 3. 整体指标对比

| Variant | hard_loop_var | soft_loop_var | mean_kl | self_loss_tail | self_rollout_tail |
|---|---:|---:|---:|---:|---:|
| scaffold + self_check | 3.6875 | 0.1875 | 1.6243 | 0.5117 | 0.4785 |
| full + self_check | 3.6875 | 0.1875 | 1.5997 | 0.5742 | 0.7246 |

解读：

- 两者在 loop 使用分布上几乎一样，说明这次差异主要不在 exit 分布本身。
- `scaffold` 的 `self_loss_tail` 更低。
- `scaffold` 的 `self_rollout_tail` 明显更低。
- 所以在整体 mixed 桶里，`scaffold` 还是更稳。

---

## 4. 分任务观察

### 4.1 math

| Variant | hard_loop_var | self_loss_tail | self_rollout_tail |
|---|---:|---:|---:|
| scaffold | 2.1875 | 0.8457 | 0.7793 |
| full | 1.1875 | 0.7324 | 0.7656 |

解读：

- `full` 在 math 桶的一阶 self loss 更低。
- rollout 尾部两者接近，`full` 略好一点。
- 但 `full` 的 `hard_loop_var` 更低，说明它没有把更深推理分布拉得更开。

### 4.2 dialogue

| Variant | hard_loop_var | self_loss_tail | self_rollout_tail |
|---|---:|---:|---:|
| scaffold | 7.25 | 0.8887 | 0.6211 |
| full | 3.5 | 0.7793 | 0.7109 |

解读：

- `scaffold` 在 dialogue 上明显更会使用更深 loops。
- `full` 的一阶 self loss 更低，但 rollout 尾部更差。
- 这更像 `full` 在对话分布上学得更保守，而不是更会展开动力学。

### 4.3 emotion

| Variant | hard_loop_var | self_loss_tail | self_rollout_tail |
|---|---:|---:|---:|
| scaffold | 0.25 | 0.6602 | 0.5801 |
| full | 0.25 | 0.6152 | 0.4238 |

解读：

- 这是本次最值得注意的桶。
- `full` 在 emotion 上同时赢了：
  - 更低的 `self_loss_tail`
  - 更低的 `self_rollout_tail`
- 这说明更完整的 world JEPA 可能对“情境状态 / 情绪上下文”这类更整体性的 latent world structure 更敏感。

---

## 5. 当前工程判断

### 5.1 默认主干仍不扶正 full

理由：

- mixed 总体上 `scaffold` 更稳
- dialogue 桶 `scaffold` 仍明显更强
- 当前正式默认仍应保持：
  - `scaffold + self_check`

### 5.2 full 的价值没有消失

这次实验反而给了一个更具体的信号：

- `full` 不一定先在“综合短程 mixed 指标”上压过 `scaffold`
- 但它在 `emotion` 桶已经显示出更完整 world modeling 的优势

所以更合理的路线不是“删掉 full”，而是：

- 继续保留 `full` 作为专项分支
- 下一步优先在以下场景继续复验：
  - 更长短程训练
  - 更强 emotion / support / multi-turn context 桶
  - 更长 rollout 与更难 latent consistency 验证

---

## 6. 一句话结论

把短程训练长度翻倍到 `32` 步后，`scaffold + self_check` 仍然是当前综合更稳的默认底座；
但 `full + self_check` 在 `emotion` 桶上的明显改善说明，它的优势更可能是“任务选择性兑现”，而不是立刻在所有混合指标上统一反超。
