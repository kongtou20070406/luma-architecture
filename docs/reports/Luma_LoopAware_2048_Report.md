# Luma Loop-Aware Self-JEPA 2048-Step Report
## 1. 实验目的
这轮实验专门回答一个问题：让 `c_t / Self-JEPA` 显式知道“自己正处于第几次循环”，能不能在当前长程基线之上带来稳定收益。
共同底座：
- `full + depth2 + self_check`
- `self_check_k = 2`
- `one-step continuation gain` 主监督
- `light two-step auxiliary = 0.25`
- `rollout_steps = 10`
- `reason_loops = 15`
- `stage2_steps = 2048`
- buckets: `math / dialogue / emotion / persona_seed / python_code / mixed`
对比组：
- `none`: 当前基线，不加 loop-awareness
- `ct_progress`: 只让 `c_t` 知道 loop progress
- `predictor_progress`: 只让 Self-JEPA predictor 知道 loop progress
- `dual_phase`: 两边都知道，并额外给离散 loop phase embedding
## 2. 一个先讲清楚的事实
当前主模型注意力仍然是 **分组 KV 注意力（GQA）** 思路：主配置是 `num_attention_heads=12`、`num_key_value_heads=3`；`run_luma_stage12.py` 里的 tiny/short-run 配置会把这个比例缩成更小的测试值。
## 3. 顶层结果
| 配置 | stage1 ct_kl | hard_loop_var | c_t_var | mixed self_tail | math self_tail | python_code self_tail |
|---|---:|---:|---:|---:|---:|---:|
| none | 1.7689 | 0.8594 | 0.5618 | 0.0259 | 0.0242 | 0.0538 |
| ct_progress | 1.6906 | 0.6875 | 0.6290 | 0.0200 | 0.0200 | 0.0338 |
| predictor_progress | 1.7700 | 0.3594 | 0.5669 | 0.0182 | 0.0156 | 0.0320 |
| dual_phase | 2.6506 | 0.1875 | 0.0894 | 0.0236 | 0.0244 | 0.0691 |

## 4. 分桶 self-tail
| 配置 | mixed | math | dialogue | emotion | persona_seed | python_code |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.0259 | 0.0242 | 0.3447 | 0.1228 | 0.5283 | 0.0538 |
| ct_progress | 0.0200 | 0.0200 | 0.4434 | 0.0897 | 0.7695 | 0.0338 |
| predictor_progress | 0.0182 | 0.0156 | 0.2354 | 0.0833 | 0.2803 | 0.0320 |
| dual_phase | 0.0236 | 0.0244 | 0.3574 | 0.0906 | 0.3066 | 0.0691 |

## 5. 关键观察
### 5.1 `predictor_progress` 是这轮最值得保留的版本
- `mixed self_tail` 最低：`0.0182`
- `math self_tail` 最低：`0.0156`
- `python_code self_tail` 最低：`0.0320`
- `persona_seed` 也最好：`0.2803`
这说明：**让 Self-JEPA predictor 知道当前 loop progress，比直接让 `c_t` 知道，更像有效的最小改动。**
### 5.2 `ct_progress` 有局部收益，但偏科
- `emotion` 和 `python_code` 有提升
- 但 `dialogue` 明显变差
- `persona_seed` 退化最严重
所以它更像“让慢环主状态变得更会往前冲”，但不够平衡。
### 5.3 `dual_phase` 太重了
- `stage1 ct_kl` 最高：`2.6506`
- 但 `c_t_var` 被压到 `0.0894`
- `hard_loop_var` 也最低：`0.1875`
这说明两边都加 loop awareness 再叠离散 phase embedding，会把系统推得太硬，`c_t` 反而更接近塌缩边缘。
### 5.4 rollout_tail 全为 0，当前不能拿它分胜负
这轮四组在所有 bucket 上的 `rollout_tail` 都是 `0.0`。这不是“它们都完美学会了 rollout”，而是说明在 `2048-step` 口径下，这个指标已经失去分辨率。
所以这轮真正能用来比较的，主要是：
- `stage2 self_loss_tail`
- `stage1 hard_loop_var / c_t_var / ct_kl`
## 6. 当前结论
- 如果目标是“再加三个让 `c_t / Self-JEPA` 知道自己第几次循环的版本，然后选一个最值得继续追的”，答案是：**优先保留 `predictor_progress`**。
- 如果目标是“让 `c_t` 本体就强感知循环相位”，当前证据不支持直接这样做。
- 如果后面要继续推进 `progress-shape self JEPA`，最自然的下一步不是再加更重的 phase 结构，而是：
  - 在 `predictor_progress` 这条线上，加轻量 `next improvement / trend / plateau` 头
  - 保持 `c_t` 主状态本体尽量干净
## 7. 一句话总结
让 Self-JEPA predictor 知道“自己现在是第几轮”，是当前这三种 loop-awareness 里最聪明、最便宜、也最不伤整体平衡的一种。
