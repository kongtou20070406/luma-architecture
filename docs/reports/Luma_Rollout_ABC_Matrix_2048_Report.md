# Luma Rollout ABC Matrix Report (2048-step)

## 1. 这次实验在比较什么

这轮实验沿着当前最值得继续追的 `A2-progress_shape_v1`，专门测试三种修 rollout 分辨率的办法：

1. 方案 A：把 rollout horizon 缩短
2. 方案 B：只保留近端 rollout 监督
3. 方案 C：在 self flow 里加入轻度连续 span mask

共同底座：

- `A2-progress_shape_v1`
- `full + depth2 + self_check`
- `self_check_k = 2`
- `one-step main + light two-step auxiliary`
- `self_loop_awareness_mode = predictor_progress`
- `self_progress_shape_weight = 0.10`
- `self_progress_trend_weight = 0.05`
- `self_progress_plateau_weight = 0.02`
- `stage2_steps = 2048`
- buckets: `math / dialogue / emotion / persona_seed / python_code / mixed`

对比组：

- `baseline_progress_full_2048`
- `horizon3_2048`
- `horizon4_2048`
- `near3_weighted_2048`
- `self_span_mask_2048`

## 1.5 每个实验项具体怎么实现

### `baseline_progress_full_2048`

就是当前的 `A2-progress_shape_v1` 原样运行：

- `rollout_steps = 10`
- `self_rollout_supervision_horizon = 0`
- `self_rollout_weighting_mode = legacy`
- `self_feature_span_mask_ratio = 0.0`

### `horizon3_2048`

这是方案 A 的最强收缩版本。

实现项：
- `rollout_steps = 3`
- `self_rollout_supervision_horizon = 3`

含义：
- 直接把 rollout 监督收回近端
- 不再让更远 horizon 参与主监督

### `horizon4_2048`

这是方案 A 的较温和版本。

实现项：
- `rollout_steps = 4`
- `self_rollout_supervision_horizon = 4`

含义：
- 保留一点点更远的 rollout
- 但仍远短于当前默认的 `10`

### `near3_weighted_2048`

这是方案 B。

实现项：
- `rollout_steps = 10`
- `self_rollout_weighting_mode = near3`

近端权重：
- `t+1 / horizon=2`: `1.0`
- `t+2 / horizon=3`: `0.5`
- `t+3 / horizon=4`: `0.2`
- 更远：`0.0`

含义：
- 保留“有 rollout”这个结构
- 但把主监督重心拉回近端有信息区间

### `self_span_mask_2048`

这是方案 C。

实现项：
- `self_feature_span_mask_ratio = 0.10`

含义：
- 在 `SelfJEPAResidualPredictor` 的输入特征上施加轻度连续 span mask
- 不是 world branch 的 mask
- 是在自流 predictor 这一侧做局部缺失鲁棒性训练

## 2. 顶层结果

| 配置 | mixed self_tail | mixed rollout_tail | mixed rollout_nonzero_ratio |
|---|---:|---:|---:|
| baseline_progress_full_2048 | 0.0225 | 0.0000 | 0.0000 |
| horizon3_2048 | 0.0164 | 0.0000 | 0.0000 |
| horizon4_2048 | 0.0183 | 0.0000 | 0.0000 |
| near3_weighted_2048 | 0.0182 | 0.0000 | 0.0000 |
| self_span_mask_2048 | 0.0225 | 0.0000 | 0.0000 |

补充：训练主线 `stage2 mixed` 的总体 `rollout_nonzero_ratio`：

| 配置 | stage2 mixed nonzero ratio |
|---|---:|
| baseline_progress_full_2048 | 0.0068 |
| horizon3_2048 | 0.0098 |
| horizon4_2048 | 0.0093 |
| near3_weighted_2048 | 0.0195 |
| self_span_mask_2048 | 0.0068 |

## 3. 分桶 self-tail

| 配置 | mixed | math | dialogue | emotion | persona_seed | python_code |
|---|---:|---:|---:|---:|---:|---:|
| baseline_progress_full_2048 | 0.0225 | 0.0208 | 0.3223 | 0.1082 | 0.7900 | 0.0510 |
| horizon3_2048 | 0.0164 | 0.0130 | 0.2339 | 0.1338 | 0.4824 | 0.0918 |
| horizon4_2048 | 0.0183 | 0.0236 | 0.9512 | 0.8015 | 1.5742 | 0.0392 |
| near3_weighted_2048 | 0.0182 | 0.0199 | 0.8066 | 0.1450 | 1.0781 | 0.0771 |
| self_span_mask_2048 | 0.0225 | 0.0208 | 0.3223 | 0.1082 | 0.7900 | 0.0510 |

## 4. rollout 分辨率观察

### 4.1 `horizon3` 最像把监督拉回模型真正在用的区间

它的特点是：

- `mixed self_tail` 最好：`0.0164`
- `math self_tail` 最好：`0.0130`
- `dialogue self_tail` 也更好：`0.2339`
- `persona_seed` 甚至明显回收：`0.4824`

同时：

- `stage2 mixed rollout_nonzero_ratio`
  - `0.0068 -> 0.0098`

这说明：

- 缩 horizon 没有把 rollout 真正“救活”成高分辨率指标
- 但它确实把监督拉回了更有用的近端区间
- 从 bucket 表现看，这一刀是有效的

代价：

- `emotion` 轻微回退：`0.1082 -> 0.1338`
- `python_code` 回退较明显：`0.0510 -> 0.0918`

所以它不是无代价碾压，而是：
- 更偏 reasoning / dialogue / persona 回收
- 但可能伤一点 code 稳定性

### 4.2 `near3_weighted` 确实把 nonzero ratio 拉高了，但 bucket 代价更大

它是这轮里 `stage2 mixed rollout_nonzero_ratio` 最好的：

- `0.0195`

而且 bucket 上也确实重新出现了非零 rollout：

- `dialogue rollout_tail = 0.3770`
- `persona_seed rollout_tail = 0.2161`

这说明：

- 方案 B 在“让 rollout 再次有信号”这件事上是有效的

但问题也很明显：

- `dialogue self_tail = 0.8066`
- `persona_seed self_tail = 1.0781`

也就是说：

- 它让 rollout 更有分辨率了
- 但这个代价太大
- 目前还不值得扶正成默认

### 4.3 `horizon4` 太松，收益不够，代价很大

`horizon4` 是这轮最不理想的收缩版之一：

- `dialogue = 0.9512`
- `emotion = 0.8015`
- `persona_seed = 1.5742`

说明：

- 只缩到 `4` 还不够像“回到近端”
- 但又已经打破了当前基线的平衡

所以这版不值得继续追。

### 4.4 `self_span_mask` 这版几乎没有产生可测效果

这轮里：

- `self_span_mask_2048`
- `baseline_progress_full_2048`

结果几乎完全相同。

这更像说明：

- 当前这版轻度 feature span mask 太弱
- 或者它在当前 predictor 结构里的插入点还不对

所以现在不能说方案 C 无价值，但至少这版接法没有兑现收益。

## 5. 当前结论

### 结论 A：方案 A 里，优先保留 `horizon3`

如果要从这轮里挑一条最值得继续追的 rollout 修复线，答案是：

- `horizon3_2048`

原因：

- `mixed / math / dialogue / persona_seed` 形状最好
- 比 baseline 更像“把监督拉回模型真正在用的范围”

### 结论 B：方案 B 更像诊断工具，而不是新默认

`near3_weighted` 很有价值，因为它证明了：

- rollout 分辨率是可以被重新拉回来的

但它当前代价太大，所以更适合：

- 做 diagnostic / research branch
- 还不适合作为新基线

### 结论 C：方案 C 当前这版不保留

`self-flow span mask` 这版没有给出可测收益，当前不建议继续沿这版参数直接追。

## 6. 建议

下一步更合理的方向是：

1. 在 `A2-progress_shape_v1` 上优先保留 `horizon3`
2. 如果还想继续挖 rollout 分辨率，再在 `horizon3` 底座上做更克制的近端加权，而不是直接上 `near3_weighted`
3. 暂时不要继续在当前强度下追 `self-flow span mask`

## 7. 一句话总结

这轮矩阵支持：

- 真正最值钱的第一刀是把 rollout 监督收回近端，尤其是 `horizon=3`
- 近端加权虽然能把 rollout 非零占比拉回来，但当前代价太高
- `self-flow span mask` 这版几乎没起作用
