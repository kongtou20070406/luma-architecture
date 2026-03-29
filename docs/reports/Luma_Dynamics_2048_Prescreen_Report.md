# Luma Dynamics 2048 Prescreen Report

## 1. 这轮在做什么

这轮是 `Luma_Dynamics_Literature_Midcourse_Plan` 的第一阶段：

- `2048-step` 短程预筛
- 目标不是直接扶正新基线
- 而是先决定：
  - 哪些候选值得进入 `4096-step` 中程复筛
  - 哪些应该直接淘汰
  - 哪些只保留研究记录，不继续烧算力

共同底座：

- `A2-progress_shape_v1-h3`
- `full + depth2 + self_check`
- `self_check_k = 2`
- `one-step main + light two-step auxiliary`
- `self_loop_awareness_mode = predictor_progress`
- `rollout supervision horizon = 3`
- buckets: `math / dialogue / emotion / persona_seed / python_code / mixed`

## 2. 预筛结果总表

| 候选 | status | guard | score | 结论 |
|---|---|---:|---:|---|
| `A2-progress_shape_v1-h3+token_selective_ct_routing` | ok | pass | `0.06347` | 晋级中程 |
| `A2-progress_shape_v1-h3+lowrank_hyperbias_ct` | ok | pass | `0.06391` | 晋级中程 |
| `A2-progress_shape_v1-h3+modulewise_ct_gate` | ok | pass | `0.06813` | 晋级中程 |
| `A2-progress_shape_v1-h3+progress_exit_readout` | ok | pass | `0.07117` | 晋级中程 |
| `A2-progress_shape_v1-h3` | ok | pass | `0.08456` | 晋级中程（基线锚点） |
| `A2-progress_shape_v1-h3-softnear` | ok | pass | `0.08456` | 淘汰：与基线重合 |
| `A2-progress_shape_v1-h3-lite_local` | ok | pass | `0.09537` | 淘汰 |
| `A2-progress_shape_v1-h3+trajectory_health_probe` | ok | fail | `0.09777` | 不晋级：诊断专用且 guard 失败 |
| `A2-progress_shape_v1-h3+local_rollout_head` | ok | pass | `0.12843` | 不晋级 |
| `A2-progress_shape_v1-h3+structured_world_mask(light)` | ok | pass | `0.13696` | 不晋级 |
| `A2-progress_shape_v1-h3+dual_rate_self_predictor` | ok | pass | `0.13865` | 不晋级 |
| `A2-progress_shape_v1-h3+backtrack_aware_progress` | ok | fail | `0.09159` | 不晋级：guard 失败 |
| `A2-progress_shape_v1-h3+film_ct_modulation` | failed | fail | - | 淘汰：实现/数值问题待修 |

## 3. 为什么保留这 5 组进中程

### 3.1 `token_selective_ct_routing`

当前短程总分第一。

优点：
- `math / python_code / mixed` 组合最强
- 说明 token 级别的 `c_t` 精细路由确实可能有价值

风险：
- 这是高风险结构线
- 所以它必须进 `4096` 继续验证，不能只看短程就扶正

### 3.2 `lowrank_hyperbias_ct`

当前短程总分第二。

优点：
- 比 `token_selective` 更保守
- 说明“比 additive 更强，但比 token routing 更稳”的 `c_t` 调制方向很有潜力

### 3.3 `modulewise_ct_gate`

当前短程总分第三。

优点：
- 这是最符合架构直觉的一条 `c_t` 调制线
- 把 `c_t` 从全局 bias 提升为模块级重心控制器

### 3.4 `progress_exit_readout`

当前短程总分第四。

优点：
- 说明 `progress-shape` 不只是当辅助 loss 有用
- 它有机会直接成为 exit/continuation 的有效读出证据

### 3.5 `A2-progress_shape_v1-h3` 基线锚点

虽然它不是前四，但必须保留进中程。

原因：
- 它是当前最稳的 dynamics 候选基线
- 后面中程/长程必须有一个稳定锚点作参照
- 否则只看新结构之间互相比较，会丢掉“到底有没有比当前主线更好”这个判断基准

## 4. 为什么这些候选被筛掉

### 4.1 `softnear`

- 分数与 `A2-progress_shape_v1-h3` 完全重合
- 当前看不出独立增益
- 所以不值得继续占中程名额

### 4.2 `lite_local`

- 总分不差，但没有明显超过基线
- `python_code` 方向也没有给出足够强的新收益
- 当前不值得占中程名额

### 4.3 `local_rollout_head`

- 这是一个很有研究价值的结构想法
- 但短程结果没有站住，尤其综合分明显偏后
- 目前不进入中程

### 4.4 `dual_rate_self_predictor`

- `dialogue / emotion` 很漂亮
- 但 `python_code` 明显拖后腿
- 当前综合代价过大，不进中程

### 4.5 `structured_world_mask(light)`

- 继续印证了旧结论：
  - world 结构化增强现在仍然容易把整体压向不理想方向
- 短程分数靠后，不进中程

### 4.6 `trajectory_health_probe`

- 它的定位更像诊断伴随头
- 而不是性能竞争组
- 这轮还触发了 guard 失败，所以不进入中程

### 4.7 `backtrack_aware_progress`

- `rollout_nonzero_ratio` 直接掉光，guard 失败
- 说明当前接法太容易把动力学监督压平
- 不进入中程

### 4.8 `film_ct_modulation`

这轮不是单纯分数差，而是**实现/数值失败**：

- CUDA 端触发了 `bernoulli` 概率断言
- 说明当前 `FiLM` 版本会把 exit/sample 路径推到不健康区间

所以它当前状态是：
- 不是“表现差”
- 而是“还没到能参加筛选比赛的实现成熟度”

## 5. 中程名单

进入 `4096-step` 中程复筛的候选固定为：

1. `A2-progress_shape_v1-h3+token_selective_ct_routing`
2. `A2-progress_shape_v1-h3+lowrank_hyperbias_ct`
3. `A2-progress_shape_v1-h3+modulewise_ct_gate`
4. `A2-progress_shape_v1-h3+progress_exit_readout`
5. `A2-progress_shape_v1-h3`

## 6. 长程规则

后续严格按这条规则执行：

- `2048` 只决定谁进中程
- `4096` 只看中程结果排前 `3`
- `10240` 长程名额固定只有 `3` 个
- 不再给结构保留位额外插队进长程

## 7. 一句话总结

短程首筛的结论很清楚：

- 真正值得继续烧到中程的，是 `c_t` 调制和 progress 读出这两类结构线
- `token_selective / lowrank / modulewise / progress_exit_readout` 是当前最强四条竞争线
- `A2-progress_shape_v1-h3` 作为基线锚点一起进入中程
- 其余候选当前都不值得继续直接推进到中程
