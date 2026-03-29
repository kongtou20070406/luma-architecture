# Luma Dynamics Literature Review And Midcourse Experiment Plan

## 1. 这份文档是干什么的

这份文档回答三件事：

1. 最近与 `JEPA / world model / latent dynamics / explicit problem modeling` 相关的新论文，有哪些对 Luma 真有参考价值
2. 从“动力学”角度，Luma 现在最值得优化的地方是什么
3. 接下来该怎么做一组中程筛选实验，而不是盲目继续堆结构

这份文档的重点不是论文综述本身，而是：

- 哪些想法能迁移到 Luma
- 哪些不该直接硬搬
- 下一步实验怎么排优先级

补充说明：
- 虽然文件名里仍叫 `Midcourse Plan`，但当前内容已经明确扩展为：
  - 先短程/中程筛选
  - 再把前 `1 / 2 / 3` 名送去长程验证
- 当前统一口径：
  - `2048` = 短程预筛
  - `4096` = 中程复筛
  - `10240` = 长程起点
  - `20480` = 长程确认
- 也就是说，这份文档现在实际上承担的是：
  - `动力学候选的分级赛制设计`
  - 而不只是中程实验清单

## 2. 当前判断：Luma 现在最缺的不是“更远”，而是“更健康的局部动力学”

结合我们已经做过的实验，当前最关键的判断是：

- Luma 不缺“能不能再多想几步”的接口
- Luma 更缺“局部推进是不是稳定、是不是有分辨率、是不是知道自己还在有效推进”

也就是说，下一阶段不应该优先追：

- 更长 rollout
- 更远 value
- 更重 crystal / uncertainty

而应该优先追：

- 更好的局部动力学监督
- 更健康的近端 rollout
- 更明确的 progress-shape 表达
- 更好的状态几何与一致性

## 3. 参考论文与对 Luma 的迁移价值

### 3.1 直接相关的一手参考

#### LeWorldModel (2026)
- 链接：https://arxiv.org/abs/2603.19312
- 对 Luma 的直接价值：非常高

关键启发：
- `world JEPA` 应优先保持 `next-embedding prediction + 简洁正则`
- 不要把 world 分支重新做回重 reconstruction 系统
- loss 项应该尽量少，监控项尽量强

对 Luma 的动作建议：
- 保留 `full world JEPA`
- 继续偏向简洁损失，而不是不断加 world-side 小项
- 把 `surprise` 留作评估维度，而不是再变成主 loss

#### Causal-JEPA (2026)
- 链接：https://arxiv.org/abs/2602.11389
- 对 Luma 的迁移价值：中高

关键启发：
- world 建模不一定只能 patch/token 级 mask
- object/slot/structured masking 往往更有助于真正的可控 dynamics

对 Luma 的动作建议：
- 继续保留 `structured world mask` 这条研究支线
- 但先不要把它扶正进主线
- 更适合等 `A2-core` 附近的局部动力学先稳下来后再回头追

#### Flow Equivariant World Models (2026)
- 链接：https://arxiv.org/abs/2601.01075
- 对 Luma 的迁移价值：中

关键启发：
- 长时程稳定性往往来自“状态表示本身更守几何/对称结构”
- 不是单纯把 dynamics predictor 做大

对 Luma 的动作建议：
- 不直接照搬 flow equivariance
- 但可以把这个思想转译成：
  - `c_t` 的局部轨迹几何约束
  - world summary drift 的稳定性约束
  - 更细的 state-geometry diagnostics

#### GeoWorld (2026)
- 链接：https://arxiv.org/abs/2602.23058
- 对 Luma 的迁移价值：中

关键启发：
- latent world model 里加入明确几何结构，可能提升长程可规划性

对 Luma 的动作建议：
- 当前不建议直接上 hyperbolic / manifold trick
- 但可以吸收它的“geometry first”思路：
  - 先把 `c_t / world_summary` 的几何健康度诊断做全
  - 再决定是否值得加更强几何正则

#### Geometrically-Regularized World Models (2025)
- 链接：https://arxiv.org/abs/2510.26782
- 对 Luma 的迁移价值：高

关键启发：
- 改善 representation geometry 本身，就可能显著改善 world-model 稳定性
- 不一定非得把 dynamics module 做得更复杂

对 Luma 的动作建议：
- 这条和我们前面关于 `local consistency / curvature / progress-shape` 的经验很一致
- 但当前证据更支持：
  - 先走 `progress-shape`
  - 再回到更轻、更克制的局部几何约束

### 3.2 对语言推理更有启发的参考

#### Model-First Reasoning (2025)
- 链接：https://arxiv.org/abs/2512.14474
- 对 Luma 的迁移价值：高

关键启发：
- 真正复杂任务上，先构建问题状态模型，再求解，往往比盲目长 CoT 更稳

对 Luma 的动作建议：
- 这和你现在的 `c_t` / `world_summary` / `know_gap` 路线非常契合
- 说明我们后续应该更重视：
  - `c_t` 是否真的在表达当前问题状态
  - `progress-shape` 是否在表达“是否还在推进”
- 而不是只把退出当成单一阈值器

### 3.3 与“状态动力学/自反性”更直接相关的新参考

#### From Latent Signals to Reflection Behavior (2026)
- 链接：https://arxiv.org/abs/2602.01999
- 对 Luma 的迁移价值：高

关键启发：
- 自反/反思行为往往不是“突然出现的一整个模块”，而是沿着一条 latent activation trajectory 逐步显现。
- 更有价值的不是单独预测 “要不要反思”，而是追踪：
  - 当前状态是不是在推进
  - 推进趋势是不是在变平
  - 是否已经接近平台期

对 Luma 的动作建议：
- 强化 `progress-shape self JEPA`
- 让 `Self-JEPA predictor` 学：
  - `next improvement`
  - `trend`
  - `plateau`
- 把它作为 exit/continuation 的上游状态信号，而不是重新堆更远 horizon

#### Internalizing LLM Reasoning via Discovery and Replay of Latent Actions (2026)
- 链接：https://arxiv.org/abs/2602.04925
- 对 Luma 的迁移价值：中高

关键启发：
- 复杂推理可以被理解成一段段可重放的 latent actions，而不是只看最终输出 token。
- 动态控制若想稳定，需要能表达“当前 latent trajectory 正处于哪个操作片段”。

对 Luma 的动作建议：
- 当前不建议直接引入新的 latent action 大结构
- 但可以吸收为：
  - `predictor_progress`
  - 更清楚的短窗轨迹诊断
  - `c_t` / `pred_delta_c` 的操作片段感知

#### Latent Particle World Models (2026)
- 链接：https://arxiv.org/abs/2603.04553
- 对 Luma 的迁移价值：中

关键启发：
- 稳定动力学往往来自更可分解的 latent state，而不是单个“大而混”的状态向量。
- object-centric 在 Luma 里不需要原样照搬，但“把状态分成更明确的角色”很有参考价值。

对 Luma 的动作建议：
- 继续维持：
  - `c_t` 负责全局/慢环
  - `h` 负责快环
- `r_t` 作为局部递推 scratchpad 仍可研究，但暂不扶正
- 先把 `c_t` 的职责做清，而不是急着再加并行状态流

#### Latent-DARM / VDLM / LaDiR (2025-2026)
- 链接：
  - https://arxiv.org/abs/2603.09184
  - https://arxiv.org/abs/2602.15870
  - https://arxiv.org/abs/2510.04573
- 对 Luma 的迁移价值：中

关键启发：
- 允许“可修正的 latent reasoning”通常会提升复杂推理
- 但一旦 latent channel 太强、太自由，往往会牺牲语言稳定性或训练可控性

对 Luma 的动作建议：
- 不建议现在直接引入 diffusion-style latent reasoner
- 但它们支持我们当前路线：
  - 先把 `continuation gain`
  - `predictor_progress`
  - `progress-shape`
  这些轻量 latent dynamics 做稳

#### EIDOS: Latent-Space Predictive Learning for Time Series Foundation Models (2026)
- 链接：https://arxiv.org/abs/2602.14024
- 对 Luma 的迁移价值：中高

关键启发：
- 比起直接预测未来观测，先约束 latent state 的可预测演化，往往更能形成稳定、可组织的时序表征。
- 轻量 target aggregation branch 往往能显著提高 latent dynamics 的稳定性。

对 Luma 的动作建议：
- 继续坚持 `Self-JEPA / World-JEPA` 的 latent prediction 主线
- 可以考虑给 `progress-shape` 加一个更轻的 target aggregation，而不是继续拉长 rollout horizon

#### Emergent Search and Backtracking in Latent Reasoning Models (2026)
- 链接：https://arxiv.org/abs/2602.08100
- 对 Luma 的迁移价值：高

关键启发：
- latent reasoning 的健康轨迹通常不是单调直冲，而是：
  - exploration
  - tentative commitment
  - convergence 或 backtracking
- 这说明“平台期 / 停滞 / 回退”本身是重要状态，不该只被当成噪声。

对 Luma 的动作建议：
- 在 `progress-shape` 中显式保留：
  - plateau
  - trend reversal / backtrack-like signal
- exit policy 不应只学“越来越小的 delta”，还要学“是否已经进入无效回路”

#### Learning When to Stop: Adaptive Latent Reasoning via RL (2025)
- 链接：https://arxiv.org/abs/2511.21581
- 对 Luma 的迁移价值：中高

关键启发：
- 停止策略的优化，最有效的往往不是硬阈值，而是让模型学会“在维持性能的同时减少无效推理长度”。
- 自适应 stopping 的关键是 continuation value，而不是单独的 confidence。

对 Luma 的动作建议：
- 继续坚持：
  - `one-step main`
  - `light two-step aux`
- 后续若要继续做 exit policy，优先加：
  - continuation budget head
  - loop-efficiency diagnostics
而不是回到简单 confidence gate

## 4. 从动力学角度，Luma 应该怎么优化

### 4.1 第一优先：强化近端动力学，而不是更远 horizon

当前最合理的判断是：

- `horizon=3` 比 `10-step` 更像模型真的在用的区间
- 过远 rollout 现在更容易被压平，而不是更聪明

所以动力学优化优先顺序应该是：

1. 近端 rollout
2. progress-shape
3. 局部状态几何
4. 再考虑更远 horizon

### 4.2 第二优先：让 Self-JEPA 更会表达“推进形状”

当前最强的新线是：
- `A2-progress_shape_v1`

说明：
- 比起强行约束 `pred_delta_c` 的几何平滑
- 让模型显式表达：
  - 下一步 improvement
  - trend
  - plateau
  更符合 Luma 当前的主问题

### 4.3 第三优先：把几何约束做轻、做局部、做可诊断

局部一致性方向没有死，但当前实现太硬。

更合理的动力学几何方向不是：
- 大力上 curvature penalty

而是：
- 轻量局部平滑
- trajectory health diagnostics
- state drift vs task performance 的相关性监控

### 4.4 第四优先：把 surprise 留在评估，而不是急着再变成训练主角

`world surprise` 很有价值，但当前更适合做：
- 状态健康检查
- bucket 对比指标
- future retrieval / know-gap 触发诊断

不适合立刻再变成一个强训练项。

### 4.5 第五优先：优先上“小结构”，不要急着再开新状态流

基于目前所有实验，我更建议优先上：
- predictor 级别的结构增强
- exit readout 级别的结构增强
- compression 到 self lane 的轻量桥接

而不是：
- 直接扶正新的并行状态流
- 再拉长 rollout horizon
- 再堆一个更重的 world/self side tower

## 4.6 结构层面的可行改进候选

下面这些不是“马上全都上”的东西，而是值得进入长程筛选清单的结构候选。

### S1：Local Rollout Head
- 做法：
  - 保留当前 `Self-JEPA predictor`
  - 单独拆一个只负责 `t+1 / t+2 / t+3` 的近端 rollout head
  - 不再让远端 horizon 与近端共享同一条 dynamics 头
- 为什么值得试：
  - 当前很多问题不是“完全不会 rollout”
  - 而是“同一个 predictor 同时兼顾远近，最后两头都不够健康”

### S2：Progress-Conditioned Exit Readout
- 做法：
  - 不改 `c_t` 本体
  - 给 exit controller 加一个专门读取：
    - `next improvement`
    - `trend`
    - `plateau`
    的小读出层
- 为什么值得试：
  - 这能把 `progress-shape` 从辅助 loss，推进成真正的 continuation 决策证据
  - 同时比重新改 `c_t` 主状态更安全

### S3：Dual-Rate Self Predictor
- 做法：
  - 给 `Self-JEPA predictor` 加两条共享底座、不同时间尺度的小支路：
    - `fast local branch`
    - `slow stabilizer branch`
  - 最终只在 predictor 级融合，不新增新的大状态流
- 为什么值得试：
  - 你想要的是“局部推进 + 全局稳态”同时存在
  - 这比正式扶正 `r_t` 风险小很多

### S4：Compression-to-Self Skip Summary
- 做法：
  - 从压缩区 block summary 到 self lane 增加一个更轻的 residual skip
  - 只提供稳定摘要，不提供完整 world latent
- 为什么值得试：
  - 可以增强 `c_t` 对当前局部证据的 anchoring
  - 同时不至于把 self lane 重新做成 world lane 的影子

### S5：Trajectory Health Probe Head
- 做法：
  - 不改变主推理结构
  - 额外加一个只做诊断/轻辅助的 head，预测：
    - 当前 trajectory 是否平滑
    - 是否进入平台期
    - 是否出现无效回环
- 为什么值得试：
  - 很可能比继续堆远期 loss，更能帮助 exit policy 找到“健康停止”

### S6：Backtrack-Aware Progress Head
- 做法：
  - 在 `progress-shape` 上增加一个轻量 reversal / backtrack logits
  - 只做辅助，不直接改 token 预测
- 为什么值得试：
  - 最近 latent reasoning 论文显示，“错误后回退再收敛”是健康推理的一部分
  - Luma 现在缺的不是只会向前冲，而是知道什么时候应该重新整理当前状态

### S7：FiLM-Style c_t Modulation
- 做法：
  - 不再只把 `c_t` 投影后直接加到 `h`
  - 而是让 `c_t` 生成每个推理子模块的：
    - `scale`
    - `shift`
  - 以 FiLM 方式调制：
    - `mamba`
    - `diff_attn`
    - `ffn`
- 为什么值得试：
  - 当前 `c_t` 更像全局 additive bias，影响偏钝
  - FiLM 可以让 `c_t` 从“整体加偏置”升级成“按模块改变工作方式”
  - 很适合 Luma 当前“慢环指导快环”的定位

### S8：Module-Wise c_t Gate
- 做法：
  - 给每个推理子模块一个单独 gate
  - gate 由 `c_t + 当前 loop summary` 共同决定
  - 不是直接改 hidden，而是调：
    - 当前轮更该偏 `mamba`
    - 还是偏 `diff_attn`
    - 还是偏 `ffn`
- 为什么值得试：
  - 这比 FiLM 更克制
  - 更像“慢环分配本轮思考重心”
  - 对 exit/continuation 也更容易形成可解释的行为

### S9：Low-Rank Hyper-Bias c_t Injection
- 做法：
  - 保留当前 additive injection 的稳定性
  - 但把 `c_t` 生成的控制量拆成低秩：
    - `u(c_t) * v(hidden_summary)`
  - 让它不再是单一固定投影
- 为什么值得试：
  - 比 full hypernetwork 便宜很多
  - 比纯线性 bias 更灵活
  - 适合作为“在不大改结构下，提升 c_t 控制表达力”的中间路线

### S10：Token-Selective c_t Routing
- 做法：
  - 不再让 `c_t` 对整段 token 一起等强注入
  - 用 `c_t + token hidden` 算 token-wise gate
  - 只让部分 token 位更强吃到慢环控制
- 为什么值得试：
  - 当前 `c_t` 注入是全局均匀的，可能把局部推理也一起抹平
  - token-selective 路由更适合：
    - math 中间变量
    - code 关键 token
    - dialogue 里的局部语义焦点
- 风险：
  - 这是这组里最不稳的一条
  - 容易引入新的训练噪声，所以优先级应低于 FiLM 和 module-wise gate

## 5. 两阶段实验筛选计划：先短程/中程筛选，再送长程

这里的目标不是“一次找最终答案”，而是：

- 先用中程实验把最有价值的动力学改进筛出来
- 再把排名前 `1 / 2 / 3` 的方案送去长程复验
- 同时避免再把系统推向 rollout 压平

### 阶段 A：短程预筛（2048-step）

这一步的目标是：
- 快速剔除会把 system dynamics 推坏的方案
- 找到真正值得继续烧更长预算的结构/损失候选
- 给每个候选一个可比较的分数

#### 中程预筛候选池

最小候选池建议是：

1. `A2-progress_shape_v1-h3`
2. `A2-progress_shape_v1-h3-softnear`
3. `A2-progress_shape_v1-h3-lite_local`
4. `A2-progress_shape_v1-h3 + local_rollout_head`
5. `A2-progress_shape_v1-h3 + progress_exit_readout`
6. `A2-progress_shape_v1-h3 + dual_rate_self_predictor`
7. `A2-progress_shape_v1-h3 + trajectory_health_probe`
8. `A2-progress_shape_v1-h3 + backtrack_aware_progress`
9. `A2-progress_shape_v1-h3 + film_ct_modulation`
10. `A2-progress_shape_v1-h3 + modulewise_ct_gate`
11. `A2-progress_shape_v1-h3 + lowrank_hyperbias_ct`
12. `A2-progress_shape_v1-h3 + token_selective_ct_routing`

如果 GPU 时间更紧，可以先缩成：

1. `A2-progress_shape_v1-h3`
2. `A2-progress_shape_v1-h3 + local_rollout_head`
3. `A2-progress_shape_v1-h3 + progress_exit_readout`
4. `A2-progress_shape_v1-h3 + dual_rate_self_predictor`

#### 实验 A1：`A2-progress_shape_v1-h3`
- 作用：作为新的 dynamics 基线
- 目的：确认 `horizon3` 在比 `2048-step` 更长的口径下是否仍稳定
- 主要看：
  - `math / python_code / mixed self_tail`
  - `rollout_nonzero_ratio`
  - `c_t_var`
  - `hard_loop_var`

#### 实验 A2：`A2-progress_shape_v1-h3-softnear`
- 做法：
  - 保持 `horizon3`
  - 再加一个比当前 `near3_weighted` 更轻的近端加权
  - 例如：`1.0 / 0.3 / 0.1`
- 目的：
  - 看能不能在不伤 `dialogue / persona_seed` 的前提下，把 `rollout_nonzero_ratio` 再拉高一点
- 停止条件：
  - `dialogue` 或 `persona_seed` 相对 `A1` 恶化超过 `15%`
  - 但 `rollout_nonzero_ratio` 没有明显回升

#### 实验 A3：`A2-progress_shape_v1-h3-lite_local`
- 做法：
  - 在 `horizon3` 上加更轻的局部一致性
  - 比当前 `local_smooth` 更弱，例如把权重减半
- 目的：
  - 看局部几何约束在近端 rollout 下是否终于变成正收益
- 主要看：
  - `pred_delta_c` 相邻夹角分布
  - `trajectory curvature`
  - `math / python_code` 是否比 `A1` 更稳

### 阶段 B：中程复筛（4096-step）

这一步只给阶段 A 表现最好的前 `3~4` 组。

目标：
- 看中程里排序是否稳定
- 排除只在 2048-step 冒尖、但一拉长就压平的方案
- 为最终长程只保留前 `1 / 2 / 3` 名

### 排名规则（用于决定前 1/2/3）

中程筛选不建议只看单个 `mixed self_tail`。

建议采用下面这个综合排序原则：

#### 硬门槛
- `math` 不恶化
- `python_code` 不恶化
- `mixed` 不崩
- `rollout_nonzero_ratio` 不能继续掉成无分辨率
- `c_t_var` 不能明显塌缩

#### 主排序指标
1. `math self_tail`
2. `python_code self_tail`
3. `mixed self_tail`

#### 副排序指标
1. `rollout_nonzero_ratio`
2. `hard_loop_var`
3. `world_summary_drift_mean`
4. `dialogue / emotion` 是否仍在可接受区间

#### 推荐排序口径
- 第 1 名：
  - 兼顾 `math + python_code + mixed`
  - 同时动力学健康指标不塌
- 第 2 名：
  - 综合次优，但在某一方向有明确结构性优势
- 第 3 名：
  - 不是最优，但具备明显研究价值，值得看长程是否反超

### 阶段 C：动态诊断层

这一步不是独立竞争组，而是给阶段 A/B 的候选统一加诊断。

目的：
- 判断性能改善到底来自“更健康的推进”，还是来自“更快压平”
- 给最终长程报告提供解释，而不只是给分数

#### 实验 B1：trajectory health probe
- 新增诊断：
  - `c_t` 短窗 curvature
  - `pred_delta_c` 相邻夹角分布
  - `world_summary` 短窗 drift 分布
- 目的：
  - 判断性能改善到底来自“更健康的推进”，还是来自“更快压平”
- 成功标准：
  - 改善组相比基线，`math / python_code` 更好时，不能伴随：
    - `c_t_var` 明显塌缩
    - `world_summary_drift` 异常归零
    - `rollout_nonzero_ratio` 大幅掉光

#### 实验 B2：progress vs rollout correlation
- 做法：
  - 记录每个 bucket 的：
    - `progress_shape_loss`
    - `rollout_nonzero_ratio`
    - `self_tail`
- 目的：
  - 判断 progress-shape 和 rollout 分辨率是否在同向工作
- 解释原则：
  - 如果 `progress_shape_loss` 明显下降，但 `rollout_nonzero_ratio` 继续归零，同时 bucket 性能也不升，那更像“状态被压硬”而不是“动力学更健康”

### 阶段 D：只保留一条高风险研究线

#### 实验 C1：structured world mask revisit
- 前提：A 阶段稳定后再做
- 做法：
  - 在 `A2-progress_shape_v1-h3` 上重新试更轻的 `structured world mask`
- 目的：
  - 验证 structured world 是不是只有在近端 dynamics 稳了之后才会发力

### 阶段 D：长程结构候选筛选

这一阶段只在 `A2-progress_shape_v1-h3` 已经站住时才开。

#### 实验 D1：`A2-progress_shape_v1-h3 + local_rollout_head`
- 目的：
  - 看近端专用 rollout head 能不能恢复 `rollout_nonzero_ratio`
  - 同时守住 `math / python_code`

#### 实验 D2：`A2-progress_shape_v1-h3 + progress_exit_readout`
- 目的：
  - 看 progress-shape 是否真的能变成 exit 证据，而不是只做辅助 loss

#### 实验 D3：`A2-progress_shape_v1-h3 + dual_rate_self_predictor`
- 目的：
  - 用 predictor 级双时间尺度，替代更重的并行状态流实验

#### 实验 D4：`A2-progress_shape_v1-h3 + trajectory_health_probe`
- 目的：
  - 先做“强诊断、弱侵入”的结构探针
  - 判断性能变化是否来自更健康轨迹

#### 实验 D5：`A2-progress_shape_v1-h3 + backtrack_aware_progress`
- 目的：
  - 验证 plateau/reversal/backtrack 这类状态是否对长程 reasoning 更关键

#### 实验 D6：`A2-progress_shape_v1-h3 + film_ct_modulation`
- 目的：
  - 验证 `c_t` 改为 FiLM 调制后，是否能提升慢环对快环的精细控制

#### 实验 D7：`A2-progress_shape_v1-h3 + modulewise_ct_gate`
- 目的：
  - 验证 `c_t` 是否更适合做“本轮思考重心分配器”，而不是单纯 bias

#### 实验 D8：`A2-progress_shape_v1-h3 + lowrank_hyperbias_ct`
- 目的：
  - 在不引入大结构的前提下，测试 `c_t` 控制表达力能否继续提升

#### 实验 D9：`A2-progress_shape_v1-h3 + token_selective_ct_routing`
- 目的：
  - 验证慢环控制是否应该更局部化，而不是全序列平均注入

## 5.1 分级执行顺序与预算

如果后面要用分级实验真正筛选，而不是继续短程碰运气，我建议：

1. 跑阶段 A：`2048-step` 全候选短程预筛
2. 选前 `3~4` 名进入阶段 B：`4096-step` 中程复筛
3. 统一补阶段 C diagnostics
4. 再决定是否给高风险线 `structured world mask` 一个名额
5. 最终只保留前 `1 / 2 / 3` 名进入长程
6. `structured world mask` 站住后，再开 D 阶段结构筛选

一个 practical 的分级筛选批次可以定成：
- `2048-step`：预筛
- `4096-step`：中程复筛
- `10240-step`：长程首轮，只给排名前 `1 / 2 / 3` 的组
- `20480-step`：长程确认，只给 `10240-step` 后的前 `1~2` 名

这样能避免把 GPU 时间浪费在高风险、低分辨率的配置上。

## 5.2 长程送测规则（10240 / 20480-step）

长程不是继续海选，而是：
- 验证中程排名前 `1 / 2 / 3` 的方案是否仍然成立
- 看中程排序在更长预算下会不会反转

### 长程送测建议

- `10240-step`
  - 第 1 名：必须送
  - 第 2 名：默认送
  - 第 3 名：建议送

- `20480-step`
  - 只送 `10240-step` 之后最强的前 `1~2` 名
  - 作用不是继续海选，而是确认谁值得真正扶正

也就是说：
- 长程第一轮固定跑 `3` 组最合适
- 长程第二轮固定跑 `1~2` 组最合适
- 这和你前面已经验证过的夜间批跑节奏也一致

### 长程报告应该回答的三个问题

1. 中程第 1 名在 `10240-step` 长程里是否继续领先？
2. 中程第 2 / 第 3 名是否出现 `10240-step` 反超？
3. 在 `20480-step` 之后，哪个方案最适合扶正为新的正式 dynamics baseline？

## 6. 建议的执行顺序

最推荐的顺序是：

1. 阶段 A：`2048-step` 全候选预筛
2. 阶段 B：前 `3~4` 名做 `4096-step`
3. 阶段 C：统一加 diagnostics
4. 按综合规则排出前 `1 / 2 / 3`
5. 把前 `1 / 2 / 3` 送去 `10240-step` 长程首轮
6. 再把 `10240-step` 最强的前 `1~2` 名送去 `20480-step` 长程确认
7. `20480-step` 结束后再决定是否扶正新基线

## 6.1 动力学候选执行表

下面这个表的目标是：
- 让后续执行时不必再从整篇文字里反推实验顺序
- 一眼看出谁该先跑、谁只适合进复筛、谁值得送长程

| 候选名 | 2048 预筛 | 4096 复筛 | 10240 长程 | 20480 确认 | 预期强项 | 主要风险 |
|---|---|---|---|---|---|---|
| `A2-progress_shape_v1-h3` | 必跑 | 必跑 | 高概率送 | 视 10240 排名而定 | 当前最稳的 dynamics 候选；近端 rollout 更有分辨率 | 可能只是“最好的一条现有线”，但未必是最终最强结构 |
| `A2-progress_shape_v1-h3-softnear` | 必跑 | 视 2048 结果而定 | 有条件送 | 低概率送 | 有机会提高 `rollout_nonzero_ratio`，同时保留近端 supervision | 容易重新伤 `dialogue / persona_seed` |
| `A2-progress_shape_v1-h3-lite_local` | 必跑 | 视 2048 结果而定 | 低概率送 | 低概率送 | 若局部几何终于变成正收益，可能增强 `math / python_code` 稳定性 | 很容易再次变成“几何好看但 bucket 变差” |
| `A2-progress_shape_v1-h3 + local_rollout_head` | 必跑 | 高优先级复筛 | 高潜长程候选 | 高潜确认候选 | 最可能恢复 rollout 的真实分辨率，同时不必拉长 horizon | 新 head 可能把 dynamics 头拆得过散，导致 mixed 不稳 |
| `A2-progress_shape_v1-h3 + progress_exit_readout` | 必跑 | 高优先级复筛 | 高潜长程候选 | 高潜确认候选 | 有机会把 progress-shape 从辅助 loss 变成真正 exit 证据 | 可能把退出策略重新压硬，导致 loops 分布收紧 |
| `A2-progress_shape_v1-h3 + dual_rate_self_predictor` | 必跑 | 高优先级复筛 | 高潜长程候选 | 高潜确认候选 | 用 predictor 级双时间尺度替代重状态流，最符合当前 Luma 风格 | 两条 predictor 支路若耦合不好，可能引入新的不稳定源 |
| `A2-progress_shape_v1-h3 + trajectory_health_probe` | 必跑 | 必跑 | 作为解释性 companion 而非主竞争组 | 不送 | 提供最强诊断价值，能帮助解释为什么某组表现更好 | 自身不一定直接带来性能提升 |
| `A2-progress_shape_v1-h3 + backtrack_aware_progress` | 建议跑 | 视 2048 结果而定 | 研究性送测 | 低概率送 | 若 plateau/reversal 真有价值，可能改善复杂 reasoning 的“卡住后回退”能力 | 容易把 progress 头做得过重，伤 dialogue / emotion |
| `A2-progress_shape_v1-h3 + film_ct_modulation` | 建议跑 | 高优先级复筛 | 高潜长程候选 | 高潜确认候选 | 让 `c_t` 从全局 bias 升级成更细粒度的模块调制 | 若 scale/shift 过强，可能把快环推硬 |
| `A2-progress_shape_v1-h3 + modulewise_ct_gate` | 建议跑 | 高优先级复筛 | 高潜长程候选 | 高潜确认候选 | 最有机会把 `c_t` 变成“本轮思考重心分配器” | gate 若过早饱和，可能导致模块使用塌缩 |
| `A2-progress_shape_v1-h3 + lowrank_hyperbias_ct` | 建议跑 | 视 2048 结果而定 | 有条件送 | 低概率送 | 在保持稳定的前提下提升 `c_t` 控制表达力 | 可能收益不明显，落成复杂但不值的中间态 |
| `A2-progress_shape_v1-h3 + token_selective_ct_routing` | 研究性跑 | 视 2048 结果而定 | 低概率送 | 低概率送 | 若成功，最可能改善 math/code 的局部推理聚焦 | 风险最高，最容易引入训练噪声与不稳定 |
| `A2-progress_shape_v1-h3 + structured_world_mask(light)` | 只在前面稳定后再跑 | 有条件复筛 | 有条件送 | 低概率送 | 若局部动力学先稳住，structured world 可能终于发力 | 历史上很容易伴随 math/rollout 压平问题 |

### 推荐的最小可执行批次

如果只想先跑一个“够有信息量，但不太夸张”的批次，我建议：

1. `A2-progress_shape_v1-h3`
2. `A2-progress_shape_v1-h3 + local_rollout_head`
3. `A2-progress_shape_v1-h3 + progress_exit_readout`
4. `A2-progress_shape_v1-h3 + dual_rate_self_predictor`
5. `A2-progress_shape_v1-h3 + trajectory_health_probe`
6. `A2-progress_shape_v1-h3 + modulewise_ct_gate`
7. `A2-progress_shape_v1-h3 + film_ct_modulation`

这 7 组已经足够回答：
- 近端 rollout head 是否值得扶正
- progress-shape 是否真能进入 exit 决策
- predictor 级双时间尺度是否比新增状态流更靠谱
- `c_t` 的调制方式是否该从 additive bias 升级
- 改善是来自真实健康轨迹，还是只是指标压平

### 当前实现状态

截至 `2026-03-29`，这批候选里已经进入代码实现、可直接被固定程序调用的有：

- `A2-progress_shape_v1-h3`
- `A2-progress_shape_v1-h3 + local_rollout_head`
- `A2-progress_shape_v1-h3 + progress_exit_readout`
- `A2-progress_shape_v1-h3 + dual_rate_self_predictor`
- `A2-progress_shape_v1-h3 + trajectory_health_probe`
- `A2-progress_shape_v1-h3 + modulewise_ct_gate`
- `A2-progress_shape_v1-h3 + film_ct_modulation`
- `A2-progress_shape_v1-h3 + backtrack_aware_progress`
- `A2-progress_shape_v1-h3 + lowrank_hyperbias_ct`
- `A2-progress_shape_v1-h3 + token_selective_ct_routing`

当前固定程序与本地 runner：

- 固定程序：`/home/kt/ai/minimind/luma_stage0/dynamics_autoresearch_program.json`
- 单候选 verifier：`/home/kt/ai/minimind/scripts/run_dynamics_candidate_eval.py`
- 纯本地 watchdog runner：`/home/kt/ai/minimind/scripts/run_dynamics_autoresearch_local.py`

补充边界：

- 这轮已用 `A2-progress_shape_v1-h3 + modulewise_ct_gate` 做过最小 CUDA smoke，候选名到开关映射链路已打通。
- 当前 `mamba/triton` 路径不支持用 CPU 做同构 smoke；候选冒烟应使用 CUDA。

### 推荐的长程送测逻辑

完成 `2048 -> 4096` 之后，长程送测规则固定为：

1. 只看 `4096` 中程结果
2. 严格按中程综合排名选前 `3` 名
3. `10240` 长程名额固定只有 `3` 个，不再额外加结构保留位

也就是说：
- `2048` 负责筛进中程
- `4096` 负责决定谁进长程
- `10240` 不再接受“分数没进前三，但结构上很有意思”的额外插队

## 7. 当前我最推荐保留的主线

如果现在必须给一个“动力学方向正式候选”，我会建议：

- `A2-progress_shape_v1 + horizon3`

原因：
- 它最符合当前实验证据
- 它不像更远 rollout 那样容易压平
- 它比更重 local consistency 更平衡
- 它也比 crystal / uncertainty 更稳定

## 8. 按当前进度对主规划的推进建议

基于目前已经完成的：
- `A2-core` 长程站住
- `A2-predictor_progress` 显示出真实潜力
- `A2-progress_shape_v1` 比局部一致性更值得继续追
- `horizon3` 比长 rollout 更有信息量

我建议主规划向前推进到下面这个表述：

### 当前正式长程基线
- `A2-core`

### 当前动力学强化主候选
- `A2-progress_shape_v1-h3`

### 当前不再优先的方向
- 继续拉长 rollout horizon
- uncertainty 直接调 two-step 权重
- 更重的 local consistency penalty
- 新增更重并行状态流

### 当前值得进入长程筛选的结构候选
- `local_rollout_head`
- `progress_exit_readout`
- `dual_rate_self_predictor`
- `trajectory_health_probe`
- `backtrack_aware_progress`
- `film_ct_modulation`
- `modulewise_ct_gate`
- `lowrank_hyperbias_ct`
- `token_selective_ct_routing`

## 9. 一句话总结

最近的动力学/world-model 论文给 Luma 最有价值的不是“再做更远”，而是：

- 让状态表示更健康
- 让近端 dynamics 更有分辨率
- 让 Self-JEPA 更会表达推进形状

所以接下来的中程实验，最该押的是：

- `A2-progress_shape_v1 + horizon3`
- 再在它上面做更轻、更克制的 dynamics 改进

## 10. 2026-03-29 当前赛果回写

截至当前，真正可计入结论的结果应按下面口径读取：

- `2048` 短程预筛：已完成，可作为候选晋级依据。
- `4096` 中程复筛：已完成，可作为长程入围依据。
- `10240` 长程首轮：当前只有 `A2-progress_shape_v1-h3+progress_exit_readout` 站住。
- `20480` 长程确认：当前不计入正式结论，因为已无活跃训练进程，仅残留旧 runtime 文件。

当前最可信的赛果是：
- `A2-progress_shape_v1-h3+progress_exit_readout`
  - `4096` 通过
  - `10240` 通过
  - 当前应视为动力学增强第一候选
- `A2-progress_shape_v1-h3`
  - `4096` 通过
  - `10240` 失败
  - 当前应视为“观察保留的基线锚点”
- `token_selective_ct_routing / lowrank_hyperbias_ct / modulewise_ct_gate`
  - 原始 `4096` 均失败
  - 当前实现版本不再继续直接送中长程

补充说明：
- 后续 `mid_rerun` 链暴露出 runner 侧补丁 bug：
  - `run_luma_stage12.py` 缺少 `import math`
- 因此该补测链不能拿来直接判结构生死，只能说明复测流程被实现 bug 污染。

## 11. token-selective 家族的更稳后继方案

基于当前 `token_selective_ct_routing` 在 `4096` 仍然太猛、太不稳，下一轮不应继续沿“直接对 token 做强选择性控制”的旧实现直推，而应优先改成下面三种更贴 Luma 的结构。

### 11.1 `summary_conditioned_chunk_film`
- 核心：`c_t` 不直接控 token，而是先控 `chunk summary / block_repr / world_summary`，再把控制量广播回 token。
- 建议实现：
  - `chunk_summary = pool(h, chunk=32/64)`
  - `gamma, beta = MLP([c_t, chunk_summary, progress_state])`
  - `h_chunk = h_chunk * (1 + gamma) + beta`
- 对 Luma 的好处：
  - 最稳定
  - 最符合“慢环控摘要，快环做局部细化”的当前架构
  - 最适合后续超长上下文扩展

### 11.2 `hierarchical_block_token_ct_routing`
- 核心：先 block-level gate，再在被选中 block 内做轻量 token gate。
- 建议实现：
  - 复用压缩区 `block_repr`
  - `block_gate = MLP([c_t, block_repr, loop_embed])`
  - 仅对 top-k block 内计算 token gate
  - 优先调 `attn bias / mamba gate / residual scale`，而不是直接整段乘法
- 对 Luma 的好处：
  - 比直接 token-selective 温和
  - 更容易稳定进入 `2048 -> 4096`

### 11.3 `progress_query_focus_routing`
- 核心：由 `c_t + progress-shape(next improvement / trend / plateau)` 共同生成 focus query，再做稀疏区域选择。
- 建议实现：
  - `focus_q = MLP([c_t, progress_state, loop_embed])`
  - 先匹配 `block_repr / chunk summary`
  - 再在选中块内做 token top-k
- 对 Luma 的好处：
  - 让 routing 直接成为 `progress-shape` 的下游执行器
  - 不是另起一条与 Self-JEPA 竞争的控制流

当前优先级建议：
1. `summary_conditioned_chunk_film`
2. `hierarchical_block_token_ct_routing`
3. `progress_query_focus_routing`

## 12. 长程赛制建议：改成晋级继训练制

后续建议不再让：
- `2048`
- `4096`
- `10240`
- `20480`
都从头 fresh start。

建议固定改成：
- `2048` 跑完保存 checkpoint
- `4096` 从对应 `2048` checkpoint 继续
- `10240` 从 `4096` 胜出 checkpoint 继续
- `20480` 从 `10240` 胜出 checkpoint 继续

这样做的理由：
- 省掉大量重复前期训练时间
- 更贴近“候选先短程站住，再看能否自然延伸到中长程”的真实筛选逻辑
- 比每一档重新从 0 开始更适合我们当前动力学实验

执行约束：
- 仅允许在同一候选、同一配置、同一 seed、同一数据桶口径下晋级继训练
- 报告必须显式写清：
  - `4096 from 2048 checkpoint`
  - `10240 from 4096 checkpoint`
  - `20480 from 10240 checkpoint`
