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

## 5. 当前 authoritative 口径（对齐当前工作区）

这一节开始，旧的“`token_selective / modulewise / lowrank` 仍在默认竞线中”的表述全部作废。

当前唯一有效、且与工作区实现对齐的动力学口径是：

- 当前动力学增强主候选：
  - `A2-progress_shape_v1-h3+progress_exit_readout`
- 当前观察锚点：
  - `A2-progress_shape_v1-h3`
- 当前实现版本不再直接送中长程：
  - `A2-progress_shape_v1-h3+token_selective_ct_routing`
  - `A2-progress_shape_v1-h3+lowrank_hyperbias_ct`
  - `A2-progress_shape_v1-h3+modulewise_ct_gate`

与当前工作区对应的程序入口：

- 固定程序：
  - `/home/kt/ai/minimind/luma_stage0/dynamics_autoresearch_program.json`
- 单候选 verifier：
  - `/home/kt/ai/minimind/scripts/run_dynamics_candidate_eval.py`
- 本地分级 runner：
  - `/home/kt/ai/minimind/scripts/run_dynamics_autoresearch_local.py`
- 矩阵生成脚本：
  - `/home/kt/ai/minimind/scripts/build_luma_dynamics_matrix.py`

与当前 runner 口径对应的赛果边界：

- `2048`：有效，可作为预筛依据
- `4096`：有效，可作为中程复筛依据
- `10240`：有效，但当前只有 `progress_exit_readout` 站住
- `20480`：当前不计正式结论，因为当时没有活着的训练进程
- `mid_rerun`：不能直接当结构失败证据，因为当时被 `run_luma_stage12.py` 的 `import math` bug 污染

## 6. 方法 Review：把三种新方案改成更符合 Luma 的实现

这里不沿用“直接强控 token”的旧实现口径，而是按 Luma 当前真实模块边界来改写：

- 慢环控制源：
  - `c_t`
  - `know_gap`
  - `progress-state(next improvement / trend / plateau)`
- 压缩区摘要源：
  - `block_repr`
  - `chunk summary`
  - `world_summary`
- 快环局部作用点：
  - `ReasonMamba input`
  - `DiffAttn residual bias`
  - `FFN residual scale`

### 6.1 `summary_conditioned_chunk_film`

这是当前最优先方案，也最符合 Luma 的慢环/快环分工。

Luma 版改写：

- `c_t` 不直接控 token
- 先构建：
  - `chunk_summary = pool(h, chunk=32/64)`
  - `block_context = fuse(last_k_block_repr, optional_world_summary)`
- 再用：
  - `summary_gate = MLP([c_t, progress_state, chunk_summary, block_context])`
- 输出：
  - `gamma_chunk`
  - `beta_chunk`
- 广播回 token 后，只做低增益调制：
  - `ReasonMamba input FiLM`
  - 或 `DiffAttn residual bias`

为什么更像 Luma：

- 慢环控制的是摘要，不是原始 token
- 直接复用 `CompressionZone -> block_repr -> ReasonLoop`
- 最容易稳定到 `10240`

论文启发落点：

- `Model-First Reasoning`
- `LeWorldModel`
- `EIDOS`
- `Geometrically-Regularized World Models`

### 6.2 `hierarchical_block_token_ct_routing`

Luma 版不做原始 token 级“全局常开”路由，而是两段式：

1. block 级筛选
- `block_gate = MLP([c_t, progress_state, block_repr, loop_embed])`
- 只保留 `top-k blocks`

2. block 内轻量 token gate
- `token_gate = sigmoid(MLP([h_local, c_t_proj, progress_proj]))`
- 但不直接做 `h = h * gate`
- 改成：
  - `attn bias` 调制
  - 或 `h = h + gate * delta_local`

为什么更像 Luma：

- 先用压缩区摘要缩小作用范围
- token 细控只在局部打开
- 更符合“先选区域，再局部精读”的推理方式

论文启发落点：

- `Causal-JEPA`
- `Latent Particle World Models`
- `Attention Residuals`

### 6.3 `progress_query_focus_routing`

Luma 版的关键不是“query from c_t”，而是“query from progress-aware slow state”。

实现口径：

- `focus_q = MLP([c_t, progress_next, progress_trend, progress_plateau, loop_embed])`
- 先匹配：
  - `block_repr`
  - `chunk summary`
- 再在选中块内做轻量 token top-k 或 span top-k
- 未选中的区域只保留弱全局 FiLM
- 被选中的区域才吃强局部 residual control

为什么更像 Luma：

- routing 成为 `progress-shape` 的下游执行器
- 不再另起一条和 Self-JEPA 竞争的控制线
- 对超长上下文更自然

论文启发落点：

- `From Latent Signals to Reflection Behavior`
- `Learning When to Stop`
- `Emergent Search and Backtracking`

## 7. 12 组实验矩阵（新基线展开）

实验矩阵全部挂在：

- `A2-progress_shape_v1-h3+progress_exit_readout`

上面，不再回到旧 `token_selective` 家族基底。

### 7.1 Summary-Conditioned Chunk FiLM 家族

1. `summary_chunk_film_v1`
- `c_t + block_repr + chunk_summary`
- 只调 `ReasonMamba input FiLM`

2. `summary_chunk_film_v2_world`
- 在 `v1` 上加入 `world_summary`
- 看慢环世界态是否帮助 chunk 选择

3. `summary_chunk_film_v3_progress`
- 在 `v1` 上加入 `progress-state`
- 看 plateau / trend 是否应该直接影响 chunk FiLM 强度

4. `summary_chunk_film_v4_dualsite`
- 弱 `Mamba input FiLM` + 弱 `DiffAttn residual bias`
- 看两点低增益调制是否优于单点

### 7.2 Hierarchical Block-Token Routing 家族

5. `hier_block_token_v1_block_only`
- 只有 block gate
- 不开 token gate
- 用来验证“先 block、再 token”是否真比旧全局 token routing 更稳

6. `hier_block_token_v2_attn_bias`
- block gate + block 内 token gate
- token gate 只调 `attn bias`

7. `hier_block_token_v3_residual_delta`
- block gate + token gate
- 被选区域走 `h = h + gate * delta_local`

8. `hier_block_token_v4_progress_rank`
- block 排名由 `c_t + progress-state` 决定
- 让 plateau / reversal 直接影响 `top-k` 范围

### 7.3 Progress-Query Focus Routing 家族

9. `progress_focus_v1_chunk_query`
- `focus_q` 先打 `chunk summary`
- 选中的 chunk 吃更强 summary-FiLM

10. `progress_focus_v2_block_then_token`
- 先打 `block_repr`
- 再在选中的 block 内做 token top-k

11. `progress_focus_v3_dense_sparse_hybrid`
- 全局保留弱 dense FiLM
- 选中 span 再加 sparse residual boost

12. `progress_focus_v4_backtrack`
- 加入 plateau / backtrack-aware top-k 调度
- 让卡住时 focus 变宽，而不是立刻塌成单点

## 8. 长程筛选赛制（至少到 10240）

赛制固定为：

- `cuda_smoke`
  - 单候选 CUDA smoke
  - 不计正式排名
- `2048`
  - 全 12 组预筛
- `4096`
  - 前 `6` 组复筛
- `10240`
  - 前 `4` 组长程首轮
- `20480`
  - 前 `2` 组确认

当前硬规则：

- 同一候选必须 checkpoint 晋级继训练
- 不允许各阶段 fresh start 混进正式排序
- 长程报告至少要明确写：
  - 哪些结果有效
  - 哪些因 stale runtime 不算
  - 哪些因实现 bug 污染不能直接盖棺

checkpoint lineage 固定写法：

- `4096 from 2048 checkpoint`
- `10240 from 4096 checkpoint`
- `20480 from 10240 checkpoint`

## 9. 排名与 guard

硬 guard：

- `math` 不掉
- `python_code` 不掉
- `mixed` 不崩
- `rollout_nonzero_ratio > 0`
- `c_t_var` 不塌

主排序指标：

1. `math self_tail`
2. `python_code self_tail`
3. `mixed self_tail`

副排序指标：

1. `rollout_nonzero_ratio`
2. `hard_loop_var`
3. `world_summary_drift_mean`
4. `dialogue / emotion` 是否守住

解释性诊断必须统一记录：

- `progress_shape_loss`
- `rollout_nonzero_ratio`
- `c_t curvature`
- `pred_delta_c` 邻接夹角
- `world_summary short-window drift`

## 10. 当前工作区中的实现边界

截至当前工作区，下面这些已经可直接跑：

- `A2-progress_shape_v1-h3+progress_exit_readout`
- `A2-progress_shape_v1-h3`
- `local_rollout_head`
- `dual_rate_self_predictor`
- `trajectory_health_probe`
- `backtrack_aware_progress`

下面这些是本轮 review 后要进入实现的新方案，而不是已经落地的现成候选：

- `summary_conditioned_chunk_film`
- `hierarchical_block_token_ct_routing`
- `progress_query_focus_routing`

因此当前动作顺序固定为：

1. 先按本节 review 实现三大家族最小版本
2. 每加一组 runner 参数先 `py_compile`
3. 每组先做单候选 CUDA smoke
4. 再进入 `2048 -> 4096 -> 10240`

## 11. 结论

当前这轮动力学 planning 的中心已经不是：

- 继续硬推旧 `token_selective`
- 或继续把旧 `modulewise / lowrank` 送中长程

而是：

- 以 `A2-progress_shape_v1-h3+progress_exit_readout` 为新基线
- 把 token-selective 家族改写成：
  - `summary-first`
  - `hierarchical`
  - `progress-driven`
- 再用 12 组矩阵做分级筛选，至少送到 `10240`

这才是当前与工作区、论文启发、以及你最新目标都一致的下一步。
