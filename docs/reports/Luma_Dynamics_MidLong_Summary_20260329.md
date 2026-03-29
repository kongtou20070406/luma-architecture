# Luma Dynamics Mid/Long Summary (2026-03-29)

## 1. 这份总结回答什么

这份总结统一回答四件事：

1. 当前这轮 `2048 -> 4096 -> 10240/20480` 动力学筛选，到底哪些结果算数
2. 哪些候选应该保留，哪些应该毙掉，哪些应当暂时观察
3. 你新提出的三种 `token_selective_ct_routing` 改良方案，结合 Luma 现有模块后该怎么改写
4. 后续长程是否应该直接在前一阶段 checkpoint 上继训练，以节约时间

## 2. 先把当前有效结果说清楚

### 2.1 已完成且有效的结果

短程 `2048` 首筛已完成：
- 报告：[Luma_Dynamics_2048_Prescreen_Report.md](/home/kt/ai/docs/reports/Luma_Dynamics_2048_Prescreen_Report.md)

中程/长程当前有效结果如下：
- `A2-progress_shape_v1-h3+progress_exit_readout`
  - `4096`：通过，`score = 0.04049072265625`
  - `10240`：通过，`score = 0.05256729125976563`
- `A2-progress_shape_v1-h3`
  - `4096`：通过，`score = 0.046511840820312504`
  - `10240`：失败
- `A2-progress_shape_v1-h3+token_selective_ct_routing`
  - 原始 `4096`：失败
- `A2-progress_shape_v1-h3+lowrank_hyperbias_ct`
  - 原始 `4096`：失败
- `A2-progress_shape_v1-h3+modulewise_ct_gate`
  - 原始 `4096`：失败

### 2.2 什么结果不算数

有两类结果现在不能直接拿来当最终判决：

1. `20480` confirm
- 当前没有活着的训练进程
- 只剩旧的 runtime/heartbeat 文件残留
- 因此这轮 `20480` 不计入正式结论

2. `mid_rerun` 补测链
- 目录：`autoresearch_dynamics_mid_rerun_20260329`
- 三条补测都显示 `failed:1`
- 但日志显示：它们都被同一个实现 bug 污染了：
  - `run_luma_stage12.py` 中 `stage2_validate` 缺少 `import math`
  - 导致在 `finite_delta_norms = [value for value in delta_norms if math.isfinite(value)]` 处直接 `NameError`
- 所以这条补测链不能当“结构复验失败”直接盖棺

一句话：
- **原始 `4096/10240` 结果算数**
- **`20480` 不算**
- **补测链仅能说明“复测流程被 bug 污染”，不能拿来直接判结构死刑**

## 3. 当前最可信的工程结论

### 3.1 当前最值得保留的动力学主候选

保留：
- `A2-progress_shape_v1-h3+progress_exit_readout`

原因：
- 它是当前唯一同时通过：
  - `4096`
  - `10240`
- 中程结果优于基线：
  - `math_self_tail = 0.03173828125`
  - `mixed_self_tail = 0.02783203125`
  - `dialogue_self_tail = 0.033447265625`
- 长程也站住：
  - `math_self_tail = 0.00921630859375`
  - `mixed_self_tail = 0.012420654296875`
- 说明把 `progress-shape` 直接读进 exit/readout，这条路线确实比“只在 Self-JEPA 内表达推进形状”更赚钱

### 3.2 当前基线怎么处理

观察保留：
- `A2-progress_shape_v1-h3`

原因：
- 它在 `4096` 通过，说明仍是可用锚点
- 但 `10240` 在 bucket probe 阶段又掉回了旧问题：
  - `torch.bernoulli(sampled_exit_score...)`
  - CUDA assert
- 这更像：
  - 基线本身在更长步数下的数值边界仍然明显
  - 不是彻底没价值

### 3.3 当前应当暂时毙掉的结构线

当前形态下建议先毙掉：
- `A2-progress_shape_v1-h3+token_selective_ct_routing`
- `A2-progress_shape_v1-h3+lowrank_hyperbias_ct`
- `A2-progress_shape_v1-h3+modulewise_ct_gate`

注意，这里的“毙掉”含义是：
- **当前实现版本不再直接送中长程**
- 不是永远放弃这个思想来源

原因：
- 它们在原始 `4096` 都是同类失败：
  - stage2 之后进入 bucket probe
  - 在 sampled exit 上数值不稳定
  - 触发 `bernoulli` 概率非法/NaN 路径
- 说明当前这类“更强 c_t 调制/更细 token 路由”在 Luma 现有慢环里还是太猛了

## 4. keep / kill / observe

### keep
- `A2-progress_shape_v1-h3+progress_exit_readout`

### observe
- `A2-progress_shape_v1-h3`
- `mid_rerun` 这条链本身不算结构证据，但它暴露了 runner/评估脚本里仍有补丁级 bug 需要先清

### kill（当前实现版本）
- `A2-progress_shape_v1-h3+token_selective_ct_routing`
- `A2-progress_shape_v1-h3+lowrank_hyperbias_ct`
- `A2-progress_shape_v1-h3+modulewise_ct_gate`

## 5. 把你新想的三个融合方案改成更贴 Luma 的实现案

你的三个方案方向是对的，但如果直接照“强 token-selective routing”去做，还是很容易重演现在的数值不稳。对 Luma 更合适的改写如下。

### 5.1 `hierarchical_block_token_ct_routing`
原方案来源：`Span-Selective Hierarchical Routing`

更贴 Luma 的实现：
- 不从原始 token 序列直接硬切 block 开始
- 先复用 Luma 已有的：
  - `compression block repr`
  - `world_summary`
  - `UnifiedAttnRes` 的跨区摘要
- 让 `c_t + know_gap + loop_progress` 先生成 **block-level gate**
- 只对 top-k block 对应的 token 区间启用轻量 token gate
- token gate 不直接乘到 `h` 全值，而是优先调：
  - `attn bias`
  - `mamba gate`
  - `ffn residual scale`

建议实现形式：
- `block_gate = MLP([c_t, block_repr, loop_embed])`
- `token_gate = sigmoid(MLP([h_local, c_t_proj]))` 只在选中 block 内计算
- 最终不做纯乘法 `h = h * gate`
- 而做更稳的残差式调制：
  - `h = h + gate * delta_local`

为什么更像 Luma：
- 它优先复用已有压缩区摘要
- 减少对全 token 细粒度 gate 的直接冲击
- 更适合慢环控制“先选区域，再细化”

### 5.2 `progress_query_focus_routing`
原方案来源：`Query-Triggered Token Focus`

更贴 Luma 的实现：
- query 不只来自 `c_t`
- 让 query 由：
  - `c_t`
  - `predicted next improvement`
  - `trend`
  - `plateau`
  一起生成
- 也就是让当前最强的 `progress-shape` 主线直接参与“该看哪里”的决定
- 匹配对象不要一开始就全 token K
- 先匹配：
  - `block_repr`
  - `chunk summary`
  - 再在被选块内做 token top-k

建议实现形式：
- `focus_q = MLP([c_t, progress_state, loop_embed])`
- `scores_block = focus_q @ block_keys`
- `top_blocks -> local token topk`
- 未选中 token 只吃轻量全局 `c_t` additive / FiLM
- 选中 token 才吃强控制

为什么更像 Luma：
- 它把 token routing 变成 progress-shape 的下游执行器
- 不是另起一条和 Self-JEPA 竞争的控制线
- 对 long context 更友好

### 5.3 `summary_conditioned_chunk_film`
原方案来源：`Summary-Conditioned Local Routing`

更贴 Luma 的实现：
- 这是我最看好的一个
- 因为它最符合 Luma 现在“慢环控摘要、快环做局部细化”的架构气质
- 不让 `c_t` 直接控 token
- 而是让 `c_t` 先控：
  - `chunk summary`
  - `block_repr`
  - `world_summary`
- 再由 summary 对 token 层做 **chunk-wise FiLM / bias modulation**

建议实现形式：
- `chunk_summary = pool(h, chunk=32 or 64)`
- `summary_gate = MLP([c_t, chunk_summary, progress_state])`
- 输出：`gamma_chunk, beta_chunk`
- 再广播回该 chunk：
  - `h_chunk = h_chunk * (1 + gamma_chunk) + beta_chunk`
- 也可只调：
  - `Mamba3 input gate`
  - `DiffAttn residual bias`

为什么更像 Luma：
- 最稳定
- 最适合超长上下文
- 与现有 `CompressionZone -> block_repr -> ReasonLoop` 的设计天然兼容
- 本质上是在“摘要层”做慢环控制，而不是让 `c_t` 直接下场控每个 token

## 6. 这三个方案我给的优先级

如果按下一轮值得实现和送 `2048 -> 4096` 的优先级排：

1. `summary_conditioned_chunk_film`
2. `hierarchical_block_token_ct_routing`
3. `progress_query_focus_routing`

理由：
- 第 1 个最稳，最贴 Luma，最适合把 `token_selective` 的想法软化成“摘要驱动局部控制”
- 第 2 个兼顾表达力和稳定性，适合做第一版真正的 `token_selective` 替代者
- 第 3 个很有想象力，但训练初期更容易抖，需要 warmup / 混合 dense 策略

## 7. 长程要不要直接在短程 checkpoint 基础上继训练

我的判断：**要，而且后面应该默认这么做。**

### 推荐策略
- `2048` 跑完后，保存该候选 checkpoint
- `4096` 不再从头训，而是从对应 `2048` checkpoint 接着训
- `10240` 再从 `4096` 的胜出 checkpoint 接着训
- `20480` 再从 `10240` checkpoint 接着训

### 这么做的收益
1. 明显省时
- 不是理论上的一点点
- 对我们这种 staged screening，通常就是省掉大约一半以上重复前期训练时间

2. 更贴近真实 curriculum
- 候选先在短程稳定下来
- 再看它能不能把这个稳定性带进更长阶段
- 这比每一档都 fresh start 更贴近“晋级赛”逻辑

3. 更容易比较动力学是否可持续
- 我们真正关心的是：
  - 短程好看的东西
  - 能不能自然延伸到中长程
- 继训练比重头跑更直接回答这个问题

### 必须加的约束
- 只允许在**同一候选、同一超参、同一数据桶口径**下继训练
- 每次晋级必须记录：
  - 来源 checkpoint
  - 训练步数区间
  - seed
  - config hash
- 报告里必须明确写：
  - `4096 from 2048 checkpoint`
  - `10240 from 4096 checkpoint`
- 如果要做公平 AB（fresh start vs warm-start）再单独做，不要混进主筛选赛制里

### 一句话
- 后面的长程筛选，我建议默认改成：
  - **晋级继训练制**
- 而不是每一档都重新从 0 开始

## 8. 当前建议的下一轮执行顺序

### 先保留的主线
- `A2-progress_shape_v1-h3+progress_exit_readout`

### 下一轮新结构候选
1. `A2-progress_shape_v1-h3 + summary_conditioned_chunk_film`
2. `A2-progress_shape_v1-h3 + hierarchical_block_token_ct_routing`
3. `A2-progress_shape_v1-h3 + progress_query_focus_routing`

### 赛制建议
- `2048`：三条都跑
- `4096`：保留前 2
- `10240`：只送前 1~2
- 全部采用 checkpoint 晋级继训练

## 9. 一句话总结

当前这轮真正站住的动力学增强线是：
- `A2-progress_shape_v1-h3+progress_exit_readout`

`token_selective` 这批“更强 c_t 细粒度调制”在现版本下还是太猛，但你的三个新方案是对的，只是要改成更符合 Luma 的“摘要优先、分层控制、progress 驱动”的版本。
