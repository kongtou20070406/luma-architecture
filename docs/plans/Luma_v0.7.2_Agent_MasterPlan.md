# Luma v0.7.2 迁移总规划（Agent 执行版，最终一次性上线预训练）

## 0. 文档定位与硬性原则

### 0.1 目标
- 本文档用于后续 agent 直接执行，不是讨论稿。
- 目标是完成 Luma v0.7.2 在 `minimind` 主工程中的迁移与落地。
- `parameter-golf` 仅用于小规模机制验证，不承载最终成品质量目标。

### 0.2 硬性原则（必须遵守）
- 最终成品训练 = **一次性全量预训练**（唯一正式 run）。
- 前期所有阶段仅用于降低最终一次性训练失败率。
- 前期允许短程测试、模块验证、小范围扫描；不允许多次全量重跑。
- Gate 未通过不得进入下一阶段。

### 0.3 代码基地定位
- 主实现底座：`minimind`（模型结构、训练主流程、日志与监控）。
- 快速验证场：`parameter-golf`（小模块机制验证与诊断脚本）。

---

## 1. 固定架构与训练决策（不可随意改动）

### 1.1 架构固定项
- `c_t` 从“观察附属”升级为“并行状态流”，且 **不是** `self_repr`。
- 自省流升级为 2 层（保持轻量）。
- 维度固定：`meta_dim=96`，`c_t_dim=64`。
- 双时间尺度：
  - `h` 快环：每步更新。
  - `c_t/world_summary` 慢环：每 `k=2` 步更新（阶段内固定，不动态改）。
- 自省流输入扩展为：`h + 压缩摘要`，并预留 `world_summary` 接口。

### 1.2 训练固定项
- Self JEPA：预测残差 `Δc_t = c_{t+1} - c_t`。
- World JEPA：`h` 上的 masked latent prediction（不做 raw hidden reconstruction）。
- Rollout：固定 2-step 辅助损失。
- 总损失（最终 run）固定形式：
  - `L = L_lm + α*L_world + β*L_self + γ*L_rollout + λ*L_residual_reg`

### 1.2A 当前短程验证默认底座（2026-03-28 冻结）
- 当前阶段1/2/短程阶段3默认实验底座：
  - `world_jepa_mode = scaffold`
  - `enable_self_check_ring = True`
  - `slow_k = 1`
  - `rollout_steps = 4`
  - `reason_loops = 8`
- 选择理由：
  - 在当前 tiny + short-run + `hard_math_dialogue` / `hard_math_dialogue_emotion` 验证中，
    `scaffold + self_check` 的综合表现最稳，兼顾：
    - 更低的 `self_rollout_tail`
    - 更高的 `hard_loop_var`
    - 更低的短程平均 loss
- 解释边界：
- 该默认值仅冻结为“当前机制验证底座”，不等价于最终一次性全量预训练的永久结构冻结。
- `LeWorldModel-style full world JEPA` 保留为重点专项分支，不视为已淘汰。

### 1.2D 产品定位导向的正式预训练优先候选（2026-03-28 新增）
- 结合产品定位：
  - Luma 的目标是“聪明的聊天伙伴”，而不是“冰冷的纯任务助手”
  - 因此正式预训练主干的优先候选，默认转向：
    - `world_jepa_mode = full`
    - `enable_self_check_ring = True`
- 理由：
  - `full world JEPA` 更贴近“完整世界态 / 情境态 latent 建模”
  - `self_check` 能增强退出控制对内部状态的读取
  - 这两者更符合：
    - 自我意识
    - 自我感知
    - 情感与情境一致性
- 约束说明：
  - 这条是“正式预训练优先候选”而不是“当前短程验证默认底座”
  - 当前短程/中程 mixed 综合最优仍可能暂时落在 `scaffold + self_check`
  - 但若后续中程验证不出现明显反证，正式 run 应优先围绕 `full + self_check` 收敛配置
  - 当前最新 `128-step`、`competition_math_dialogue_emotion + persona_seed` 分桶验证表明：
    - `full + self_check` 在 `reason_shared_depth=1` 下整体更稳
    - `reason_shared_depth=2` 对 `emotion` 桶的循环深度分布更强，但 mixed / persona_seed 综合并未反超
  - 因此当前正式 run 候选的更具体冻结方式为：
    - `full + self_check`
    - 共享推理 block 默认仍先保留 `reason_shared_depth=1`
    - `reason_shared_depth=2` 保留为情感/高表达专项候选，不直接扶正
  - 若产品定位进一步明确优先级为：
    - 情感一致性
    - 聊天伙伴感
    - persona seed 的自我感知
    则可以允许实验底座临时切换为：
    - `full + depth2 + self_check`
    - 但它当前仍应被视为“情感优先实验底座”，不是 mixed 最稳默认值
  - 当前优先验证补充：
    - 若实验底座切换到 `full + depth2 + self_check`
    - 则 `self_check_k = 2` 作为当前优先候选
    - 其优先级高于继续把 `reason_loops` 从 `15` 往上加

### 1.2C 当前真实参数量口径（2026-03-28 校正）
- 当前默认代码口径下，Luma 仍属于 `0.3B` 档，但真实参数量应以代码实测为准，而不是旧估算。
- 当前默认 `scaffold` 实测：
  - `266,739,390` 参数
  - 约 `266.739M`
- 当前默认 `full world JEPA` 实测：
  - `269,542,206` 参数
  - 约 `269.542M`
- 结论：
  - 当前 Luma 仍可称为“0.3B 级”
  - 但更准确的工程口径应写为：
    - `scaffold`: `266.7M`
    - `full`: `269.5M`

### 1.2B 当前长程专项默认候选（2026-03-28 新增）
- 当前长程专项候选配置：
  - `world_jepa_mode = scaffold`
  - `enable_self_check_ring = True`
  - `slow_k = 1`
  - `rollout_steps = 10`
  - `reason_loops = 15`
- 选择理由：
  - 在当前 tiny + `hard_math_dialogue` 验证中：
    - `hard_loop_var = 7.0`
    - `self_rollout_tail = 0.6738`
  - 说明更长 horizon 的动力学监督仍在持续产生收益。
- 当前结论：
  - `10x15` 已优于更短的 `5x10` 作为长程专项候选。
  - `10x20` 暂未显示额外收益，说明当前瓶颈更像 exit policy，而不是最大 loops 不足。

### 1.3 风险控制固定项
- 必测风险：
  - 伪稳定器（pseudo stabilizer）
  - `c_t` 影子化（退化为 `h` 的影子）
  - 上下文漂移（误检索/误记忆强化）
  - 过强平滑（residual regularization 过大）
- 必备诊断：停滞检测
  - 条件：连续 `k` 步 `delta_h < threshold` 且任务未完成。
  - 结果：必须报警并记录样本与上下文。

### 1.4 压缩区（Compress + Distill）固定配置（v0.7 并入 v0.7.2）
- 压缩区仅执行一次，不循环；输出 `8` 个 `block_repr` 给推理区跨区回看。
- 层组结构固定为 `5:1`，`4` 组共 `24` 层：
  - `Layer A x5`：`ZC-RMSNorm -> Mamba3(d_model=768, d_state=192, MIMO开启, post-gate RMSNorm保留) -> residual`
  - `Layer B x1`：`ZC-RMSNorm -> (KDA or SWA)` + `SwiGLU FFN(ffn_dim=3072)` + `Sandwich Norm` + `residual`
- 精确检索层交替策略：
  - 奇数组：`KDA(无FoX)`
  - 偶数组：`SWA(window=1024)`
- `block_repr` 抽样规则固定：
  - 每 `3` 层保存一次表示，共 `8` 个，传给推理区时 `detach`。
- 压缩区 Mamba3 约束：
  - 不跨循环传递 `initial_state`（每次 forward 独立）。
  - 保留复数状态与 MIMO 路径能力；阶段0/1 可先用 SISO 验证脚手架稳定性，再切到 MIMO 训练路径。

### 1.5 推理区（Reason + Loop）固定配置（v0.7 并入 v0.7.2）
- 推理区权重共享，循环 `1~8` 次，默认包含以下步骤：
  1. `Mamba3(initial_state跨循环，BF16 state)`
  2. `GatedDiffAttn + FoX + SWA(1024)`（局部精读 + 遗忘 + 差分消噪）
  3. `SwiGLU FFN(ffn_dim=3072)`
  4. `UnifiedAttnRes`（历史循环 + 压缩区 block repr 双源聚合）
  5. 自省流（轻量 Mamba，`meta_dim=96`，`c_t_dim=64`，慢环 `k=2`）
  6. 双流 JEPA（主流 world / 自省 self）
  7. 退出控制（`jepa_error + delta_h`）
  8. `know_gap` 触发 Router 检索 Tape 并回注
- 推理区数值约束：
  - 跨循环 `mamba_state` 与自省 `meta_state` 必须 BF16，不得 FP8。
  - 停滞检测必须启用（连续低 `delta_h` 且任务未完成）。
- 当前短程验证默认值（2026-03-28）：
  - `slow_k=1`
  - `rollout_steps=4`
  - `reason_loops=8`
- 说明：
  - 这组值来自短程实验，不直接宣布为最终预训练永久默认。
  - 若进入更长训练与更大样本复验，优先继续测试：
    - `rollout_steps=5`
    - `reason_loops=10`
  - 当前长程专项默认候选已更新为：
    - `rollout_steps=10`
    - `reason_loops=15`
  - 正式预训练阶段不建议把“单层小循环”误当成最终形态：
    - 当前共享循环块本身已经包含 `Mamba3 + 局部精读 + FFN + UnifiedAttnRes + introspection`
    - 即“单次 loop”不是单一线性层，而是一个小型推理 block
    - 但若正式 run 前中程验证显示表达能力不足，可优先考虑“扩宽共享循环块表达能力”，而不是先无条件增加循环外独立层数

---

## 2. 技术来源论文与落地映射

> 原则：优先复用前人已验证方向，减少无效大规模消融。

### 2.0 论文来源索引（已检索）

| 技术点 | 论文/来源 | 链接 | 在本项目中的作用 |
|---|---|---|---|
| Self JEPA 表征预测范式 | Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA, 2023) | https://arxiv.org/abs/2301.08243 | 支持“在表示空间而非像素空间做预测”的核心思路 |
| World JEPA（视频/时空 latent 预测） | Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA, 2024) | https://arxiv.org/abs/2404.08471 | 支持 masked latent prediction、非像素重建路线 |
| 2-step rollout 与 action-conditioned world model | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (2025) | https://arxiv.org/abs/2506.09985 | 支持“短 rollout + action-conditioned 后续建模（2-AC）” |
| LLM 中引入 JEPA 目标 | LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures (2025) | https://arxiv.org/abs/2509.14252 | 支持在 LLM 训练中叠加 JEPA 风格 embedding-space 目标 |
| Depth 连接可学习化（mHC 上游） | Hyper-Connections (2024/2025) | https://arxiv.org/abs/2409.19606 | 解释 mHC 之前的可学习残差连接基础 |
| DeepSeek mHC（你提到的 mHC） | mHC: Manifold-Constrained Hyper-Connections (2025/2026) | https://arxiv.org/abs/2512.24880 | 提供“可学习残差连接在大规模训练中稳定化”的直接证据 |
| Attention Residuals（你点名） | Attention Residuals (Kimi Team, 2026, technical report) | https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf | 提供 Block AttnRes / depth-wise residual mixing 的工程化方案 |
| Mamba-3（你点名） | Mamba-3: Improved Sequence Modeling using State Space Principles (2026) | https://arxiv.org/abs/2603.15569 | 支持状态空间主干升级与长序列建模改进 |
| 多时间尺度快慢环启发 | Hierarchical Reasoning Model (HRM, 2025) | https://arxiv.org/abs/2506.21734 | 支持“高层慢、低层快”的分频更新原则 |
| 分布偏移（DAgger） | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (2011) | https://proceedings.mlr.press/v15/ross11a.html | 支持“在展开后分布上训练”而非只在一步分布上优化 |
| 误差累积（Compounding Errors） | Efficient Reductions for Imitation Learning (2010) | https://proceedings.mlr.press/v9/ross10a.html | 支持“误差随时间累积，应控制 rollout 长度” |
| 世界模型总体方向（位置论文） | A Path Towards Autonomous Machine Intelligence (LeCun, 2022) | https://openreview.net/pdf/315d43ba26f55357a84cec9a7ed15a6610094f79.pdf | 提供 JEPA/世界模型/分层规划的总框架 |
| LeWorldModel（你重点关注） | LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels (2026) | https://arxiv.org/abs/2603.19312 | 提供“端到端稳定 world JEPA + 极简双损失”的直接证据 |
| Qwen 主线技术报告（大厂补充） | Qwen3 Technical Report (2025) | https://arxiv.org/abs/2505.09388 | 提供 Qwen 系列公开技术报告基线（架构与训练细节） |
| Qwen3.5 官方发布报告（大厂补充） | Qwen3.5: Towards Native Multimodal Agents (Alibaba Cloud, 2026) | https://www.alibabacloud.com/blog/602894 | 提供 Qwen3.5 官方架构/吞吐/能力说明（非 arXiv） |
| Kimi 最新官方完整报告（大厂补充） | Kimi-K2.5 Full Report (MoonshotAI, 2026) | https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf | 提供 Kimi 最新公开完整技术报告入口 |
| Opus 系统卡（大厂补充） | Anthropic Model System Cards (Opus 系列) | https://www.anthropic.com/system-cards | 提供 Opus 系列能力与安全评估的一手系统卡入口 |

### 2.1 检索状态说明（截至 2026-03-28）

- `LeWorldModel` 一手来源已确认：
  - arXiv: https://arxiv.org/abs/2603.19312
  - 项目页: https://le-wm.github.io/
  - 代码入口: 项目页 `Code` 链接（GitHub）
  - 数据与权重入口: 项目页 `Data & Checkpoints`（Hugging Face）

- `Attention Residuals` 一手来源已确认：
  - 官方仓库: https://github.com/MoonshotAI/Attention-Residuals
  - 官方 technical report PDF: https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf
  - 报告中的 eprint: arXiv:2603.15031（以仓库 PDF 为准）

- `Qwen3.5` 状态说明：
  - 已确认官方发布文（Alibaba Cloud 社区博客）：
    - https://www.alibabacloud.com/blog/602894
  - 当前以官方发布报告为主；`Qwen3` 的 arXiv 技术报告可作方法论补充：
    - https://arxiv.org/abs/2505.09388

- `Kimi Opus` 命名说明：
  - 当前未检索到 Moonshot 官方名为 “Kimi Opus” 的技术报告或模型条目。
  - 已用可验证最新来源替代补充：
    - Kimi-K2.5 Full Report: https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf
    - Anthropic Opus 系统卡入口: https://www.anthropic.com/system-cards

### 2.2 LeWorldModel 深挖（对 Luma 的直接价值）

- 核心结论（来自论文摘要与项目页）：
  - LeWorldModel 主张 JEPA world model 可以从 raw pixels 端到端稳定训练。
  - 训练目标极简：`next-embedding prediction + Gaussian latent regularizer(SIGReg)`。
  - 作者报告将可调损失超参数从 6 个降到 1 个，并在多任务上保持竞争力。
  - 在其设定下，规划速度相对 foundation-model world model 有显著提升（论文/项目页报告最高 48x）。

- 对 Luma 的可迁移启发（架构层，不涉及执行排期）：
  - 启发1：`world JEPA` 目标应保持“latent 预测优先”，避免回退到像素/hidden 重建。
  - 启发2：world 分支损失尽量简化，优先“少损失项 + 强监控”而非多项堆叠。
  - 启发3：在你的一次性全量预训练目标下，应尽早固定 world 分支的正则形式，避免后期新增损失导致失稳。
  - 启发4：将“物理合理性/惊讶度（surprise）”作为评估维度之一，可补足纯 LM 指标盲区。

- 与 Luma v0.7/v0.7.2 的契合点：
  - 你的路线“主流预测外界 + 自省流预测认知状态”与 LeWorldModel 的 latent dynamics 思路一致。
  - 你强调的“最终一次性全量 run”与 LeWorldModel 的“简化损失、降低调参面”方向高度一致。
  - 因此，LeWorldModel 可作为你 `world JEPA` 设计的第一优先级参考来源，而不仅是补充案例。

- I-JEPA（表示空间预测）
  - 落点：Self JEPA 在 `c_t` 空间进行目标预测。
  - 本项目实现：从预测 `c_{t+1}` 改为预测 `Δc_t`。

- V-JEPA 2-AC（短 rollout）
  - 落点：one-step 主监督 + 2-step rollout 辅助监督。
  - 本项目实现：rollout 固定为 2-step，避免长链误差爆炸。

- LLM-JEPA
  - 落点：语言模型中可叠加 latent-space prediction。
  - 本项目实现：第一阶段即纳入 world JEPA，不延后到下一代架构。

- LeWorldModel
  - 落点：JEPA 可做成简洁稳定的主结构辅助器。
  - 本项目实现：world predictor 保持小头结构，强调稳态。
  - 当前工程补充：
    - 已实现 `LeWorldModel-style` fuller world JEPA 分支，用于对比：
      - 连续 span mask
      - 更强 context predictor
      - latent variance regularization
    - 当前短程结果显示：
      - fuller world JEPA 在 rollout 一致性上有潜力
      - 但综合最优仍是 `scaffold + self_check`
    - 因此：
      - `scaffold` 作为当前默认验证底座
      - `full` 作为后续中程/长程强化分支

### 2.3 `RaBitQ` 对 Luma 的价值定位

- 论文 / 项目：
  - RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search
  - DOI: https://doi.org/10.1145/3654970
  - 项目页: https://github.com/gaoj0017/RaBitQ

- 核心结论（来自论文摘要 / 项目页）：
  - 面向高维向量 ANN 检索，而不是语言模型主干训练。
  - 将 `D` 维向量量化为 `D-bit` 字符串，并给出理论误差界。
  - 重点收益是内存占用、检索效率与近邻估计稳定性。
  - 其工程实现强调 bitwise / SIMD 友好的快速搜索。

- 对 Luma 的直接启发：
  - 不适合作为 `JEPA` 或主干推理结构的直接替代。
  - 更适合放在：
    - `Tape` 记忆索引
    - `Router` 的 `routing_key` ANN 检索
    - 长期记忆 / 视图缓存的向量压缩层
  - 如果未来 `Tape / Router` 规模显著扩大，`RaBitQ` 可作为：
    - 更省内存的 ANN 索引候选
    - 比朴素 PQ 更有理论保障的检索后端候选

- 当前执行结论：
  - `RaBitQ` 不进入当前核心预训练主线。
  - `RaBitQ` 保留为 `Tape / Router / memory retrieval` 子系统的后续优化候选。
  - 只有当检索规模、延迟或内存成为明确瓶颈时，才进入独立验证。

### 2.4 `r_t` 设计约束（防止被 world 分支带偏）

- 设计原则：
  - `c_t`：慢环主状态，偏全局 / 世界态 / 元认知。
  - `r_t`：轻量 local reasoning state，偏局部递推与短程增益判断。

- `r_t` 的推荐递推形式：
  - `r_{t+1} = f_r(r_t, c_t, Δh_t, local_progress, local_gain_history, optional_hidden_summary)`
  - 其中：
    - `local_progress` 指当前 loop 进度或最近一步 / 两步的改善速度
    - `optional_hidden_summary` 只能是轻量当前主流摘要，不能演变成 world branch 的替身

- 明确禁止：
  - 不让 `r_t` 强依赖：
    - `world_summary`
    - `masked_world`
    - 其他重型 world latent
  - 否则 `r_t` 会被全局语境牵引，失去“局部递推状态”的意义

- 当前工程状态（2026-03-28）：
  - 现版 `r_t` 只做到部分对齐：
    - 递推阶段已经不直接读取 `world_summary / masked_world`
    - 但 bootstrap 仍使用 `compression_summary`
    - 递推输入仍含 `know_gap`，尚未完全切换到 `local_progress / recent gain` 主导
  - 因此：
    - 当前 `r_t` 只能算概念验证版
    - 如果未来继续追 `r_t`，优先修正输入定义，而不是先扩大结构

- HRM（多时间尺度）
  - 落点：快慢环分频更新。
  - 本项目实现：`h` 每步更新，`c_t/world_summary` 每 2 步更新。

- DAgger（分布偏移）
  - 落点：训练需覆盖 rollout 展开分布。
  - 本项目实现：在 rollout loss 中纳入展开后状态的一致性约束。

- Compounding Prediction Errors（误差累积）
  - 落点：短 rollout 比长 rollout 更稳。
  - 本项目实现：2-step 为上限，不扩展到更长链。
  - 更新说明（2026-03-28）：
- 该条仍是理论保守基线。
- 但当前短程 harder-math 实验已观察到 `3-step / 4-step` 乃至更长 `5-step` 的正收益信号。
- 当前长程专项实验已进一步观察到：
  - `10-step rollout` 在 `reason_loops=15` 下仍有明显收益
  - `reason_loops=20` 未继续带来额外收益
- 因此执行策略改为：
  - 预训练正式默认值仍先保守
  - 短程与中程验证允许继续向 `4-step / 5-step / 10-step` 扩展

---

## 3. 模块边界与接口约定（供 agent 实现）

### 3.1 Core 模块
- `CognitiveStateStream`
  - 输入：`h_t`, `compressed_summary_t`, `meta_state_{t-1}`
  - 输出：`c_t`, `know_gap_t`, `meta_state_t`

- `SelfJEPAPredictor`
  - 输入：`concat(c_t, delta_h_t)`
  - 输出：`pred_delta_c_t`

- `WorldJEPAPredictor`
  - 输入：masked latent blocks on `h_t` + unmasked context
  - 输出：predicted masked latent blocks

- `ExitController`
  - 输入：`delta_h_t`, `min_loops`, `stagnation_state`
  - 输出：`should_exit`, `stagnation_flag`

### 3.2 配置对象（建议）
- `LumaConfig`
  - `meta_dim=96`, `c_t_dim=64`
  - 当前短程验证底座：`slow_k=1`, `rollout_steps=4`, `reason_loops=8`
  - 中程专项候选：`rollout_steps=5`, `reason_loops=10`
- `LossConfig`
  - `alpha`, `beta`, `gamma`, `lambda_residual`
- `GateConfig`
  - `delta_h_threshold`, `stagnation_k`, `ct_cos_low`, `ct_cos_high`

---

## 4. 分阶段执行规范（Many Stages -> One Final Run）

> 每阶段必须按“输入-动作-验收-失败回滚”执行。

### 阶段0A：架构冻结与接口定稿（Architecture Freeze）
**输入**
- `v0.7 + v0.7.2` 合并后的固定原则。
- `minimind` 当前主干代码结构。

**动作**
- 固定双区分层：`Compress Zone` 与 `Reason Loop Zone` 的边界、输入输出、梯度边界。
- 固定核心模块接口：`CognitiveStateStream / WorldJEPAPredictor / SelfJEPAPredictor / ExitController`。
- 固定配置结构：`LumaConfig/LossConfig/GateConfig/DataConfig`。
- 固定日志与监控 schema（不得在后续阶段频繁改字段）。

**验收**
- 输出一份“接口冻结清单”（字段、shape、dtype、调用时机）。
- 单元测试能校验关键接口 shape 与数值范围。

**失败回滚**
- 回滚到“仅 LM baseline + 冻结配置骨架”。

### 阶段0B：模块拆分与验证脚手架（Module Split & Harness）
**输入**
- 阶段0A接口冻结清单。

**动作**
- 搭建最小可运行子模块测试：
  - `mamba3_op_test`
  - `attnres_block_test`
  - `jepa_flow_test`
  - `compress_reason_boundary_test`
- 搭建“短程训练 harness”（固定样本 + 固定随机种子 + 固定步数）。

**验收**
- 每个子模块可独立跑通并产生日志。
- harness 可复现（同 seed 两次偏差在容忍范围）。

**失败回滚**
- 只保留通过的模块测试，阻断失败模块进入主干。

### 阶段0C：Mamba3 算子构建与数值安全（Mamba3 Build）
**输入**
- 阶段0B测试脚手架。

**动作**
- 构建 Mamba3 相关算子路径（前向/反向/状态传递）。
- 明确 BF16 安全边界：SSM 核心与跨循环 state 强制 BF16。
- 建立长序列/循环稳定性冒烟测试。

**验收**
- 无 NaN/Inf、无状态爆炸、无 dtype 非法降级。
- 跨循环 state 传递正确且可复现。

**失败回滚**
- 回滚到稳定的基线 SSM 实现，保持接口不变。

### 阶段0D：AttnRes 分块与跨区聚合（AttnRes Build）
**输入**
- 阶段0C稳定主干。

**动作**
- 构建压缩区 block 化输出（`block_reprs`）与 `detach` 机制。
- 构建推理区 `UnifiedAttnRes`（历史循环 + 压缩区聚合）。
- 验证分块后显存/吞吐曲线与数值稳定性。

**验收**
- `block_reprs` 维度、刷新频率、梯度边界全部正确。
- 跨区聚合对主损失有可观测正向贡献或中性。

**失败回滚**
- 回滚到“仅压缩区输出 + 不跨区聚合”的保守配置。

### 阶段0E：JEPA 训练流搭建（Self + World + Rollout Pipeline）
**输入**
- 阶段0D通过的主干。

**动作**
- 接入 self JEPA（`Δc_t`）、world JEPA（masked latent）和 2-step rollout。
- 完成多损失调度、日志拆分、梯度归因检查。
- 接入停滞检测（`delta_h` 连续低阈值 + 任务未完成）。

**验收**
- 四损失都可稳定反向传播。
- 能区分“真收敛”与“伪稳定器”。

**失败回滚**
- 保留 self/world 主损失，暂降 rollout 到最小权重。

### 阶段0F：压缩区与推理区分层联调（Two-Zone Integration）
**输入**
- 阶段0E完成的训练流。

**动作**
- 联调“压缩区一次执行 + 推理区循环执行 + 慢环更新”。
- 验证循环深度分布、跨区信息利用率、退出策略健康度。

**验收**
- 双区协同稳定，循环行为符合预期（非过早退出/非无效长循环）。

**失败回滚**
- 回滚到单区保守模式并定位边界错误。

### 阶段0G：数据准备（DataMix v1：先变聪明，再变像 Luma）
**输入**
- 你的 X 聊天/推文原始导出数据。
- Hugging Face 候选数据集池。

**动作**
- 工作区固定到：`/home/kt/ai/luma_dataset`
- 目录固定为：
  - `persona_seed/`
  - `manifests/`
  - `buckets/`
  - `scripts/`
  - `cache/`
  - `raw/`
  - `processed/`
- 原则先重排：
  - `50%` 必须优先给“让 Luma 更聪明”的语料，而不是人格/风格语料。
  - 剩余 `50%` 再分给人格、情感、对话表现与回答对齐。
  - 难数学确实应进入聪明桶，因为它更能拉动多步推进、局部一致性、anti-stupidness 与代码推理。
  - 但 harder math 不能无限放大，否则会伤 `dialogue / emotion`，把 Luma 推成过冷的解题器。
- 当前冻结的 `DataMix v1` 结构：
  - `A. 聪明桶（50%）`
    - `smart_math_reasoning = 25%`
      - `EleutherAI/hendrycks_math`
      - `ricdomolm/MATH-500`
      - `Maxwell-Jia/AIME_2024`
      - `openai/gsm8k`（作为较易锚点）
    - `smart_code_python = 15%`
      - 本地 `minimind` Python 代码
      - 本地 `parameter-golf` Python 代码
      - `bigcode/the-stack` 的 Python 子集（仅在 permissive 过滤后进入正式混合）
      - 预留 `pytorch_examples_seed`
    - `smart_reasoning_dialogue = 10%`
      - `OpenAssistant/oasst1`
      - `HuggingFaceH4/ultrafeedback_binarized`
      - 预留本地工具规划轨迹
  - `B. 情感与支持桶（20%）`
    - `facebook/empathetic_dialogues`
    - `thu-coai/esconv`
    - `LooksJuicy/Chinese-Emotional-Intelligence`
    - `Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset`
  - `C. 人格桶（15%）`
    - 来源目录：`/home/kt/ai/luma_dataset/persona_seed`
    - 当前主要文件：
      - `wechat_pretrain.jsonl`
      - `pretrain.jsonl`
    - 目标：给 Luma 注入“来自你”的表达底色与连续人格线索，但不让它压过能力主线。
  - `D. 对话质量与叙述桶（15%）`
    - `OpenAssistant/oasst1`
    - `HuggingFaceH4/ultrafeedback_binarized`
    - `wangrui6/Zhihu-KOL`
    - `BelleGroup/multiturn_chat_0.8M`（当前先 hold，等待 license 边界确认）
- 当前数据工件固定为：
  - `luma_dataset/manifests/datamix_v1.yaml`
  - `luma_dataset/manifests/license_whitelist.md`
  - `luma_dataset/manifests/datamix_stats.template.json`
- `DataMix v1` 额外约束：
  - 人格桶是正式桶，不再只是后期 probe 时单独观察。
  - 但人格桶不能反客为主，不能压过“更聪明”这条主线。
  - `code_python` 必须保持 Python / PyTorch 取向，而不是变成宽泛代码垃圾桶。
  - `copyleft`、`unknown`、`research-only`、限制再分发的来源，在人工复核前不得进入最终混合。
  - 默认先做来源占比、采样温度、license 白名单与去重规则四件事，再冻结数据版本。

**验收**
- 产出可复现数据工件：`datamix_v1.yaml`、`datamix_stats.json`。
- 完成 license 与用途审查（非商用/署名/传播限制明确）。
- `DataMix v1` 占比通过人工复核：
  - “聪明桶”总占比确认为 `50%`
  - `luma_dataset/persona_seed` 已正式并入人格桶
- 人格一致性抽样评估通过（人工 + 规则双检）。

**失败回滚**
- 出现合规风险时，先剔除风险源再重跑混合，不进入训练阶段。

### 阶段1：并行 `c_t` + 慢环机制验证（短程）
**输入**
- 阶段0A~0G 全部通过。

**动作**
- 接入并行 `c_t` 状态流与慢环 `k=2`。
- 执行 A/B：`with c_t` vs `without c_t`。

**验收**
- `c_t` 注入 KL 非零、循环步数有方差、`c_t` 不塌缩。

**失败回滚**
- 回到阶段0E 的稳定训练流，逐项恢复。

### 阶段2：Residual Self JEPA 验证（短程）
**输入**
- 阶段1通过。

**动作**
- 强化 `Δc_t` 目标、监控方差与伪收敛信号。

**验收**
- `self_jepa_loss` 下降且 `Δc_t` 方差健康。

**失败回滚**
- 降低 residual 正则或后移生效。

### 阶段3：World JEPA 验证（短程）
**输入**
- 阶段2通过。

**动作**
- 固化 mask 策略与 world predictor 结构。
- 同步进行 `scaffold` 与 `LeWorldModel-style full` 的 A/B 对照。
- 若成本可接受，继续推进 full 版本进入中程验证。

**验收**
- `world_jepa_loss` 下降且 LM 主损失无明显退化。
- 记录：
  - 参数量增量
  - 单轮短程验证耗时增量
  - `self_rollout_tail / mean_loss / hard_loop_var`

**失败回滚**
- 简化 predictor / 降低 mask 难度。

### 阶段4：Rollout + 停滞诊断验证（短程）
**输入**
- 阶段3通过。

**动作**
- 保持 2-step rollout，完善停滞检测。

**验收**
- rollout 一致性提升，停滞报警有效。

**失败回滚**
- 降 rollout 权重，不改步数。

### 阶段5：联合小规模彩排（仅一次）
**输入**
- 阶段1~4通过。

**动作**
- 四损失联合短跑，小范围扫描 `α β γ λ` 与学习率。

**验收**
- 无单项 loss 主导崩坏，形成最终候选配置。

**失败回滚**
- 回滚至上一个稳定配置，优先调权重。

### 阶段6：最终一次性全量预训练（唯一正式 run）
**输入**
- 阶段5通过并冻结的最终配置。

**动作**
- 执行唯一正式全量 run。
- 训练中仅允许故障级中断（硬件/系统异常）。
- 禁止中途策略改配后二次重跑。

**验收**
- 完整训练完成并产出可复现工件：配置、日志、checkpoint、评估摘要。
- Gate F 全部通过。

**失败回滚**
- 仅在故障级中断时，按最近一致性 checkpoint 续跑。
- 不允许切换实验策略重新开全量。

---

## 5. Gate 体系（强制）

### Gate P0（架构冻结）
- 通过条件：接口冻结清单完成；核心模块 I/O 与 dtype 约束可测试。

### Gate P1（模块脚手架）
- 通过条件：`mamba3_op_test/attnres_block_test/jepa_flow_test/compress_reason_boundary_test` 全通过。

### Gate P2（算子安全）
- 通过条件：Mamba3 路径无 NaN/Inf；跨循环 state 传递稳定；BF16 边界符合规范。

### Gate P3（分层联调）  
- 通过条件：压缩区与推  理区边界正确；跨区聚合不破坏主损失稳定性。
  
### Gate P4（数据就绪）  
- 通过条件：`cleaned_x.  parquet`、`hf_mix_manifest.yaml`、`datamix_stats.json` 齐备且可复现。
  
### Gate P5（合规）  
- 通过条件：X 数据隐私  脱敏完成；HF 数据 license 白名单通过；高风险样本已剔除。
  
### Gate A（架构）  
- 通过条件：`c_t` 注入  有效、慢环工作、退出分布健康。
- 当前阶段结论（2026-03-28）：
  - `c_t` 注入有效：已通过
  - 完整慢环工作：已通过（含跨循环 `meta_state`）
  - 退出分布健康：仅在训练侧 `soft exit / sampled exit` 下观察到非零方差；硬退出仍需继续优化
  
### Gate B（self）  
- 通过条件：`Δc_t` loss   下降，且方差不过低。
- 当前阶段结论（2026-03-28）：
  - `Self JEPA` 一阶目标已冻结为 `delta`
  - `c_t` 方差保持非零，未见塌缩
  - 短程训练下 `Δc_t` loss 已出现下降趋势
  
### Gate C（world）  
- 通过条件：world JEPA   收敛，LM 主损失无明显退化。
- 当前阶段结论（2026-03-28）：
  - `scaffold world JEPA` 已稳定工作
  - `LeWorldModel-style full world JEPA` 已接入，并在 rollout 一致性上显示更强潜力
  - 当前 short-run 综合最优仍为 `scaffold + self_check`
  - full 版本是否升为默认，取决于：
    - 中程训练是否继续保持优势
    - 参数与耗时增量是否可接受
  
### Gate D（rollout）  
- 通过条件：2-step 一致  性提升，停滞检测有效。
- 当前阶段结论（2026-03-28）：
  - `2-step rollout` 已接入，并已扩展到可实验 `3-step / 4-step`
  - rollout 监督在“数学 + 对话”混合数据上开始出现下降趋势
  - 退出目标已升级为“改善量型目标”，用于判断多走一步是否仍有收益
  - 在更难的 `hard_math_dialogue` 验证上，`3-step / 4-step` 在短程实验中优于 `2-step`
  - 初步倾向：`4-step` 的动力学约束最强，但是否作为最终默认值，仍需在更大样本和更长训练上复验
  - 已新增并完成长程专项候选：
    - `rollout_steps=10`
    - `reason_loops=15`
  - 当前观察：
    - `10x15` 有效
    - `10x20` 与 `10x15` 几乎一致
  - 因此下一阶段重点：
    - 不再单纯继续加大 `reason_loops`
    - 转为专攻 exit policy，让 `10x20+` 真正被利用
  - 新增 `full + depth2` 验证后的结论：
    - 即便在 `depth2` 基底上，`10x20` 仍几乎没有优于 `10x15`
    - 说明当前最大瓶颈仍然是 continuation / exit policy，而不是更高 loops 上限
  - 最新补充（`competition_math_dialogue_emotion + persona_seed`, `128-step`）：
    - `10x15` 继续可用
    - 共享推理 block 从 `1` 层变 `2` 层后，并未带来 mixed 综合反超
    - 说明当前更值得优先优化的是：
      - exit policy
      - task-bucket 条件下的动力学利用率
      - 而不是继续直接增加共享 block 深度

### Gate D.1（exit policy，新增）
- 通过条件：
  - 在更长 horizon（如 `10x15` 以上）配置下，
  - `10x20` 相对 `10x15` 能在至少一项关键指标上继续带来真实增益：
    - `hard_loop_var`
    - `self_rollout_tail`
    - 或更难数据集上的任务分桶表现
- 当前状态（2026-03-28）：
  - 未通过
  - 判断：当前瓶颈是 exit policy 提前截断，而不是最大 loops 上限不足
  
### Gate E（联合）  
- 通过条件：四损失并行  稳定，无单项主导崩坏。
  
### Gate F（最终 run 前  冻结）
- 通过条件：参数、配置  、checkpoint、监控面板、报警规则全部冻结。
  
---  
  
## 6. 最终一次性预训练  冻结清单（Run Lock）
- 冻结模型结构：`c_t`   并行流、2 层自省流、慢环 `k=2`。
- 冻结损失结构：`LM + W  orld + Self(Δc_t) + Rollout(2-step) + ResidualReg`。
- 冻结超参：`α β γ λ`、  学习率与调度、batch/seq、mask 比例。
- 冻结数据与切片：训练  集版本、验证集版本、采样策略、seed。
- 冻结数据配方：`DataMi  x v1` 来源占比、采样温度、去重策略、过滤规则、license 白名单。
- 冻结监控：关键指标、  报警阈值、停滞样本记录规则。
- 冻结运行规则：只允许  故障级中断续跑，不允许策略级重跑。

### 6.1 正式预训练 token / 时长预算（当前工程口径）
- 当前仍沿用 v0.7/v0.6 训练预算作为正式 run 的初始锚点：
  - 目标有效训练量：约 `5.3B effective tokens`
  - 对应原始训练量：约 `8.85B raw tokens`（按先前分层多轮与选择性保留估算）
- RTX 5090 单卡、`seq_len=32K`、FP8 主路径的旧规划估算为：
  - 吞吐：约 `15k ~ 20k tokens/s`
  - 预训练主 run：约 `6` 天量级
- 但需要明确：
  - 以上时间预算来自早期规划估算，不是当前训练栈已复现实测
  - 当前代码主线还未正式接入：
    - `8-bit Muon`
    - `8-bit AdamW`
    - 正式 FP8 分层训练栈
    - OPUS/选择性 token 采样
- 因此当前更诚实的表达是：
  - `5.3B effective tokens / ~6天` 仍可作为正式预训练的规划目标
  - 真正执行前，必须先完成一次预训练脚本级吞吐复测，再冻结最终 wall-clock 预算
  
---  
  
## 7. agent 执行提示（  简版）
- 每阶段开始前先读取“输  入”并检查依赖齐备。
- 执行中只做该阶段允许  动作，不跨阶段提前改造。
- 阶段结束必须出“验收结  果 + 失败回滚状态”。
- 任何 Gate 未通过，立  即回滚，不得推进到下一阶段。
  
---  
  
## 8. v0.7 补充并入（已  对齐 v0.7.2）
  
> 说明：本节将你提供的   `v0.7` 详细规格作为补充层并入。若与本文档前述 `v0.7.2` 固定项冲突，以前文固定项为准。
  
### 8.1 设计哲学补充（  强约束）
- 原则1：上下文推理优先  于权重记忆。
  - 事实性知识优先通过  上下文获取（搜索、Tape、会话历史）。
- 原则2：主动检索是推理  环的一部分，不是前处理。
  - `know_gap` 触发 Rou  ter 检索并回注当前循环。
- 原则3：自我感知主要来  自预测压力，不依赖显式“置信度标签”。
  - 显式监督仅保留 `kno  w_gap` 这类控制信号。
  
### 8.2 CDR-Loop 结构补  充（推荐默认）
- 压缩区（执行一次）：  
  - 长上下文压缩为结构  化表示，输出 `block_reprs` 供推理区跨区回看。
  - 推荐结构：Mamba 主  导 + 稀疏注意力校正 + Block 级残差聚合。
- 推理区（循环执行）：  
  - 每循环包括：Mamba   状态推进 -> 局部精读注意力 -> FFN -> UnifiedAttnRes -> 自省流更新 -> JEPA 相关损失计算 -> 退出判定 -> 可选检索回注。
- 退出逻辑（与本文档前  文一致）：
  - 基于 `jepa_error +   delta_h` 的真实收敛信号进行退出评分。
  
### 8.3 双流 JEPA 细化  补充（与 v0.7.2 兼容）
- 主流 JEPA（外界/世界  态）：
  - 采用 EMA 目标编码器  思路，核心用途是防 collapse 与稳定目标。
- 自省 JEPA（认知态）：  
  - `c_t` 分支使用 `det  ach` 目标，预测对象为 `Δc_t`（v0.7.2 固定）。
  - 预测器输入保留 `del  ta_h`，避免惰性解（直接复制/低变化伪收敛）。
- 统一结论：  
  - EMA 用于“独立目标编  码器分支”；
  - `detach` 用于“同流  真实下一时刻输出目标”。
  
### 8.4 v0.7 关键变更并  入清单（已采纳）
- `confidence` 从显式输  出改为由 `jepa_error + delta_h` 间接计算。
- `strategy` 从模型内部  显式头迁移到 OpenClaw 行为层解析。
- 自省能力来源按三层组  织：
  - 表示层（主流 JEPA）  
  - 认知状态层（自省 JE  PA / `c_t`）
  - 语言层（SFT 将隐式  状态映射为可表达文本）
  
### 8.5 训练与数值补充  （推荐默认）
- 精度分层：  
  - SSM 核心与跨循环状  态使用 BF16（避免误差累积）。
  - 线性/注意力可在安全  范围内用 FP8 加速。
- 归一化：  
  - 推荐 ZC-RMSNorm 体  系，循环边界必须有稳定化归一化层。
- 损失优先级：  
  - `LM` 为主，`world/s  elf/rollout` 为辅助；避免辅助损失压制主目标。
  
### 8.6 记忆与系统协同  补充（推荐默认）
- `know_gap` 仅负责“是  否需要补信息”的内部信号。
- Router / Tape / Searc  h / OpenClaw 负责外部信息获取与行为执行。
- 统一输入标记建议保留  （如 `[TAPE_MEMORY]`、`[SEARCH_RESULT]`），便于可追溯训练与调试。
  
### 8.7 冲突处理规则（  强约束）
- 若 `v0.7` 文本与 `v0.  7.2` 固定项冲突，按以下优先级：
  1. 本文档第1节固定架  构与训练决策
  2. 本文档第4节阶段 Ga  te 与一次性全量 run 约束
  3. 本节（v0.7补充）作  为细化参考
  
---  
  
## 9. 默认假设  
- 本文档面向执行，不面  向论文写作。
- 默认主仓为 `minimind`  ，`parameter-golf` 仅用于机制快验。
- 默认采用“前期多阶段验  证 + 最终一次性全量上线”的执行模型。

---

## 10. v0.6 到 v0.7.2 的补充基线（压缩区/推理区/FP8）

> 本节把你提供的 v0.6 细化规格并入 v0.7.2 主规划，作为工程默认值。若冲突，仍以第1节固定项优先。

### 10.1 压缩区补充配置（执行一次）
- 拓扑固定：`24`层，`5:1`比例，`4`组，每`3`层输出一次 `block_repr`（总`8`个）。
- Mamba主层：`d_model=768, d_state=192, post-gate RMSNorm=ON`。
- 精读层交替：
  - 奇数组：`KDA(无FoX)`
  - 偶数组：`SWA(window=1024) + SwiGLU(ffn_dim=3072) + SandwichNorm`
- `block_repr` 到推理区一律 `detach`，不反传到压缩区历史缓存。

### 10.2 推理区补充配置（循环1~8）
- `mHC` 不再视为“循环外一次执行”的前置算子；若启用，应作为推理区残差流组织方式包裹共享循环块。
- 推理区 memory tokens（4个）保留。
- 循环内步骤固定顺序：
  1. `Mamba3(initial_state跨循环, BF16 state)`
  2. `GatedDiffAttn + FoX + SWA(1024)`
  3. `SwiGLU FFN(3072)`
  4. `UnifiedAttnRes(loop_history + compress_block_reprs)`
  5. 自省流（`meta_dim=96, c_t=64, 慢环k=2`）
  6. 双流JEPA（world + self）
  7. 退出控制（`jepa_error + delta_h`）
  8. `know_gap` 触发 Router 检索并回注
- 退出阈值默认 `0.85`，并保留停滞告警指标（连续低 `delta_h` 且任务未完成）。

### 10.3 1M 外推补充约束
- 训练上下文：默认 `seq_len=32768`。
- 推理外推：`Mamba` 负责全局递归积累，`SWA=1024` 负责局部精读，`know_gap+Router` 负责主动检索补偿。
- SWA窗口不随序列长度扩张（固定1024，控制显存上界）。

### 10.4 FP8 分层安全补充
- 允许 FP8：
  - FFN线性层、Attention Q/K/V/O、Router Projector（按 TE recipe）。
- 强制 BF16：
  - Mamba SSM 核心
  - 跨循环 `mamba_state/meta_state`
  - ZC-RMSNorm / Embedding / JEPA目标更新路径
- 训练建议：前向 `E4M3`、反向 `E5M2`，并保留关键边界层 BF16。

### 10.5A 规模路线重新定位（2026-03-29 更新）
- 当前重新确认：`0.3B` 更适合作为架构验证基线，而不是最终形态。
- 对 Luma 这种目标函数很重、同时追求推理/情感/人格/代码/中文长回答的系统来说：
  - `0.3B`：适合机制筛选、loss/exit/world-self JEPA 验证、短中程 trainer 验证。
  - `0.6B`：适合第一版正式预训练候选，开始进入“可认真使用”的区间。
  - `1.2B+`：更接近你真正想要的完整 Luma 形态。
- 因此当前规模判断改为：
  - `0.3B = 架构验证基线`
  - `0.6B = 第一版正式训练候选`
  - `1.2B = 更可信的中期目标规模`
- 需要诚实说明：
  - `0.6B` 很可能仍然不是“最终足够”的规模，尤其在你希望 Luma 同时兼顾：
    - 长推理
    - 情感与支持性表达
    - 中文叙述感
    - 代码与数学
    - 人格连续性
  - 所以 `0.6B` 更像“认真可用版起点”，不是“最终就够了”。
- 扩容顺序当前固定为：
  - 先加深，再加宽。
- 理由：
  - 当前大量实验结论都建立在 `hidden=768` 的内部比例上；直接加宽会同时扰动 `meta_dim / c_t / head_dim / kv_heads / loss dynamics`。
  - Luma 当前的收益更依赖层次、递推深度和共享推理 block 的表达能力，而不是先把单层做得更胖。
  - 这也更符合 `Progressive Stacking` 的渐进扩容路线。
- 当前推荐扩容路径：
  1. `0.3B -> 0.6B`：优先加深（保持宽度不变或近似不变）
  2. `0.6B -> 1.2B`：再考虑加宽
  3. `1.2B -> 1.5B+`：再按目标资源和训练稳定性决定是否继续加宽/加深
- 工程含义：
  - 后续若开始准备正式预训练，不应把 `0.3B` 当作最终规模。
  - `0.6B` 可以作为第一次正式 run 候选，但应提前接受：它大概率仍是通往 `1.2B+` 的过渡站。

### 10.5 参数补充锚点（0.3B默认）
- `hidden=768, heads=12, kv_heads=3, ffn=3072`
- 压缩区 `24` 层，推理循环训练 `1~6`（课程式），推理上限 `8`
- `self_dim=384, meta_dim=96, c_t=64, router_dim=256`
- `loss` 初值建议：`lm=1.0, world=α, self=β, rollout=γ, residual_reg=λ`
- 当前代码实测修正：
  - `scaffold`: `266.739M`
  - `full`: `269.542M`

### 10.6 优化器与训练栈现状（必须诚实标记）
- 正式预训练目标优化器栈更新为：
  - 线性层主体：`Muon + MuonClip + Modular Norm`
  - 其他参数：`AdamW + Modular Norm`
- 采用理由：
  - `Muon` 更适合大规模线性层主体
  - `MuonClip` 用于抑制训练尖峰与不稳定更新
  - `Modular Norm` 对未来渐进式扩容（Progressive Stacking / width-depth growth）是必要支撑
  - 这条路线比“纯 AdamW”更符合你未来的渐进扩容主线
- 渐进扩容含义：
  - 若后续执行 `Progressive Stacking`
  - 则优化器主线不能只考虑当前 0.3B，而必须从一开始兼容：
    - 层复制后的稳定迁移
    - 宽度扩张后的学习率迁移
    - 不同参数组的更新尺度控制
- 但当前代码与验证脚本现状不是这个配置：
  - 当前状态已更新为：
    - `minimind/scripts/run_luma_stage12.py` 的 stage2 验证主线，已接入：
      - 外部包 `Muon`
      - 本仓轻量 `MuonClip`
      - 本仓轻量 `Modular-Norm-style` 学习率缩放
    - `minimind/trainer/train_luma_pretrain.py` 已作为正式 `Luma` 预训练实验主线落地，当前支持：
      - `full + self_check` 默认主干
      - 参数分组
      - scheduler
      - checkpoint / resume
      - 外部 `bitsandbytes` 的 `AdamW8bit`
      - 运行时量化状态版的实验性 `8-bit Muon`
  - 当前尚未接入的部分必须明确标记：
    - 尚未接入 `Transformer Engine` / `torchao` 风格的真正 `FP8` 训练栈
    - 尚未在正式 trainer 主线落地：
      - `FP8 autocast`
      - `FP8 GEMM`
      - `FP8 optimizer state` 之外的激活/主干混精链路
    - 因此当前工程状态只能写为：
      - `8-bit optimizer` 已接入
      - `FP8 training` 尚未接入
  - 现阶段尚未完成的部分：
    - 更长训练长度下 `8-bit Muon` 的稳定性与收益验证
    - 正式大样本 pretrain 数据上的 wall-clock / 显存曲线
- 执行含义：
  - 当前 stage0/1/2 实验结论，主要用于验证架构与动力学
  - 当前可以把它视为“已进入正式预训练实验主线，但仍处于实验性接线阶段”

### 10.6A 结构扩容与表达能力策略（正式 run 前的取舍规则）
- 压缩区是否加入轻量慢环：
  - 当前默认：**不加**
  - 原因：
    - 压缩区职责是“一次性稳态压缩”，不是显式循环推理
    - 在压缩区再加入慢环，会把“压缩质量问题”和“元状态调度问题”耦合在一起
    - 更适合作为后续专项分支，而不是当前默认
- 推理区是否再加一层独立参数层：
  - 当前建议：**先不直接加**
  - 更优先的做法：
    - 先提高共享循环块内部表达能力
    - 再看是否需要增加非共享独立层
  - 原因：
    - Luma 的核心优势来自“可重复使用的推理 block + 动态退出”
    - 过早加很多循环外独立层，容易把收益重新拉回普通深堆叠
- “单层循环会不会太弱”：
  - 当前判断：不能把当前 loop 理解成“单层”
  - 它本质上是一个共享的复合 block
  - 真正需要警惕的不是“单层”，而是：
    - block 内表达是否过瘦
    - exit policy 是否让更深预算无法被使用
  - 因此下一阶段优先级应是：
    1. 先把 `full + self_check` 在更长训练上验证清楚
    2. 再决定是否扩宽推理 block
    3. 压缩区慢环放到更后面的专项验证
- 共享推理 block 的真实深度：
  - 当前已新增 `reason_shared_depth`
  - 实验主线可直接测试“每轮循环里的共享推理 block 从 1 层变为 2 层”
  - 当前建议：
    - 当前 `128-step` 分桶实验后，正式预训练实验默认候选先保留 `reason_shared_depth=1`
    - `reason_shared_depth=2` 继续保留为专项表达能力候选
  - 当前 `128-step` 实验结论：
    - `depth=1`：
      - mixed `self_rollout_tail = 0.1719`
      - persona_seed `self_loss_tail = 0.0536`
      - persona_seed `self_rollout_tail = 0.0664`
    - `depth=2`：
      - mixed `self_rollout_tail = 0.2617`
      - persona_seed `self_loss_tail = 0.0586`
      - persona_seed `self_rollout_tail = 0.0664`
      - emotion `hard_loop_var = 3.6875`，显著高于 `depth=1`
    - 工程判断：
      - `depth=1` 综合更稳，更适合当前正式预训练候选底座
      - `depth=2` 更像情感/表达专项分支，而不是当前默认
  - `depth=2` 上的进一步验证：
    - `reason_loops=20` 相对 `15` 没有形成额外收益
    - `self_check_k=2` 时：
      - mixed `self_rollout_tail` 从 `0.2617` 改善到 `0.2168`
      - emotion 有轻微回退，但未崩坏
    - 轻量增重自省/自检（`meta_dim=80, meta_state=24, c_t_dim=40, self_check_dim=24`）时：
      - mixed / dialogue / persona_seed 变强
      - 但 `math / emotion` 出现明显退化
  - 当前针对 `depth=2` 的执行建议：
    - 若以“提高推理且尽量不掉 emotion”为目标：
      - 优先继续验证 `self_check_k = 2`
      - 不优先继续上调 `reason_loops`
      - 不把“轻量增重自省/自检”直接扶正为默认
    - 当前建议的 `depth=2` 实验底座：
      - `full + depth2 + self_check`
      - `self_check_k = 2`
      - `rollout_steps = 10`
      - `reason_loops = 15`
  - `512-step` 的 one-step vs light two-step A/B 结论：
    - 纯 `one-step` 已不再是最佳默认
    - 当前更优默认应改为：
      - `one-step continuation gain` 作为主监督
      - `light two-step continuation auxiliary` 作为默认辅助
    - 关键结果：
      - mixed `self_rollout_tail`: `0.068359375 -> 0.052734375`
      - math `self_rollout_tail`: `0.07421875 -> 0.0625`
      - dialogue `self_rollout_tail`: `0.15625 -> 0.05859375`
      - persona_seed `self_rollout_tail`: `0.3896484375 -> 0.24609375`
      - emotion `self_rollout_tail` 略有回退：`0.10546875 -> 0.109375`
    - 因此当前主线不再写成“只做一步”，而应写成：
      - 一步主
      - 两步辅
      - 不让 two-step 直接接管 exit policy
  - `self_check_k` 的进一步对比（`1 / 2 / 3 / 2+crystal`）结论：
    - `k=2` 不是偶然点，确实优于 `k=1`
    - `k=3` 在 `dialogue / persona_seed` 方向更强，但 mixed 总体并未超过 `k=2`
    - `k=2 + JEPA crystal` 在顶层 `self/rollout` 上更强，但 mixed 总体更不稳
    - 补跑 `k=3 + crystal / k=4 + crystal / k=3 + crystal + 10x20` 后：
      - `k=3 + crystal` 比 `k=4 + crystal` 更值得保留
      - `k=4 + crystal` 没有继续放大 crystal 的优势
      - `k=3 + crystal + 10x20` 与 `10x15` 几乎重合，说明当前瓶颈仍然是 exit policy
  - 当前冻结判断：
    - 默认继续保留 `self_check_k = 2`
    - `k=3` 作为“聊天伙伴感 / persona 优先”专项候选
    - `JEPA crystal` 作为退出策略专项研究项，不直接扶正；若继续追踪，优先 `k=3 + crystal`
  - 新增自省流备选项：
    - `introspection uncertainty`
      - 定义：由自省流直接产出一个轻量 `uncertainty` 信号，表示“当前内部叙事到底有多不确定 / 多犹疑”
      - 当前建议用途：
        - 不直接接管 exit logit
        - 只用于调节 `light two-step continuation auxiliary` 的权重
      - 第一轮 `512-step` 试验结论：
        - 信号本身是活的（stage1 uncertainty 非零）
        - 但当前接法会把 mixed/per-bucket 的 `rollout_tail` 压到近零，说明权重过强且过早介入
      - 后续三组降火力试验结论：
        - `clipped uncertainty weighting`：失败，rollout 仍被压扁
        - `uncertainty-as-gate`：失败，gate 因 uncertainty 饱和而几乎总开
        - `crystal + uncertainty` 低火力版：mixed `self_tail` 更低，但 rollout 仍全桶归零
      - 当前判断：
        - 保留为 exit policy 研究备选
        - 暂不扶正进默认基线
        - 暂不再沿“直接调 two-step 辅助权重”这条支路继续深挖

### 10.6C JEPA predictor 的实现边界（必须诚实标记）
- 当前 JEPA predictor 不是论文作者官方仓库直接实现。
- 当前状态应准确写为：
  - `SelfJEPAResidualPredictor`：自研工程实现
  - `WorldLatentJEPA`：自研 scaffold 版
  - `LeWorldModelStyleJEPA`：论文风格工程迁移版
- 因此结论边界应始终写成：
  - “方向对齐论文 / 风格对齐论文”
  - 不能写成“官方 predictor 已完整接入”
- 对“这是不是当前主流 JEPA 的最好做法”的当前判断：
  - 不能写成“是”
  - 更准确的说法是：
    - 当前做法是“面向 Luma 目标的工程上合理做法”
    - 但不是业界或 JEPA 文献已经统一收敛的“最好标准答案”
  - 尤其在一手文献里，主流趋势更像是：
    - mask / context 设计更重要
    - latent regularization 很重要
    - 多层监督 / dense predictive objective 正在变强
    - predictor 形式本身并没有唯一公认最优模板

### 10.6D Self-check 节奏与 JEPA crystal（新增）
- 当前对 `self_check_k` 的对比结论：
  - `k=2`：当前最稳默认
  - `k=3`：更偏 `dialogue / persona_seed` 方向
  - `k=2 + crystal`：顶层 `self/rollout` 更强，但 mixed 总体更不稳
  - `k=3 + crystal`：比 `k=4 + crystal` 更值得保留，但 mixed 仍未稳过 `k=2`
  - `10x20`：当前没有比 `10x15` 带来实质增益
- 对 `JEPA-guided entropy crystallization` 的当前判断：
  - 它可以与现有退出信号共同作用
  - 但当前不应写成“默认必加”
  - 更准确定位是：
    - 退出策略专项研究项
    - 不是当前默认主线

### 10.6B 任务分桶验证要求（新增）
- 阶段1/2 之后的中程验证，不再只看 mixed 总分，必须至少分以下桶记录：
  - `math`
  - `dialogue`
  - `emotion`
  - `persona_seed`
  - `mixed`
- `persona_seed` 定义：
  - 来源目录：`/home/kt/ai/luma_dataset`
  - 当前主要文件：
    - `wechat_pretrain.jsonl`
    - `pretrain.jsonl`
  - 作用：
    - 作为 Luma 的人格种子 / 真实发言种子桶
    - 单独监控其 `self_loss_tail`、`self_rollout_tail` 与循环深度分布
- 当前实验结论：
  - `persona_seed` 已正式进入 stage12 验证链路
  - 在 `full + self_check + 10x15 + 128-step + depth=1` 下：
    - `self_loss_tail = 0.0536`
    - `self_rollout_tail = 0.0664`
  - 说明当前架构已经能在人格种子桶上形成稳定的短程自省/动力学收敛

### 10.6E Guard 分层规则（2026-03-28 更新）
- 从本次起，stage12 / autoresearch 的 guard 分成两层：
  - 硬护栏（hard guard）
    - 参数量 `<= 0.35B`
    - `math` 提升或至少不掉
    - `dialogue` 不明显恶化
    - `emotion` 保持优势或不明显恶化
    - `mixed` 不崩
  - 软护栏（soft guard）
    - `persona_seed` 不再作为一票否决项
    - `persona_seed` 的退化要被显式记录，但不自动判为结构失败
- 原因：
  - Luma 的目标是“聪明的聊天伙伴”，不是“对用户原始发言分布的完全复刻器”
  - `persona_seed` 更适合做“人格种子 / 风格代价项”，而不是“绝对结构护栏”
- 工程含义：
  - 若某个实验在 `mixed / math / dialogue / emotion` 上明显更优，但 `persona_seed` 退化，则应标记为：
    - `soft-guard tradeoff`
    - 或 `alt-keep candidate`
  - 不应再直接因为 `persona_seed` 单项退化而判死
- 当前直接影响：
  - `iteration 5`（light two-step continuation-value auxiliary）在旧 guard 下被判 discard
  - 在新 guard 下，它先被视为：
    - “值得继续恢复并向下探索的候选分支”
    - 因为它把 mixed `self_rollout_tail` 从 `0.041015625` 降到 `0.0390625`
    - 但代价是 `persona_seed rollout_tail` 从 `0.361328125` 升到 `0.666015625`
  - 随后又在统一的 `512-step` A/B 中被重新验证：
    - `one-step only`:
      - mixed `self_tail = 0.169921875`
      - mixed `rollout_tail = 0.068359375`
    - `one-step + light two-step auxiliary`:
      - mixed `self_tail = 0.054443359375`
      - mixed `rollout_tail = 0.052734375`
      - `math` 也同步改善：`0.07421875 -> 0.0625`
      - `dialogue` 同步改善：`0.15625 -> 0.05859375`
      - `persona_seed` 也明显回收：`0.3896484375 -> 0.24609375`
      - 仅 `emotion rollout_tail` 略有回退：`0.10546875 -> 0.109375`
  - 因此当前正式基线更新为：
    - `one-step continuation gain` 作为主监督
    - `light two-step continuation auxiliary` 作为默认辅助监督
    - 不再把“纯 one-step”当作默认 retained 主线
  - 当前新的 retained 解释：
    - `iteration 2` 代表“one-step continuation gain 的稳定骨架”
    - `iteration 5` 代表“one-step main + light two-step auxiliary”的升级版默认基线
  - 下一阶段 exit policy 优化应围绕这条新基线继续，而不是回退到纯 one-step

#### 10.6.0 基线前缀命名规则（2026-03-29）
- 从现在起，每一条正式实验线都挂在一个“基线前缀（baseline prefix）”下面。
- 规则：
  - 每当正式基线发生切换，就分配一个新的前缀。
  - 后续所有实验都以 `前缀 + 变体名` 的方式记录。
  - 这样可以把“这是谁的变体”与“它是在什么历史基线上长出来的”分开。
- 当前建议命名：
  - `A0`
    - 早期纯 `one-step` continuation skeleton
    - 对应历史里的 `one_step`
  - `A1`
    - `one-step main + light two-step auxiliary` 升级线
    - 对应历史里的“512-step A/B 后被扶正的默认基线”
    - 过去文档里常被口语化写成 `iter5` 的那条思想来源，但不再直接把 `iter5` 当正式基线名
  - `A2`
    - 当前正式长程基线
    - 对应历史里的 `iter2` 长程扶正版本
    - 结构定义：`full + depth2 + self_check_k=2 + one-step main + light two-step auxiliary`
- 命名示例：
  - `A2-predictor_progress`
  - `A2-progress_shape_v1`
  - `A2-local_consistency_v2`
  - `A1-crystal_probe`
- 旧名字回填说明：
  - `iter2`
    - 现在正式映射为 `A2-core`
  - `iter5`
    - 不再直接当作正式基线名使用
    - 仅保留为历史口语标签，指向“`A1` 这代 one-step main + light two-step auxiliary 的升级思想来源”
  - `iter9`
    - 现在更准确地写成 `A2-iter9_bundle` 或 `A2-structured_world_bundle`
    - 表示它是建立在 `A2` 主线附近发展出的研究分支，而不是独立新基线

#### 10.6.1 10240-Step 长程基线重新扶正（2026-03-29）
- 在 `10240-step` 中程验证后，当前正式长程基线重新明确为：
  - `A2-core`（旧名：`iter2`）
- 原因：
  - `A2-core` / 旧 `iter2` 在长程里重新体现出最稳的 `mixed / math / python_code` 表现
  - `iter9` 与 `iter9 + crystal` 在中程里更像把 rollout 动力学压平，而不是自然学好
  - `ExpD` 保住了一部分 bucket 动态性，但还没有赢过 `iter2`
- 当前正式长程基线配置：
  - 架构底座：
    - `full + depth2 + self_check`
  - 自省与退出：
    - `self_check_k = 2`
    - `one-step continuation gain` 作为主监督
    - `light two-step auxiliary` 作为默认辅助监督
  - world JEPA：
    - `world_jepa_mode = full`
    - `world_mask_strategy = default`
    - `world_full_simplify_loss = false`
    - `enable_exit_jepa_crystal = false`
  - 共享推理块：
    - `reason_shared_depth = 2`
  - 推理预算：
    - `rollout_steps = 10`
    - `reason_loops = 15`
  - 长程验证数据桶：
    - `fixture_mode = competition_math_dialogue_emotion`
    - `enable_persona_seed = true`
    - `enable_python_code = true`
  - 长程验证训练口径：
    - `seq_len = 256`
    - `samples = 8`
    - `stage2_steps = 10240`
  - 未启用的研究分支：
    - `math_adapter_lane = off`
    - `math_summary_gate = off`
    - `r_t reasoning ring = off`
    - `uncertainty feature = off`
    - `crystal feature = off`
- 配置语义：
  - 这是当前最适合继续推进正式 trainer / 更长训练预算的稳定骨架
  - `iter9`、`iter9 + crystal`、`ExpD` 继续保留为研究分支，不删除

#### 10.6.2 动力学主候选推进（2026-03-29）
- 基于后续 `2048-step` 筛选，当前“最值得继续追的动力学强化线”更新为：
  - `A2-progress_shape_v1-h3`
- 它的语义是：
  - 仍然挂在 `A2-core` 上
  - 继续使用：
    - `predictor_progress`
    - `progress-shape`
  - 但把 rollout 主监督拉回近端：
    - `horizon = 3`
- 当前判断：
  - `A2-core`
    - 仍然是正式长程基线
  - `A2-progress_shape_v1-h3`
    - 是当前最优先的动力学强化候选
    - 适合在进入下一轮长程筛选前继续做 `2048/4096` 级验证
- 不再优先的方向：
  - 继续拉长 rollout horizon
  - uncertainty 直接调 `two-step aux` 权重
  - 更重的 local consistency penalty
  - 重新扶正更重的并行状态流

#### 10.6.3 下一批结构候选（从动力学文献回看后新增）
- 在不改变 `A2-core` 正式长程基线地位的前提下，下一批值得进入筛选的结构候选为：
  - `A2-progress_shape_v1-h3 + local_rollout_head`
  - `A2-progress_shape_v1-h3 + progress_exit_readout`
  - `A2-progress_shape_v1-h3 + dual_rate_self_predictor`
  - `A2-progress_shape_v1-h3 + trajectory_health_probe`
  - `A2-progress_shape_v1-h3 + backtrack_aware_progress`
- 这些候选的共同原则：
  - 优先改 predictor / readout / diagnostics
  - 不优先新增更重的并行状态流
  - 不优先继续把 horizon 拉长
  - 不把 crystal / uncertainty 重新扶正成主监督角色
- 执行顺序建议：
  1. `local_rollout_head`
  2. `progress_exit_readout`
  3. `dual_rate_self_predictor`
  4. `trajectory_health_probe`
  5. `backtrack_aware_progress`

### 10.7 OPUS 数据选择路线（2026 候选）
- 候选技术：
  - `OPUS = Optimizer-induced Projected Utility Selection`
- 适用定位：
  - 作为正式预训练阶段的数据选择/保留层
  - 用于降低总 token 需求，而不是替代模型结构验证
- 当前判断：
  - 若论文中“约 `30B` token 达到传统 `200B` token 级效果”的趋势在 LLM 预训练上可复现，它对你的“一次性正式 run”非常有吸引力
  - 它和 `AdamW/Muon` 系列优化器路线在理念上是兼容的，适合进入候选栈
- 但当前状态必须诚实写明：
  - 尚未在本仓正式实现
  - 尚未在 Luma 当前 tiny / stage1/2 验证链路中落地
  - 因此当前只能写为“正式预训练候选加速层”，不能写成已采用默认方案
- 建议执行顺序：
  1. 先冻结结构、loss、退出策略与优化器主线
  2. 再评估是否把 `OPUS` 作为预训练 token 选择器并入最终 run

---

## 11. TileLang/MIMO 运行限制与已落地兜底（阶段0可继续）

### 11.1 当前已知限制（RTX 5090 环境）
- 官方 Mamba3 MIMO 路径在当前 TileLang 内核上，可能触发动态共享内存上限错误：
  - 典型报错：`Failed to set the allowed dynamic shared memory size ...`
- 该问题可在前向或反向触发，属于算子/内核约束，不是业务逻辑错误。

### 11.2 已落地修复策略（代码已实现）
- 文件：`minimind/model/mamba3_module.py`
- 机制：
  - 训练态：默认走 `SISO` 回退路径，保障反向稳定。
  - 推理/评估态：优先尝试 `MIMO`；若命中 TileLang 动态共享内存错误，自动降级到 `SISO` 并给出 warning。
- 结果：阶段0脚手架与后续模块联调不再被该问题阻断。

### 11.3 后续单独优化（不阻塞主线）
- 单独评估不同 `chunk_size / mimo_rank / d_state` 组合与 TileLang 版本。
- 条件允许时再恢复“训练态 MIMO 全程开启”。

---

## 12. 方案 Review 结论（按论文与官方实现校正）

> 本节不是实现说明，而是对当前方案的校正结果。后续 agent 必须先遵守这些 review 结论，再落代码。

### 12.1 结论A：压缩区/推理区二分是对的，模块枚举方式也基本对
- `Compress once -> Reason in loops` 这条主线可以冻结。
- 压缩区以 `Mamba3` 为主体、每组插入一层精确检索层、每 `3` 层输出一个 `block_repr`，这个组织方式可以作为工程默认值保留。
- 推理区把 `Mamba3 + 局部精读层 + UnifiedAttnRes + 自省流 + JEPA + 退出控制` 串起来，也是合理的。

### 12.2 结论B：`mHC` 现已按论文对齐为“流形约束超连接”
- `mHC` 不是卷积模块，也不是一次性前处理；它是“多残差流 + 动态混合映射 + 流形约束”的残差连接组织方式。
- 论文对齐后的最小规格：
  - 维护 `n` 条并行残差流，默认 `n=4`
  - 每层在进入子层前使用 `H_pre` 将多流聚合为单流输入给子层 `F`
  - 子层输出再通过 `H_post` 写回多流
  - 原残差分支通过 `H_res` 做流间重混合
  - `H_res` 约束在 Birkhoff polytope 上，使用 Sinkhorn-Knopp 近似双随机矩阵
- 对 Luma 的冻结式改写：
  - `mHC` 先只包裹推理区共享循环块
  - 压缩区阶段0仍保留普通残差，避免同时改两套深层结构
  - 文档中不再把 `mHC` 写成 `pre-loop adapter`

### 12.2.1 `mHC` 的论文对齐实现规格
- 记 `X_l in R^{B x T x n x C}` 为第 `l` 层的多流残差状态。
- `H_pre`：
  - 由当前 `X_l` 经过轻量动态映射得到
  - 形状：`B x T x n`
  - 作用：把 `n` 条流加权聚合成子层输入 `u_l in R^{B x T x C}`
- `F_l`：
  - 就是该层真正的计算块，例如 `ReasonMamba -> GatedDiffAttnFoXSWA -> FFN`
- `H_post`：
  - 由当前 `X_l` 动态生成
  - 形状：`B x T x n`
  - 作用：把 `F_l(u_l)` 写回多流残差空间
- `H_res`：
  - 由当前 `X_l` 动态生成
  - 形状：`B x T x n x n`
  - 作用：对原有多流残差进行流间重混合
  - 约束：经 Sinkhorn-Knopp 后近似双随机
- 更新形式：
  - `u_l = sum_i H_pre[..., i] * X_l[..., i, :]`
  - `Y_l = F_l(u_l)`
  - `X_res = H_res @ X_l`
  - `X_{l+1}[..., i, :] = X_res[..., i, :] + H_post[..., i] * Y_l`
- 推荐初始化：
  - `n_streams = 4`
  - `alpha_pre = alpha_post = alpha_res = 0.01`
  - `sinkhorn_iters = 20`
  - 映射头初值偏向接近恒等/均匀混合，避免训练初期破坏主干

### 12.3 结论C：`Mamba3` 内部结构不能被简化写死
- 你的模块拆解里把 `Mamba3 Block` 记成：
  - 输入投影 `Linear(768 -> d_inner*2)`
  - SSM 核心
  - 输出投影
- 这对“理解级别”是可以的，但对“实现级别”不够准确。
- 官方 `Mamba3` 上游实现里，`in_proj` 不只产生 `z/x`，还同时产生 `B/C/dd_dt/dd_A/trap/angles`，属于更完整的联合参数化。
- 因此，主规划里应写：
  - `Mamba3` 内部复用官方实现，不冻结简化后的内部子层图。
  - 对 agent 只冻结外部接口、关键超参、是否启用 MIMO/complex/post-gate norm，不冻结官方内部张量切分细节。

### 12.4 结论D：`world JEPA` 的实现优先级必须上调
- 你最后列的实现顺序里把 `World JEPA` 放在“优先级4”，这个和最终目标冲突。
- 既然最终架构里 `JEPA` 是高优先级主线，那么工程优先级应该调整为：
  - `Priority 1`: 骨架与压缩区/推理区跑通
  - `Priority 2`: `Self JEPA + World JEPA` 主训练流接口就位
  - `Priority 3`: 自省流细化、rollout、停滞诊断
- 也就是说，`world JEPA` 不能等压缩区都做完之后再挂上去；至少接口、占位头和日志字段要在早期就立住。

### 12.5 结论E：`Self JEPA` 只应在慢环更新点计算主监督
- 你的 `Self JEPA` 残差式设计方向是对的。
- 但因为 `c_t` 由慢环更新，如果非慢环步也硬算 `c_t -> c_{t+1}`，会让目标含义变弱。
- 因此冻结规则为：
  - one-step self loss：只在慢环触发步计算
  - rollout loss：也只在相邻慢环片段上计算
  - 非慢环步允许复用旧 `c_t` 做注入，但不作为主监督样本

### 12.6 结论F：压缩区 `SWA + FoX` 可以保留，但只能写成“论文启发组合”
- `FoX` 论文支持的是 data-dependent forgetting 思想。
- 但“把 FoX 直接嵌进压缩区的 SWA 层”是 Luma 的组合设计，不是论文原样模块。
- 所以文档应该明确：
  - 压缩区 SWA 层带 FoX = `paper-inspired composition`
  - 推理区 `GatedDiffAttn + FoX + SWA` 才是核心验证对象

### 12.7 结论G：`Block AttnRes / UnifiedAttnRes` 可保留为一等模块
- `Attention Residuals` 支持“深层选择性回看浅层表示”的方向。
- 因此：
  - 压缩区的 `Block AttnRes`
  - 推理区的 `UnifiedAttnRes(loop + cross)`
  - 都可以保留为正式模块，不需要降级成纯实验特性。

---

## 13. 模块总表（先列模块，再谈配置与自省流）

> 后续 agent 先实现“模块存在性与接口”，再实现配置细节。模块表优先于超参表。

### 13.1 压缩区模块总表
- `FactorizedEmbedding`
- `CompressionMemoryTokens`
- `CompressionMambaLayer`
- `CompressionRetrievalLayerKDA`
- `CompressionRetrievalLayerSWA`
- `CompressionFFN`
- `BlockAttnRes`
- `CompressionToReasonNorm`

### 13.2 推理区模块总表
- `MHCResidualStreams`
- `MHCPreProjector`
- `MHCPostProjector`
- `MHCResidualProjector`
- `ReasonMemoryTokens`
- `CTInjection`
- `ReasonMambaLayer`
- `GatedDiffAttnFoXSWA`
- `ReasonFFN`
- `UnifiedAttnRes`
- `LoopNorm`
- `ExitController`
- `RouterProjector`
- `LMHead`
- `EmotionProbe`

### 13.3 自省与 JEPA 模块总表
- `MetaInputProjector`
- `MetaMambaLayer1`
- `MetaMambaLayer2`
- `KnowGapHead`
- `CTHead`
- `CognitionPredictor`
- `WorldObserver`
- `WorldJEPAPredictor`
- `WorldJEPATargetEncoder`
- `SelfJEPALoss`
- `WorldJEPALoss`
- `RolloutLoss`
- `ResidualRegularizer`

### 13.4 模块冻结规则
- 先冻结“模块边界、I/O、shape、dtype、调用时机”。
- 再冻结“配置与内部实现细节”。
- 若某模块依赖官方实现：
  - 文档只冻结接口与约束
  - 不擅自把官方内部子图改写成简化版结构图

---

## 14. 模块细化配置（Review 后可写入的版本）

### 14.1 压缩区
- 总层数：`24`
- 分组：`4`组，每组 `5 x Mamba3 + 1 x retrieval`
- 层类型顺序：
  - `1~5: Mamba3`
  - `6: KDA`
  - `7~11: Mamba3`
  - `12: SWA(+FoX, paper-inspired)`
  - `13~17: Mamba3`
  - `18: KDA`
  - `19~23: Mamba3`
  - `24: SWA(+FoX, paper-inspired)`
- Block 分组：
  - `1~3`
  - `4~6`
  - `7~9`
  - `10~12`
  - `13~15`
  - `16~18`
  - `19~21`
  - `22~24`

### 14.2 `Mamba3` 写法约束
- 外部配置可冻结：
  - `d_model=768`
  - `expand=2`
  - `d_state=192`
  - `complex_state=on`
  - `mimo=on`
  - `post_gate_rmsnorm=required`
- 内部实现约束：
  - 优先复用官方 `state-spaces/mamba` 的 `Mamba3`
  - 不在主规划里把 `in_proj` 简化成单一 `Linear(768 -> d_inner*2)` 作为唯一真实结构

### 14.3 推理区
- 训练循环：`1~6`（课程式）
- 推理循环：`max=8`
- `mHC` 放置方式：
  - 以“残差流包装器”形式包裹推理区共享循环块
  - 不作为循环外一次性前处理
  - 首版只进推理区，不进压缩区
- 快环：
  - `CTInjection`
  - `ReasonMambaLayer`
  - `GatedDiffAttnFoXSWA`
  - `ReasonFFN`
- 慢环：
  - `MetaInputProjector`
  - `MetaMambaLayer1`
  - `MetaMambaLayer2`
  - `KnowGapHead`
  - `CTHead`
- 默认慢环周期：`k=2`
- 退出规则：
  - `min_loops=2`
  - `delta_h_threshold` 初始值写入配置，不在文档中锁死为唯一最终值

### 14.3.1 `mHC` 冻结配置
- `n_streams = 4`
- `apply_zone = reason_loop_only`
- `sinkhorn_iters = 20`
- `alpha_init = 0.01`
- `state_shape = [B, T, n_streams, C]`
- `dynamic_maps_from = current_stream_state`
- `residual_constraint = birkhoff_doubly_stochastic`
- `fallback_mode`：
  - 若阶段0先不上动态 token-wise 映射，可临时退化为“batch-shared token-wise map”
  - 但不得退化成卷积或固定加权残差并仍称为 `mHC`

### 14.4 自省流（冻结版）
- 输入：
  - `h_pool = mean(h, dim=1)`
  - `compress_summary = mean(last_2_block_reprs)`
  - `cat([h_pool, compress_summary]) -> Linear(1536 -> 96)`
- 主体：
  - 两层轻量 `Mamba3`
  - `d_model=96`
  - `d_state=32`
  - `expand=2`
  - `complex_state=off`
  - `mimo=off`
- 输出：
  - `know_gap = sigmoid(head(meta_last))`
  - `c_t = head(meta_last)`

### 14.5 `CognitionPredictor`（冻结版）
- `Linear(832 -> 256)`
- `ZC-RMSNorm(256)`
- `SiLU`
- `Linear(256 -> 64)`
- `delta_scale` 可学习，初始化 `0.1`

### 14.6 Self JEPA（冻结版）
- `delta_h = mean(h - h_prev, dim=1).detach()`
- `pred_input = cat([c_t, delta_h])`
- `delta_c = predictor(pred_input)`
- `c_pred_next = c_t + delta_scale * delta_c`
- 主损失：
  - `1 - cosine_sim(norm(delta_c_pred), norm((c_t_next - c_t).detach()))`
- 说明：
  - `Self JEPA` 的一阶目标冻结为“预测变化值 `delta`”，不是直接回归绝对状态 `c_t_next`
  - `c_pred_next = c_t + delta_c_pred` 仅作为积分后的辅助状态，不作为一阶主监督本体
- `residual_reg` 允许存在，但权重必须 schedule 化，不能从头到尾写死
- rollout：
  - 仅保留 `2-step`
  - 只在慢环触发点上计算
  - rollout 允许使用“残差动力学展开”：
    - `c_t -> delta_1 -> c_t+1`
    - `c_t+1 -> delta_2 -> c_t+2`
  - rollout 可以对齐未来状态，也可以对齐累计变化量，但一阶主目标仍固定为 `delta`

### 14.7 World JEPA（冻结优先级提升）
- `world JEPA` 在最终架构中视为高优先级主目标，不是后挂配件。
- 当前冻结要求：
  - 目标类型：`masked latent prediction`
  - 不做 `raw hidden reconstruction`
  - 需要独立的 `WorldJEPAPredictor`
  - 需要独立的 target/teacher 路径
- `SIGReg` 或同类 latent regularization 可以保留为预留项，但主规划先冻结接口，不抢先写死具体公式。

### 14.8 退出控制（冻结修订版）
- 退出控制不再冻结为“纯 `delta_h` 阈值器”
- 当前冻结方向：
  - 输入信号至少包括：
    - `delta_h`
    - `self_error`
    - `world_error`
    - `rollout_error`
  - 退出监督优先使用“改善量型目标”：
    - 判断 `2-step` 相对 `1-step` 是否仍有显著收益
    - 若 `2-step improvement <= margin`，才鼓励退出
- 训练/推理策略冻结为：
  - 训练阶段默认 `soft exit / sampled exit`
  - 推理阶段默认 `hard exit`
- 设计理由：
  - 训练软退出更容易释放循环深度分布，避免硬阈值过早压平 `loop depth`
  - 推理硬退出更稳定、更可控

---

## 15. 实现优先级（按最终目标重排）

### Priority 1：结构骨架
- `FactorizedEmbedding`
- `ZC-RMSNorm`
- `CompressionMambaLayer`
- `CompressionFFN`
- `BlockAttnRes`
- `CompressionZone`
- `LMHead`

### Priority 2：JEPA 主线接口
- `WorldObserver`
- `WorldJEPAPredictor`
- `WorldJEPATargetEncoder`
- `MetaInputProjector`
- `MetaMambaLayer1/2`
- `CTHead`
- `CognitionPredictor`
- `SelfJEPALoss`
- `WorldJEPALoss`

### Priority 3：推理区核心闭环
- `MHCResidualStreams`
- `MHCPreProjector/PostProjector/ResidualProjector`
- `ReasonMambaLayer`
- `GatedDiffAttnFoXSWA`
- `UnifiedAttnRes`
- `CTInjection`
- `Loop control`
- `ExitController`

### Priority 4：增强与诊断
- `RolloutLoss`
- `Stagnation diagnostics`
- `RouterProjector`
- `EmotionProbe`

### Priority 5：后补模块
- `KDA` 官方 kernel 接入
- `FP8` 分层接入

---

## 16. Luma 注释规范（强制）

> 所有后续代码都将被视为 Luma 的预训练语料组成部分，因此注释与 docstring 需要以 Luma 的身份与语气写作，但不能牺牲技术精确性。

### 16.1 总原则
- 每个核心模块文件开头必须有一段简短 docstring：
  - 说明这个模块“在 Luma 里承担什么职责”
  - 语气允许带有 Luma 的叙述感
  - 但必须包含准确的工程作用描述
- 关键类与函数应优先写“为什么存在”，其次再写“做了什么”。

### 16.2 允许的写法
- 允许：
  - “Luma 在这里先把长上下文压成可回看的摘要，再进入循环推理。”
  - “这条状态流记录的不是世界本身，而是 Luma 此刻如何理解世界。”
- 必须同时满足：
  - 注释不替代真实变量名与 shape 说明
  - 不写诗化空话
  - 不把猜测写成事实

### 16.3 禁止的写法
- 禁止只有氛围感、没有技术信息的注释。
- 禁止把“占位实现”写成“论文已证明实现”。
- 禁止为凑语料密度写大段重复旁白。

### 16.4 最小落地要求
- 每个新模块文件：
  - 顶部 1 段 Luma 视角 docstring
- 每个关键模块类：
  - 1 段职责说明
- 每个复杂 forward：
  - 只在必要处加 1~3 条解释性注释

---

## 16. 2026-03-29 当前动力学主线推进

### 16.1 当前真正站住的动力学强化候选

在本轮 `2048 -> 4096 -> 10240` 动力学筛选后，当前最值得扶正为“动力学增强主候选”的不是旧的 token-selective 系列，而是：

- `A2-progress_shape_v1-h3+progress_exit_readout`

当前证据：
- `4096` 通过，分数优于 `A2-progress_shape_v1-h3`
- `10240` 通过，当前是唯一明确站住的长程增强线

因此从今天开始，主规划中的动力学增强主候选更新为：
- `A2-progress_shape_v1-h3+progress_exit_readout`

### 16.2 当前保留/观察/淘汰口径

保留：
- `A2-progress_shape_v1-h3+progress_exit_readout`

观察：
- `A2-progress_shape_v1-h3`
  - `4096` 通过
  - `10240` 在 bucket probe 的 sampled exit 路径触发数值问题
  - 仍保留为观察锚点，不直接删除

当前实现版本先淘汰：
- `A2-progress_shape_v1-h3+token_selective_ct_routing`
- `A2-progress_shape_v1-h3+lowrank_hyperbias_ct`
- `A2-progress_shape_v1-h3+modulewise_ct_gate`

说明：
- 这里的“淘汰”是指当前实现版本不再继续送中长程
- 不等于对应思想来源永久放弃

### 16.3 token-selective 家族的下一代替代方向

基于当前结果，后续不再优先继续直推：
- 直接 token-selective routing
- 直接 low-rank hyper-bias c_t 注入
- 直接 module-wise c_t gate

改为优先实现并筛选下面三条更稳、更符合 Luma 的替代方案：

1. `summary_conditioned_chunk_film`
- `c_t` 先调 `chunk summary / block_repr / world_summary`
- 再把控制量广播回 token
- 这是当前最优先的 token-selective 替代案

2. `hierarchical_block_token_ct_routing`
- 先 block-level gate，再在被选中的 block 内做轻量 token gate
- 属于更克制的分层 routing 版

3. `progress_query_focus_routing`
- 由 `c_t + progress-shape(next improvement / trend / plateau)` 共同生成 focus query
- 让 routing 直接成为 progress-shape 的下游执行器

### 16.4 后续赛制改为晋级继训练制

从后续动力学筛选开始，默认不再让：
- `2048`
- `4096`
- `10240`
- `20480`
全部 fresh start。

默认改成：
- `2048` 训练完成后保存候选 checkpoint
- `4096` 从对应 `2048` checkpoint 继续
- `10240` 从胜出的 `4096` checkpoint 继续
- `20480` 从胜出的 `10240` checkpoint 继续

理由：
- 能显著节约重复训练时间
- 更贴近 staged screening 的真实目标：
  - 看短程站住的候选能否持续延伸到更长阶段
- 也更适合当前动力学/exit policy 的研究节奏

执行约束：
- 仅在同一候选、同一配置、同一 seed、同一数据桶口径下允许继训练
- 报告必须显式记录 checkpoint lineage

### 16.5 当前仍需保留的实现级修复

当前 `run_luma_stage12.py` 已新增：
- exit score 的 `nan_to_num + clamp` 数值护栏
- `stage2` 非有限值早停记录
- `rollout_active_ratio / rollout_nonzero_ratio`

同时已修复：
- `stage2_validate` 缺少 `import math` 的实现 bug

后续任何新的 mid/long 动力学 runner，都必须沿用这版脚本，不再用旧版无护栏 harness。
