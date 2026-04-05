# Luma 研究报告：小模型变聪明 + 无反向传播训练

> 2026-04-05 | 双方向调研汇总

---

## 一、让小模型 (<1B) 更聪明

### Tier 1: 最高影响力（推荐立即采用）

#### 1. 循环深度 + 潜空间推理 (Recurrent Depth Transformers)
- **论文**: "Scaling up Test-Time Compute with Latent Reasoning" — Geiping et al., 2025 (NeurIPS 2025 Spotlight)
- **核心**: 一个可循环执行的 recurrent block（如 4 层），推理时展开到任意深度。3.5B 模型 × 32 次迭代 = 132 层等效深度。**不需要 CoT token，不需要特殊训练数据。**
- **对 Luma 的意义**: Luma 的 CDR-Loop 已经是循环架构！reason_shared_depth=2 + loops=12 正是这个思路。可以进一步增大循环深度比（更少唯一层 × 更多循环），用 ~200M 唯一参数达到 500M 等效。
- **Mamba3 兼容性**: ★★★★★ — SSM 天然是循环结构
- **代码**: [github.com/seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)

#### 2. 合成持续预训练 (EntiGraph)
- **论文**: "Synthetic Continued Pretraining" — Yang & Band, 2024 (ICLR 2025 Oral)
- **核心**: 从小语料提取实体，通过实体交叉连接合成大量多样化文本。从 1.3M tokens 生成 600M tokens。不是简单 paraphrase（论文证明 paraphrase 很快饱和）。
- **对 Luma 的意义**: 直接用于 persona/empathy 数据扩充 — 从少量高质量人格数据合成大量训练语料，符合"先变聪明再变像 Luma"原则。
- **Mamba3 兼容性**: ★★★★★ — 纯数据技术，架构无关

#### 3. 推理时 A* 搜索 (Test-Time A* Search)
- **论文**: "Test-Time Scaling for Multistep Reasoning in SLMs via A* Search" — 2025
- **核心**: 将推理建模为树搜索，用 A* 代价函数引导。**Llama-1B + TTS 超越 8B，3B 超越 70B**。无需训练、无需外部监督、drop-in 解码器。
- **对 Luma 的意义**: 部署时免费提升推理质量。但 SSM 的树搜索需要保存/恢复状态，实现比 Transformer 稍复杂。
- **Mamba3 兼容性**: ★★★☆☆ — 需要 SSM state checkpoint 机制

#### 4. 数据质量课程 (SmolLM2 / Phi 策略)
- **论文**: SmolLM2 (HuggingFace, 2024), Phi-3/MiniCPM
- **核心**: 质量课程（先通用后高质量）+ perplexity 数据修剪（**30% 数据匹配全量性能**）+ 模型级过滤（**15% token 匹配 MMLU**）
- **对 Luma 的意义**: 当前 61M tokens 可能有大量低效数据。先做 perplexity 过滤，再做质量分层。
- **Mamba3 兼容性**: ★★★★★ — 纯数据/训练策略

### Tier 2: 高影响力

#### 5. DistiLLM — 偏斜 KL 蒸馏
- **论文**: Ko et al., ICML 2024
- **核心**: Skew KL divergence 解决 teacher-student 分布差异大时的梯度不稳定，比 MiniLLM 快 4.3 倍。
- **Mamba3 兼容性**: ★★★★★

#### 6. GaLore — 梯度低秩投影
- **论文**: Zhao et al., 2024; GaLore 2, 2025
- **核心**: 梯度矩阵低秩投影，减少 65.5% 优化器内存。24GB GPU 预训练 7B。
- **对 Luma**: 可能允许在 32GB 卡上跑更大 batch。
- **Mamba3 兼容性**: ★★★★★ — 优化器级技术

#### 7. IBM Granite 4.0 — 9:1 SSM/Attention 混合比
- **论文**: IBM Granite 4.0, 2025
- **核心**: 9 个 Mamba block : 1 个 Transformer block，线性复杂度 + 选择性全局注意力。Nemotron-H 用 92% SSM 替换 attention，吞吐量 3 倍。
- **对 Luma**: Luma 的压缩区已经是类似设计（Mamba 为主 + 少量 SWA/KDA），可参考比例优化。

#### 8. 渐进式训练 (Progressive Training)
- **论文**: Apollo, 2024
- **核心**: 先训浅网络再扩展到目标深度，低值优先采样 + 权重共享。
- **对 Luma**: 可以先训 22 层模型，再通过层复制/插值扩展到 44 层。

### Tier 3: 值得关注

| 技术 | 核心思路 | 兼容性 |
|------|---------|--------|
| CoT 课程蒸馏 | 教师 CoT + 从易到难 | ★★★★★ |
| Dual-Space 蒸馏 | logit + hidden state 同时蒸馏 | ★★★☆☆ |
| Inner Thinking Transformer | 每 token 自适应深度 | ★★★★☆ |
| 序列长度课程 | 短→长渐进训练 | ★★★★★ |

---

## 二、无反向传播训练

### 核心发现

**2025 年是无 BP 训练的突破年。** 进化策略 (ES) 已经能在 LLM 规模上与梯度方法匹敌甚至超越。这不再是理论讨论 — 三组独立团队在 2025 年验证了这一点。

### Tier 1: 立即可用

#### 1. ES at Scale — 进化策略全参数微调 14B 模型
- **论文**: "Evolution at Scale" — Lange et al., ICML 2025
- **核心**: 用进化策略（ES）全参数微调 14B 模型。在 Countdown / Sudoku / ARC-AGI 上**大幅超越 GRPO（梯度强化学习）**。仅需 N=30 种群，训练稳定性比 GRPO 高 **15.5 倍**。
- **规模**: 14B 全参数微调
- **vs BP**: **超越** GRPO baseline（在推理任务上）
- **VRAM**: 仅需前向传播 × N 个扰动，可并行
- **Mamba3 兼容性**: ★★★★★ — 只需前向传播，架构完全无关
- **代码**: 已开源

#### 2. EGGROLL — 进化策略预训练整数语言模型（在 RWKV7 上验证）
- **论文**: "EGGROLL: Evolving GPU-native bitwise Recurrent Operations for LLMs" — NVIDIA + Oxford, 2025
- **核心**: 用 ES 从零预训练**纯整数**语言模型。在 **RWKV7（线性循环模型，与 SSM/Mamba 同族）** 上直接验证，达标准推理吞吐量的 91%。
- **规模**: RWKV7 架构预训练
- **vs BP**: 91% 推理吞吐量
- **VRAM**: 极低（整数运算 + 无梯度）
- **Mamba3 兼容性**: ★★★★★ — **直接在 SSM 同族架构上验证过**，最相关的工作
- **代码**: 开源

#### 3. ESSA — 进化策略 + 自适应 32B 微调
- **论文**: "ESSA: Evolutionary Strategy with Stochastic Adaptation" — 2025
- **核心**: 32B 模型用 INT4 推理级显存即可微调，比 GRPO 收敛快 **2-6 倍**，仅需 100 个样本。
- **规模**: 32B 微调
- **vs BP**: 收敛更快，最终性能可比
- **VRAM**: INT4 推理级（32B 模型仅需 ~16GB）
- **Mamba3 兼容性**: ★★★★★

#### 4. MeZO 家族 — 零阶优化
- **论文**: "Fine-Tuning Language Models with Just Forward Passes" — Malladi et al., 2023; Sparse MeZO (2024), AGZO (2025)
- **核心**: 基于 SPSA（同时扰动随机近似），仅用两次前向传播估算梯度。30B 级 fine-tune，显存仅等同推理。
- **规模**: 30B 微调
- **vs BP**: 在 11/15 任务上匹配 full fine-tune
- **VRAM**: 等同推理（无激活缓存、无梯度）
- **Mamba3 兼容性**: ★★★★★ — 仅需前向传播
- **代码**: [github.com/princeton-nlp/MeZO](https://github.com/princeton-nlp/MeZO)

### Tier 2: 有前景但规模有限

#### 5. NoProp — 扩散去噪替代传播
- **论文**: "NoProp: Training Neural Networks without Back-propagation or Forward-propagation" — 2025
- **核心**: 将隐藏层表征视为含噪样本，通过扩散去噪学习。每层独立训练，天然可并行。
- **规模**: 仅 CIFAR 验证
- **vs BP**: CIFAR 上接近
- **Mamba3 兼容性**: ★★★☆☆ — 概念新颖但未在序列模型上验证

#### 6. Mono-Forward — 单向前传播训练
- **论文**: Kohan et al., 2025
- **核心**: 用信号传播理论，仅前向传播更新权重。MLP 上**超越 BP**，能耗降 41%。
- **规模**: MLP 级别
- **vs BP**: 在 MLP 上超越
- **Mamba3 兼容性**: ★★☆☆☆ — 未扩展到复杂架构

#### 7. Predictive Coding — 预测编码
- **论文**: Millidge et al., 2024; iPC (2025)
- **核心**: 生物启发的局部误差信号，每层仅需局部预测误差更新。ResNet-18 上追平 BP。
- **规模**: ResNet-18
- **vs BP**: 接近
- **Mamba3 兼容性**: ★★★☆☆ — 理论上适配但工程挑战大

### Tier 3: 硬件/理论驱动

| 方法 | 核心 | 规模 | 相关度 |
|------|------|------|--------|
| 光学 DFA | 1.3B Transformer 光学训练 (Nature 2024) | 1.3B | 需光学硬件 |
| Equilibrium Propagation | 能量最小化，适合模拟硬件 | 小规模 | 低 |
| Forward-Forward (Hinton) | 正负样本对比，逐层训练 | MNIST/CIFAR | 低 (未扩展到 LLM) |

---

## 三、综合建议：Luma 的两条腿

### 路线 A：小模型变聪明（短期，可立即执行）

**推荐组合**（按优先级）：

| 优先级 | 技术 | 对 Luma 的具体应用 | 预计收益 |
|--------|------|-------------------|---------|
| P0 | 循环深度扩展 | 增大 reason_loops（12→20+），减少唯一层数 | 等效 2-3x 深度 |
| P0 | 数据质量课程 | perplexity 过滤 + 先通用后高质量 | 30% 数据 = 100% 效果 |
| P1 | EntiGraph 合成 | 从 persona/empathy 小语料合成 10x 训练数据 | 数据量 10x |
| P1 | GaLore | 替换当前优化器，释放 VRAM | 可能 bs=2 |
| P2 | 推理时 A* 搜索 | 部署时 drop-in，500M 逼近 2B 推理质量 | 推理 4x |
| P2 | DistiLLM 蒸馏 | 从 Qwen-72B 蒸馏到 Luma | 知识注入 |

### 路线 B：无反向传播训练（中期，需要实验验证）

**推荐路线图**：

```
Phase 1 (1-2 周): MeZO fine-tune 验证
  └─ 在当前 482M Luma 上用 MeZO 做 fine-tune
  └─ 对比 BP fine-tune 的效果差距
  └─ VRAM 应该降到推理级 (~2GB)

Phase 2 (2-4 周): ES 微调验证
  └─ 实现 ES at Scale 的 antithetic sampling
  └─ 在 Luma 上做推理任务微调 (math, code)
  └─ 对比 MeZO 和 BP baseline

Phase 3 (探索性): EGGROLL 式预训练
  └─ 参考 EGGROLL 在 RWKV7 上的成功
  └─ 尝试用 ES 从零预训练小 Luma (100M)
  └─ 如果可行，逐步扩大到 482M
```

**为什么进化策略最适合 Luma**：
1. Mamba3 SSM 是循环架构，与 EGGROLL 验证的 RWKV7 同族
2. 仅需前向传播 → VRAM 从训练级降到推理级
3. ES 在推理任务上已超越梯度 RL (GRPO) → 与"让模型变聪明"目标一致
4. 天然支持非可微组件（如离散 exit decision）
5. 训练稳定性高 15.5 倍 → 解决当前 OOM/crash 问题

### 风险评估

| 路线 | 风险 | 最坏情况 | 缓解 |
|------|------|---------|------|
| A (小模型变聪明) | 低 | 收益不如预期 | 每项技术独立，可逐个验证 |
| B (无 BP) | 中-高 | 预训练不收敛 | 先从 fine-tune 验证，渐进式 |
| A+B 组合 | 中 | 研究周期延长 | 并行推进，A 不依赖 B |

---

## 四、关键论文索引

### 小模型变聪明
1. Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning" (NeurIPS 2025)
2. Yang & Band, "Synthetic Continued Pretraining / EntiGraph" (ICLR 2025 Oral)
3. "Test-Time A* Search for SLMs" (2025)
4. SmolLM2 (HuggingFace, 2024)
5. Ko et al., "DistiLLM" (ICML 2024)
6. Zhao et al., "GaLore / GaLore 2" (2024-2025)
7. IBM Granite 4.0 Hybrid Architecture (2025)

### 无反向传播训练
1. Lange et al., "Evolution at Scale" (ICML 2025) — **14B ES 微调超越 GRPO**
2. "EGGROLL" — NVIDIA+Oxford, 2025 — **RWKV7 ES 预训练**
3. "ESSA" — 2025 — **32B ES 微调，INT4 显存**
4. Malladi et al., "MeZO" (NeurIPS 2023) + Sparse MeZO, AGZO 后续
5. "NoProp" — 2025 — 扩散去噪替代传播
6. "Mono-Forward" — 2025 — 单向前传播超越 BP (MLP)
7. Millidge et al., "Predictive Coding" (2024)
