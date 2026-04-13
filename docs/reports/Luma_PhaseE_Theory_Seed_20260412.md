# Luma Phase E — 能量梯度流推理架构 · 理论种子

**日期**: 2026-04-12
**状态**: 种子稿（seed）— 随 Phase E 实验同步演化，**严禁事后改写实验前的推论**
**作者**: Claude + 用户联合设计
**前序文档**: `Luma_Crisis_20260412.md`, `Luma_Theory_Grounding_20260412.md`

---

## 0. 为什么要有这份文档

Phase 2 用十次 V 系列实验验证了一件事：
> 当前 CR-Loop `h_{k+1} = h_k + Δ(h_k, c_t, k)` 是一个 **marginal stable** 的显式迭代，ρ(∂F/∂h) 恒在 1 附近徘徊，必须靠 `ct_inj_max`、`max_ct_norm`、`force_adamw` 白名单、低 wd 补丁等**主动反馈控制**才能稳定。这是托卡马克等离子体的等价物：用磁场约束不稳定流体。

用户在 2026-04-12 下午做出架构决策：
> **"我希望我的模型不是一个托卡马克 而是更像一个仿星器 通过设计模型让梯度自然成形"**

本文档是 Phase E 的理论起点 — 把"仿星器"从哲学隐喻翻译成可执行的架构规范。**同步建设**原则：理论先行于实验、但实验数据立刻回喂理论。每次实验后，这份文档的相应章节会被标记为 "✅ 验证" 或 "❌ 证伪"，绝不改原文。

---

## 1. 目标动力系统的形式化

### 1.1 我们想要什么（用户原话 + 数学翻译）

| 用户原话 | 数学翻译 |
|---|---|
| "自洽" | 系统定态由内部规则决定，不由外部 clamp 强制 |
| "动力学" | h 的更新是某个连续/离散时间演化 |
| "认知流体" | h 在表征空间中有**流形结构**，轨迹具方向性 |
| "c_t 不是人格而是地基" | c_t 是**慢变量**，塑造 h 的演化规则 |
| "不动点太死板" | h 不必收敛到零维点；**高维吸引子（亚稳态/极限环/NESS）也算自洽** |
| "梯度自然成形" | 训练目标的梯度场本身**蕴含动力学的稳定性** |

### 1.2 吸引子维度的谱

按吸引子维度从低到高：

| 维度 | 类型 | 认知解释 | Phase E 态度 |
|---|---|---|---|
| 0 | 点吸引子 | "唯一答案" | 默认，但不是终点 |
| 1 | 极限环 | "节奏性思考"（呼吸式迭代） | 允许 |
| 2 | 环面/拟周期 | "多尺度节奏嵌套" | 允许 |
| fractal | 奇怪吸引子 | "混沌但有界的探索" | 允许 |
| 距离化的 | 亚稳态链 | "候选答案间跳跃" | **强期待** |
| 连续 | 非平衡稳态 (NESS) | "永远在流动的稳态" | **理想目标** |

**设计约束**：Phase E 架构必须**允许**高维吸引子，但**不禁止**点吸引子（简单任务应自然落到点）。这是通过**温度调度**（Langevin 热噪声）自然实现的 — T > 0 时分布非退化，T → 0 时退化为点。

### 1.3 正式问题陈述

给定输入 x、隐状态 h ∈ R^D、慢变量 c_t ∈ R^{d_c}，设计可学习的标量能量函数：

```
E : R^D × R^{d_c} × X → R_{≥0}
```

以及迭代规则：

```
h_{k+1} = h_k - η_k · ∇_h E(h_k; c_t, x) + σ_k · ξ_k,    ξ_k ~ N(0, I)
```

使得：
1. **构造性收缩**（至少局部）：∃ T_c 使得当 σ ≤ σ(T_c) 时，系统收敛到 Boltzmann 分布 p(h) ∝ exp(-E/T)
2. **c_t 调制地貌**：c_t 改变 E 的形状，不改变 h 的更新规则
3. **自适应深度涌现**：停止条件 `||∇_h E|| < ε` 让简单 token 在 1-2 步内停，困难 token 自然跑多步
4. **托卡马克补丁归零**：不需要 ct_inj_max / max_ct_norm / force_adamw 等主动控制，系统自然保持在可解域

这是完整的仿星器契约。

---

## 2. 与文献的精确定位

2026-04-12 文献搜索（agent 调研）发现以下直接相关工作：

### 2.1 直接前身：Energy-Based Transformers (EBT)

> Gladstone et al., *"Energy-Based Transformers are Scalable Learners and Thinkers"*, arXiv:2507.02092, 2025

- 做到了 **800M 参数**的能量式 LM 预训练（当前能量式 LM 最大规模）
- 前向传播**正是**梯度流：给 (input, candidate) 对打分 E(input, candidate)，对 candidate 做 `candidate ← candidate - η∇E`
- 训练开销：Transformer++ 的 3.3-6.6×
- **没有** c_t 这种 slow-context 调制变量
- **没有** gradient-norm early stopping

**Luma 的差异**（可写进论文 "delta" 章节）：
1. c_t-conditioned 能量地形（E 依赖于外部慢变量）
2. 梯度范数自适应停止（不是固定迭代次数）
3. Langevin 松弛（EBT 是确定性梯度流，Luma 加噪声）
4. 规模：293M（比 EBT 小 2.7×，但在同一阶段合理）

### 2.2 概念最近邻：∇-Reasoner

> *"LLM Reasoning via Test-Time Gradient Descent in Latent Space"*, OpenReview 2025

- **推理时**在 latent 空间做梯度下降（LLM likelihood + reward model 梯度）
- 不是训练架构，是**现成 LLM 的 test-time patch**
- 证明了"forward pass = gradient descent in latent space"这一观点可行

**对 Luma 的意义**：把"推理 = 梯度下降"从 test-time 提升到 train-time，是 Luma 的核心创新点。

### 2.3 Langevin 推理的近亲：LangevinFlow

> *"Langevin Flows for Modeling Neural Latent Dynamics"*, arXiv:2507.11531, 2025

- latent 时间演化服从**欠阻尼 Langevin**（含惯性项 + 阻尼 + 学到的势能 + 随机力）
- 势函数参数化为**耦合振子网络** — 天然包含极限环/振荡结构
- 规模：neuroscience benchmark，非 LM

**对 Luma 的意义**：
1. 证明 Langevin 推理在表征学习是 viable 的
2. "耦合振子势"是允许非点吸引子的具体参数化范例
3. **如果 Luma 把 Langevin 加到 LM 规模的前向传播，是 LM 领域首创**

### 2.4 对"不动点太死板"的理论后盾

> *"Self-orthogonalizing attractor neural networks from the free energy principle"*, arXiv:2505.22749, 2025

- 在 FEP（Free Energy Principle）下推导吸引子网络
- 顺序数据 → **非对称耦合** → **非平衡稳态**动力学（非点吸引子）
- 明确论证：序列建模**不应**强制收敛到固定点，NESS 更自然

**对 Luma 的意义**：这篇论文**直接支持用户的直觉**（不动点太死板）。Phase E 论文可引用作为"非收敛是 feature 不是 bug"的理论背书。

### 2.5 全景综述

> *"Energy-Based Dynamical Models for Neurocomputation, Learning, and Optimization"*, arXiv:2604.05042, 2026

- 统一能量地形 + 梯度流 + 不同类型吸引子（fixed-point, limit-cycle, other）的综述
- 明确区分吸引子类型对应的计算能力

**对 Luma 的意义**：论文 Related Work 章节的总览锚点。

### 2.6 对照组（非能量但规模相近）

- **Inner Thinking Transformer (ITT)** arXiv:2502.13842 — token 级别 learned router 做自适应深度，Luma 的 MoR 类似
- **Liquid Reasoning Transformers** arXiv:2512.12792 — recurrent reasoning + learned stopping
- **Retrofitted Recurrence** arXiv:2511.07384 — 把预训练 LM 改造成迭代推理
- **C-DEQ** arXiv:2602.03024 — Consistency distillation 加速 DEQ，Route 2 的最新进展
- **Modern Hopfield in Transformer** arXiv:2511.20698 — Hopfield 视角能 scale 到 GPT 级

这些都是 Luma Phase E 论文中的 **comparison baseline 候选**。

### 2.7 文献空白（Luma 的独特性定位）

搜索失败的方向（说明 Luma 进入的是**未被占领的交叉区**）：

- ❌ "slow context variable modulating energy landscape at LM scale" — 0 hit
- ❌ "heteroclinic chain reasoning transformer" — 0 hit
- ❌ "gradient-norm stopping at LLM inference loop" — 0 hit（只有训练期的 GradES）

这三个空白正是 Luma 的贡献点。

---

## 3. Phase E 架构规范（V1.0 种子）

### 3.1 核心公式

```
h_{k+1} = h_k - η_k · ∇_h E(h_k; c_t, x) + √(2 η_k T_k) · ξ_k
```

其中：
- **η_k**：步长（可以是固定常数，也可以按 loop 衰减）
- **T_k**：温度（按 loop 衰减，模拟退火；k=0 时 T 高鼓励探索，k=K 时 T 低收敛到答案附近）
- **ξ_k ~ N(0, I)**：各向同性高斯噪声

当 T_k ≡ 0，这退化为确定性梯度下降（Route 3 naive 版本）。当 T_k > 0，这是 Langevin SGLD，采样 p(h) ∝ exp(-E/T)。

### 3.2 能量函数的候选参数化

**方案 A（最简）— "transformer 层反向构造"**：
```python
def compute_energy(h, c_t, x):
    h_next = shared_transformer_layer(h, c_t, x)   # 原有的 layer 前向
    return 0.5 * ((h - h_next) ** 2).sum()          # "预测残差的平方"
```
- 优点：直接复用现有 shared_layers 参数
- 缺点：E 不一定是严格凸的；Hessian 难保证正定
- ∇_h E = h - h_next + (∂h_next/∂h)^T (h_next - h) ≈ h - h_next（当 ∂h_next/∂h 小）
- 所以 h_{k+1} ≈ h_k - η (h_k - h_next) = (1-η) h_k + η h_next
- **这正是 α 门控残差**，但从能量框架导出，不是拍脑袋加 α

**方案 B（Hopfield）**：
```python
def compute_energy(h, c_t, x):
    memory = memory_table(c_t)                         # [M, D]，c_t 决定的记忆槽
    attention_term = -torch.logsumexp(beta * h @ memory.T, dim=-1)
    self_term = 0.5 * (h ** 2).sum() / alpha
    return attention_term + self_term
```
- 这是 Modern Hopfield Network 的标准能量
- **闭式梯度**：∇_h E = -β · memory^T · softmax(β · h @ memory.T) + h/α
- 更新规则：`h ← (1/α) · memory^T · softmax(β · h @ memory.T)` （近似）
- 就是 **attention with c_t-conditioned memory**
- 优点：有严格收敛证明（Ramsauer 2020）；能量有下界；容量指数级
- 缺点：表达能力受记忆槽数量限制

**方案 C（自由形式 MLP + Jacobian 正则）**：
```python
def compute_energy(h, c_t, x):
    return energy_mlp(torch.cat([h, c_t_broadcast, x_pooled], dim=-1)).sum()
```
加辅助 loss：
```python
hessian_trace_penalty = λ · (estimated_hessian_trace - target)**2
```
- 优点：最灵活
- 缺点：难保证凸性，训练不稳定风险高

**选择原则**：Step 1 先用方案 A（最简、复用现有参数），验证骨架能跑。Step 2+ 根据实验决定是否换 B 或 C。

### 3.3 温度调度 T_k

三个候选：
- **线性**：T_k = T_0 · (1 - k/K)
- **指数**：T_k = T_0 · γ^k
- **可学习**：T_k = softplus(θ_k)，θ_k 作为参数

**初始选择**：指数衰减，T_0 = 1.0，γ = 0.5。这样 k=0 时温度高（探索），k=5 时 T ≈ 0.03（基本收敛）。

**后续**：观察 Luma 在不同 T 下的行为，根据实验数据决定。

### 3.4 c_t 的时间尺度（关键设计）

**决定**：c_t **不**在 loop 轴演化。它在 **token 轴**（或 sequence 轴）演化。

具体：
- 一次前向传播开始时，c_t 从上一个 token 的 c_t 继承（可加 EMA 衰减）
- 整个 loop 迭代过程中，c_t **冻结**，E 的地貌不变
- h 的 Langevin 流在冻结地貌上运行 K 步
- K 步结束后，h* 到达谷底附近
- 用 h* 更新 c_t：`c_t_next = ρ · c_t + (1 - ρ) · g(h*)`，ρ ≈ 0.9-0.99

这样：
- h 的快动力学（loop 轴，微秒）和 c_t 的慢动力学（token 轴，毫秒）彻底解耦
- token_depth_routing 变不变 loop 数都不影响 c_t 的训练信号
- **奇异摄动理论**适用 — h 在 c_t 近似冻结的前提下分析，天然有双时间尺度

### 3.5 自适应深度 — 梯度范数 early stopping

替换 `TokenDepthRouter`：

```python
for k in range(K_max):
    E_k = compute_energy(h, c_t, x)
    grad_h = torch.autograd.grad(E_k.sum(), h, create_graph=True)[0]
    h = h - eta * grad_h + noise(T_k)
    if grad_h.norm(dim=-1).max() < eps_stop and T_k < T_floor:
        break
```

- 简单 token 的 ||∇E|| 很快就小（平坦谷底）→ 自动停
- 困难 token 的 ||∇E|| 保持大（陡峭谷壁）→ 自动跑满
- **不需要学习 router，不需要 balance loss**
- 物理可解释：能量梯度就是"未解明的紧张度"

### 3.6 Hessian 谱约束（构造性收缩的理论保证）

要保证 `ρ(∂F/∂h) < 1`，其中 `F(h) = h - η ∇_h E`：
```
∂F/∂h = I - η · ∂²E/∂h²
```
只要 `0 < η · λ_min(∇²E)` 且 `η · λ_max(∇²E) < 2`，则 ρ < 1。

**辅助 loss**：
```python
# 估算 Hessian trace via Hutchinson trick
v = torch.randn_like(h)
grad = torch.autograd.grad(E, h, create_graph=True)[0]
Hv = torch.autograd.grad((grad * v).sum(), h, create_graph=True)[0]
hessian_trace_est = (v * Hv).sum() / v.numel()
# 约束 trace 为正（半正定 proxy）
hessian_positive_loss = F.relu(-hessian_trace_est).mean()
```

这个 loss 的权重是 Phase E Step 4 要调的超参数。

### 3.7 Probe 重写

现有 `measure_theory_probes` 只测 `shared_layers` 子集的 Jacobian。Phase E 里要改成测**完整的** `F(h) = h - η∇E`：

```python
def measure_phase_e_probes(self, h, c_t, x, k, rel_eps=0.05):
    # 1. 完整 F forward
    E = self.compute_energy(h, c_t, x)
    grad_h = torch.autograd.grad(E.sum(), h, create_graph=False)[0]
    F_h = h - self.eta * grad_h
    
    # 2. 扰动 h，测 ρ_h_frozen
    delta = rel_eps * h.norm() * torch.randn_like(h) / torch.randn_like(h).norm()
    h_pert = h + delta
    E_pert = self.compute_energy(h_pert, c_t, x)
    grad_h_pert = torch.autograd.grad(E_pert.sum(), h_pert)[0]
    F_h_pert = h_pert - self.eta * grad_h_pert
    rho_h = (F_h_pert - F_h).norm() / delta.norm()
    
    # 3. Hessian 谱估计（Hutchinson）
    v = torch.randn_like(h)
    grad_for_hvp = torch.autograd.grad(E.sum(), h, create_graph=True)[0]
    Hv = torch.autograd.grad((grad_for_hvp * v).sum(), h)[0]
    hessian_trace = (v * Hv).sum() / v.numel()
    
    # 4. c_t 敏感度（保持现有定义）
    ...
    
    return {"rho_h_full": rho_h, "hessian_trace": hessian_trace, ...}
```

关键区别：
- **`rho_h_full`** 是完整 F 的 ρ，不是 shared_layers 子集
- **`hessian_trace`** 直接估 λ_max + λ_min 的代理，验证构造性收缩
- 两者一起给出**动力学 + 能量几何**的联合诊断

---

## 4. 实验路线（Phase E Step 1 → 4）

### Step 1：骨架跑通（~1-2 天）

**目标**：让 `EnergyReasonCore` 能 forward + backward + 不爆炸。

- 用方案 A 的最简能量参数化
- K_max = 5, eta = 0.1, T_k ≡ 0（先不加 Langevin 噪声）
- c_t 保持现有 loop 轴演化（暂不动）
- token_depth_routing 关掉
- 保留现有所有托卡马克补丁作为安全网
- 验收：loss 能下降 + 不 OOM

### Step 2：时间尺度解耦（~1 天）

**目标**：把 c_t 搬到 token 轴，验证双时间尺度是否自然。

- 修改 reason_core：c_t 进循环体前 detach，循环结束后用 h* 更新 c_t
- 验收：loss_lm 不退化 + rho_h 稳定

### Step 3：Langevin 噪声引入（~1 天）

**目标**：打开温度调度，验证噪声不破坏训练。

- T_0 = 1.0, γ = 0.5
- 验收：生成质量不变 + h 轨迹方差增加（通过 probe 测）

### Step 4：梯度范数早停（~1 天）

**目标**：替换 token_depth_routing，让自适应深度自然涌现。

- `eps_stop = 1e-3`（待调）
- 验收：token 深度分布 bimodal（简单-快停，困难-跑满）+ 总 compute ≈ V8

### Step 5：托卡马克补丁拆除（~2-3 天，最重要）

**目标**：验证仿星器成立 — 不需要任何主动控制。

**拆除顺序**（每步只改一个，验证）：
1. 先拆 `ct_inj_max`（设 ∞）
2. 再拆 `max_ct_norm`（设 0）
3. 再拆低 wd 白名单（恢复 wd=0.1）
4. 最后拆 `FORCE_ADAMW`（全走 Muon）

如果 Step 5.4 还能稳定训练 + probe 通过，**仿星器证毕**。

### Step 6：Hessian 约束调参

**目标**：把 `hessian_positive_loss` 权重调到最合适，验证 ρ < 1 是构造性的而非偶然。

---

## 5. 风险与开放问题

### 5.1 已知风险

| 风险 | 来源 | 缓解 |
|---|---|---|
| Langevin 噪声破坏 LM loss | SGLD 在高维可能不采样到训练分布 | Step 3 先用 T≡0，逐步开温度 |
| `∇_h E` 需要 create_graph=True | 双倍内存 | 用 Hutchinson trick 做 HVP，不存完整 Hessian |
| 能量函数非凸 → 局部极小 | 方案 A/C 不保证凸性 | 多温度 + 退火 + 必要时换方案 B |
| Hessian 约束太强 → 表达能力 cap | 强制 ρ<1 可能压扁地貌 | 从软约束（trace 惩罚）开始，不直接 clip |
| Probe 新测量值不可比较 | 旧 probe 和新 probe 测不同算子 | WORKLOG 明确标注"Phase E 之后的 rho_h_full 和 Phase 2 的 rho_h_frozen 不直接可比" |

### 5.2 开放问题

**O1. 能量函数是否需要对 c_t 也凸？**
不凸的话，c_t 更新可能震荡。也许需要 `E(h; c_t, x) = E_h(h; c_t) + λ · ||c_t - c_t_target||²` 形式给 c_t 额外的凸约束。

**O2. LM head 直接读 h_K 还是 h*?**
h_K 是 K 步后的状态，h* 是梯度为零的理想不动点（未必达到）。用 h_K 更直接但受 K 影响；用 h* 更干净但需要求解。

**O3. 训练时的梯度路径**
`loss_lm(lm_head(h_K))` 通过 h_K = h_{K-1} - η∇E(h_{K-1}) 反传到 h_{K-1}，再到 h_{K-2}... 这是 K 层的展开图，比现有 CR-Loop 的展开更深（因为每步有一次额外的 autograd.grad）。显存风险高，可能需要 gradient checkpointing。

**O4. c_t 在 token 轴演化的梯度流**
如果 c_t_{t+1} = ρ c_t_t + (1-ρ) g(h*_t)，那 loss 对 c_t_{t-1} 的梯度要穿过 g 回到 h*_{t-1}，再穿过整个 h*_{t-1} 的 Langevin 流回到 c_t_{t-1}。需要 BPTT through time。

**O5. 如何证明 Langevin 流的训练信号没被噪声淹没？**
SGLD 理论上在 T > 0 时采样 Boltzmann，但训练信号来自 log p(data)。需要导出类似 diffusion 模型的 score matching 目标。

---

## 6. 验收准则（Phase E 成功标准）

Phase E 被认为成功，当且仅当以下**全部**满足：

1. ✅ 训练稳定，loss_lm ≤ Phase 2 V8 最终水平（~6.0）
2. ✅ 在**完全拆除所有托卡马克补丁**后（Step 5.4），训练仍稳定
3. ✅ `rho_h_full` < 1.0 全程 p50 < 1 且 p95 < 1.2
4. ✅ `hessian_trace_est` 全程 > 0
5. ✅ Token 自适应深度分布 bimodal（简单快停，困难跑满）
6. ✅ 生成质量不退化（human eval 或 perplexity）

任何一项失败，回到对应 Step 诊断并更新本文档。

---

## 7. 文档演化规则

- 本文档是 **append-only 的种子** — 章节可新增，已有结论不删不改
- 每次实验后，在相应章节末尾添加 `**实验 X 结果（日期）**：...`
- ✅ 验证 / ❌ 证伪 / ⚠️ 部分 — 用明确标记
- 文档超过 1500 行后，归档到 `Luma_PhaseE_Theory_v1_archive.md`，新建 v2

---

## 附录 A：和 Phase 2 的对照表

| 概念 | Phase 2 | Phase E |
|---|---|---|
| 状态更新 | `h_{k+1} = h_k + Δ(h_k)` 自由 | `h_{k+1} = h_k - η∇E(h_k) + noise` 保守场 |
| ρ < 1 保证 | 没有，靠 clamp | 构造性（Hessian 正 + η 小） |
| c_t 演化轴 | loop 轴 | token 轴 |
| 自适应深度 | 学出来的 router | `||∇E|| < ε` 早停 |
| 不动点哲学 | 点吸引子 | Boltzmann 分布 / NESS |
| 托卡马克补丁 | 必须 | **目标：全部拆除** |
| probe 测什么 | shared_layers 子集 Jacobian | 完整 F 算子 + Hessian 谱 |

---

**本文档结束，等 Phase E Step 1 开工后首次更新。**
