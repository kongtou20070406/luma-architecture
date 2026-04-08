# 循环深度坍缩分析报告

> 日期: 2026-04-07
> 问题: ExitController 在所有配置下学到 avg_loops=2，即使给予 20 loops 预算
> 架构: Luma 482M (A1), reason_shared_depth=2, compression_layers=44

---

## 1. 现象

在 Matrix 5 (E9 winner) 配置下，ExitController 始终在第 2 次循环后退出：

| 实验 | 数据 | avg_loops | loss |
|------|------|-----------|------|
| E9 baseline | pretrain_h_python | 2.0 | 2.5417 |
| LD1 hard_math | MATH competition L4-5 | 2.0 | 6.6919 |
| SJ0 baseline | OpenR1-Math 2k | 2.3 | 11.0830 |
| SJ1 +SigReg ct | OpenR1-Math 2k | 2.3 | 10.4337 |

强制更深循环（Phase F 实验）全部更差：
- F0 min_loops=4: +0.6%
- F1 bias=-2.0: +1.2%
- F2 warmup=100: +0.4%

换更难数据（LD 实验）也无法驱动更深循环。

## 2. 根因分析

### 2.1 假设 A: reason_shared_depth=2 表达力已饱和

只有 2 个独立共享层在循环复用。2 层 × 2 次 = 4 次前向传播，之后表征收敛。更多循环只是重复相同变换，ExitController 正确识别了这一点。

**证据支持**:
- 强制更深循环全部退步（F 实验）
- 梯度范数显示 reason_shared_1 (1.63) > reason_shared_0 (0.58)，第二层已在努力工作

**反驳**: 待 SD 实验验证（depth=1 是否被迫用更多 loop，depth=3 是否自然更深）

### 2.2 假设 B: 压缩区过重，推理区无事可做

**这可能是更深层的根因。**

当前架构：
- 压缩区 44 层 → 参数占比 ~75%
- 推理区 2 层共享 → 参数占比 ~8%（~40M / 482M）

44 层的压缩区本身就是一个完整的中型 LLM。大多数同规模模型（GPT-2 Medium 355M = 24 层，Qwen 0.5B = 24 层）总层数不超过 32。压缩区可能已经完成了绝大部分表征学习，推理循环只做边际精修。

**证据支持**:
- 梯度质量在压缩区和推理区之间大致平衡（~2.2 vs ~2.2），但压缩区用 44 层分摊，推理区只有 2 层 → 推理层承压更大
- 强制更深循环反而更差 → 推理区确实没有足够的"工作"来填充更多循环
- 早期（无 ExitController 梯度时）循环深度高（hard_loop_var=7.0）但 loss 并不因此更好

**预测**: 如果减少压缩层数（如 32 或 24 层），推理循环可能被迫承担更多工作 → avg_loops 自然增加

### 2.3 假设 C: 训练信号不足（中间循环无 credit）

ExitController 只通过最终输出的 loss 反向传播获得梯度。中间循环步没有独立的训练信号，模型发现"2 步到终点"是最短路径。

**文献支持**:
- RLTT (arXiv 2602.10520): 标准 GRPO 只对最终 state 分配 credit，导致与内部计算的根本不匹配。trajectory-level dense reward 在 MATH-500 上 +14.4%
- Thinking Deeper, Not Longer (arXiv 2603.21676): Silent thinking objective 只监督最终输出，但通过 identity-biased recurrence 创建梯度高速公路，使深层循环获得有效梯度

### 2.4 假设 D: 缺少 identity-biased recurrence

每个循环步的变换是 `h → shared_layers(h)`，没有显式的恒等映射偏置。梯度在多步循环中衰减，深层循环信号太弱。

**文献支持**:
- Thinking Deeper, Not Longer (arXiv 2603.21676): 三个稳定化技术中 identity-biased recurrence 最关键
- LoopFormer (arXiv 2602.11451): shortcut-consistency training 要求不同深度都产生有效输出

## 3. 已进行的实验

### 3.1 Phase F: 强制深循环（全部失败）
| 实验 | 方法 | vs baseline |
|------|------|-------------|
| F0 | min_loops=4 | +0.6% |
| F1 | bias_init=-2.0 | +1.2% |
| F2 | warmup=100 steps | +0.4% |

**结论**: 强制不是答案。模型确实不需要更深循环（在当前架构下）。

### 3.2 LD: 难数据驱动深循环（失败）
| 实验 | 数据 | avg_loops |
|------|------|-----------|
| LD0 easy_data | pretrain_h_python | 2.0 |
| LD1 hard_math | MATH competition | 2.0 |

**结论**: 数据难度不影响循环深度。根因在架构/训练信号侧。

### 3.3 SJ: Self-JEPA 激活实验（进行中）
| 实验 | 配置 | loss_lm | avg_loops |
|------|------|---------|-----------|
| SJ0 | E9 baseline | 11.0830 | 2.3 |
| SJ1 | +SigReg ct | 10.4337 | 2.3 |
| SJ2 | +SigReg ct+rollout | 进行中 | ~3 |
| SJ3-SJ5 | c_t drift 系列 | 待跑 | — |

**初步发现**: SigReg ct 显著改善 loss（-5.9%）但不影响循环深度。SJ2 加入 rollout SigReg 后出现 loops=3 的迹象。

### 3.4 SD: Shared Depth 实验（待跑）
| 实验 | depth | loops | 假设 |
|------|-------|-------|------|
| SD0 | 1 | 20 | 单层太弱 → 被迫多循环？ |
| SD1 | 2 | 20 | 对照 baseline |
| SD2 | 3 | 20 | 更多独立层 → 更深？ |
| SD3 | 3 | 12 | 收紧预算验证 |

## 4. 相关前沿文献

### 4.1 最直接相关（循环深度坍缩问题）

| 论文 | 发表 | 核心启发 |
|------|------|----------|
| **Thinking Deeper, Not Longer** (2603.21676) | arXiv 2026 | Silent thinking + identity-biased recurrence + LayerScale 稳定深度递归 |
| **LoopFormer** (2602.11451) | arXiv 2026 | Shortcut-consistency training 防止坍缩 — 不同深度都要有效 |
| **RLTT** (2602.10520) | arXiv 2026 | Trajectory-level dense reward 替代 outcome-only credit |
| **Two-Scale Latent Dynamics** (2509.23314) | NeurIPS 2025 | 二阶 step-size 差分作为自监督退出信号 |
| **Ouro** (2510.25741) | arXiv 2025 | Entropy-regularized objective 防止捷径学习 |

### 4.2 架构参考

| 论文 | 发表 | 核心启发 |
|------|------|----------|
| **MoR** (2507.10524) | NeurIPS 2025 | Per-token depth routing + 端到端 router |
| **Inner Thinking Transformer** (2502.13842) | ACL 2025 | Residual Thinking Connections 防循环退化 |
| **MIND** (ICLR 2025) | ICLR 2025 | Fixed-point iteration + introspection network |
| **Coconut** (2412.06769) | ICLR 2025 | 连续空间推理 = BFS over latent paths |
| **Huginn-3.5B** + 后续分析 (2507.02199) | 2025 | 深递归模型中未发现结构化 latent CoT 证据 |

### 4.3 JEPA + 语言模型

| 论文 | 发表 | 核心启发 |
|------|------|----------|
| **LLM-JEPA** (2509.14252) | arXiv 2025 | JEPA objective 跨模型家族优于标准预训练 |
| **NextLat** (2511.05963) | NeurIPS 2025 | Latent prediction 收敛到 belief states |

## 5. SJ 实验结果（Self-JEPA 激活）

| 实验 | 配置 | loss_lm | avg_loops | vs SJ0 |
|------|------|---------|-----------|--------|
| SJ0 | E9 baseline | 11.083 | 2.3 | — |
| **SJ1** | **+SigReg ct (0.05)** | **10.434** | **2.3** | **-5.9%** |
| SJ2 | +SigReg ct+rollout | 10.723 | 2.3 | -3.2% |
| SJ3 | +c_t drift 0.5 | 11.380 | 2.3 | +2.7% |
| SJ4 | +c_t drift 1.5 | 11.802 | 2.3 | +6.5% |
| SJ5 | SigReg ct + drift | 11.240 | 2.4 | +1.4% |

**SJ 结论**:
- SigReg ct 防坍缩 = 纯收益（-5.9%），之前一直没开
- c_t drift 参与退出 = 纯毒药，drift 越强越差
- avg_loops 全部 2.3，退出信号改不了循环深度

**SJ 最优 = SJ1**: `--enable_sigreg_ct 1 --sigreg_ct_weight 0.05`

## 6. CR 实验结果（压缩区/推理区比例）— 核心突破

### 第一轮

| 实验 | 压缩层 | depth | 参数量 | loss_lm | avg_loops | vs CR0 |
|------|--------|-------|--------|---------|-----------|--------|
| CR0 | 44 | 2 | 482M | 11.412 | 2.3 | — |
| CR1 | 36 | 2 | 415M | 10.646 | 2.3 | -6.7% |
| CR2 | 32 | 3 | 401M | 12.030 | 2.4 | +5.4% |
| CR3 | 24 | 4 | 354M | 10.283 | 2.4 | -9.9% |

### 第二轮

| 实验 | 压缩层 | depth | 参数量 | loss_lm | avg_loops | vs CR0 |
|------|--------|-------|--------|---------|-----------|--------|
| CR3b | 24 | 2 | 314M | 10.865 | 2.1 | -4.8% |
| CR4 | 20 | 4 | 320M | 9.815 | 2.2 | -14.0% |
| **CR5** | **16** | **4** | **286M** | **8.944** | **2.5** | **-21.6%** |
| CR6 | 12 | 4 | 254M | 10.665 | 2.7 | -6.5% |

### CR 关键发现

1. **CR5 (c16_d4, 286M) 碾压全场** — loss=8.94，比原始 482M 模型低 21.6%，参数只有 59%
2. **44 层压缩区严重过重** — 砍到 16 层仍有收益，12 层开始回升 → 16 是甜区
3. **depth=4 有独立贡献** — CR3 (24+d4)=10.28 vs CR3b (24+d2)=10.87，depth 带来 -5.4%
4. **avg_loops 随压缩减少而增加** — CR0:2.3 → CR5:2.5 → CR6:2.7，趋势明确
5. **CR2 (32+d3) 异常** — depth=3 可能是不稳定点，depth=2 和 4 都正常

### CR 结论

**假设 B 确认：压缩区过重是循环浅的根因之一。**

- 压缩区从 44→16（-64%），推理区从 2→4（+100%），总参数从 482M→286M（-41%）
- loss 下降 21.6%，avg_loops 从 2.3→2.5
- 最优比例：**压缩区 16 层 + 推理区 depth=4**

### 新架构推荐: A2 (暂定)

```
--compression_layers 16 --reason_shared_depth 4
--hidden_size 768 --intermediate_size 3072
--num_attention_heads 12 --num_key_value_heads 3
```

总参数 ~286M，相比 A1 (482M) 减少 41%，loss 下降 21.6%。

## 7. 可行的后续改进方向

### P1: Identity-Biased Recurrence（来自 "Thinking Deeper, Not Longer" arXiv 2603.21676）

**做法**: 在共享层循环中加入 identity shortcut：
```python
h_new = shared_layers(h) 
h = h + alpha * (h_new - h)  # alpha 从小到大 warmup
```

**成本**: ~3 行代码，零参数增加。在 CR5 基础上可能进一步推深循环。

### P2: Shortcut-Consistency Training（来自 LoopFormer arXiv 2602.11451）

**做法**: 训练时随机在不同循环深度取输出计算 loss，要求所有深度都产生有效预测。

**成本**: 增加约 30% 训练时间。直接破解"早退出 = 捷径"的问题。

### P3: Trajectory-Level Dense Credit（来自 RLTT arXiv 2602.10520）

**做法**: 每个循环步的 hidden state 都计算辅助预测 loss，给中间步提供独立训练信号。

### P4: 在 CR5 基础上微调压缩层数

CR5 (16层) 和 CR6 (12层) 之间有较大跳跃，可测试 14 层。

### P5: 旋度/散度分析

等循环深度 >5 后，分析 h_t 轨迹的几何性质（delta_h 夹角序列 = curl proxy，delta_h 范数比 = divergence proxy）。

## 8. 历史背景

### 早期退出策略探索（2026-03-28 ~ 04-06）

1. **Autoresearch 迭代 1-2**: one-step continuation gain 有效（tail 0.094→0.041）
2. **Autoresearch 迭代 3-7**: two-step value / gain gating 全部失败（persona/emotion 崩坏）
3. **迭代 8 PIVOT**: 放弃 two-step value，保留 one-step gain
4. **Rollout Depth**: 10x15 有效（hard_loop_var=7.0），10x20 无额外收益 → 瓶颈是 exit policy
5. **Matrix 2**: EX5 胜出（exit_aux=0.01 + second_order=0.3 + 20 loops → -1.3%）
6. **现在**: 即使 EX5 配置，实际只用 2 loops → ExitController 学到了"2 就够"

### 关键认知转变

- 早期 hard_loop_var=7.0 是**随机退出**的结果（ExitController 无梯度，权重随机），不代表模型真的需要深循环
- M2 的 -1.3% 收益来自**训练信号改善**（auxiliary loss 正则化），不是来自更深推理
- **压缩区 44 层可能才是根因**：已经是一个完整 LLM，推理循环只是附加的微调层

---

*报告生成时间: 2026-04-07*
*关联实验: SJ0-SJ5, CR0-CR6, CR3b*
*关联脚本: run_selfjepa_exit_test.sh, run_compress_ratio_test.sh, run_compress_ratio_test2.sh*
*核心发现: 44层压缩区过重是循环浅的根因。CR5 (c16_d4, 286M) 以 59% 参数量实现 -21.6% loss。*
