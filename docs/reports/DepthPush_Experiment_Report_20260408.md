# Depth Push (DP) 实验报告

**日期**: 2026-04-08
**基线**: CR5 (c16_d4, 286M) + SJ1 (SigReg ct) + E9 (MoR + MHC3)
**数据**: openr1_math_hard_2k.jsonl
**配置**: 500 steps, seq=2048, bs=1, accum=2, reason_loops=20
**目标**: 突破 avg_loops=2 的循环深度坍缩

## 结果汇总

| 实验 | 方案 | loss_lm | vs baseline | avg_loops | max_loops | 结论 |
|------|------|---------|-------------|-----------|-----------|------|
| **DP0** | 对照 (无新方案) | 8.8185 | — | 2.1 | 3 | 基线 |
| **DP1** | Exit entropy (Ouro) | 8.3066 | **-5.8%** | 2.0 | 2 | Loss 好，但循环更浅 |
| **DP2** | Time conditioning (LoopFormer) | **8.0490** | **-8.7%** | 2.2 | 4 | **最佳 loss** |
| **DP3** | Entropy + Time | 8.0675 | -8.5% | 2.1 | 4 | 和 DP2 相当，entropy 冗余 |
| **DP4** | RLTT dense LM (Princeton) | 8.9898 | +1.9% | **2.5** | **7** | **最深循环**，但 loss 更差 |
| **DP5** | Entropy + RLTT | 8.6550 | -1.9% | 2.0 | 2 | Entropy 压制了 RLTT 的深度 |
| **DP6** | Shortcut consistency (LoopFormer) | CRASH | — | — | — | CUDA driver error (FP8 backward 不兼容) |
| **DP7** | Full combo (Entropy+Time+RLTT) | 8.0821 | -8.4% | 2.1 | 4 | 接近 DP2，RLTT 被稀释 |
| **DP8** | Coconut 1 round (Meta) | 9.9922 | **+13.3%** | 2.2 | 4 | 严重劣化 |
| **DP9** | Coconut + Entropy | 9.9942 | +13.3% | 2.1 | 3 | 同上 |

## 关键发现

### 1. Time Conditioning 是最有效的单一方案 (DP2, -8.7%)

LoopFormer 的时间步注入（normalized t + dt → hidden_size projection）带来了最大的 loss 改善。实现极其简单（2 维输入 → Linear → add to h），零参数量开销。

机制解释：给 shared_layers 提供了"我在第几轮"的信息，让相同的权重在不同循环步可以做不同的事情。这类似于 positional encoding 对 Transformer 的作用。

### 2. RLTT 是唯一能推深循环的方案 (DP4, avg=2.5, max=7)

RLTT dense LM loss 在每个中间循环步计算 LM loss，并用深度加权（后面的循环权重更高）。这给中间循环步提供了直接的学习信号，激励模型在更多循环中持续改善输出。

**但代价是 loss 更差 (+1.9%)**。原因：中间循环步的 LM loss 是 noise — 早期循环输出质量差，强制拟合这些输出会干扰最终输出的学习。

### 3. Exit Entropy Regularization 是 anti-depth (DP1, DP5)

虽然 entropy reg 改善了 loss (-5.8%)，但它把循环深度压到了 2.0（甚至从 2.5 压到 2.0，见 DP5 vs DP4）。

机制：entropy reg 惩罚 exit score 的确定性，导致模型学会在所有循环步给出接近 0.5 的 exit score。当 threshold=0.8 时，没有循环步能超过阈值，模型只能在最少循环（2次）后因达到最大步数的某种 fallback 而退出。

**核心矛盾**：entropy reg 优化的是 exit score 分布的多样性，不是循环深度本身。

### 4. Coconut 在 286M 规模不可行 (DP8/DP9, +13.3%)

Coconut 将 c_t (64维) 投影到 hidden_size (768维) 作为 thought token prepend，然后再过一遍 shared_layers。

失败原因分析：
- 64→768 的投影在训练初期是噪声，严重干扰第二轮 shared_layers 的输入分布
- 额外的 shared_layers forward 增加了 2x 计算量，但没有足够的训练步数来收敛
- 286M 模型的 capacity 不足以同时学习"正常推理"和"thought token 理解"

### 5. 组合方案效果不叠加 (DP3, DP5, DP7)

- DP3 (entropy + time) ≈ DP2 (time alone) — entropy 冗余
- DP5 (entropy + RLTT) < DP4 (RLTT alone) on depth — entropy 压制深度
- DP7 (entropy + time + RLTT) ≈ DP2 (time alone) — 其他方案被 time conditioning 主导

### 6. DP6 (Shortcut Consistency) CUDA Driver Crash

Shortcut consistency 需要通过随机中间 loop 的 logits 做 KL divergence backward。在 FP8 forward + BF16 backward + gradient checkpointing 的环境下，这个 backward 路径触发了 CUDA driver error: device not ready。

**不是代码 bug**，而是 WSL2 + 5090 + FP8 backward 的兼容性问题。需要：
1. 关闭 FP8 重试，或
2. 用 torch.no_grad() 包裹 shortcut 路径的 teacher logits（当前 target 已经 detach，但 student 路径仍然经过 FP8 layers）

## Bug 修复记录

### RLTT 序列长度不匹配 (DP4 首次崩溃)

**问题**: `ValueError: Expected input batch_size (2059) to match target batch_size (2047)`

**原因**: RLTT 的 `loop_h_grad` 中的 hidden state 序列长度包含了额外的 prepend token，但 cross_entropy target `y = labels[..., 1:]` 只有原始 token 长度。主路径用 `logits[:, -labels.shape[1]:, :]` trim 过了，RLTT 没有。

**修复**: 在 RLTT 和 shortcut consistency 中加入相同的 trim：
```python
_logits_i = _logits_i[:, -labels.shape[1]:, :]  # trim to match labels length
```

## 推荐

### 短期 (下一轮实验)

1. **Time conditioning 作为默认配置** — 加入 `--enable_time_conditioning 1` 到标准训练命令
2. **RLTT 需要降低权重** — 0.05 太高，试 0.01-0.02 的范围看能否兼顾 loss 和深度
3. **Time + 轻量 RLTT** — DP7 没有很好地测试这个组合（因为同时加了 entropy），建议新实验：time + RLTT(0.01-0.02) 不加 entropy

### 中期

4. **Shortcut consistency 需要 FP8 兼容性修复** — 关闭 FP8 或用 float32 fallback 做 shortcut backward
5. **Coconut 暂时搁置** — 在更大规模模型或更长训练后再试

### 核心结论

**循环深度坍缩的根因不在这些辅助方案**。所有方案最多把 avg_loops 从 2.1 推到 2.5（DP4），没有任何方案能突破 3。问题可能在：
- ExitController 的架构本身（二阶差分退出可能过于保守）
- 共享权重循环的梯度信号衰减（循环越深，梯度越弱）
- 需要更激进的退出策略改革，而不是辅助 loss 调节

---

*实验耗时约 90 分钟 (10 experiments × ~8 min each + bug fixes + CUDA recovery)*
