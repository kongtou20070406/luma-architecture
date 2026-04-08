# 赫布可塑性在循环推理架构中的适用性分析

> 基于 NM 实验矩阵 (2026-04-08)，IS9 基线 (Time Cond + LoRA32 + Memory K=4 + CMDA)

## 1. 实验事实

### Rank 消融

| rank | loss | vs baseline | loss 曲线模式 |
|------|------|-------------|---------------|
| 0 (无 hebb) | 8.89 | +6.6% | 反弹 |
| 4 | 11.13 | +33.4% | 崩溃 |
| 8 | 9.17 | +9.9% | 大反弹 |
| **16** | **7.10** | **-14.9%** | **持续下降** |
| 24 | 待跑 | | |
| 32 | 待跑 | | |

### 关键观察

1. **NM6 (rank=16) 是唯一一个 loss 后期不反弹的实验。** baseline 从 6.7 反弹到 8.3 (+24%)，NM6 从 8.6 持续降到 7.1 (-17%)。赫布项的核心价值不是"让模型学得更好"，而是**防止灾难性遗忘**。

2. **rank 存在明确的相变点。** rank=4/8 都比不开 hebb 更差，rank=16 突然变成全场最佳。这不是线性改善，是相变。

3. **赫布项本身的参数极少 (~14K)，但影响巨大。** 占模型 292M 参数的 0.005%，却带来 -14.9% 的 loss 改善。

## 2. 为什么 rank=16 是相变点

c_t 是 64 维向量。赫布项的本质是：

```
Δc_t_hebb = surprise × hebb_out(hebb_proj_h(δh) ⊙ hebb_proj_c(prev_c_t))
```

这是一个 **低秩双线性映射**：从 (δh, prev_c_t) 的联合空间中提取 rank 个关联模式，映射回 c_t 空间。

- **rank=4**: 4 个关联通道，只能覆盖 c_t 64 维中的 6.25%。噪声比信号大 → 反而干扰。
- **rank=8**: 12.5% 覆盖，仍然不足以捕捉"主流变化"和"自省状态"之间的核心关联。
- **rank=16**: 25% 覆盖。这是 c_t 维度的 1/4，恰好足够表示"4 类不同的主流变化模式应该如何影响自省状态"。

**假说：赫布 rank 的有效下界 ≈ c_t_dim / 4。** 低于此阈值，关联通道太少无法编码有用模式；高于此阈值，开始真正起作用。

## 3. 赫布防遗忘的机制

标准的 c_t 更新是：
```
c_t = introspection_stream(h)  # 每次完全由当前 h 决定
```

加了赫布后变成：
```
c_t = introspection_stream(h) + surprise × hebb(δh, prev_c_t)
```

赫布项引入了 **prev_c_t 对新 c_t 的残差影响** — 即使自省流给出了新的 c_t，赫布项仍然保留了"上一轮 c_t 和当前主流变化之间的关联"。这是一种**隐式的经验回放**：

1. surprise 高时（JEPA 预测不准），赫布项强写入 → c_t 快速适应新模式
2. surprise 低时（正常训练），赫布项弱写入 → c_t 保留已学到的关联
3. 训练后期（已见过大部分数据），surprise 整体降低 → 赫布项自动变成稳定器

**这解释了为什么 NM6 不反弹：赫布项在训练后期自动切换为"记忆保持"模式。**

## 4. 适用性拓宽：赫布可以用在哪里

### 4.1 已验证：c_t 写入调制

当前实现。效果显著。

### 4.2 高潜力：Shared Layer 的 LoRA 更新

当前 Loop LoRA 是 per-loop Embedding lookup，各轮独立。可以加赫布调制：

```python
# 当前: lora_delta = lora_B(lora_A(loop_idx))  # 查表
# 改进: lora_delta = lora_B(lora_A(loop_idx)) + surprise × hebb(δh, prev_lora_state)
```

让 LoRA 的权重更新也受 surprise 调制。高 surprise 时 LoRA 做更大的调整 → shared_layers 在深循环中真正做不同的计算。

这直接对准了"循环深了但没有用"的核心问题：目前深循环中每轮 LoRA 变化是固定的（从 Embedding 查表），不会根据当前状态动态调整。加了赫布后，LoRA 的调整量和方向会受"上一轮发生了什么"影响。

### 4.3 中潜力：Memory Token 的写入门控

当前 memory tokens 通过 cross-attention 从主流读取，残差累积。可以加赫布式的选择性写入：

```python
# 当前: memory = memory + cross_attn(memory, h)  # 每轮全部 slot 都更新
# 改进: write_gate = surprise × σ(hebb(δh, memory))  # 只有 surprise 高时才写入
#       memory = memory + write_gate × cross_attn(memory, h)
```

这就是 NTM 风格的可寻址记忆，但用赫布规则代替了 NTM 的 learned addressing。

### 4.4 低潜力但有趣：注意力权重的赫布调制

在 shared_layers 的 attention 中，用赫布项调制 attention scores：

```python
attn_scores = Q @ K.T / sqrt(d) + surprise × hebb_attn_bias(prev_attn_pattern, current_Q)
```

类似于"如果上一轮关注了某些 token 并且 surprise 高，这一轮继续关注它们"。这模拟了注意力的持续性（sustained attention），是认知科学中的核心概念。

## 5. 生物学对应

| Luma 组件 | 生物学对应 | 机制 |
|-----------|-----------|------|
| surprise (1-self_check) | 去甲肾上腺素 (NE) | 检测新异性 |
| hebb_term | 突触可塑性 | 共激活→连接增强 |
| surprise × hebb | NE 调制可塑性 | 注意力调制学习率 |
| c_t | 工作记忆内容 | 前额叶持续活动 |
| rank=16 相变 | 突触聚类 | ~16 个树突分支独立整合 |

## 6. 下一步建议

1. **确认 rank 扫描**：等 rank=24/32 结果，确认是否有最优点或持续改善
2. **JEPA surprise 替代 self_check**：更直接的预测误差信号
3. **融合消融**：hebb16 + (confidence/curvature/entropy) 的组合效果
4. **1000 步长训练**：验证赫布防遗忘效应在更长训练中是否更显著
5. **LoRA 赫布调制**：下一个实验矩阵的核心方向
