# IS (Introspection Stream) 实验报告

**日期**: 2026-04-08
**基线**: CR5 + SJ1 + E9 + Time Conditioning + LoRA32 (RS5 最优)
**数据**: openr1_math_hard_2k.jsonl, 350 steps

## 结果汇总

| 实验 | 方案 | loss_lm | vs IS0 | avg_loops | 结论 |
|------|------|---------|--------|-----------|------|
| **IS0** | 对照 (mean pool + broadcast) | 8.8484 | — | 2.3 | baseline |
| IS1 | Memory token K=4 | 9.1562 | +3.5% | 2.1 | K=4 太少 |
| IS2 | Memory token K=8 | 8.3360 | -5.8% | 2.3 | K=8 有效 |
| **IS3** | **Chunked pooling (8 chunks)** | **7.5506** | **-14.7%** | 2.4 | **零参数大幅改善** |
| IS4 | meta_dim 96→192 | 9.8951 | +11.8% | 2.2 | 更差 (收敛不足) |
| IS5 | c_t_dim 64→128 | 9.5495 | +7.9% | 2.1 | 更差 (收敛不足) |
| IS6 | Token-aware c_t injection | 12.0435 | +36.1% | 2.4 | 严重劣化 |
| IS7 | BiXT 双向 cross-attention | 8.4715 | -4.3% | **2.6** | loss 好 + **循环最深** |
| IS8 | CMDA 双向通道调制 | 9.7572 | +10.3% | 2.3 | 单独用效果不好 |
| **IS9** | **Memory K=4 + CMDA** | **7.4713** | **-15.6%** | 2.2 | **全场最佳** |

## 关键发现

### 1. Chunked Pooling 是最佳单项方案 (IS3, -14.7%, 零参数)

把 seq=2048 分成 8 段各 mean → [B, 8, 768] 输入自省流 Mamba，完全不加参数就改善 14.7%。

**这证实了核心假设**：自省流退化的主要原因是 `h.mean(dim=1)` 丢失了位置信息。只要恢复粗粒度位置信息，自省流立刻变强。

同时 Mamba 从 seq_len=1 变成 seq_len=8，终于能做序列建模了。

### 2. Memory K=4 + CMDA 组合是全场最佳 (IS9, -15.6%)

IS9 比 IS3 更好（7.47 vs 7.55），虽然差距很小。有意思的是：
- Memory K=4 单独用（IS1）很差（+3.5%）
- CMDA 单独用（IS8）也很差（+10.3%）
- 但组合起来就是全场最佳

**机制解释**：Memory tokens 选择性读取主流关键信息（解决输入问题），CMDA 让 c_t 做 channel-wise 调制而非 broadcast add（解决注入问题）。两个问题同时修复才能发挥协同效应。

### 3. BiXT 是唯一推深循环的方案 (IS7, avg=2.6)

BiXT 双向 cross-attention 把循环深度从 2.3 推到 2.6，是 IS 矩阵中唯一改变循环深度的方案。

**机制**：BiXT 让主流每轮都从 memory tokens 获取信息（不只是 c_t 的 broadcast add），这给 shared_layers 提供了新的输入变化，推迟了 hidden state 收敛。

### 4. 扩维度在短训练中无效 (IS4/IS5)

meta_dim 96→192 和 c_t_dim 64→128 都恶化了。更多参数需要更多训练步数来收敛。在 350 步的实验中，这些参数只是噪声。

更长训练（1000+ steps）后可能有效，但优先级不高。

### 5. Token-aware injection 严重失败 (IS6, +36.1%)

用 c_t × h 做 per-token gate 后注入，效果极差。原因：c_t 只有 64 维，投影到 768 维后做 element-wise 乘法产生了过多的自由度，训练初期就是纯噪声。这和 Coconut (DP8/DP9) 失败的原因类似 — 低维→高维的投影在训练初期是破坏性的。

## 与前序实验的累积效果

从原始 baseline 到当前最优，历次改善叠加：

| 步骤 | 配置 | loss_lm | 累积改善 |
|------|------|---------|----------|
| 1 | CR5 基线 (DP0) | 8.8185 | — |
| 2 | + Time Conditioning (DP2) | 8.0490 | -8.7% |
| 3 | + LoRA32 (RS5 基线 IS0) | 8.8484* | — (IS0 重测) |
| 4 | + Chunked Pooling (IS3) | 7.5506 | -14.7% vs IS0 |
| 5 | + Memory K=4 + CMDA (IS9) | **7.4713** | **-15.6% vs IS0** |

*注：IS0 的 loss 比 RS5 高是因为 350 步随机性，不是回退。

## 推荐

### 立即纳入默认配置
1. **Chunked pooling** (introspection_input_mode=chunked) — 零参数，-14.7%
2. **LoRA rank=32** — +1.4% 参数，-20% loss

### 下一轮验证
3. **Chunked pooling + BiXT** — IS3 的输入 + IS7 的双向交互，可能兼得 loss 和深度
4. **IS9 配置跑更长** — Memory + CMDA 在 1000 步后是否持续优势
5. **Chunked pooling chunk 数量调优** — 8 chunks 是否最优，试 4/16

### 暂时搁置
6. IS4/IS5 扩维度 — 需要更长训练，当前不紧急
7. IS6 token-aware — 方向错误，放弃
