# Matrix 9: AttnRes 改造实验报告

> 日期: 2026-04-06  
> 架构: A1 (482M, 768h, L44, 12/3 heads, shared_depth=2)  
> 基线: B2' (world_sigreg=0.10, mask=0.25, mhc_alpha_init=0.01)  
> 参考: arxiv 2603.15031 (Kimi Block Attention Residuals)  

---

## 1. 实验目标

验证 Kimi 论文的 Block Attention Residuals 是否优于当前的 lerp 残差注意力。

**当前实现 (legacy lerp)**:
- 输出: `α * old + (1-α) * new` (插值)
- Query: 全局共享 pseudo_query
- Value: V 经过 RMSNorm

**论文实现 (paper)**:
- 输出: softmax attention 加权和 (direct replace)
- Query: 每层/每块独立 pseudo_query (zero-init)
- Value: V 保持 raw (只 norm K)

## 2. 实验设计

| 实验 | CompressionZone AttnRes | ReasoningLoop AttnRes | Query 类型 | 假设 |
|---|---|---|---|---|
| AR0 | legacy (lerp) | legacy (lerp) | global | Baseline |
| AR1 | **paper (direct)** | legacy (lerp) | per-block | 压缩区受益于更强的跨层信息选择 |
| AR2 | legacy (lerp) | **paper (direct)** | per-loop | 推理循环受益于动态残差路径 |
| AR3 | **paper (direct)** | **paper (direct)** | per-block/loop | 全量替换 |
| AR5 | **paper_global_q** | **paper_global_q** | global | 论文输出方式 + 全局 query (消融) |

每实验 2100 步, seq=2048, bs=1, reason_loops=12, FP8.

## 3. 结果总览

| 实验 | loss_lm | vs AR0 | Peak VRAM | v2_rank (终) | mode1% (终) | MHC | 结论 |
|---|---|---|---|---|---|---|---|
| AR0 baseline | 3.6804 | — | 9.95 GB | 11/52 | 75.2% | dead | 基线 |
| **AR1 compress_paper** | **3.0867** | **-16.1%** | **10.46 GB** | **10/52** | **78.6%** | **dead** | **胜出 ✅** |
| AR2 reason_paper | 4.2593 | +15.7% | 10.22 GB | 8/52 | 90.2% | dead | 退步 |
| AR3 full_paper | 3.5493 | -3.6% | 10.53 GB | 5/52 | 86.9% | dead | 略好，但不如 AR1 |
| AR5 paper_global_q | 3.8328 | +4.1% | 10.53 GB | 11/52 | 80.3% | **alive** | MHC 存活但 loss 退步 |

## 4. 深度分析

### 4.1 AR1 为什么胜出？

AR1 在压缩区使用 paper AttnRes，推理循环保持 legacy lerp。这是最佳组合：

**压缩区特性**：44 层深度前馈，每层只执行一次。Paper AttnRes 的 per-block 独立 query + direct replace 允许每层自主选择从哪些前序层提取信息，比固定比例 lerp 更灵活。

**为什么推理循环不适合 paper AttnRes** (AR2 退步 15.7%):
- 推理循环是 **迭代式** 的：同一组共享层反复执行 12 次
- lerp (`α * old + (1-α) * new`) 提供了天然的 **指数移动平均** 效应，让表征在迭代中平滑演化
- direct replace 在迭代场景中打断了这种平滑性，导致每个 loop 的输出波动更大

**AR3 (全量 paper) 只好 3.6%** — 压缩区的 16% 增益被推理循环的 15% 退步大幅抵消。

### 4.2 DOD 动力学对比

**v2_rank 轨迹** (越高 = 表征多样性越好):

| Step | AR0 | AR1 | AR2 | AR3 | AR5 |
|------|-----|-----|-----|-----|-----|
| 200 | 5 | 4 | 7 | 6 | 6 |
| 600 | 5 | 4 | 4 | 4 | 4 |
| 1000 | 6 | 5 | 6 | 3 | 7 |
| 1400 | 10 | 7 | 6 | 4 | 9 |
| 2100 | 11 | 10 | 8 | 5 | 11 |

- AR0/AR1/AR5: v2_rank 持续上升至 10-11（健康的表征多样化）
- AR2: v2_rank 上升到 8 后停滞
- **AR3: v2_rank 始终极低 (3-5)**，mode1% 始终 >86% — **近表征坍缩**

AR3 的 v2_rank=5 + mode1%=86.9% 是危险信号：全量 paper AttnRes 虽然 loss 略好于 baseline，但表征多样性严重不足。长期训练可能恶化。

### 4.3 mode1_energy 分析

mode1_energy 反映梯度方向集中度，越低 = 梯度来源越多样：

| 实验 | mode1% 起始 | mode1% 终止 | 趋势 |
|---|---|---|---|
| AR0 | 95.6% | 81.7% | ↓ 健康分散 |
| AR1 | 85.9% | 87.1% | → 稳定 |
| AR2 | 92.0% | 92.4% | → 停滞 |
| AR3 | 58.6% | 94.1% | ↑ 梯度集中化！ |
| AR5 | 87.1% | 73.9% | ↓ 最佳分散 |

AR3 的 mode1 从 58% 上升到 94% — 梯度越来越集中到单一方向，配合 v2_rank=5 证实了表征坍缩趋势。

### 4.4 AR5 (paper_global_q) 的 MHC 存活

AR5 是唯一一个 MHC 全程存活的配置：
- 全局 pseudo_query (不是 per-block/per-loop)
- 论文式 direct replace 输出 + V raw

MHC 存活的原因可能是全局 query 提供了更稳定的残差路径，不像 per-block query 每层学到完全不同的注意力模式。但 loss 比 baseline 差 4%，说明**全局 query 的表达能力不足**。

### 4.5 VRAM 开销

| 实验 | Peak VRAM | vs AR0 |
|---|---|---|
| AR0 | 9.95 GB | — |
| AR1 | 10.46 GB | +0.51 GB |
| AR2 | 10.22 GB | +0.27 GB |
| AR3 | 10.53 GB | +0.58 GB |
| AR5 | 10.53 GB | +0.58 GB |

Paper AttnRes 增加 0.3-0.6 GB VRAM，在 32GB RTX 5090 上完全可接受。

## 5. 结论与决策

### 胜出配置: AR1 (压缩区 paper + 推理循环 legacy)

```
--attnres_compress_mode paper --attnres_reason_mode legacy
```

**理由**:
1. **loss 降低 16.1%** — 实验中最大收益
2. **动力学健康** — v2_rank=10/52，mode1%=78.6%，与 baseline 相当
3. **VRAM 可接受** — +0.51 GB (10.46 vs 9.95 GB)
4. **架构合理性** — 压缩区单次前馈适合 paper AttnRes 的自由信息选择，推理循环的迭代特性需要 lerp 的平滑

### 关键洞察

**残差策略应因场景而异：**
- **单次前馈 (CompressionZone)**: paper AttnRes (per-block query, direct replace) 更优 — 每层自主选择信息源
- **迭代循环 (ReasoningLoop)**: legacy lerp 更优 — 提供 EMA 平滑，防止迭代振荡

### 未跳过实验

AR4 (input-dependent query, `W_q @ h`) 仍可探索，但 AR1 已提供 16% 增益，优先级降低。

## 6. 下一步

1. **采纳 AR1 配置为新默认** — 更新训练配置
2. **启动 M7 (训练吞吐量)** — gradient accum=2，基于 AR1 配置
3. MHC 在所有实验中不稳定 — 可能需要在 M10 中专门调试

---

*实验耗时: 4h8m (23:20 - 03:28)*  
*脚本: `minimind/scripts/run_matrix9_attnres.sh`*
