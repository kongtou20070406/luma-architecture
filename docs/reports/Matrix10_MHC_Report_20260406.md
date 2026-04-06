# Matrix 10: MHC 门控调优实验报告

> 日期: 2026-04-06  
> 架构: A1 (482M) + AR1 (compress paper + reason legacy) + GL1 (accum=2)  
> 基线: B2' 配置 (sigreg=0.10, mask=0.25)  

---

## 1. 实验目标

Multi-Head Compression (MHC) 在之前所有实验中频繁进入 dead 状态。本实验调优 MHC 的两个核心超参数：
- **alpha_init**: softmax 温度系数，控制 routing 差异化程度
- **n_streams**: 残差流数量，影响路由复杂度

## 2. 结果

| 实验 | alpha | streams | loss_lm | vs MH0 | Peak VRAM | MHC 复活 step | v2_rank (终) |
|---|---|---|---|---|---|---|---|
| MH0 | 0.01 | 4 | 2.9385 | — | 11.60 GB | ~600 | 5/52 |
| MH1 | 0.03 | 4 | 2.7041 | -8.0% | 11.20 GB | ~600 | 5/52 |
| MH2 | 0.02 | 4 | 3.0322 | +3.2% | 11.20 GB | ~600 | 4/52 |
| MH3 | 0.10 | 4 | 2.6963 | -8.2% | 11.20 GB | ~400 | 4/52 |
| **MH4** | **0.01** | **2** | **2.6914** | **-8.4%** | **11.14 GB** | **~400** | **6/52** |
| MH5 | 0.01 | 8 | 3.2507 | +10.6% | 11.35 GB | **永久 dead** | 4/52 |

## 3. 分析

### 3.1 MHC 并非永久死亡

关键发现：**MHC 在所有 4-stream 配置中 step 400-600 后自动复活**（最终 DOD 只剩 exit_ctrl dead）。之前 Matrix 1/9 观察到的 "MHC dead" 只是前期 warmup 阶段的瞬态现象。

唯一永久死亡的是 MH5 (8 streams) — 路由空间 8×8 太大，Sinkhorn 正则化无法在有限信号下建立差异化。

### 3.2 两条独立的改善路径

**路径 A: 提高 alpha（温度系数）**
- MH3 (alpha=0.10): loss 降 8.2%, MHC step 400 复活
- MH1 (alpha=0.03): loss 降 8.0%, MHC step 600 复活
- 更高 alpha → softmax 输出更尖锐 → routing 差异化更快

**路径 B: 减少 streams**
- MH4 (2 streams): loss 降 8.4%, MHC step 400 复活, v2_rank=6 最佳
- 2 stream 的 routing 矩阵只有 2×2 → Sinkhorn 更容易收敛
- VRAM 最低 (11.14 GB)

两条路径效果几乎相同（8.2% vs 8.4%），但 MH4 的 v2_rank=6/52 优于 MH3 的 4/52。

### 3.3 MH2 (alpha=0.02) 为什么最差？

MH2 的 loss 比 MH0 (alpha=0.01) 更差（+3.2%）。可能解释：
- alpha=0.02 处于一个不稳定区间：比 0.01 高到足以让 routing 尝试差异化，但不够高到快速收敛到有意义的模式
- 导致前期 routing 震荡，干扰了主损失优化
- alpha 响应曲线不是单调的，存在 "死谷"

### 3.4 数据量的影响

你的直觉部分正确：当前数据集（427 packs × 2048 tokens ≈ 0.87M tokens 有效）确实偏小。但实验证明 MHC 在 1500 步后**所有 4-stream 配置都能复活** — 问题不是"信号不足导致 MHC 永久死亡"，而是"warmup 阶段的假死需要 alpha/streams 调优来加速唤醒"。

更大数据集下 MHC 的差异化应该更明显。当前实验已经确认 MHC 是有效的，只是需要正确的超参数。

## 4. 结论

### 推荐配置: MH4 (streams=2, alpha=0.01)

```
--mhc_streams 2 --mhc_alpha_init 0.01
```

**理由**:
1. **loss 最低** (2.6914, -8.4% vs baseline)
2. **v2_rank 最高** (6/52, 表征多样性最好)
3. **VRAM 最低** (11.14 GB, 2 stream 参数少)
4. **MHC 快速复活** (step 400)

备选: MH3 (alpha=0.10, streams=4) — loss 接近 (2.6963)，如果 2 stream 容量不够未来可考虑。

### 正式预训练完整配置

结合 M1/M9/M7/M10 所有结果：

```
# 架构
--hidden_size 768 --compression_layers 44 --num_attention_heads 12 --num_key_value_heads 3
--reason_shared_depth 2 --mamba_chunk_size 32

# AttnRes (M9 AR1)
--attnres_compress_mode paper --attnres_reason_mode legacy

# MHC (M10 MH4)
--mhc_streams 2 --mhc_alpha_init 0.01

# 训练效率 (M7 GL1)
--accumulation_steps 2 --batch_size 1

# World-JEPA (M1 B2')
--world_sigreg_weight 0.10 --world_mask_ratio 0.25
```

---

*实验耗时: 2h48m (09:07 - 11:55)*  
*脚本: `minimind/scripts/run_matrix10_mhc.sh`*
