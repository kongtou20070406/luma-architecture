# Luma 执行计划 v4 (2026-04-05)

> 本文档取代 Luma_Execution_Plan_20260405.md (v3)。

---

## 0. v3 → v4 关键变化

| 变化 | 影响 |
|------|------|
| **架构瘦身 588M → 482M** | Matrix 0 选定 A1 (768h, L44)，更深更窄，loss 最优 |
| **bs=2 不可行确认** | 32GB VRAM 下 seq=2048 只能 bs=1，所有实验统一 bs=1 |
| **Flash Mamba 调研完成** | Mamba3 TileLang 已是 flash-style 实现，无额外优化空间 |
| **arxiv_dl_code 数据就绪** | 20K 条 ML/DL 研究代码替代 the-stack（即时下载，无 gating） |
| **全局实验矩阵制定** | Matrix 0-4 五阶段路线图，依赖链清晰 |

---

## 1. 已完成工作

### 1.1 架构演进
- Stage A: C5 架构胜出 (660M, 32L, shared=2)
- MIMO Mamba3 升级 (rank=2, chunk=32, TileLang kernel)
- SDPA Flash Attention 升级 (4 处手写 attention → F.scaled_dot_product_attention)
- FP8 混精度训练 (162 Linear layers → FP8 forward)

### 1.2 Matrix 0: 架构定型 (2026-04-05)

**目标**：从 588M 瘦身到 ~450M，平衡 VRAM / 容量 / 训练速度。

| 配置 | 参数量 | Step-200 Loss | Peak VRAM | Reserved | 结果 |
|---|---|---|---|---|---|
| A0 (768h, L36) | 415M | 5.94 | 9.15 GB | 14.79 GB | 基线 |
| **A1 (768h, L44)** | **482M** | **5.28** | **9.71 GB** | **16.44 GB** | **胜出 ✅** |
| A2 (832h, L36) | 473M | 6.56 | 9.47 GB | 15.23 GB | 宽但浅，不如深 |
| A3 (800h, L40) | 479M | CRASH | — | — | head_dim 不整除 |

**结论**：**深度优于宽度** — 768h×44L 在同等参数下比 832h×36L loss 低 19%。

**A1 定型配置**：
```python
hidden_size = 768
intermediate_size = 3072
compression_layers = 44
num_attention_heads = 12
num_key_value_heads = 3
reason_shared_depth = 2
factorized_vocab_dim = 256
mamba_d_state = 192
mamba_chunk_size = 32  # MIMO rank=2, chunk*rank=64
```

### 1.3 Flash Mamba 调研结论

**不需要额外集成。** Tri Dao 既是 FlashAttention 作者也是 Mamba 作者，flash-style IO-aware 优化从 Mamba-1 起就内建在 Mamba 的 CUDA/Triton kernel 中。

当前 Mamba3 + TileLang MIMO kernel 已是公开最优实现：
- Mamba-1: 自定义 CUDA kernel (kernel fusion + recomputation + SRAM residency)
- Mamba-2: Triton SSD kernel (matmul-based, tensor cores, 比 Mamba-1 快 2-8x)
- Mamba-3: TileLang + CuTe DSL (MIMO prefill 需要细粒度 shared memory 控制)

**未来增量**：PyTorch 团队 5-kernel Triton fusion (1.5-2.5x SSD 加速)，但尚未开源且只针对 Mamba-2 SSD。

**结论**：Mamba 侧无低垂果实，优化重心在模型瘦身 + 数据扩量。

### 1.4 基础设施修复
- VRAM 碎片化修复：每步 `torch.cuda.empty_cache()`
- `_fp8_act_ctx` 残留引用修复
- 过时代码清理 (fp8_activation_compress, _make_local_causal_mask, 旧脚本)

### 1.5 数据资产

| 数据集 | 条数 | 大小 | 桶归属 | 状态 |
|---|---|---|---|---|
| math_real | — | 9.5 MB | smart_math | ✅ |
| arc_agi | — | 1.7 MB | smart_math | ✅ |
| python_code | — | 14.2 MB | smart_code | ✅ |
| **arxiv_dl_code** | **20,000** | **79.2 MB** | **smart_code** | **✅ 新** |
| chinese_scifi | — | 6.0 MB | empathy | ✅ |
| persona_private | — | 3.4 MB | persona | ✅ |
| wechat_sft | — | 1.9 MB | persona | ✅ |
| zhihu_kol | — | 532.6 MB | dialogue | ✅ |

---

## 2. 全局实验矩阵路线图

> **核心原则：先解锁训练速度，再优化架构细节。**

```
═══ 第一优先级：解锁训练速度 ═══

Matrix 0 (架构定型) ──── 完成 ✅, A1 (482M)
    ↓
Matrix 1 (B' World-JEPA, ~4h) ──── 完成 ✅, B2' (sig=0.10) 胜出
    ↓
┌───────────────────────────────────────────────┐
│  M1 完成后立即并行:                            │
│                                                │
│  M9 (AttnRes 改造, ~4h) ──── 完成 ✅, AR1 胜出  │
│    └─ compress paper + reason legacy (-16% loss)│
│                                                │
│  M7 (训练吞吐量, ~1h) ──── 完成 ✅, GL1 胜出    │
│    └─ accum=2, 吞吐量 2.55x                    │
│                                                │
│  M10 (MHC 门控, ~3h) ──── 完成 ✅, MH4 胜出     │
│    └─ streams=2, MHC 救活 (-8.4% loss)         │
│                                                │
│  M5 (ES 预训练验证, ~3天) ←── 探索性验证       │
│  M6 (数据效率, ~3-5天) ←── 并行                │
└───────────────────────────────────────────────┘
    ↓
Gate F: 配置冻结 ←── M9+M7+M10 结果已确定
    ↓
正式预训练 (BP + accum=2)
    ↓
═══ 第二优先级：架构改进（预训练后）═══

M2 (Exit Policy) ←── fine-tune 阶段
M4 (MoR Routing) ←── 同上
M3 (数据扩量) ←── 并行准备数据
    ↓
M8 (A* 推理搜索) ←── 部署阶段
```

### Matrix 1: Stage B' — seq=2048 + World-JEPA

**目标**：在 A1 架构上验证 World-JEPA 变体，确定最佳世界模型训练策略。
**前置**：Matrix 0 ✅
**预计**：每实验 2100 步 × ~1.4s/step ≈ 49min，5 实验 ≈ 4h

| 实验 | JEPA mode | SIGreg | mask_ratio | EMA decay | 说明 |
|---|---|---|---|---|---|
| B0' | none | — | — | — | Baseline (无 JEPA) |
| B1' | full (LeWM) | 0.05 | 0.25 | — | 标准 LeWM |
| B2' | full (LeWM) | 0.10 | 0.25 | — | 更强正则化 |
| B3' | scaffold (EMA) | 0.05 | 0.25 | 0.996 | EMA 对比组 |
| B4' | full (LeWM) | 0.05 | 0.50 | — | 激进 masking |

**架构参数** (A1 winner):
```
hidden=768, intermediate=3072, layers=44, heads=12/3
reason_shared_depth=2, mamba_chunk_size=32
```

**训练参数**:
```
iters=2100, batch_size=1, max_seq_len=2048, reason_loops=12
fp8=1, gradient_checkpointing=1, cpu_offload_optimizer=1
```

**判胜标准**:
1. loss_lm 持续下降
2. DOD v2_rank 收敛到 ≥5
3. World-JEPA 不在 dead list（表明 JEPA 有梯度流）
4. c_t batch 方差正增长

**结果 (2026-04-05 18:48)**:

| 实验 | loss_lm (均值) | v2_rank | dead modules | 状态 |
|---|---|---|---|---|
| B0' baseline | 3.74 | 13/52 | exit_ctrl, world_jepa | 基线 |
| B1' sig=0.05 | 3.94 | 14/52 | exit_ctrl, mhc | MHC 死亡 |
| **B2' sig=0.10** | **3.91** | **6/52** | **exit_ctrl** | **胜出 ✅** |
| B3' EMA | 4.09 | 17/52 | exit_ctrl, mhc | 收敛慢，弃用 |
| B4' mask=0.50 | 3.89 | 1/52 ⚠️ | exit_ctrl, mhc | 表征坍缩，弃用 |

**关键发现**: B2'（sig=0.10）是唯一 MHC 存活、动力学健康的配置。详见 [Matrix1 报告](../reports/Matrix1_WorldJEPA_Report_20260405.md)。

### Matrix 2: Exit Policy — 预训练中培养自适应退出

**目标**：在预训练阶段打开 exit_aux_weight，让 ExitController 学会自适应退出。
**前置**：M1 + M9 + M7 + M10 ✅
**脚本**：`minimind/scripts/run_matrix2_exit_policy.sh`

| 实验 | exit_aux | 2nd_order | loops | loss_lm | vs EX0 | 结论 |
|---|---|---|---|---|---|---|
| EX0 | 0.0 | 0.0 | 12 | 2.6367 | — | baseline (exit_ctrl dead) |
| EX1 | 0.01 | 0.0 | 12 | 2.6530 | +0.6% | 无害 |
| EX2 | 0.05 | 0.0 | 12 | 2.6869 | +1.9% | 权重太高 |
| EX3 | 0.01 | 0.3 | 12 | 2.7985 | +6.1% | 12 loops 下 2nd_order 有害 |
| EX4 | 0.01 | 0.0 | 20 | 2.6455 | +0.3% | 20 loops 零 VRAM 开销 |
| **EX5** | **0.01** | **0.3** | **20** | **2.6032** | **-1.3%** | **胜出 ✅** |

**关键发现**: second_order 在 20 loops 下有益（-1.3%），在 12 loops 下有害（+6.1%）。更宽的 loop 预算让收敛检测真正有效。20 loops 零 VRAM 开销。
**胜出配置**: `--reason_loops 20 --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3`
详见 [Matrix2 报告](../reports/Matrix2_ExitPolicy_Report_20260406.md)。

**状态**: ✅ 完成

### Matrix 3: Stage C — 数据扩量

**核心数据原则：先变聪明，再变像 Luma。**

**目标**：从 ~61M tokens 扩到 ≥500M tokens。

| 实验 | 数据量 | smart% | persona+empathy% | 说明 |
|---|---|---|---|---|
| C0 | 61M tokens | 50% | 25% | 当前 DataMix v1 |
| C1 | 200M tokens | 55% | 25% | 补充 arxiv_dl + math |
| C2 | 500M tokens | 50% | 25% | 目标数据量 |
| C3 | 500M tokens | 60% | 20% | smart 偏重对比 |

**配比红线**: persona+empathy 永远 ≥25%

**判胜标准**: perplexity 持续下降 + persona/empathy eval 不崩

### Matrix 4: Stage D — MoR Per-Token Routing

**目标**：Mixture-of-Reasoning 让不同 token 走不同深度的推理循环。

| 实验 | routing 方式 | 说明 |
|---|---|---|
| D0 | fixed-depth | Baseline (所有 token 同深度) |
| D1 | token-level router | 每个 token 独立决策 exit |
| D2 | chunk-level router | 每 chunk (32 tokens) 决策 |
| D3 | budget-constrained | 总 loop budget 约束下的分配 |

### Matrix 5: ES 进化策略验证 (探索性)

**目标**：用最小代价验证 ES 在 Mamba3 上能否收敛，为未来参数扩容后的 ES 微调铺路。
**前置**：Matrix 1 完成；与 M7 **并行**
**定位**：探索性实验，不阻塞主线。单卡 ES 预训练比 BP 慢，真正价值在于：
1. **未来参数扩容**（1B+）后 BP VRAM 不够 → ES 微调是唯一选择
2. **多卡场景** → N 个扰动天然并行到 N 张卡
3. **非可微组件** → exit decision 等离散操作可直接优化

**参考论文**：
- "Evolution at Scale" (Lange et al., ICML 2025) — 14B ES 微调超越 GRPO
- "EGGROLL" (NVIDIA+Oxford, 2025) — RWKV7 ES 预训练，91% 推理吞吐量
- "ESSA" (2025) — 32B INT4 显存微调，收敛快 2-6 倍

**最小验证 (N=2 antithetic, ~3 天)**:

| 实验 | 方法 | 模型规模 | N | 训练类型 | 说明 |
|---|---|---|---|---|---|
| F0-es-n2-100m | ES antithetic | 100M | 2 | pretrain 500步 | 最小验证: 能否收敛？ |
| F1-es-n2-482m | ES antithetic | 482M | 2 | pretrain 200步 | 全尺寸 VRAM 和速度基准 |
| F2-es-n2-ft | ES antithetic | 482M | 2 | fine-tune | 预训练模型上 ES 微调 |

**ES 核心算法 (N=2 antithetic)**:
```
每步:
  1. 采样 1 个高斯扰动 ε
  2. 评估 loss(θ + σε) 和 loss(θ - σε)   ← 2 次前向，无 BP
  3. 更新: θ ← θ - lr × (loss+ - loss-) / (2σ) × ε
```

**判胜标准**:
1. F0 的 loss 在 500 步内**有下降趋势** → ES 在 Mamba3 上可收敛
2. F1 的 VRAM < 4GB → 确认 VRAM 优势
3. 记录收敛速度 vs BP baseline → 评估未来多卡 ROI

**未来路线** (如验证成功):
- 参数扩容到 1B+ 后，BP VRAM 超 32GB → 切换到 ES 微调
- 多卡可用时 → N=30 ES 并行，吞吐量线性扩展
- 非可微 exit decision → ES 直接优化端到端 reward

---

### Matrix 6: 数据效率 — EntiGraph 合成 + Perplexity 修剪

**目标**：用更少/更好的数据达到同等效果，解决 Luma 数据瓶颈。
**前置**：Matrix 1 完成（需要基准 loss 对比）
**可与 Matrix 2-4 并行**

#### Phase 6a: Perplexity 数据修剪 (~1 天)

用当前 BP 预训练的 Luma 模型计算每条训练数据的 perplexity，按 perplexity 分桶过滤。

| 实验 | 数据策略 | 保留比例 | 说明 |
|---|---|---|---|
| P0 | 全量 (baseline) | 100% | 当前 DataMix |
| P1 | 去除 top-10% 最简单 | 90% | 去掉模型已掌握的 |
| P2 | 去除 top-30% 最简单 | 70% | 更激进过滤 |
| P3 | 中等难度区间 | 50% | 只保留 "可学习" 数据 |

**参考**: ICLR 2025 "When Less is More" — 30% 数据匹配全量性能
**判胜标准**: P2/P3 的 loss ≤ P0 的 loss + 5%

#### Phase 6b: EntiGraph 合成扩充 (~3-5 天)

从 Luma 的 persona/empathy 小语料（~10MB）提取实体图，合成 10x 训练数据。

| 实验 | 源数据 | 合成目标 | 方法 |
|---|---|---|---|
| G0 | persona_private (3.4MB) | 30MB | GPT-4 实体交叉合成 |
| G1 | chinese_scifi (6MB) | 60MB | 同上 |
| G2 | G0 + G1 合并 | 90MB | 合成数据混入 DataMix |
| G3 | G2 + perplexity 过滤 | ~60MB | 合成后再修剪 |

**参考**: "Synthetic Continued Pretraining" (ICLR 2025 Oral) — 1.3M tokens → 600M tokens
**关键**: 不是简单 paraphrase（论文证明 paraphrase 饱和快），而是实体图扩展
**判胜标准**: persona/empathy eval 提升 + loss 不恶化

### Matrix 7: 训练吞吐量提升 — 等效 bs=2

**VRAM 分析结论 (2026-04-05)**：
- 优化器状态已经 8-bit，仅占 **0.67 GB** — GaLore 节省空间极小
- 瓶颈是**激活内存** (7.31 GB) 和 CUDA **碎片化** (reserved 16.44 vs peak 9.90)
- **GaLore 单独无法解锁 bs=2**

**新策略**：gradient accumulation（零 VRAM 开销）+ activation offload 验证

| 实验 | 方法 | bs | accum | 等效 bs | 说明 |
|---|---|---|---|---|---|
| GL0 | 当前 baseline | 1 | 1 | 1 | 当前 VRAM: 9.90 peak |
| GL1 | gradient accumulation | 1 | 2 | 2 | 零 VRAM 开销，速度 ~×2 step |
| GL2 | activation offload compress | 1 | 2 | 2 | CPU offload 压缩区激活 |
| GL3 | 真实 bs=2 + offload | 2 | 1 | 2 | ���试是否 OOM |
| GL4 | 真实 bs=2 + offload + empty_cache | 2 | 1 | 2 | 碎片化控制 |

**关键指标**:
- GL1: 确认 accum=2 的 loss 收敛 ≈ baseline（理论上等价）
- GL3/GL4: 真实 bs=2 是否 OOM？如果不 OOM → 速度翻倍
- 如果真实 bs=2 不行 → GL1 (accum=2) 是最佳方案

**判胜标准**: 等效 bs=2 训练不 OOM + loss 不超过 baseline 5%

**结果 (2026-04-06 04:20)**:

| 实验 | loss_lm | Peak VRAM | Wall-clock | 状态 |
|---|---|---|---|---|
| GL0 baseline | 4.2273 | 10.26 GB | 23.7 min | ✅ |
| **GL1 accum=2** | **3.9946 (-5.5%)** | **11.20 GB** | **18.6 min** | **✅ 胜出** |
| GL2 offload+accum | — | — | — | ❌ crash (FP8+offload 冲突) |
| GL3/GL4 | — | — | — | 未运行 |

**关键发现**: accum=2 吞吐量 **2.55x** (3674 vs 1441 tok/s)，预训练 ~16天 → **~6天**。
**胜出配置**: `--accumulation_steps 2 --batch_size 1`
详见 [Matrix7 报告](../reports/Matrix7_Throughput_Report_20260406.md)。

**状态**: ✅ 完成

### Matrix 9: AttnRes 改造 — Kimi Block Attention Residuals

**目标**：将当前 lerp 残差注意力替换为论文 (arxiv 2603.15031) 的 Block Attention Residuals，验证对收敛和动力学的影响。
**前置**：Matrix 1 ✅ (B2' baseline)
**预计**：每实验 2100 步 × ~1.4s/step ≈ 49min，5 实验 ≈ 4h
**脚本**：`minimind/scripts/run_matrix9_attnres.sh`

**当前 vs 论文的三个关键差异**：
1. **输出方式**：lerp (α·old + (1-α)·new) → 直接替换 (softmax attention 加权和)
2. **Query**：全局共享 pseudo_query → 每层/每块独立 pseudo_query (zero-init)
3. **Value 归一化**：V 经过 RMSNorm → V 保持 raw (只 norm K)

| 实验 | CompressionZone | ReasoningLoop | Query 类型 | 说明 |
|---|---|---|---|---|
| AR0 | legacy (lerp) | legacy (lerp) | global | Baseline = B2' |
| AR1 | paper (direct) | legacy (lerp) | per-block | 仅压缩区用论文式 |
| AR2 | legacy (lerp) | paper (direct) | per-loop | 仅推理循环用论文式 |
| AR3 | paper (direct) | paper (direct) | per-block/loop | 全量论文式 |
| AR5 | paper_global_q | paper_global_q | global | 论文输出 + 全局 query |

**实现细节**：
- `PaperBlockAttnRes`: per-block 独立 query, RMSNorm on K only, V raw, direct replace
- `PaperBlockAttnResGlobalQ`: 同上但单个全局 pseudo_query
- `PaperUnifiedAttnRes`: per-loop 独立 query, 合并 loop_history + block_reprs
- `PaperUnifiedAttnResGlobalQ`: 同上但单个全局 pseudo_query
- AR4 (input-dependent query) 暂跳过，AR3 有效再加

**判胜标准**:
1. loss_lm ≤ AR0 baseline
2. DOD v2_rank 健康 (≥5/52)
3. MHC 不死亡
4. VRAM 不显著增加 (< +1 GB)

**结果 (2026-04-06 03:28)**:

| 实验 | loss_lm | vs AR0 | Peak VRAM | v2_rank | MHC | 结论 |
|---|---|---|---|---|---|---|
| AR0 baseline | 3.6804 | — | 9.95 GB | 11/52 | dead | 基线 |
| **AR1 compress_paper** | **3.0867** | **-16.1%** | **10.46 GB** | **10/52** | **dead** | **胜出 ✅** |
| AR2 reason_paper | 4.2593 | +15.7% | 10.22 GB | 8/52 | dead | 退步 |
| AR3 full_paper | 3.5493 | -3.6% | 10.53 GB | 5/52 | dead | 表征坍缩风险 |
| AR5 paper_global_q | 3.8328 | +4.1% | 10.53 GB | 11/52 | alive | MHC 活但 loss 差 |

**关键发现**: 残差策略应因场景而异 — 压缩区(单次前馈)适合 paper AttnRes，推理循环(迭代)需要 lerp 平滑。
**胜出配置**: `--attnres_compress_mode paper --attnres_reason_mode legacy`
详见 [Matrix9 报告](../reports/Matrix9_AttnRes_Report_20260406.md)。

**状态**: ✅ 完成

### Matrix 10: MHC 门控调优

**目标**：调优 MHC 的 alpha 温度系数和 stream 数量，救活 MHC 梯度流。
**前置**：Matrix 9 ✅ + Matrix 7 ✅
**脚本**：`minimind/scripts/run_matrix10_mhc.sh`

| 实验 | alpha | streams | loss_lm | vs MH0 | MHC 复活 | 结论 |
|---|---|---|---|---|---|---|
| MH0 | 0.01 | 4 | 2.9385 | — | step 600 | 基线 |
| MH1 | 0.03 | 4 | 2.7041 | -8.0% | step 600 | alpha 提升有效 |
| MH2 | 0.02 | 4 | 3.0322 | +3.2% | step 600 | 死谷区间 |
| MH3 | 0.10 | 4 | 2.6963 | -8.2% | step 400 | 高温有效 |
| **MH4** | **0.01** | **2** | **2.6914** | **-8.4%** | **step 400** | **胜出 ✅** |
| MH5 | 0.01 | 8 | 3.2507 | +10.6% | 永久 dead | 路由太难 |

**关键发现**: MHC 并非永久死亡，step 400-600 后自动复活。streams=2 是最佳选择 — 路由简单、VRAM 最低、v2_rank 最高。
**胜出配置**: `--mhc_streams 2 --mhc_alpha_init 0.01`
详见 [Matrix10 报告](../reports/Matrix10_MHC_Report_20260406.md)。

**状态**: ✅ 完成

### Matrix 8: 推理时 A* 搜索 — 部署增强

**目标**：验证 A* 树搜索能否让 482M Luma 在推理任务上逼近 2B 模型。
**前置**：正式预训练完成（需要成熟模型）
**这是部署阶段优化，不影响训练流程**

| 实验 | 搜索策略 | beam width | 说明 |
|---|---|---|---|
| S0 | Greedy decode (baseline) | 1 | 标准自回归生成 |
| S1 | Beam search | 4 | 经典 beam search |
| S2 | A* search (self-eval) | 4 | 模型自评估引导的 A* |
| S3 | A* search (self-eval) | 8 | 更宽搜索 |
| S4 | Best-of-N sampling | 8 | 采样 N 个 → 选最好 |

**SSM 特殊考量**:
- Transformer 的 KV cache 天然支持树搜索回溯
- Mamba3 SSM 需要 **state checkpoint 机制** — 在每个搜索节点保存/恢复 SSM hidden state
- 实现复杂度高于 Transformer，但潜在收益也大（SSM state 比 KV cache 小得多）

**参考**: "Test-Time A* Search for SLMs" (2025) — 1B + TTS 超越 8B
**判胜标准**: 
- 数学/代码任务准确率提升 ≥30%
- 延迟增加 < 5x（可接受的推理时间代价）

---

### 补充技术：DistiLLM 蒸馏

正式预训练后可选的知识注入步骤：
- 从 Qwen-72B / DeepSeek-V3 蒸馏到 Luma
- 使用 DistiLLM (skew KL) 或 Dual-Space (logit + hidden state) 蒸馏
- 与 Mamba3 兼容（蒸馏目标函数架构无关）
- 预计 1-2 天 fine-tune

详细调研见 [Research Report](../research/Luma_Research_Report_SmallModel_NoBP.md)。

---

## 3. 经验教训汇总

| 发现 | 原因 | 解决方案 | 教训 |
|------|------|---------|------|
| 手写 softmax attention OOM | [bs,16,2048,2048] FP32 矩阵 | SDPA flash kernel | **永远不要手写 attention** |
| CUDA cache 碎片化 | reserved 从 16GB 增长到 29GB+ | `empty_cache()` 每步 | reserved ≠ allocated |
| MIMO 测试 VRAM 估算偏低 | 用 228M 测试，实际 628M | 必须用真实模型估算 | **VRAM 估算用真实配置** |
| bs=2 seq=2048 OOM | 32GB 不够 482M+bs2+seq2048 | 固定 bs=1 | **先确认 VRAM 再设计实验** |
| 深度 > 宽度 | 768h×44L > 832h×36L (同参数量) | A1 胜出 | **Scaling law: 深度更高效** |
| fp8_activation_compress NaN | seq≥1024 时精度不够 | 已删除 | 不保留已知有毒的代码路径 |
| 孤儿 GPU 进程 | nohup 子进程 kill 后残留 | `nvidia-smi` 检查 + kill -9 | 启动前必须检查 |
| head_dim 必须整除 | 800/12=66.67 | hidden 必须被 heads 整除 | **配置前先算 head_dim** |
| the-stack 下载极慢 | parquet index 巨大 | 换 arxiv_dl_code (1min 完成) | 优先选小数据集验证 |

---

## 4. 硬件约束

- **GPU**: RTX 5090 32GB
- **VRAM 实测**: 482M 模型 + seq=2048 + bs=1 = 9.71 GB peak, 16.44 GB reserved
- **bs=2 不可行**: 所有 seq=2048 实验固定 bs=1
- **训练速度**: ~1.4s/step (seq=2048, bs=1)
- **Mamba3**: TileLang MIMO kernel, rank=2, chunk=32 (shared memory 极限)
- **FP8**: 162 Linear layers FP8 forward, BF16 backward

---

## 5. 时间线估算

| 优先级 | 阶段 | 预计耗时 | 状态 | 说明 |
|--------|------|---------|------|------|
| — | Matrix 0 (架构定型) | ~1h | **完成 ✅** | A1 (482M) 胜出 |
| — | Matrix 1 (B' World-JEPA) | ~4h | **完成 ✅** | B2' (sig=0.10) 胜出 |
| — | Matrix 9 (AttnRes 改造) | ~4h | **完成 ✅** | AR1 胜出 (compress paper + reason legacy) |
| — | Matrix 7 (训练吞吐量) | ~1h | **完成 ✅** | GL1 胜出 (accum=2, 2.55x 吞吐) |
| — | Matrix 10 (MHC 门控) | ~3h | **完成 ✅** | MH4 胜出 (streams=2, -8.4%) |
| P1 | Matrix 5 (ES 验证) | ~3 天 | 探索性 | N=2 快速验证能否收敛 |
| P1 | Matrix 6 (数据效率) | 3-5 天 | 与 M5/M7 并行 | EntiGraph 合成 + PPL 修剪 |
| P1 | Matrix 3 (数据扩量) | 并行准备 | 数据收集中 | 不阻塞训练 |
| — | Matrix 2 (Exit Policy) | ~3h | **完成 ✅** | EX5 胜出 (20 loops + 2nd_order, -1.3%) |
| P2 | Matrix 4 (MoR Routing) | 1-2 周 | 预训练后 | fine-tune 阶段加入 |
| — | Gate F (配置冻结) | 0.5 天 | — | M9 + M7 结果决定 |
| — | **正式预训练** | **~10-16 天** | — | **BP + accum=2 (主线)** |
| P3 | Matrix 8 (A* 推理搜索) | ~3 天 | 部署阶段 | 推理时免费提升质量 |

**决策树**:
```
M9 (AttnRes 改造)           M7 (训练吞吐量)
  ├─ paper 胜出 → 更新 AttnRes   ├─ accum=2 收敛 → 等效 bs=2
  └─ legacy 胜出 → 保持现状       └─ 真实 bs=2 不 OOM → 速度翻倍

M5 (ES N=2 收敛?)
  ├─ YES → 留作多卡方案
  └─ NO  → 放弃 ES
```

**VRAM 分析修正 (2026-04-05)**：
- 优化器状态已 8-bit，仅 **0.67 GB** — GaLore/MLorc/APOLLO 节省极小
- 瓶颈是激活内存 (7.31 GB)，不是优化器
- gradient accumulation=2 零 VRAM 开销，是最务实的提速路径

**预训练时间估算** (482M, 2B tokens):
| 方案 | bs | tokens/step | 速度 | 耗时 |
|------|-----|-------------|------|------|
| BP 当前 | 1 | 2048 | 1.4s/step | ~16 天 |
| **BP + accum=2** | **1** | **4096** | **~2.8s/step** | **~10 天** |
| BP + 真实 bs=2 | 2 | 4096 | ~2s/step (如不 OOM) | ~7 天 |

**预训练时间重估** (482M 模型):
- 482M < 588M → 可能更快
- seq=2048 bs=1 → 2048 tokens/step → 1.4s/step
- 目标 2B tokens → 976K steps → ~16 天
- 目标 5.3B tokens → 2.6M steps → ~42 天
- **推荐先跑 2B tokens 验证效果**

---

## 6. 立即行动清单

1. ✅ Matrix 0 完成，A1 (482M) ���型
2. ✅ arxiv_dl_code 20K 条拉取完成
3. ✅ Matrix 1 完成，B2' (sig=0.10, mask=0.25) 胜出
4. ✅ Matrix 9 完成，AR1 胜出 (compress paper + reason legacy, loss -16.1%)
5. ✅ Matrix 7 完成，GL1 胜出 (accum=2, loss -5.5%, 吞吐 2.55x)
6. ✅ Matrix 10 完成，MH4 胜出 (streams=2, loss -8.4%, MHC 救活)
7. **→ Gate F: 配置冻结** — M1+M9+M7+M10 结果已确定正式预训练配置
8. **→ 正式预训练启动** — 完整配置见 M10 报告
9. → Matrix 5 (ES N=2): 探索性验证（不阻塞主线）
10. → 整合 arxiv_dl_code: 更新 DataMix，重建 pretrain 数据
