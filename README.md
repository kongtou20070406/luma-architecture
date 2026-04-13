# Luma Architecture

**Luma** — 一个从零设计的小型语言模型架构，核心是 **CR-Loop**（Compress → Reason Loop）：用 Mamba3 SSM 做高效一次性压缩，用**能量梯度下降**（Phase E）做深度思考，用赫布可塑性实现人格强化。

## 架构概览

```
Input → FactorizedEmbedding → CompressionZone ──→ Phase E Reason Loop ──→ LM Head → Output
                                   │                      ↑ ↓
                                   │              c_t (人格/情绪基调)
                                   │              introspection (自省流)
                                   │              Phase E damped: h ← (1-η)h + η·F(h)
                                   │              Hebbian associative memory
                                   │              PC error correction
                                   │                      ↑
                                   └── block_reprs ───────┘
```

**数据流**：
1. **Embedding** 将 token 映射到 hidden space（分解式，省参数）
2. **CompressionZone** 单次前向，混合 SSM+Attention，输出压缩表征 + 各 block 的摘要 `block_reprs`
3. **Phase E Reason Loop**：外层 loop（最多 4 轮）× 内层 damped fixed-point iteration（K=3 步）
   - **Phase E damped**：`h ← (1-η)h + η·F(h)`，η=0.5，K=3 步逼近不动点 h* = F(h*)
   - 理论上等价于在 `E(h) = 0.5·‖h - F(h)‖²` 上做能量梯度下降，一阶近似
   - **人格流 `c_t`**：64 维慢变量，方向稳定（ct_perp≈0，见下文 c_t 范式），范数随经验增长
   - **自省流**：Memory Token K=4 + CMDA 双向调制 + Mamba3 序列建模 → 产出 c_t
   - **赫布关联记忆**：`c_t += surprise × hebb(δh ⊗ prev_c_t)` 人格强化
   - **PC 误差修正**：`h -= α × (h - c_t预测(h))` 情绪反应
   - **Loop LoRA**：per-loop 低秩适配，让每轮循环差异化
   - **MoR**：Token-level depth routing，简单 token 早退出
4. **LM Head** 输出下一 token 概率

**关键特性：**
- CompressionZone 只执行一次，循环只在 Reason Loop 内部
- Phase E body `F(h) = shared_layers(h)` 每步用同一组参数（weight tying + 深度扩展）
- damped 模式避开 `autograd.grad(create_graph=True)` 的 bf16 二阶导数值不稳定问题
- 所有 `LumaZCRMSNorm` **无可学习 scale**（真正的 RMSNorm），保证长训练范数不漂移

### 模块详细拆解

#### 1. Embedding 层

```
Token IDs → FactorizedEmbedding → [B, T, 768]
```

| 类 | 说明 |
|---|---|
| `FactorizedEmbedding` | vocab→256 dim→768 dim 两步映射，节省 80%+ embedding 参数 |
| `FactorizedLMHead` | 768→256→vocab 反向映射，与 embedding 共享权重 |

#### 2. 压缩区 (CompressionZone)

```
[B, T, 768] → 12 层混合 SSM+Attn → [B, T, 768] + block_reprs[12]
```

| 类 | 层数 | 说明 |
|---|---|---|
| `CompressionMambaLayer` | ~10 层 | Mamba3 MIMO SSM (d_state=192, expand=2) |
| `CompressionRetrievalLayerSWA` | ~2 层 | 滑窗注意力（seq>window 时 chunked SWA 自动启用） |
| `CompressionBlockAttentionResiduals` | 每层 | 跨 block 注意力残差 — 后层可以回看前层摘要 |
| `LumaSwiGLUFFN` | 每层 | SwiGLU FFN (768→3072→768) |

**规模决策**：`compression_layers=12` 是 Phase E damped 在 bf16 + 216M 下的稳定上限。扩到 16 层会在 seq=2048 下直接溢出（Gap 13 v9/v10 验证）。每层输出一个 `block_repr`（层摘要），供推理区的 UnifiedAttnRes 回看。

#### 3. 推理区 — Phase E Damped Reason Loop

```python
# 外层 loop (outer): 最多 reason_loops=4 轮
for outer_loop in range(reason_loops):
    c_t → CTInjection → h += proj(c_t)    # 注入人格偏置
    # 内层 loop (inner): Phase E damped K=3 步
    for k in range(K_max):
        h_new = F(h)                        # F = shared_layers (depth=2)
        h = (1 - η) × h + η × h_new         # damped fixed-point, η=0.5
    h = UnifiedAttnRes(h, block_reprs)     # 回看压缩区
    h = PCErrorCorrector(h, c_t)           # 情绪反应（预测编码）
    # slow update: 更新 c_t
    memory → MemoryTokenReader(h)
    h → CMDAModulation(h, c_t)
    c_t_new = IntrospectionStateStream(h)
    c_t += NeuromodulatedCTWriter(c_t_new, c_t, δh, surprise)  # 人格强化
    if ExitController.should_exit: break
```

**Phase E damped 的等价性**：当 `‖J_F‖` 小时，`∇_h E ≈ (h - F(h))`，所以 `h - η·∇E = (1-η)h + η·F(h)`。damped 模式是 Phase E 能量梯度的一阶近似，理论保留所有核心性质（不动点收敛 + 构造性收缩），工程上完全避开 `autograd.grad(create_graph=True)` 的 bf16 数值陷阱。

##### 3a. 共享推理层 (LumaReasonSharedLayer × 2 — Phase E body)

| 子模块 | 说明 |
|--------|------|
| `ReasonMambaLayer` | Mamba3 SSM (d_state=192) |
| `GatedDiffAttnFoXSWA` | Gated Differential Attention + FoX + 滑窗 |
| `LumaSwiGLUFFN` | SwiGLU FFN + FiLM conditioning (c_t 调制) |
| `CTInjection` | c_t → Linear → broadcast add 注入主流 |
| Loop LoRA | `Embedding(20, 768×32)` per-loop 低秩适配 |
| Time Conditioning | `Linear(2→768)` 注入循环位置 [t, dt] |
| Loop FFN Gate | 可选，per-loop sigmoid gate 控制 FFN 强度 |

4 层共享权重但每轮通过 LoRA + Time Conditioning 差异化。

##### 3b. 自省流 (IntrospectionStateStream)

```
h [B,T,768] → MemoryTokenReader (K=4 cross-attention) → [B,4,96]
                         ↓
            Mamba3 layer1 → Mamba3 layer2 → c_t_head → c_t [B,64]
                         ↓
            CMDA: c_t→sigmoid gate 调制主流 + spatial attention 回传
```

| 类 | 说明 |
|---|---|
| `MemoryTokenReader` | 4 个可学习 query 对主流做 cross-attention，残差累积 |
| `IntrospectionStateStream` | 2 层 Mamba3 (meta_dim=96)，产出 c_t (64 dim) |
| `CMDAModulation` | c_t→per-channel sigmoid gate 调制 h + spatial attention pooling 回传 |
| `BiXTCrossAttention` | 可选，memory tokens 和主流双向 cross-attention |

##### 3c. 赫布关联记忆 (NeuromodulatedCTWriter)

```
surprise = 1 - self_check_score  (或 JEPA prediction error)
gain = 1 + σ(MLP(surprise))                    # 调制强度 [1, 2]
hebb_term = hebb_out(hebb_proj_h(δh) ⊙ hebb_proj_c(prev_c_t))  # rank=32 低秩外积
c_t = prev_c_t + gain × Δc_t + surprise × hebb_term
```

surprise 高时强写入新关联，surprise 低时保持记忆 → 防灾难性遗忘。

##### 3d. 预测编码修正 (PCErrorCorrector)

```
pred_h = MLP(c_t) → [B, 1, 768]     # 自省流预测主流状态
error = h - pred_h                    # 预测误差
h = h - α × error                    # 抑制已知，保留新信息 (α=0.1)
```

PC 和赫布协同：PC 过滤掉 c_t 已知的信息 → δh 是纯新信号 → 赫布写入更精准。

##### 3e. JEPA 系统

| 类 | 输入 | 预测目标 | 说明 |
|---|---|---|---|
| `SelfJEPAResidualPredictor` | c_t, δh | Δc_t (下一轮 c_t 变化) | 自省流自我预测 |
| `SelfJEPAProgressShapeHead` | c_t | 训练进展方向 | 渐进式学习曲线塑形 |
| `LeWorldModelStyleJEPA` | h (masked) | h (unmasked) | token 级世界模型 |
| `CtWorldJEPA` | c_t 序列 | 被 mask 的 c_t 步 | c_t 轨迹预测 |

##### 3f. 退出控制 (ExitController)

```
exit_logit = bias + Σ(weight_i × signal_i) - gain_weight × predicted_gain
                                              + jepa_surprise_weight × self_error
```

| 信号 | 含义 | 权重方向 |
|------|------|----------|
| delta_signal (1 - \|δh\|) | h 不再变化 | + (退出) |
| self_signal (1 - JEPA err) | JEPA 预测准 | + (退出) |
| world_signal (1 - world err) | 世界模型收敛 | + (退出) |
| self_check_signal | 自检一致 | + (退出) |
| predicted_gain | 预测还有收益 | - (继续) |
| jepa_surprise | JEPA 预测不准 | - (继续) |
| entropy_proxy | 输出不确定 | - (继续) |
| confidence_gap | top-2 gap 小 | - (继续) |
| ct_curvature | c_t 方向在变 | - (继续) |

##### 3g. 其他推理区组件

| 类 | 说明 |
|---|---|
| `UnifiedAttnRes` | 回看压缩区 block_reprs + 循环历史 loop_history |
| `MHCResidualStreams` | 多头通道残差流 (Sinkhorn 路由) |
| `TokenDepthRouter` (MoR) | per-token Gumbel-Sigmoid 路由，简单 token 早退出 |
| `TinySlowSelfCheckRing` | 极简自检环 — 跟踪内部叙事一致性 |
| `TinyReasoningStateRing` | 推理状态环 — 跟踪推理信任度 |
| `TrajectoryHealthProbe` | 轨迹健康探针 |
| `ExitQualityProbe` | 退出质量探针 (entropy/confidence/token sensitivity) |

### 核心组件总览

| 组件 | 实现 | 说明 |
|------|------|------|
| **压缩区** (Compression Zone) | Mamba3 MIMO + SWA/DiffAttn | 16 层混合架构，~90% SSM + ~10% Attention |
| **推理区** (Reasoning Zone) | 权重共享 Mamba3 + DiffAttn | 4 层共享，循环最多 20 次 |
| **认知流** (c_t stream) | 64-dim 持续状态 | 跨循环传递推理上下文 |
| **自省流** (Introspection) | Memory Token K=4 + Mamba3 | 选择性读取主流 → Mamba 序列建模 → c_t |
| **赫布关联记忆** (Hebbian) | rank=32 低秩双线性映射 | surprise × hebb(δh ⊗ prev_c_t) → Δc_t |
| **预测编码** (PC) | c_t → pred_h → error correction | 抑制已知部分，保留新信息 |
| **CMDA** | 双向通道调制 | c_t→sigmoid gate 调制主流 + spatial attention 回传自省流 |
| **Loop LoRA** | rank=32 per-loop adaptation | Embedding(max_loops, D×rank) 让每轮循环差异化 |
| **Self-JEPA** | 残差预测器 + SigReg ct | 自监督：预测 c_t 的变化量 |
| **World-JEPA** | 潜空间世界模型 | 学习 token 序列的因果结构 |
| **ExitController** | 多信号退出 + JEPA surprise | 推理循环自适应退出 |
| **MoR** (Token Depth Routing) | Gumbel-Sigmoid per-token mask | 简单 token 早退出，难 token 继续循环 |

### 关键技术选择

- **Mamba3 MIMO**: rank=2, chunk=32, TileLang kernel (Blackwell GPU 优化)
- **SDPA Flash Attention**: PyTorch 2.11 F.scaled_dot_product_attention
- **FP8 混精度**: ~130 Linear layers FP8 forward, BF16 backward
- **8-bit Muon + AdamW**: CPU offload 优化器状态

## 当前状态 (2026-04-13)

**生产架构 (Hero v7 = Phase E damped + 216M + scaffold JEPA)**:
- **参数量：216.955M** (bf16, fp8=0)
- hidden=768, **compression_layers=12**, heads=12/3, **reason_shared_depth=2, reason_loops=4**
- **Phase E damped**: K_max=3, η=0.5, damped_mode=1（生产形态）
- **scaffold World-JEPA**: mask=0.6, sigreg=0.05, block_mean=32（双流 JEPA + EMA target）
- 全记忆栈：introspection memory K=4 + CMDA + Loop LoRA rank=32 + Hebbian rank=32 + MoR
- Peak VRAM **~10-11 GB** @ seq=2048（RTX 5090 32GB 还有 20+ GB 余量）

**Hero v6 历史数据（seq=2048, 10538 iter, NaN 终止）**:
| 指标 | 值 | 说明 |
|------|-----|------|
| best ema | **7.38** @ step 6475 | 超越 v11 seq=1024 final 8.51 |
| best loss_lm | **0.56** @ step 9475 | dramatic minimum |
| best loss_c | **1.73** @ step 9475 | compression 真实学习 |
| Peak VRAM | 14.33 GB | 全 hero stack |
| 崩溃点 | step 10538 NaN | 累积 grad spike，根因已修复 |

**长训稳定性修复 (2026-04-13)**:
- 根因：`LumaZCRMSNorm` 的可学习 `scale` 参数被优化器推大 → 所有 residual stream / meta_state 的"归一化"变成"放大" → residual stream 协同放大 → grad spike → NaN
- 修复：删掉 `LumaZCRMSNorm.scale`，变成真正的 RMSNorm（参考 T5/Mistral），输出范数严格 = √dim
- 同时移除所有 ct_inj clamp、行范数归一化、ct 范数 hard clamp 等症状层修复
- Hero v7 (216.955M = 216.973M − 0.018M scale params) 从头跑 1 epoch 中，验证修复有效性

**c_t = 人格/情绪，不是工作记忆**（2026-04-11 范式转换）：
- **c_t 方向** = 人格（稳定，不随推理步变化 — 你做数学题时性格不会变）
- **c_t 范数** = 人格强度（由 `_ct_out_norm` 严格固定为 √64≈8）
- **h** = 工作记忆（h_diversity 自发涌现，spiral refinement 就是思考过程）
- **赫布写入** = 人格强化（surprise × hebb(δh ⊗ prev_c_t) 强化既有方向）
- **PC 误差** = 情绪反应（pred_h - h = 人格视角下的预期违背）
- **Loop LoRA** = 思考阶段（per-loop 差异化）

这解释了为什么所有强迫 c_t 方向变化的实验都恶化 loss — 相当于强迫模型每步换人格。

**实验矩阵进度**:

| Matrix | 内容 | 状态 | 最优 |
|--------|------|------|------|
| M0-M10, CR, SJ, IS, RS, DP, LD, NM+ES, PC | 4.4-4.8 早期矩阵 | 完成 | CR5/IS9/NM8/PC7/SJ1 组合 |
| H/I/J/K/M | 循环动力学 + per-layer 注入 (25+ 实验) | 完成 | G0 (无干预最优) |
| Phase E Gap 11-23 | Phase E 集成 + seq=2048 解锁 | 完成 (4.12-4.13) | damped K=3 η=0.5, 216M 稳定 |
| **Hero v7 1 epoch** | **Phase E + no-scale + scaffold JEPA** | **进行中 (4.13+)** | **80444 iter，ETA ~18h** |

详见 [WORKLOG](artifacts/WORKLOG.md) | [4.11 进展报告](docs/reports/Progress_Report_20260411.md) | [4.10 进展报告](docs/reports/Progress_Report_20260410.md) | [动力学分析技能](../Luma_Dynamics_Analysis_Skill.md)

## 项目结构

```
luma-architecture/
├── docs/
│   ├── plans/          # 执行计划
│   ├── reports/        # 实验报告
│   └── reference/      # 参考文档
├── minimind/
│   ├── model/
│   │   ├── model_minimind.py   # Luma 主模型 (~4500 LOC)
│   │   ├── mamba3_module.py    # Mamba3 MIMO wrapper
│   │   └── fp8_linear.py       # FP8 混精度 Linear
│   ├── trainer/
│   │   └── train_luma_refactor.py  # 训练脚本
│   ├── scripts/        # 实验矩阵脚本
│   └── artifacts/      # 训练产物（metrics, dynamics）
├── luma_dataset/
│   ├── synthetic/      # 公开数据集 (math, code, scifi, etc.)
│   └── rebuild_mixes.py # DataMix 重建
└── third_party/
    └── mamba-official/ # Mamba3 TileLang/Triton kernels
```

## 快速开始

```bash
# 环境: Python 3.12, PyTorch 2.11+, CUDA 12.8+, RTX 5090 (32GB)

# 推荐配置 (IS9 + NM8)
cd minimind/trainer
python train_luma_refactor.py \
  --hidden_size 768 --compression_layers 16 \
  --num_attention_heads 12 --num_key_value_heads 3 \
  --reason_shared_depth 4 --mamba_chunk_size 32 \
  --max_seq_len 2048 --batch_size 1 --reason_loops 20 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 --phase 6 \
  --world_jepa_mode full --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --enable_sigreg_ct 1 --sigreg_ct_weight 0.05 \
  --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3 \
  --enable_token_depth_routing 1 --mor_target_continue_ratio 0.7 \
  --enable_time_conditioning 1 --loop_lora_rank 32 \
  --introspection_input_mode memory --introspection_memory_tokens 4 \
  --introspection_inject_mode cmda \
  --enable_neuromod_ct 1 --neuromod_mode surprise --neuromod_hebb_rank 32
```

## 研究方向

1. **长程预训练** — G0 配置跑完整 v5 (532M tokens)，验证模型质量（进行中，0.5 epoch ~10h）
2. **人格注入** — 推理时初始化 c_t 方向 = 选择 persona，无需改训练
3. **per-layer c_t 注入** — 加强人格渗透力（当前单次注入被 ρ⁴ 衰减），需解决梯度路径 OOM
4. **acceleration-based exit** — 用 h 轨迹几何判断替代/辅助 learned exit

## 数据原则

**先变聪明，再变像 Luma。**

- smart 桶 (math + code + reason): >= 50%
- persona + empathy: >= 25% (红线)
- dialogue: 15-20%

## 硬件环境

- WSL2 + RTX 5090 (32GB VRAM)
- CR5 + NM8: seq=2048 bs=1, ~10 GB peak VRAM
- Mamba3 TileLang kernel 需要 Blackwell 架构 GPU

## License

Apache-2.0
