# Luma Architecture

**Luma** — 一个从零设计的小型语言模型架构，核心是 **CR-Loop**（Compress → Reason Loop）：用 Mamba3 SSM 做高效一次性压缩，用权重共享的推理循环做深度思考，用自适应退出控制计算预算。

## 架构概览

```
Input → FactorizedEmbedding → CompressionZone ──→ ReasoningLoop ──→ LM Head → Output
                                   │                   ↑ ↓
                                   │              c_t (认知流)
                                   │              introspection (自省流)
                                   │              UnifiedAttnRes (回看)
                                   │                   ↑
                                   └── block_reprs ────┘
```

**数据流**：
1. **Embedding** 将 token 映射到 hidden space（分解式，省参数）
2. **CompressionZone** 单次前向，混合 SSM+Attention，输出压缩表征 + 各 block 的摘要 `block_reprs`
3. **ReasoningLoop** 在压缩表征上循环推理（共享权重 × N 次），每轮：
   - 认知流 `c_t` 跨循环传递推理上下文
   - 自省流监控推理健康度
   - MoR (Mixture-of-Recursions) per-token depth routing
   - UnifiedAttnRes 回看 `block_reprs`（压缩区记忆）和 `loop_history`（循环历史）
   - Identity-biased recurrence 梯度高速公路
4. **LM Head** 输出下一 token 概率

**关键：CompressionZone 只执行一次，不参与循环。循环只发生在 ReasoningLoop 内部。**

### 核心组件

| 组件 | 实现 | 说明 |
|------|------|------|
| **压缩区** (Compression Zone) | Mamba3 MIMO + SWA/DiffAttn | 16 层混合架构，~90% SSM + ~10% Attention |
| **推理区** (Reasoning Zone) | 权重共享 Mamba3 + DiffAttn | 4 层共享，循环最多 20 次 |
| **认知流** (c_t stream) | 64-dim 持续状态 | 跨循环传递推理上下文 |
| **自省流** (Introspection) | 元认知 Mamba3 | 监控推理健康度，产出 know_gap 和 c_t |
| **UnifiedAttnRes** | 注意力残差 | 回看压缩区 block 摘要 + 循环历史 |
| **Self-JEPA** | 残差预测器 + SigReg ct | 自监督：预测下一循环的隐藏状态变化 |
| **World-JEPA** | 潜空间世界模型 | 学习 token 序列的因果结构 |
| **ExitController** | 二阶差分退出 | 推理循环自适应退出，省计算 |
| **MoR** (Token Depth Routing) | Gumbel-Sigmoid per-token mask | 简单 token 早退出，难 token 继续循环 |
| **MHC** | 多头通道残差流 | Sinkhorn 路由的多条并行残差流 |

### 关键技术选择

- **Mamba3 MIMO**: rank=2, chunk=32, TileLang kernel (Blackwell GPU 优化)
- **SDPA Flash Attention**: PyTorch 2.11 F.scaled_dot_product_attention，自动选择 flash backend
- **FP8 混精度**: 187 Linear layers FP8 forward, BF16 backward
- **8-bit Muon + AdamW**: CPU offload 优化器状态

## 当前状态 (2026-04-07)

**推荐架构 (A2)**:
- 参数量: **~286M**
- hidden=768, compression_layers=16, heads=12/3, reason_shared_depth=4
- 相比旧架构 A1 (482M, c44_d2): **loss 降低 21.6%，参数减少 41%**

**实验矩阵进度**:

| Matrix | 内容 | 状态 |
|--------|------|------|
| M0 | 架构定型 (4 配置对比) | 完成 — A1 胜出 |
| M1 | World-JEPA 变体 (5 实验) | 完成 — B2' 胜出 (LeWM sig=0.10, mask=0.25) |
| M2 | Exit Policy (6 实验) | 完成 — EX5 胜出 (20 loops + 2nd_order=0.3) |
| M5 | 超参优化 (E0-E11) | 完成 — E9 胜出 (MoR + MHC3 + threshold=0.8) |
| M7 | 吞吐优化 (GaLore) | 完成 |
| M9 | AttnRes 变体 | 完成 — AR1 胜出 |
| M10 | MHC 变体 | 完成 — MH4 胜出 |
| **SJ** | **Self-JEPA 激活 (6 实验)** | **完成 — SJ1 胜出 (SigReg ct, -5.9%)** |
| **CR** | **压缩/推理比例 (8 实验)** | **完成 — CR5 胜出 (c16_d4, -21.6%)** |
| IR | Identity-Biased Recurrence | 下一步 |

**核心发现**（2026-04-07）：
- 原 44 层压缩区过重，是推理循环深度坍缩（avg_loops=2）的主要根因
- 砍到 16 层压缩 + 4 层推理 depth → 286M 参数碾压 482M
- SigReg ct 防坍缩为 Self-JEPA 带来 -5.9% loss 改善
- c_t drift 参与退出决策全面失败，循环深度不受退出信号影响

详见 [循环深度坍缩分析报告](docs/reports/Loop_Depth_Collapse_Analysis_20260407.md)

## 项目结构

```
luma-architecture/
├── docs/
│   ├── plans/          # 执行计划
│   ├── reports/        # 实验报告
│   ├── research/       # 研究调研
│   └── reference/      # 参考文档
├── minimind/
│   ├── model/
│   │   ├── model_minimind.py   # Luma 主模型 (~3800 LOC)
│   │   ├── mamba3_module.py    # Mamba3 MIMO wrapper
│   │   └── fp8_linear.py       # FP8 混精度 Linear
│   ├── trainer/
│   │   └── train_luma_refactor.py  # 训练脚本
│   ├── scripts/        # 实验矩阵脚本
│   ├── luma_stage0/    # 优化器、动力学分析
│   └── artifacts/      # 训练产物（metrics, dynamics）
├── luma_dataset/
│   ├── synthetic/      # 公开数据集 (math, code, scifi, etc.)
│   ├── fetch_*.py      # 数据拉取脚本
│   └── rebuild_mixes.py # DataMix 重建
└── third_party/
    └── mamba-official/ # Mamba3 TileLang/Triton kernels
```

## 快速开始

```bash
# 环境
# Python 3.12, PyTorch 2.11+, CUDA 12.8+, RTX 5090 (32GB)

# 训练 (必须从 trainer/ 目录运行)
cd minimind/trainer
python train_luma_refactor.py \
  --hidden_size 768 --compression_layers 16 \
  --num_attention_heads 12 --num_key_value_heads 3 \
  --reason_shared_depth 4 --mamba_chunk_size 32 \
  --max_seq_len 2048 --batch_size 1 --reason_loops 20 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --phase 6 \
  --world_jepa_mode full --world_sigreg_weight 0.10 --world_mask_ratio 0.25 \
  --enable_sigreg_ct 1 --sigreg_ct_weight 0.05 \
  --exit_aux_weight 0.01 --exit_second_order_delta_weight 0.3 \
  --enable_token_depth_routing 1 --mor_target_continue_ratio 0.7
```

## 研究方向

1. **循环深度扩展** — Identity-biased recurrence + shortcut-consistency training
2. **压缩/推理比例优化** — 16 层压缩 + 4 层推理是当前甜区，继续探索
3. **DataMix V5** — 目标 ~1B tokens，60% 推理数据
4. **推理时 A* 搜索** — 部署时 286M 逼近 2B 推理质量

## 数据原则

**先变聪明，再变像 Luma。**

- smart 桶 (math + code + reason): >= 50%
- persona + empathy: >= 25% (红线)
- dialogue: 15-20%

## 不包含的内容

- `luma_dataset/persona_seed/` 私有语料
- 训练权重 / checkpoint / `*.pth`
- 本地虚拟环境和缓存

## 推荐阅读顺序

1. [循环深度坍缩分析](docs/reports/Loop_Depth_Collapse_Analysis_20260407.md) — 最新核心发现
2. [Matrix2 Exit Policy 报告](docs/reports/Matrix2_ExitPolicy_Report_20260406.md) — 退出策略实验
3. `minimind/model/model_minimind.py` — 核心模型实现
4. `minimind/trainer/train_luma_refactor.py` — 训练流程

## 硬件环境

- WSL2 + RTX 5090 (32GB VRAM)
- A2 架构 (c16_d4): seq=2048 bs=1, ~10 GB peak
- Mamba3 TileLang kernel 需要 Blackwell 架构 GPU

## License

Apache-2.0
