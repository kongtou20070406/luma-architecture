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
2. **CompressionZone** 单次前向，44 层混合 SSM+Attention，输出压缩表征 + 各 block 的摘要 `block_reprs`
3. **ReasoningLoop** 在压缩表征上循环推理（2 层共享权重 × 12 次 = 24 层等效深度），每轮：
   - 认知流 `c_t` 跨循环传递推理上下文
   - 自省流监控推理健康度
   - UnifiedAttnRes 回看 `block_reprs`（压缩区记忆）和 `loop_history`（循环历史）
4. **LM Head** 输出下一 token 概率

**关键：CompressionZone 只执行一次，不参与循环。循环只发生在 ReasoningLoop 内部。**

### 核心组件

| 组件 | 实现 | 说明 |
|------|------|------|
| **压缩区** (Compression Zone) | Mamba3 MIMO + SWA/DiffAttn | 44 层混合架构，~90% SSM + ~10% Attention |
| **推理区** (Reasoning Zone) | 权重共享 Mamba3 + DiffAttn | 2 层共享，循环 12 次 = 24 层等效深度 |
| **认知流** (c_t stream) | 64-dim 持续状态 | 跨循环传递推理上下文 |
| **自省流** (Introspection) | 元认知 Mamba3 | 监控推理健康度，产出 know_gap 和 c_t |
| **UnifiedAttnRes** | 注意力残差 | 回看压缩区 block 摘要 + 循环历史 |
| **Self-JEPA** | 残差预测器 | 自监督：预测下一循环的隐藏状态变化 |
| **World-JEPA** | 潜空间世界模型 | 学习 token 序列的因果结构 |
| **ExitController** | 二阶差分退出 | 推理循环提前退出，省计算 |
| **MHC** | 多头通道残差流 | Sinkhorn 路由的多条并行残差流 |

### 关键技术选择

- **Mamba3 MIMO**: rank=2, chunk=32, TileLang kernel (Blackwell GPU 优化)
- **SDPA Flash Attention**: PyTorch 2.11 F.scaled_dot_product_attention，自动选择 flash backend
- **FP8 混精度**: 187 Linear layers FP8 forward, BF16 backward
- **8-bit Muon + AdamW**: CPU offload 优化器状态

## 当前状态 (2026-04-05)

**定型架构 (A1)**:
- 参数量: **482M**
- hidden=768, layers=44, heads=12/3, shared_depth=2
- Peak VRAM: 9.90 GB (seq=2048, bs=1, RTX 5090)

**实验矩阵进度**:

| Matrix | 内容 | 状态 |
|--------|------|------|
| M0 | 架构定型 (4 配置对比) | 完成 — A1 胜出 |
| M1 | World-JEPA 变体 (5 实验) | 完成 — B2' 胜出 (LeWM sig=0.10, mask=0.25) |
| M7 | GaLore 优化器 (解锁 bs=2) | 下一个 (最高优先级) |
| M5 | ES 进化策略验证 (N=2) | 探索性 (与 M7 并行) |
| M6 | 数据效率 (EntiGraph + PPL 修剪) | 规划中 |
| M2 | Exit Policy | 预训练后 |
| M4 | MoR Per-Token Routing | 预训练后 |
| M8 | 推理时 A* 搜索 | 部署阶段 |

详见 [执行计划 v4](docs/plans/Luma_Execution_Plan_v4_20260405.md)

## 项目结构

```
luma-architecture/
├── docs/
│   ├── plans/          # 执行计划（v4 为当前版本）
│   ├── reports/        # 实验报告
│   ├── research/       # 研究调研（小模型优化、无BP训练）
│   └── reference/      # 参考文档、loss 说明
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
  --hidden_size 768 --compression_layers 44 \
  --num_attention_heads 12 --num_key_value_heads 3 \
  --reason_shared_depth 2 --mamba_chunk_size 32 \
  --max_seq_len 2048 --batch_size 1 --reason_loops 12 \
  --fp8 1 --use_gradient_checkpointing 1 \
  --cpu_offload_optimizer 1 \
  --phase 6 \
  --world_jepa_mode full --world_sigreg_weight 0.10 --world_mask_ratio 0.25
```

## 研究方向

1. **GaLore 优化器** — 梯度低秩投影，解锁 bs=2，预训练速度翻倍
2. **ES 进化策略** — 无反向传播训练，为未来参数扩容和多卡场景储备
3. **EntiGraph 数据合成** — 从小语料合成 10x 训练数据
4. **推理时 A* 搜索** — 部署时 500M 逼近 2B 推理质量
5. **循环深度扩展** — 更多推理循环，更少唯一参数

详见 [研究报告](docs/research/Luma_Research_Report_SmallModel_NoBP.md)

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

1. [Luma_Execution_Plan_v4](docs/plans/Luma_Execution_Plan_v4_20260405.md) — 当前全局路线图
2. [Matrix1 报告](docs/reports/Matrix1_WorldJEPA_Report_20260405.md) — 最新实验结果
3. [Research Report](docs/research/Luma_Research_Report_SmallModel_NoBP.md) — 技术调研
4. `minimind/model/model_minimind.py` — 核心模型实现
5. `minimind/trainer/train_luma_refactor.py` — 训练流程

## 硬件环境

- WSL2 + RTX 5090 (32GB VRAM)
- seq=2048 bs=1: 9.90 GB peak, ~1.4s/step
- Mamba3 TileLang kernel 需要 Blackwell 架构 GPU

## License

Apache-2.0
