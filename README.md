# Luma Architecture

**Luma** — 一个从零设计的小型语言模型架构，核心是 **CR-Loop**（Compress → Reason Loop）：用 Mamba3 SSM 做高效一次性压缩，用权重共享的推理循环做深度思考，用自适应退出控制计算预算。

## 架构概览

```
Input → FactorizedEmbedding → CompressionZone ──→ ReasoningLoop ──→ LM Head → Output
                                   │                   ↑ ↓
                                   │              c_t (认知流)
                                   │              introspection (自省流)
                                   │              UnifiedAttnRes (回看)
                                   │              Hebbian associative memory (赫布关联记忆)
                                   │              PC error correction (预测编码修正)
                                   │                   ↑
                                   └── block_reprs ────┘
```

**数据流**：
1. **Embedding** 将 token 映射到 hidden space（分解式，省参数）
2. **CompressionZone** 单次前向，混合 SSM+Attention，输出压缩表征 + 各 block 的摘要 `block_reprs`
3. **ReasoningLoop** 在压缩表征上循环推理（共享权重 × N 次），每轮：
   - 认知流 `c_t` 跨循环传递推理上下文
   - 自省流监控推理健康度，通过 Memory Token + CMDA 双向交互
   - Hebbian associative memory: surprise-gated 赫布外积，防灾难性遗忘
   - PC error correction: 自省流预测主流状态，误差修正主流
   - Loop LoRA: per-loop 低秩适配，让每轮循环做不同计算
   - Time Conditioning: 循环位置 [t, dt] 注入
   - MoR (Mixture-of-Recursions) per-token depth routing
   - UnifiedAttnRes 回看 `block_reprs`（压缩区记忆）和 `loop_history`（循环历史）
4. **LM Head** 输出下一 token 概率

**关键：CompressionZone 只执行一次，不参与循环。循环只发生在 ReasoningLoop 内部。**

### 核心组件

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

## 当前状态 (2026-04-08)

**推荐架构 (CR5 + NM8)**:
- 参数量: **~293M**
- hidden=768, compression_layers=16, heads=12/3, reason_shared_depth=4
- Time Conditioning + Loop LoRA rank=32 + Memory K=4 + CMDA
- **Hebbian rank=32: loss -23.9%** (500 步实验)

**实验矩阵进度**:

| Matrix | 内容 | 状态 | 最优 |
|--------|------|------|------|
| M0 | 架构定型 (4 配置) | 完成 | A1 → A2 (CR5) |
| M1 | World-JEPA 变体 | 完成 | B2' (LeWM sig=0.10, mask=0.25) |
| M2 | Exit Policy | 完成 | EX5 (20 loops + 2nd_order=0.3) |
| M5 | 超参优化 | 完成 | E9 (MoR + MHC3 + threshold=0.8) |
| CR | 压缩/推理比例 | 完成 | CR5 (c16_d4, **-21.6%**) |
| SJ | Self-JEPA 激活 | 完成 | SJ1 (SigReg ct, **-5.9%**) |
| IS | 自省流优化 (10 实验) | 完成 | IS9 (Memory K=4 + CMDA, **-15.6%**) |
| RS | 推理结构 (9 实验) | 完成 | RS5 (LoRA rank=32, **-20%**) |
| DP | 循环深度推送 (10 实验) | 完成 | DP2 (Time Conditioning, **-8.7%**) |
| LD | 循环深度 v2 (10 实验) | 完成 | LD1 (bias=-1, avg=2.4 安全推深) |
| **NM+ES** | **赫布+退出信号 (28 实验)** | **完成** | **NM8 (hebb32, -23.9%)** |
| **PC** | **预测编码 (8 实验)** | **完成** | **PC7 (PC+hebb32, -11.5%)** |
| **Long** | **1000 步长训练验证** | **进行中** | 待定 |

**核心发现**（2026-04-08）：
- **Hebbian rank=32 是全场最强单项改进 (-23.9%)**，核心价值是防灾难性遗忘
- Rank 消融呈双峰：rank=16 (-14.9%) 和 rank=32 (-23.9%) 是两个峰，中间 rank=20/24 塌陷
- FUSE 组合全军覆没：hebb32 单独最强，叠加 ES 信号反而干扰 (+16~24%)
- PC 符号修正后和 hebb 协同：PC(修正)+hebb32 = -11.5%（修正前 +3.5%）
- 自省流瓶颈（mean pool + 1536→96 压缩）是循环坍缩的主因之一
- warmup=200 可训出 avg=8.6 深循环，但需要配合赫布才有用

详见 [赫布可塑性分析报告](docs/reports/Hebbian_Neuromodulation_Analysis_20260408.md) | [循环深度坍缩分析](docs/reports/Loop_Depth_Collapse_Analysis_20260407.md)

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

1. **赫布+PC 协同** — 修正版 PC 和赫布协同防遗忘，长训练验证中
2. **Warmup + Hebbian** — warmup 训出深循环能力，赫布让深循环有用
3. **推理时 PC** — 不改训练，推理循环内用 PC 收敛检测替代 learned exit
4. **自省流 v2** — 滑窗注意力补充 SSM、NTM-style slot memory

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
