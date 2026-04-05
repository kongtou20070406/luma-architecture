# Luma 执行计划 v2 (2026-04-04)

> 本文档取代 Luma_Execution_Plan_20260402.md。旧 Luma_v0.7.2_Agent_MasterPlan.md 保留为历史参考。

---

## 1. 已完成工作 (归档)

### 1.1 架构验证 -- 312M 基线 (Phase 0-4, Gate A-D 全部通过)

**结论**: Luma CDR-Loop 架构可行，核心机制全部验证通过。

| Gate | 验证内容 | 关键结果 |
|------|----------|---------|
| A | c_t 并行流 + 慢环 | c_t 注入有效, KL 非零, 循环步数有方差 |
| B | Self JEPA (delta c_t) | loss 下降, 方差健康, 未塌缩 |
| C | World JEPA | scaffold 稳定; full 在 rollout 一致性上更强 |
| D | Rollout + 停滞诊断 | one-step main + light two-step auxiliary 确认为最优监督形式 |

**Phase 4 最优配置 (35 个实验)**:
- `self_progress_shape_weight=0.05` + `self_rollout_weight=0.1` (F2 组合)
- mode1=52.0% (全场最佳梯度分布)
- near3 加权模式有毒 (A2/A6/D4 反复坍缩), 已永久禁用
- 三模块叠加不如双模块 (F4 87.4% vs F2 52.0%)

**数据结论 (312M 容量限制)**:
- G5 (persona + 真实数学) mode1=43.9% 全场最佳
- 312M 无法消化 5 类以上数据 (G1/G2 rank=2 坍缩)
- 数据质量 > 数量 (G4 2.2万 不如 G5 6000)
- H2 (加 5% python) 可行 mode1=53.8%; H3 (加 ARC) 直接坍缩

### 1.2 模型扩容 -- S2 级 570M (进行中)

**扩容路径**: 0.3B -> 0.6B (先加深再加宽)

**S2 配置**:
```python
hidden_size = 1024          # 768 -> 1024
intermediate_size = 4096    # 3072 -> 4096
num_attention_heads = 16    # 12 -> 16
num_key_value_heads = 4     # 3 -> 4
mamba_d_state = 256         # 192 -> 256
factorized_vocab_dim = 256  # 192 -> 256
```

**S2 变体对比 (1500 步短程)**:

| 变体 | 层数 | Depth | c_t | meta | 参数 | 最终 mode1 | 判定 |
|------|------|-------|-----|------|------|-----------|------|
| S2-A | 32 | 2 | 96 | 128 | 660M | 49.7% | 基线 |
| C1 | 24 | 2 | 96 | 128 | 533M | 76.9% | 较差 |
| C2 | 24 | 3 | 96 | 128 | 569M | -- | 中等 |
| **C4** | **24** | **3** | **128** | **192** | **570M** | -- | **最优候选** |
| C5 | 32 | 2 | 128 | 192 | ~600M | -- | 对比候选 |

**关键发现**: c_t/meta 维度提升比单纯加深更有效。

### 1.3 训练基础设施

**优化器栈** (已落地):
- Muon (矩阵参数) + AdamW (标量参数)
- MuonClip (per-param RMS 比例裁剪)
- Modular-Norm-style 学习率缩放
- 8-bit Muon + 8-bit AdamW (bitsandbytes)
- CPU offload (optimizer states -> pinned CPU memory)
- LumaCosineScheduler

**FP8 混合精度** (已落地):
- Forward: FP8 E4M3 tensor core GEMM (212/185 Linear 层)
- Backward: 从 FP8 反量化
- 强制 BF16: Mamba SSM 核心, RMSNorm, JEPA 目标路径
- 节省: 24GB -> 13GB (312M), 使 S2 级可行

**VRAM 优化** (已落地):
- P0: Optimizer CPU offload -- 节省 667MB (8-bit) / 1.3GB (fp32)
- P1: Reason loop checkpoint -- 已确认 Mamba3 Triton kernel 不兼容额外 checkpointing, 当前已最优
- P2: Activation offload (compress zone) -- 节省 18GB, 但速度代价 2.3x, **暂不启用**
- **P3: FP8 Activation Compression (GPU)** -- 节省 3,706MB (18.4%), 速度代价 14%, `--fp8_activation_compress 1`
  - BF16 saved tensors -> FP8 E4M3 per-channel (Quamba 方法)
  - FP32 saved tensors -> BF16 downcast
  - 基于 saved_tensors_hooks, model-wide 覆盖
- Gradient checkpointing: 全程启用

**Checkpoint 断点续训** (已落地):
- `--save_interval N` 每 N 步保存 (model + optimizer + scheduler + step)
- `--resume path/to/checkpoint.pt` 从断点恢复
- `--ckpt_keep 3` 自动清理旧 checkpoint
- 每个 checkpoint ~1.3GB (660M 模型)

**动力学分析 v2** (已落地):
- LayerGradTracker: 逐层梯度 norm POD (dim 约 30-40, 真实判别力)
- CtStateTracker: batch variance 追踪 (不再只看 mean)
- ExitDepthTracker: 退出深度分布 (entropy, histogram, peaks)
- 旧 v1 3D POD (max rank=3) 保留兼容, v2 指标输出 `v2_rank=X/Y v2_mode1=Z%`

### 1.4 数据集 -- DataMix v2

| 数据源 | 条数 | 用途 |
|--------|------|------|
| persona_private | 43,053 | 人格种子 + 中文对话 |
| math_real | 14,227 | GSM8K + hendrycks_math |
| python_code | 26,802 | python_code_18k + CodeAlpaca |
| chinese_scifi | 4,636 | 刘慈欣 63 本 |
| arc_agi | 1,702 | ARC-AGI 400 tasks |
| **总计** | **90,420** | |

已清理: emotion_real (ESConv 96%超长), chinese_dialog (Belle 低质量), 旧合成模板数据

---

## 2. 当前状态 (2026-04-05)

### 2.1 Stage A 完成: C5 胜出

| 实验 | 配置 | 步数 | 状态 | 最终 v2_rank | 最终 mode1 |
|------|------|------|------|-------------|-----------|
| **C4-long** | 24L, d3, 570M | 5000 | **完成** | 1/33 (坍塌) | 100% |
| **C5-long** | 32L, d2, 660M | 5000 | **完成** | 14/40 (健康) | 82.5% |

**C5 胜出理由**: 跑满 5000 步未坍塌 (C4 在 3500 步后不可逆坍塌); 更深 compress zone 提供更多梯度多样性。前 2500 步两者 loss 几���相同。C5 每步慢 60% (1.3s vs 0.8s), 可接受。

### 2.2 Stage B 运行中: seq=1024 + World-JEPA

5 实验串行, ETA 约 14:30。

| ID | 配置 | 状态 |
|---|---|---|
| B0 | Phase 4 baseline (无 JEPA) | **运行中** |
| B1-B4 | World-JEPA (LeWM/EMA variants) | 排队中 |

**发现**: FP8 activation compress 与 seq=1024 不兼容 (10-20 步后 NaN), 已���用。VRAM ~23GB allocated (够用)。

### 2.3 Stage E 规划中: Exit Policy 改进 (4.1)

已实现二阶差分 exit (Phase 1), 实验矩阵已准备, 待 Stage B 完成后启动。详见 4.1。

### 2.4 代码修复 (Review 2026-04-05)

| 修复 | 位置 | 说明 |
|------|------|------|
| float16 clamp 下溢 | model_minimind.py:880 | `amax().float().clamp_()` 防止 scale=0 |
| cleanup_checkpoints 误删 | train_luma_refactor.py:107 | 按 phase 过滤 glob |
| Peak VRAM 累计值 | train_luma_refactor.py:464 | 训练前 `reset_peak_memory_stats()` |
| Phase 6 (World-JEPA) 配置 | train_luma_refactor.py:335 | 新增 `build_phase6_config()` + EMA update 调用 |
| CUDA 碎片化 | run_stage_b_matrix.sh | `expandable_segments:True` 环境变量 |

### 2.5 待决策

1. **Stage B 结果** → 选 LeWM vs EMA, 确定 SIGreg 强度和 mask ratio
2. **Exit policy** → Stage E 实验矩阵 (可与 Stage C 并行)
3. **数据扩量** → 正式预训练需大语料, 500K tokens 不够

---

## 3. 下一步执行路线

### Stage A: S2 配置选型 (完成 ✅)

**结论**: **C5 胜出** — compress=32, shared=2, 660M。

| ID | 配置 | v2_rank@2500 | mode1@2500 | 5000步坍塌? |
|---|---|---|---|---|
| A1 (C4) | 24L, shared=3, 570M | 7/33 (21%) | 39.2% (优) | **是** (rank→1) |
| A2 (C5) | 32L, shared=2, 660M | 6/40 (15%) | 96.1% (spike) | **否** (rank=14) |

C4 前 2500 步 mode1 更优但后期坍塌; C5 抗坍塌更强, 长期安全。

### Stage B: seq=1024 + World-JEPA 实验矩阵

**前置**: Stage A 选出 winner

全部使用: **bs=2, seq=1024, cosine LR decay, 2500 步, fp8_activation_compress=1, save_interval=500**

| ID | World-JEPA | Target 模式 | SIGreg | mask_ratio | 目的 |
|---|---|---|---|---|---|
| **B0** | OFF | -- | -- | -- | seq=1024 baseline, 确认长上下文不崩 |
| **B1** | ON | **LeWM** (stop-grad) | 0.05 | 0.25 | LeWM 基础配置 |
| **B2** | ON | **LeWM** (stop-grad) | 0.10 | 0.25 | 更强 SIGreg 防坍缩 |
| **B3** | ON | **EMA** (decay=0.996) | 0.05 | 0.25 | EMA 对照组 |
| **B4** | ON | **LeWM** (stop-grad) | 0.05 | 0.50 | 更高 mask ratio |

**预估**: 5 实验 x 2500 步 x ~2.5s/step = ~8.7 小时 (串行)

**VRAM 实测**: bs=2 seq=1024, allocated ~23GB, reserved ~31GB. FP8 activation compress 与 seq=1024 不兼容 (NaN), 已禁用。

**判据**:

| 指标 | 好 | 可接受 | 差 |
|---|---|---|---|
| v2 rank (2500 步) | >5 | 3-5 | <3 |
| mode1% | <60% | 60-80% | >80% |
| world-JEPA loss | 持续下降 | 平台 | 发散 |
| c_t batch 方差 | 正增长 | 稳定 | 萎缩 |

**LeWM vs EMA 选择逻辑**:
- LeWM (stop-grad): 省 ~1.3GB VRAM, 与现有 self-JEPA 架构一致, 需 SIGreg 防坍缩
- EMA: 更稳定靶标, 但多占 1.3GB, 实现更复杂
- **优先 LeWM**, 如果坍缩则 fallback 到 EMA

### Stage C: 数据扩量 + 配置冻结

**前置**: Stage B 选出最优 JEPA 配置

1. **数据验证**: D1-D4 矩阵 (persona+math, +python, +scifi, 全部), 确认 570M 消化能力
2. **序列长度渐进**: 1024 -> 2048 -> 4096 (Mamba3 线性复杂度, 瓶颈仅在 attention 层)
3. **配置冻结 (Gate F)**: 架构/数据/LR/seq_len 全部锁定

### Stage E: Exit Policy 改进 (可与 Stage B/C 并行)

**目标**: 解决 "10x20 ≈ 10x15" 瓶颈 — 模型无法有效利用更多 reasoning loops。

**Phase 1: 二阶差分 Exit (已实现)**

监测 `|delta_h_t - delta_h_{t-1}|`，当变化率本身不再变化时退出。比一阶 delta_h 更能识别真正收敛 vs 小值震荡。

- 实现: `ExitController` 新增 `second_order_weight` + `prev_delta_h` 输入
- CLI: `--exit_second_order_delta_weight <float>` (0=禁用, 0.3-0.5 推荐)

**实验矩阵** (seq=512, bs=4, 2500 步, Phase 4):

| ID | loops | 二阶权重 | 目的 |
|---|---|---|---|
| E0 | 12 | 0 | Baseline (同 C5 配置) |
| E1 | 12 | 0.3 | 二阶差分基础效果 |
| E2 | 12 | 0.5 | 更强二阶信号 |
| E3 | 20 | 0 | 更多 loops 无二阶 (对照) |
| E4 | 20 | 0.3 | **核心**: 二阶 + 更多 loops |
| E5 | 20 | 0.3 | E4 + sampling exit |

**成功标准**: E4 的 loss 或 v2_rank 显著优于 E0 和 E3。如果 E3 ≈ E0 但 E4 > E3, 证明二阶差分让模型真正利用了额外 loops。

**Phase 2: MoR Per-Token Routing (待实现)**

基于 arXiv:2507.10524, 每个 token 独立决定循环次数。需新增 per-token router 模块。
Phase 1 验证后实施, 预计 1-2 天。

### Phase Pretrain: 正式预训练

**预算估算**:
- 570M x Chinchilla 20x = 11.4B tokens
- 保守方案: 5.3B effective tokens
- 预估: ~0.9s/step, bs=2, seq=2048 -> ~4096 tokens/step -> 5.3B = 1.3M steps = 13.5 天
- 断点续训已就绪 (`--save_interval 1000 --resume`)

---

## 4. 研究方向候选 (2025-2026 论文调研)

以下基于最新论文调研, 按对当前瓶颈的影响力排序。

### 4.1 Exit Policy 改进 (P0 -- 最高优先)

**现状**: 10x20 约等于 10x15, 模型无法有效利用更多 reasoning loops。这是正式预训练前必须解决的核心瓶颈。

| 方向 | 论文 | 核心思想 | 实操建议 |
|------|------|---------|---------|
| **Mixture-of-Recursions (MoR)** | arXiv:2507.10524 NeurIPS 2025 | **per-token** router 决定每个 token 是否继续循环, 而非全局统一退出 | **最推荐**: 直接解决 Luma 的问题。代码开源 |
| **二阶差分 Exit** | arXiv:2509.23314 NeurIPS 2025 | 监测 delta_h 的变化率而非绝对值, 当 delta_h 不再减小时退出 | **最低成本**: 只需多存一步 delta_h, 几行代码 |
| **MIND (Introspection Switch)** | ICLR 2025 Oral | 轻量 introspection net 观察 hidden repr 决定走 FPI 还是 no-op | 与 Luma c_t 流天然契合 |
| **Inner Thinking Transformer** | arXiv:2502.13842 ACL 2025 | Thinking Step Encoding + Adaptive Token Routing | 162M 达 466M 的 96.5%, 减 43% 数据 |
| **FR-Ponder (Steering + Halting)** | arXiv:2509.24238 | <1M controller 观察 hidden states 决定停/继续, 加 steering vector 引导方向 | 与 c_t 认知流结合 |
| **Latent Reasoning Scaling** | arXiv:2502.05171 NeurIPS 2025 | Recurrent block latent iteration 实现 test-time scaling, 3.5B+800B tokens 约等于 50B 性能 | 验证了 Luma Reason Loop 路线正确 |

**推荐实施路线**:
1. Phase 1 (快速): 二阶差分 exit -- 监测 `|delta_h_t - delta_h_{t-1}|`, 收敛时退出
2. Phase 2 (核心): MoR per-token adaptive depth -- 每个 token 独立决定循环次数
3. Phase 3 (增强): 让 exit controller 读 c_t 多维信号 (norm, 方向变化, 对齐度)

### 4.2 序列长度扩展 (P1)

| 技术 | 论文 | 要点 | 优先级 |
|------|------|------|--------|
| **CoPE (Clipped RoPE)** | arXiv:2602.05258 (2026) | 对 RoPE 低频分量 soft clipping, 零成本获取 length generalization | 立即实施 |
| **LongRoPE2** | ICML 2025 | Evolutionary search 最优 RoPE rescaling + mixed window training | 128K 保持 98.5%+ 精度 |
| **Progressive Length Curriculum** | ACL 2025 | 逐步增长 context length, 同 FLOPs 下 1.5x 更快收敛 | 训练策略标配 |
| **Mamba-3 ICLR 2026** | arXiv:2603.15569 | Trapezoidal discretization + MIMO + data-dependent RoPE | 长期升级路径 |

**推荐**: CoPE 零成本可立即加入 -> progressive curriculum (512->1024->2048->4096) -> 如需 >8K 用 LongRoPE2。Mamba3 主干天然线性复杂度, 瓶颈仅在 SWA/KDA attention 层。

### 4.3 Progressive Scaling 0.6B -> 1.2B (P2)

| 技术 | 论文 | 要点 |
|------|------|------|
| **G_stack (Depthwise Stacking)** | arXiv:2405.15319 ICLR 2025 | 复制层堆叠扩容, 7B 仅需 194B tokens 达到 300B 效果, 54.6% 加速 |
| **Distilled Pretraining** | arXiv:2509.01649 | 蒸馏 init 的模型展现更好 test-time scaling (与 Reason Loop 直接相关) |
| **Minitron (Pruning+Distill)** | arXiv:2407.14679 NVIDIA | 反向: 先训大后裁剪, 产出高效推理版本 |

**推荐**: Compress Zone 用 G_stack (24->48 层), Reason Zone 用 width growth (weight-shared 不适合 stacking)。预期节省约 50% token 预算。

### 4.4 训练效率 (P1)

| 技术 | 论文 | 要点 |
|------|------|------|
| **Turbo-Muon** | arXiv:2512.04632 | Spectral preconditioning 加速 Newton-Schulz 2.8x, 整体减 5-10% 训练时间 |
| **Curriculum Learning** | arXiv:2505.11643 | 四阶段 easy->hard curriculum, 50% 训练步数, 激活更多 reasoning heads |
| **Data Mixing Laws** | ICLR 2025 | 性能关于 data mix 比例可预测, 小模型拟合 -> 预测大模型最优 mix |
| **8-bit Muon** | arXiv:2509.23106 | Blockwise 8-bit 量化, 74% memory reduction (已在用) |

**推荐**: Turbo-Muon 可直接替换当前 Newton-Schulz; Data Mixing Laws 在 S2 数据验证时使用; Curriculum 在正式预训练时实施。

### 4.5 World Model / JEPA (P2)

| 技术 | 论文 | 要点 |
|------|------|------|
| **LLM-JEPA** | arXiv:2509.14252 (LeCun 组) | 首个 LLM + JEPA, 显著超越标准 LLM 目标, embedding-space prediction + different views |
| **VL-JEPA** | arXiv:2512.10942 | 50% 更少可训参数, selective decoding 减少 2.85x 操作 |
| **JEPA + World Models 综述** | SSRN 2025-2026 | JEPA 和 World Models 正在融合 -- 验证 Luma 路线正确 |

**推荐**: 参考 LLM-JEPA 将 World JEPA 从 input-space 迁移到 embedding-space prediction; 引入 random loss dropout 缓解 2x compute 开销。

---

## 5. 风险与约束

### 5.1 已知风险

| 风险 | 严重性 | 缓解措施 | 状态 |
|------|--------|---------|------|
| seq=512 -> 2048 数值不稳 | 高 | Stage B/C 渐进验证 | 待验 |
| Mamba3 Triton kernel chunk_size 限制 | 中 | 已知 K>=16 约束, seq>=128 安全 | 已知 |
| 正式预训练 27 天太慢 (seq=512) | 高 | 用 seq>=2048, bs=2 | 规划中 |
| exit policy 瓶颈 | 中 | MoR/二阶差分 exit 研究 | 规划中 |
| VRAM 29GB/32GB 余量小 | **已缓解** | FP8 act compress 省 3.7GB (seq=512); seq=1024 不兼容, 回退到 23GB | Done |
| FP8 act compress + seq>=1024 NaN | 高 | **已禁用**: BF16→FP8 per-channel 在 ~15 步后 NaN | 已发现, 已绕过 |
| 小数据 5000 步过拟合 | 中 | 2500 步 + cosine decay + 大语料 | 已发现 |
| exit policy 瓶颈 (10x20≈10x15) | 高 | 二阶差分 exit 已实现, 实验矩阵待跑 | Stage E |

### 5.2 硬件约束

- **硬件**: RTX 5090 单卡 32GB, PCIe 5.0
- **CPU RAM**: 需确认 >=32GB (activation offload 需要)
- **训练时间**: 目标 <=14 天完成正式预训练
- **一次性 run**: Gate F 通过后不允许策略级重跑

---

## 6. 时间线估算 (更新于 2026-04-05)

| 阶段 | 预计耗时 | 依赖 | 状态 |
|------|---------|------|------|
| Stage A: C4/C5 选型 | ~80min | -- | **完成 ✅** C5 胜出 |
| Stage B: seq=1024 + JEPA 矩阵 | ~8.7 小时 (5 实验) | Stage A | **运行中** ETA ~14:30 |
| Stage E: Exit policy (二阶差分) | ~3.5 小时 (6 实验) | -- | **待启动** (B 完成后) |
| Stage C: 数据扩量 + seq 渐进 | 2-3 天 | Stage B + E | 规划中 |
| 配置冻结 (Gate F) | 0.5 天 | 以上全部 | 规划中 |
| 正式预训练 | ~13.5 天 | Gate F | 规划中 |
| **总计** | **约 17-20 天** | |

---

## 7. 文件索引

### 计划文档
- 本文档: `Luma_Execution_Plan_20260404.md`
- 历史 MasterPlan: `Luma_v0.7.2_Agent_MasterPlan.md` (保留参考)
- 旧执行计划: `Luma_Execution_Plan_20260402.md` (已取代)

### 报告
- Phase 4 实验矩阵: `Experiment_Matrix_Phase4_Full_Report_20260404.md`
- G5 + H 组报告: `H_Group_and_G5_Extended_Report_20260404.md`
- 模型扩容评估: `Model_Scaling_Evaluation_20260404.md`
- VRAM 优化: `VRAM_Optimization_Plan_20260404.md`
- 动力学综合报告: `Luma_Dynamics_Consolidated_Report_20260402.md`

### 关键代码
- 训练器: `minimind/trainer/train_luma_refactor.py`
- 模型: `minimind/model/model_minimind.py`
- 优化器: `minimind/luma_stage0/optimizers.py`
- 动力学分析: `minimind/luma_stage0/dynamics_analysis.py`
- S2 实验日志: `minimind/artifacts/experiment_matrix_20260403/`
