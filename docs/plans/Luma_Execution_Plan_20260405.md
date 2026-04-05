# Luma 执行计划 v3 (2026-04-05)

> 本文档取代 Luma_Execution_Plan_20260404.md。

---

## 0. 关键变化总结 (v2 → v3)

| 变化 | 影响 |
|------|------|
| **MIMO Mamba3 升级完成** (rank=2, chunk=32) | VRAM 从 23GB → 7.6GB (seq=2048), 15.3GB (seq=4096)。**彻底解除 VRAM 瓶颈** |
| **Pretrain 数据集 v1 重建** (735K 条, ~61M tokens) | DataMix v1 基本就绪，smart_code 桶从 the-stack 补充中 |
| C5 (660M) → C5-slim (314M) + MIMO | 参数量缩小但效率更高，VRAM headroom 巨大 |
| FP8 act compress + seq≥1024 NaN | 不再需要 — MIMO 本身已解决 VRAM 问题 |
| Stage B (seq=1024 JEPA 矩阵) | 计划内容不变，但 VRAM 余量更大 |

**核心结论**：MIMO 升级是 game-changer。之前在 32GB 卡上勉强跑 seq=1024，现在 seq=4096 + loops=12 只要 15.3GB，为后续 seq 扩展和更大 batch size 打开了巨大空间。

---

## 1. 已完成工作 (归档)

*(与 v2 §1 相同，此处省略。详见 Luma_Execution_Plan_20260404.md §1)*

### 1.5 MIMO Mamba3 升级 (2026-04-05, 新增)

**升级内容**：将 Mamba3 SSM 层从 SISO (单输入单输出) 升级到 MIMO (多输入多输出)。

**配置**：
```python
is_mimo = True
mimo_rank = 2        # 多输出分支数
chunk_size = 32      # rank * chunk = 64 (RTX 5090 TileLang 极限)
auto_fallback_on_mimo_error = True  # TileLang 编译失败时自动回退 SISO
```

**VRAM 验证结果**：

| seq | loops | Peak VRAM (allocated) | 状态 |
|-----|-------|-----------------------|------|
| 2048 | 2 | 7.59GB | ✅ |
| 2048 | 12 | 7.59GB | ✅ (gradient checkpointing 生效) |
| 4096 | 2 | 15.28GB | ✅ |
| 4096 | 12 | ~15.3GB | ✅ |

**关键发现**：
- loops 增加不增加 VRAM（gradient checkpointing 将每个 loop 的激活在 backward 时重计算）
- seq=4096 loops=12 仅 15.3GB，32GB 卡还剩 ~17GB headroom
- 自动 fallback 机制确保 TileLang 编译失败不阻塞训练

### 1.6 Pretrain 数据集 v1 (2026-04-05, ��增)

**pretrain_v1.jsonl**: 735K 条, 667MB, ~61M tokens

| 桶 | 目标占比 | 数据源 |
|---|---|---|
| smart_math (25%) | math_real + arc_agi |
| smart_code (15%) | python_code + stack_python (补充中) |
| smart_reason (10%) | *(空 — oasst/ultrafeedback 被排除)* |
| empathy (20%) | chinese_scifi |
| persona (15%) | persona_private + wechat_sft |
| dialogue (15%) | zhihu_kol |

**待补充**：
- smart_code: the-stack Python 子集拉取中 (~15K 条)
- smart_reason: 需要编程思维链 + tool use 数据（见 §3 Stage C）

---

## 2. 当前状态 (2026-04-05 晚, 更新)

### 2.1 已完成
- [x] Stage A: C5 胜出 (660M, 32L, shared=2)
- [x] MIMO Mamba3 升级 + VRAM 验证
- [x] Pretrain 数据集 v1 主体
- [x] 二阶差分 exit 实现
- [x] HF 认证配置
- [x] **SDPA Flash Attention 升级** — 所有手写 softmax attention 替换为 F.scaled_dot_product_attention
- [x] **VRAM 碎片化修复** — 每步后 `torch.cuda.empty_cache()`，reserved 从 29GB 降到 16GB
- [x] **过时代码清理** — 删除 fp8_activation_compress、旧 mask 函数、旧 Stage B 脚本

### 2.2 进行中
- [🔄] the-stack Python 数据拉取 (~15K 条，后台运行)
- [🔄] **Stage B' B0p_baseline 运行中** — seq=2048, bs=1, loops=12, ETA ~38min (step 500/2100)

### 2.3 已决策
1. ✅ **Stage B 直接升级到 seq=2048** — SDPA 升级 + empty_cache 修复解决了 OOM
2. **smart_reason 桶** — 待补充 CoT + tool use 数据
3. **正式预训练目标** — 5.3B tokens (标准方案)

### 2.4 今日经验教训

| 发现 | 原因 | 解决方案 | 教训 |
|------|------|---------|------|
| 手写 softmax attention OOM | [bs,16,2048,2048] FP32 矩阵太大 | SDPA flash kernel | **永远不要手写 attention** |
| 训练 5 步后 CUDA driver error | PyTorch CUDA 缓存碎片化 | `empty_cache()` 每步调用 | reserved ≠ allocated，碎片是杀手 |
| nohup 后台总是崩 | 前几次是 OOM 伪装成 driver error | 修复 OOM 后 nohup 正常 | 先在前台跑 smoke test 确认稳定 |
| MIMO 测试说 7.6GB | 测试用的是 228M 小模型 | 628M 模型实际 9.75GB | **VRAM 估算必须用真实模型** |
| fp8_activation_compress NaN | BF16→FP8 per-channel 精度不够 | 已删除，SDPA 解决了 VRAM | 不要保留已知有毒的代码路径 |

---

## 3. 推进路线 (更新)

### Stage B' (更新): seq=2048/4096 + World-JEPA

> 原 Stage B (seq=1024) 因 MIMO VRAM 突破而升级。

**配置变更**：
- seq: 1024 → **2048** (首选) 或 4096
- VRAM: 7.6GB (seq=2048) / 15.3GB (seq=4096)，远低于 32GB 上限
- 可考虑 bs=4 (seq=2048) 以加速实验

| ID | seq | World-JEPA | Target | SIGreg | mask_ratio | 目的 |
|---|---|---|---|---|---|---|
| **B0'** | 2048 | OFF | -- | -- | -- | seq=2048 baseline |
| **B1'** | 2048 | ON | LeWM | 0.05 | 0.25 | LeWM 基础 |
| **B2'** | 2048 | ON | LeWM | 0.10 | 0.25 | 更强 SIGreg |
| **B3'** | 2048 | ON | EMA (0.996) | 0.05 | 0.25 | EMA 对照 |
| **B5'** | **4096** | ON | (B1'/B2' winner) | -- | 0.25 | seq=4096 可行性 |

**预估**：5 实验 x 2500 步 x ~2.5s/step = ~8.7 小时
**判据**：同 v2 (v2_rank>5, mode1<60%, JEPA loss 下降, c_t 方差正增长)

### Stage E (不变): Exit Policy 二阶差分

**可与 Stage B' 并行。** 用 seq=512 bs=4 跑 6 组实验 (E0-E5)，验证二阶差分 exit 是否让模型真正利用更多 reasoning loops。

成功标准：E4 (20 loops + 二阶 0.3) 显著优于 E3 (20 loops 无二阶)。

### Stage C (更新): 数据扩量 + 配置验证

**核心数据原则：先变聪明，再变像 Luma。**

> 代码和数学语料能显著提升模型的推理和思考能力（"变聪明"），但不能让这些硬数据完全压倒 Luma 的情感理解和对话能力（"像 Luma"）。策略是：**pretrain 阶段 smart 桶占主导 (≥50%)，但始终保持 persona/empathy 桶作为底色；后续 SFT 阶段再强化 Luma 人格。**

**目标**：将 61M tokens 扩充到 ≥500M tokens，验证 660M 模型消化能力。

**数据扩量路线**：

| 优先级 | 数据源 | 预估条数 | 桶归属 | 目的 |
|--------|--------|---------|--------|------|
| P0 | the-stack Python (MIT/Apache) | ~15K | smart_code | 补齐代码推理 |
| P1 | OpenMathInstruct-2 / MetaMathQA | ~50K | smart_math | CoT 数学推理链 |
| P2 | SlimPajama (English, filtered) | ~200K | smart_reason | 通用英文推理 |
| P3 | Chinese web (WuDaoCorpora / BAAI) | ~100K | dialogue | 中文通用 + 对话质量 |
| P4 | 合成 tool use / agent traces | ~20K | smart_reason | Agent 能力种子 |
| P5 | 情感对话扩充 (empathetic_dialogues等) | ~30K | empathy | **防止情感能力被稀释** |

**配比原则** (pretrain 阶段)：
- smart 桶 (math + code + reason): **50-55%** — 推理是第一优先
- persona + empathy: **25-30%** — 人格和情感不能低于这个底线
- dialogue + 通用: **15-20%** — 中文表达质量
- **red line**: 任何扩量不得让 persona+empathy 低于总量的 25%

**验证矩阵** (Stage B' winner 配置)：
1. D1: persona + math (baseline，沿用 G5 经验)
2. D2: D1 + python + stack_python (代码推理)
3. D3: D2 + zhihu + scifi (中文质量 + 情感)
4. D4: D3 + SlimPajama + CoT (全量)
5. **D5: 情感回归测试** — D4 配置下跑 empathy 子集评估，确认情感理解未降级

**序列长度渐进** (MIMO 后不再有 VRAM 瓶颈)：
- Phase 1: seq=2048, bs=4 (7.6GB × 2 ≈ 15GB)
- Phase 2: seq=4096, bs=2 (15.3GB)
- Phase 3: seq=4096, bs=4 (~30GB, 接近上限)

### Stage D (新增): 高级 Exit Policy — MoR Per-Token Routing

**前置**：Stage E 验证二阶差分有效

基于 arXiv:2507.10524 (Mixture-of-Recursions)，实现 per-token router：
- 每个 token 独立决定是否继续循环
- 比全局统一退出更精细 — "简单 token 早退，难 token 多想"
- 代码开源，预计 1-2 天实现

**实施步骤**：
1. 在 `ExitController` 旁新增 `PerTokenRouter` 模块
2. Router 读取当前 hidden state，输出 per-token continue/exit 概率
3. 训练：Gumbel-softmax 可微采样
4. 推理：硬 threshold

### Gate F: 配置冻结

**前置**：Stage B' + C + D/E 全部完成

冻结项：
- 架构: MIMO rank/chunk, compress layers, reason shared depth, c_t/meta dim
- 数据: DataMix v1 final 配比 + 数据量
- 训练: LR, schedule, seq_len, bs, optimizer
- Exit: 二阶差分权重 / MoR router 配置

### Phase Pretrain: 正式预训练

**预算重新估算** (SDPA + empty_cache 实测)：

实测 VRAM (628M 模型, SDPA, empty_cache):
- seq=1024 bs=1: peak 5.62GB, reserved 16GB ✅
- seq=2048 bs=1: peak 9.75GB, reserved 16GB ✅
- seq=2048 bs=2: **OOM** (reserved > 32GB)
- seq=4096 bs=1: 未测试，预计 ~20GB

| 配置 | seq | bs | tokens/step | 目标 tokens | 步数 | 速度 | 时间 |
|------|-----|----|-------------|-------------|------|------|------|
| 保守 | 2048 | 1 | 2048 | 2B | 976K | 1.65s | ~18.6 天 |
| **标准** | **2048** | **1** | **2048** | **5.3B** | **2.6M** | **1.65s** | **~49 天** |
| 高效 | 1024 | 2 | 2048 | 5.3B | 2.6M | 1.3s | ~39 天 |

**现实问题**：bs=1 seq=2048 是当前 32GB 卡的极限。要真正提速需要：
1. 减小模型（比如回到 312M）→ 不可取
2. 用 seq=1024 bs=2（相同 tokens/step，更稳定）
3. 解决 VRAM 碎片化以支持更大 bs — 需要更深入的 activation offload 或 tensor parallelism

**结论**：正式预训练可能需要 seq=1024 bs=2 而非 seq=2048 bs=1，两者 tokens/step 相同但前者更稳定。

断点续训就绪：`--save_interval 1000 --resume`

---

## 4. 研究方向 (与 v2 相同，按优先级)

1. **Exit Policy** (P0): 二阶差分 → MoR per-token routing → c_t 多维信号
2. **序列扩展** (P1): CoPE (零成本) → progressive curriculum → LongRoPE2
3. **训练效率** (P1): Turbo-Muon → curriculum learning → data mixing laws
4. **Progressive Scaling** (P2): G_stack → distilled pretraining
5. **World Model** (P2): LLM-JEPA embedding-space prediction

---

## 5. 时间线估算 (2026-04-05 晚更新)

| 阶段 | 预计耗时 | 依赖 | 状态 |
|------|---------|------|------|
| ~~Stage A~~ | -- | -- | **完成 ✅** |
| ~~MIMO 升级~~ | -- | -- | **完成 ✅** |
| ~~SDPA 升级~~ | -- | -- | **完成 ✅** |
| ~~代码清理~~ | -- | -- | **完成 ✅** |
| Pretrain 数据 v1 | -- | -- | **基本完成** (the-stack 补充中) |
| **Stage B': seq=2048 + JEPA** | ~4.8h (5×58min) | ✅ | **B0' 运行中** (step ~500/2100) |
| **Stage E: Exit policy** | ~3.5h (6 实验) | -- | 可与 B' 并行 |
| **Stage C: 数据扩量** | 2-3 天 | B' 完成 | 规划中 |
| **Stage D: MoR routing** | 1-2 天 | E 完成 | 规划中 |
| Gate F: 配置冻结 | 0.5 天 | B'+C+D/E | 规划中 |
| **正式预训练** | **~39 天** (seq=1024 bs=2) | Gate F | 需重新评估 |
| **总计** | **约 45-50 天** | | |

**重要修正**：之前估算正式预训练 7 天是基于 bs=2 seq=4096 的假设，但实测 628M 模型在 32GB 卡上 bs=2 seq=2048 就 OOM。实际可行配置是 seq=1024 bs=2 或 seq=2048 bs=1，两者 ~2048 tokens/step，5.3B tokens 需要 ~39 天。

**加速方案**（待探索）：
1. 更激进的 activation offload（牺牲 ~20% 速度换 bs=2 seq=2048）
2. 缩小模型到 ~400M（牺牲容量换速度）
3. 多卡训练（需要硬件）
4. 减少目标 tokens（2B 而非 5.3B，~15 天）

---

## 6. 风险与约束 (更新)

| 风险 | 严重性 | 缓解措施 | 状态 |
|------|--------|---------|------|
| MIMO TileLang 编译失败 | 低 | `auto_fallback_on_mimo_error` 自动回退 SISO | 已缓解 |
| MIMO rank*chunk=64 上限 | 中 | rank=2 chunk=32 已是极限，rank=4 需 chunk≤16 | 已知 |
| seq=4096 数值稳定性 | 中 | Stage B' 验证 + BF16 关键路径 | 待验 |
| 数据量不足 (61M vs Chinchilla 13.2B) | **高** | Stage C 数据扩量是关键路径 | 规划中 |
| exit policy 瓶颈 (10x20≈10x15) | 高 | Stage E + Stage D (MoR) | 规划中 |
| smart_reason 桶为空 | 中 | CoT + tool use 合成数据 | 规划中 |

**硬件约束**：
- RTX 5090 32GB → MIMO 后 VRAM headroom 充足
- 正式预训练目标: ≤7 天 (之前 ≤14 天)

---

## 7. 立即行动清单 (Today)

1. ✅ HF 认证配置
2. 🔄 the-stack Python 数据拉取 (后台运行中)
3. **→ 启动 Stage B' (seq=2048)**: 先跑 B0' baseline 确认 MIMO + seq=2048 稳定
4. **→ 同时启动 Stage E**: exit policy 实验矩阵 (独立于 B')
5. **→ 完善 rebuild_mixes.py**: 加入 stack_python 数据后重建 pretrain_v1.jsonl
