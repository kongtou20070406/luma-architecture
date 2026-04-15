# Rho-1 Self-Reference Epoch 2 规划

**创建时间**: 2026-04-15
**目标**: Luma v19 epoch 1 跑完后，epoch 2 启用 Rho-1 Self-Reference 加速收敛 + 过滤噪声 token
**前置**: v19 epoch 1 跑完 (step 80444), ema ≤ 7.5
**预期收益**: epoch 2 ema ≤ 5-6，为能说话打基础

---

## 1. 理论动机

### 1.1 Rho-1 核心思想

来自 Microsoft 论文 **"Rho-1: Not All Tokens Are What You Need" (2024)**：

不是所有 token 都值得同等权重训练。每个 token 有三种类型：

| 类型 | 描述 | 训练策略 |
|------|------|---------|
| **Easy** | 所有模型都轻松预测 | 跳过（浪费算力） |
| **Noise** | 所有模型都预测不了（tokenizer 噪声/乱码/断链） | 跳过（有害）|
| **Valuable** | 当前模型不会，但好模型能 | 重点训 |

**关键公式**:

```
excess_loss(x_i) = loss_current(x_i) - loss_reference(x_i)
```

- `excess_loss` 高 → valuable（reference 会但 current 不会）
- `excess_loss` 低 → easy 或 noise
- 取 top-k% excess_loss 做 loss，其他丢弃

**论文报告**: 只用 30% token 训练达到 100% 的效果，加速 5-10x。

### 1.2 Luma 当前的 selective_loss (简化版)

trainer.py line 772-797 已实现一个简化版：

```python
_per_tok = cross_entropy(...)    # 当前模型 per-token loss
_topk = _per_tok.topk(k)          # 直接取 top-k% 最高 loss
loss_lm = _topk.mean()
```

**问题**: 只看"绝对 loss 高低"，分不清 valuable 和 noise。高 loss 包括：
- 噪声 token (tokenizer 错误) — 不该训
- 真正难的 token — 该训

**标准 Rho-1 用 excess_loss 才能区分**:
- Noise: current_loss 高 + reference_loss 也高 → excess ≈ 0 → 跳过 ✓
- Valuable: current_loss 高 + reference_loss 低 → excess 大 → 选中 ✓

### 1.3 为什么对 Luma 特别重要

v5 数据集 330M tokens 远低于 Chinchilla-optimal (4.3B for 215M model)。Rho-1 的"等效数据放大"正好补这个缺口：

- 如果只训 top-60% → 相当于 198M 有效 tokens
- 但跳过的 40% 都是低价值 → **剩下的 60% 学得比 100% 更快**
- 等效数据量可能从 330M → 提升到 ~800M-1B "有效 tokens"

同时 v5 混合了 smart + persona + empathy + 对话 + 毛选等，不同类型 token 的学习价值差异大：
- **对话/问候语**: 高频，easy，应跳过
- **数学 CoT / 代码**: 低频，难，应重点训
- **毛选里的古词**: 罕见，接近 noise，应跳过
- **persona 里的情绪词**: 中等，valuable，应训

**Rho-1 正好做这个区分**。

---

## 2. 方案对比

### 方案 A: Self-Reference Rho-1 ⭐ 推荐

**Reference**: v19 epoch 1 end 的冻结 checkpoint (phase6_step80444.pt)
**Current**: epoch 2 从 epoch 1 end 继续训练

```python
# Epoch 2 启动:
reference_model = load_checkpoint("phase6_step80444.pt")
reference_model.eval()
for p in reference_model.parameters():
    p.requires_grad = False

current_model = load_checkpoint("phase6_step80444.pt")  # 同起点
# 继续 trainable
```

**每步**:
```python
with torch.no_grad():
    ref_out = reference_model(input_ids, labels=labels)
    ref_per_tok = per_token_loss(ref_out.logits, labels)

cur_out = current_model(input_ids, labels=labels)  
cur_per_tok = per_token_loss(cur_out.logits, labels)

excess = cur_per_tok - ref_per_tok
# 第一步 excess ≈ 0（两个 model 完全相同），随着 current 学习，excess 开始分化
top_k_idx = excess.topk(int(N * 0.6)).indices
loss_lm = cur_per_tok[top_k_idx].mean()
```

**优点**:
- 同架构同 tokenizer，梯度信号 100% 兼容
- 不需要额外预训练
- 初始 excess=0 自然 warm-up
- 保留 Luma 的人格/persona 特征（reference 也是 Luma，不会被通用模型的偏见污染）

**缺点**:
- Epoch 2 头几百步 excess ≈ 0，top-k 几乎随机（但无害）
- 显存多 ~0.4GB (bf16 reference)

**显存估算**:
- Current (trainable + grad + optimizer): ~11 GB (v19 现状)
- Reference (eval bf16, no grad): ~0.4 GB weights + ~0.5 GB activations
- Total: ~12 GB (vs 32 GB 上限，富余)

### 方案 B: External Qwen Reference

**Reference**: Qwen2.5-0.5B (和 Luma 同 tokenizer)

**优点**:
- Qwen 见过几 T tokens，excess_loss 信号强
- 第一步就有实质信号

**缺点**:
- Qwen 是通用模型，对 Luma 的 persona/empathy 有偏见
- "简单的 persona token" 可能被 Qwen 判为 easy → 被跳过 → 损害 Luma 独特性
- 需要额外下载和加载 Qwen checkpoint
- 参数量可能比 Luma 大，显存紧

**不推荐** ❌（Luma 走人格路线，不该用通用模型当老师）

### 方案 C: Noise-Filter-Only 简化方案

保留当前 selective_loss_ratio，只加 **上限过滤**:

```python
# 剔除 loss 极高的 token (噪声 token)
MAX_REASONABLE_LOSS = 11.0  # log(vocab)=11.9, 留 0.9 buffer
valid_mask = _per_tok < MAX_REASONABLE_LOSS
# top-k of valid
```

**优点**:
- 不需要 reference model
- 实现最简单 (+5 行代码)
- 零显存开销

**缺点**:
- 不能区分 easy 和 valuable（都是"loss 不高"）
- 只是过滤噪声不是真正的 Rho-1
- 阈值 MAX_REASONABLE_LOSS hack 没原理支撑

**作为 Rho-1 的 fallback**，如果方案 A 出问题可以快速切换。

---

## 3. 实现计划（方案 A）

### 3.1 代码改动清单

| 文件 | 改动 | 行数估计 |
|------|------|---------|
| `trainer/train_luma_refactor.py` | 加载 reference model，每步算 excess | +60 |
| `trainer/train_luma_refactor.py` | CLI: `--rho1_ref_ckpt`, `--rho1_ratio` | +10 |
| `docs/plans/Rho1_Epoch2_Plan.md` | 本文档 | (完成) |
| 可选：`luma_stage0/rho1.py` | 独立模块，封装 reference 加载和 excess 计算 | +80 |

### 3.2 详细实现步骤

#### Step 1: CLI 参数

```python
parser.add_argument("--rho1_ref_ckpt", type=str, default="",
                    help="Rho-1: reference model checkpoint path (空=禁用 Rho-1)")
parser.add_argument("--rho1_ratio", type=float, default=0.6,
                    help="Rho-1: top-k ratio of excess_loss tokens to train (默认 60%%)")
parser.add_argument("--rho1_warmup_steps", type=int, default=500,
                    help="Rho-1: 前 N 步用完整 loss，然后切换到 excess-based selection")
```

#### Step 2: Reference 加载

```python
rho1_reference = None
if args.rho1_ref_ckpt:
    Logger(f"Rho-1: loading reference from {args.rho1_ref_ckpt}")
    rho1_reference = LumaForCausalLM(luma_config).to(device)
    _ref_ckpt = torch.load(args.rho1_ref_ckpt, map_location=device, weights_only=False)
    rho1_reference.load_state_dict(_ref_ckpt["model"])
    rho1_reference.eval()
    for p in rho1_reference.parameters():
        p.requires_grad = False
    # bf16 进一步节省显存
    rho1_reference = rho1_reference.to(dtype=torch.bfloat16)
    Logger(f"Rho-1: reference loaded ({sum(p.numel() for p in rho1_reference.parameters())/1e6:.1f}M params)")
```

#### Step 3: 每步 Rho-1 loss 计算（替换现有 selective_loss 分支）

```python
# 在 current model forward 后
if rho1_reference is not None and step >= args.rho1_warmup_steps:
    with torch.no_grad():
        ref_out = rho1_reference(input_ids, labels=labels)
        ref_logits = ref_out.logits
        if ref_logits.size(-2) > labels.size(-1):
            ref_logits = ref_logits[:, -labels.size(-1):, :]
        _ref_logits = ref_logits[..., :-1, :].contiguous()
        _labels = labels[..., 1:].contiguous()
        ref_per_tok = F.cross_entropy(
            _ref_logits.view(-1, _ref_logits.size(-1)),
            _labels.view(-1),
            ignore_index=-100, reduction="none",
        )

    # Current per-token loss
    cur_logits = res.logits
    if cur_logits.size(-2) > labels.size(-1):
        cur_logits = cur_logits[:, -labels.size(-1):, :]
    _cur_logits = cur_logits[..., :-1, :].contiguous()
    cur_per_tok = F.cross_entropy(
        _cur_logits.view(-1, _cur_logits.size(-1)),
        _labels.view(-1),
        ignore_index=-100, reduction="none",
    )

    # Excess loss
    excess = (cur_per_tok - ref_per_tok).detach()
    valid = (_labels.view(-1) != -100)
    excess_valid = excess[valid]
    cur_valid = cur_per_tok[valid]
    k = max(1, int(excess_valid.numel() * args.rho1_ratio))
    top_excess_idx = excess_valid.topk(k).indices
    loss_lm = cur_valid[top_excess_idx].mean()
    _total_lm = loss_lm + aux_loss
    # 诊断: 平均 excess, ratio of noise (excess<0), ratio of easy (excess<epsilon)
    _dbg_excess_mean = float(excess_valid.mean().item())
    _dbg_noise_pct = float((excess_valid < 0).float().mean().item() * 100)
    _dbg_easy_pct = float((excess_valid.abs() < 0.01).float().mean().item() * 100)
else:
    # 原有分支（完整 loss 或简化 selective）
    ...
```

#### Step 4: 日志打印

```python
if rho1_reference is not None and step >= args.rho1_warmup_steps:
    rho1_line = f"  rho1: excess={_dbg_excess_mean:.3f} noise={_dbg_noise_pct:.1f}% easy={_dbg_easy_pct:.1f}%"
else:
    rho1_line = ""

Logger(f"[{step}/{args.iters}] loss_lm={loss_lm:.3f} ... {rho1_line}")
```

### 3.3 显存影响预估

v19 当前 peak VRAM ≈ 11 GB (bs=1 seq=2048 FP8+grad_ckpt+CPU offload)。

Reference model 开销：
- Weights: 215M × 2 bytes (bf16) = 430 MB
- Forward activation (no grad): ~500 MB (因为无 backward graph，可以复用 buffer)
- **Total: ~1 GB 额外**

Epoch 2 预计 peak VRAM: **~12 GB** (仍远低于 32 GB 上限)。

### 3.4 计算开销预估

每步多一次 reference forward（无 backward），相当于 **每步多 1 次 inference**。

- Current forward + backward ≈ 3-4x inference time
- Reference forward = 1x inference time
- **Rho-1 开销 ≈ 25-33% step time 增加**

Epoch 2 预计 ETA：从 v19 epoch 1 的 17 小时增加到 ~22-23 小时。

---

## 4. 启动命令模板

```bash
# Epoch 2 启动（Rho-1 self-reference）
cd /home/kt/ai/luma-architecture/minimind/scripts && \
bash run_experiment.sh hero_v20_rho1_epoch2 \
  --resume ../artifacts/checkpoints/phase6_step80444.pt \
  --rho1_ref_ckpt ../artifacts/checkpoints/phase6_step80444.pt \
  --rho1_ratio 0.6 \
  --rho1_warmup_steps 500 \
  --iters 160888  # epoch 2 从 step 80444 继续到 160888
```

---

## 5. 判据 / 成功标准

### 5.1 Epoch 2 必须满足

- **fp_proxy L < 1.0** 全程（仿星器稳定性不能退化）
- **sig_raw < 10** 全程
- **loss_w < 0.05** 全程
- **不 NaN**
- **通过 step 95000 / 110000** 等里程碑

### 5.2 Epoch 2 成功判定

从 epoch 1 end 的 **ema ≈ 7.0-7.5** 继续：

| epoch 2 结果 | ema (end) | 判定 |
|------------|-----------|------|
| **纯赢** | ≤ 5.5 | Rho-1 有效，继续 epoch 3 |
| **部分赢** | 5.5 - 6.5 | 有效但收益低于预期，评估是否 rho1_ratio 调整 |
| **无效** | 6.5 - 7.0 | Rho-1 没帮助，回退标准训练 |
| **退化** | > 7.0 | Rho-1 有害，停训 + 分析 |

### 5.3 诊断指标

除了 ema，观察：
- **`rho1_noise_pct`**（excess<0 的占比）: 预期 20-40%，表示 reference 和 current 有分化
- **`rho1_easy_pct`**（excess≈0 的占比）: 预期 30-50%，表示容易 token 被正确跳过
- **`rho1_excess_mean`**: 正值且上升 = current 在学新东西；接近 0 = stall

---

## 6. 风险点和缓解

### 风险 1: Reference 和 Current 太像

**场景**: epoch 2 开始时 reference 和 current 完全一样，excess 全为 0，top-k 几乎随机。
**缓解**: `--rho1_warmup_steps 500`，前 500 步仍用完整 loss，让 current 先偏离一点。

### 风险 2: Excess 分布病态

**场景**: 某些 batch 里所有 token excess 都很小，top-k 退化为随机。
**缓解**: 监控 `excess_mean` 和 `excess_std`，如果 std 太小说明没区分度，自动回退到完整 loss。

### 风险 3: Reference 过时

**场景**: Current 学了很久，reference 仍是 epoch 1 end 的状态，信号失效。
**缓解**: epoch 3 时把 epoch 2 end 当新的 reference。或者更激进的做法：每 5000 步更新 reference (EMA 式)。

### 风险 4: 显存溢出

**场景**: Reference bf16 + activations 比预期大。
**缓解**: 如果 VRAM 紧，用 `--fp8 1` 把 reference 也转 FP8，或者用 CPU offload reference。

### 风险 5: Rho-1 把 persona token 全丢了

**场景**: persona token 在 v5 里高频，reference 已经学得很好，excess 低 → 全被 skip。
**缓解**:
- 可以加一个 **protected token mask**: 某些 token_id 保证进入训练
- 或者提高 `rho1_ratio` 到 0.8 确保不丢太多

---

## 7. 可选增强

### 7.1 Weighted Rho-1

不是硬筛选 top-k，而是按 excess_loss 加权:

```python
weights = softmax(excess / temperature)
loss_lm = (weights * cur_per_tok).sum()
```

**优点**: 平滑的 importance weighting，不丢失信息。
**缺点**: 引入 temperature 超参，复杂度增加。

**判断**: 先跑硬筛选版，如果 ema 停滞再试 weighted。

### 7.2 Dynamic Ratio

`rho1_ratio` 随训练阶段变化：

- epoch 2 前 50%: `ratio=0.8`（温和，让 current 先展开）
- epoch 2 后 50%: `ratio=0.4`（激进，筛掉 easy 的）

### 7.3 Token-type specific ratio

对不同数据源 token 用不同 ratio：

- persona/对话 bucket: ratio=0.9（保留多数）
- 数学/代码 bucket: ratio=0.4（严格筛选高 excess）

---

## 8. 实现时机

### 8.1 **不要现在做**

v19 epoch 1 还在跑 (step 60550, 75% 进度)。提前启动 epoch 2 会污染 checkpoint。

### 8.2 触发条件

v19 epoch 1 满足以下**全部**条件才启动 Rho-1 epoch 2：

- ✅ 跑完 step 80444
- ✅ 最终 ema ≤ 7.5
- ✅ fp_proxy L 全程 <1
- ✅ sig_raw 全程 <10
- ✅ sanity check (方案 A-D) 通过

如果 epoch 1 失败，先诊断 epoch 1 再考虑 Rho-1。

### 8.3 实现顺序

1. **epoch 1 跑完** (ETA: 6 小时)
2. **sanity check** (1 小时)
3. **实现 Rho-1 代码** (2 小时)
4. **Rho-1 smoke test** (256 步，20 分钟)
5. **Rho-1 epoch 2 正式跑** (22 小时)

---

## 9. 备选：如果 Rho-1 没用怎么办

如果 epoch 2 ema 没改善（> 6.5），考虑以下方向（**不全开**，按优先级试一个）:

1. **纯标准训练继续 epoch 2-3**：先排除 Rho-1 干扰因素
2. **扩数据**：把 v5 升级到 v6 (加 FineWeb 子集或 RedPajama)
3. **模型规模**：215M 可能不够，考虑 400M 或 600M
4. **超参调整**：scalar_lr 从 6e-4 调到 3e-4 (更稳但慢)

---

## 10. 相关参考

- **Rho-1 论文**: "Not All Tokens Are What You Need" (Lin et al., 2024, arxiv 2404.07965)
- **Chinchilla scaling**: 215M 最优 ~4.3B tokens (Hoffmann et al. 2022)
- **SlimPajama / RedPajama**: 备选扩数据源
- **Luma v5 data mix**: [docs/plans/v5_data_mix.md](v5_data_mix.md) (如存在)
- **v19 stellarator**: [WORKLOG 4.15 01:30 条目](../../artifacts/WORKLOG.md)
