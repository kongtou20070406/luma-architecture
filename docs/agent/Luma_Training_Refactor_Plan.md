# Luma 训练体系重构执行计划

> **给 Claude Code 的上下文说明**：本文档是 Luma architecture 项目的训练体系重构指令。
> 项目仓库：`https://github.com/kongtou20070406/luma-architecture`
> 主实现文件：`minimind/model/model_minimind.py`
> 硬件环境：WSL2 + RTX 5090 (32GB VRAM)，当前参数量 0.27B，fp32 训练。

---

## 0. 问题诊断摘要

通过 DOD (Dynamic Orthogonal Decomposition) 和 DMD (Dynamic Mode Decomposition) 分析，发现以下严重问题：

1. **DOD rank 仅为 1-2**：整个模型的参数更新方向塌缩到 1-2 个方向，远低于健康值 3-4。
2. **能量聚集在单一模态**：某一条梯度信号完全主导了训练动态。
3. **压缩区（encoder/embedding）长期未被优化**：推理区的梯度无法有效回传到压缩区。
4. **推理区过度主导**：所有优化都集中在推理区，self_check、JEPA 等辅助 loss 的梯度"污染"了主干。

**根因**：多 loss 之间缺乏梯度隔离，压缩区没有独立梯度驱动，shared block 的双次 forward 导致梯度叠加，系统进入 Edge of Stability 状态后对称性破缺不可逆。

---

## 1. Phase 0 — 代码清理与基础设施搭建

### 1.1 分支管理

```bash
git checkout -b luma-refactor
# 保持 main 分支不动，所有修改在 luma-refactor 上进行
```

### 1.2 关闭所有辅助 loss

在训练循环中，将以下 loss 的权重设为 0 或注释掉其 `.backward()` 调用：

- `loss_jepa`（JEPA 预测 loss）
- `loss_sc`（self_check loss）
- `loss_ct`（c_t 相关 loss，如果有）
- 其他任何非 next-token prediction 的辅助 loss

**只保留 `loss_lm`（主 next-token prediction loss）。**

> ⚠️ 注意：不要删除这些 loss 的代码，只关闭它们参与反向传播。结构（forward 计算图）保持不变。

### 1.3 切换为固定学习率

将 lr scheduler 从余弦退火改为固定 lr：

```python
# 移除或注释掉余弦退火 scheduler
# scheduler = CosineAnnealingLR(...)

# 使用固定 lr，取原 max_lr 的 30%-50%
# 例如原 max_lr = 3e-4，则使用 1e-4
FIXED_LR = 1e-4  # 根据你的实际 max_lr 调整

# 为 shared block 设置独立的更小 lr
param_groups = [
    {'params': compress_params,       'lr': FIXED_LR},
    {'params': shared_block_params,   'lr': FIXED_LR * 0.5},
    {'params': other_reasoning_params,'lr': FIXED_LR},
]
optimizer = torch.optim.AdamW(param_groups)
```

**原因**：固定 lr 确保 DOD 变化可以干净地归因到 loss 结构修改，而非 lr 变化。余弦退火会在训练前期导致不可逆的对称性破缺。

### 1.4 搭建评估监控体系

在训练循环中添加以下指标的日志记录（每 N 步记录，建议 N=50-100）：

```python
import torch

def compute_grad_metrics(model, named_module_groups):
    """
    计算各模块组的梯度范数。
    named_module_groups: dict, e.g. {
        'compress': compress_encoder.parameters(),
        'shared': shared_block.parameters(),
        'reasoning': reasoning_params,
    }
    """
    metrics = {}
    norms = []
    for name, params in named_module_groups.items():
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        metrics[f'grad_norm_{name}'] = total_norm
        norms.append(total_norm)

    if len(norms) > 0 and min(norms) > 0:
        metrics['grad_ratio'] = max(norms) / min(norms)
    else:
        metrics['grad_ratio'] = float('inf')

    return metrics

# 在训练循环中 loss.backward() 之后、optimizer.step() 之前调用：
# grad_metrics = compute_grad_metrics(model, module_groups)
# logger.log(grad_metrics)
```

**需要记录的完整指标集：**

| 指标 | 说明 | 频率 |
|------|------|------|
| `loss_lm` | 主 LM loss | 每步 |
| `loss_compress` | 压缩区辅助 loss（Phase 2 加入后） | 每步 |
| `loss_jepa` | JEPA loss（Phase 3 加入后） | 每步 |
| `loss_sc` | self_check loss（Phase 4 加入后） | 每步 |
| `grad_norm_compress` | 压缩区梯度范数 | 每 50 步 |
| `grad_norm_shared` | shared block 梯度范数 | 每 50 步 |
| `grad_norm_reasoning` | 推理区梯度范数 | 每 50 步 |
| `grad_ratio` | 最大/最小梯度范数比值 | 每 50 步 |
| `dod_rank` | DOD 分析的有效 rank | 每 500 步 |

**判断标准：`grad_ratio` 越接近 1 越好。如果超过 10，说明梯度严重失衡。**

---

## 2. Phase 1 — 基线验证（只有 LM loss）

### 目标

确认在只有一条梯度通路（LM loss）时，梯度能从 LM head 一路回传到压缩区 embedding 层。

### 操作

1. 使用 Phase 0 的配置（只有 `loss_lm`，固定 lr）
2. 训练 1000-2000 步
3. 记录各层 `grad_norm` 和 DOD 基线

### 检查点

- [ ] `grad_norm_compress` > 0 且不为极小值（如 < 1e-7），确认梯度能传到压缩区
- [ ] DOD rank 记录为基线值（此时 rank=1 是正常的，因为只有一条通路）
- [ ] 如果 `grad_norm_compress` ≈ 0，说明**结构本身就有梯度阻断**，需要先执行预案 S1（跳连捷径），再继续

---

## 3. Phase 2 — 添加压缩区辅助 loss

### 目标

给压缩区一条独立的梯度驱动信号，使 DOD rank 从 1 上升到 2。

### 实现

```python
# 在模型中添加一个轻量辅助头（训练完可丢弃）
compress_probe = nn.Linear(d_model, vocab_size)

# forward 中：
h_compress = compress_encoder(x)       # 压缩区输出
h_final = reasoning_blocks(h_compress) # 推理区处理
logits = lm_head(h_final)              # 主 LM logits

# 压缩区辅助 logits（直接从压缩区输出预测）
logits_compress = compress_probe(h_compress)

# loss 计算
loss_lm = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
loss_compress = F.cross_entropy(logits_compress.view(-1, vocab_size), targets.view(-1))

# 合并，权重 0.1-0.3
total_loss = loss_lm + 0.2 * loss_compress
total_loss.backward()
```

### 检查点

- [ ] DOD rank 从 1 上升到 2
- [ ] 能量分布在两个模态间有分散（不是 95/5，而应接近 60/40 或 70/30）
- [ ] `grad_norm_compress` 明显上升
- [ ] `grad_ratio` 下降（接近 1）

**如果 rank 没升，调大 `loss_compress` 权重到 0.3-0.5。如果仍然不升，说明压缩区和推理区的梯度方向完全重合，需要执行预案 S1。**

---

## 4. Phase 3 — 添加 JEPA loss（带 stop-gradient）

### 目标

加回 JEPA loss，但通过 `.detach()` 确保它不污染主干梯度流。DOD rank 应保持 2 或微升。

### 实现

```python
# JEPA 分支：从主干取特征时切断梯度回传
jepa_input = h.detach()                          # 关键：主干不收 JEPA 梯度
pred_z = jepa_predictor(jepa_input)
loss_jepa = jepa_loss(pred_z, target_z.detach()) # target 侧也要 detach

total_loss = loss_lm + 0.2 * loss_compress + w_jepa * loss_jepa
```

### 检查点

- [ ] DOD rank ≥ 2（不应下降）
- [ ] `grad_norm_shared` 没有因为 JEPA 而突然增大
- [ ] 如果 rank 下降了 → JEPA 的 detach 边界没切干净，检查是否有遗漏的梯度路径

---

## 5. Phase 4 — 添加 self_check loss（带 stop-gradient）

### 实现

```python
# self_check 分支：同样切断对主干的梯度
sc_input = h.detach()
sc_output = self_check_head(sc_input)
loss_sc = self_check_loss(sc_output, target)

total_loss = loss_lm + 0.2 * loss_compress + w_jepa * loss_jepa + w_sc * loss_sc
```

### 检查点

- [ ] DOD rank ≥ 2
- [ ] 能量分布保持均匀
- [ ] 各 `grad_norm` 之间比值稳定

---

## 6. Phase 5 — 添加 c_t 慢环 loss / 梯度路径

### 目标

如果 c_t 有独立的 loss，加回来。如果没有，确认 c_t 的梯度路径是否构成独立的动力学方向。

### 实现要点

```python
# c_t 从 h 获取信息时，做部分 detach
c_t_input = 0.8 * h + 0.2 * h.detach()  # 只让 20% 梯度回传
# 或者使用 GradScale（见预案 C）

# c_t 影响 h 时，也要控制梯度
h = h + gate * c_t.detach()  # c_t → h 的信号不回传梯度到 c_t
```

### 检查点

- [ ] DOD rank 升到 3（理想目标）
- [ ] c_t 相关参数的 `grad_norm` 非零且稳定

---

## 7. 结构改动预案（按需执行）

> 以下预案按侵入性从低到高排列。只在对应 Phase 的检查点未通过时执行。

### 预案 S1：跳连捷径（Phase 1 检查点未通过时使用）

**问题**：梯度完全无法从 LM head 回传到压缩区。
**解法**：加一条绕过推理区的残差连接。

```python
h_compress = compress_encoder(x)
h_reason = reasoning_blocks(h_compress)

# 跳连：让 LM loss 梯度有一条短路径直达压缩区
alpha = 0.1  # 初始值设小，可用 nn.Parameter 做可学习
h_final = h_reason + alpha * h_compress
logits = lm_head(h_final)
```

### 预案 S2：解耦 shared block 的两次 forward

**问题**：depth-2 shared block 导致梯度叠加，Hessian 特征值偏高，加速 EOS。
**解法**：两次 forward 共享权重但使用独立的 LayerNorm。

```python
# 原来：
# h = shared_block(h)
# h = shared_block(h)

# 改为：给 shared_block 添加一个 norm 参数
class SharedBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = ...
        self.ffn = ...
        self.norm_pass1 = nn.LayerNorm(d_model)
        self.norm_pass2 = nn.LayerNorm(d_model)

    def forward(self, h, pass_id=1):
        norm = self.norm_pass1 if pass_id == 1 else self.norm_pass2
        h_normed = norm(h)
        # ... 后续计算
        return h

# 调用：
h = shared_block(h, pass_id=1)
h = shared_block(h, pass_id=2)
```

### 预案 S3：c_t 慢环走独立路径

**问题**：c_t 和 h 的梯度完全耦合，c_t 没有独立的动力学方向。
**解法**：双向梯度控制。

```python
# h → c_t 方向：部分 detach
c_t_input = GradScale.apply(h, 0.2)  # 只传 20% 梯度
c_t_new = gate * c_t_old + (1 - gate) * c_t_transform(c_t_input)

# c_t → h 方向：完全 detach
h = h + output_gate * c_t_new.detach()

# GradScale 实现
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None
```

### 预案 S4：多尺度梯度注入

**问题**：即使加了压缩区辅助 loss，rank 仍然不够。
**解法**：在网络不同深度加入辅助探针。

```python
# 浅层探针（压缩区之后）
loss_shallow = F.cross_entropy(shallow_probe(h_compress), targets)

# 中层探针（第一次 shared block 之后）
loss_mid = F.cross_entropy(mid_probe(h_after_pass1), targets)

# 深层（原始 LM loss）
loss_deep = F.cross_entropy(lm_head(h_final), targets)

# 所有探针训练完后可丢弃
total_loss = loss_deep + 0.15 * loss_mid + 0.1 * loss_shallow
```

---

## 8. 训练集与评估方法调整

### 8.1 数据检查清单

- [ ] 检查语料各来源的比例，确认没有某类数据严重偏多
- [ ] 检查序列长度分布，考虑按长度分桶或做 length-normalized loss
- [ ] 确认 tokenizer 覆盖率，生僻 token 不应占比过高

### 8.2 评估集设计

评估集需要覆盖每条梯度通路的效果：

```python
eval_results = {
    'eval_loss_lm': evaluate_lm(model, eval_data),
    'eval_loss_compress': evaluate_compress_probe(model, eval_data),
    # 以下在对应 Phase 启用后加入
    'eval_loss_jepa': evaluate_jepa(model, eval_data),
    'eval_loss_sc': evaluate_self_check(model, eval_data),
    # 动力学指标
    'eval_dod_rank': compute_dod_rank(model, eval_data),
    'eval_energy_distribution': compute_energy_distribution(model, eval_data),
}
```

### 8.3 长度归一化 loss（可选）

```python
# 如果序列长度差异大，用 per-token 归一化
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='none')
loss = loss.view(batch_size, seq_len)
# 按有效 token 数归一化（排除 padding）
loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
loss = loss.mean()
```

---

## 9. 梯度流优化方法论文参考

完成基础重构后，可考虑引入以下方法进一步优化多 loss 平衡：

| 方法 | 论文 | 核心思路 | 适用阶段 |
|------|------|----------|----------|
| **LDC-MTL** | arxiv 2502.08585 (2025) | 双层优化做 loss 差异控制，O(1) 开销 | Phase 5 之后 |
| **Jacobian Descent** | arxiv 2406.16232 (2024-2025) | 用 Jacobian 矩阵投影消除梯度冲突，PyTorch 库 `torchjd` | Phase 5 之后 |
| **GradNorm** | arxiv 1711.02257 (2018) | 经典梯度归一化，动态调 loss 权重 | 任何阶段 |

**安装：**
```bash
pip install torchjd              # Jacobian Descent
# LDC-MTL: github.com/OptMN-Lab/LDC-MTL
```

---

## 10. 完整执行流程图

```
Phase 0: 代码清理
  ├── 关闭所有辅助 loss
  ├── 切固定 lr + shared block 独立低 lr
  └── 搭建 eval_metrics 日志
         │
Phase 1: 基线验证（LM loss only）
  ├── 训练 1000-2000 步
  ├── 检查 grad_norm_compress > 0？
  │     ├── YES → 继续
  │     └── NO  → 执行预案 S1（跳连捷径），然后重跑
  └── 记录 DOD 基线
         │
Phase 2: 加压缩区辅助 loss
  ├── 加 compress_probe + loss_compress
  ├── 训练 1000-2000 步
  ├── DOD rank 升到 2？
  │     ├── YES → 继续
  │     └── NO  → 调大权重 / 执行预案 S1 或 S4
  └── 记录能量分布
         │
Phase 3: 加 JEPA loss（带 detach）
  ├── jepa_input = h.detach()
  ├── 训练 1000 步
  ├── DOD rank ≥ 2？
  │     ├── YES → 继续
  │     └── NO  → 检查 detach 边界
  └── 记录指标
         │
Phase 4: 加 self_check loss（带 detach）
  ├── sc_input = h.detach()
  ├── 训练 1000 步
  └── DOD rank ≥ 2，grad_ratio 稳定？
         │
Phase 5: 加 c_t 慢环
  ├── 双向梯度控制（预案 S3）
  ├── 训练 1000 步
  └── DOD rank = 3？能量均匀？
         │
Phase 6: 确认最终配置
  ├── 所有 loss 到位
  ├── DOD rank ≥ 3，能量分布均匀
  ├── grad_ratio < 5
  └── 冻结配置 → 准备正式训练
         │
Phase 7: 正式训练准备
  ├── 切回余弦退火（或 warmup-stable-decay）
  ├── 关键：所有 loss 在 step 0 就全部启用
  ├── warmup 阶段所有通路同时建立
  └── 开始一次性全量预训练
```

---

## 11. 关键原则提醒

1. **每次只改一个变量**：每个 Phase 只加一个 loss 或一个结构修改，DOD 变化才能归因。
2. **rank 只能升不能降**：每加一个机制，DOD rank 要么不变要么升。降了就说明新机制在抢别人的梯度方向。
3. **能量分布比 rank 数值更重要**：rank=3 但 90/8/2 的能量分布，实际上还是接近 rank=1。
4. **结构不动优先**：先通过 loss 设计和 detach 边界解决问题。只有检查点明确未通过时才执行结构预案。
5. **compress_probe 等辅助头训练完可以丢掉**：它们不影响最终推理时的模型结构。
6. **固定 lr 是实验室环境，余弦退火是最终配置**：调好了再切。
