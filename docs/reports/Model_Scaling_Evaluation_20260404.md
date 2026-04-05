# Luma 模型扩容评估

**日期**: 2026-04-04  
**硬件**: RTX 5090 (32GB VRAM, FP8 Tensor Cores)  
**当前**: 312M params (实际 294M), bf16, bs=4 占 24GB

---

## 1. 当前架构参数分布

| 组件 | 参数量 | 占比 |
|------|--------|------|
| compress_zone (24层) | 214.3M | 73.0% |
| reason_zone (loops=15) | 46.1M | 15.7% |
| embedding (factorized) | 29.5M | 10.0% |
| mamba + norms + other | 3.7M | 1.3% |
| **总计** | **293.6M** | 100% |

**关键特征**: 
- compress_zone 占绝对主导，扩容时 hidden_size 的影响是二次方的
- reason_loops=15 导致激活内存 ~40x param size（远高于标准 transformer 的 ~10x）
- 实际 VRAM: 24GB @ bf16, bs=4, seq=512, gradient_checkpointing=ON

---

## 2. 扩容方案对比

基于实际观测的 VRAM 校准（294M → 24GB = 40.7x param size）：

| 方案 | 参数量 | 主要变化 | FP8 bs=4 | FP8 bs=2 | FP8 bs=1 |
|------|--------|----------|----------|----------|----------|
| **Current** | 294M | — | 13GB ✅ | 9GB ✅ | 7GB ✅ |
| **S5** | 431M | h=960, 同结构 | 19GB ✅ | 13GB ✅ | 10GB ✅ |
| **S1** | 492M | h=1024, L=24 | 22GB ✅ | 15GB ✅ | 11GB ✅ |
| **S2** | 635M | h=1024, L=32, 扩 mamba | 29GB ✅ | 19GB ✅ | 14GB ✅ |
| **S3** | 943M | h=1280, L=32 | 42GB ❌ | 28GB ✅ | 21GB ✅ |
| **S4** | 1.05B | h=1536, L=24, loops=12 | 47GB ❌ | 31GB ⚠️ | 23GB ✅ |

---

## 3. 推荐方案: S2 (635M)

### 为什么选 S2

1. **VRAM 安全**: FP8 bs=4 刚好 29GB，留 3GB 余量给系统和峰值
2. **参数量翻倍**: 294M → 635M，表达能力显著提升
3. **deeper > wider**: 32 层比 24 层提供更深的特征层次，对 compress/reason zone 的梯度流更有利
4. **保持 bs=4**: 不需要降 batch size，保证训练稳定性

### S2 配置

```python
hidden_size = 1024         # 768 → 1024
intermediate_size = 4096   # 3072 → 4096
compression_layers = 32    # 24 → 32
num_attention_heads = 16   # 12 → 16
num_key_value_heads = 4    # 3 → 4
c_t_dim = 96               # 64 → 96
meta_dim = 128             # 96 → 128
mamba_d_state = 256        # 192 → 256
factorized_vocab_dim = 256 # 192 → 256
reason_loops = 15          # 不变
```

### 次选: S1 (492M) — 保守方案
如果 S2 的 VRAM 实际超出预期，S1 (h=1024, L=24) 只用 22GB，确保安全。

---

## 4. FP8 混合精度策略

### 需要 FP8 的组件（大头，节省 VRAM）
- compress_zone 所有线性层（gate_proj, up_proj, down_proj, q/k/v/o_proj）
- reason_zone 线性层
- embedding 的 factorized projection

### 必须保留 bf16/fp32 的关键位点
- **RMSNorm**: 小参数量但对数值稳定性至关重要，保留 fp32
- **Softmax + Cross Entropy**: 已在模型内显式 `.float()`
- **SIGreg 计算**: 涉及 SVD 等高精度操作
- **DOD/DMD 分析**: 梯度矩阵的 SVD/DMD 必须 fp32
- **Mamba SSM kernel**: d_state 维度的 scan 操作，保留 bf16（Triton kernel 限制）
- **Loss 聚合和 backward scaling**: loss_scaler 用 fp32

### 实现方案

**推荐: PyTorch 原生 FP8 (torchao)**
```python
# RTX 5090 支持 torch.float8_e4m3fn / torch.float8_e5m2
from torchao.float8 import convert_to_float8_training

# 只对线性层做 FP8 量化，norm/mamba 等保持 bf16
convert_to_float8_training(model, module_filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear))
```

**备选: MS-AMP / TransformerEngine**
- TransformerEngine 原生支持 FP8 GeMM + bf16 非线性
- 但需要适配 Luma 的非标准架构（Mamba、reason loops）

### FP8 Master Weights 策略
```
Forward:  Linear weights in FP8-E4M3 (1 byte)
Backward: Gradients in FP8-E5M2 (1 byte)  
Optim:    Master weights in bf16 (2 bytes), Adam states in bf16 (4 bytes)
```
每个参数: 1 (FP8 weight) + 1 (FP8 grad) + 2 (master) + 4 (optim) = 8 bytes  
对比 bf16: 2 + 2 + 2 + 4 = 10 bytes → **节省 20% 固定开销**  
加上**激活内存 FP8 节省 ~50%** → 总节省 ~45%

---

## 5. 扩容对动力学的预期影响

### 正面预期
- **mode1 波动减少**: 更多参数 → 梯度方向更多样 → DOD rank 可能达到 4-5
- **数据类型上限提升**: 312M 只能 2-3 类，635M 可能承受 4-5 类
- **训练窗口变宽**: 312M 的甜蜜点只有几百步，635M 应该能稳定几千步
- **ARC-AGI 可能变得可行**: 更大的参数空间容纳异构数据域

### 风险
- **训练速度下降**: 参数翻倍 + 层数增加，每步时间可能 2-3x
- **FP8 精度问题**: reason_loops 的梯度累积在 FP8 下可能不稳定
- **新的超参数搜索**: 所有实验矩阵需要重跑

---

## 6. 实施路线

### Phase 1: FP8 基础设施 (先在 312M 上验证)
1. 集成 torchao FP8 或 TransformerEngine
2. 312M + FP8 跑 G5 baseline，对比 bf16 结果确认精度无损
3. 确认 DOD/DMD 在 FP8 下的数值稳定性

### Phase 2: 模型扩容
4. 实现 S2 (635M) 配置
5. FP8 + S2 跑 G5 baseline，确认 VRAM 在 32GB 内
6. 重跑 H2 (persona+math+python) 验证数据类型上限

### Phase 3: 全面实验
7. 在 635M 上重建实验矩阵（可能只需 F+G+H 子集）
8. 评估是否需要进一步扩容到 S3 (943M, bs=2)
