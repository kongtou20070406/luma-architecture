# VRAM 优化计划 (中等成本)

**日期**: 2026-04-04  
**硬件**: RTX 5090 (32GB VRAM)  
**当前状态**: S2 (660M) + FP8 forward + gradient_checkpointing → 30.2GB VRAM @ bs=4

已完成的低成本优化:
- [x] FP8 forward (tensor core GEMM): 24GB → ~13GB (312M), 使 S2 可行
- [x] Gradient checkpointing: 已启用
- [x] FP8 activation saving: backward 从 FP8 反量化, 省 50% activation 保存内存

---

## 目标

从 30.2GB 降到 ~24-26GB, 为以下场景留空间:
1. S2 (660M) bs=4 稳定运行 (当前 30.2GB, 余量仅 1.8GB)
2. 未来 S3 (943M) bs=2 可行 (预估 28GB)
3. 更长序列 (512→1024) 的可能性

---

## 优化 1: Optimizer State CPU Offload

### 原理
Adam optimizer 每个参数需要 2 个状态 (m, v), 各占参数大小的内存。  
660M params × bf16 = 1.3GB params → Adam states 占 ~2.6GB (m + v, 各 bf16)。  
把 m/v 放到 CPU RAM, 只在 step() 时搬回 GPU 做更新, 更新完搬回 CPU。

### 预期节省
**~2.6GB VRAM** (Adam states for 660M params)

### 实现方案
```python
# 方案 A: 手动 CPU offload (简单可控)
class CPUOffloadAdam(torch.optim.Optimizer):
    """Adam with m/v states on CPU, pin_memory for fast transfer."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
        # state tensors 分配在 CPU pinned memory
        # step() 时: GPU grad → CPU, 在 CPU 做 Adam update, 更新后的 param → GPU
        pass

# 方案 B: DeepSpeed ZeRO-Offload (更成熟但依赖重)
# from deepspeed.ops.adam import DeepSpeedCPUAdam

# 方案 C: PyTorch 原生 (2.3+)
# torch.optim.Adam with foreach=False + manual pin_memory
```

### 推荐: 方案 A (手动)
- 不引入 DeepSpeed 依赖
- 可以精确控制哪些参数 offload (大 Linear 层), 哪些留 GPU (小 norm 层)
- pin_memory + non_blocking transfer 可以和 forward 重叠

### 代价
- 每步训练增加 ~2-5ms (PCIe 传输 2.6GB, RTX 5090 PCIe 5.0 = 64GB/s → ~40ms 理论值)
- 实际因为可以分组流水线, 开销约 10-15ms/step (当前 ~900ms/step for S2, 增加 ~1.5%)

### 风险
- 低: CPU→GPU 传输是确定性的, 不影响数值精度
- 需要 pin_memory, 确保系统 RAM 充足 (>8GB free)

---

## 优化 2: Activation Offload (选择性)

### 原理
Gradient checkpointing 已经大幅减少了 activation 内存, 但 reason_loops 的循环特性导致:
- 每个 loop 的 checkpoint boundary 仍然保存了 hidden state (bs × seq × hidden_size)
- 15 loops × 4 × 512 × 1024 × 2 bytes = ~60MB (这部分不大)

更大的 activation 内存在 compress_zone 的 32 层:
- 即使有 checkpointing, 每个 checkpoint segment 仍保存输入 activation
- 32 layers / segment_size × bs × seq × hidden_size

### 预期节省
**~1-3GB**, 取决于 checkpoint segment 粒度

### 实现方案
```python
# 在 gradient_checkpointing 的基础上, 对 compress_zone 最底层的 segment 做 CPU offload
# 策略: 只 offload 前 N 层的 activation (这些层最早计算, 最晚被 backward 使用)

class ActivationOffloadCheckpoint:
    """Wrap torch.utils.checkpoint to offload saved tensors to CPU."""
    def __init__(self, offload_first_n_layers=8):
        self.offload_layers = offload_first_n_layers
    
    def pack(self, tensor):
        # 前 N 层: tensor.to('cpu', non_blocking=True)
        # 后面的层: 保留 GPU
        pass
    
    def unpack(self, tensor):
        # CPU tensor → GPU, non_blocking
        pass

# PyTorch 2.1+ 原生支持:
# torch.utils.checkpoint.checkpoint(..., preserve_rng_state=True)
# + saved_tensors_hooks(pack_hook, unpack_hook)
```

### 代价
- 每步增加 ~20-50ms (取决于 offload 层数)
- 前 8 层 offload: ~16MB × 8 = 128MB 搬运, PCIe 5.0 < 5ms
- 但 unpack 发生在 backward, 可能和计算产生竞争

### 风险
- 中等: saved_tensors_hooks 和 FP8 autograd.Function 的交互需要测试
- 如果 _FP8Matmul.save_for_backward 的 tensor 被 hook 拦截, 可能影响 FP8 反量化逻辑

---

## 优化 3: Reason Loop Activation Recomputation (零额外代码)

### 原理
当前 reason_loops 的每次循环可能在 checkpointing 外部保存了中间状态。  
确保整个 reason loop 被包裹在一个 checkpoint 里, 让 PyTorch 在 backward 时重算。

### 预期节省
**~0.5-1GB** (如果当前实现没有完全 checkpoint reason loops)

### 实现
检查 `train_luma_refactor.py` 中 reason zone 是否已经被 checkpoint 包裹。如果没有:
```python
# 在 forward 中
if self.training and self.use_gradient_checkpointing:
    reason_out = torch.utils.checkpoint.checkpoint(
        self.reason_zone, h, c_t, ..., use_reentrant=False
    )
```

### 代价
- 计算时间增加 (重算 15 loops), 但 reason zone 只占 15% 参数, 重算代价可控
- 前提: Mamba3 Triton kernel 必须兼容 recomputation (已知限制)

---

## 实施优先级

| 优先级 | 优化 | 预期节省 | 实际节省 | 状态 |
|--------|------|----------|----------|------|
| **P0** | Optimizer CPU Offload | ~2.6GB (fp32) | **667MB (8bit) / 1316MB (fp32)** | ✅ 已实施 |
| **P1** | Reason Loop Checkpoint 检查 | ~0.5-1GB | — | 待定 |
| **P2** | Activation Offload (前 N 层) | ~1-3GB | — | 待定 |

### P0 实施记录 (2026-04-04)
- 实现位置: `luma_stage0/optimizers.py` → `LumaOptimizerBundle.enable_cpu_offload()`
- 训练器接入: `train_luma_refactor.py` → `--cpu_offload_optimizer 1`
- 原理: step() 前将 CPU pinned memory 状态搬回 GPU，step() 后搬回 CPU
- 实测 633M 模型 (8-bit Muon + 8-bit AdamW): 优化器状态 652MB → offload 后节省 667MB
- 实测 633M 模型 (full-precision): 优化器状态 1299MB → offload 后节省 1316MB
- 注: 原预估 2.6GB 基于 fp32 Adam states，实际已用 8-bit 优化器压缩了大部分

### 修正后预期
当前配置 (8-bit optimizers + CPU offload): ~30.2GB → **~29.5GB** (节省 ~0.7GB)
如切回 fp32 optimizers + CPU offload: ~30.2GB → **~28.9GB** (节省 ~1.3GB)

P2 (Activation Offload) 对大模型更有效，预计额外节省 1-3GB。

---

## 优化 3 (新增): FP8 Activation Compression (On-GPU)

### 原理
利用 `saved_tensors_hooks` 拦截所有 `save_for_backward` 张量:
- BF16/FP16 张量 -> FP8 E4M3 per-channel quantization (50% savings)
- FP32 张量 -> BF16 downcast (50% savings)

基于 Quamba (arXiv:2410.13229) 和 COAT (arXiv:2410.19313) 的研究:
Mamba activations 有 channel-wise outliers, per-channel scaling 是质量保证的关键。

### 实测结果 (S2 570M, bs=4, seq=512)

| 指标 | Baseline | FP8 Compress | 变化 |
|------|----------|-------------|------|
| Peak VRAM (PyTorch) | 20,083 MB | 16,377 MB | **-3,706 MB (-18.4%)** |
| Step Time | ~1.36 s | ~1.55 s | **+14% slower** |
| Loss (step 30) | 13.6554 | 13.6081 | 无差异 |
| Gradient cosine sim | — | 0.9996 | 近乎无损 |

### 为什么只省 3.7GB 而非预估 9-10GB

FP8Linear 已经将 185 个 Linear 层的 saved activations 存为 FP8 (~8.7GB), hooks 只压缩剩余部分:
- Mamba3 kernel 的 Q/K/V/SSM_States/Out 等 (BF16 -> FP8)
- FP32 元数据张量 DA_CS/ADT/DT 等 (FP32 -> BF16)

### 实现位置
- 函数: `model_minimind.py` -> `_activation_fp8_compress_ctx()`
- 训练器: `train_luma_refactor.py` -> `--fp8_activation_compress 1`
- Model-wide 应用 (覆盖 compress zone + reason zone)

### 进一步压缩方向
- Checkpoint Mamba3 (当前因 Triton kernel 不兼容被跳过): 预计 +5-8GB
- reason_loops 12->8: 预计 +2GB
- INT4 activation (GACT 方案): 在 FP8 基础上再省 ~2GB, 质量风险较高

---

## 实施优先级 (更新)

| 优先级 | 优化 | 预期节省 | 实际节省 | 状态 |
|--------|------|----------|----------|------|
| **P0** | Optimizer CPU Offload | ~2.6GB (fp32) | **667MB (8bit) / 1316MB (fp32)** | Done |
| **P1** | Reason Loop Checkpoint 检查 | ~0.5-1GB | Mamba3 不兼容 | Done (无法实施) |
| **P2** | Activation Offload (CPU) | ~18GB | 18GB, **2.3x 速度损失** | Done (暂不启用) |
| **P3** | FP8 Activation Compress (GPU) | ~9-10GB | **3,706MB, 14% 速度** | Done |

---

## 时间线

1. ~~**S2 A/B 对比完成后**: 实施 P0 (Optimizer Offload)~~ Done
2. ~~**P3: FP8 Activation Compression**~~ Done (2026-04-04)
3. **如需进一步扩容**: 尝试 Checkpoint Mamba3 (验证 Triton kernel determinism)
