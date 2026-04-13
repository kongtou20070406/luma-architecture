# FP8 Mamba3 MIMO Triton Kernel 重写计划

**创建日期**: 2026-04-13
**优先级**: 🟡 长期（等 Phase E hero run 稳定后再启动）
**预估工期**: 10-14 天（单人 triton kernel 重写 + 二阶导 + 数值验证）
**背景**: 用户希望支持长上下文 + 节省显存，当前 `--fp8 1` 只对 Linear 层生效，Mamba3 SSM 算子仍走 bf16，是长 seq 时显存瓶颈。

---

## 1. 目标与约束

### 1.1 核心目标
1. **原生 FP8 Mamba3 MIMO 前向**：输入/状态/输出张量用 FP8 (e4m3 或 e5m2)，kernel 内部 FP16 accumulate，最终 FP8 写回
2. **FP8 backward** 支持一阶导（训练可用）
3. **二阶导兼容**（Phase E grad mode 的长期兼容目标，当前 damped mode 不需要但留接口）
4. **数值精度**：最终 loss 相比 bf16 基线误差 < 5%（参考 FP8 Linear 的实测漂移）
5. **活性内存节省**：相比 bf16 Mamba3，显存占用 降 ~40-50%（FP8 = 1 byte vs bf16 = 2 bytes）

### 1.2 硬约束
- **不破坏 bf16 回退路径**：`fp8=0` 时必须走原 triton kernel
- **和 Phase E damped 兼容**：不引入新的 reentrant/saved_tensors_hooks 冲突
- **和 grad_ckpt 兼容**：Mamba3Block 的 `use_gradient_checkpointing=True` 路径保持可用（当前用 `use_reentrant=True`）
- **MIMO rank 保持 2**：不重写 MIMO 多路径逻辑
- **chunk_size 灵活**：支持 32/64/128 不改 kernel

### 1.3 范围外
- 不重写 SISO kernel（不用于生产训练）
- 不搞 FP4/INT8 量化
- 不做 per-tensor scaling（只用 per-channel 或 per-block scaling）

---

## 2. 当前代码结构调研

### 2.1 涉及文件（已备份到 `backups/mamba3_original_20260413/`）

| 文件 | 行数 | 角色 |
|---|---|---|
| `third_party/mamba-official/mamba_ssm/ops/triton/mamba3/angle_dt.py` | 431 | dt 角度投影 |
| `third_party/mamba-official/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py` | 711 | SISO 前向 |
| `third_party/mamba-official/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py` | 1777 | SISO 反向（主工作量） |
| `third_party/mamba-official/mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py` | 403 | 前后向组合 autograd Function |
| `third_party/mamba-official/mamba_ssm/ops/triton/mamba3/mamba3_mimo_utils.py` | 878 | MIMO 实用函数 |
| `third_party/mamba-official/mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py` | 397 | MIMO rotary + step 推理 |
| `third_party/mamba-official/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py` | ? | MIMO 主入口 |
| `third_party/mamba-official/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py` | ? | MIMO tilelang 前向 |
| `third_party/mamba-official/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py` | ? | MIMO tilelang 反向 |
| `minimind/model/mamba3_module.py` | 166 | Luma 包装器 + `Mamba3Block.forward` |

### 2.2 当前数据流
```
x: [B, T, d_model] bf16
  ↓ pre_norm (ZCRMSNorm)
  ↓ linear_proj (fp8 via FP8Linear if fp8=1)
  ↓ reshape → dt_bias, A, B, C 计算 (bf16)
  ↓ mamba3_mimo triton kernel (ALL bf16)     ← FP8 化目标区
  ↓ output_proj (fp8)
  ↓ post_norm + dropout + residual
y: [B, T, d_model] bf16
```

### 2.3 当前显存大头（seq=2048, hidden=768, d_state=192, expand=2, headdim=64）
每层 Mamba3 forward 保存：
- x_proj: [B, T, 2*d_model] ~ 6 MB bf16
- dt, A, B, C: [B, T, nheads, ...] ~ 30 MB
- 内部 SSM state: [B, nheads, headdim, d_state] ~ 存一次
- chunked scan 临时张量: O(B × T × nheads × chunk × d_state) → 可观

总估算每层 Mamba3 activation ~80-100 MB bf16 seq=2048。8 层 Mamba = 800 MB。
FP8 化可降到 ~400 MB，长 seq 下节省显著。

---

## 3. 实施阶段（Phased，每 5 step 一次子 agent 审查）

### Phase 0: 环境 & 基准测试（1 天）

**步骤 0.1** (1 step): 搭建 FP8 triton 开发环境
- 验证 `triton` 版本 ≥ 2.3（FP8 原生 tl.tensor(dtype=tl.float8e4m3fn) 支持）
- 运行 `scripts/test_mamba3_mimo_fallback.py` 确认当前 MIMO 前后向 bf16 基线通过

**步骤 0.2** (1 step): 建立数值基准
- 写 `scripts/mamba3_fp8_bench.py`：固定输入 seed → 记录 bf16 output 的 mean/std/max/min
- 记录 backward 梯度的 L2 范数

**步骤 0.3** (1 step): 建立显存基准
- 测量 bf16 MIMO forward 占 activation 内存（torch.cuda.memory_allocated diff）
- 目标：FP8 版本 < 50% bf16 基线

**步骤 0.4** (1 step): 创建特性开关
- `mamba3_module.py` 加 `use_fp8_kernel` 配置
- Luma `LumaConfig` 加 `mamba_fp8_kernel: bool` flag
- fallback 默认 False

**步骤 0.5** (1 step): **子 agent 审查 #1**
- 任务：review Phase 0 changes 确认 baseline 建立，没破坏现有 fp8=0 路径
- 审查清单：基准值记录完整，回退路径完好

### Phase 1: FP8 SISO forward kernel（2-3 天）

**步骤 1.1**: Quantize helper + descale 表
- 加 `_quantize_fp8_per_block(x, block_size, dtype=fp8e4m3)` → 输出 (x_fp8, scale)
- per-block scale 粒度 = 64 或 128 tokens
- per-channel 备选

**步骤 1.2**: 改 `mamba3_siso_fwd.py` 接受 fp8 输入张量
- 加 `x_dtype_flag`：`bf16` / `fp8e4m3fn`
- kernel 内部：load FP8 → cast to fp16 → matmul → cast back to fp16 accumulator
- 输出仍是 fp16 / bf16（避免 output scale 复杂性）

**步骤 1.3**: 对 `x`, `dt`, `A`, `B`, `C` 逐个 tensor 加 FP8 路径
- 保留 `dt_bias` 为 bf16（小张量，不值得量化）
- `A` 是 learnable log-parameter，保留 fp32

**步骤 1.4**: 数值验证
- bf16 vs fp8 forward output max abs diff
- 阈值：< 5e-3

**步骤 1.5**: **子 agent 审查 #2**
- 审查清单：kernel diff 合理，fp8 路径不破坏 bf16，数值验证通过

### Phase 2: FP8 SISO backward kernel（3-4 天，核心工作量）

**步骤 2.1**: 改 `mamba3_siso_bwd.py`（1777 行）接受 FP8 cache
- forward 时保存的 state tensors 以 FP8 保存（需 scale 元信息）

**步骤 2.2**: 在 backward kernel 加 FP8 load → fp16 compute → grad 输出
- dA, dB, dC, dx 都返回 bf16（grads 留高精度，最终 optimizer step 时再量化）

**步骤 2.3**: 数值验证
- `torch.autograd.gradcheck` 不适用于 triton，用 random input + finite diff 验证
- FP8 grad vs bf16 grad max rel diff < 10%（FP8 backward 容忍更宽）

**步骤 2.4**: saved_tensors 单次 unpack 合规
- backward 里不要多次访问 `ctx.saved_tensors`（之前和 `use_reentrant=False` 冲突的那个 bug）
- 这次重写借机修复，兼容 non-reentrant checkpoint

**步骤 2.5**: **子 agent 审查 #3**
- 审查清单：backward 数值正确，saved_tensors 单次 unpack 合规

### Phase 3: MIMO tilelang 适配（2-3 天）

**步骤 3.1**: `tilelang_mamba3/mamba3_mimo_fwd.py` 接受 FP8
- MIMO = rank=2 有两条 SSM 路径并行，两路独立 FP8 输入

**步骤 3.2**: `tilelang_mamba3/mamba3_mimo_bwd.py` FP8 backward
- rank=2 意味着梯度是两条路径的和，两路都用 FP8

**步骤 3.3**: 数值验证（MIMO 尤其敏感）
- seq=1024 forward/backward FP8 vs bf16 对比

**步骤 3.4**: 集成到 `mamba3_module.py`
- `use_fp8_kernel=True` 时走 FP8 路径
- 失败自动 fallback 到 bf16（含警告日志）

**步骤 3.5**: **子 agent 审查 #4**
- 审查清单：MIMO FP8 正确，fallback 路径完好

### Phase 4: Luma 集成 & 训练验证（1-2 天）

**步骤 4.1**: Luma 端配置 flag 接入
- `LumaConfig.mamba_fp8_kernel`
- `train_luma_refactor.py --mamba_fp8_kernel 1`

**步骤 4.2**: Smoke test 最小模型
- hidden=384, 2 layers Mamba3 MIMO
- fp8_kernel=1 forward/backward 通过

**步骤 4.3**: 显存测量
- seq=2048 Luma 全栈：bf16 vs fp8 activation 内存对比
- 目标：Mamba 部分 ≥ 40% 节省

**步骤 4.4**: 数值稳定性（50 iter 短训）
- 同 config 对比 bf16 和 fp8 的 loss 曲线
- max rel diff < 10%

**步骤 4.5**: **子 agent 审查 #5**
- 审查清单：端到端训练稳定，显存节省达标，fallback 正确

### Phase 5: 长 seq 实验（1-2 天）

**步骤 5.1**: seq=4096 能否跑通（之前 bf16 下 ~52 GB 预估 OOM）

**步骤 5.2**: seq=6144 极限测试

**步骤 5.3**: 长 seq loss 曲线健康（无 NaN）

**步骤 5.4**: 和 bf16 基线比对

**步骤 5.5**: **子 agent 审查 #6**（最终）
- 审查清单：长 seq 工作，显存达标，精度达标，回退路径完好

---

## 4. 风险 & 缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| triton 2.x 对 tl.float8e4m3fn 支持不全 | 🟡 中 | 先验证，不行则用 `bitcast(int8)` 方案 |
| FP8 backward 数值误差太大 | 🟡 中 | backward 保留关键中间态为 fp16 |
| 破坏现有 bf16 训练 | 🔴 高影响 | feature flag + 默认关闭 |
| Phase E damped 长期稳定性受影响 | 🟡 中 | 先独立验证，再组合测试 |
| MIMO rank=2 两路同步 FP8 尺度偏差 | 🟡 中 | 两路用独立 scale factor |
| 二阶导（Phase E grad 模式）不再工作 | 🟡 中 | 只保障一阶导，二阶导做 soft fallback 到 bf16 |

---

## 5. 审查 Agent 交互规范

每个 Phase 末尾的 "子 agent 审查" 步骤：

```
Agent 调用:
  description: "FP8 Mamba3 Phase N review"
  subagent_type: code-reviewer
  prompt: |
    Review recent changes to mamba3 kernel for Phase N.
    Files modified: [list]
    Context: FP8 triton kernel rewrite for Mamba3 MIMO.
    Specific checks:
      1. Numerical parity test results (paste values)
      2. Memory savings measurements (paste values)
      3. bf16 fallback path preserved (verify)
      4. No new saved_tensors_hooks conflicts
      5. Grad correctness (relative diff < 10%)
    Return: PASS / FAIL with specific issues.
```

审查 FAIL 时立即停止后续步骤，修复后重跑当前 Phase 的失败步骤再次审查。

---

## 6. 回滚策略

任何阶段发现不可修复问题：

```bash
# 完整回滚
rm -rf /home/kt/ai/luma-architecture/third_party/mamba-official/mamba_ssm/ops/triton/mamba3
rm -rf /home/kt/ai/luma-architecture/third_party/mamba-official/mamba_ssm/ops/tilelang/mamba3
cp -r /home/kt/ai/luma-architecture/backups/mamba3_original_20260413/triton_mamba3 \
      /home/kt/ai/luma-architecture/third_party/mamba-official/mamba_ssm/ops/triton/mamba3
cp -r /home/kt/ai/luma-architecture/backups/mamba3_original_20260413/tilelang_mamba3 \
      /home/kt/ai/luma-architecture/third_party/mamba-official/mamba_ssm/ops/tilelang/mamba3
cp /home/kt/ai/luma-architecture/backups/mamba3_original_20260413/mamba3_module.py \
   /home/kt/ai/luma-architecture/minimind/model/mamba3_module.py
```

---

## 7. 启动前置条件

**此计划不在 Phase E hero run（gap17）期间启动**。等待以下信号之一：
1. Phase E 15k iter hero run 完成且 loss 稳定下降
2. 用户明确指示开始 FP8 Mamba3 工作
3. 当前 bf16 训练达到预期 milestone 后有空档

**启动时第一步**: 重读本文档，确认所有备份文件还在，创建 `artifacts/fp8_mamba3/baseline_metrics.json` 记录当前 bf16 基线数值。

---

## 附录 A: 参考资料

- FlashAttention v3 FP8 kernel (支持 per-tensor scale)
- TorchAO FP8 tensor API（`torch.float8_e4m3fn`）
- Mamba2 FP8 实验（社区第三方实现）
- NVIDIA Transformer Engine FP8 recipes（per-block scaling reference）

## 附录 B: 成功标准（可量化）

- [ ] bf16 ↔ fp8 forward MAX abs diff < 5e-3
- [ ] bf16 ↔ fp8 grad max rel diff < 10%
- [ ] Mamba 激活内存节省 ≥ 40%
- [ ] 50 iter 训练 loss 曲线 max rel diff < 10%
- [ ] seq=4096 训练可跑（之前 bf16 下 OOM）
- [ ] bf16 fallback 路径 100% 工作（fp8_kernel=0）
- [ ] 6 个子 agent 审查全部 PASS
