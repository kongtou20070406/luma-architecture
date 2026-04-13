# NaN 崩溃调查报告 (2026-04-11)

## 背景

G0 长训练（v5 + seq=2048 + FP8 + Muon）在 step 5800 崩溃。本报告记录了对 NaN 根因的系统性调查。

---

## 一、原始崩溃数据

**配置：** G0 baseline, v5 数据集, seq=2048, FP8, Muon for W_c, 无保护

**崩溃时间线：**

| step | ct_inj | ct 范数 | L_est | hebb_norm | 状态 |
|------|--------|---------|-------|-----------|------|
| 2000 | 0.025 | 15 | 0.55 | ~100 | 正常 |
| 5000 | 0.037 | 20 | 0.59 | ~500 | 正常 |
| 5800 | 0.047 | 25 | 0.59 | ~700 | 接近边界 |
| 5850 | 0.056 | 25 | 0.57 | — | 越过 α_crit |
| 6200 | 0.115 | 29 | 2.02 | — | 发散开始 |
| 6350 | 0.263 | 568 | 1.95 | 1544 | 正反馈失控 |
| 6550 | 0.763 | 1832 | 2.11 | — | 即将 NaN |
| 6585 | NaN | — | — | — | 崩溃 |

**NaN watchdog 捕获位置：** `loss.item()` — loss 本身变成 NaN。

---

## 二、验证实验矩阵

所有验证实验使用 **小数据集 + seq=512 + FP8**。

### 2.1 极端初始化实验（200步内触发 NaN 的尝试）

| 实验 | scale | hebb_init | ct_out_scale | 保护 | 结果 | ct_max | L_est_max |
|------|-------|-----------|-------------|------|------|--------|-----------|
| scale3 | 3 | 0 | 0 | 开 | 不NaN | — | 1.02 |
| scale5 | 5 | 0 | 0 | 关 | 不NaN | ~15 | 10.0 |
| scale20 | 20 | 0 | 0 | 关 | 不NaN | ~10 | 5.66 |
| hebb10 | 1 | 10 | 10 | 关 | 不NaN | 162 | 33.5 |
| hebb100 | 1 | 100 | 100 | 关 | 不NaN | 4352 | 3040 |
| hebb300 | 1 | 300 | 300 | 关 | 不NaN | 4352 | 27264 |
| force20 | 10 | 0 | 0 | 关 | 不NaN | ~13 | 2.5 |

**结论：200 步内无法触发 NaN，无论初始化多极端。**

### 2.2 长步数验证（Muon + scale=3 + 无保护，小数据集）

| 实验 | 步数 | ct_max | ct_inj_max | L_est_max | NaN |
|------|------|--------|-----------|-----------|-----|
| trace1 | ~800 | 354 | 6.7 | 6.4 | 否 |
| trace2 | ~1500+ | 616 | 15.9 | 7.8 | 否（进行中） |

**结论：小数据集 + seq=512 条件下，即使 ct=616, ct_inj=16, L_est=8 也不 NaN。**

### 2.3 NaN 探针结果

在 forward（h, c_t, logits）和 backward（所有参数梯度）部署了 NaN 探针。

**截至 ct=616：所有探针零触发。** h 和 c_t 全程 finite，所有参数梯度全程 finite。

---

## 三、已证伪的假说

| 假说 | 预测 | 实测 | 状态 |
|------|------|------|------|
| ct 数值溢出 (bf16 ct²>65504) | ct 元素>256 → NaN | ct=4352(元素=544) 不NaN | **证伪** |
| h 发散 (ρ>1 → h 范数爆炸) | L_est>1 → h→inf | L_est=27264, h 被RMSNorm约束 | **证伪** |
| ρ^20 指数放大 | 强制20轮 → overflow | 20轮后 h 仍 finite | **证伪** |
| 极端初始化可复现 NaN | 初始化到崩溃值 → 立即NaN | ct=4352, ct_inj=2598 不NaN | **证伪** |
| 赫布正反馈触发 NaN | hebb_norm 大 → ct 爆炸 → NaN | hebb_init=300, ct 大但不NaN | **证伪** |

## 四、仍然成立的预测

| 预测 | 精度 | 数据来源 |
|------|------|---------|
| ct_inj 二次增长 α(t)=0.0188+2.17e-6·t+4.63e-10·t² | 3.5% (scale=3 step50) | G0 原始数据 5 点拟合 |
| α_crit≈0.045 (L_est 越过 1.0) | ✅ scale=3 时 ct_inj=0.055→L_est=1.02 | 多实验一致 |
| 崩溃步数 t_crash=(α_crit-α₀)/κ≈5531 | 4.6% 误差 (实测5800) | 原始 G0 崩溃 |
| ct_perp 指数衰减 τ≈60步 | ✅ 多实验一致 | no-LoRA, G0, M1 |
| Muon 下 W_c 持续增长 | ✅ ct_inj 单调递增 | trace1/trace2 |
| 行范数归一化+ct clamp 阻止 ct_inj 发散 | ✅ scale=5 保护开不崩 | VERIFY_crash_scale5 |

## 五、原始崩溃 vs 验证实验的关键差异

| 因素 | 原始崩溃 | 验证实验 | 可能影响 |
|------|---------|---------|---------|
| 数据集 | v5 (160889 packs) | 小诊断集 (255 packs) | 数据多样性影响梯度分布 |
| seq_len | 2048 | 512 | 4× 更长序列，更大中间张量 |
| 训练步数 | 5800 | 200-1500 | 权重累积效应 |
| 优化器状态 | 5800步 Muon momentum | 从零开始 | momentum 矩阵可能近奇异 |
| FP8 精度 | 129 层 FP8 forward | 129 层 FP8 forward | 相同 |

**最可能的差异来源：**

1. **seq=2048 的中间张量更大**：attention scores [B, heads, 2048, 2048] vs [B, heads, 512, 512]。极端 logits 下 softmax 可能产生更精确的 0 → log(0) = -inf。

2. **Muon momentum 累积 5800 步**：rank-1 梯度持续累积，momentum 矩阵可能接近 singular，Newton-Schulz 正交化数值不稳定 → 产生 inf 更新 → NaN。这在 200 步内无法复现。

3. **backward pass 梯度链更长**：seq=2048 时，attention 的 backward 涉及 [2048, 2048] 矩阵求导，bf16 精度下梯度乘积可能溢出。

---

## 六、下一步调查方向

1. **用 v5 + seq=2048 复现**：在原始条件下跑，带 NaN 探针，看是否在 step 5800 附近崩溃，探针定位具体层。
2. **关闭 FP8 测试**：用 bf16 全精度跑长训练，看是否仍崩溃。如果不崩 → FP8 是根因。
3. **Muon momentum 监控**：在 Muon step 里加 momentum 矩阵条件数监控，看是否在崩溃前急剧增大。
4. **torch.autograd.detect_anomaly**：开启自动异常检测，精确定位 backward 中的 NaN 源。

---

## 七、实验日志索引

| 实验 | 日志文件 |
|------|---------|
| 原始 G0 半epoch v1（首次崩溃） | `artifacts/G0_half_epoch.log` |
| scale=3 保护开 | `artifacts/VERIFY_crash_scale3.log` |
| scale=5 保护关 | `artifacts/VERIFY_A_no_protect.log` |
| scale=20 保护关 | `artifacts/VERIFY_A_scale20.log` |
| hebb=10 + scale=10 全大 | `artifacts/VERIFY_F_all_big.log` |
| hebb=100 保护关 | `artifacts/VERIFY_G_ct_overflow.log` |
| hebb=300 保护关 | `artifacts/VERIFY_H_scale300.log` |
| 强制20轮 scale=10 | `artifacts/VERIFY_E_force20loops.log` |
| Muon+scale=3 NaN探针 trace1 | `artifacts/VERIFY_NaN_trace.log` |
| Muon+scale=3 NaN探针+梯度探针 trace2 | `artifacts/VERIFY_NaN_trace2.log` |
