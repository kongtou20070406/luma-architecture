# Luma 4.11 进展报告

## 核心发现

**长训练 NaN 崩溃的根因是 Muon 优化器 + c_t 方向固定的耦合效应。**

c_t 方向冻结（人格特征）→ W_c 梯度每步同方向 → Muon 正交化更新持续推高 W_c 范数 → ct_inj 越过 α_crit → 循环发散 → NaN。

修复方案：将 ct_injection/c_t_head/hebb_out 从 Muon 转到 AdamW，让 weight decay 直接约束范数增长。

---

## 一、c_t = 人格（范式转换）

### 1.1 旧框架（4.10 报告时）

```
问题：c_t 方向冻结（ct_perp≈0）是需要修复的缺陷
方向：per-layer 注入、独立 proj、提高 β
```

### 1.2 新框架（4.11）

```
核心认知：c_t 方向稳定不是 bug，是人格特征

c_t 方向 = 人格（稳定，不随推理步变化）
c_t 范数 = 人格强度（随训练/经验增长）
h = 工作记忆（h_diversity=0.33，spiral refinement）
赫布写入 = 人格强化（surprise × hebb → 强化已有方向）
PC 误差 = 情绪反应（人格视角下的预期违背）
Loop LoRA = 思考阶段（per-loop 差异化 = 分阶段处理）
```

**关键推论：** 所有强迫 c_t 方向变化的实验（cos_sigreg, ct-LoRA, FiLM, dt_inject）都恶化 loss — 相当于强迫模型每步换人格。G0（零干预）最优是因为人格稳定本来就是对的。

---

## 二、长训练 NaN 崩溃分析

### 2.1 崩溃时间线（v1，step 5800-6585）

| step | ct_inj | ct 范数 | L_est | 状态 |
|------|--------|---------|-------|------|
| 5000 | 0.025 | 15 | 0.55 | 正常 |
| 5800 | 0.047 | 20 | 0.59 | 接近边界 |
| 5850 | 0.056 | 25 | 0.57 | 越过 α_crit |
| 6200 | 0.115 | 29 | 2.02 | 发散开始 |
| 6350 | 0.263 | 568 | 1.95 | 正反馈失控 |
| 6585 | NaN | — | — | 崩溃 |

### 2.2 根因链

```
c_t 方向固定 (ct_perp≈0)
  → W_c 梯度 ∂L/∂W_c ∝ c_t，每步同方向
  → Muon 正交化更新持续推高 W_c 范数
  → weight_decay = lr × wd = 0.02 × 0.1 = 0.002/step，不够压
  → ct_inj = ‖W_c · c_t‖/‖h‖ 线性增长 (~0.005/1000步)
  → step ~5800: ct_inj 越过 α_crit≈0.04-0.05
  → ρ(α) = ρ₀ + γ·α² > 1.0 (Eq.3 验证)
  → L_est > 1.0，循环不再收缩而是发散
  → δh 大 → 赫布写入更多 → ct 范数爆炸 → 正反馈 → NaN
```

### 2.3 为什么 2000 步实验没发现

ct_inj 增长是线性的。预测崩溃时间：

```
t_crash = (α_crit - α₀) / κ = (0.045 - 0.02) / 0.000005 ≈ 5000步
```

实测 5800 步崩溃，和预测高度吻合。2000 步实验系统性地低估了长训练风险。

### 2.4 修复尝试履历

| 尝试 | 方案 | 结果 | 失败原因 |
|------|------|------|----------|
| v1 | soft clamp ct_inj (sigmoid, α_crit=0.05) | ct_inj→0.001 | W_c 梯度消失，ct_inj 萎缩 |
| v2 | hard clamp ct_inj=0.04 | ct_inj→0.001 | 同上 |
| v2 续 | 改 hard clamp=0.04 + detach | ct 范数仍爆炸 | 赫布写入不经过 ct_inj clamp |
| v3 | weight_norm on W_c | FP8 crash | weight_norm 和 FP8 不兼容 |
| v3 续 | W_c 手动行范数归一化 | ct_inj 正常 | 但 ct 范数从其他路径增长 |
| v4 | ct 范数 hard clamp=50 (NeuromodWriter) | ct=360 穿越 | clamp 只在 loop_idx>0 生效 |
| v5 | ct 范数 clamp 放到 Backbone.forward | ct 被压到 50 但 ct_inj=0.001 | checkpoint 的 W_c 配合 ct=360 训练的 |
| v5 续 | max_ct_norm=50 从头跑 | step 1500 ct_inj=0.046 | 50 太高，ct 撞墙前已发散 |
| v6 | max_ct_norm=20 从头跑 | 待验证 | — |
| **v7** | **ct_injection/c_t_head/hebb_out → AdamW** | **待验证** | **根本修复** |

### 2.5 最终修复方案（v7）

**根本修复：** 将三个发散源头从 Muon 转到 AdamW

```python
# optimizers.py CONTROL_TENSOR_NAME_PATTERNS 新增：
"ct_injection",   # W_c: Muon正交化 + 固定方向梯度 → wd不够压
"c_t_head",       # introspection输出层
"hebb_out",       # 赫布写入层
```

**安全网（保留）：**
1. W_c 行范数归一化（forward 里 clamp 行范数 ≤ 1.0）
2. hebb_out 行范数归一化（同上）
3. c_t_head 行范数归一化（同上）
4. ct 范数 hard clamp ≤ 20（Backbone.forward + NeuromodWriter）
5. ct_output_rmsnorm（归一化方向）

---

## 三、M 矩阵（per-layer c_t 注入）

### 3.1 实验结果

| 实验 | 配置 | ct_perp | 结果 |
|------|------|---------|------|
| M1 | per-layer 共享 proj (200步) | 0.929→0.043 | 延迟冻结但未阻止 |
| M2 | per-layer 独立 Embedding proj | OOM | 梯度路径穿透 shared layers |
| M2v2 | detach(c_bias) × direction vectors | 0.012 | detach 切断信号 |

### 3.2 OOM 根因追踪

连续 5 次 OOM 的根因不是 per-layer 注入，而是启动参数遗漏：

| 遗漏参数 | 默认值 | 正确值 | 影响 |
|----------|--------|--------|------|
| batch_size | 4 | 1 | 4× VRAM |
| compression_layers | 24 | 16 | 模型结构不同 |
| reason_shared_depth | 2 | 4 | 模型结构不同 |
| phase | 4 | 6 | 缺少 World-JEPA |
| gradient_checkpointing | 0 | 1 | 循环激活不释放 |
| cpu_offload_optimizer | 0 | 1 | 优化器状态占 VRAM |

**修复：** 创建 `run_experiment.sh` 固化全套参数，写入 CLAUDE.md。

### 3.3 结论

在人格框架下，per-layer 注入的价值不是改变 c_t 方向（那是破坏人格），而是加强人格渗透力（单次注入被 ρ⁴ 衰减）。暂时搁置，先完成长训练稳定性验证。

---

## 四、动力学方程更新

### 4.1 Eq.3 修正：α_crit 不是静态的

原公式假设 α 是常数。实际 α(t) 在训练中线性增长：

```
α(t) ≈ α₀ + κ·t    (κ ≈ 0.005/1000步)
t_crash = (α_crit - α₀) / κ
```

G0 实测：α₀=0.02, α_crit≈0.045, κ≈0.000005 → t_crash≈5000步。实测 5800 步崩溃。

### 4.2 新增 Eq.6：W_c 范数增长方程

```
d‖W_c‖/dt = ‖∂L/∂W_c‖ - wd × lr × ‖W_c‖

Muon: ‖∂L/∂W_c‖ ≈ const (c_t 方向固定 → 梯度方向不变)
      wd_eff = 0.002/step → 不够压

AdamW: wd_eff = 0.1 × lr → 更有效
       且 Adam 的自适应学习率会随梯度稳定而降低
```

---

## 五、工程改进

| 改动 | 说明 |
|------|------|
| **run_experiment.sh** | 标准实验启动脚本，固化 293M 架构全套参数 |
| **CLAUDE.md** | 写入完整训练启动模板 |
| **packed_dataset.py** | Path.resolve() 修复缓存路径 |
| **optimizers.py** | ct_injection/c_t_head/hebb_out → AdamW |
| **CTInjection** | W_c 行范数归一化 |
| **NeuromodulatedCTWriter** | hebb_out 行范数归一化 + ct 范数 clamp |
| **IntrospectionStateStream** | c_t_head 行范数归一化 |
| **Backbone.forward** | ct 范数 clamp ≤ 20（所有路径的最终出口） |
| **hebb_write 监控** | 新增 (hebb_gate × hebb_term).norm() 真实写入量 |

---

## 六、当前状态

### 6.1 待验证

**v7 配置**（AdamW + 三重行范数归一化 + ct clamp 20）从头跑 0.5 epoch。

预期行为：
- ct 范数稳定在 8-20（不再无限增长）
- ct_inj 稳定在 0.015-0.030（不越过 α_crit）
- hebb_norm 稳定（不单调递增）
- 通过 step 6000+ 不崩溃

### 6.2 G0_jepa_enhanced 实验（进行中）

**核心改动：** 让赫布重新苏醒。

| 参数 | G0 v7 | jepa_enhanced |
|------|-------|---------------|
| world_mask_ratio | 0.25 | **0.7** |
| h_mask_ratio | 0 | **0.25** |
| h_mask_surprise_weight | — | **0.3** |

**机制：**
- mask=70% 让 World-JEPA 预测任务持续有难度 → surprise 不归零
- h_mask_predictor：c_t 直接预测 h 的随机 mask 维度 → c_t→h 理解程度的测量
- h_mask 误差以 30% 权重混入 surprise → 赫布不再沉睡

**Step 50 初步数据：**
- jepa=[0.176, 0.242]（vs v7 的 0.000）— JEPA 在工作
- gain=1.34（vs v7 的 1.00）— 赫布在写入
- ct_perp=0.395（vs v7 的 0.000）— c_t 方向在变化
- ct_traj cos=0.482（vs v7 的 1.000）— 人格在进化

**人格框架解读：** 不再是"僵死的人格"，而是"有持续经验输入的活人格"。surprise 来自真实的预测困难，不是人为强制（cos_sigreg），所以方向变化是有信息内容的。

**风险：** 赫布积极写入可能加速 ct 范数增长→触发 clamp=20。需要监控。

### 6.3 下一步

1. 监控 G0_jepa_enhanced 长训练稳定性
2. 如果稳定，对比 loss：jepa_enhanced vs v7
3. 评估 ct_perp 是否维持 >0.1（不像之前掉到 0）
4. 人格注入实验（推理时初始化 c_t 方向）
