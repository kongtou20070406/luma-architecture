# Luma 4.10 进展报告

## 核心发现

**G0 的 h_diversity=0.33 改变了整个理论框架。**

之前认为：需要 slow_k=2 + cos_sigreg 才能创造循环间的 h 多样性。
现在发现：**循环本身就自然产生 h_diversity**，不需要人为干预。slow_k=2 和 cos_sigreg 不是在解决问题，而是在制造问题然后部分补偿。

---

## 一、理论框架演变

### 1.1 旧框架（4.9 报告时）

```
问题：c_t 冻结 → F 不变 → 循环坍缩
方案：打破 c_t 冻结（cos_sigreg）+ 让 c_t 改变 F（ct-LoRA/dt注入/FiLM）
```

### 1.2 中间框架（H/I 矩阵后）

```
修正：目标追踪动力系统
  h 追踪移动目标 h*(c_t)
  有效信息 ∝ ||h*(c_t_new) - h*(c_t_old)|| × (1 - L^n)
  slow_k=2 给 h 时间追踪，cos_sigreg 让目标移动
```

### 1.3 当前框架（G0 诊断后）

```
核心发现：h_diversity 是自发涌现的，不需要人为创造
  G0 (无干预): h_div=0.33, loss=5.53
  J0 (slow_k+cos): h_div=0.36, loss=5.78
  差异不显著（0.33 vs 0.36），但 loss 代价 6%

新问题：c_t 的方向多样性极低（ct_perp=0.01-0.05）
  c_t 像累加器，沿固定方向增长
  不是"c_t 不更新"，是"c_t 更新缺乏方向多样性"
  
新方向：不强迫 c_t 变化，而是让 c_t 的自然更新更有效率
```

---

## 二、实验结果汇总

### 2.1 H 矩阵 — 单一机制筛选

基底：G0 配置（hebb32 + jepa_surprise + c_t RMSNorm + cosine decay）
步数：2000，cosine_total_steps=3500

| 实验 | 配置 | loss | L_avg | slow | dead | loops3+ |
|------|------|------|-------|------|------|---------|
| G0 | baseline (slow_k=1) | **5.53** | 0.30 | 0 | — | 19% |
| H1 | ct_lora (raw c_t, tanh×0.1) | 13.40 | 0.81 | 8 | 0.1 | 17% |
| H2 | cos_sigreg=0.05 | 5.81 | 0.21 | 1 | 4.3 | 11% |
| H3 | ct_lora + cos_sigreg | 7.46 | 0.25 | 3 | 3.5 | 16% |
| **H4** | **slow_k=2** | **6.10** | **0.62** | **8** | **1.3** | **22%** |
| **H5** | **ct_momentum=0.5** | **5.80** | 0.31 | 8 | 2.8 | **23%** |
| H6 | ct_lora + cos + slow_k=2 | 7.55 | 0.45 | 7 | 3.0 | 18% |

**结论：** H4 (slow_k=2) 和 H5 (momentum=0.5) 是两个赢家。H4 结构最优（L=0.62），H5 loss 最优（5.80）。

### 2.2 I 矩阵 — slow_k=2 基础上组合

基底：G0 + slow_k=2

| 实验 | 配置 | loss | L_avg | slow | loops3+ |
|------|------|------|-------|------|---------|
| I1 | delta_ct_lora | 9.57 | 0.55 | 6 | 23% |
| I2 | delta_ct_lora + cos=0.05 | 12.96 | 0.26 | 6 | 28% |
| **I3** | **cos_sigreg=0.05** | **5.87** | **0.26** | **7** | **25%** |
| I4 | delta_ct_lora + momentum=0.3 | 10.49 | 0.57 | 6 | 27% |

**结论：** I3 (slow_k=2 + cos_sigreg) 是最优组合。ct-LoRA（无论 raw 还是 delta）在长训练中都不稳定。

### 2.3 J 矩阵 — I3 基础上直接改变 F

基底：I3 (slow_k=2 + cos_sigreg=0.05)，带 h_diversity 诊断

| 实验 | 配置 | loss | L_est | h_div | slow | loops3+ |
|------|------|------|-------|-------|------|---------|
| **J0** | **I3 重跑** | **5.78** | **0.68** | **0.36** | **8** | **24%** |
| J1 | dt_inject (Mamba SSM) | 7.84 | 0.97 | 0.48 | 8 | 19% |
| J2 | FiLM on CTInjection | 7.73 | 10.86 | 7.16 | 8 | 33% |

**结论：** 所有试图"增强"I3 的改动都在恶化。J1 dt 注入让收敛太快（h_diversity 降低），J2 FiLM 爆炸。

### 2.4 K 矩阵 — W_c 方向验证

| 实验 | 配置 | loss | h_div | 结论 |
|------|------|------|-------|------|
| K1 | I3 + ct_inject_scale=2 | 12.12 | 42.75 | ❌ P2 否定，W_c 方向关闭 |

### 2.5 G0 完整诊断 — 关键验证

| 指标 | G0 (无干预) | J0 (I3 配置) |
|------|------------|------------|
| loss | **5.87** | 5.78 |
| h_diversity | **0.33** | 0.36 |
| L_est | 0.5-0.7 | 0.7-0.8 |
| slow | 0-2 | 8 |
| ct_perp | **0.01-0.05** | ~1.0 |
| ct_traj cos | **0.97-0.99** | 0.95-0.99 |
| DOD rank | **5→7** | 5→8 |
| mode1% | **89→71** | 90→85 |

---

## 三、改动履历

### 3.1 代码改动

| 改动 | 原因 | 效果 |
|------|------|------|
| **赫布 hebb_norm_h/c RMSNorm** | 赫布正反馈导致 hebb_term 0.26→44→NaN | ✅ 打断正反馈 |
| **c_t output RMSNorm** | c_t 范数漂移导致下游 h 异常 | ✅ loss 从 9.88→5.56 |
| **gain 零参数化** | MLP 初始化零→sigmoid(0)=0.5→恒定 gain=1.5 | ✅ gain 跟随 surprise |
| **RLTT 采样限 3 份** | 19 份 logits 同时存在→22GB OOM | ✅ 防 OOM |
| **RLTT detach 中间 h** | 深循环计算图太大 | ✅ 防 OOM |
| **渐进热身** | 强制 step%20 遍历所有深度 | ❌ 放开后回退 |
| **ct-LoRA (raw c_t)** | 让 c_t 改变 F | ❌ 正反馈发散→NaN |
| **ct-LoRA (delta_c_t)** | 打断正反馈 | ⚠️ 稳定但 loss 差 |
| **ct-LoRA tanh×0.1** | 限制 LoRA 强度 | ❌ 太弱无效果 |
| **cos_sigreg** | 惩罚相邻 loop c_t 方向相似 | ⚠️ 有效但代价 6% |
| **loop SigReg** | 间接版 cos_sigreg | ❌ 不持久 |
| **CMDA token wish gate** | per-token 调制程度 | ❌ loss 翻倍 |
| **c_t gated attn** | c_t 条件门控注意力 | ❌ loss 翻倍 |
| **FiLM on CTInjection** | 乘法调制替代加法 | ❌ h_diversity 爆炸 |
| **dt_inject (Mamba SSM)** | c_t 调制 SSM 时间步长 | ❌ loss +36% |
| **ct_inject_scale=2** | 增大 W_c | ❌ h_diversity 爆炸 |
| **6 模块后置 RMSNorm** | 防数值漂移 | ❌ 导致 baseline NaN，已回滚 |
| **SWA in introspection** | 滑窗注意力补充 SSM | ❌ loss 恶化 |
| **FoX decay** | 赫布遗忘门 | ❌ 过度遗忘 |
| **PC error correction** | 自省流预测主流→误差修正 | ⚠️ 短实验有效，长训练拖后腿 |

### 3.2 诊断系统改动

| 指标 | 层级 | 说明 |
|------|------|------|
| L_est | 每步 | δh[1]/δh[0] 收缩率估计 |
| ct_perp | 每步 | c_t 变化方向和 c_t 自身的垂直度 |
| dt_ratio | 每步 | dt 注入量/基线比（dt_inject 专属） |
| step_norms | 每步 (loops≥2) | 每轮更新量绝对大小 |
| step_angles | 每步 (loops≥3) | 相邻更新方向 cosine（spiral refinement 指标） |
| accel_norms | 每步 (loops≥3) | 归一化加速度（二阶变化） |
| h_diversity | DOD 时 | 跨循环 h 标准差均值 |
| fixed_point L/slow/dead | 每步 (loops≥3) | SVD 方向分解 |
| interval max_loops (peak) | 每步 | 区间内最大循环数 |
| loss_pos head/mid/tail | 每步 | 按 token 位置分段 loss |
| world_jepa_loss | 每步 | World-JEPA loss 单独显示 |
| NaN watchdog | 每步 | 检测到 NaN 自动停训 |
| tok/s | 每步 | 训练速度 |

### 3.3 数据集改动

v5 数据集 532M tokens：
- Chinese Wikipedia 307M (STEM 优先)
- 知乎 KOL 80M
- 数学 CoT 91M (openr1 + numina + openmath + metamath + ultramath + deeptheorem)
- 代码 15M
- 对话 29M
- 科幻/小说 7M
- persona ×6 5M
- 毛选 2M

---

## 四、已验证无效的方向

| 方向 | 实验 | 为什么无效 |
|------|------|-----------|
| 强制深循环 | LD3/LD4/warmup | LoRA 没学到，放开后回退 |
| ct-LoRA (所有变体) | H1/H3/H6/I1/I2/I4 | 正反馈发散或太弱无效 |
| 直接改变 F | J1(dt)/J2(FiLM)/K1(scale) | 要么不稳定要么方向错 |
| 注入方式精细化 | G2/G3 | c_t 内容没变，精细化无用 |
| 增大 W_c | K1 | 目标移动量不是瓶颈 |
| 6 模块后置 RMSNorm | baseline NaN | 破坏 h 的自然范数动态 |
| RLTT + 深循环 | LD7-LD9 | OOM (logits 累积) |

---

## 五、当前状态

### 5.1 最优配置

**G0 配置**（loss=5.53，DOD rank=7，mode1=71%）：
```
hebb32 + jepa_surprise + c_t RMSNorm + cosine decay
slow_k=1, 无 cos_sigreg, 无 ct-LoRA
```

### 5.2 未解决的问题

**c_t 方向多样性极低（ct_perp=0.01-0.05）**

c_t 沿固定方向持续增长，像累加器而非工作记忆。每轮 c_t 更新方向和 c_t 自身几乎平行（cos≈0.98），没有根据不同循环步调整方向。

这意味着 c_t 在当前架构中没有发挥"工作记忆"的设计意图 — 它是一个静态偏置而非动态状态。

### 5.3 M 矩阵 — per-layer c_t 注入（4.10 新增）

**核心思路：** β≈0 不是 Mamba 结构限制，是单次注入→ρ⁴衰减→introspection 看不到差异的反馈环路。per-layer 注入让 c_t 在每层被重新施加，打破衰减链。

| 实验 | 配置 | ct_perp | loss | 结论 |
|------|------|---------|------|------|
| M1 | per-layer 共享 proj (200步) | 0.929→0.043 | 7.41 | 延迟冻结但未阻止（共享 proj → rank=1） |
| M2 | per-layer 独立 proj | OOM | — | nn.Embedding 梯度路径穿透 shared layers |
| M2v2 | detach(c_bias) × direction vectors | 0.012@400步 | 进行中 | ct_perp 和 G0 一样低，detach 切断了信号 |

**M2v2 失败分析：** detach(c_bias) 意味着 per-layer injection 是常数偏置，不随循环变化。c_t 方向冻结后，所有 direction-scaled 版本也冻结。方向多样性需要 c_t 在循环间变化，但 detach 恰好切断了这个来源。

**OOM 根因追踪（教训）：**
- 默认 batch_size=4（应为 1）
- 默认 compression_layers=24（应为 16）
- 默认 reason_shared_depth=2（应为 4）
- 默认 phase=4（应为 6）
- 未开 gradient_checkpointing、cpu_offload_optimizer
- 以上遗漏导致连续 OOM 5 次，已固化到 run_experiment.sh 和 CLAUDE.md

### 5.4 五个核心动力学方程（4.10 新增）

| 方程 | 公式 | 测量量 | 预测能力 |
|------|------|--------|----------|
| Eq.1 | 残差 = ρ^k × 初始残差 | L_est | 收敛速度 |
| Eq.2 | ∂h*/∂c_t ≈ ρ/(1-ρ) · ‖W_c‖ | ct_inj + ρ | c_t 影响力放大 |
| Eq.3 | α_crit = √((1-ρ₀)/γ), γ≈200-250 | ct_inj | 相变边界 |
| Eq.4 | β = f(rank(W_eff)) | ct_perp | 注入架构→方向多样性 |
| Eq.5 | ct_perp(t) = ct_perp₀·e^(-t/τ) + β·h_div | ct_perp 轨迹 | 终值预测, τ≈60步 |

完整推导见 Luma_Dynamics_Analysis_Skill.md §2.5。

### 5.5 下一步方向

1. **per-layer 注入不 detach** — 需要解决 OOM：用 gradient checkpointing 包裹 per-layer inject，或预计算 bias 后只让 proj 参数通过辅助 loss 学习
2. **长程预训练** — G0 配置跑完整 v5，验证模型质量
3. **acceleration-based exit** — 用 h 轨迹几何判断替代/辅助 learned exit

---

## 六、理论总结

### 6.1 压缩映射与自发多样性

推理循环 F 是压缩映射（L≈0.3-0.7），h 快速收敛到不动点。但收敛过程本身产生了 h_diversity=0.33 — 这不是 bug，是 spiral refinement 的自然结果。

### 6.2 人为干预的代价

所有试图"增强"循环动态的干预（cos_sigreg, ct-LoRA, dt_inject, FiLM, scale）都有代价：
- 要么 loss 恶化（+6% 到 +120%）
- 要么数值不稳定（NaN, h_diversity 爆炸）
- 要么无效果（被压缩映射的数学结构吸收）

### 6.3 β 的真实本质（4.10 修正）

旧结论"β≈0 是 Mamba 结构限制"被 M1 实验否定。

β 是注入架构与 h 收敛动态的反馈环路产物：
- 单次注入 → c_t 影响被 ρ⁴ 衰减 → introspection 看不到差异 → β≈0
- Per-layer 注入 → c_t 每层重新施加 → introspection 看到差异 → β>0（至少暂时）
- 但共享 proj → rank(W_eff)=1 → 仍然趋向 β=0
- 独立 proj → rank(W_eff)=4 → β>0（但 detach 会切断信号）

### 6.4 c_t 的真实角色

c_t 在当前架构中是**静态偏置**而非**动态工作记忆**。它每轮更新但方向不变（ct_perp≈0.01），实质上只是在平移不动点而不是改变思考方向。

要让 c_t 成为真正的工作记忆，需要让 per-layer 注入的梯度路径可行（不 OOM），使 c_t 的变化能真正影响 introspection 的输入。

---

## 七、工程改进（4.10 新增）

| 改动 | 说明 |
|------|------|
| **run_experiment.sh** | 标准实验启动脚本，固化 ARCH+TRAIN+IS9+G0 全套参数 |
| **packed_dataset.py 缓存修复** | Path.resolve() 确保相对/绝对路径命中同一缓存 |
| **per-layer inject 重构** | 从 SharedLayer 内的 Embedding 迁移到 ReasonCore 的 direction vectors |
| **CLAUDE.md 更新** | 写入完整训练启动模板，防止参数遗漏 |
