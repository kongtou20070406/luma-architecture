# Luma 4.9 进展报告

## 核心目标

**让循环推理真正有用 — loop 3+ 能降低 loss，而不只是跑空轮。**

当前 Luma 的推理循环在 86% 的样本上 loop 2 就退出。不是 exit 机制有问题，而是 loop 3+ 确实没有产生有效计算。

---

## 一、问题诊断

### 1.1 现象

```
loop 1: δh = 1.000  (基准)
loop 2: δh = 0.350  (快速衰减)
loop 3: δh = 0.120  (几乎为零)
loop 4: δh = 0.040  (完全无效)
```

循环深度分布：loops=2 占 86%，loops=3 占 10%，loops=4+ 占 4%。

### 1.2 根因链

```
shared_layers 是压缩映射 (L ≈ 0.3)
    ↓
h 在 loop 2 后距不动点只剩 9%
    ↓
c_t 方向在 loop 2 后冻结 (cosine > 0.99)
    ↓
c_t 注入 h 的信号每轮几乎相同 (加法偏移不变)
    ↓
JEPA 完美预测 "什么都不变" → surprise = 0
    ↓
赫布不写入 → c_t 继续不变
    ↓
exit controller 看到 "已收敛" → 退出
```

**自我强化的死循环。** 每个环节都在加固"不需要深循环"这个结论。

### 1.3 数学本质

推理循环 $h_{t+1} = F(h_t, c_t)$ 中，$F$ 的 Jacobian 谱半径 $\rho(J_h) \approx 0.3$：

$$\|F(h_1, c) - F(h_2, c)\| \leq 0.3 \|h_1 - h_2\|$$

loop 2 后残差 $0.3^2 = 9\%$。**不是"还没收敛"，是"收敛到了一个不够好的不动点"。**

当前 c_t 对不动点的影响力通过隐函数定理：

$$\frac{\partial h^*}{\partial c_t} = (I - J_h)^{-1} J_h W_c \approx 0.3 \cdot W_c$$

c_t 的影响被压缩了 70%。**c_t 在"耳边低语"，不是在"改变规则"。**

---

## 二、实验历程

### 2.1 IS 矩阵 — 自省流优化 (10 实验)

**核心发现：** Memory Token K=4 + CMDA 双向调制是最优自省流配置 (-15.6%)。

| 方案 | loss | vs baseline |
|------|------|------------|
| IS0 baseline | 8.85 | — |
| IS9 Memory K=4 + CMDA | **7.47** | **-15.6%** |
| IS3 Chunked pooling | 7.55 | -14.7% |

### 2.2 NM+ES 矩阵 — 赫布 + 退出信号 (28 实验)

**核心发现：** Hebbian rank=32 是短实验冠军 (-23.9%)，但长训练崩溃。

| 方案 | 500步 loss | 1000步 loss |
|------|-----------|------------|
| NM8 hebb32 (surprise) | **6.35** (-23.9%) | 10.61 (崩了) |
| NM10 hebb32 (jepa) | 6.55 (-21.6%) | **7.33** (-14.9%) |
| L8 jepa+hebb32+cosine | — | **6.33** (-26.5%) |

赫布在短实验防遗忘，长训练噪声累积。JEPA surprise 比 self_check surprise 稳定。

### 2.3 赫布数值稳定性

| 问题 | 原因 | 修复 |
|------|------|------|
| hebb_term norm 爆炸 (0.26→44→NaN) | 正反馈: c_t 大→hebb 大→c_t 更大 | hebb 输入 RMSNorm |
| c_t 范数漂移 | 赫布写入后 c_t 无归一化 | c_t 输出 RMSNorm |
| RLTT OOM (22GB logits) | 19份 [1,2048,151936] 同时存在 | 采样 3 份 |
| gain 恒等于 1.5 | MLP 初始化为零, sigmoid(0)=0.5 | 改成零参数 1+surprise |

### 2.4 长程预训练 (v5 数据集, 7K-17K 步)

v5 数据集: 532M tokens (中文 Wikipedia + 知乎 + 数学 + 代码 + persona)

| 实验 | loss | 状态 |
|------|------|------|
| baseline (IS9 + cosine) | 5.47 @ 6850步 | 稳定 |
| hebb (无 RMSNorm) | NaN @ 1950步 | 赫布爆炸 |
| hebb (input RMSNorm) | 9.88 @ 7001步 | 不够好 |
| **hebb (c_t output RMSNorm)** | **5.56 @ 7001步** | **最优** |
| hebb + FoX decay | 9.43 | FoX 过度遗忘 |
| hebb + SWA | 8.83 | SWA 干扰 Mamba |
| hebb + warmup | 34.81 | 深循环崩溃 |

### 2.5 循环深度探索

| 方案 | 效果 | 结论 |
|------|------|------|
| exit bias=-1 | avg 2.2→2.4, loss +1.9% | 安全推深但没用 |
| warmup=200 (强制深循环) | avg=8.6, loss +8.9% | 能训出深度但 loss 差 |
| 渐进热身 (step%20 遍历深度) | 热身后回退到 loops=2 | LoRA 没学到 |
| RLTT (per-loop loss) | 没推深, loss 恶化 | 信号太弱 |
| loop SigReg | 偶尔 loops=3, 不持久 | 间接作用不够 |
| JEPA dropout 0.2 | 热身期 δh 稳定, 放开后衰减 | 暂时有效 |

### 2.6 G 矩阵 — 注入方式 (3 实验)

| 方案 | loss | 结论 |
|------|------|------|
| G0 baseline | **5.53** | 最优 |
| G2 CMDA token wish | 14.80 | 崩了 |
| G3 c_t gated attn | 11.53 | 差 |

**注入方式的精细度不是问题。** 问题是注入的内容 (c_t) 没变。

### 2.7 H 矩阵 — 不动点结构改变 (进行中)

| 方案 | L_global | slow_dirs | loss | 状态 |
|------|---------|-----------|------|------|
| G0 baseline | 0.3 | 0 | 5.53 | 对照 |
| H1 ct-LoRA (tanh×0.1) | 0.27→1.0 | 0→7 | 13.4 | **推高了 L 但发散** |
| H2 cos_sigreg | 0.1-0.36 | 0 | 进行中 | L 没变 |
| H3 ct-LoRA + cos_sigreg | 0.1-0.7 | 0-3 | 进行中 | **偶尔活** |
| H4 slow_k=2 | — | — | — | 排队 |
| H5 ct_momentum=0.5 | — | — | — | 排队 |
| H6 ct-LoRA + cos + slow_k=2 | — | — | — | 排队 |

---

## 三、动力学监控体系

### 3.1 已实现的指标

| 指标 | 说明 | 采样频率 |
|------|------|----------|
| loss_lm / loss_c / loss_j / loss_w | 主 loss + 压缩 + Self-JEPA + World-JEPA | 每 25 步 |
| loops / peak | 当前循环数 / 区间最大循环数 | 每 25 步 |
| per_loop δh / ct_change / jepa_err | 每轮的主流变化 / c_t 变化 / JEPA 误差 | 每 25 步 |
| hebb gain / norm | 赫布调制强度 / 赫布项范数 | 每 25 步 |
| ct_traj cosine | 相邻循环 c_t 方向相似度 | 每 25 步 |
| loss_pos head/mid/tail | 按 token 位置分段 loss | 每 25 步 |
| **fixed_point L / dirs / slow / dead** | **不动点收缩率 / 方向分解 / 慢方向数 / 死方向数** | 每 25 步 (loops≥3) |
| DOD v2_rank / mode1 / dmd_radius | 梯度多样性 / 集中度 / 稳定性 | 每 100 步 |
| NaN watchdog | 自动停训 | 每步 |

### 3.2 不动点分析 (新增)

```
fixed_point: L=0.310  dirs=[0.41, 0.28, 0.19, 0.08]  slow=1  dead=2
```

- **L** = 全局收缩率 (<0.3 太快, 0.5-0.7 理想, >0.95 发散)
- **slow** = L>0.5 的方向数 (**核心指标, 目标从 0 推到 3-5**)
- **dead** = L<0.05 的方向数

---

## 四、当前困境

### 4.1 主要矛盾

**c_t 对 F 的影响力太弱 — "耳边低语"而非"改变规则"。**

当前 c_t 通过加法偏移注入 h: `h = h + proj(c_t)`。这只平移不动点，不改变 F 的 Jacobian。不管 c_t 怎么变，F 的收缩结构不变，收敛速度不变。

### 4.2 ct-conditioned LoRA 的困境

ct-LoRA 理论上正确 — 让 c_t 改变 F 本身: `F(h; c_t) = (W + ΔW(c_t)) h`

但实验暴露了**稳定性-有效性困境**:

- **缩放 ×1.0**: L 从 0.3 推到 0.97 → slow=6 但 NaN (H1 旧版)
- **缩放 ×0.1**: L 保持 0.27 → slow=0, 无效果 (H1 新版)
- **缩放 ×0.1 + cos_sigreg**: L 偶尔到 0.7 → slow=3 偶尔, 不稳定 (H3)

**正反馈循环**: c_t 变 → LoRA 变 → F 变 → h 变 → c_t 更变 → 发散

### 4.3 已排除的方向

| 方向 | 为什么不行 |
|------|-----------|
| 强制深循环 (warmup/min_loops) | LoRA 没学到, 放开后回退 |
| 退出信号优化 (ES1-ES5) | 推深了循环但 loop 3+ 仍是噪声 |
| 注入方式精细化 (token wish, gated attn) | c_t 内容没变, 精细化无用 |
| RLTT (per-loop loss) | OOM + 梯度信号太弱 |
| loop SigReg | 间接作用, 不改变 F |
| FoX 遗忘 / SWA 注意力 | 副作用大于收益 |

---

## 五、下一步方向

### 5.1 核心问题

**如何让 c_t 改变 F 的不动点结构, 同时保持数值稳定?**

### 5.2 待验证方案 (H 矩阵进行中)

1. **H4 slow_k=2** — 恢复快慢时间尺度分离
2. **H5 ct_momentum=0.5** — EMA 式慢更新
3. **H6 ct-LoRA + cos_sigreg + slow_k=2** — 完整理论组合

### 5.3 待探索方向

1. **ct-LoRA 缩放系数调参** — 找 0.1-1.0 之间的 sweet spot (可能 0.3)
2. **ct-LoRA 输入改为 delta_c_t** — 用变化量而非绝对值, 打断正反馈
3. **谱归一化 ct-LoRA** — Lipschitz 约束保证 L < 1
4. **DEQ (Deep Equilibrium) 训练** — 隐式微分, 强迫不动点对输入敏感
5. **数据驱动**: 只在困难样本 (math competition) 上推深循环

### 5.4 架构级思考

> **c_t 不应该是 F 的参数, 而应该是 F 的控制变量。**

区别:
- 参数: c_t 进入 F 后被 F 的计算结构吸收, 影响力受限于 F 的 Lipschitz 常数
- 控制变量: c_t 直接改变 F 的结构 (Jacobian), 影响力不受 F 内部约束

ct-conditioned LoRA 是走控制变量路线的第一步, 但需要解决稳定性问题。

---

## 六、基础设施

### 6.1 数据集

- v5: 532M tokens (Wikipedia 307M + 知乎 80M + 数学 91M + 代码 15M + 对话 29M + persona 5M)
- 中文 48%, 英文 52%
- 待补: SwallowCode/Math (100M tokens 中文代码/数学, 网络问题未拉到)

### 6.2 训练配置

- 模型: 293M 参数, CR5 (c16_d4)
- 硬件: RTX 5090 32GB, WSL2
- FP8 forward + BF16 backward
- CPU offload optimizer
- 训练速度: ~4000 tok/s

### 6.3 最优基线 (v5_hebb_ct_norm_7k)

```
loss = 5.56 (7001 步)
v2_rank = 8/26 (历史最高)
mode1 = 82.7%
配置: IS9 + hebb32 + jepa_surprise + c_t RMSNorm + cosine decay
```
