# Luma 长训练动力学危机 — 决策文档

**日期：** 2026-04-12
**核心目标：** 不是加更多约束，而是从架构和动力学数学层面找出根因并重构。

---

## 参考文档（必读）

- **动力学分析框架：** [`/home/kt/ai/Luma_Dynamics_Analysis_Skill.md`](../../Luma_Dynamics_Analysis_Skill.md)
  - 完整的数学推导（§1.4.1-1.4.10）
  - 五个核心方程（§2.5）
  - 所有变量→公式→效果映射表（§1.5）
  - 已有实验数据拟合（§2.4）

- **代码路径：**
  - 模型主文件：[`minimind/model/model_minimind.py`](../../minimind/model/model_minimind.py) (~5000 LOC)
  - 关键类：`CTInjection` (1186), `NeuromodulatedCTWriter` (1963), `IntrospectionStateStream` (~1600), `LumaReasonCore` (2986), `LumaBackbone` (~3870)
  - 优化器：[`minimind/luma_stage0/optimizers.py`](../../minimind/luma_stage0/optimizers.py)

---

## 一、核心目标

**Luma 是一个 293M 参数的循环推理语言模型。** 架构：

```
Input → Embedding → Compression(16 层，一次) → ReasonLoop(4 层×N 轮) → LMHead
                                                ↑ ↓
                                            c_t (认知流/人格)
                                            introspection (自省流)
                                            Hebbian 赫布写入
```

**目标：** 用 v5 数据集（~330M tokens）跑完 0.5 epoch（40222 步）的预训练，验证模型质量。

**当前的范式共识：**
- c_t = 人格/情绪（方向稳定 = feature，不是 bug）
- h = 工作记忆（spiral refinement）
- 赫布 = 人格强化（surprise 驱动）
- PC = 情绪反应（人格视角的预期违背）

详见 Skill §1.4.6 人格框架的数学基础。

---

## 二、当前困境

### 2.1 表面症状

所有长训练（step 3000-7000 范围）都会恶化：
- `L_est`（循环收缩率估计）从 ~0.6 升到 2+
- `hebb_norm` 单调递增或 `write` 爆炸
- `loss ema` 先降后反弹
- `h_diversity` 从 0.33 升到 >1.0（混沌而非螺旋收敛）
- 偶尔 NaN（但不总是）

### 2.2 尝试过的修复（全部失败）

| 修复 | 机制 | 失败原因 |
|------|------|----------|
| ct_inj soft clamp (v1) | sigmoid 限制注入比 | W_c 梯度被截断 → 萎缩 → ct_inj→0.001 |
| ct_inj hard clamp (v2) | 硬截断 | 同上 |
| ct 范数 hard clamp=20 | 限制 c_t 总量 | 撞墙后锁死 ct_perp=0 |
| W_c 移到 AdamW wd=0.1 | 用 AdamW 控制增长 | 初期好，后期撞 clamp 锁死 |
| W_c 行范数归一化 (Muon) | forward 时截断权重 | Muon 正交化每步推回，截断无效 |
| hebb_out 行范数归一化 | 同上 | 同上 |
| **hebb_proj_h/c + hebb_out 全链行范数归一化** | 三层输入输出全归一化 | **step 4000 L_est=2.4, step 7200 L_est=4.1 发散** |
| hebb 移到 AdamW wd=0.01 | 轻量 weight decay | **hebb 被压死（norm=0.02），但 L_est 仍涨到 3.31 — 证明发散和赫布无关** |
| c_t_head 行范数归一化 | 限制 introspection 输出 | 不够 |
| h_mask_predictor + loss=0.1 + surprise=0.3 | 给 c_t 直接梯度信号 | **可能是新发散源** |
| Span masking (mean=32/48) | World-JEPA 更难 | loss_w 仍快速归零 |
| World-JEPA mask 0.7→0.85→0.9 | 提高难度 | 同上 |

### 2.3 最关键的发现

**最近一次实验（hebb 在 AdamW wd=0.01 + 全链归一化）：**
- 赫布被 AdamW 压死：`hebb norm=0.026, write=0.008`
- 但 `L_est` 仍从 1.00 涨到 **3.31**
- `ct_perp` 从 0.07 降到 0.03（但未归零 — 自省流还活）
- `ct_inj` 从 0.033 撞到 **0.050 clamp**
- `ct 范数` 从 20 涨到 24

**这否定了之前的假设："赫布写入扰动不动点导致 L_est 升高"。**

真正的发散源可能是：
1. **h_mask_loss（权重 0.1）** 的梯度通过 `c_t_head` 反传到 introspection → 推动 c_t 方向快速变化
2. c_t 方向变 → 每步不动点 h*(c_t) 移动 → `L_est`（测残差比）虚高
3. ct 范数被 c_t_head 权重增长推大 → `ct_inj` 撞 clamp → 每步注入被截断 → c_t_head 梯度被放大 → 恶性循环

---

## 三、数学层面的深层问题

### 3.1 原罪：权重共享循环 + BPTT

Skill §1.4.8 已经指出：
- 20 轮循环 × 4 层 = 80 层 Jacobian 乘积
- 即使每层 J 谱半径 ≈ 1.05，backward 梯度 ≈ 1.05^80 ≈ 50
- bf16 + 80 层乘积 = 数值不稳定

**文献支撑（见 Skill §1.4.9）：**
- DEQ 论文：权重共享循环应用隐式微分而非 BPTT
- Mamba 论文：Mamba 是 Lyapunov 稳定的（forward 不会爆），不稳定来自循环外部
- RNN 梯度理论：forward 稳定 ≠ backward 稳定

### 3.2 赫布写入的梯度路径问题

当前赫布路径：
```
loss → c_t → hebb_gate × hebb_term → hebb_proj_h/c/out 的权重
                                    → surprise → _h_mask_err (detach)
                                    → prev_c_t (back through loops)
```

surprise 是 detach 的（明确切断），但 hebb_term 通过 `modulated_c_t += _hebb_write` 仍然进入 c_t，然后 c_t 在下一轮循环中作为输入——**这构成了一个时间依赖的循环梯度路径**。BPTT 展开后，hebb_out 的权重收到 80+ 层 Jacobian 乘积的梯度。

### 3.3 为什么 clamp 治标不治本

所有 clamp 都是 forward 时的"截断器"。反向传播时：
- clamp 激活 → 梯度通过 `x * min(r, 1)` 流过 → r<1 时梯度被缩放
- 被缩放的梯度仍然更新权重 → 权重继续在同方向增长 → 下一步 clamp 激活更深 → 梯度缩放更狠 → 更新变慢但不停
- 优化器（Muon/AdamW）的 momentum/variance 继续积累历史梯度 → 即使当前梯度小，权重仍会被推动

**Muon 的正交化更糟：** Newton-Schulz 正交化输出的幅度不依赖梯度大小。一个被 clamp 缩放到 0.01 的梯度，正交化后变成恒定幅度 η 的更新。**clamp 对 Muon 完全失效。**

**AdamW 的饱和也不够：** `m/√v` 在梯度稳定时饱和在 η·g/(g+ε) ≈ η。weight decay = η·λ·w 形成均衡 w*=1/λ。但 1/λ=10 对于 hebb_out 这种 zero-init 的权重来说是"永远爬不上来"——一开始梯度就小，AdamW 的 v 也小，步长反而大，但 zero-init 让权重从 0 开始，wd 每步减去的比例比梯度推的比例更大。

---

## 四、候选的结构性修复方向

### 方向 A：切断赫布的反向传播

让 hebb_term 完全从计算图中 detach，只保留前向传递。赫布成为"纯经验记忆"，不参与梯度训练：

```python
_hebb_write = hebb_gate * hebb_term.detach()  # 完全 stop-grad
```

赫布 proj 权重通过一个独立的辅助 loss 训练（比如直接 regress 到 δh 的重构），不参与主 loss 的反向传播。

**优点：** 从根上切断赫布对循环梯度的污染。
**风险：** 赫布失去任何训练信号，等效于"用手写规则定义赫布变换"——可能变成随机噪声。

### 方向 B：隐式微分（DEQ 风格）

参考 DEQ 论文，把循环展开的 BPTT 改成 `torch.autograd.Function` 里的隐式微分：
- Forward: 迭代求 h* = F(h*)
- Backward: 通过 `(I - J_F)^{-1} · ∂L/∂h*` 一步算出梯度

**优点：** 从根上解决 80 层 Jacobian 乘积的问题，backward 数值稳定。
**风险：** 工程量大；需要保证 J_F 可逆；Mamba SSM 的状态可能让 Jacobian 非平凡。

### 方向 C：减少循环深度，增加宽度

当前 reason_shared_depth=4, reason_loops=20 → 80 层展开。
改成 depth=8, loops=10 → 80 层但循环次数少一半，梯度路径更直接。
或 depth=12, loops=5 → 循环几乎消失，退化为深度网络。

**优点：** 缓解 BPTT 深度。
**风险：** 放弃"循环推理"的设计理念。

### 方向 D：让 c_t 的更新在梯度上停下

c_t 在 slow_update 轮被 introspection 重新生成。如果在生成时 detach：
```python
c_t_fresh = introspection(...)  # 新值
c_t = c_t_fresh.detach() + (c_t_fresh - c_t_fresh.detach())  # 保持值不变，但不参与 prev_c_t 的梯度
```
或者更激进：`c_t = c_t.detach()` 在每次 slow_update 之后，切断跨循环的 c_t 梯度链。

**优点：** c_t 仍被 introspection 训练，但不再是循环梯度的载体。
**风险：** introspection 失去"看到自己过去"的能力。

### 方向 E：把 h_mask_predictor 从 loss 里移除

h_mask_loss 可能是新的发散源。移除它，只保留 h_mask_err 作为 surprise 信号（不反向传播）。

**优点：** 回到上一个稳定状态。
**风险：** c_t 又变成只有间接梯度，回到"人格僵死"问题。

### 方向 F：重新设计 c_t 的角色

当前 c_t 既是人格（稳定）又是梯度载体（跨循环反向传播）。这两个角色冲突：
- 稳定的 c_t → 方向固定 → 梯度方向固定 → W_c 单方向累积 → 发散
- 变化的 c_t → ct_perp>0 → 梯度方向多样 → 但扰动不动点 → L_est 虚高

**结构性方案：** 引入两个 c_t 变量
- `c_t_persona`（稳定，不参与反向传播，用辅助 loss 训练）
- `c_t_working`（动态，参与循环和反向传播）
- 注入 h 时用两者的组合：`h += W_p · c_t_persona + W_w · c_t_working`

**优点：** 角色分离，各自优化。
**风险：** 参数量增加；设计复杂度上升。

---

## 五、需要 codex 帮忙决策的问题

1. **方向 A-F 中哪个最可行？** 或者是组合？
2. **隐式微分（方向 B）的实施复杂度是否值得？** Luma 的 reason_core 有 Mamba、DiffAttn、FFN、LoRA 混合，Jacobian 不好求。
3. **是否应该先回退到 "G0 baseline（无 h_mask_predictor + 无赫布扰动）" 然后先跑通 0.5 epoch，再迭代加功能？** 还是坚持现在的路径修 bug？
4. **Muon 的 rank-1 梯度不稳定性是否是根因？** Skill §1.4.8 引用了 Dao Lab 的文章说 Newton-Schulz 在近零奇异值下不稳定。如果是，切换到 AdamW 全量（不只是 hebb）是否合理？
5. **当前的诊断指标是否足够？** L_est 是 `dh[1]/dh[0]`，在不动点移动时会虚高。是否应该用 `fixed_point L`（SVD-based）作为主要指标？

---

## 六、已知的数据锚点

### G0 baseline（无 h_mask_predictor, 无增强, Muon W_c）
- 2000 步 loss_lm = 5.53
- h_diversity = 0.33
- ct_perp = 0.01-0.05
- L_est = 0.5-0.7
- **step 5800 W_c 增长 → ct_inj > α_crit → step 6585 NaN**（已证实，Eq.3 完美拟合）

### G0_jepa_enhanced v1（Muon W_c + hebb Muon）
- step 2000 loss ema = 8.61
- ct_perp = 0.05, ct_traj cos = 0.79（活！）
- hebb norm = 536, write = 137（暴力写入）
- Mamba L2_cos = 0.85（没冻结）
- **step 4000+ L_est 持续 >2.0，h_diversity 涨到 1.4, hebb norm→2496**

### G0_jepa_enhanced v2（全链行范数归一化）
- step 2500 loss ema = 8.29
- hebb norm = 784（压不住）
- **step 4900 L_est=2.4, hebb norm=1848**

### G0_jepa_enhanced v3（hebb AdamW wd=0.01）
- step 2000 hebb norm = 0.016（压死）
- **step 4350 L_est=2.44, ct_inj 撞 0.05 clamp**
- **证明：发散和赫布无关**

---

## 七、关键未解之谜

1. **为什么 v3 里 hebb 死了 L_est 还是涨到 3.31？** 谁在推动？候选：
   - h_mask_loss 的梯度推 c_t_head → c_t 范数涨 → ct_inj 撞 clamp
   - c_t_head 在 Muon 上无约束
   - W_c 虽然在 Muon 上但配合 ct 范数增长，ct_inj 最终还是上去了

2. **ct_inj 撞 clamp 之后 W_c 的梯度会怎样？** clamp 是硬截断，梯度通过缩放传回，但 Muon 的正交化可能绕过缩放。

3. **为什么 G0 baseline 能跑到 5800 步才崩，而 enhanced 版本 4000 步就开始恶化？** 差异：
   - enhanced 加了 h_mask_loss
   - enhanced 加了 hebb surprise boost
   - enhanced 用了更高的 World-JEPA mask
   - enhanced 的 c_t_head 有行范数归一化

---

## 八、行动建议（待决策）

**选项 1（保守）：** 回退到纯 G0（删除 h_mask_predictor，hebb 回 Muon 无 decay），先跑通 0.5 epoch 确认长训练基线。然后再加功能。

**选项 2（中度激进）：** 保留 h_mask_predictor 但删除它的 loss（只用作 surprise 源），hebb 回 Muon。这样 c_t 仍有间接刺激但没有直接梯度推动。

**选项 3（激进）：** 实现方向 B（隐式微分）或方向 F（双 c_t 分离）。投入 1-2 天重构。

**选项 4（诊断优先）：** 在 G0 上跑 10000 步并详细记录所有权重范数、梯度范数、Jacobian 谱半径的完整时间序列，找到真正的发散驱动因子。不改代码。

---

## 附录 A：完整的训练启动命令

```bash
cd /home/kt/ai/luma-architecture/minimind/scripts && \
bash run_experiment.sh <exp_name> \
  --iters 40222 --cosine_total_steps 80444 \
  [--world_mask_ratio 0.9] \
  [--h_mask_ratio 0.25] [--h_mask_surprise_weight 0.3]
```

脚本内部固化了 293M 架构参数和 Phase 6 配置。见 [`run_experiment.sh`](../../minimind/scripts/run_experiment.sh)。

## 附录 B：当前代码状态

所有已加的约束（截至本文档时间）：
- `ct_injection.proj`: Muon + 行范数归一化（get_bias 里 clamp ≤1.0）
- `c_t_head`: Muon + 行范数归一化
- `hebb_proj_h/c/out`: **AdamW wd=0.01** + 行范数归一化
- `ct_inj` clamp = 0.05（total norm 方式）
- `ct_output_rmsnorm` 在 NeuromodWriter 里
- Mamba span masking mean=48（多 span 几何分布）
- World-JEPA mask=0.9

决策文档结束。等待 codex 输入。
