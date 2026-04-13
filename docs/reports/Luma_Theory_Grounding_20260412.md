# Luma 动力学理论落地 — 从"猜测"到"可运行探针"

**日期：** 2026-04-12
**目的：** Skill 文档中的所有公式和断言都必须由一个**可运行的 Python 探针**验证。不允许"我觉得"、"基于直觉"、"大概是"。

**参考文档：**
- 动力学 Skill: [`/home/kt/ai/Luma_Dynamics_Analysis_Skill.md`](../../Luma_Dynamics_Analysis_Skill.md)
- 危机决策文档: [`Luma_Crisis_20260412.md`](Luma_Crisis_20260412.md)
- 模型代码: [`minimind/model/model_minimind.py`](../../minimind/model/model_minimind.py)

---

## 零、核心原则

**"可证伪性"是动力学理论的底线。** Skill 文档目前有 5 个核心方程和 7 张变量映射表，但绝大多数是**事后拟合**（用已有实验的 loss 反推参数），不是**前向预测**（先预测再验证）。

**本文档的任务：**
1. 把 Skill 文档里的每个主张分类为「已测量」「事后拟合」「纯推测」三档
2. 为「事后拟合」和「纯推测」类的主张指定**具体的 Python 探针**
3. 探针必须满足：
   - 输入是训练中的一个 step 的所有状态（h, c_t, 权重, 梯度）
   - 输出是一个或多个数值
   - 可以在训练循环里每 N 步调用一次，不影响训练
4. 把 Skill 文档的每个公式改写成 `probes/eq_N.py` 的形式，让它变成"这个数值在这个 step 是多少"而不是"这个公式大概成立"

---

## 零点五、代码对齐修正（2026-04-12 晚）

### 0.5.1 先承认系统是时变的

当前代码一旦开启任一 loop-index 相关模块，主循环就不再是自治映射 `h_{k+1}=F(h_k,c_t)`，而是：

```text
h_{k+1} = F_k(h_k, c_t_k)
```

原因不是一个，而是四类同时存在：

- `Loop LoRA`：每轮 `loop_idx` 取不同的 `A[k], B[k]`
- `loop_ffn_gate`：每轮门控不同
- `phase_embed`：每轮 phase embedding 不同
- `time_proj(t, dt)`：每轮显式注入不同时间特征

因此：

- `L_est = dh[1]/dh[0]` 只能看作**时变轨迹的局部步长变化率**
- `_analyze_fixed_point()` 只能看作**旧口径 proxy**
- 只有在 `loop_lora_delta_ratio_mean`、`phase/time 注入比` 都很小的时候，才允许把系统近似成“受小扰动的自治系统”

### 0.5.2 新优先级：先测“时变性”，再谈“不动点”

本轮代码同步后，探针优先级调整为：

1. `loop_lora_delta_ratio_mean`
   - 定义：`||Δ_lora|| / ||FFN_residual||`
   - 含义：Loop LoRA 对当前 loop 算子的相对扰动强度
2. `ct_inj_pre`
   - 定义：`||proj(c_t)|| / ||h||`
   - 含义：裁剪前的 c_t 注入强度
3. `alpha_true`
   - 定义：`||applied_ct_bias|| / ||h||`
   - 含义：真正进入 forward 的 additive 注入强度
4. `wc_sv_top1`, `wc_cond`
   - 含义：`W_c` 是否进入单奇异值主导和病态条件数区
5. `grad_lm_to_wc`, `grad_hmask_to_ct_head`, `grad_selfjepa_to_hebb`
   - 含义：关键参数到底主要被哪个 loss 驱动

只有当 `loop_lora_delta_ratio_mean` 长期很小，才继续把 `rho_h_frozen / rho_c_drift / eta_moving_fp` 当作主判据。

### 0.5.3 结构化日志字段（与代码同步）

本轮详细日志写入 `artifacts/dynamics/<run_name>.jsonl`，关键字段定义如下：

| 字段 | 定义 |
|------|------|
| `ct_inj_pre` | `||proj(c_t)|| / ||h||`，裁剪前注入比 |
| `alpha_true` | `||applied_ct_bias|| / ||h||`，真实前向注入比 |
| `ct_norm_raw` | introspection 直接输出的 `next_c_t` 范数 |
| `ct_norm_after_writer` | writer/RMSNorm/可选 clamp 后的 `c_t` 范数 |
| `meta_last_norm` | introspection layer2 `meta_last` 范数 |
| `c_t_head_out_norm` | `c_t_head` 投影输出范数 |
| `loop_lora_delta_ratio_mean` | 各 shared layer 的 `||Δ_lora|| / ||FFN_residual||` 平均 |
| `wc_sv_top1` | `W_c` 第一奇异值 |
| `wc_cond` | `W_c` 条件数估计 |
| `grad_lm_to_wc` | `lm_loss` 对 `W_c` 的梯度范数 |
| `grad_hmask_to_ct_head` | `h_mask_loss` 对 `c_t_head` 的梯度范数 |
| `grad_selfjepa_to_hebb` | `self_jepa_loss` 对 `hebb_out` 的梯度范数 |

注：`rho_h_frozen`、`rho_c_drift`、`eta_moving_fp` 在未冻结 `loop_idx` 的情况下不能乱记；如果该步未接线，日志中应为 `null`，而不是拿别的 proxy 冒充。

---

## 一、Skill 文档主张分类审查

### 1.1 已直接测量（high trust）

| 主张 | 来源 | 测量方式 | 日志字段 |
|------|------|----------|---------|
| L_est = dh[1]/dh[0] ∈ [0.5, 0.7] | §1.4.3 | 训练中算 | `L_est` |
| ct_inj = ‖c_bias‖/‖h‖ ∈ [0.02, 0.03] | §1.4.5 | 训练中算 | `ct_inj` |
| ct_perp ∈ [0.01, 0.05] | §1.4.4 | cross product | `ct_perp` |
| h_diversity ≈ 0.33 | §1.4.6 | std over loops | `h_diversity` |
| L_global（SVD-based） | §1.4.3 | `_analyze_fixed_point` | `fixed_point L` |
| Mamba L1_cos, L2_cos | §1.4.4 | 相邻 loop 方向余弦 | `mamba_diag` |
| hebb_norm, hebb_write | §2.5 | forward 时算 | `hebb: norm/write` |

**这些可信。但注意：`L_est` 只用到 loop 1→2（第一对），不代表真正的 Jacobian 谱半径。**

### 1.2 事后拟合（medium trust）

| 主张 | 来源 | 证据强度 | 问题 |
|------|------|---------|------|
| α_crit ≈ 0.045 | §2.5 Eq.3 | K1 一个实验 | 1 个数据点 |
| γ ≈ 200-250 | §2.5 Eq.3 | 从 α_crit 反推 | 圆循环论证 |
| κ ≈ 5×10⁻⁶/步 | §1.4.5 | G0 一次 | 对 enhanced 版本不适用 |
| t_crash 预测 5000 步 | §1.4.5 | G0 一次 | enhanced 4000 步就崩 |
| ρ_Li ≈ 0.88 per layer | §1.4.3 | 几何级数反推 | 没单独测每层 |
| 1/λ = 10 Muon 均衡 | §1.4.8 | 数学推导 | 没验证 W_c 实际到达 10 |
| τ ≈ 60 步 | §1.4.4 | no-LoRA 数据 | 对当前配置未验证 |

### 1.3 纯推测（low trust）

| 主张 | 来源 | 为什么是推测 |
|------|------|------------|
| Muon Newton-Schulz 不稳定 → NaN | §1.4.8 | 文献支撑，未在 Luma 直接验证 |
| backward 梯度爆炸是 NaN 根因 | §1.4.8 | 探针没触发过 |
| RMSNorm 的 Jacobian 放大梯度 | §1.4.8 | 没测过 |
| c_t 方向固定 → ∂L/∂W_c ∝ ĉ·gᵀ (rank-1) | §1.4.5 | 从未直接检查梯度矩阵的奇异值 |
| h_mask_loss 的梯度推 c_t_head（crisis 文档猜测） | crisis §2.3 | 需要测 c_t_head 的梯度范数时间序列 |

---

## 二、需要新增的探针

### 2.1 探针 P1 — 真实 Jacobian 谱半径

**目的：** Skill §1.4.3 说 J_F 的谱半径是 ρ_total ≈ 0.6，但训练里测的 L_est 是 dh[1]/dh[0]，这是"第一对 δh 的衰减率"，不是 Jacobian 的真正谱半径。

**实现：** 在 ReasonCore 的 forward 里，每 log_interval 步对当前 h 计算 F 的数值 Jacobian 的 top-k 特征值：

```python
# probes/jacobian.py
import torch

@torch.no_grad()
def measure_jacobian_spectrum(reason_core, h, c_t, n_probes: int = 8):
    """测量 F = (shared_layers ×4) 在当前 h 的雅可比谱半径。
    用随机向量 probe + power iteration 估 top-k 特征值。
    """
    device = h.device
    D = h.shape[-1]
    eigenvalues = []
    for _ in range(n_probes):
        v = torch.randn_like(h) * 0.01
        # 计算 F(h+v) - F(h) ≈ J_F · v
        h_plus = h + v
        # 跑一遍 shared_layers
        h_out = _run_shared(reason_core, h_plus, c_t)
        h_base = _run_shared(reason_core, h, c_t)
        Jv = h_out - h_base
        # Rayleigh quotient: v^T J v / v^T v
        rayleigh = (v * Jv).sum() / (v * v).sum().clamp(min=1e-8)
        eigenvalues.append(rayleigh.item())
    return {
        "jacobian_max_eig": max(eigenvalues),
        "jacobian_mean_eig": sum(eigenvalues) / len(eigenvalues),
        "jacobian_eig_samples": eigenvalues,
    }

def _run_shared(reason_core, h, c_t):
    h_cur = h
    for layer in reason_core.shared_layers:
        h_cur = layer(h_cur, c_t=c_t, use_gradient_checkpointing=False, loop_idx=0)
    return h_cur
```

**测量什么：**
- `jacobian_max_eig`: 真实的 J_F 谱半径估计
- 预期：G0 健康时 0.5-0.7；发散时 >1.0
- **如果这个值 <1.0 但 L_est >2.0**，则 L_est 是"不动点移动"造成的虚高（而非真正发散），我们一直在盯错指标

### 2.2 探针 P2 — W_c 行范数时间序列

**目的：** Skill §1.4.8 说 Muon 下 W_c 沿 ĉ 方向分量单调增长。需要验证。

**实现：**
```python
@torch.no_grad()
def measure_wc_growth(model):
    W = model.reason_core.ct_injection.proj.weight  # [768, 64]
    row_norms = W.norm(dim=1)  # [768]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return {
        "wc_fro_norm": W.norm().item(),
        "wc_row_max": row_norms.max().item(),
        "wc_row_mean": row_norms.mean().item(),
        "wc_sv_top3": S[:3].tolist(),  # top-3 奇异值
        "wc_sv_ratio": (S[0] / S[-1]).item() if len(S) > 1 else 1.0,  # 条件数
    }
```

**测量什么：**
- 如果 `wc_sv_top3[0]` 单调增长而其他不动 → rank-1 梯度主导（Skill §1.4.5 的断言）
- 如果 `wc_row_max` 很快达到行范数归一化上限 1.0 → clamp 活跃
- `wc_sv_ratio` 爆炸 → Muon 正交化不稳定的前兆

### 2.3 探针 P3 — 梯度流向分析

**目的：** 找到真正驱动 W_c / c_t_head / hebb 增长的梯度来源。

**实现：** 在 backward 后立即捕获各关键参数的梯度：
```python
@torch.no_grad()
def measure_gradient_sources(model, loss_dict):
    """loss_dict: {'lm': L_lm, 'h_mask': L_hm, 'jepa': L_j, ...}
    对每个 loss 单独 backward，测各 loss 对关键参数的梯度贡献"""
    key_params = {
        "wc": model.reason_core.ct_injection.proj.weight,
        "c_t_head": model.introspection_state_stream.c_t_head.weight,
        "hebb_out": model.neuromod_ct_writer.hebb_out.weight,
    }
    contributions = {}
    for loss_name, loss_val in loss_dict.items():
        grads = torch.autograd.grad(loss_val, list(key_params.values()), retain_graph=True, allow_unused=True)
        for (pname, _), g in zip(key_params.items(), grads):
            key = f"{loss_name}→{pname}"
            contributions[key] = g.norm().item() if g is not None else 0.0
    return contributions
```

**测量什么：**
- `h_mask→c_t_head` 的梯度范数 → 直接回答 "h_mask_loss 是否在快速推动 c_t_head"
- `lm→wc` vs `h_mask→wc` 对比 → 哪个 loss 是 W_c 的主要推手

### 2.4 探针 P4 — 不动点移动速度

**目的：** Skill §1.4.6 说 c_t 变化会让 h* 移动，但从未直接测过。L_est 虚高的诊断需要这个。

**实现：**
```python
@torch.no_grad()
def measure_fixed_point_drift(reason_core, h_prev, h_curr, c_t_prev, c_t_curr, n_iter: int = 10):
    """对两个连续的 (h, c_t) 状态，各自迭代 F 到不动点，测两个不动点之间的距离"""
    def iterate_to_fp(h0, c_t, n):
        h = h0
        for _ in range(n):
            c_bias = reason_core.ct_injection.get_bias(c_t).unsqueeze(1)
            h_with_ct = h + c_bias
            for layer in reason_core.shared_layers:
                h_with_ct = layer(h_with_ct, c_t=c_t, use_gradient_checkpointing=False, loop_idx=0)
            h = h_with_ct
        return h
    h_star_prev = iterate_to_fp(h_prev, c_t_prev, n_iter)
    h_star_curr = iterate_to_fp(h_curr, c_t_curr, n_iter)
    drift = (h_star_curr - h_star_prev).norm().item()
    baseline = (h_curr - h_prev).norm().item()
    return {
        "fp_drift": drift,
        "h_actual_change": baseline,
        "fp_to_h_ratio": drift / max(baseline, 1e-8),
    }
```

**测量什么：**
- 如果 `fp_drift` 小而 `h_actual_change` 大 → h 在收敛到稳定不动点（好）
- 如果 `fp_drift` 大 → 不动点本身在移动，L_est 的虚高是这个原因

### 2.5 探针 P5 — backward 梯度爆炸监控

**目的：** Skill §1.4.8 猜测 NaN 来自 backward 梯度爆炸（80 层 Jacobian 乘积）。需要直接测。

**实现：**
```python
def install_gradient_monitor(model):
    """给每个 shared_layer 注册 backward hook，记录梯度通过时的范数"""
    grad_norms = {}
    def make_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad_norms[name] = grad_output[0].norm().item()
        return hook
    for i, layer in enumerate(model.reason_core.shared_layers):
        layer.register_full_backward_hook(make_hook(f"shared_{i}"))
    return grad_norms  # 训练中每步读这个 dict
```

**测量什么：**
- `grad_norms["shared_0"]` vs `grad_norms["shared_3"]` → 如果前者 >> 后者，说明梯度在展开的循环里累积放大
- NaN 之前的 step 应该能看到这些值指数增长

### 2.6 探针 P6 — Muon momentum 条件数

**目的：** Skill §1.4.8 引用 Dao Lab 说 Muon 的 Newton-Schulz 在近零奇异值时不稳定。需要监控 momentum buffer 的条件数。

**实现：**
```python
@torch.no_grad()
def measure_muon_momentum_health(optimizer, param_names: list[str]):
    """读 Muon optimizer 的 momentum buffer，计算每个的奇异值分布"""
    results = {}
    for group in optimizer.matrix_optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.matrix_optimizer.state.get(p, {})
            if "momentum" not in state and "momentum_q" not in state:
                continue
            m = state.get("momentum", None)
            if m is None:
                # 8bit 反量化
                from luma_stage0.optimizers import _dequantize_symmetric_8bit
                m = _dequantize_symmetric_8bit(state["momentum_q"], state["momentum_scale"], p)
            if m.ndim < 2:
                continue
            S = torch.linalg.svdvals(m.float())
            name = next((n for n in param_names if getattr(model, n, None) is p), f"p@{id(p):x}")
            results[name] = {
                "sv_max": S[0].item(),
                "sv_min": S[-1].item(),
                "cond_number": (S[0] / S[-1].clamp(min=1e-8)).item(),
                "effective_rank": (S.sum() / S[0]).item() if S[0] > 0 else 0,
            }
    return results
```

**测量什么：**
- `cond_number` > 1000 → Newton-Schulz 可能数值不稳定
- `effective_rank` = 1 → rank-1 梯度主导，正交化的行为变得不可预测

### 2.7 探针 P7 — c_t 范数增长的分解

**目的：** 回答危机文档的未解之谜 #1 —— "v3 里 hebb 死了谁在推 ct 范数涨"

**实现：** 把 c_t 的更新公式展开成各个加项，分别测范数：

```python
@torch.no_grad()
def decompose_ct_update(prev_c_t, next_c_t, hebb_term, hebb_gate, rmsnorm_scale):
    """
    modulated_c_t = prev_c_t + gain*Δc + surprise*hebb_term
    c_t_out = RMSNorm(modulated_c_t)
    
    分解：
    - 来自 introspection 的贡献：gain*Δc
    - 来自赫布的贡献：surprise*hebb_term
    - RMSNorm 的作用（scale 可学习）
    """
    delta_c = next_c_t - prev_c_t
    gain_contribution = delta_c  # gain=1+surprise 已经在外面乘了
    hebb_contribution = hebb_gate * hebb_term
    return {
        "ct_in_norm": prev_c_t.norm().item(),
        "ct_intro_contrib": gain_contribution.norm().item(),
        "ct_hebb_contrib": hebb_contribution.norm().item(),
        "ct_out_norm": next_c_t.norm().item(),  # 实际下一步的 c_t
        "rmsnorm_scale_mean": rmsnorm_scale.mean().item() if rmsnorm_scale is not None else 1.0,
    }
```

**测量什么：**
- `ct_intro_contrib` vs `ct_hebb_contrib` 的比例 → 谁在主导 c_t 变化
- 如果 hebb 死了但 `ct_intro_contrib` 单调增长 → introspection 是元凶
- `rmsnorm_scale_mean` > 1 → ct_output_rmsnorm 的可学习 scale 在放大 c_t 范数

---

## 三、待回答的诊断问题 × 探针组合

### 问题 A：v3 里 hebb 死了 L_est 还涨到 3.31，谁在推？

**用探针：** P1 (真 Jacobian) + P4 (不动点移动) + P7 (ct 分解) + P3 (梯度流向)

**判定树：**
1. 如果 P1.jacobian_max_eig > 1.0 → 真发散，和 Skill §1.4.3 一致
2. 如果 P1.jacobian_max_eig < 1.0 但 P4.fp_drift 大 → L_est 虚高来自不动点移动
3. 如果 P4 确认了虚高，再看 P7 决定谁推动 c_t 变化
4. P3 回答"这个变化的梯度来自 lm / h_mask / jepa 哪个"

### 问题 B：Eq.3 的 γ 真的是 225 吗？

**用探针：** P1 + P2

**实验设计：**
1. 扫 `ct_inject_scale` ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
2. 每个值跑 200 步
3. 每 50 步测 P1.jacobian_max_eig 和 P2.wc_row_max
4. 拟合 ρ(α) = ρ₀ + γα² → 报告拟合优度 R²

**当前只有一个数据点（K1，α=0.05）。** 需要至少 4 个点才能拟合 2 参数曲线。

### 问题 C：Muon 真的是 NaN 的根因吗？

**用探针：** P5 + P6

**验证：**
1. 在 G0 baseline 跑到 step 5500（接近崩溃前）
2. P6 测 Muon momentum 的条件数 → 是否 >1000
3. P5 测 backward 梯度范数 → 是否在 step 5800 爆炸
4. 同步测 ct_inj 和 L_est → 谁先越界

**如果 P6 的条件数正常，但 P5 的梯度爆炸 → NaN 来自 forward 数值问题，不是 Muon。**

### 问题 D：RMSNorm 的可学习 scale 是否在绕过归一化？

**用探针：** P7 + 直接读 `ct_output_rmsnorm.scale`

**断言：** RMSNorm 应该把 c_t 范数归到 1 附近（乘以 scale），如果 scale 持续增长（>2），则归一化失效。

---

## 四、重构 Skill 文档的具体方案

把 Skill 文档的每个章节改写成「声明 + 探针 + 验证状态」格式：

### 示例：§2.5 Eq.1 重写

**原文（§2.5）：**
> #### Eq.1 — 循环收缩（残差衰减）
> ‖h^(k+1) - h*‖ = ρ · ‖h^(k) - h*‖
> k 轮后残差 = ρ^k × 初始残差。ρ 直接从日志的 L_est 测量。

**改写：**

#### Eq.1 — 循环收缩（残差衰减）

**断言：** F 是压缩映射，存在不动点 h*，满足 ‖h^(k+1) - h*‖ = ρ · ‖h^(k) - h*‖。

**探针（probes/eq1_contraction.py）：**
```python
def verify_eq1(loop_history, reason_core, c_t):
    """检查 dh 序列是否按 ρ^k 几何衰减"""
    dhs = [(loop_history[i+1] - loop_history[i]).norm().item() 
           for i in range(len(loop_history)-1)]
    if len(dhs) < 3:
        return {"verified": None, "reason": "not enough loops"}
    # 拟合 log(dh_k) = log(dh_0) + k·log(ρ)
    import numpy as np
    ks = np.arange(len(dhs))
    logs = np.log(np.array(dhs) + 1e-12)
    slope, intercept = np.polyfit(ks, logs, 1)
    rho_fit = np.exp(slope)
    # 残差平方和
    residuals = logs - (slope * ks + intercept)
    r_squared = 1 - (residuals.var() / logs.var())
    return {
        "verified": r_squared > 0.9,
        "rho_fit": rho_fit,
        "r_squared": r_squared,
        "dh_series": dhs,
    }
```

**验证条件：** R² > 0.9 说明几何衰减模型成立；否则 Eq.1 失效。

**已知现状：** 训练日志只采集 loop 1→2 的 L_est，没有 verify_eq1 所需的完整 dh 序列。

**下一步：** 把 `loop_history` 的完整 dh 序列采到 metrics JSON 里。

---

### 示例：§1.4.3 ρ_total = Π ρ_Li 重写

**原文：** "每层 ρ_Li ≈ 0.9 → 4 层后 ρ ≈ 0.65"

**改写：**

**断言：** ρ_total = Π_{i=1..4} ρ_Li，其中 ρ_Li 是第 i 层 SharedLayer 在不动点处的 Jacobian 谱半径。

**探针（probes/per_layer_rho.py）：**
```python
@torch.no_grad()
def measure_per_layer_rho(reason_core, h, c_t, n_probes: int = 4):
    """对每层 SharedLayer 分别测 Jacobian 谱半径"""
    results = []
    h_cur = h + reason_core.ct_injection.get_bias(c_t).unsqueeze(1)
    for i, layer in enumerate(reason_core.shared_layers):
        # 测这一层的 Jacobian 最大特征值
        eigs = []
        for _ in range(n_probes):
            v = torch.randn_like(h_cur) * 0.01
            h_base = layer(h_cur, c_t=c_t, loop_idx=0)
            h_perturbed = layer(h_cur + v, c_t=c_t, loop_idx=0)
            Jv = h_perturbed - h_base
            rayleigh = ((v * Jv).sum() / (v * v).sum().clamp(min=1e-8)).item()
            eigs.append(abs(rayleigh))
        results.append(max(eigs))
        h_cur = layer(h_cur, c_t=c_t, loop_idx=0)
    return {
        "per_layer_rho": results,
        "rho_total_predicted": 1.0 * results[0] * results[1] * results[2] * results[3],
    }
```

**验证条件：** `rho_total_predicted` 应该和训练中的 L_est 在同一数量级（0.5-0.7）。

**如果不一致：** ρ_total = Π 的假设错了，可能 ρ_total 由非乘法关系给出（比如 softmax-like 组合）。

---

## 五、立即可做的最小行动

### 第一步：加两个关键探针到训练循环

1. **P1 Jacobian 谱半径** — 每 200 步采样一次，日志输出 `jac_eig`
2. **P4 不动点移动速度** — 每 200 步采样一次，日志输出 `fp_drift`

这两个可以立即回答最关键的问题："L_est 的高值是真发散还是不动点移动？"

### 第二步：跑一轮 G0 baseline + 两个探针

不改任何架构，只加探针。跑 8000 步（过 G0 v1 的崩溃点 5800），观察：
- `jac_eig` 是否在 step 5800 越过 1.0
- `fp_drift` 是否同步爆炸

**如果 `jac_eig` 越过 1.0 → Skill §1.4.3 正确，是真发散，继续找 W_c 增长源。**

**如果 `jac_eig` 一直 <1.0 而 `fp_drift` 越过某值 → L_est 诊断有问题，我们一直在盯错指标，整个 Skill §1.4.5 需要重写。**

### 第三步：根据结果决定下一步

- 情况 1（真发散）：聚焦 W_c 的结构性修复（方向 B/D/F）
- 情况 2（虚高）：重新定义发散判据，可能当前所有 "crisis" 都是误判

---

## 六、需要删除 / 降级的 Skill 文档内容

以下章节的结论依赖单次实验或纯推理，应该加「**待验证**」标注或移到附录：

| 章节 | 问题 |
|------|------|
| §1.4.3 ρ_Li ≈ 0.9 | 没单独测每层 |
| §1.4.4 τ ≈ 60 | 单次 no-LoRA 实验 |
| §1.4.5 t_crash 预测 | 对 enhanced 版本失效 |
| §1.4.7.1 "使用示例" | 基于未验证的 §1.4.3/5 |
| §1.4.8 Muon 不稳定 | 文献，未 Luma 验证 |
| §1.4.9 DEQ/Mamba 文献联系 | 类比，未实测 |
| §2.5 Eq.2 ∂h*/∂c_t ≈ ρ/(1-ρ)·‖W_c‖ | 从未数值验证 |
| §2.5 Eq.3 α_crit = √((1-ρ₀)/γ) | 1 个数据点 |
| §2.5 Eq.4 rank(W_eff) 预测 β | M2 已证伪 |
| §2.5 Eq.5 ct_perp(t) 指数衰减 | 未做时间序列拟合 |

---

## 七、行动清单

- [ ] 实现 P1 (Jacobian)、P4 (FP drift) 两个探针，挂到 Backbone.forward 里
- [ ] 每 200 步采样，输出到日志
- [ ] 跑 G0 baseline 8000 步，观察 jac_eig 和 fp_drift 的时间序列
- [ ] 根据结果决定：
  - 真发散 → 读 Crisis 文档的方向 B/D/F，挑一个结构性修复实施
  - 虚高 → 重写 Skill §1.4.3-5 的发散判据，可能当前"崩溃"都不是崩溃
- [ ] 跑完后把剩下的探针 P2/P3/P5/P6/P7 逐个加上
- [ ] 每个探针对应 Skill 文档里的一条断言，通过或证伪
- [ ] 最终目标：Skill 文档的每个章节都有 `probes/XX.py` 支持，删除所有"直觉推导"

---

**结论：** 当前的 Skill 文档更像"事后编撰的神话"，不是可证伪的理论。要治本，先把"测量"和"推测"分开，然后让测量来约束推测。本文档定义的 7 个探针是第一步。
