# Luma Loss Reference

> 2026-04-02 代码对齐版。  
> 本文以 `minimind/model/model_minimind.py` 的当前实现为准，不再沿用旧的简化损失口径。

## 1. 当前总损失（代码真实口径）

`LumaForCausalLM.forward()` 里的训练总损失是：

```text
L_total
= L_lm
+ L_world
+ L_self
+ w_rollout * L_self_rollout
+ w_exit * L_exit_aux
+ L_rollout_zone
+ L_routing_entropy
+ L_trajectory_vitality
```

对应代码位置：
- 总和组装：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L3515)

说明：
- `L_rollout_zone`、`L_routing_entropy`、`L_trajectory_vitality` 是“抗塌缩/动力学健康”正则，不是日志装饰项。
- `local_alive_floor_loss` 与 `rollout_alive_floor_loss` 当前仅记录在 aux 中，默认未并入 `L_total`。

---

## 2. `L_lm`（主语言建模损失）

- 形式：标准 next-token cross-entropy
- 代码位置：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L3504)
- 作用：保持可生成能力与 token 级预测主目标

---

## 3. `L_world`（World JEPA）

`L_world` 来自 `aux["world_jepa_loss"]`，按 world 模式分为两类。

### 3.1 `scaffold` 模式

核心是 masked latent cosine 对齐：

```text
L_world_scaffold = 1 - cos(pred_masked_world, target_masked_world)
```

### 3.2 `full` / LeWorldModel-style 模式

当前实现：

```text
L_world_full
= L_cosine
+ w_world_sigreg * L_world_sigreg        (enable_sigreg_world=true 时生效)
+ w_delta * L_world_delta                 (world_full_simplify_loss=false 时生效)
```

对应代码位置：
- world full 分支计算：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L1608)

开关与权重：
- `enable_sigreg_world`
- `world_sigreg_weight`
- `world_full_simplify_loss`（为 `true` 时跳过 delta 子项）

---

## 4. `L_self`（Self JEPA 主损失）

`L_self` 对应 `aux["self_jepa_loss"]`，不是单一项，而是组合项。

基础骨架：

```text
L_self_base
= mean(L_self_main)
+ mean(L_residual_reg)
+ w_coupling * L_self_world_coupling      (可选)
+ L_progress_shape                         (可选)
+ L_local_consistency                      (可选)
+ L_trajectory_health                      (可选)
```

在开启 SIGReg 子路后会继续叠加：

```text
L_self
= L_self_base
+ w_sigreg_delta   * L_sigreg_delta       (enable_sigreg_delta=true)
+ w_sigreg_rollout * L_sigreg_rollout     (enable_sigreg_rollout=true)
```

对应代码位置：
- self 总组装：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L3143)
- `sigreg_delta_loss` / `sigreg_rollout_loss` 注入：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L3150)

说明：
- `L_self_main` 的核心语义仍是 `Δc_t` 预测（残差式自省动力学）。
- `L_progress_shape`、`L_local_consistency`、`L_trajectory_health` 都是“形状/稳定性约束”，并入 `L_self` 主体。

---

## 5. `L_self_rollout`（多步动力学监督）

- 对应项：`aux["self_rollout_loss"]`
- 在总损失中的权重：`self_rollout_weight`
- 代码位置：
  - rollout 形成与对齐：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L2863)
  - 加入总损失：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L3520)

说明：
- 当前主线是“一步主监督 + 两步轻辅助的 continuation 学习口径”，并允许更高 horizon 作为动态筛选信号。

---

## 6. `L_exit_aux`（退出策略辅助损失）

`L_exit_aux` 不是 BCE 分类头，而是 continuation gain 回归（SmoothL1）为主。

当前实现：

```text
L_exit_aux
= SmoothL1(predicted_gain_1step, continuation_gain_1step)
+ w_two_step * SmoothL1(predicted_gain_2step, continuation_gain_2step)  (当开启 two-step aux)
```

对应代码位置：
- 退出辅助损失计算：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L3325)

权重与增强项：
- `exit_aux_weight`（总入口权重）
- `exit_two_step_aux_weight`
- 可选的 uncertainty / crystal 对 two-step 权重调制：
  - `exit_uncertainty_two_step_weight`
  - `exit_crystal_two_step_weight`

---

## 7. 三个动力学健康正则（直接进总损失）

### 7.1 `L_rollout_zone`

目的：把 rollout 活性维持在健康区间，防止“过早归零”或“异常爆活”。

结构：

```text
L_rollout_zone = w_zone * zone_penalty(
  rollout_nonzero_ratio,
  rollout_active_ratio,
  future_delta_var
)
```

代码位置：
- 计算：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L3405)

### 7.2 `L_routing_entropy`

目的：防止 tier/chunk/block 路由塌缩到单一路径。

结构：

```text
L_routing_entropy
= w_entropy * entropy_floor_violation
+ w_local_floor * local_share_violation
```

代码位置：
- 计算：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L3433)

### 7.3 `L_trajectory_vitality`

目的：防止 `c_t` 与 world 漂移长期归零，避免“假稳定”。

代码位置：
- 计算：[model_minimind.py](/home/kt/ai/luma-architecture/minimind/model/model_minimind.py#L3459)

---

## 8. 当前 SIGReg 三干预点（与矩阵对齐）

当前 8 组合矩阵对应三个独立开关：

1. `enable_sigreg_world`
   - 作用对象：`world_online`
   - 注入位置：`L_world`
2. `enable_sigreg_rollout`
   - 作用对象：`rollout_state_preds[:3]`
   - 注入位置：`L_self`
3. `enable_sigreg_delta`
   - 作用对象：`pred_delta_c`
   - 注入位置：`L_self`

相关参数入口（stage12 runner）：
- [run_luma_stage12.py](/home/kt/ai/luma-architecture/minimind/scripts/run_luma_stage12.py#L1218)

---

## 9. 当前读取建议（避免误解）

- 先看 `loss`（包含 `L_lm`）判断整体训练目标是否健康。
- 再看 `aux_loss` 与各子项，定位是哪条动力学分支在拉偏。
- `rollout_tail` 单独变低不等于变好，必须结合：
  - `rollout_nonzero_ratio`
  - `future_delta_var`
  - 分桶指标（`math/python_code/mixed/dialogue/emotion/persona_seed/arc_agi`）

---

## 10. 一句话结论

当前 Luma 不是“LM + 两个 JEPA 附件”的简式训练，而是“主 LM + world/self 动力学 + 退出收益建模 + 三个健康正则”的组合训练系统。  
因此任何 keep/kill 结论都必须基于完整损失分解和分桶健康指标，而不是只看单个 tail 分数。
