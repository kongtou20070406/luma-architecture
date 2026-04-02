# Luma Stage12 Consolidated Report (2026-04-02)

## 1) Scope
本报告合并了早期 Stage12 的同类散报告，统一保留“仍对当前主线有效”的结论边界，作为正式预训练前的结构基线说明。

已并入的报告簇包括：
- world/self 基础验证：`Luma_Stage12_Experiment_Report`、`Luma_SlowFastWorld_Experiment_Report`、`Luma_LeWorldModel_Comparison_Report`
- rollout / exit 相关：`Luma_Rollout_Depth_Experiment_Report`、`Luma_OneStep_vs_TwoStepAux_512_Report`
- depth / self-check / crystal：`Luma_Persona_Depth128_Report`、`Luma_Depth2_Check_Loops_Report`、`Luma_SelfCheck_Crystal_Report`
- uncertainty / r_t / 局部路由：`Luma_Uncertainty_*`、`Luma_rT_*`
- 补充实验：`Luma_32Step_Full_vs_Scaffold_Report`、`Luma_Muon_Width_Experiment_Report`、`Luma_ABC_Math_Repair_512_Report`、`Luma_DEF_LightStructure_Compression_512_Report`

## 2) Consolidated Stable Conclusions

### 2.1 当前结构主线（可继续前推）
- 基线语义：`A2-progress_shape_v1-h3+progress_exit_readout`
- 主干方向：`full world JEPA + self_check`
- 推理块：`reason_shared_depth=2` 作为当前 A2 系实验底座
- continuation 口径：`one-step main + light two-step auxiliary`

### 2.2 rollout / loops 的当前边界
- `10x15` 在中长程里是当前可用预算点。
- `10x20` 在多轮验证里未稳定形成额外收益，瓶颈更偏 exit/continuation policy，而非单纯 loops 上限。

### 2.3 self-check / uncertainty / crystal 的当前定位
- `self_check_k=2` 仍是当前默认稳点。
- `k=3`、`crystal`、`uncertainty` 保留为专项研究项，不进入默认主线。

### 2.4 world 分支的执行边界
- `scaffold` 在短程快验上更稳，仍可做低成本冒烟。
- `full` 更贴近正式预训练目标（世界态 latent 建模），当前作为正式主干优先候选。

## 3) Keep / Observe / Not Default

| 状态 | 分支 |
|---|---|
| Keep | `A2-progress_shape_v1-h3+progress_exit_readout` |
| Keep (probe path) | `summary_chunk_film_v2_progress+s_local_floor`（见 Matrix13/retest 边界） |
| Observe | `memory_tiered_routing_v1`（中程易塌缩，需 anti-collapse 约束） |
| Not default | `uncertainty/crystal` 直接接管 exit 或 two-step 权重 |
| Not default | 旧版 token-selective 强控（易导致 rollout 活性塌缩） |

## 4) Validity Boundary
- 本报告只保留对“当前仓库可复现实验链”仍有价值的结论。
- 已知被实现 bug、stale runtime、或 NaN 污染的结果，全部按“不可直接盖棺”处理。
- 具体边界以如下两份报告为准：
  - `Luma_Dynamics_Matrix13_ARCAGI_Report_20260330.md`
  - `Luma_Artifacts_Cleanup_20260402.md`

