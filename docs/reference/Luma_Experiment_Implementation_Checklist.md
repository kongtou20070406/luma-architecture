# Luma Experiment Implementation Checklist

这份清单专门回答一件事：

- 某个实验名字在代码里到底改了什么
- 它改的是哪条训练压力、哪条状态流、哪种退出信号
- 以后要复现实验时，最少需要哪些开关

这份文档只记录“目前仍然可考、仍然值得追踪”的实验分支，不追求把所有已经淘汰的临时想法都写进去。

## 1. 基线谱系

### `A0-core`

最早的纯 `one-step continuation` 骨架。

核心特征：
- `full + depth2 + self_check`
- `one-step continuation gain`
- 不启用 `light two-step auxiliary`

作用：
- 作为后续 `A1/A2` 的历史起点

### `A1-core`

`one-step main + light two-step auxiliary` 的升级线。

核心特征：
- `exit_two_step_aux_weight > 0`
- 两步只做轻量辅助，不接管 exit policy

作用：
- 证明“一步主、两步辅”比纯一步更好

### `A2-core`

当前正式长程基线。

核心特征：
- `world_jepa_mode = full`
- `reason_shared_depth = 2`
- `enable_self_check_ring = true`
- `self_check_k = 2`
- `exit_two_step_aux_weight = 0.25`
- `rollout_steps = 10`
- `reason_loops = 15`

明确不启用：
- crystal
- uncertainty feature
- math adapter lane
- math summary gate
- r_t reasoning ring

## 2. World JEPA 分支

### `A2-structured_world_bundle`（历史上常口语化叫 `iter9 bundle`）

在 `A2-core` 上加入更激进的 world/self 动力学强化。

实现项：
- `world_mask_strategy = structured`
- `world_full_simplify_loss = true`
- `self_world_coupling_weight > 0`
- `self_rollout_hierarchical = true`

作用：
- 让 world JEPA 更接近 `LeWorldModel` 风格
- 强化 structured mask 与简化 latent 监督

风险：
- 中长程里容易把 rollout 压平
- `math` 容易被拉伤

### `A2-structured_world_crystal`

在 `A2-structured_world_bundle` 上再加低火力 `JEPA crystal`。

实现项：
- `enable_exit_jepa_crystal = true`
- 让 exit controller 额外读 `jepa_crystal_signal`

作用：
- 试图用 JEPA 熵结晶度做排序信号

风险：
- 很容易进一步压平 rollout / exit 动力学

## 3. Self-JEPA / Loop-Awareness 分支

### `A2-predictor_progress`

让 `SelfJEPAResidualPredictor` 知道当前 loop progress。

实现项：
- `self_loop_awareness_mode = predictor_progress`
- predictor 在预测 `pred_delta_c` / rollout state 时读取：
  - `loop_progress`
  - `loop_index`

不做的事：
- 不把 phase 信息直接写进 `c_t`
- 不给 `c_t` 本体额外 phase embedding

作用：
- 在不污染慢环主状态的前提下，让 Self-JEPA predictor 更知道“自己现在在第几轮”

### `A2-progress_shape_v1`

当前最值得继续追的 Self-JEPA 强化候选。

实现项：
- 基于 `A2-predictor_progress`
- `self_progress_shape_weight = 0.10`
- `self_progress_trend_weight = 0.05`
- `self_progress_plateau_weight = 0.02`

它让 Self-JEPA 额外学习：
- `next improvement`
- `improvement trend`
- `plateau`

代码意义：
- 不是单纯多一个 loss 名字
- 而是在成熟的 slow-step 上，把 predictor 对推进节奏的表达能力单独监督出来

### `A2-local_smooth`

局部一致性最轻版本。

实现项：
- `self_local_delta_consistency_weight = 0.05`
- `self_local_curvature_weight = 0.0`

作用：
- 惩罚相邻 `pred_delta_c` 方向乱跳

### `A2-local_curvature`

局部一致性更重版本。

实现项：
- `self_local_delta_consistency_weight = 0.05`
- `self_local_curvature_weight = 0.02`

作用：
- 除了平滑，还约束短窗 trajectory curvature

## 4. Exit / Continuation 分支

### `one-step main + light two-step auxiliary`

这是当前 continuation 学习的正式主口径。

实现项：
- `one-step continuation gain` 作为主监督
- `exit_two_step_aux_weight = 0.25` 作为轻量辅助

作用：
- 一步做主、两步做辅
- 避免 two-step 直接接管 exit policy

### `JEPA crystal`（辅助信号）

实现项：
- 从 `self / rollout / world` 信号构造 entropy crystal
- 接进 exit feature 或 exit ranking

当前定位：
- 研究项
- 不作为默认基线一部分

### `introspection uncertainty`（辅助信号）

实现项：
- 自省流额外产出 uncertainty
- 试过几种接法：
  - two-step weighting
  - gate
  - exit feature

当前定位：
- 研究项
- 当前还没有稳定进入正式主线

## 5. Math / Routing / Compression 分支

### `ExpD-math-adapter`

这是迄今最值得保留的 math 修复线。

实现项：
- `enable_math_adapter_lane = true`

作用：
- 在压缩与后续自省/推理过渡处，给 math-like 信号一条轻量适配旁路

当前定位：
- 修复线
- 不是主线替代者

### `math summary gate`

实现项：
- `enable_math_summary_gate = true`

作用：
- 给 compressed summary -> self lane 的融合加 math-aware gate

当前定位：
- 研究项
- 目前不如 `ExpD`

### `compression MHC`

实现项：
- `enable_compression_mhc = true`

作用：
- 把 MHC 约束更早放入 compression/fusion 段

当前定位：
- 研究项
- 目前收益不足，不进入主线

## 6. r_t 局部递推分支

### `r_t parallel / predictor / blend`

实现项：
- `enable_reasoning_state_ring = true`
- `r_t_mode in {blend, parallel, predictor}`

作用：
- 在 `c_t` 旁边保一个轻量局部递推状态 `r_t`

当前判断：
- 方向成立，但收益还不够稳定
- 目前不扶正进主线

## 7. Rollout 监督分支

### `default rollout`

实现项：
- `self_rollout_steps = N`
- 默认远端权重衰减：
  - `<=2`: 1.0
  - `<=4`: 0.5
  - `>4`: 0.25

### `near3 weighting`

实现项：
- `self_rollout_weighting_mode = near3`

权重：
- `t+1 / horizon=2`: 1.0
- `t+2 / horizon=3`: 0.5
- `t+3 / horizon=4`: 0.2
- 更远：0.0

作用：
- 保留 rollout 结构
- 但把主监督拉回近端有信息区间

### `short horizon`

实现项：
- `self_rollout_supervision_horizon = 3 or 4`

作用：
- 直接缩短 rollout 监督参与的最远 horizon
- 用来检验 rollout 是否因为太远而失去分辨率

### `self-flow span mask`

实现项：
- `self_feature_span_mask_ratio > 0`

作用：
- 在 `SelfJEPAResidualPredictor` 的输入特征上施加轻度连续 span mask
- 不是 world mask
- 是让自流 predictor 对局部特征缺失更鲁棒

## 8. 诊断指标清单

### rollout 相关
- `self_rollout_tail`
- `rollout_active_ratio`
- `rollout_nonzero_ratio`

解释：
- `active_ratio` 高但 `nonzero_ratio` 低：通常说明 rollout 逻辑走到了，但监督缺少分辨率
- `tail` 很低不自动等于“真学会了 rollout”

### slow loop / self 检查相关
- `ct_kl`
- `hard_loop_var`
- `soft_loop_var`
- `c_t_var`
- `c_t_drift_mean`
- `self_check_mean`

### world / surprise 相关
- `world_surprise_mean`
- `world_summary_drift_mean`

### 研究性附加信号
- `jepa_crystal_mean`
- `uncertainty_mean`
- `math_lane_score_mean`
- `math_summary_gate_mean`
- `r_t_drift_mean`
- `r_t_switch_mean`

## 9. 当前推荐怎么用这份清单

以后每份实验报告都建议至少明确三件事：

1. 它基于哪条基线
   - 比如 `A2-core` / `A2-predictor_progress`
2. 它到底多加了什么
   - 用这份清单里的开关名直接写清楚
3. 它主比较的指标是什么
   - `self_tail` / `rollout_tail` / `rollout_nonzero_ratio` / 分桶平衡

一句话总结：

- 名字只是索引
- 这份清单负责把名字落回代码实现
