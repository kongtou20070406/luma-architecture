# Luma 10240-Step Midcourse Plan

## 1. 目标

这轮长程验证的目的不是继续做局部 512-step 微调，而是回答：

- `iter2` 这条稳基线在中程是否仍然最可靠
- `iter9` 这条更接近 LeWorldModel 的 bundle 是否会在更长训练里兑现优势
- `iter9 + crystal` 是否属于“短程不稳、中程转正”的类型
- `ExpD` 作为 iter9 的 math 修复线，是否值得在有余量时补跑

## 2. 数据桶

统一从 mixed 主线派生 probe：

- `math`
- `dialogue`
- `emotion`
- `persona_seed`
- `python_code`
- `mixed`

## 3. 实验组

### 3.1 `iter2_10240`
- `full + depth2 + self_check_k=2`
- `one-step main + light two-step auxiliary`

### 3.2 `iter9_10240`
- `iter2` 底座
- `structured world mask`
- `simplified full world`
- `self/world coupling`
- `hierarchical rollout`

### 3.3 `iter9_crystal_10240`
- `iter9`
- `JEPA crystal`
- `self_check_k=3`

### 3.4 `expd_iter9_math_adapter_10240`（仅在总耗时允许时补跑）
- `iter9`
- `math adapter lane`

## 4. 时间预算

保守估计：

- 每组约 `2.5 ~ 3.0` 小时
- 主跑 3 组：约 `8 ~ 9.5` 小时
- 若主跑偏快，再补 `ExpD`

## 5. 运行脚本

- `/home/kt/ai/minimind/scripts/run_luma_midcourse_10240_batch.sh`

## 6. 主要比较指标

优先看：

- `self_rollout_tail`
- `self_loss_tail`
- `hard_loop_var`
- `world_surprise_mean`
- `intermediate_state_variance`
- `c_t_drift_mean`
- `world_summary_drift_mean`

## 7. 护栏

- 参数量 `<= 0.35B`
- `math` 不掉
- `dialogue` 不明显恶化
- `emotion` 不明显恶化
- `mixed` 不崩
- `persona_seed` 仅软记录
