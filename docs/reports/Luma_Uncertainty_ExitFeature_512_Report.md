# Luma Uncertainty As Exit Feature Report

## 1. 这次实验在比较什么

这次实验专门验证一个更克制的问题：

- 不再让 `uncertainty` 直接调 `two-step auxiliary loss` 的权重
- 而是只把它作为 `exit feature` 输入给退出控制
- 看它能否保留不确定性信号，同时避免把 rollout supervision 压扁

共同底座：

- `full + depth2 + self_check`
- `self_check_k = 2`
- `one-step continuation gain` 主监督
- `light two-step continuation auxiliary`
- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `stage2_steps = 512`

对比组：

- baseline: `two_step_aux`
- `uncertainty feature = 0.05`
- `uncertainty feature = 0.10`
- `uncertainty feature = 0.05 + crystal feature = 0.05`

## 2. 结果

待填充。
