# Reference Index (Current)

## 1) 这个目录现在放什么
- 当前训练实现需要反复引用、且不适合写进单次实验报告的“稳定参考项”。
- 包括：
  - loss 定义与作用边界
  - 实验实现清单（开关语义、候选映射）

## 2) Canonical References
1. `Luma_Loss_Reference.md`
2. `Luma_Experiment_Implementation_Checklist.md`

## 3) 路径与执行口径
- 当前唯一主项目路径：
  - `/home/kt/ai/luma-architecture/minimind`
- `minimind_runtime_dynamics` 已移除，不再作为有效执行入口。
- `parameter-golf` 仅保留为可选参考材料，默认实验链不依赖它。

## 4) 和 reports / plans 的关系
- `plans/` 负责“该做什么、按什么 gate 推进”。
- `reports/` 负责“已经发生了什么、哪些结论有效”。
- `reference/` 负责“长期稳定可复用的定义和约束”。

