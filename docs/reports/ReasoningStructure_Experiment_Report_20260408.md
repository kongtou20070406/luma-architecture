# RS (Reasoning Structure) 实验报告

**日期**: 2026-04-08
**基线**: CR5 + SJ1 + E9 + Time Conditioning (DP2 最优)
**数据**: openr1_math_hard_2k.jsonl, 350 steps

## 结果汇总

| 实验 | 方案 | loss_lm | vs RS0 | avg_loops | max | 参数量 |
|------|------|---------|--------|-----------|-----|--------|
| **RS0** | 对照 (min=2, time_cond) | 10.6353 | — | 2.3 | 4 | 286.3M |
| RS1 | min_loops=5 | 12.4690 | +17.2% | 5.4 | 8 | 286.3M |
| RS2 | min_loops=8 | 12.1442 | +14.2% | 8.3 | 11 | 286.3M |
| RS3 | min_loops=5, no time | 12.1717 | +14.4% | 5.3 | 7 | 286.3M |
| RS4 | LoRA rank=16 | 13.1277 | +23.4% | 2.5 | 5 | 288.2M |
| **RS5** | **LoRA rank=32** | **8.5127** | **-20.0%** | 2.3 | 4 | 290.2M |
| RS6 | LoRA rank=16 + min=5 | 13.0788 | +23.0% | 5.3 | 7 | 288.2M |
| RS7 | Loop FFN Gate | 12.2140 | +14.8% | 2.4 | 4 | 286.3M |
| RS8 | FFN Gate + min=5 | 13.0312 | +22.5% | 5.2 | 6 | 286.3M |

## 关键发现

### 1. LoRA rank=32 碾压全场 (RS5, -20.0%)

Loop LoRA rank=32 把 loss 从 10.64 打到 8.51，**改善幅度达 20%**。这比 DP 矩阵中最好的 time conditioning (-8.7%) 还大一倍多。

机制：每个 loop step 有独立的 LoRA A/B 矩阵作用在 FFN 上，让 shared_layers 在不同循环步做不同的计算。这相当于用 +3.9M 参数（290.2M vs 286.3M, +1.4%）换来了 20% 的 loss 改善。

### 2. LoRA rank=16 完全失败 (RS4, +23.4%)

rank=16 比 baseline 差了 23%，而 rank=32 好了 20%。这不是线性关系 — rank=16 可能低于 FFN 变化所需的最小秩，导致 LoRA 退化为噪声。

### 3. 强制深循环全面失败 (RS1/RS2/RS3, +14~17%)

min_loops=5/8 都显著恶化 loss。这证实了循环坍缩的根因不是 ExitController 太保守，而是 **shared_layers 在更多循环中没有学到有用的东西**。

RS2 (min=8) 比 RS1 (min=5) 稍好，可能是因为更多循环提供了更多梯度信号。但都远不如 baseline。

### 4. FFN Gate 没有明显效果 (RS7, +14.8%)

Loop FFN Gating 不如 LoRA — 可能是因为 gate 只能缩放 FFN 输出，而 LoRA 可以完全改变 FFN 的映射。gate 的表达力不够。

### 5. Time conditioning 效果无法叠加 min_loops (RS1 vs RS3)

RS1 (min=5 + time) = 12.47，RS3 (min=5 no time) = 12.17。去掉 time conditioning 反而更好？这可能是因为 time conditioning 的效果在强制深循环时成为噪声。

## 推荐

1. **LoRA rank=32 纳入默认配置** — +1.4% 参数换 -20% loss 是极好的交易
2. **放弃 min_loops 方向** — 强制深循环无效
3. **放弃 FFN Gate** — 表达力不够
4. **尝试 LoRA rank=48/64** — 看是否还有继续增益的空间
5. **IS 实验基于 RS5 (LoRA32 + Time) 做基线** — 当前最优配置
