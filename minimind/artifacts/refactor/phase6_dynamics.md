# Luma Phase 6 Dynamics Report
*501 training steps*

## v2 逐层分析 (推荐)
| 指标 | 值 | 说明 |
|---|---|---|
| 逐层 DOD rank | **6** / 26 | 梯度独立方向数 / 总层数 |
| 第一模态能量 | 93.07% | <50% 优秀, <70% 可接受 |
| 死层 | ['exit_ctrl', 'mhc'] | 梯度 <1% 均值 |

**v2 判定**: RANK_LOW: 6/26 (23%) | mode1=93.1% 集中 | DEAD_LAYERS: ['exit_ctrl', 'mhc']

### Rank 轨迹: `[5, 5, 6, 6]`
### Mode1% 轨迹: `[93.3, 89.0, 93.1, 93.1]`

## c_t Batch 方差
| 指标 | 值 | 说明 |
|---|---|---|
| batch 方差均值 | 0.000000 | >0 = adaptive depth 有效 |
| 方差趋势 | 0.000000 | 正=增长(好), 负=萎缩(坏) |
| norm 散布 | 0.0000 | max-min 范数差 |

## v1 三维分析 (兼容)
判定: FAIL: dod_rank=3 < target 5, mode1=96.1%
### DOD Rank 轨迹: `[3, 3, 3, 3]`
### Mode1% 轨��: `[95.5, 92.3, 96.1, 96.1]`
