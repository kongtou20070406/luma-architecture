# Luma Phase 0 Dynamics Report
*4 training steps*

## v2 逐层分析 (推荐)
| 指标 | 值 | 说明 |
|---|---|---|
| 逐层 DOD rank | **1** / 32 | 梯度独立方向数 / 总层数 |
| 第一模态能量 | 99.97% | <50% 优秀, <70% 可接受 |
| 死层 | ['exit_ctrl', 'mhc', 'self_jepa', 'world_jepa'] | 梯度 <1% 均值 |

**v2 判定**: RANK_LOW: 1/32 (3%) | mode1=100.0% 集中 | DEAD_LAYERS: ['exit_ctrl', 'mhc', 'self_jepa', 'world_jepa']

### Rank 轨迹: `[1, 1]`
### Mode1% 轨迹: `[100.0, 100.0]`

## c_t Batch 方差
| 指标 | 值 | 说明 |
|---|---|---|
| batch 方差均值 | 0.000000 | >0 = adaptive depth 有效 |
| 方差趋势 | 0.000000 | 正=增长(好), 负=萎缩(坏) |
| norm 散布 | 0.0000 | max-min 范数差 |

## v1 三维分析 (兼容)
判定: WARN: dod_rank=1 ok但能量仍集中 mode1=100.0% (目标<80%)
### DOD Rank 轨迹: `[1, 1]`
### Mode1% 轨��: `[100.0, 100.0]`
