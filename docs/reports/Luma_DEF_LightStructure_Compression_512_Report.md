# Luma D/E/F Light-Structure And Compression 512 Report

## 1. 这次实验在比较什么

第一组 `A/B/C` 没有把 `math` 修回来，所以这轮直接回到 `iter9 bundle` 底座，改试更有针对性的轻结构增强：

- `Exp D`: 在压缩区前后加一个轻量 `math adapter lane`
- `Exp E`: 让 `compressed summary -> self lane` 的融合加一个 `math-aware gate`
- `Exp F`: 把 `MHC` 放进 compression / fusion block

共同底座：

- `full + depth2 + self_check_k=2`
- `reason_loops = 15`
- `rollout_steps = 10`
- `world_mask_strategy = structured`
- `world_full_simplify_loss = true`
- `self_world_coupling_weight = 0.05`
- `self_rollout_hierarchical = true`
- `exit_two_step_aux_weight = 0.25`
- `fixture_mode = competition_math_dialogue_emotion`
- `enable_persona_seed = true`
- `stage2_steps = 512`

比较基线：

- `iter2 retained baseline`
- `iter9 bundle`

## 2. 核心结果表

| 配置 | mixed rollout_tail | math rollout_tail | dialogue rollout_tail | emotion rollout_tail | persona rollout_tail |
|---|---:|---:|---:|---:|---:|
| iter2 | 0.0410 | 0.0371 | 0.0410 | 0.0879 | 0.3613 |
| iter9 | 0.0391 | 0.0410 | 0.0371 | 0.0801 | 0.6133 |
| Exp D | 0.0313 | 0.0410 | 0.0410 | 0.0898 | 0.2988 |
| Exp E | 0.0996 | 0.0625 | 0.1035 | 0.0879 | 0.1680 |
| Exp F | 0.0664 | 0.0605 | 0.0742 | 0.0957 | 0.6230 |

补充 `self_tail`：

| 配置 | mixed self_tail | math self_tail | dialogue self_tail | emotion self_tail | persona self_tail |
|---|---:|---:|---:|---:|---:|
| iter2 | 0.0587 | 0.0563 | 0.0513 | 0.0704 | 0.2656 |
| iter9 | 0.0637 | 0.0632 | 0.0635 | 0.0969 | 0.4863 |
| Exp D | 0.0480 | 0.0516 | 0.0488 | 0.0645 | 0.3018 |
| Exp E | 0.1033 | 0.0984 | 0.1069 | 0.0706 | 0.1831 |
| Exp F | 0.0563 | 0.0555 | 0.0583 | 0.0820 | 0.4756 |

## 3. 结构诊断指标（mixed）

| 配置 | intermediate state variance | c_t drift | world summary drift |
|---|---:|---:|---:|
| Exp D | 0.00539 | 104.77 | 8.09 |
| Exp E | 0.00529 | 14.90 | 7.05 |
| Exp F | 0.00582 | 104.50 | 7.40 |

## 4. 逐项判断

### Exp D：当前最值得保留的轻结构增强

优点：

- `mixed rollout_tail` 直接降到 `0.03125`，是这轮最好的 mixed 结果
- `math rollout_tail` 没有继续恶化，维持在 `iter9` 同级 (`0.0410`)
- `dialogue` 没崩，回到 `iter2` 级别
- `persona_seed` 大幅回收：`0.6133 -> 0.2988`
- `emotion self_tail` 也变好

限制：

- `math rollout_tail` 仍然没有回到 `iter2` 的 `0.0371`
- 所以它更像“把 iter9 的副作用压下去”，而不是“彻底修好 math”

当前判断：

- `Exp D` 是这轮最像 `keep-candidate` 的方向
- 但如果 `iter2 math` 被视为硬底线，它还只能算“最优候选”，不能算“完全修复”

### Exp E：专项偏置明显，不适合默认主线

优点：

- `persona_seed` 很强：`0.1680`
- `emotion` 没掉

问题：

- `mixed` 直接坏掉：`0.0996`
- `dialogue` 明显变差
- `math` 也没有修好

当前判断：

- `Exp E` 不是默认主线候选
- 如果以后要做人格/自省表达专项，它可以保留为分支

### Exp F：compression-side MHC 目前不值得继续推

特点：

- mixed 比 `iter9` 差
- `math` 没修好
- `persona_seed` 又回到很差的区间

当前判断：

- 在当前阶段，compression 还不能确认是主要瓶颈
- 所以把 `MHC` 引进 compression / fusion block 这一步，暂时收益不够

## 5. 关于额外诊断的一点诚实说明

这轮我还记录了：

- `math_lane_score_mean`
- `math_summary_gate_mean`

但目前这两个量在报告里仍接近 `0.0`，没有像 mixed 指标那样明显亮起来。

这更像说明两种可能：

1. 诊断口径还不够好，没把结构真实激活程度量出来
2. 这轮收益主要来自“轻结构带来的优化路径变化”，而不是 gate 标量本身真的学成了强控制器

所以这一轮的主判断，仍应以任务指标为主，而不是先过度解读这两个 gate 诊断量。

## 6. 最终结论

如果只看这轮 `D/E/F`：

- `Exp D` 是明确的第一名
- `Exp E` 是偏专项、但不适合默认主线
- `Exp F` 目前不值得继续推进

一句话总结：

- 如果我们要在 `iter9 bundle` 基础上继续修 `math`，最值得继续的是 `Exp D` 这条线
- 下一步不该继续推 `E/F`，而应在 `D` 的基础上做更细的针对性修 math
