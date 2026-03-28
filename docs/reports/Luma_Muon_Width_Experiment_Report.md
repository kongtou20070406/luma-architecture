# Luma Muon + Width Experiment Report

## 1. 这次实验做了什么

这次实验分三步：

1. 把训练脚手架从单一 `AdamW` 升级为：
   - 外部包 `muon-optimizer` 提供的 `Muon`
   - 本地轻量 `MuonClip` 包装
   - 本地轻量 `Modular-Norm-style` 学习率缩放包装
2. 在这条新训练主线上，先跑：
   - `full + self_check`
   - `rollout=10`
   - `reason_loops=15`
   - `competition_math_dialogue_emotion`
   - `64-step / 128-step`
3. 再测试“扩宽共享推理 block”：
   - 只扩宽共享推理 block 内部 FFN
   - `reason_width_mult = 1.5`

注意：

- 这次 `Muon` 本体来自外部包 `muon-optimizer`
- `MuonClip` 与 `Modular Norm` 目前是本仓的轻量工程包装，不应假装成官方独立库

---

## 2. 相关文件

- 优化器包装：
  - [optimizers.py](/home/kt/ai/minimind/luma_stage0/optimizers.py)
- 验证脚本：
  - [run_luma_stage12.py](/home/kt/ai/minimind/scripts/run_luma_stage12.py)
- 模型配置变更：
  - [model_minimind.py](/home/kt/ai/minimind/model/model_minimind.py)

实验结果：

- `64-step base`:
  - [stage12_muon_full_selfcheck_10x15_64.json](/home/kt/ai/minimind/artifacts/stage12_muon_full_selfcheck_10x15_64.json)
- `64-step wide15`:
  - [stage12_muon_full_selfcheck_10x15_64_wide15.json](/home/kt/ai/minimind/artifacts/stage12_muon_full_selfcheck_10x15_64_wide15.json)
- `128-step base`:
  - [stage12_muon_full_selfcheck_10x15_128.json](/home/kt/ai/minimind/artifacts/stage12_muon_full_selfcheck_10x15_128.json)
- `128-step wide15`:
  - [stage12_muon_full_selfcheck_10x15_128_wide15.json](/home/kt/ai/minimind/artifacts/stage12_muon_full_selfcheck_10x15_128_wide15.json)

---

## 3. 先看优化器主线是否真的跑起来了

答案是：跑起来了，而且不是空转。

在 `full + self_check + 10x15` 上：

- `64-step base`
  - `self_loss_tail = 0.1577`
  - `self_rollout_tail = 0.3672`
- `128-step base`
  - `self_loss_tail = 0.0708`
  - `self_rollout_tail = 0.1172`

这说明：

- 新优化器主线在更长短程里明显能继续下降
- 不是“换了优化器但没有稳定学习”

---

## 4. 共享推理 block 扩宽的结果

### 4.1 64-step

| Variant | top-level self_tail | top-level rollout_tail |
|---|---:|---:|
| base | 0.1577 | 0.3672 |
| wide15 | 0.1553 | 0.2539 |

在 `64-step` 上：

- `wide15` 的整体 rollout 更好
- 说明更宽的共享推理 block 确实能加快动力学一致性学习

但分任务不能忽略：

| Task | base rollout_tail | wide15 rollout_tail |
|---|---:|---:|
| math | 0.3477 | 0.2441 |
| dialogue | 0.1523 | 0.4746 |
| emotion | 0.1016 | 0.1270 |
| mixed | 0.2637 | 0.2500 |

解读：

- `wide15` 明显提升了 `math`
- 但明显伤了 `dialogue`
- 说明扩宽共享推理 block 会把模型往“更强解题器”方向推，不是无条件提升聊天伙伴表现

### 4.2 128-step

| Variant | top-level self_tail | top-level rollout_tail |
|---|---:|---:|
| base | 0.0708 | 0.1172 |
| wide15 | 0.1875 | 0.1875 |

在 `128-step` 上：

- 默认宽度整体反超
- `wide15` 没有延续 `64-step` 的总体优势

这说明：

- `wide15` 更像短中程加速器
- 不是当前正式主干的稳定最优结构

不过如果只看 `per_task.mixed`（它在 `128-step` 报告里对应半长训练对照），`wide15` 仍有更强信号：

- `base mixed`: `self_tail = 0.1672`, `rollout_tail = 0.1699`
- `wide15 mixed`: `self_tail = 0.1128`, `rollout_tail = 0.0996`

这说明：

- 宽共享 block 并不是“完全没用”
- 它更像在较短或中等 horizon 上带来更强推进
- 但完整 `128-step` 顶层训练里，当前还没稳过默认宽度

---

## 5. 当前工程判断

### 5.1 优化器主线

当前可以确认：

- `Muon` 已经真正接入 stage2 验证主线
- 它不是纸面规划项了
- 但必须诚实说明：
  - `Muon` 本体来自外部包
  - `MuonClip` 和 `Modular Norm` 仍是本仓轻量工程包装

### 5.2 full + self_check

在新优化器主线下：

- `full + self_check` 是能稳定学下去的
- `64 -> 128 step` 明显继续收敛
- 这比之前单 AdamW 下更让人放心

### 5.3 宽共享推理 block

当前最稳的结论不是“要不要加”，而是：

- 它值得保留为专项开关
- 不宜立刻扶正为默认主干

原因：

- `64-step` 上它对 rollout 有帮助
- `128-step` 上默认宽度反而更稳
- 它对不同任务桶的影响不均衡，尤其会拉偏 `dialogue`

---

## 6. 一句话总结

- `Muon` 主线已经真正跑起来了
- `full + self_check` 在新优化器下能继续稳定收敛
- 扩宽共享推理 block 有真实价值，但当前更像“专项增压器”，还不是“默认主干升级”
