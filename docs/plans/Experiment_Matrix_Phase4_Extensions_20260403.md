# Phase 4 Extensions 实验矩阵 (2026-04-03)

## 基线
- **Phase 4**：self-JEPA (0.1) + SIGreg delta (0.05) + self_check_ring (0.1) + compress_probe (0.2)
- loss_lm=1.25, DOD rank=3, mode1=65.4%, DMD radius=0.41, ratio≈4
- 数据集：pretrain_diag.jsonl (6445条闲聊, avg 42 chars)
- 配置：1500 steps, batch=4, seq=512, gradient_checkpointing=1

## 设计原则
1. 文献指引：优先近端动力学 > progress-shape > 几何约束 > 远端 horizon
2. 每个实验只改 1-2 个变量，保证因果归因清晰
3. 数据集实验与结构实验正交，析因组合
4. 判定标准：DOD rank≥3, mode1<80%, loss_lm≤Phase4

---

## A组：Rollout Loss 回归实验 (6个)

Self-JEPA predictor 的多步验证，当前 Phase 4 关着 (weight=0)。
文献依据：近端 rollout (horizon 2-3) 比远端更有效。

| ID | 改动 | 假设 |
|----|------|------|
| A1 | self_rollout_weight=0.1 | rollout 基本验证，给 reasoning 区额外梯度 |
| A2 | self_rollout_weight=0.1, self_rollout_weighting_mode=near3 | 近端加权 (h2=1.0, h3=0.5, h4=0.2)，文献推荐 |
| A3 | self_rollout_weight=0.2, self_rollout_weighting_mode=near3 | A2 加强版，更大权重 |
| A4 | self_rollout_weight=0.1, rollout_zone_weight=0.01 | rollout + 活跃度守卫，防 rollout 退化 |
| A5 | self_rollout_weight=0.1, trajectory_vitality_weight=0.01 | rollout + 轨迹防冻，防 c_t 停滞 |
| A6 | self_rollout_weight=0.1, self_rollout_weighting_mode=near3, rollout_zone_weight=0.01, trajectory_vitality_weight=0.01 | 全套组合 |

## B组：Progress-Shape 与退出决策 (5个)

文献依据：progress-shape (next/trend/plateau) 是比远端 rollout 更稳的信号源。
"Emergent Search and Backtracking" 论文指出平台期/回退是健康推理的一部分。

| ID | 改动 | 假设 |
|----|------|------|
| B1 | self_progress_shape_weight=0.05 | progress-shape 辅助 loss，让 predictor 学习推进形状 |
| B2 | self_progress_shape_weight=0.05, enable_progress_exit_readout=1 | B1 + 把 progress 信号接入退出决策 |
| B3 | self_progress_shape_weight=0.05, enable_progress_exit_readout=1, enable_backtrack_aware_progress=1 | B2 + 回退感知，允许模型识别无效回路 |
| B4 | self_progress_shape_weight=0.05, self_rollout_weight=0.1, self_rollout_weighting_mode=near3 | progress-shape + 近端 rollout 组合 |
| B5 | self_progress_shape_weight=0.05, enable_progress_exit_readout=1, self_rollout_weight=0.1, self_rollout_weighting_mode=near3 | B2 + A2 全组合 |

## C组：状态几何与正则 (4个)

文献依据：Geometrically-Regularized World Models 论文表明改善 representation geometry 比加复杂 dynamics 更有效。

| ID | 改动 | 假设 |
|----|------|------|
| C1 | self_local_delta_consistency_weight=0.01 | c_t 增量方向的局部一致性（相邻步方向不应剧烈翻转）|
| C2 | self_local_curvature_weight=0.005 | c_t 轨迹曲率正则（二阶差分惩罚，平滑轨迹）|
| C3 | enable_sigreg_ct=1, sigreg_ct_weight=0.03 | c_t 本身加 SIGreg，防认知状态空间坍缩 |
| C4 | self_local_delta_consistency_weight=0.01, self_local_curvature_weight=0.005, enable_sigreg_ct=1, sigreg_ct_weight=0.03 | 全套几何正则组合 |

## D组：数据集多样性实验 (5个)

当前数据只有闲聊（avg 42 chars），这是所有辅助任务"太简单"的根本原因之一。
混入 hard math / emotion / persona_seed 数据，让模型面对真正需要推理的内容。

需要先准备数据集（见下方数据准备节）。

| ID | 数据集 | 结构 | 假设 |
|----|--------|------|------|
| D1 | diag + hard_math (50:50 混合) | Phase 4 原始 | math 需要多步推理，应更好利用 reasoning loops |
| D2 | diag + emotion + persona_seed (40:30:30) | Phase 4 原始 | 多样语境，测试 compress/reasoning 分工 |
| D3 | diag + hard_math + emotion + persona_seed (25:25:25:25) | Phase 4 原始 | 全混合基线 |
| D4 | 全混合 (同D3) | + self_rollout_weight=0.1, near3 | 数据多样性 × 近端 rollout |
| D5 | 全混合 (同D3) | + B2 (progress-shape + exit readout) | 数据多样性 × progress 信号 |

## E组：最优组合候选 (4个)

从 A-D 组的正交实验中，选出各组最佳组合。
如果 A-D 都跑完了则按结果选，否则按先验设计：

| ID | 配方 | 假设 |
|----|------|------|
| E1 | A2 最佳 rollout + B2 progress | 近端动力学 + 退出决策双管齐下 |
| E2 | E1 + C1 局部一致性 | 三层叠加（动力学+退出+几何）|
| E3 | E1 + D3 全混合数据 | 结构最佳 + 数据最佳 |
| E4 | A2 + B2 + C1 + D3 | 全组合——验证各维度是否叠加正向 |

---

## 数据准备

### 需要的数据集
1. **hard_math**：数学题（GSM8K/MATH 的中文翻译子集，或自行生成的数学推理数据）
2. **emotion**：情感表达/分析数据（需要理解隐含情绪）
3. **persona_seed**：角色设定+对话（需要保持角色一致性）

### 格式
与现有 pretrain_diag.jsonl 相同：`{"text": "..."}`，每行一个样本。

### 混合方式
按比例随机交错写入一个新的 jsonl 文件，保持总量 ~6000-8000 条。

---

## 执行计划

### 串行顺序（总计约 24 × 12min ≈ 5 小时）
1. A1-A6（72 min）
2. B1-B5（60 min）
3. C1-C4（48 min）
4. D1-D5（60 min）— 需要先准备数据
5. E1-E4（48 min）— 根据前面结果调整配方

### 自动化
写一个 runner 脚本，串行跑所有实验，每个实验：
1. 运行 1500 步训练
2. 提取 DOD/DMD 快照
3. 记录最终 loss_lm, DOD rank, mode1_energy, DMD radius, ratio
4. 输出到统一的 results.jsonl

### 判定标准
- ✅ PASS：rank≥3 且 mode1<80% 且 loss_lm ≤ 1.30
- ⚠️ WARN：rank≥3 但 mode1 80-90% 或 loss 1.30-1.40
- ❌ FAIL：rank<3 或 mode1>90% 或 loss>1.40
