# Luma Phase 4 Extensions 实验矩阵完整报告

**日期**: 2026-04-04
**模型**: Luma 312M (LumaBackbone)
**训练**: 1500 steps, bs=4, gradient_checkpointing, max_seq_len=512

---

## 1. 实验概览

共 35 个实验，分 7 组（A-G），覆盖三个维度：
- **A-C 组**: 自监督模块消融（rollout, progress-shape, geometry regularization）
- **D-F 组**: 模块组合优化
- **G 组**: 新数据源验证

### 核心指标说明
| 指标 | 含义 | 健康范围 |
|------|------|---------|
| loss_lm | 语言模型损失 | 单类型数据 <1.30, 多类型 <5.0 |
| DOD rank | 梯度方向维度 | =3 (最大) |
| mode1_energy | 第一主成分占比 | <80% (越低越健康) |
| ratio | compress_grad / reasoning_grad | 4-25 (太高=reasoning被压制) |

---

## 2. 全量实验结果

### A 组：Rollout Loss 变体
| ID | Config | loss | rank | mode1 | ratio | 判定 |
|----|--------|------|------|-------|-------|------|
| A1 | rollout basic (w=0.1) | 1.21 | 3 | 54.4% | 21.4 | PASS |
| A2 | rollout near3 | 1.17 | **1** | 100% | 26.4 | FAIL |
| A3 | rollout near3 strong (w=0.2) | 1.19 | 3 | 86.5% | 31.8 | WARN |
| A4 | rollout + zone guard | 1.21 | 3 | 54.4% | 21.4 | PASS |
| A5 | rollout + vitality | 1.21 | 3 | 54.4% | 21.4 | PASS |
| A6 | rollout full suite | 1.17 | **1** | 100% | 26.4 | FAIL |

**结论**: near3 加权模式有毒，直接导致 rank 坍缩。legacy rollout (w=0.1) 安全且有效。zone_guard 和 vitality 对指标无显著影响。

### B 组：Progress-Shape 与退出决策
| ID | Config | loss | rank | mode1 | ratio | 判定 |
|----|--------|------|------|-------|-------|------|
| **B1** | **progress-shape (w=0.05)** | **1.07** | **3** | **59.1%** | **17.5** | **PASS** |
| B2 | + exit readout | 1.30 | 3 | 75.3% | 22.2 | WARN |
| B3 | + exit + backtrack | 1.29 | 3 | 67.2% | 30.6 | PASS |
| B4 | + rollout near3 | 1.28 | 3 | 55.3% | 20.3 | PASS |
| B5 | + exit + rollout near3 | 1.22 | 3 | 80.5% | 25.3 | WARN |

**结论**: B1 (progress-shape=0.05) 是单模块最佳，loss 全场最低 (1.07)。exit readout 和 backtrack 增加复杂度但不提升核心指标。

### C 组：状态几何与正则
| ID | Config | loss | rank | mode1 | ratio | 判定 |
|----|--------|------|------|-------|-------|------|
| **C1** | **local delta consistency (w=0.01)** | **1.24** | **3** | **52.8%** | **25.3** | **PASS** |
| C2 | curvature reg (w=0.005) | 1.28 | 3 | 62.9% | 26.0 | PASS |
| C3 | sigreg on c_t (w=0.03) | 1.20 | 3 | 65.3% | 19.6 | PASS |
| C4 | full geometry suite | 1.21 | 3 | 85.6% | 23.9 | WARN |

**结论**: C1 mode1=52.8% 是 A-C 单模块中最佳几何健康度。C4 全套叠加反而 mode1 升高至 85.6%，模块间有干扰。

### D 组：旧合成数据集多样性
| ID | Config | loss | rank | mode1 | ratio | 判定 |
|----|--------|------|------|-------|-------|------|
| D1 | diag + math 50:50 | 0.52 | **1** | 100% | 23.4 | FAIL |
| D2 | diag + emo + persona 40:30:30 | 0.56 | 3 | 71.9% | 18.9 | PASS |
| **D3** | **full mix 25:25:25:25** | **0.53** | **3** | **64.7%** | **15.2** | **PASS** |
| D4 | full mix + rollout near3 | 0.49 | 3 | 96.1% | 32.9 | FAIL |
| D5 | full mix + progress + exit | 0.46 | 3 | 98.0% | 8.9 | FAIL |

**结论**: D3 (四类均匀) 是旧数据中最佳配比。数学 50:50 (D1) 直接坍缩。near3 在数据实验中同样有毒。

### E 组：早期组合探索 (部分使用旧配方)
| ID | Config | loss | rank | mode1 | ratio | 判定 |
|----|--------|------|------|-------|-------|------|
| E1 | rollout near3 + progress exit | 1.22 | 3 | 80.5% | 25.3 | WARN |
| E2 | E1 + local consistency | 1.22 | 3 | 72.0% | 19.2 | PASS |
| E3 | E1 + full mix data | 0.48 | 3 | 98.2% | 16.4 | FAIL |
| E4 | full combo + data | 0.47 | 3 | 77.5% | 12.0 | PASS |

### F 组：最优双模块组合验证
| ID | Config | loss | rank | mode1 | ratio | 判定 |
|----|--------|------|------|-------|-------|------|
| F1 | B1+C1 (progress+consistency) | 1.20 | 3 | 61.0% | 21.9 | PASS |
| **F2** | **B1+A1 (progress+rollout legacy)** | **1.20** | **3** | **52.0%** | **23.9** | **PASS** |
| F3 | B1+C1+full mix data | 0.54 | 3 | 71.1% | 13.0 | PASS |
| F4 | B1+A1+C1 (三模块) | 1.24 | 3 | **87.4%** | 18.0 | WARN |
| F5 | B1+A1+C1+full mix | — | — | — | — | SKIP |
| F6 | B1+C3 (progress+sigreg) | 1.19 | 3 | 70.8% | 23.1 | PASS |

**结论**: F2 (progress+rollout legacy) 确认为最优双模块配置，mode1=52.0% 全场最低。三模块 F4 反而更差 (87.4%)，叠加有干扰。

### G 组：新真实数据源验证 (使用 F2 配置)
| ID | 数据 | loss | rank | mode1 | ratio | 判定 |
|----|------|------|------|-------|-------|------|
| G1 | full_mix (五类均匀) | 7.23 | **2** | 96.7% | 11.7 | FAIL |
| G2 | reasoning_mix (math+arc+python) | 6.96 | **2** | 94.3% | 9.3 | FAIL |
| G3 | chinese_heavy (中文为主) | 4.40 | 3 | 72.5% | 9.9 | WARN |
| G4 | full_mix_large (2.2万全量) | 7.70 | 3 | 81.7% | 10.2 | WARN |
| **G5** | **diag_math (persona+math)** | **1.81** | **3** | **43.9%** | **16.4** | **PASS** |

**结论**: G5 (persona+真实数学) mode1=43.9% 是全部 35 个实验中最健康的梯度分布！多类型数据在 312M 下容量不足，英文为主 (G1/G2) 直接坍缩。中文为主 (G3) 勉强可用。persona+math (G5) 是 312M 的最佳数据配比。

---

## 3. 核心发现

### 3.1 模块层面
1. **progress-shape (w=0.05) 是最有价值的单一模块** — 将 loss 从 baseline ~1.25 降至 1.07
2. **rollout legacy (w=0.1) 是最佳搭档** — 与 progress-shape 组合后 mode1 降至 52.0%
3. **near3 加权模式有毒** — 在 A2/A6/D4 中反复导致坍缩，必须禁用
4. **三模块叠加不如双模块** — F4 (三模块) mode1=87.4% 远差于 F2 (双模块) 52.0%

### 3.2 数据层面
1. **真实数学数据 (GSM8K + hendrycks_math) 改善梯度健康度** — G5 mode1=43.9%，比纯 persona 的 F2 (52.0%) 更好
2. **312M 参数无法消化 5 种以上数据类型** — G1 (五类) rank=2 坍缩
3. **中文为主的配比 (G3) 是多类型数据的安全上限** — mode1=72.5% 尚可
4. **数据越多样，ratio 越低** — G 组 ratio 9-16 vs F2 24，reasoning zone 获得了更多梯度

### 3.3 容量约束
1. 312M 模型在单类型/双类型数据上表现最佳
2. 多类型数据需要更大模型容量 — 这是扩参数量的关键动机
3. 数据质量比数量重要 — G4 (2.2万) 不如 G5 (6000)

---

## 4. 最优配置推荐

### Phase 4 最终 baseline (312M)
```
--self_progress_shape_weight 0.05
--self_rollout_weight 0.1
--data_path pretrain_diag_math.jsonl    # persona + 真实数学
```

### 预期指标
| 指标 | 值 |
|------|---|
| loss_lm | ~1.80 (真实数学更难) |
| DOD rank | 3 |
| mode1_energy | ~44% |
| ratio | ~16 |

---

## 5. 数据集现状 (DataMix v2)

| 数据源 | 条数 | 来源 | 状态 |
|--------|------|------|------|
| persona_private | 43,053 | pretrain_diag + wechat (私有) | ✅ 清洗完毕 |
| math_real | 14,227 | GSM8K + hendrycks_math | ✅ |
| python_code | 26,802 | python_code_18k + CodeAlpaca | ✅ |
| chinese_scifi | 4,636 | 刘慈欣 63 本 (水印已清除) | ✅ |
| arc_agi | 1,702 | ARC-AGI 400 tasks | ✅ |
| **总计** | **90,420** | | |

已删除: emotion_real (ESConv, 96%超长无法用), chinese_dialog (Belle, 质量太低), 旧合成模板数据 (hard_math/emotion/persona_seed)

---

## 6. 下一步方向

1. **G5 加长训练 (3000-5000 步)** — 验证 persona+math 在更长训练中 loss 能否降到 1.4 以下
2. **H 组：在 G5 基础上逐步加入第三类数据** — 找到 312M 的数据多样性上限
   - H1: G5 + 少量 scifi (5%)
   - H2: G5 + 少量 python (5%)
   - H3: G5 + 少量 ARC (5%)
3. **模型扩容评估** — 如果 312M 确认是容量瓶颈，规划 ~1B 参数版本
4. **World JEPA 回归** — 在 Phase 4 baseline 稳定后，重新设计 Phase 6 的 World JEPA 方案

---

## 附录：Judge 阈值 (v2)

```python
# 单类型数据 (A-C, F 组)
FAIL: rank < 3 OR mode1 > 95% OR loss > 1.40
WARN: mode1 > 80% OR loss > 1.30

# 多类型数据 (D, G 组)
FAIL: rank < 3 OR mode1 > 95% OR loss > 8.0
WARN: mode1 > 85% OR loss > 5.0
```
