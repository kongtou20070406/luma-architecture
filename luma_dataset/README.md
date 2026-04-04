# Luma Dataset Workspace

这个目录现在不是“随手放几份数据”的地方，而是 Luma 正式预训练与更贴近预训练实验的数据工作区。

## 目标

Luma 的数据目标不是只做一个会聊天的壳，也不是只做一个解题器。
我们现在明确追求三件事同时成立：

- 足够聪明：能做较长链条的数学、代码、规划推理
- 理解情感：会共情、支持、识别情绪与语境
- 像你但不复制你：保留人格底色与聊天手感，但不把 persona 桶当成唯一中心

所以当前 `DataMix v1` 的总原则是：

- `50%` 先让 Luma 更聪明
- 其余 `50%` 再分给情感支持、人格底色、对话质量与叙述能力

## 目录结构

```
luma_dataset/
├── datamix_v1.yaml              — 数据混合方案 (bucket 定义、来源、比例)
├── license_whitelist.md         — license 白名单
├── source_status.md             — 来源注册表
├── datamix_stats.template.json  — 统计模板
├── generate_mixed_data.py       — 合成数据生成 (seed=42, 可复现)
├── fetch_enabled_sources.py     — 从 HuggingFace 拉取真实数据
├── prepare_datamix_v1.py        — 正式 DataMix v1 混合
├── synthetic/                   — 程序化生成的合成训练数据
│   ├── pretrain_diag.jsonl        (6445条 原始闲聊)
│   ├── hard_math.jsonl            (2000条, 12种题型)
│   ├── emotion.jsonl              (1500条 情感表达与分析)
│   └── persona_seed.jsonl         (1500条 角色设定对话)
└── mixes/                       — 混合训练数据集（实验矩阵用）
    ├── pretrain_diag_math.jsonl         (diag 3000 + math 3000)
    ├── pretrain_diag_emo_persona.jsonl  (diag 2400 + emo 1800 + persona 1800)
    └── pretrain_full_mix.jsonl          (四类各 1500, 共 6000)
```

`minimind/dataset/` 下有 symlinks 指向 `mixes/`，保持 trainer 兼容。

## 为什么要把 harder math 放进聪明桶

是的，较难数学通常能提升模型的“智能感”，但前提是混得对。

原因：

- harder math 会逼模型维护更长的局部依赖与中间状态
- 它通常能提升多步推进、局部一致性、错误恢复能力
- 它对“anti-stupidness”很有帮助，也常常能带动代码与结构化思考

但它也不能无限加大，因为：

- 数学比例太高，会伤对话与情感表现
- 会把模型往“冷硬解题器”方向推过头

所以现在我们的策略不是“全上 hardest math”，而是：

- 用 harder math 做聪明桶的骨架
- 再用代码、情感支持和 persona 桶把 Luma 拉回聊天伙伴的形状

## 当前推荐的 DataMix v1

详见：
- `datamix_v1.yaml`
- `license_whitelist.md`

## 说明

- `persona_seed/` 是私有数据区，不应该进入公开仓
- 公开分享时，只分享 `manifests/`、清洗脚本、统计方法，不分享私有原始语料
