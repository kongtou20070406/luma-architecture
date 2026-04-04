#!/usr/bin/env python3
"""
统一混合数据集重建脚本。

数据源（7 类）：
  - persona_private.jsonl  — 人格种子（私有真实发言）
  - math_real.jsonl        — GSM8K + hendrycks_math
  - emotion_real.jsonl     — ESConv 情感对话
  - python_code.jsonl      — Python 代码指令
  - chinese_scifi.jsonl    — 刘慈欣科��小说
  (chinese_dialog 已删除 — Belle 质量太低)
  - arc_agi.jsonl          — ARC-AGI 抽象推理（从原始 JSON grid 转换）

混合方案：
  pretrain_diag_math.jsonl     — persona 3000 + math 3000
  pretrain_full_mix.jsonl      — 七类均匀混合，共 7000
  pretrain_chinese_heavy.jsonl — 中文为主: persona 2000 + scifi 1500 + dialog 1500 + math 500 + emo 250 + python 250
  pretrain_reasoning_mix.jsonl — 推理强化: math 2000 + arc 1500 + python 1500 + scifi 500 + persona 500
  pretrain_full_mix_large.jsonl — 全量: 每类取 min(实际数量, 5000)，不做上采样

运行: python rebuild_mixes.py
"""

import json
import os
import re
import random
from pathlib import Path

random.seed(42)

LUMA_DATASET = Path(__file__).resolve().parent
SYNTHETIC_DIR = LUMA_DATASET / "synthetic"
MIXES_DIR = LUMA_DATASET / "mixes"
TRAINER_DATASET = LUMA_DATASET.parent / "minimind" / "dataset"
ARC_DATA_DIR = LUMA_DATASET.parent / "data" / "ARC-AGI" / "data"

# 当前 trainer max_seq_len=512 tokens
# 中文 ~1 char/tok, 英文 ~3-4 chars/tok, 留 margin → 用 450 chars 作为截断阈值
MAX_CHARS = 450


# ── 工具函数 ────────────────────────────────────────────────

def truncate_at_boundary(text: str, max_chars: int) -> str:
    """截断到 max_chars 以内，在句子边界处切断。"""
    if len(text) <= max_chars:
        return text
    # 在 max_chars 范围内找最后一个句���结束符
    truncated = text[:max_chars]
    # 中文句号/问号/感叹号/换行 作���切断点
    for sep in ['\n', '。', '！', '？', '…', '. ', '! ', '? ']:
        pos = truncated.rfind(sep)
        if pos > max_chars * 0.5:  # 至少保留一半内容
            return truncated[:pos + len(sep)].rstrip()
    # 找不到好的边界就硬切
    return truncated.rstrip()


def load_jsonl(path: Path, truncate: bool = True) -> list[dict]:
    if not path.exists():
        print(f"  ⚠️ 不存在: {path.name}")
        return []
    data = []
    truncated_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if truncate and len(d["text"]) > MAX_CHARS:
                d["text"] = truncate_at_boundary(d["text"], MAX_CHARS)
                truncated_count += 1
            data.append(d)
    if truncated_count:
        print(f"    ↪ {path.name}: {truncated_count} 条被截断到 {MAX_CHARS} chars")
    return data


def write_jsonl(path: Path, data: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  ✅ {path.name}: {len(data)} 条")


def sample_or_oversample(data: list[dict], n: int) -> list[dict]:
    if not data:
        return []
    if len(data) >= n:
        return random.sample(data, n)
    result = data * (n // len(data)) + random.sample(data, n % len(data))
    return result[:n]


def sample_no_oversample(data: list[dict], n: int) -> list[dict]:
    """取 min(len(data), n) 条，不做上采样。"""
    if not data:
        return []
    return random.sample(data, min(len(data), n))


def ensure_symlink(name: str, target: Path):
    link = TRAINER_DATASET / name
    if link.exists() or link.is_symlink():
        link.unlink()
    rel = os.path.relpath(target, TRAINER_DATASET)
    link.symlink_to(rel)
    print(f"  🔗 {name} → {rel}")


# ── ARC-AGI 转换 ────────────────────────────────────────────

def grid_to_text(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def convert_arc_agi() -> list[dict]:
    """将 ARC-AGI 的 JSON grid 任务转成文本训练样本。

    每个任务生成多条样本：
    - few-shot 完整样本（含所有 train examples + test，作为训练数据）
    - 每个 train pair 单独作为一条样本（增加数据量）
    """
    results = []
    training_dir = ARC_DATA_DIR / "training"

    if not training_dir.exists():
        print(f"  ⚠️ ARC-AGI 训练目录不存在: {training_dir}")
        return results

    task_files = sorted(training_dir.glob("*.json"))
    print(f"  处理 ARC-AGI 训练任务: {len(task_files)} 个 ...")

    for tf in task_files:
        with open(tf) as f:
            task = json.load(f)

        train_pairs = task.get("train", [])
        test_pairs = task.get("test", [])

        # 样本 1: few-shot 完整 prompt + answer（最核心的训练格式）
        if train_pairs and test_pairs:
            parts = ["Below are input-output grid transformation examples. "
                     "Study the pattern and predict the output for the test input.\n"]
            for i, ex in enumerate(train_pairs):
                parts.append(f"Example {i+1}:")
                parts.append(f"Input:\n{grid_to_text(ex['input'])}")
                parts.append(f"Output:\n{grid_to_text(ex['output'])}")
                parts.append("")

            for test in test_pairs:
                parts.append("Test:")
                parts.append(f"Input:\n{grid_to_text(test['input'])}")
                parts.append(f"Output:\n{grid_to_text(test['output'])}")

            results.append({"text": "\n".join(parts)})

        # 样本 2: 每个 train pair 单独成一条（简单的 input→output 映射）
        for ex in train_pairs:
            text = (f"Grid transformation:\n"
                    f"Input:\n{grid_to_text(ex['input'])}\n"
                    f"Output:\n{grid_to_text(ex['output'])}")
            results.append({"text": text})

    print(f"    ARC-AGI: {len(results)} 条 (from {len(task_files)} tasks)")
    return results


# ── 主流程 ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("统一混合数据集重建")
    print("=" * 60)

    # 加载所有数据源
    persona = load_jsonl(SYNTHETIC_DIR / "persona_private.jsonl")
    math = load_jsonl(SYNTHETIC_DIR / "math_real.jsonl")
    python_code = load_jsonl(SYNTHETIC_DIR / "python_code.jsonl")
    scifi = load_jsonl(SYNTHETIC_DIR / "chinese_scifi.jsonl")
    # ARC-AGI: 从原始 JSON 转换
    print("\n[ARC-AGI] 转换训练数据 ...")
    arc = convert_arc_agi()
    write_jsonl(SYNTHETIC_DIR / "arc_agi.jsonl", arc)

    sources = {
        "人格种子": persona,
        "数学推理": math,
        "Python代码": python_code,
        "科幻小说": scifi,
        "ARC-AGI": arc,
    }

    print(f"\n数据源统计:")
    total = 0
    for name, data in sources.items():
        print(f"  {name:<10} {len(data):>6}")
        total += len(data)
    print(f"  {'总计':<10} {total:>6}")

    # ── 生成混合数据集 ──────────────────────────────────────

    print(f"\n生成混合数据集 ...")

    # Mix 1: persona + math 50:50 (对照组，和之前的 D 组实验兼容)
    mix1 = sample_or_oversample(persona, 3000) + sample_or_oversample(math, 3000)
    random.shuffle(mix1)
    write_jsonl(MIXES_DIR / "pretrain_diag_math.jsonl", mix1)

    # Mix 2: 五类均匀混合 (各 1200 = 6000)
    mix2 = (sample_or_oversample(persona, 1200)
            + sample_or_oversample(math, 1200)
            + sample_or_oversample(python_code, 1200)
            + sample_or_oversample(scifi, 1200)
            + sample_or_oversample(arc, 1200))
    random.shuffle(mix2)
    write_jsonl(MIXES_DIR / "pretrain_full_mix.jsonl", mix2)

    # Mix 3: 中文为主（persona + 科幻 + 数学 + python）
    mix3 = (sample_or_oversample(persona, 2500)
            + sample_or_oversample(scifi, 1500)
            + sample_or_oversample(math, 1000)
            + sample_or_oversample(python_code, 1000))
    random.shuffle(mix3)
    write_jsonl(MIXES_DIR / "pretrain_chinese_heavy.jsonl", mix3)

    # Mix 4: 推理强化（数学 + ARC + 代码 为主）
    mix4 = (sample_or_oversample(math, 2000)
            + sample_or_oversample(arc, 1500)
            + sample_or_oversample(python_code, 1500)
            + sample_or_oversample(scifi, 500)
            + sample_or_oversample(persona, 500))
    random.shuffle(mix4)
    write_jsonl(MIXES_DIR / "pretrain_reasoning_mix.jsonl", mix4)

    # Mix 5: 全量混合（每类取 min(实际数量, 5000)，不上采样，保真实分布）
    mix5 = []
    for name, data in sources.items():
        sampled = sample_no_oversample(data, 5000)
        mix5.extend(sampled)
    random.shuffle(mix5)
    write_jsonl(MIXES_DIR / "pretrain_full_mix_large.jsonl", mix5)

    # ── 清理旧文件 ──────────────────────────────────────────

    for old_name in ["pretrain_diag_emo_python.jsonl", "pretrain_diag_emo_persona.jsonl"]:
        old = MIXES_DIR / old_name
        if old.exists():
            old.unlink()
            print(f"  🗑️  删除旧 {old.name}")
        link = TRAINER_DATASET / old_name
        if link.is_symlink():
            link.unlink()
            print(f"  🗑️  删除旧 symlink {old_name}")

    # ── 更新 symlinks ──────────────────────────────────────

    print(f"\n更新 trainer symlinks ...")
    for mix_name in ["pretrain_diag_math.jsonl", "pretrain_full_mix.jsonl",
                     "pretrain_chinese_heavy.jsonl", "pretrain_reasoning_mix.jsonl",
                     "pretrain_full_mix_large.jsonl"]:
        ensure_symlink(mix_name, MIXES_DIR / mix_name)

    # ── 汇总 ──────────────────────────────────────────────

    print(f"\n混合数据集汇总:")
    for p in sorted(MIXES_DIR.glob("*.jsonl")):
        n = sum(1 for _ in open(p))
        print(f"  {p.name:<40} {n:>6} 条")

    print(f"\n完成！")


if __name__ == "__main__":
    main()
