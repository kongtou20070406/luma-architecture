#!/usr/bin/env python3
"""
统一混合数据集重建脚本 v2。

数据源（扩展后 11 类）：
  原有:
    - persona_private.jsonl  — 人格种子（私有真实发言）
    - math_real.jsonl        — GSM8K + hendrycks_math
    - python_code.jsonl      — Python 代码指令
    - chinese_scifi.jsonl    — 刘慈欣科幻小说
    - arc_agi.jsonl          — ARC-AGI 抽象推理
  新增:
    - oasst1.jsonl           — OpenAssistant 多轮对话
    - ultrafeedback.jsonl    — 高质量回答偏好
    - empathetic_dialogues.jsonl — 共情对话
    - zhihu_kol.jsonl        — 知乎高质量中文长回答
    - stack_python.jsonl     — the-stack 真实 Python 代码

混合方案 (DataMix v1):
  pretrain_v1.jsonl — 正式预训练数据集, 按 DataMix v1 配比:
    A. 聪明桶 50%: math 25% + code 15% + reasoning_dialogue 10%
    B. 情感桶 20%: empathetic_dialogues + esconv
    C. 人格桶 15%: persona_private
    D. 对话质量桶 15%: zhihu_kol + oasst1

  保留旧 mix 用于回归测试:
    pretrain_h_python.jsonl  — 旧 H-Python mix (persona + math + python 300)

运行: python rebuild_mixes.py
"""

import json
import os
import random
from pathlib import Path

random.seed(42)

LUMA_DATASET = Path(__file__).resolve().parent
SYNTHETIC_DIR = LUMA_DATASET / "synthetic"
MIXES_DIR = LUMA_DATASET / "mixes"
TRAINER_DATASET = LUMA_DATASET.parent / "minimind" / "dataset"
ARC_DATA_DIR = LUMA_DATASET.parent / "data" / "ARC-AGI" / "data"

# seq=4096 时，4096 tokens ≈ 14000 chars (混合中英文)
# 不做硬截断，让 trainer 的 tokenizer 自然处理
MAX_CHARS = 16000  # 安全上限，只截断极端长文


# ── 工具函数 ────────────────────────────────────────────────

def truncate_at_boundary(text: str, max_chars: int) -> str:
    """截断到 max_chars 以内，在句子边界处切断。"""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in ['\n\n', '\n', '。', '！', '？', '…', '. ', '! ', '? ']:
        pos = truncated.rfind(sep)
        if pos > max_chars * 0.5:
            return truncated[:pos + len(sep)].rstrip()
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
            text = d.get("text", "")
            # 过滤空文本和极短文本
            if len(text) < 5:
                continue
            if truncate and len(text) > MAX_CHARS:
                d["text"] = truncate_at_boundary(text, MAX_CHARS)
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


def estimate_tokens(data: list[dict]) -> int:
    """粗估 token 数 (中英混合 ~3.5 chars/token)。"""
    return int(sum(len(d["text"]) for d in data) / 3.5)


# ── ARC-AGI 转换 ────────────────────────────────────────────

def grid_to_text(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def convert_arc_agi() -> list[dict]:
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
    print("统一混合数据集重建 v2 (DataMix v1)")
    print("=" * 60)

    # ── 加载所有数据源 ──────────────────────────────────────

    # 原有数据
    print("\n加载原有数据源 ...")
    persona = load_jsonl(SYNTHETIC_DIR / "persona_private.jsonl")
    math = load_jsonl(SYNTHETIC_DIR / "math_real.jsonl")
    python_code = load_jsonl(SYNTHETIC_DIR / "python_code.jsonl")
    scifi = load_jsonl(SYNTHETIC_DIR / "chinese_scifi.jsonl")

    print("\n[ARC-AGI] 转换训练数据 ...")
    arc = convert_arc_agi()
    write_jsonl(SYNTHETIC_DIR / "arc_agi.jsonl", arc)

    # 新增数据（只保留纯人类真实语料 + 数学/代码/推理）
    print("\n加载扩展数据源 ...")
    # oasst1/ultrafeedback 是人类-AI 对话 → 按用户要求排除
    zhihu = load_jsonl(SYNTHETIC_DIR / "zhihu_kol.jsonl")
    stack_python = load_jsonl(SYNTHETIC_DIR / "stack_python.jsonl")
    wechat_sft = load_jsonl(SYNTHETIC_DIR / "wechat_sft.jsonl")

    # ── 数据源统计 ──────────────────────────────────────────

    all_sources = {
        "persona_private": persona,
        "math_real": math,
        "python_code": python_code,
        "chinese_scifi": scifi,
        "arc_agi": arc,
        "zhihu_kol": zhihu,
        "stack_python": stack_python,
        "wechat_sft": wechat_sft,
    }

    print(f"\n数据源统计:")
    total_samples = 0
    total_tokens = 0
    for name, data in all_sources.items():
        tokens = estimate_tokens(data)
        print(f"  {name:<25} {len(data):>8} 条  ~{tokens/1e6:.1f}M tokens")
        total_samples += len(data)
        total_tokens += tokens
    print(f"  {'总计':<25} {total_samples:>8} 条  ~{total_tokens/1e6:.0f}M tokens")

    # ── DataMix v1 正式预训练数据集 ────────────────────────

    print(f"\n生成 DataMix v1 正式预训练数据集 ...")

    # 目标总量：用完所有 unique 数据，不上采样（除了 persona 桶）
    # 按 DataMix v1 配比分配:
    #   A. 聪明桶 50%
    #     - smart_math (25%): math_real + arc_agi
    #     - smart_code (15%): python_code + stack_python
    #     - smart_reasoning (10%): oasst1 + ultrafeedback
    #   B. 情感桶 (20%): empathetic_dialogues + scifi (替代 esconv/emotion_real)
    #   C. 人格桶 (15%): persona_private
    #   D. 对话质量桶 (15%): zhihu_kol

    # 合并各桶数据源
    bucket_smart_math = math + arc
    bucket_smart_code = python_code + stack_python
    bucket_smart_reason = []  # oasst/ultrafeedback removed (human-AI dialogue)
    bucket_empathy = scifi  # pure human fiction
    bucket_persona = persona + wechat_sft  # wechat_sft 加入人格桶
    bucket_dialogue = zhihu

    bucket_info = {
        "smart_math (25%)": bucket_smart_math,
        "smart_code (15%)": bucket_smart_code,
        "smart_reason (10%)": bucket_smart_reason,
        "empathy (20%)": bucket_empathy,
        "persona (15%)": bucket_persona,
        "dialogue (15%)": bucket_dialogue,
    }

    # 计算各桶实际 token 数
    print(f"\n  桶统计:")
    bucket_tokens = {}
    for name, data in bucket_info.items():
        t = estimate_tokens(data)
        bucket_tokens[name] = t
        print(f"    {name:<25} {len(data):>8} 条  ~{t/1e6:.1f}M tokens")

    # 策略：以实际数据量为准，按比例缩放
    # 找到"最紧缺"的桶（token数/目标占比 最小），以它为基准
    target_ratios = {
        "smart_math (25%)": 0.25,
        "smart_code (15%)": 0.15,
        "smart_reason (10%)": 0.10,
        "empathy (20%)": 0.20,
        "persona (15%)": 0.15,
        "dialogue (15%)": 0.15,
    }

    # 策略：使用所有 unique 数据，对 token 稀少的桶做温和上采样 (最多 5x)
    # 正式预训练前需要补充 empathy/persona 数据源
    v1_mix = []
    for name, data in bucket_info.items():
        if not data:
            continue
        actual_tokens = bucket_tokens[name]
        target_pct = target_ratios[name]
        # 计算理想 token 数（以总 unique tokens 为基准）
        ideal_tokens = total_tokens * target_pct
        if actual_tokens > 0:
            oversample_ratio = min(5.0, ideal_tokens / actual_tokens)  # 最多 5x 上采样
            target_n = max(len(data), int(len(data) * oversample_ratio))
            sampled = sample_or_oversample(data, target_n)
        else:
            sampled = data
        v1_mix.extend(sampled)

    random.shuffle(v1_mix)

    # 实际比例
    total_v1 = len(v1_mix)
    print(f"\n  DataMix v1 实际配比 (总 {total_v1} 条):")
    for name, data in bucket_info.items():
        actual_pct = len(data) / total_v1 * 100
        target_pct = target_ratios[name] * 100
        diff = actual_pct - target_pct
        print(f"    {name:<25} {len(data):>8} ({actual_pct:5.1f}% vs 目标 {target_pct:.0f}%, {'↑' if diff > 0 else '↓'}{abs(diff):.1f}%)")

    write_jsonl(MIXES_DIR / "pretrain_v1.jsonl", v1_mix)
    ensure_symlink("pretrain_v1.jsonl", MIXES_DIR / "pretrain_v1.jsonl")

    # ── 保留旧 mix（向后兼容 + 回归测试）──────────────────

    print(f"\n保留旧混合数据集 ...")

    # H-mix: persona 3000 + math 3000 + X 300
    for h_name, h_data in [("scifi", scifi), ("python", python_code), ("arc", arc)]:
        h_mix = (sample_or_oversample(persona, 3000)
                 + sample_or_oversample(math, 3000)
                 + sample_or_oversample(h_data, 300))
        random.shuffle(h_mix)
        fname = f"pretrain_h_{h_name}.jsonl"
        write_jsonl(MIXES_DIR / fname, h_mix)
        ensure_symlink(fname, MIXES_DIR / fname)

    # 推理强化 mix（小型快速实验用）
    mix_reason = (sample_or_oversample(math, 2000)
                  + sample_or_oversample(arc, 1500)
                  + sample_or_oversample(python_code, 1500)
                  + sample_or_oversample(scifi, 500)
                  + sample_or_oversample(persona, 500))
    random.shuffle(mix_reason)
    write_jsonl(MIXES_DIR / "pretrain_reasoning_mix.jsonl", mix_reason)

    # Full mix large（旧版兼容）
    mix_large = []
    for name, data in all_sources.items():
        mix_large.extend(sample_no_oversample(data, 5000))
    random.shuffle(mix_large)
    write_jsonl(MIXES_DIR / "pretrain_full_mix_large.jsonl", mix_large)

    # ── 更新 symlinks ──────────────────────────────────────

    print(f"\n更新 trainer symlinks ...")
    for mix_name in ["pretrain_reasoning_mix.jsonl", "pretrain_full_mix_large.jsonl"]:
        ensure_symlink(mix_name, MIXES_DIR / mix_name)

    # ── 清理旧文件 ──────────────────────────────────────────

    for old_name in ["pretrain_diag_emo_python.jsonl", "pretrain_diag_emo_persona.jsonl",
                     "pretrain_diag_math.jsonl", "pretrain_full_mix.jsonl",
                     "pretrain_chinese_heavy.jsonl"]:
        old = MIXES_DIR / old_name
        if old.exists():
            old.unlink()
            print(f"  🗑️  删除旧 {old.name}")
        link = TRAINER_DATASET / old_name
        if link.is_symlink():
            link.unlink()
            print(f"  🗑️  删除旧 symlink {old_name}")

    # ── 汇总 ──────────────────────────────────────────────

    print(f"\n混合数据集汇总:")
    for p in sorted(MIXES_DIR.glob("*.jsonl")):
        n = sum(1 for _ in open(p))
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:<40} {n:>8} 条  {size_mb:.1f}MB")

    print(f"\n完成！")


if __name__ == "__main__":
    main()
