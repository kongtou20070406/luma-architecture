#!/usr/bin/env python3
"""
从 HuggingFace 拉取真实学术数据集，清洗转成 Luma trainer 兼容的 {"text": ...} jsonl 格式。
替换原有的合成模板数据。

数据桶：
  - math:    GSM8K + hendrycks_math → math_real.jsonl
  - emotion: ESConv → emotion_real.jsonl
  - python:  python_code_18k + CodeAlpaca-20k → python_code.jsonl
  - persona: pretrain_diag (清洗) + wechat_pretrain → persona_private.jsonl

运行: python fetch_real_datasets.py
"""

import json
import os
import random
import hashlib
from pathlib import Path

random.seed(42)

LUMA_DATASET = Path(__file__).resolve().parent
SYNTHETIC_DIR = LUMA_DATASET / "synthetic"
MIXES_DIR = LUMA_DATASET / "mixes"

# Trainer 兼容目录
TRAINER_DATASET = LUMA_DATASET.parent / "minimind" / "dataset"


def write_jsonl(path: Path, data: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  ✅ {path.name}: {len(data)} 条")


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def dedup_by_text(data: list[dict]) -> list[dict]:
    """按 text 内容去重。"""
    seen = set()
    result = []
    for d in data:
        h = hashlib.md5(d["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(d)
    return result


def sample_or_oversample(data: list[dict], n: int) -> list[dict]:
    if len(data) >= n:
        return random.sample(data, n)
    result = data * (n // len(data)) + random.sample(data, n % len(data))
    return result[:n]


# ============================================================
# 1. 数学推理 — GSM8K + hendrycks_math
# ============================================================

def fetch_math() -> list[dict]:
    from datasets import load_dataset

    results = []

    # GSM8K: question + step-by-step answer
    print("  拉取 openai/gsm8k ...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for item in ds:
        q = item["question"].strip()
        a = item["answer"].strip()
        # GSM8K answer 格式: 推理过程 \n#### 最终答案
        text = f"问题：{q}\n解答：{a}"
        results.append({"text": text})
    print(f"    GSM8K: {len(results)} 条")

    # hendrycks_math: 6 个子集
    math_subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra",
        # "precalculus" — 部分子集可能不存在
    ]
    math_count = 0
    for subject in math_subjects:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
            for item in ds:
                p = item["problem"].strip()
                s = item["solution"].strip()
                text = f"[{subject}] {p}\n\n{s}"
                results.append({"text": text})
                math_count += 1
        except Exception as e:
            print(f"    ⚠️ hendrycks_math/{subject}: {e}")
    print(f"    hendrycks_math: {math_count} 条")

    results = dedup_by_text(results)
    print(f"    去重后: {len(results)} 条")
    return results


# ============================================================
# 2. 情感对话 — ESConv
# ============================================================

def fetch_emotion() -> list[dict]:
    from datasets import load_dataset

    results = []

    # ESConv: 情感支持对话
    print("  拉取 thu-coai/esconv ...")
    ds = load_dataset("thu-coai/esconv", split="train")
    for item in ds:
        parsed = json.loads(item["text"])
        emotion = parsed.get("emotion_type", "")
        situation = parsed.get("situation", "")
        dialog = parsed.get("dialog", [])

        if not dialog or len(dialog) < 2:
            continue

        # 构建对话文本
        turns = []
        for turn in dialog:
            speaker = "求助者" if turn["speaker"] == "usr" else "支持者"
            turns.append(f"{speaker}：{turn['text']}")

        header = f"[情感：{emotion}]"
        if situation:
            header += f" 背景：{situation}"

        text = header + "\n" + "\n".join(turns)
        results.append({"text": text})

    print(f"    ESConv: {len(results)} 条")
    results = dedup_by_text(results)
    print(f"    去重后: {len(results)} 条")
    return results


# ============================================================
# 3. Python 代码 — python_code_18k + CodeAlpaca-20k
# ============================================================

def fetch_python() -> list[dict]:
    from datasets import load_dataset

    results = []

    # python_code_instructions_18k_alpaca
    print("  拉取 iamtarun/python_code_instructions_18k_alpaca ...")
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    for item in ds:
        instruction = item["instruction"].strip()
        output = item["output"].strip()
        if not output or len(output) < 20:
            continue
        text = f"# Task: {instruction}\n\n{output}"
        results.append({"text": text})
    print(f"    python_code_18k: {len(results)} 条")

    # CodeAlpaca-20k (只取 Python 相关)
    print("  拉取 sahil2801/CodeAlpaca-20k ...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    code_count = 0
    for item in ds:
        instruction = item["instruction"].strip()
        output = item["output"].strip()
        if not output or len(output) < 20:
            continue
        # 粗过滤：只保留明确 Python 相关或通用算法题
        lower_instr = instruction.lower()
        lower_out = output.lower()
        is_python = (
            "python" in lower_instr
            or "def " in lower_out
            or "import " in lower_out
            or "print(" in lower_out
        )
        if not is_python:
            continue
        text = f"# Task: {instruction}\n\n{output}"
        results.append({"text": text})
        code_count += 1
    print(f"    CodeAlpaca (Python filtered): {code_count} 条")

    results = dedup_by_text(results)
    print(f"    去重后: {len(results)} 条")
    return results


# ============================================================
# 4. 人格种子 — pretrain_diag (清洗) + wechat_pretrain
# ============================================================

def build_persona() -> list[dict]:
    print("  合并人格种子数据 ...")

    results = []

    # pretrain_diag — 只取中文闲聊部分 (过滤掉混入的英文数学/esconv)
    diag_path = SYNTHETIC_DIR / "pretrain_diag.jsonl"
    if diag_path.exists():
        raw = load_jsonl(diag_path)
        clean = []
        for d in raw:
            t = d["text"]
            # 过滤条件：排除英文数学、LaTeX、ESConv JSON
            if "\\boxed" in t or "\\frac" in t:
                continue
            if t.startswith('{"experience_type"'):
                continue
            if len(t) > 10 and all(ord(c) < 0x4e00 or ord(c) > 0x9fff for c in t[:20] if c.isalpha()):
                # 前20字符没有中文字符且超过10字符 → 可能是英文混入
                has_cn = any('\u4e00' <= c <= '\u9fff' for c in t)
                if not has_cn:
                    continue
            clean.append(d)
        print(f"    pretrain_diag: {len(raw)} → 清洗后 {len(clean)} 条")
        results.extend(clean)

    # wechat_pretrain
    wechat_path = Path("/home/kt/ai/wechat_pretrain.jsonl")
    if wechat_path.exists():
        wechat = load_jsonl(wechat_path)
        print(f"    wechat_pretrain: {len(wechat)} 条")
        results.extend(wechat)
    else:
        print("    ⚠️ wechat_pretrain.jsonl 未找到")

    results = dedup_by_text(results)
    print(f"    合并去重后: {len(results)} 条")
    return results


# ============================================================
# 5. 混合数据集生成
# ============================================================

def build_mixes(persona: list, math: list, emotion: list, python_code: list):
    """生成混合训练数据集。"""
    print("\n生成混合数据集 ...")

    # Mix 1: persona + math 50:50
    n_half = 3000
    mix1 = sample_or_oversample(persona, n_half) + sample_or_oversample(math, n_half)
    random.shuffle(mix1)
    write_jsonl(MIXES_DIR / "pretrain_diag_math.jsonl", mix1)

    # Mix 2: persona + emotion + python 40:30:30
    mix2 = (sample_or_oversample(persona, 2400)
            + sample_or_oversample(emotion, 1800)
            + sample_or_oversample(python_code, 1800))
    random.shuffle(mix2)
    write_jsonl(MIXES_DIR / "pretrain_diag_emo_python.jsonl", mix2)

    # Mix 3: full mix 四类各 1500 = 6000
    mix3 = (sample_or_oversample(persona, 1500)
            + sample_or_oversample(math, 1500)
            + sample_or_oversample(emotion, 1500)
            + sample_or_oversample(python_code, 1500))
    random.shuffle(mix3)
    write_jsonl(MIXES_DIR / "pretrain_full_mix.jsonl", mix3)

    # 更新 symlinks
    symlinks = {
        "pretrain_diag_math.jsonl": MIXES_DIR / "pretrain_diag_math.jsonl",
        "pretrain_full_mix.jsonl": MIXES_DIR / "pretrain_full_mix.jsonl",
    }
    for name, target in symlinks.items():
        link = TRAINER_DATASET / name
        if link.exists() or link.is_symlink():
            link.unlink()
        rel = os.path.relpath(target, TRAINER_DATASET)
        link.symlink_to(rel)
        print(f"  🔗 {link.name} → {rel}")


def main():
    print("=" * 60)
    print("Luma 真实数据集拉取与处理")
    print("=" * 60)

    # 1. 人格种子（本地，不需要网络）
    print("\n[1/4] 人格种子（私有数据）")
    persona = build_persona()
    write_jsonl(SYNTHETIC_DIR / "persona_private.jsonl", persona)

    # 2. 数学推理
    print("\n[2/4] 数学推理")
    math = fetch_math()
    write_jsonl(SYNTHETIC_DIR / "math_real.jsonl", math)

    # 3. 情感对话
    print("\n[3/4] 情感对话")
    emotion = fetch_emotion()
    write_jsonl(SYNTHETIC_DIR / "emotion_real.jsonl", emotion)

    # 4. Python 代码
    print("\n[4/4] Python 代码")
    python_code = fetch_python()
    write_jsonl(SYNTHETIC_DIR / "python_code.jsonl", python_code)

    # 5. 生成混合数据集
    build_mixes(persona, math, emotion, python_code)

    # 6. 删除旧的合成数据
    print("\n清理旧合成数据 ...")
    old_files = ["hard_math.jsonl", "emotion.jsonl", "persona_seed.jsonl"]
    for f in old_files:
        p = SYNTHETIC_DIR / f
        if p.exists():
            p.unlink()
            print(f"  🗑️  删除 {f}")

    # 旧 mix 文件
    old_mixes = ["pretrain_diag_emo_persona.jsonl"]
    for f in old_mixes:
        p = MIXES_DIR / f
        if p.exists():
            p.unlink()
            print(f"  🗑️  删除 mixes/{f}")
        # 清理旧 symlink
        link = TRAINER_DATASET / f
        if link.is_symlink():
            link.unlink()
            print(f"  🗑️  删除 symlink {f}")

    # 汇总
    print("\n" + "=" * 60)
    print("数据汇总")
    print("=" * 60)
    print(f"  人格种子:  {len(persona):>6} 条")
    print(f"  数学推理:  {len(math):>6} 条")
    print(f"  情感对话:  {len(emotion):>6} 条")
    print(f"  Python 代码: {len(python_code):>6} 条")
    print(f"  总计:      {len(persona)+len(math)+len(emotion)+len(python_code):>6} 条")


if __name__ == "__main__":
    main()
