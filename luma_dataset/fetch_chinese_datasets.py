#!/usr/bin/env python3
"""
拉取中文数据：
  1. 刘慈欣科幻小说全集 → 切段成训练样本 (chinese_scifi.jsonl)
  2. BelleGroup 多轮中文对话 → 采样 (chinese_dialog.jsonl)

运行: python fetch_chinese_datasets.py
"""

import json
import os
import re
import random
import hashlib
from pathlib import Path

random.seed(42)

LUMA_DATASET = Path(__file__).resolve().parent
SYNTHETIC_DIR = LUMA_DATASET / "synthetic"


def write_jsonl(path: Path, data: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  ✅ {path.name}: {len(data)} 条")


def dedup_by_text(data: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for d in data:
        h = hashlib.md5(d["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(d)
    return result


# ============================================================
# 1. 刘慈欣科幻小说 → 段落切分
# ============================================================

def build_scifi(txt_dir: str = "/tmp/lcx_texts", chunk_chars: int = 512) -> list[dict]:
    """
    把每本小说按行累积切成训练样本。
    - chunk_chars: 每个样本的目标字符数
    - 不做 overlap，避免从句子中间截断
    """
    print("  处理刘慈欣科幻小说 ...")
    results = []
    txt_dir = Path(txt_dir)

    for txt_file in sorted(txt_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            raw = f.read()

        # 清理：去掉多余空行、制表符
        raw = raw.replace("\r\n", "\n").replace("\t", " ")
        raw = re.sub(r"\n{3,}", "\n\n", raw)

        # 清除盗版站水印行（每 ~300 行嵌一个，每次编码不同）
        # 策略：逐行检查，含有水印特征字符组合的整行删除
        clean_lines = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if not stripped:
                clean_lines.append(line)
                continue
            # 水印特征：短行 + 包含 丅/ㄒ/Т/Х/Н/亅/郃/匼/閤/磼/雧/粭 等非常规字符
            watermark_chars = set('丅ㄒТＴχㄨ〤Х亅Ｈ郃匼閤磼雧粭峆鏶潗')
            hit = sum(1 for c in stripped if c in watermark_chars)
            # 如果一行中有 3+ 个水印特征字符且行长 <60，基本确定是水印
            if hit >= 3 and len(stripped) < 60:
                continue
            # 也处理 "ＴＸＴ合集网独家整理" 和 "www.TXTHJ.com"
            if 'ＴＸＴ合集' in stripped or 'TXTHJ' in stripped.upper():
                continue
            clean_lines.append(line)
        raw = "\n".join(clean_lines)

        raw = raw.strip()

        if len(raw) < 100:
            continue

        # 提取书名（从文件名）
        book_name = txt_file.stem
        # e.g. "刘慈欣01.三体I：地球往事" → "三体I：地球往事"
        match = re.match(r"刘慈欣\d+\.(.*)", book_name)
        title = match.group(1) if match else book_name

        # 按单行换行分段（这些 txt 文件用 \n 而非 \n\n 分段）
        lines = [l.strip() for l in raw.split("\n") if l.strip()]

        # 跳过开头的网站水印行
        while lines and ("txthj" in lines[0].lower() or "www." in lines[0].lower()
                         or "ＴＸＴ" in lines[0] or "txt" in lines[0].lower()):
            lines.pop(0)

        # 按整行累积切成 chunk（不做 overlap，避免断句）
        chunks = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > chunk_chars and current:
                chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line

        if current:
            chunks.append(current)

        for chunk in chunks:
            if len(chunk) < 50:  # 跳过太短的
                continue
            results.append({"text": chunk})

        print(f"    {title}: {len(raw)} chars → {len([c for c in chunks if len(c)>=50])} 段")

    results = dedup_by_text(results)
    print(f"    总计去重后: {len(results)} 段")
    return results


# ============================================================
# 2. Belle 中文多轮对话
# ============================================================

def fetch_belle(max_samples: int = 10000) -> list[dict]:
    from datasets import load_dataset

    print(f"  拉取 BelleGroup/multiturn_chat_0.8M (采样 {max_samples} 条) ...")
    ds = load_dataset("BelleGroup/multiturn_chat_0.8M", split="train", streaming=True)

    results = []
    for i, item in enumerate(ds):
        if i >= max_samples * 3:  # 多拉一些再筛
            break

        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()

        if not instruction or not output:
            continue

        # instruction 里已经是多轮对话格式: Human: ... Assistant: ... Human: ...
        # output 是最后一轮 assistant 的回复
        # 拼成完整对话
        text = instruction.strip()
        if not text.endswith("\n"):
            text += "\n"
        text += f"Assistant: {output}"

        # 质量过滤：太短的跳过
        if len(text) < 100:
            continue
        # 太长的也跳过（避免超出 max_seq_len）
        if len(text) > 2000:
            continue

        results.append({"text": text})

        if len(results) >= max_samples:
            break

    print(f"    Belle: {len(results)} 条")
    results = dedup_by_text(results)
    print(f"    去重后: {len(results)} 条")
    return results


# ============================================================
# main
# ============================================================

def main():
    print("=" * 60)
    print("中文数据集拉取与处理")
    print("=" * 60)

    # 1. 刘慈欣科幻
    print("\n[1/2] 刘慈欣科幻小说")
    scifi = build_scifi()
    write_jsonl(SYNTHETIC_DIR / "chinese_scifi.jsonl", scifi)

    # 2. Belle 中文多轮对话
    print("\n[2/2] Belle 中文多轮对话")
    belle = fetch_belle(max_samples=10000)
    write_jsonl(SYNTHETIC_DIR / "chinese_dialog.jsonl", belle)

    # 汇总
    print("\n" + "=" * 60)
    print("中文数据汇总")
    print("=" * 60)
    print(f"  科幻小说:    {len(scifi):>6} 段")
    print(f"  中文对话:    {len(belle):>6} 条")
    print(f"  总计:        {len(scifi)+len(belle):>6} 条")


if __name__ == "__main__":
    main()
