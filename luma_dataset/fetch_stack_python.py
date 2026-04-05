#!/usr/bin/env python3
"""
从 bigcode/the-stack 拉取 Python 子集，过滤后输出 stack_python.jsonl。

过滤条件：
  - 仅 permissive license (mit, apache-2.0, bsd-2-clause, bsd-3-clause)
  - 代码长度 200-8000 chars（太短无意义，太长超 seq）
  - alphanum_fraction > 0.25（过滤二进制/数据文件）
  - max_stars_count >= 2（基本质量过滤）
  - 目标：~15000 条高质量 Python 代码

运行: python fetch_stack_python.py
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path(__file__).resolve().parent / "synthetic" / "stack_python.jsonl"

PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"}
MIN_LEN = 200
MAX_LEN = 8000
MIN_ALPHANUM = 0.25
MIN_STARS = 2
TARGET_COUNT = 15000

def is_good(sample: dict) -> bool:
    content = sample.get("content", "")
    if not (MIN_LEN <= len(content) <= MAX_LEN):
        return False
    if sample.get("alphanum_fraction", 0) < MIN_ALPHANUM:
        return False
    # License check
    licenses = sample.get("max_stars_repo_licenses", [])
    if not any(lic in PERMISSIVE for lic in licenses):
        return False
    # Stars check
    stars = sample.get("max_stars_count", 0)
    if stars is None or stars < MIN_STARS:
        return False
    return True


def main():
    print(f"从 bigcode/the-stack 拉取 Python 代码 ...")
    print(f"目标: {TARGET_COUNT} 条, 过滤: permissive license, {MIN_LEN}-{MAX_LEN} chars, stars>={MIN_STARS}")

    ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)

    collected = []
    scanned = 0

    try:
        for sample in ds:
            scanned += 1
            if is_good(sample):
                collected.append({
                    "text": sample["content"],
                    "source": "the-stack-python",
                    "repo": sample.get("max_stars_repo_name", ""),
                    "path": sample.get("max_stars_repo_path", ""),
                    "stars": sample.get("max_stars_count", 0),
                })

                if len(collected) % 1000 == 0:
                    print(f"  已收集 {len(collected)}/{TARGET_COUNT} (扫描 {scanned}, 通过率 {len(collected)/scanned*100:.1f}%)")

                if len(collected) >= TARGET_COUNT:
                    break

            if scanned % 100000 == 0:
                print(f"  扫描 {scanned}, 收集 {len(collected)} ...")

    except KeyboardInterrupt:
        print(f"\n中断! 保存已收集的 {len(collected)} 条")

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in collected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_chars = sum(len(d["text"]) for d in collected)
    print(f"\n完成! {len(collected)} 条, ~{total_chars/3.5/1e6:.1f}M tokens")
    print(f"扫描 {scanned} 条, 通过率 {len(collected)/max(scanned,1)*100:.1f}%")
    print(f"输出: {OUTPUT}")


if __name__ == "__main__":
    main()
