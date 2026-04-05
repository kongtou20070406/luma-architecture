#!/usr/bin/env python3
"""
从 AlgorithmicResearchGroup/arxiv_deep_learning_python_research_code
拉取深度学习研究代码，过滤后输出 arxiv_dl_code.jsonl。

过滤条件：
  - 包含 ML/DL 关键 import (torch, tensorflow, sklearn, jax, keras)
  - 代码长度 200-8000 chars
  - alphanum_fraction > 0.25
  - 目标：~20000 条高质量 ML/DL Python 代码

运行: python fetch_arxiv_dl_code.py
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path(__file__).resolve().parent / "synthetic" / "arxiv_dl_code.jsonl"

ML_KEYWORDS = [
    "import torch", "from torch",
    "import tensorflow", "from tensorflow",
    "import sklearn", "from sklearn",
    "import jax", "from jax",
    "import keras", "from keras",
    "import numpy", "from numpy",
    "import gym", "from gym",
    "import transformers", "from transformers",
]

MIN_LEN = 200
MAX_LEN = 8000
MIN_ALPHANUM = 0.25
TARGET_COUNT = 20000


def is_good(code: str) -> bool:
    if not (MIN_LEN <= len(code) <= MAX_LEN):
        return False
    alphanum = sum(c.isalnum() for c in code) / max(len(code), 1)
    if alphanum < MIN_ALPHANUM:
        return False
    if not any(kw in code for kw in ML_KEYWORDS):
        return False
    return True


def main():
    print(f"从 arxiv_deep_learning_python_research_code 拉取 ML/DL 代码 ...")
    print(f"目标: {TARGET_COUNT} 条, 过滤: ML/DL imports, {MIN_LEN}-{MAX_LEN} chars")

    ds = load_dataset(
        "AlgorithmicResearchGroup/arxiv_deep_learning_python_research_code",
        split="train",
        streaming=True,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    scanned = 0

    with open(OUTPUT, "w", encoding="utf-8") as fout:
        for sample in ds:
            scanned += 1
            code = sample.get("code", "")
            if not is_good(code):
                continue

            record = {
                "text": code,
                "source": "arxiv_dl_code",
                "repo": sample.get("repo", ""),
                "file": sample.get("file", ""),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 2000 == 0:
                print(f"  kept {kept}/{scanned} scanned ...", flush=True)

            if kept >= TARGET_COUNT:
                break

    print(f"完成: {kept} 条 (scanned {scanned}), 保存到 {OUTPUT}")
    print(f"文件大小: {OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
