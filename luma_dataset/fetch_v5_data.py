"""
v5 数据集拉取脚本
目标: 500M+ tokens, 中文 ≥ 55%

Phase 1: 中文 Wikipedia + 毛选
Phase 2: 中文数学讲义 (BELLE/Firefly)
Phase 3: 代码补充 (The Stack Python)
"""
import os
import json
import random
from pathlib import Path

# 使用 HF 镜像（国内网络）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

OUT_DIR = Path(__file__).parent / "synthetic"
OUT_DIR.mkdir(exist_ok=True)


def fetch_chinese_wikipedia():
    """拉取中文 Wikipedia，清洗后输出 jsonl"""
    from datasets import load_dataset

    out_path = OUT_DIR / "chinese_wikipedia.jsonl"
    if out_path.exists():
        print(f"SKIP: {out_path} already exists ({sum(1 for _ in open(out_path))} lines)")
        return

    print("Fetching Chinese Wikipedia (常识 + 计算机/数学/物理)...")
    # wikimedia/wikipedia 新格式
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
    except Exception:
        try:
            ds = load_dataset("wikipedia", "20220301.zh", split="train")
        except Exception:
            ds = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train")

    # 计算机/数学/物理相关关键词 — 这些不受长度限制
    STEM_KEYWORDS = [
        # 计算机
        "算法", "编程", "计算机", "软件", "程序", "数据结构", "操作系统", "网络",
        "人工智能", "机器学习", "深度学习", "神经网络", "自然语言", "计算",
        "Python", "Linux", "CPU", "GPU", "API", "数据库",
        # 数学
        "数学", "代数", "几何", "拓扑", "微积分", "线性", "矩阵", "向量",
        "概率", "统计", "函数", "方程", "定理", "证明", "群论", "集合",
        "微分", "积分", "级数", "空间", "映射", "变换",
        # 物理
        "物理", "量子", "相对论", "力学", "电磁", "热力学", "光学", "粒子",
        "引力", "黑洞", "宇宙", "暗物质", "暗能量", "弦理论", "熵",
        "波函数", "薛定谔", "费曼", "爱因斯坦", "牛顿",
    ]

    count = 0
    stem_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item.get("text", "")
            title = item.get("title", "")
            # 过滤消歧义、列表、模板
            if any(kw in title for kw in ["消歧义", "列表", "年表", "模板:", "维基", "档案"]):
                continue
            # 判断是否 STEM 相关
            is_stem = any(kw in title or kw in text[:500] for kw in STEM_KEYWORDS)
            if is_stem:
                # STEM: 放宽长度限制，只过滤太短的
                if len(text) < 300:
                    continue
                stem_count += 1
            else:
                # 非 STEM: 常识级别，过滤太短和太长
                if len(text) < 500 or len(text) > 15000:
                    continue
            f.write(json.dumps({"text": text, "source": "zh_wikipedia", "title": title}, ensure_ascii=False) + "\n")
            count += 1

    print(f"Chinese Wikipedia: {count} articles (STEM: {stem_count}) → {out_path}")


def fetch_mao_selected_works():
    """毛选全文，公开文本"""
    from datasets import load_dataset

    out_path = OUT_DIR / "mao_selected_works.jsonl"
    if out_path.exists():
        print(f"SKIP: {out_path} already exists")
        return

    print("Fetching 毛选...")
    # 尝试从 HuggingFace 拉取
    try:
        ds = load_dataset("bingwork/mao_selected_works", split="train", trust_remote_code=True)
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds:
                text = item.get("text", item.get("content", ""))
                if len(text) < 100:
                    continue
                f.write(json.dumps({"text": text, "source": "mao_selected_works"}, ensure_ascii=False) + "\n")
                count += 1
        print(f"毛选: {count} passages → {out_path}")
    except Exception as e:
        print(f"毛选 HF 下载失败: {e}")
        print("请手动准备毛选文本放到 synthetic/mao_selected_works.jsonl")


def fetch_chinese_math():
    """中文数学数据 — BELLE + Firefly 数学子集"""
    from datasets import load_dataset

    out_path = OUT_DIR / "chinese_math_cot.jsonl"
    if out_path.exists():
        print(f"SKIP: {out_path} already exists")
        return

    print("Fetching Chinese math CoT...")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        # BELLE 数学
        try:
            ds = load_dataset("BelleGroup/school_math_0.25M", split="train", trust_remote_code=True)
            for item in ds:
                instruction = item.get("instruction", "")
                output = item.get("output", "")
                text = f"问题：{instruction}\n\n解答：{output}"
                if len(text) < 50:
                    continue
                f.write(json.dumps({"text": text, "source": "belle_math"}, ensure_ascii=False) + "\n")
                count += 1
            print(f"  BELLE math: {count} items")
        except Exception as e:
            print(f"  BELLE math failed: {e}")

        # Firefly 数学子集
        try:
            ds = load_dataset("YeungNLP/firefly-train-1.1M", split="train", trust_remote_code=True)
            math_count = 0
            for item in ds:
                kind = item.get("kind", "")
                if "math" not in kind.lower() and "数学" not in kind:
                    continue
                text = item.get("input", "") + "\n" + item.get("target", "")
                if len(text) < 50:
                    continue
                f.write(json.dumps({"text": text, "source": "firefly_math"}, ensure_ascii=False) + "\n")
                count += 1
                math_count += 1
            print(f"  Firefly math: {math_count} items")
        except Exception as e:
            print(f"  Firefly math failed: {e}")

    print(f"Chinese math total: {count} items → {out_path}")


def fetch_code_python():
    """补充 Python 代码 — The Stack v2 采样"""
    from datasets import load_dataset

    out_path = OUT_DIR / "python_code_extra.jsonl"
    if out_path.exists():
        print(f"SKIP: {out_path} already exists")
        return

    print("Fetching Python code (The Stack subset)...")
    try:
        ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train",
                          streaming=True, trust_remote_code=True)
        count = 0
        target = 50000  # 5 万条
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds:
                text = item.get("content", "")
                # 过滤太短或太长
                if len(text) < 200 or len(text) > 10000:
                    continue
                # 随机采样 (streaming 模式下用概率采样)
                if random.random() > 0.1:
                    continue
                f.write(json.dumps({"text": text, "source": "starcoderdata_python"}, ensure_ascii=False) + "\n")
                count += 1
                if count >= target:
                    break
                if count % 10000 == 0:
                    print(f"  Python code: {count}/{target}")
        print(f"Python code: {count} items → {out_path}")
    except Exception as e:
        print(f"Python code failed: {e}")


def fetch_chinese_math_textbook():
    """中文数学讲义 — 几何、线性代数、分析"""
    from datasets import load_dataset

    out_path = OUT_DIR / "chinese_math_textbook.jsonl"
    if out_path.exists():
        print(f"SKIP: {out_path} already exists")
        return

    print("Fetching Chinese math textbook data...")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        # 中文数学教材/讲义
        try:
            ds = load_dataset("math-ai/AutoMathText", "zh-web", split="train",
                              streaming=True, trust_remote_code=True)
            for item in ds:
                text = item.get("text", "")
                if len(text) < 200:
                    continue
                # 过滤非数学内容
                f.write(json.dumps({"text": text, "source": "automathtext_zh"}, ensure_ascii=False) + "\n")
                count += 1
                if count >= 30000:
                    break
                if count % 5000 == 0:
                    print(f"  AutoMathText zh: {count}")
            print(f"  AutoMathText zh: {count} items")
        except Exception as e:
            print(f"  AutoMathText zh failed: {e}")

    print(f"Chinese math textbook: {count} items → {out_path}")


if __name__ == "__main__":
    import sys

    phases = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    if "wiki" in phases or "all" in phases:
        fetch_chinese_wikipedia()

    if "mao" in phases or "all" in phases:
        fetch_mao_selected_works()

    if "math" in phases or "all" in phases:
        fetch_chinese_math()
        fetch_chinese_math_textbook()

    if "code" in phases or "all" in phases:
        fetch_code_python()

    print("\n=== Done ===")
    # 统计
    for f in sorted(OUT_DIR.glob("*.jsonl")):
        lines = sum(1 for _ in open(f))
        size_mb = f.stat().st_size / 1e6
        print(f"  {lines:>8} lines  {size_mb:>7.1f}MB  {f.name}")
