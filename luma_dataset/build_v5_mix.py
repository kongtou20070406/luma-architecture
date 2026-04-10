"""
v5 数据集混合脚本
按比例从各数据源采样，输出单个 shuffled jsonl

目标配比:
  聪明 55%: 数学 + 代码 + 推理
  知识 25%: Wikipedia + 知乎
  思维 10%: 毛选 + 科幻 + 小说
  对话  9%: ultrafeedback + oasst1
  人格  1%: persona×6 + wechat×6
"""
import json
import os
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "synthetic"
OUT_DIR = Path(__file__).parent / "mixes"
OUT_DIR.mkdir(exist_ok=True)

# 每个数据源: (文件名, 最大采样行数 或 None=全量, 类别)
SOURCES = {
    # ═══ 聪明 55% ═══
    # 数学
    "openr1_math_hard": (None, "smart"),
    "numina_math_cot": (None, "smart"),
    "openmath_cot": (None, "smart"),
    "metamathqa": (None, "smart"),
    "ultradata_math_l3": (None, "smart"),
    "deeptheorem": (None, "smart"),
    "math_competition_hard": (None, "smart"),
    "math_real": (None, "smart"),
    "gsm8k": (None, "smart"),
    "textbook_linear_algebra": (None, "smart"),
    # 代码
    "python_code": (None, "smart"),
    "arxiv_dl_code": (None, "smart"),
    # 推理
    "open_platypus": (None, "smart"),
    "arc_agi": (None, "smart"),

    # ═══ 知识 25% ═══
    "chinese_wikipedia": (None, "knowledge"),  # 会按比例下采样
    "zhihu_kol": (None, "knowledge"),  # 会按比例下采样

    # ═══ 思维 10% ═══
    "mao_selected_works": (None, "thinking"),
    "chinese_scifi": (None, "thinking"),
    "novel_dawn_sword": (None, "thinking"),
    "novel_game_real": (None, "thinking"),

    # ═══ 对话 9% ═══
    "ultrafeedback": (None, "dialogue"),
    "oasst1": (None, "dialogue"),

    # ═══ 人格 1% ═══
    "persona_private_x6": (None, "persona"),
    "wechat_sft_x6": (None, "persona"),
}

# 不要用 openr1_math_hard_2k/3k/4k (是 openr1_math_hard 的子集)
EXCLUDE = {"openr1_math_hard_2k", "openr1_math_hard_3k", "openr1_math_hard_4k",
           "persona_private", "wechat_sft",  # 用 x6 版本
           "chinese_math_cot",  # 空文件
           "ultradata_math",  # 旧版，用 l3
           "swallow_code", "swallow_math",  # 还没拉到
           }


def estimate_tokens(text: str) -> int:
    """粗估 token 数：中文 ~0.7 token/char, 英文 ~0.25 token/char"""
    cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = len(text) - cn
    return int(cn * 0.7 + en * 0.25)


def load_source(name: str) -> list:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"  SKIP: {name} (not found)")
        return []
    lines = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", ""))
                # persona 短消息也要保留
                min_len = 10 if "persona" in name or "wechat" in name else 50
                if len(text) >= min_len:
                    lines.append(line)
            except json.JSONDecodeError:
                continue
    return lines


def main():
    # 1. 加载所有数据源并统计 token
    category_data = {"smart": [], "knowledge": [], "thinking": [], "dialogue": [], "persona": []}
    category_tokens = {"smart": 0, "knowledge": 0, "thinking": 0, "dialogue": 0, "persona": 0}

    print("Loading sources...")
    for name, (max_lines, cat) in SOURCES.items():
        if name in EXCLUDE:
            continue
        lines = load_source(name)
        if max_lines and len(lines) > max_lines:
            random.shuffle(lines)
            lines = lines[:max_lines]
        # 估算 token
        sample = lines[:min(100, len(lines))]
        if sample:
            avg_tokens = sum(estimate_tokens(json.loads(l).get("text", "")) for l in sample) / len(sample)
            total_tokens = avg_tokens * len(lines)
        else:
            total_tokens = 0
        category_data[cat].extend(lines)
        category_tokens[cat] += total_tokens
        print(f"  {name}: {len(lines)} lines, ~{total_tokens/1e6:.1f}M tokens → {cat}")

    # 2. 计算目标比例
    TARGET_RATIO = {"smart": 0.55, "knowledge": 0.25, "thinking": 0.10, "dialogue": 0.09, "persona": 0.01}

    # 固定总量目标，各类别按比例分配
    # 如果某类别不够就全量用，多出的配额分给 knowledge（中文最多）
    total_target = 300_000_000  # 300M tokens 目标

    print(f"\n目标总量: {total_target/1e6:.0f}M tokens")

    # 3. 按比例下采样其他类别
    final_lines = []
    for cat in ["smart", "knowledge", "thinking", "dialogue", "persona"]:
        target_tokens = total_target * TARGET_RATIO[cat]
        current_tokens = category_tokens[cat]
        data = category_data[cat]

        if current_tokens <= target_tokens:
            # 不够，全量使用
            final_lines.extend(data)
            actual_ratio = current_tokens / total_target * 100
            print(f"  {cat}: 全量 {len(data)} lines, {current_tokens/1e6:.0f}M tokens ({actual_ratio:.1f}% 不足目标 {TARGET_RATIO[cat]*100:.0f}%)")
        else:
            # 过多，按比例采样
            sample_ratio = target_tokens / current_tokens
            random.shuffle(data)
            n = int(len(data) * sample_ratio)
            sampled = data[:n]
            final_lines.extend(sampled)
            print(f"  {cat}: 采样 {n}/{len(data)} lines, {target_tokens/1e6:.0f}M tokens ({TARGET_RATIO[cat]*100:.0f}%)")

    # 4. Shuffle 并写入
    print(f"\n总计: {len(final_lines)} lines")
    random.shuffle(final_lines)

    out_path = OUT_DIR / "v5_pretrain.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(line.strip() + '\n')

    # 5. 统计
    size_mb = out_path.stat().st_size / 1e6
    tokens_est = size_mb * 0.6 / 4 * 1e6  # 粗估
    print(f"\n输出: {out_path}")
    print(f"大小: {size_mb:.0f}MB, ~{tokens_est/1e6:.0f}M tokens")
    print(f"行数: {len(final_lines)}")

    # 6. 验证中文比例
    cn_count = en_count = 0
    for line in random.sample(final_lines, min(1000, len(final_lines))):
        text = json.loads(line).get("text", "")[:500]
        for c in text:
            if '\u4e00' <= c <= '\u9fff':
                cn_count += 1
            elif c.isascii() and c.isalpha():
                en_count += 1
    cn_pct = cn_count / max(cn_count + en_count, 1) * 100
    print(f"中文比例 (采样): {cn_pct:.0f}%")


if __name__ == "__main__":
    main()
