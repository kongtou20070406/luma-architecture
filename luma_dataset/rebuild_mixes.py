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

    # ── DataMix v2: 实验用 mix (M4+ 实验) ─────────────────

    print(f"\n生成 DataMix v2 实验数据集 ...")

    # persona + wechat 内容复制 2x（1 epoch = 2 epoch 人格曝光）
    persona_2x = (persona + wechat_sft) * 2  # ~176K 条

    # 加载 oasst1 和 ultrafeedback（v2 恢复使用）
    oasst = load_jsonl(SYNTHETIC_DIR / "oasst1.jsonl")
    ultrafeedback = load_jsonl(SYNTHETIC_DIR / "ultrafeedback.jsonl")

    # v2 配比：persona+wechat ≥25%, smart ≥40%, 叙事+对话 ≤35%
    # 总量以 persona_2x 为锚点，倒推其他桶的采样量
    n_persona = len(persona_2x)                            # ~176K (25%)
    total_target = int(n_persona / 0.25)                   # ~704K
    n_math = int(total_target * 0.15)                      # 15% math+arc
    n_code = int(total_target * 0.10)                      # 10% code
    n_reason = int(total_target * 0.15)                    # 15% oasst+ultrafeedback
    n_zhihu = int(total_target * 0.15)                     # 15% zhihu
    n_scifi = int(total_target * 0.05)                     # 5% scifi
    n_arc = int(total_target * 0.05)                       # 5% arc
    # 剩余 ~10% 由 persona_2x 多出部分自然填充

    v2_mix = []
    v2_mix.extend(persona_2x)
    v2_mix.extend(sample_or_oversample(math, n_math))
    v2_mix.extend(sample_or_oversample(python_code + stack_python, n_code))
    v2_mix.extend(sample_or_oversample(oasst + ultrafeedback, n_reason))
    v2_mix.extend(sample_or_oversample(zhihu, n_zhihu))
    v2_mix.extend(sample_or_oversample(scifi, n_scifi))
    v2_mix.extend(sample_or_oversample(arc, n_arc))

    random.shuffle(v2_mix)

    # 统计实际配比
    total_v2 = len(v2_mix)
    persona_pct = len(persona_2x) / total_v2 * 100
    print(f"\n  DataMix v2 总量: {total_v2} 条")
    print(f"  persona+wechat (2x): {len(persona_2x)} ({persona_pct:.1f}%)")
    print(f"  math:                {n_math}")
    print(f"  code:                {n_code}")
    print(f"  reason (oasst+uf):   {n_reason}")
    print(f"  zhihu:               {n_zhihu}")
    print(f"  scifi:               {n_scifi}")
    print(f"  arc:                 {n_arc}")

    write_jsonl(MIXES_DIR / "pretrain_v2.jsonl", v2_mix)
    ensure_symlink("pretrain_v2.jsonl", MIXES_DIR / "pretrain_v2.jsonl")

    # ── DataMix v3: 正式预训练 (50% 聪明 + 25% 性格 + 25% 叙事) ──

    print(f"\n生成 DataMix v3 正式预训练数据集 ...")

    # 加载新增数据源
    oasst = load_jsonl(SYNTHETIC_DIR / "oasst1.jsonl")
    ultrafeedback = load_jsonl(SYNTHETIC_DIR / "ultrafeedback.jsonl")
    novel_game = load_jsonl(SYNTHETIC_DIR / "novel_game_real.jsonl")
    novel_dawn = load_jsonl(SYNTHETIC_DIR / "novel_dawn_sword.jsonl")
    textbook_la = load_jsonl(SYNTHETIC_DIR / "textbook_linear_algebra.jsonl")

    # 性格桶: persona + wechat 3x 复制（短文本高频曝光 → 烙印风格）
    persona_bucket = (persona + wechat_sft) * 3

    # 聪明桶 (50%): 以性格桶样本数为锚点反推
    n_persona = len(persona_bucket)
    total_target = int(n_persona / 0.25)  # persona=25% → total
    n_smart = int(total_target * 0.50)
    n_narrative = int(total_target * 0.25)

    # 聪明桶内部配比: math 20% + code 15% + oasst+uf 10% + arc 3% + textbook 2%
    smart_pool = math + python_code + arc + oasst + ultrafeedback + textbook_la
    smart_bucket = sample_or_oversample(smart_pool, n_smart)

    # 叙事桶: 网文 + 科幻 + 知乎（下采样知乎，突出网文风格）
    n_novels = len(novel_game) + len(novel_dawn) + len(scifi)  # ~5.3K
    n_zhihu_v3 = max(n_narrative - n_novels, 0)
    narrative_bucket = novel_game + novel_dawn + scifi + sample_no_oversample(zhihu, n_zhihu_v3)
    # 如果叙事桶不够，上采样网文
    if len(narrative_bucket) < n_narrative:
        narrative_bucket = sample_or_oversample(narrative_bucket, n_narrative)

    v3_mix = persona_bucket + smart_bucket + narrative_bucket
    random.shuffle(v3_mix)

    # 统计
    total_v3 = len(v3_mix)
    v3_tokens = estimate_tokens(v3_mix)
    print(f"\n  DataMix v3 总量: {total_v3} 条, ~{v3_tokens/1e6:.0f}M tokens")
    print(f"  性格桶 (persona+wechat 3x): {n_persona} ({n_persona/total_v3*100:.1f}%)")
    print(f"  聪明桶 (math+code+arc+oasst+uf+教材): {len(smart_bucket)} ({len(smart_bucket)/total_v3*100:.1f}%)")
    print(f"  叙事桶 (网文+科幻+知乎): {len(narrative_bucket)} ({len(narrative_bucket)/total_v3*100:.1f}%)")

    write_jsonl(MIXES_DIR / "pretrain_v3.jsonl", v3_mix)
    ensure_symlink("pretrain_v3.jsonl", MIXES_DIR / "pretrain_v3.jsonl")

    # ── DataMix v4: v3 + smart 桶补充 (arxiv_dl_code + numina + platypus + gsm8k) ──

    print(f"\n生成 DataMix v4 (v3 基础 + smart 补充) ...")

    # 加载 v4 新增数据源
    arxiv_dl = load_jsonl(SYNTHETIC_DIR / "arxiv_dl_code.jsonl")
    numina_cot = load_jsonl(SYNTHETIC_DIR / "numina_math_cot.jsonl")
    platypus = load_jsonl(SYNTHETIC_DIR / "open_platypus.jsonl")
    gsm8k = load_jsonl(SYNTHETIC_DIR / "gsm8k.jsonl")

    # 性格桶: 同 v3 (persona + wechat 3x)
    v4_persona_bucket = (persona + wechat_sft) * 3
    n_v4_persona = len(v4_persona_bucket)

    # 以性格桶为锚点，保持 50/25/25
    v4_total_target = int(n_v4_persona / 0.25)
    n_v4_smart = int(v4_total_target * 0.50)
    n_v4_narrative = int(v4_total_target * 0.25)

    # 聪明桶: v3 原有 + 新增 4 个数据源（不上采样，池子够大直接采样）
    v4_smart_pool = (math + python_code + arc + oasst + ultrafeedback
                     + textbook_la + arxiv_dl + numina_cot + platypus + gsm8k)
    v4_smart_bucket = sample_or_oversample(v4_smart_pool, n_v4_smart)

    # 叙事桶: 同 v3
    n_v4_novels = len(novel_game) + len(novel_dawn) + len(scifi)
    n_v4_zhihu = max(n_v4_narrative - n_v4_novels, 0)
    v4_narrative_bucket = novel_game + novel_dawn + scifi + sample_no_oversample(zhihu, n_v4_zhihu)
    if len(v4_narrative_bucket) < n_v4_narrative:
        v4_narrative_bucket = sample_or_oversample(v4_narrative_bucket, n_v4_narrative)

    v4_mix = v4_persona_bucket + v4_smart_bucket + v4_narrative_bucket
    random.shuffle(v4_mix)

    total_v4 = len(v4_mix)
    v4_tokens = estimate_tokens(v4_mix)
    print(f"\n  DataMix v4 总量: {total_v4} 条, ~{v4_tokens/1e6:.0f}M tokens")
    print(f"  性格桶 (persona+wechat 3x): {n_v4_persona} ({n_v4_persona/total_v4*100:.1f}%)")
    print(f"  聪明桶 (+arxiv_dl+numina+platypus+gsm8k): {len(v4_smart_bucket)} ({len(v4_smart_bucket)/total_v4*100:.1f}%)")
    print(f"  叙事桶 (网文+科幻+知乎): {len(v4_narrative_bucket)} ({len(v4_narrative_bucket)/total_v4*100:.1f}%)")
    print(f"  聪明桶池子大小: {len(v4_smart_pool)} (新增 arxiv_dl={len(arxiv_dl)} numina={len(numina_cot)} platypus={len(platypus)} gsm8k={len(gsm8k)})")

    write_jsonl(MIXES_DIR / "pretrain_v4.jsonl", v4_mix)
    ensure_symlink("pretrain_v4.jsonl", MIXES_DIR / "pretrain_v4.jsonl")

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
