#!/usr/bin/env python3
"""
Luma DataMix v1 扩展数据拉取。

拉取 datamix_v1.yaml 中已启用但尚未下载的大型 HuggingFace 数据源。
目标：从 ~8M tokens 扩充到 ~300M+ tokens，让正式预训练不会每 epoch 都重复。

新增数据源:
  A. 聪明桶 (50%)
    - smart_reasoning: oasst1, ultrafeedback_binarized
    - smart_code:      the-stack Python (采样 100K)
  B. 情感桶 (20%)
    - empathetic_dialogues
  C. 对话质量桶 (15%)
    - Zhihu-KOL (采样 200K)

已有数据源 (不重新拉取):
  - math_real.jsonl (GSM8K + hendrycks_math)
  - python_code.jsonl (python_code_18k + CodeAlpaca)
  - chinese_scifi.jsonl
  - persona_private.jsonl
  - arc_agi.jsonl

运行: python fetch_expanded_datasets.py
"""

import json
import hashlib
import random
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


def filter_short(data: list[dict], min_chars: int = 20) -> list[dict]:
    """过滤过短的样本。"""
    return [d for d in data if len(d.get("text", "")) >= min_chars]


# ============================================================
# 1. OpenAssistant/oasst1 — 多轮高质量对话
# ============================================================
def fetch_oasst1() -> list[dict]:
    from datasets import load_dataset

    print("  拉取 OpenAssistant/oasst1 ...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    # 按 message tree 组织对话
    # 每条是独立 message，用 parent_id 连接
    msg_by_id = {}
    roots = []
    for item in ds:
        msg = {
            "id": item["message_id"],
            "parent_id": item["parent_id"],
            "text": item["text"],
            "role": item["role"],
            "lang": item.get("lang", ""),
        }
        msg_by_id[msg["id"]] = msg
        if msg["parent_id"] is None:
            roots.append(msg)

    # 构建对话链：从每个 leaf 回溯到 root
    children = {}
    for m in msg_by_id.values():
        if m["parent_id"]:
            children.setdefault(m["parent_id"], []).append(m)

    results = []

    def build_chains(msg_id, chain):
        chain = chain + [msg_by_id[msg_id]]
        kids = children.get(msg_id, [])
        if not kids:
            # leaf — 输出这条完整对话链
            if len(chain) >= 2:
                turns = []
                for m in chain:
                    role = "user" if m["role"] == "prompter" else "assistant"
                    turns.append(f"{role}: {m['text']}")
                text = "\n".join(turns)
                results.append({"text": text})
        else:
            for kid in kids:
                build_chains(kid["id"], chain)

    for root in roots:
        build_chains(root["id"], [])

    print(f"    oasst1 对话链: {len(results)} 条")
    results = dedup_by_text(results)
    results = filter_short(results, 50)
    print(f"    去重+过滤后: {len(results)} 条")
    return results


# ============================================================
# 2. ultrafeedback_binarized — 回答质量偏好
# ============================================================
def fetch_ultrafeedback() -> list[dict]:
    from datasets import load_dataset

    print("  拉取 HuggingFaceH4/ultrafeedback_binarized ...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    results = []
    for item in ds:
        # 用 chosen response（高质量回答）
        prompt = item["prompt"]
        chosen = item["chosen"]
        if isinstance(chosen, list):
            # chat format: [{"role": ..., "content": ...}, ...]
            turns = []
            for turn in chosen:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                turns.append(f"{role}: {content}")
            text = "\n".join(turns)
        elif isinstance(chosen, str):
            text = f"user: {prompt}\nassistant: {chosen}"
        else:
            continue
        results.append({"text": text})

    print(f"    ultrafeedback: {len(results)} 条")
    results = dedup_by_text(results)
    results = filter_short(results, 50)
    print(f"    去重+过滤后: {len(results)} 条")
    return results


# ============================================================
# 3. empathetic_dialogues — 共情对话
# ============================================================
def fetch_empathetic_dialogues() -> list[dict]:
    from datasets import load_dataset

    print("  拉取 facebook/empathetic_dialogues ...")
    ds = load_dataset("facebook/empathetic_dialogues", split="train", trust_remote_code=True)

    # 按 conv_id 分组
    convos = {}
    for item in ds:
        cid = item["conv_id"]
        if cid not in convos:
            convos[cid] = {
                "context": item.get("context", ""),
                "turns": [],
            }
        speaker = "listener" if item["speaker_idx"] == 1 else "speaker"
        convos[cid]["turns"].append((speaker, item["utterance"]))

    results = []
    for cid, conv in convos.items():
        if len(conv["turns"]) < 2:
            continue
        parts = []
        if conv["context"]:
            parts.append(f"[situation: {conv['context']}]")
        for speaker, utt in conv["turns"]:
            # 清理 _comma_ 等 artifact
            utt = utt.replace("_comma_", ",").replace("_pipe_", "|")
            parts.append(f"{speaker}: {utt}")
        text = "\n".join(parts)
        results.append({"text": text})

    print(f"    empathetic_dialogues: {len(results)} 条")
    results = dedup_by_text(results)
    results = filter_short(results, 50)
    print(f"    去重+过滤后: {len(results)} 条")
    return results


# ============================================================
# 4. Zhihu-KOL — 中文长篇高质量回答
# ============================================================
def fetch_zhihu_kol(max_samples: int = 200_000) -> list[dict]:
    from datasets import load_dataset

    print(f"  拉取 wangrui6/Zhihu-KOL (上限 {max_samples} 条) ...")
    ds = load_dataset("wangrui6/Zhihu-KOL", split="train", streaming=True)

    results = []
    for i, item in enumerate(ds):
        if i >= max_samples * 2:  # 多拉一些再过滤
            break
        # Zhihu-KOL: INSTRUCTION (问题) + RESPONSE (回答)
        q = (item.get("INSTRUCTION") or "").strip()
        a = (item.get("RESPONSE") or "").strip()
        if not a or len(a) < 50:
            continue
        text = f"问：{q}\n答：{a}" if q else a
        results.append({"text": text})

    print(f"    Zhihu-KOL 拉取: {len(results)} 条")
    # 采样到目标数量
    if len(results) > max_samples:
        results = random.sample(results, max_samples)
    results = dedup_by_text(results)
    results = filter_short(results, 100)
    print(f"    去重+过滤后: {len(results)} 条")
    return results


# ============================================================
# 5. the-stack Python — 真实 Python 代码
# ============================================================
def fetch_stack_python(max_samples: int = 100_000) -> list[dict]:
    from datasets import load_dataset

    print(f"  拉取 bigcode/the-stack Python 子集 (上限 {max_samples} 条) ...")
    # the-stack v1 uses data_dir for language filtering
    try:
        ds = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"    ⚠️ the-stack-dedup 加载失败: {e}")
        print("    尝试 bigcode/starcoderdata ...")
        try:
            ds = load_dataset(
                "bigcode/starcoderdata",
                data_dir="python",
                split="train",
                streaming=True,
            )
        except Exception as e2:
            print(f"    ⚠️ starcoderdata 也失败: {e2}")
            return []

    results = []
    for i, item in enumerate(ds):
        if len(results) >= max_samples:
            break
        content = item.get("content", "")
        if not content or len(content) < 100:
            continue
        # 过滤非 Python 或质量过低的
        if "def " not in content and "class " not in content and "import " not in content:
            continue
        # 截断超长文件（保留前 8000 chars ≈ 2000 tokens）
        if len(content) > 8000:
            # 在函数/类边界截断
            cut = content[:8000]
            last_def = max(cut.rfind("\ndef "), cut.rfind("\nclass "))
            if last_def > 4000:
                content = cut[:last_def]
            else:
                content = cut
        results.append({"text": content})

    print(f"    the-stack Python: {len(results)} 条")
    results = dedup_by_text(results)
    print(f"    去重后: {len(results)} 条")
    return results


# ============================================================
# 6. wechat_sft — 真实微信对话 (SFT 格式转 pretrain)
# ============================================================
def convert_wechat_sft() -> list[dict]:
    sft_path = Path("/home/kt/ai/wechat_sft.jsonl")
    if not sft_path.exists():
        print("    ⚠️ wechat_sft.jsonl 不存在")
        return []

    print(f"  转换 wechat_sft.jsonl ...")
    results = []
    with open(sft_path) as f:
        for line in f:
            d = json.loads(line)
            messages = d.get("messages", [])
            if len(messages) < 2:
                continue
            turns = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if content:
                    turns.append(f"{role}: {content}")
            text = "\n".join(turns)
            if len(text) >= 10:  # 过滤过短
                results.append({"text": text})

    results = dedup_by_text(results)
    print(f"    wechat_sft: {len(results)} 条")
    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Luma DataMix v1 扩展数据拉取")
    print("=" * 60)

    # 0. wechat_sft (本地，不需要网络)
    print("\n[0/5] wechat_sft (本地)")
    wechat_sft = convert_wechat_sft()
    write_jsonl(SYNTHETIC_DIR / "wechat_sft.jsonl", wechat_sft)

    # 1. oasst1
    print("\n[1/5] OpenAssistant/oasst1")
    oasst = fetch_oasst1()
    write_jsonl(SYNTHETIC_DIR / "oasst1.jsonl", oasst)

    # 2. ultrafeedback
    print("\n[2/5] ultrafeedback_binarized")
    uf = fetch_ultrafeedback()
    write_jsonl(SYNTHETIC_DIR / "ultrafeedback.jsonl", uf)

    # 3. empathetic_dialogues
    print("\n[3/5] empathetic_dialogues")
    ed = fetch_empathetic_dialogues()
    write_jsonl(SYNTHETIC_DIR / "empathetic_dialogues.jsonl", ed)

    # 4. Zhihu-KOL
    print("\n[4/5] Zhihu-KOL")
    zhihu = fetch_zhihu_kol(max_samples=200_000)
    write_jsonl(SYNTHETIC_DIR / "zhihu_kol.jsonl", zhihu)

    # 5. the-stack Python
    print("\n[5/5] the-stack Python")
    stack = fetch_stack_python(max_samples=100_000)
    if stack:
        write_jsonl(SYNTHETIC_DIR / "stack_python.jsonl", stack)

    # 汇总
    print("\n" + "=" * 60)
    print("扩展数据拉取汇总")
    print("=" * 60)

    total_chars = 0
    for name, data in [("wechat_sft", wechat_sft), ("oasst1", oasst),
                       ("ultrafeedback", uf), ("empathetic_dialogues", ed),
                       ("zhihu_kol", zhihu), ("stack_python", stack)]:
        chars = sum(len(d["text"]) for d in data)
        tokens_est = chars / 3.5
        total_chars += chars
        print(f"  {name:<25} {len(data):>8} 条  ~{tokens_est/1e6:.1f}M tokens")

    print(f"  {'新增总计':<25} ~{total_chars/3.5/1e6:.0f}M tokens")

    # 加上已有数据
    existing = 0
    for fname in ["math_real.jsonl", "python_code.jsonl", "chinese_scifi.jsonl",
                   "persona_private.jsonl", "arc_agi.jsonl"]:
        p = SYNTHETIC_DIR / fname
        if p.exists():
            with open(p) as f:
                for line in f:
                    d = json.loads(line)
                    existing += len(d.get("text", ""))
    print(f"  {'已有数据':<25} ~{existing/3.5/1e6:.0f}M tokens")
    print(f"  {'总计':<25} ~{(total_chars + existing)/3.5/1e6:.0f}M tokens")


if __name__ == "__main__":
    main()
