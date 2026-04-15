#!/usr/bin/env python
"""
Luma Sanity Check v1 (2026-04-15)
=================================
训练完成后的基础能力验证。不是评测，是"训练没白跑、模型学到了东西"的 sanity check。

6 个测试模块：
  1. test_prompt_loss        基础困惑度（loss < 11.9 = 比随机好）
  2. test_topk_prediction    top-k 词汇合理性
  3. test_language_isolation 中英文 token 分离度
  4. test_exit_depth         不同难度 prompt 的循环深度分布
  5. test_c_t_clustering     c_t 自发聚类（采样 + t-SNE 可选）
  6. test_c_t_injection ⭐   手动初始化 c_t 看输出差异（人格机制验证）

用法:
    python sanity_check.py --checkpoint <path> --output <json_path>
    python sanity_check.py --checkpoint <path> --tests prompt_loss,topk,lang
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# 让 scripts/ 下的脚本能 import minimind 内部模块
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "trainer"))

from model.model_minimind import LumaConfig, LumaForCausalLM
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# 测试 prompts
# ---------------------------------------------------------------------------

PROMPTS_BASIC = {
    "cn_weather": "今天天气很好，我想出去走走。",
    "cn_coding": "编程的乐趣在于",
    "cn_science": "量子力学告诉我们，",
    "en_fox": "The quick brown fox jumps over the lazy dog.",
    "en_ml": "Attention mechanism allows the model to",
    "code_py": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "math": "2 + 3 × 4 =",
    "dialogue": "A: 你好啊。\nB: 你好，",
}

PROMPTS_DIFFICULTY = {
    "easy_greeting": "你好",
    "easy_daily": "今天星期",
    "medium_concept": "什么是机器学习？",
    "medium_reason": "为什么天空是蓝色的？",
    "hard_math": "如果一个三角形三边长是 3, 4, 5，它的面积是",
    "hard_abstract": "自由意志和决定论之间的矛盾",
    "hard_trick": "一只猫有 3 条腿，它还能",
    "edge_noise": "asdfghjkl qwerty",
}

# 情绪类 prompts - 用于采集 c_t，做 PCA 得到情绪轴
PROMPTS_EMOTION = {
    "happy_1": "今天是个好日子，我感到非常",
    "happy_2": "收到礼物的那一刻，心里",
    "happy_3": "和朋友相聚总是令人",
    "sad_1": "失去亲人的痛苦让人",
    "sad_2": "分别的时候心里总是",
    "sad_3": "漫长的等待让人",
    "calm_1": "静静地坐在湖边，听风吹过",
    "calm_2": "清晨的森林里，一切都那么",
    "calm_3": "冥想时内心变得",
    "angry_1": "不公正的待遇让人",
    "angry_2": "被背叛后心中",
    "angry_3": "连续的挫折让我",
}


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cuda:0") -> tuple:
    """加载 checkpoint。返回 (model, tokenizer, config)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" in ckpt:
        cfg_args = ckpt["args"]
    else:
        cfg_args = {}

    # 重建 config（尽量匹配训练时的 arch 参数）
    cfg = LumaConfig(
        hidden_size=cfg_args.get("hidden_size", 768),
        intermediate_size=cfg_args.get("intermediate_size", 3072),
        compression_layers=cfg_args.get("compression_layers", 12),
        reason_shared_depth=cfg_args.get("reason_shared_depth", 2),
        num_attention_heads=cfg_args.get("num_attention_heads", 12),
        num_key_value_heads=cfg_args.get("num_key_value_heads", 3),
        c_t_dim=cfg_args.get("c_t_dim", 64),
        meta_dim=cfg_args.get("meta_dim", 96),
        factorized_vocab_dim=cfg_args.get("factorized_vocab_dim", 256),
        mamba_d_state=cfg_args.get("mamba_d_state", 192),
        mamba_chunk_size=cfg_args.get("mamba_chunk_size", 32),
        reason_loops=cfg_args.get("reason_loops", 4),
        # Phase 6 + stellarator
        enable_world_jepa=True,
        disable_self_jepa=False,
        enable_self_check_ring=True,
        enable_sigreg_delta=True,
        enable_energy_reason_core=True,
        phase_e_K_max=cfg_args.get("phase_e_K_max", 3),
        phase_e_eta=cfg_args.get("phase_e_eta", 0.5),
        phase_e_damped_mode=True,
        stellarator_mode=bool(cfg_args.get("stellarator_mode", True)),
        stellarator_mod_rank=cfg_args.get("stellarator_mod_rank", 8),
        loop_lora_rank=cfg_args.get("loop_lora_rank", 0),
        enable_token_depth_routing=True,
        enable_time_conditioning=True,
        introspection_input_mode="memory",
        introspection_memory_tokens=4,
        introspection_inject_mode="cmda",
        enable_neuromod_ct=True,
        neuromod_hebb_rank=32,
        neuromod_mode="jepa_surprise",
        world_jepa_mode="scaffold",
        world_jepa_weight=0.3,
        world_sigreg_weight=0.02,
        world_mask_ratio=0.4,
    )
    model = LumaForCausalLM(cfg).to(device).eval()
    # 允许非严格 load，兼容老 checkpoint
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys, e.g. {missing[:3]}", file=sys.stderr)
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}", file=sys.stderr)

    # tokenizer
    tok_path = cfg_args.get("tokenizer_path", str(_ROOT / "model" / "qwen3_5_tokenizer"))
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    return model, tokenizer, cfg


@torch.no_grad()
def _forward_with_loss(model, tokenizer, text: str, device: str):
    """单条 prompt forward，返回 (logits, per_token_loss, token_ids, aux)."""
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    labels = ids.clone()
    out = model(input_ids=ids, labels=labels)
    logits = out.logits
    # 对齐 reason_memory prefix
    if logits.size(-2) > labels.size(-1):
        logits = logits[:, -labels.size(-1):, :]
    _logits = logits[..., :-1, :].contiguous()
    _labels = labels[..., 1:].contiguous()
    per_tok = F.cross_entropy(
        _logits.view(-1, _logits.size(-1)),
        _labels.view(-1),
        ignore_index=-100, reduction="none",
    )
    raw_model = getattr(model, "_orig_mod", model)
    aux = getattr(raw_model.model, "last_aux", {}) if hasattr(raw_model, "model") else {}
    return logits, per_tok, ids, aux


# ---------------------------------------------------------------------------
# 测试 1: prompt loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_prompt_loss(model, tokenizer, device: str) -> dict:
    """测基础困惑度。目标：所有 prompt loss < 11.9 (random baseline for 151936 vocab)."""
    vocab_size = model.config.vocab_size
    baseline = math.log(vocab_size)

    results = {}
    for name, text in PROMPTS_BASIC.items():
        _, per_tok, _, _ = _forward_with_loss(model, tokenizer, text, device)
        mean_loss = float(per_tok.mean().item())
        ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
        results[name] = {
            "text": text,
            "mean_loss": round(mean_loss, 3),
            "perplexity": round(ppl, 1),
            "baseline": round(baseline, 2),
            "beats_random": mean_loss < baseline - 1.0,
        }

    losses = [r["mean_loss"] for r in results.values()]
    summary = {
        "mean": round(float(np.mean(losses)), 3),
        "max": round(float(np.max(losses)), 3),
        "min": round(float(np.min(losses)), 3),
        "baseline": round(baseline, 2),
        "all_beat_random": all(r["beats_random"] for r in results.values()),
    }
    return {"summary": summary, "per_prompt": results}


# ---------------------------------------------------------------------------
# 测试 2: top-k prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_topk_prediction(model, tokenizer, device: str, k: int = 10) -> dict:
    """给 prompt 看 top-k next token。目标：top-k 是合理 token。"""
    results = {}
    for name, text in PROMPTS_BASIC.items():
        logits, _, ids, _ = _forward_with_loss(model, tokenizer, text, device)
        last_logits = logits[0, -1, :]
        topk_probs, topk_ids = torch.softmax(last_logits.float(), dim=-1).topk(k)
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids.cpu().tolist())
        results[name] = {
            "prompt": text,
            "top_k": [
                {"token": t, "prob": round(float(p.item()), 4)}
                for t, p in zip(topk_tokens, topk_probs)
            ],
            "top1": topk_tokens[0],
        }
    return {"per_prompt": results}


# ---------------------------------------------------------------------------
# 测试 3: 语言分离度
# ---------------------------------------------------------------------------

def _is_chinese_token(tok: str) -> bool:
    """粗略判断：含有中文字符的 token。"""
    return any("\u4e00" <= c <= "\u9fff" for c in tok)


def _is_english_token(tok: str) -> bool:
    """粗略判断：纯 ASCII 字母的 token（剔除空格/标点）。"""
    clean = tok.lstrip("▁ ").strip()
    return len(clean) > 0 and all(c.isascii() and c.isalpha() for c in clean)


@torch.no_grad()
def test_language_isolation(model, tokenizer, device: str, k: int = 20) -> dict:
    """中文 prefix 看 top-k 是中文 token 的比例。"""
    cn_prompts = [("cn_weather", PROMPTS_BASIC["cn_weather"]),
                  ("cn_coding", PROMPTS_BASIC["cn_coding"])]
    en_prompts = [("en_fox", PROMPTS_BASIC["en_fox"]),
                  ("en_ml", PROMPTS_BASIC["en_ml"])]

    def _pct(prompts):
        cn_pcts, en_pcts = [], []
        for name, text in prompts:
            logits, _, _, _ = _forward_with_loss(model, tokenizer, text, device)
            topk_ids = logits[0, -1, :].float().topk(k).indices.cpu().tolist()
            topk_toks = tokenizer.convert_ids_to_tokens(topk_ids)
            cn_pcts.append(sum(_is_chinese_token(t) for t in topk_toks) / k * 100)
            en_pcts.append(sum(_is_english_token(t) for t in topk_toks) / k * 100)
        return {
            "cn_token_pct": round(float(np.mean(cn_pcts)), 1),
            "en_token_pct": round(float(np.mean(en_pcts)), 1),
        }

    return {
        "cn_prompt": _pct(cn_prompts),
        "en_prompt": _pct(en_prompts),
        "expect": "cn_prompt.cn_token_pct > en_token_pct 且反之亦然",
    }


# ---------------------------------------------------------------------------
# 测试 4: 退出深度分布
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_exit_depth_distribution(model, tokenizer, device: str) -> dict:
    """不同难度 prompt 的 exit loops 分布。目标：难度和循环数正相关。"""
    results = {}
    for name, text in PROMPTS_DIFFICULTY.items():
        _, _, _, aux = _forward_with_loss(model, tokenizer, text, device)
        exit_loops = aux.get("exit_loops")
        if exit_loops is None:
            loops_val = -1
        elif hasattr(exit_loops, "item"):
            loops_val = int(exit_loops.float().mean().item())
        else:
            loops_val = int(exit_loops)
        results[name] = {
            "prompt": text,
            "exit_loops": loops_val,
            "loops_detail_from_aux": str(aux.get("per_loop_delta_h", []))[:120],
        }
    loops_vals = [r["exit_loops"] for r in results.values() if r["exit_loops"] > 0]
    return {
        "per_prompt": results,
        "summary": {
            "mean_loops": round(float(np.mean(loops_vals)), 2) if loops_vals else 0,
            "max_loops": int(max(loops_vals)) if loops_vals else 0,
            "min_loops": int(min(loops_vals)) if loops_vals else 0,
        },
    }


# ---------------------------------------------------------------------------
# 测试 5: c_t 聚类采样
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_c_t_clustering(model, tokenizer, device: str) -> dict:
    """采集情绪 prompt 的最终 c_t。返回每条 prompt 的 c_t 向量和标签。
    不做 t-SNE（需要 sklearn），只保存 raw 向量供后续分析。"""
    samples = []
    for name, text in PROMPTS_EMOTION.items():
        _, _, _, aux = _forward_with_loss(model, tokenizer, text, device)
        ct_history = aux.get("c_t_history", [])
        if ct_history and len(ct_history) > 0:
            last_ct = ct_history[-1]
            if hasattr(last_ct, "cpu"):
                vec = last_ct.float().cpu().squeeze().numpy().tolist()
            else:
                vec = []
        else:
            vec = []
        emotion = name.split("_")[0]
        samples.append({
            "name": name,
            "emotion": emotion,
            "prompt": text,
            "c_t_dim": len(vec),
            "c_t_vec": vec[:64],  # 截前 64 维（应该就是 c_t_dim）
            "c_t_norm": float(np.linalg.norm(vec)) if vec else 0.0,
        })
    # 按情绪统计 c_t 平均
    emotion_stats = {}
    for emo in {s["emotion"] for s in samples}:
        vecs = [s["c_t_vec"] for s in samples if s["emotion"] == emo and s["c_t_vec"]]
        if vecs:
            arr = np.array(vecs)
            emotion_stats[emo] = {
                "mean_c_t": arr.mean(axis=0).tolist(),
                "norm_mean": float(arr.mean(axis=0).__abs__().mean()),
            }
    return {
        "samples": samples,
        "emotion_stats": emotion_stats,
        "note": "t-SNE 可视化需要 sklearn，本脚本只导出 raw 向量",
    }


# ---------------------------------------------------------------------------
# 测试 6: c_t 注入 ⭐
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_c_t_injection(model, tokenizer, device: str) -> dict:
    """手动覆盖 c_t 方向看输出差异。
    方法: 先采集 PROMPTS_EMOTION 的 c_t，做 PCA 取前 2 主方向作为"情绪轴"，
    然后用 ±主轴 × 不同强度覆盖 c_t，生成 top-k 看差异。"""
    # 1. 采集情绪 c_t
    ct_samples = []
    for name, text in PROMPTS_EMOTION.items():
        _, _, _, aux = _forward_with_loss(model, tokenizer, text, device)
        ct_history = aux.get("c_t_history", [])
        if ct_history:
            last_ct = ct_history[-1]
            if hasattr(last_ct, "cpu"):
                ct_samples.append(last_ct.float().cpu().squeeze().numpy())
    if len(ct_samples) < 2:
        return {"error": "collected <2 c_t samples, cannot do PCA"}

    arr = np.stack(ct_samples)  # (N, c_t_dim)
    mean = arr.mean(axis=0)
    centered = arr - mean
    # PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    principal_dir1 = Vt[0]  # (c_t_dim,)
    principal_dir2 = Vt[1] if Vt.shape[0] > 1 else np.zeros_like(Vt[0])

    # 2. 构造 4 种注入方向: neutral / +dir1 / -dir1 / +dir2
    test_ct_vectors = {
        "neutral": np.zeros_like(mean),
        "positive_axis1": mean + 3.0 * principal_dir1,
        "negative_axis1": mean - 3.0 * principal_dir1,
        "positive_axis2": mean + 3.0 * principal_dir2,
    }

    # 3. 对测试 prompt 做注入。
    # 注入机制: monkey-patch introspection_state_stream 的 forward
    # 让它永远返回指定的 c_t
    test_prompt = "今天下雨了，我感到"
    raw_model = getattr(model, "_orig_mod", model)
    backbone = raw_model.model
    introspection = backbone.introspection_state_stream

    original_forward = introspection.forward

    def make_patched_forward(override_ct):
        override_ct_t = torch.tensor(override_ct, dtype=torch.float32, device=device).unsqueeze(0)

        def patched(h, block_reprs, slow_state, loop_progress=None, loop_index=0, meta_override=None):
            # 调原 forward 拿 know_gap 和 new_slow_state
            know_gap, original_c_t, new_slow_state = original_forward(
                h, block_reprs, slow_state,
                loop_progress=loop_progress, loop_index=loop_index, meta_override=meta_override,
            )
            # 强行用 override_ct 替换
            bsz = original_c_t.size(0)
            new_ct = override_ct_t.expand(bsz, -1).to(original_c_t.dtype)
            return know_gap, new_ct, new_slow_state

        return patched

    results = {}
    for inj_name, inj_vec in test_ct_vectors.items():
        introspection.forward = make_patched_forward(inj_vec)
        try:
            logits, per_tok, _, _ = _forward_with_loss(model, tokenizer, test_prompt, device)
            mean_loss = float(per_tok.mean().item())
            last_logits = logits[0, -1, :]
            topk_probs, topk_ids = torch.softmax(last_logits.float(), dim=-1).topk(10)
            topk_toks = tokenizer.convert_ids_to_tokens(topk_ids.cpu().tolist())
            results[inj_name] = {
                "injected_norm": float(np.linalg.norm(inj_vec)),
                "prompt_loss": round(mean_loss, 3),
                "top_10_tokens": topk_toks,
                "top_10_probs": [round(float(p.item()), 4) for p in topk_probs],
            }
        finally:
            introspection.forward = original_forward

    # 4. 计算不同注入之间的 top-k 分布 KL 散度
    if len(results) >= 2:
        keys = list(results.keys())
        # 简单做法: 对比每一对的 top-10 token 重叠率
        overlap_matrix = {}
        for i, k1 in enumerate(keys):
            for k2 in keys[i+1:]:
                set1 = set(results[k1]["top_10_tokens"])
                set2 = set(results[k2]["top_10_tokens"])
                overlap = len(set1 & set2) / 10.0
                overlap_matrix[f"{k1}_vs_{k2}"] = round(overlap, 2)
        results["_overlap_matrix"] = overlap_matrix
        # 判定: 如果所有 overlap > 0.9 → c_t 注入没效果（人格框架弱）
        all_overlaps = list(overlap_matrix.values())
        results["_verdict"] = {
            "mean_overlap": round(float(np.mean(all_overlaps)), 2),
            "c_t_has_influence": float(np.mean(all_overlaps)) < 0.7,
        }

    return {
        "test_prompt": test_prompt,
        "pca_explained_variance_top2": [float(s**2 / (S**2).sum()) for s in S[:2]],
        "injections": results,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "prompt_loss": test_prompt_loss,
    "topk": test_topk_prediction,
    "lang": test_language_isolation,
    "exit": test_exit_depth_distribution,
    "ct_cluster": test_c_t_clustering,
    "ct_inject": test_c_t_injection,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output", default="", help="结果 JSON 输出路径")
    parser.add_argument("--tests", default="all",
                        help=f"测试列表: all 或 逗号分隔 ({','.join(ALL_TESTS.keys())})")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Luma Sanity Check")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  device: {args.device}")
    print(f"  tests: {args.tests}")
    print()

    model, tokenizer, cfg = load_model(args.checkpoint, device=args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print()

    if args.tests == "all":
        test_names = list(ALL_TESTS.keys())
    else:
        test_names = [t.strip() for t in args.tests.split(",") if t.strip()]

    results = {
        "checkpoint": args.checkpoint,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "tests": {},
    }

    for name in test_names:
        if name not in ALL_TESTS:
            print(f"  [skip] unknown test: {name}")
            continue
        print(f"[{name}] running...")
        try:
            results["tests"][name] = ALL_TESTS[name](model, tokenizer, args.device)
            print(f"  [ok]")
        except Exception as e:
            import traceback
            results["tests"][name] = {"error": str(e), "traceback": traceback.format_exc()[:2000]}
            print(f"  [error] {e}")

    # 输出
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved: {out_path}")

    # 人类可读 summary
    print("\n=== Summary ===")
    if "prompt_loss" in results["tests"]:
        s = results["tests"]["prompt_loss"].get("summary", {})
        print(f"  prompt_loss: mean={s.get('mean')} max={s.get('max')} beat_random={s.get('all_beat_random')}")
    if "lang" in results["tests"]:
        lr = results["tests"]["lang"]
        print(f"  language: cn-prefix {lr.get('cn_prompt', {}).get('cn_token_pct')}% cn / en-prefix {lr.get('en_prompt', {}).get('en_token_pct')}% en")
    if "exit" in results["tests"]:
        es = results["tests"]["exit"].get("summary", {})
        print(f"  exit_loops: mean={es.get('mean_loops')} max={es.get('max_loops')}")
    if "ct_inject" in results["tests"]:
        ci = results["tests"]["ct_inject"].get("injections", {}).get("_verdict", {})
        print(f"  ct_inject: mean_overlap={ci.get('mean_overlap')} has_influence={ci.get('c_t_has_influence')}")


if __name__ == "__main__":
    main()
