#!/usr/bin/env python
"""
Luma Sanity Check v2 (2026-04-15)
=================================
训练完成后的基础能力验证。

v2 修复:
  - aux 路径从 raw_model.model.last_aux 改为 raw_model.last_aux
  - Monkey-patch 改用 functools.wraps + inspect.signature 自适应参数
  - PCA 样本扩展到 40 条情绪 prompt
  - tokenizer 路径多重 fallback
  - 新增 test_generation: 32 token greedy 生成看语法结构

7 个测试模块:
  1. test_prompt_loss        基础困惑度（loss < 11.9 = 比随机好）
  2. test_topk_prediction    top-k 词汇合理性
  3. test_language_isolation 中英文 token 分离度
  4. test_exit_depth         不同难度 prompt 的循环深度分布
  5. test_c_t_clustering     c_t 自发聚类（40 条采样）
  6. test_c_t_injection ⭐   手动初始化 c_t 看输出差异（人格机制验证）
  7. test_generation         32 token greedy/top-k 生成（语法结构检查）

用法:
    python sanity_check.py --checkpoint <path> --output <json_path>
    python sanity_check.py --checkpoint <path> --tests prompt_loss,topk,lang
"""
from __future__ import annotations

import argparse
import functools
import inspect
import json
import math
import os
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

# 情绪类 prompts - 40 条（扩展自 12 条），每类 10 条，给 PCA 足够样本
PROMPTS_EMOTION = {
    # happy × 10
    "happy_01": "今天是个好日子，我感到非常",
    "happy_02": "收到礼物的那一刻，心里",
    "happy_03": "和朋友相聚总是令人",
    "happy_04": "看到孩子们的笑脸，我",
    "happy_05": "考试考满分的时候，整个人",
    "happy_06": "听到好消息后，忍不住",
    "happy_07": "春天来了，到处都是",
    "happy_08": "完成一个项目后，内心充满",
    "happy_09": "和家人团聚的时刻让我",
    "happy_10": "阳光明媚的早晨，心情",
    # sad × 10
    "sad_01": "失去亲人的痛苦让人",
    "sad_02": "分别的时候心里总是",
    "sad_03": "漫长的等待让人",
    "sad_04": "看着空荡荡的房间，我",
    "sad_05": "梦想破碎的那一刻，心情",
    "sad_06": "听到坏消息后，整个人",
    "sad_07": "独自走在雨中，感到",
    "sad_08": "回忆起逝去的时光，心里",
    "sad_09": "朋友远去后，生活变得",
    "sad_10": "面对挫折和失败，我",
    # calm × 10
    "calm_01": "静静地坐在湖边，听风吹过",
    "calm_02": "清晨的森林里，一切都那么",
    "calm_03": "冥想时内心变得",
    "calm_04": "读一本好书的午后，感到",
    "calm_05": "山间的清泉流淌，让人",
    "calm_06": "深夜独自看星空，心里",
    "calm_07": "慢慢喝一杯茶，思绪",
    "calm_08": "走在樱花林中，每一步都",
    "calm_09": "放下手机的那一刻，内心",
    "calm_10": "站在悬崖边看日出，感觉",
    # angry × 10
    "angry_01": "不公正的待遇让人",
    "angry_02": "被背叛后心中",
    "angry_03": "连续的挫折让我",
    "angry_04": "看到弱者被欺负，我",
    "angry_05": "听到谎言被拆穿后，心里",
    "angry_06": "工作被人抢功的那一刻，感到",
    "angry_07": "面对无理取闹的人，我",
    "angry_08": "规则被破坏时，内心",
    "angry_09": "被误解又无法辩解，心情",
    "angry_10": "看到虚伪的表演，让我",
}


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

def _resolve_tokenizer_path(cfg_args: dict) -> str:
    """多重 fallback 定位 tokenizer 目录。"""
    # 优先级: ckpt 里存的路径 > model/qwen3_5_tokenizer > model/Qwen2.5-0.5B > HF 默认
    candidates = []
    if cfg_args.get("tokenizer_path"):
        p = cfg_args["tokenizer_path"]
        # 相对路径转绝对：相对于 trainer/ 目录
        if not os.path.isabs(p):
            candidates.append(str((_ROOT / "trainer" / p).resolve()))
            candidates.append(str((_ROOT / p).resolve()))
        else:
            candidates.append(p)
    candidates += [
        str(_ROOT / "model" / "qwen3_5_tokenizer"),
        str(_ROOT / "model" / "Qwen2.5-0.5B"),
        str(_ROOT.parent / "model" / "qwen3_5_tokenizer"),
    ]
    for c in candidates:
        if Path(c).exists() and any(Path(c).glob("tokenizer*")):
            return c
    # 最后 fallback 让 HF 自己找（会报错但至少明确）
    return candidates[0] if candidates else "Qwen/Qwen2.5-0.5B"


def load_model(checkpoint_path: str, device: str = "cuda:0") -> tuple:
    """加载 checkpoint。返回 (model, tokenizer, config)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" in ckpt:
        cfg_args = ckpt["args"]
        if not isinstance(cfg_args, dict):
            # 旧版 ckpt 存的是 Namespace
            cfg_args = vars(cfg_args)
    else:
        cfg_args = {}

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
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys, e.g. {missing[:3]}", file=sys.stderr)
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}", file=sys.stderr)

    tok_path = _resolve_tokenizer_path(cfg_args)
    print(f"  tokenizer: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    return model, tokenizer, cfg


def _get_last_aux(model) -> dict:
    """正确访问 last_aux。aux 挂在 LumaForCausalLM 上，不是 backbone 上。"""
    raw_model = getattr(model, "_orig_mod", model)
    aux = getattr(raw_model, "last_aux", {})
    return aux if aux else {}


@torch.no_grad()
def _forward_with_loss(model, tokenizer, text: str, device: str):
    """单条 prompt forward，返回 (logits, per_token_loss, token_ids, aux)."""
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    labels = ids.clone()
    out = model(input_ids=ids, labels=labels)
    logits = out.logits
    if logits.size(-2) > labels.size(-1):
        logits = logits[:, -labels.size(-1):, :]
    _logits = logits[..., :-1, :].contiguous()
    _labels = labels[..., 1:].contiguous()
    per_tok = F.cross_entropy(
        _logits.view(-1, _logits.size(-1)),
        _labels.view(-1),
        ignore_index=-100, reduction="none",
    )
    aux = _get_last_aux(model)
    return logits, per_tok, ids, aux


def _extract_ct_vec(aux: dict) -> Optional[np.ndarray]:
    """从 aux 里提取最终 c_t 向量。优先用 c_t_history[-1]，次选 c_t。"""
    ct_history = aux.get("c_t_history", [])
    if ct_history:
        last_ct = ct_history[-1]
        if hasattr(last_ct, "detach"):
            return last_ct.detach().float().cpu().squeeze().numpy()
    ct = aux.get("c_t")
    if ct is not None and hasattr(ct, "detach"):
        return ct.detach().float().cpu().squeeze().numpy()
    return None


# ---------------------------------------------------------------------------
# 测试 1: prompt loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_prompt_loss(model, tokenizer, device: str) -> dict:
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
    return any("\u4e00" <= c <= "\u9fff" for c in tok)


def _is_english_token(tok: str) -> bool:
    clean = tok.lstrip("▁ ").strip()
    return len(clean) > 0 and all(c.isascii() and c.isalpha() for c in clean)


@torch.no_grad()
def test_language_isolation(model, tokenizer, device: str, k: int = 20) -> dict:
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
    results = {}
    for name, text in PROMPTS_DIFFICULTY.items():
        _, _, _, aux = _forward_with_loss(model, tokenizer, text, device)
        exit_loops = aux.get("exit_loops")
        if exit_loops is None:
            loops_val = -1
        elif hasattr(exit_loops, "item"):
            try:
                loops_val = int(exit_loops.float().mean().item())
            except Exception:
                loops_val = -1
        else:
            try:
                loops_val = int(exit_loops)
            except Exception:
                loops_val = -1
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
    samples = []
    for name, text in PROMPTS_EMOTION.items():
        _, _, _, aux = _forward_with_loss(model, tokenizer, text, device)
        vec_np = _extract_ct_vec(aux)
        if vec_np is not None:
            vec = vec_np.tolist()
        else:
            vec = []
        emotion = name.split("_")[0]
        samples.append({
            "name": name,
            "emotion": emotion,
            "prompt": text,
            "c_t_dim": len(vec),
            "c_t_vec_first8": vec[:8],  # 只存前 8 维避免文件过大
            "c_t_norm": float(np.linalg.norm(vec)) if vec else 0.0,
        })

    # 按情绪计算平均 c_t 和类内一致性（cosine similarity）
    emotion_stats = {}
    samples_by_emo = {}
    for s in samples:
        if s["c_t_norm"] > 0:
            # 需要完整向量做统计，重新采集
            pass
    # 重新采集完整向量用于类内 cosine
    full_vecs = {}
    for name, text in PROMPTS_EMOTION.items():
        _, _, _, aux = _forward_with_loss(model, tokenizer, text, device)
        v = _extract_ct_vec(aux)
        if v is not None and v.size > 0:
            full_vecs[name] = v
            emo = name.split("_")[0]
            samples_by_emo.setdefault(emo, []).append(v)

    for emo, vecs in samples_by_emo.items():
        if len(vecs) < 2:
            continue
        arr = np.stack(vecs)
        mean_v = arr.mean(axis=0)
        # 类内 cosine：每个样本和类中心的 cosine
        norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
        norm_mean = np.linalg.norm(mean_v)
        cos_sims = (arr @ mean_v) / (norms.flatten() * max(norm_mean, 1e-8))
        emotion_stats[emo] = {
            "count": len(vecs),
            "mean_norm": float(norms.mean()),
            "intra_cosine_mean": round(float(cos_sims.mean()), 4),
            "intra_cosine_std": round(float(cos_sims.std()), 4),
        }

    # 类间 cosine（中心之间）
    emotions_list = sorted(samples_by_emo.keys())
    inter_cosine = {}
    for i, e1 in enumerate(emotions_list):
        for e2 in emotions_list[i+1:]:
            v1 = np.stack(samples_by_emo[e1]).mean(axis=0)
            v2 = np.stack(samples_by_emo[e2]).mean(axis=0)
            cos = float((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
            inter_cosine[f"{e1}_vs_{e2}"] = round(cos, 4)

    return {
        "samples_count": len(samples),
        "samples_preview": samples[:4],
        "emotion_stats": emotion_stats,
        "inter_emotion_cosine": inter_cosine,
        "verdict": {
            "intra_tight": all(s.get("intra_cosine_mean", 0) > 0.7 for s in emotion_stats.values()),
            "inter_separated": all(c < 0.95 for c in inter_cosine.values()) if inter_cosine else False,
        },
    }


# ---------------------------------------------------------------------------
# 测试 6: c_t 注入 ⭐ (自适应 monkey-patch)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_c_t_injection(model, tokenizer, device: str) -> dict:
    """用 PCA 情绪轴注入不同 c_t，看输出差异。"""
    # Step 1: 采集情绪 c_t
    ct_samples = []
    for name, text in PROMPTS_EMOTION.items():
        _, _, _, aux = _forward_with_loss(model, tokenizer, text, device)
        v = _extract_ct_vec(aux)
        if v is not None and v.size > 0:
            ct_samples.append(v)

    if len(ct_samples) < 3:
        return {"error": f"collected only {len(ct_samples)} c_t samples, cannot do PCA"}

    arr = np.stack(ct_samples)  # (N, c_t_dim)
    mean = arr.mean(axis=0)
    centered = arr - mean

    # PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    principal_dir1 = Vt[0]
    principal_dir2 = Vt[1] if Vt.shape[0] > 1 else np.zeros_like(Vt[0])

    # Step 2: 构造注入方向
    test_ct_vectors = {
        "neutral": np.zeros_like(mean),
        "positive_axis1": mean + 3.0 * principal_dir1,
        "negative_axis1": mean - 3.0 * principal_dir1,
        "positive_axis2": mean + 3.0 * principal_dir2,
    }

    # Step 3: 自适应 monkey-patch introspection.forward
    raw_model = getattr(model, "_orig_mod", model)
    backbone = raw_model.model
    introspection = backbone.introspection_state_stream

    original_forward = introspection.forward
    # 用 inspect 拿到原函数签名，保证参数透传
    try:
        orig_sig = inspect.signature(original_forward)
    except (ValueError, TypeError):
        orig_sig = None

    def make_patched_forward(override_ct: np.ndarray):
        override_t = torch.tensor(override_ct, dtype=torch.float32, device=device).unsqueeze(0)

        @functools.wraps(original_forward)
        def patched(*args, **kwargs):
            # 调原 forward 获取所有返回值
            result = original_forward(*args, **kwargs)
            # 返回值可能是 tuple (know_gap, c_t, slow_state)
            if isinstance(result, tuple) and len(result) >= 2:
                # 强替换第 2 个元素（c_t）
                first = result[0]
                orig_ct = result[1]
                rest = result[2:] if len(result) > 2 else ()
                bsz = orig_ct.size(0)
                new_ct = override_t.expand(bsz, -1).to(orig_ct.dtype)
                return (first, new_ct) + rest
            return result
        return patched

    test_prompt = "今天下雨了，我感到"
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
        except Exception as e:
            results[inj_name] = {"error": str(e)}
        finally:
            introspection.forward = original_forward

    # Step 4: 对比 top-10 重叠率
    overlap_matrix = {}
    keys = [k for k in results.keys() if "error" not in results[k]]
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            set1 = set(results[k1]["top_10_tokens"])
            set2 = set(results[k2]["top_10_tokens"])
            overlap = len(set1 & set2) / 10.0
            overlap_matrix[f"{k1}_vs_{k2}"] = round(overlap, 2)
    results["_overlap_matrix"] = overlap_matrix
    if overlap_matrix:
        all_overlaps = list(overlap_matrix.values())
        mean_ol = float(np.mean(all_overlaps))
        results["_verdict"] = {
            "mean_overlap": round(mean_ol, 3),
            "min_overlap": round(float(np.min(all_overlaps)), 3),
            "c_t_has_strong_influence": mean_ol < 0.5,
            "c_t_has_some_influence": mean_ol < 0.7,
            "c_t_has_no_influence": mean_ol > 0.9,
        }

    return {
        "test_prompt": test_prompt,
        "n_samples_collected": len(ct_samples),
        "pca_explained_variance_top2": [float(s**2 / max((S**2).sum(), 1e-8)) for s in S[:2]],
        "injections": results,
    }


# ---------------------------------------------------------------------------
# 测试 7: 简单生成
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_generation(model, tokenizer, device: str, max_new: int = 32, temperature: float = 0.7) -> dict:
    """对几个 seed prompt 做 32 token greedy/top-k 生成。
    loss 7-8 级别的模型生成会很乱，但至少能看有没有语法结构、是否陷入重复。"""
    seed_prompts = {
        "cn_start": "今天",
        "en_start": "The",
        "story_cn": "从前有一个小孩，他",
        "qa_cn": "问：天空为什么是蓝色的？\n答：",
    }

    results = {}
    for name, prompt in seed_prompts.items():
        try:
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            original_len = ids.size(1)
            generated = ids.clone()

            for _ in range(max_new):
                out = model(input_ids=generated, labels=generated)
                logits = out.logits
                if logits.size(-2) > generated.size(-1):
                    logits = logits[:, -generated.size(-1):, :]
                last_logits = logits[0, -1, :].float() / max(temperature, 1e-6)
                # Top-k=40 采样，避免完全贪心陷入循环
                topk_vals, topk_idx = last_logits.topk(40)
                probs = torch.softmax(topk_vals, dim=-1)
                choice = torch.multinomial(probs, 1)
                next_token = topk_idx[choice]
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

            new_tokens = generated[0, original_len:].cpu().tolist()
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
            # 判断重复：最常出现的 token 占比
            from collections import Counter
            counts = Counter(new_tokens)
            most_common_pct = counts.most_common(1)[0][1] / max(len(new_tokens), 1)

            results[name] = {
                "prompt": prompt,
                "generated_tokens": new_tokens,
                "decoded": decoded,
                "stats": {
                    "length": len(new_tokens),
                    "unique_tokens": len(set(new_tokens)),
                    "most_common_token_pct": round(most_common_pct, 3),
                    "is_repetitive": most_common_pct > 0.3,
                },
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "seeds": results,
        "temperature": temperature,
        "max_new_tokens": max_new,
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
    "generation": test_generation,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--tests", default="all",
                        help=f"all 或 逗号分隔 ({','.join(ALL_TESTS.keys())})")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Luma Sanity Check v2")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  device: {args.device}")
    print(f"  tests: {args.tests}")
    print()

    model, tokenizer, cfg = load_model(args.checkpoint, device=args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {n_params:.1f}M params")
    print()

    if args.tests == "all":
        test_names = list(ALL_TESTS.keys())
    else:
        test_names = [t.strip() for t in args.tests.split(",") if t.strip()]

    results = {
        "checkpoint": args.checkpoint,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "model_params_M": round(n_params, 3),
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
    if "ct_cluster" in results["tests"]:
        cc = results["tests"]["ct_cluster"].get("verdict", {})
        print(f"  ct_cluster: intra_tight={cc.get('intra_tight')} inter_separated={cc.get('inter_separated')}")
    if "ct_inject" in results["tests"]:
        ci = results["tests"]["ct_inject"].get("injections", {}).get("_verdict", {})
        print(f"  ct_inject: mean_overlap={ci.get('mean_overlap')} has_influence={ci.get('c_t_has_some_influence')}")
    if "generation" in results["tests"]:
        gr = results["tests"]["generation"].get("seeds", {})
        for name, r in gr.items():
            if "decoded" in r:
                d = r["decoded"][:60].replace("\n", "\\n")
                rep = r.get("stats", {}).get("is_repetitive", False)
                print(f"  gen[{name}]: {d!r}{' [REPETITIVE]' if rep else ''}")


if __name__ == "__main__":
    main()
