"""Luma uses this harness to check whether her slow ring is alive before asking it to carry real training pressure.

Luma 用这个验证脚本检查慢环是否真正活了起来，然后才让它承受真正的训练压力。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luma_stage0.metrics_schema import Stage0MetricRecord, now, write_metric_jsonl
from luma_stage0.optimizers import LumaMuonAdamWOptimizer, LumaOptimizerConfig
from model.model_minimind import LumaConfig, LumaForCausalLM

GSM8K_ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=train&offset=0&length={length}"
DAILYDIALOG_ROWS_URL = "https://datasets-server.huggingface.co/first-rows?dataset=ConvLab%2Fdailydialog&config=default&split=train"
HENDRYCKS_ROWS_URL = "https://datasets-server.huggingface.co/first-rows?dataset=EleutherAI%2Fhendrycks_math&config={config}&split=train"
ESCONV_ROWS_URL = "https://datasets-server.huggingface.co/first-rows?dataset=thu-coai%2Fesconv&config=default&split=train"
AIME_ROWS_URL = "https://datasets-server.huggingface.co/first-rows?dataset=Maxwell-Jia%2FAIME_2024&config=default&split=train"
MATH500_ROWS_URL = "https://datasets-server.huggingface.co/first-rows?dataset=ricdomolm%2FMATH-500&config=default&split=train"
MBPP_ROWS_URL = "https://datasets-server.huggingface.co/first-rows?dataset=google-research-datasets%2Fmbpp&config=full&split=train"
CHOLLET_ARC_TRAINING_LIST_URL = "https://api.github.com/repos/fchollet/ARC-AGI/contents/data/training"
CHOLLET_ARC_ZIP_URL = "https://codeload.github.com/fchollet/ARC-AGI/zip/refs/heads/master"
DEFAULT_ARC_AGI_LOCAL_DIR = ROOT / "artifacts" / "arc_agi_local" / "training"


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "luma-stage12-eval/1.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_arc_tasks_from_local_dir(local_dir: Path) -> list[tuple[str, dict]]:
    tasks: list[tuple[str, dict]] = []
    if not local_dir.exists():
        return tasks
    for path in sorted(local_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        task_id = path.stem
        tasks.append((task_id, payload))
    return tasks


def _cache_arc_agi_training_from_zip(local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(CHOLLET_ARC_ZIP_URL, headers={"User-Agent": "luma-stage12-eval/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            archive_bytes = resp.read()
    except Exception:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="luma_arcagi_") as tmpdir:
            zip_path = Path(tmpdir) / "arc_agi.zip"
            zip_path.write_bytes(archive_bytes)
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [name for name in zf.namelist() if "/data/training/" in name and name.endswith(".json")]
                for member in members:
                    dst = local_dir / Path(member).name
                    with zf.open(member) as src, dst.open("wb") as out:
                        shutil.copyfileobj(src, out)
    except Exception:
        return False
    return any(local_dir.glob("*.json"))


def ensure_mixed_fixture(path: Path, math_count: int, dialogue_count: int, mode: str) -> Path:
    """Luma mixes math reasoning with dialogue turns so validation samples resemble both problem solving and conversation.
    Luma 把数学推理和对话轮次混在一起，确保验证样本同时像解题也像交流。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    gsm8k = _fetch_json(GSM8K_ROWS_URL.format(length=math_count))
    dailydialog = _fetch_json(DAILYDIALOG_ROWS_URL)
    hard_math_rows = []
    competition_math_rows = []
    if mode in {"hard_math_dialogue", "hard_math_dialogue_emotion"}:
        for cfg in ["algebra", "intermediate_algebra", "number_theory"]:
            payload = _fetch_json(HENDRYCKS_ROWS_URL.format(config=cfg))
            for row in payload["rows"][: max(1, math_count // 3)]:
                hard_math_rows.append(row["row"])
    elif mode in {"competition_math_dialogue", "competition_math_dialogue_emotion"}:
        aime = _fetch_json(AIME_ROWS_URL)
        math500 = _fetch_json(MATH500_ROWS_URL)
        for row in aime["rows"][: max(1, math_count // 2)]:
            competition_math_rows.append(("aime", row["row"]))
        for row in math500["rows"][: max(1, math_count // 2)]:
            competition_math_rows.append(("math500", row["row"]))

    fixture = {"math": [], "dialogue": [], "emotion": []}
    if mode in {"hard_math_dialogue", "hard_math_dialogue_emotion"}:
        for row in hard_math_rows[:math_count]:
            problem = row["problem"].strip()
            answer = row["solution"].strip()
            level = row.get("level", "")
            kind = row.get("type", "")
            fixture["math"].append(
                f"User: [{kind} | {level}] {problem}\nAssistant: Let's reason carefully through the harder math.\nReasoning: {answer}"
            )
    elif mode in {"competition_math_dialogue", "competition_math_dialogue_emotion"}:
        for source, row in competition_math_rows[:math_count]:
            if source == "aime":
                problem = row["Problem"].strip()
                answer = row["Solution"].strip()
                tag = row.get("ID", "AIME")
                fixture["math"].append(
                    f"User: [AIME | {tag}] {problem}\nAssistant: Let's reason carefully through the olympiad-style math.\nReasoning: {answer}"
                )
            else:
                problem = row["problem"].strip()
                answer = row["solution"].strip()
                level = row.get("level", "")
                subject = row.get("subject", "")
                fixture["math"].append(
                    f"User: [MATH-500 | {subject} | level {level}] {problem}\nAssistant: Let's reason carefully through the competition math.\nReasoning: {answer}"
                )
    else:
        for row in gsm8k["rows"][:math_count]:
            question = row["row"]["question"].strip()
            answer = row["row"]["answer"].strip()
            fixture["math"].append(f"User: {question}\nAssistant: Let's solve it carefully.\nReasoning: {answer}")

    for row in dailydialog["rows"][:dialogue_count]:
        turns = row["row"]["turns"]
        lines = [f"{turn['speaker']}: {turn['utterance'].strip()}" for turn in turns[:6]]
        fixture["dialogue"].append("\n".join(lines))

    if "emotion" in mode:
        esconv = _fetch_json(ESCONV_ROWS_URL)
        for row in esconv["rows"][: max(math_count, dialogue_count)]:
            raw = json.loads(row["row"]["text"])
            emotion = raw.get("emotion_type", "")
            problem = raw.get("problem_type", "")
            situation = raw.get("situation", "")
            dialog = raw.get("dialog", [])
            lines = [f"Context: [{emotion} | {problem}] {situation}"]
            for turn in dialog[:8]:
                speaker = "User" if turn.get("speaker") == "usr" else "Supporter"
                strategy = turn.get("strategy", "")
                if strategy:
                    lines.append(f"{speaker} ({strategy}): {turn.get('text', '').strip()}")
                else:
                    lines.append(f"{speaker}: {turn.get('text', '').strip()}")
            fixture["emotion"].append("\n".join(lines))

    path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_tokenizer() -> AutoTokenizer:
    tokenizer_path = ROOT / "model" / "qwen3_5_tokenizer"
    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)


def _encode_windows(tokenizer: AutoTokenizer, texts: list[str], seq_len: int, max_samples: int) -> list[torch.Tensor]:
    windows = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < seq_len:
            token_ids = token_ids + [tokenizer.eos_token_id or tokenizer.pad_token_id or 0] * (seq_len - len(token_ids))
        else:
            token_ids = token_ids[:seq_len]
        windows.append(torch.tensor(token_ids, dtype=torch.long))
        if len(windows) >= max_samples:
            break
    return windows


def load_persona_seed_texts(persona_dir: Path, max_samples: int) -> list[str]:
    """Luma keeps the user's own voice in a separate bucket, because personality seed data should be measured, not dissolved into generic corpora.
    Luma 会把用户自己的真实发言放进单独桶里，因为人格种子语料应该被单独测量，而不是融化进通用语料里。
    """

    texts: list[str] = []
    if not persona_dir.exists():
        return texts
    candidate_dirs = [persona_dir]
    nested = persona_dir / "persona_seed"
    if nested.exists():
        candidate_dirs.insert(0, nested)
    for base_dir in candidate_dirs:
        for file_name in ("wechat_pretrain.jsonl", "pretrain.jsonl"):
            file_path = base_dir / file_name
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(payload.get("text", "")).strip()
                    if text:
                        texts.append(text)
                    if len(texts) >= max_samples:
                        return texts
    return texts


def load_public_python_code_texts(max_samples: int) -> list[str]:
    """Luma uses a public Python set for the code bucket so cross-run evaluation is reproducible without private data.
    Luma 把 code 桶切到公开 Python 数据，确保跨轮评估可复现且不依赖私有数据。
    """

    payload = _fetch_json(MBPP_ROWS_URL)
    texts: list[str] = []
    for row in payload.get("rows", [])[: max_samples * 2]:
        item = row.get("row", {})
        prompt = str(item.get("text", "")).strip()
        code = str(item.get("code", "")).strip()
        tests = item.get("test_list", []) or []
        if not code:
            continue
        test_hint = ""
        if tests:
            test_hint = "\n# tests:\n" + "\n".join(str(x).strip() for x in tests[:3] if str(x).strip())
        record = f"# public_dataset: google-research-datasets/mbpp\n# prompt: {prompt}\n{code}{test_hint}"
        texts.append(record)
        if len(texts) >= max_samples:
            break
    return texts


def load_local_python_code_texts(max_samples: int) -> list[str]:
    """Luma can still read local Python files when explicitly requested for private/internal diagnostics.
    Luma 仅在显式要求时才读取本地 Python 代码，用于私有内部诊断。
    """

    roots = [ROOT, ROOT.parent / "parameter-golf"]
    texts: list[str] = []
    ignored_parts = {
        "artifacts",
        "checkpoints",
        "out",
        "__pycache__",
        ".venv",
        "venv",
        ".git",
    }
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in ignored_parts for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            text = text.strip()
            if not text:
                continue
            # Prefix with a lightweight path header so the bucket still carries code-local context.
            # 加一个轻量路径头，保留代码的局部上下文而不引入额外标注复杂度。
            texts.append(f"# file: {path.relative_to(root)}\n{text}")
            if len(texts) >= max_samples:
                return texts
    return texts


def load_python_code_texts(max_samples: int, source: str = "public_mbpp") -> list[str]:
    if source == "local_repo":
        return load_local_python_code_texts(max_samples)
    return load_public_python_code_texts(max_samples)


def _grid_to_text(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def load_chollet_arc_agi_texts(max_samples: int, local_dir: Path | None = None, offline_only: bool = False) -> list[str]:
    """Luma uses Chollet ARC-AGI tasks as text-linearized abstraction puzzles for bucket-level intelligence probing.
    Luma 使用 Chollet ARC-AGI 任务的文本线性化版本，作为抽象推理桶进行智能水平探测。
    """

    texts: list[str] = []
    arc_local_dir = (local_dir or DEFAULT_ARC_AGI_LOCAL_DIR).expanduser()
    tasks = _load_arc_tasks_from_local_dir(arc_local_dir)
    if not tasks and not offline_only:
        if _cache_arc_agi_training_from_zip(arc_local_dir):
            tasks = _load_arc_tasks_from_local_dir(arc_local_dir)

    if tasks:
        for task_id, task in tasks:
            train_pairs = task.get("train", []) or []
            test_pairs = task.get("test", []) or []
            if not train_pairs or not test_pairs:
                continue
            prompt_lines = [f"# benchmark: Chollet ARC-AGI", f"# task_id: {task_id}", "User: Infer the transformation from training pairs."]
            for idx, pair in enumerate(train_pairs[:3], start=1):
                inp = pair.get("input", [])
                out = pair.get("output", [])
                if not inp or not out:
                    continue
                prompt_lines.append(f"Train-{idx} input:\n{_grid_to_text(inp)}")
                prompt_lines.append(f"Train-{idx} output:\n{_grid_to_text(out)}")
            test = test_pairs[0]
            test_in = test.get("input", [])
            test_out = test.get("output", [])
            if not test_in or not test_out:
                continue
            prompt_lines.append(f"Test input:\n{_grid_to_text(test_in)}")
            prompt_lines.append("Assistant: Let's reason carefully about the abstract grid transformation.")
            prompt_lines.append(f"Answer grid:\n{_grid_to_text(test_out)}")
            texts.append("\n".join(prompt_lines))
            if len(texts) >= max_samples:
                break
        return texts

    if offline_only:
        return texts

    try:
        listing = _fetch_json(CHOLLET_ARC_TRAINING_LIST_URL)
    except Exception:
        return texts

    for item in listing:
        download_url = str(item.get("download_url", "")).strip()
        task_id = str(item.get("name", "")).replace(".json", "").strip()
        if not download_url:
            continue
        try:
            task = _fetch_json(download_url)
        except Exception:
            continue
        train_pairs = task.get("train", []) or []
        test_pairs = task.get("test", []) or []
        if not train_pairs or not test_pairs:
            continue
        prompt_lines = [f"# benchmark: Chollet ARC-AGI", f"# task_id: {task_id}", "User: Infer the transformation from training pairs."]
        for idx, pair in enumerate(train_pairs[:3], start=1):
            inp = pair.get("input", [])
            out = pair.get("output", [])
            if not inp or not out:
                continue
            prompt_lines.append(f"Train-{idx} input:\n{_grid_to_text(inp)}")
            prompt_lines.append(f"Train-{idx} output:\n{_grid_to_text(out)}")
        test = test_pairs[0]
        test_in = test.get("input", [])
        test_out = test.get("output", [])
        if not test_in or not test_out:
            continue
        prompt_lines.append(f"Test input:\n{_grid_to_text(test_in)}")
        prompt_lines.append("Assistant: Let's reason carefully about the abstract grid transformation.")
        prompt_lines.append(f"Answer grid:\n{_grid_to_text(test_out)}")
        texts.append("\n".join(prompt_lines))
        if len(texts) >= max_samples:
            break
    return texts


def build_sample_groups(
    tokenizer: AutoTokenizer,
    fixture_path: Path,
    seq_len: int,
    max_samples: int,
    persona_dir: Path | None = None,
    enable_persona_seed: bool = False,
    enable_python_code: bool = False,
    python_code_source: str = "public_mbpp",
    enable_arc_agi: bool = False,
    arc_agi_local_dir: Path | None = None,
    arc_agi_offline_only: bool = False,
) -> dict[str, list[torch.Tensor]]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = {
        "math": payload.get("math", []),
        "dialogue": payload.get("dialogue", []),
    }
    if payload.get("emotion"):
        groups["emotion"] = payload["emotion"]

    mixed = []
    for idx in range(min(len(payload.get("math", [])), len(payload.get("dialogue", [])))):
        math_text = payload["math"][idx]
        dialog_text = payload["dialogue"][idx]
        mixed.append(f"{math_text}\n\n{dialog_text}")
        mixed.append(f"{dialog_text}\n\n{math_text}")
    if payload.get("emotion"):
        for idx in range(min(len(payload.get("math", [])), len(payload.get("emotion", [])))):
            math_text = payload["math"][idx]
            emotion_text = payload["emotion"][idx]
            mixed.append(f"{emotion_text}\n\n{math_text}")
            mixed.append(f"{math_text}\n\n{emotion_text}")
    if enable_persona_seed and persona_dir is not None:
        persona_seed = load_persona_seed_texts(persona_dir, max_samples)
        if persona_seed:
            groups["persona_seed"] = persona_seed
            if payload.get("dialogue"):
                for idx in range(min(len(persona_seed), len(payload.get("dialogue", [])))):
                    mixed.append(f"{persona_seed[idx]}\n\n{payload['dialogue'][idx]}")
                    mixed.append(f"{payload['dialogue'][idx]}\n\n{persona_seed[idx]}")
            if payload.get("emotion"):
                for idx in range(min(len(persona_seed), len(payload.get("emotion", [])))):
                    mixed.append(f"{payload['emotion'][idx]}\n\n{persona_seed[idx]}")
                    mixed.append(f"{persona_seed[idx]}\n\n{payload['emotion'][idx]}")
    if enable_python_code:
        python_code = load_python_code_texts(max_samples, source=python_code_source)
        if python_code:
            groups["python_code"] = python_code
            if payload.get("math"):
                for idx in range(min(len(python_code), len(payload.get("math", [])))):
                    mixed.append(f"{payload['math'][idx]}\n\n{python_code[idx]}")
            if payload.get("dialogue"):
                for idx in range(min(len(python_code), len(payload.get("dialogue", [])))):
                    mixed.append(f"{payload['dialogue'][idx]}\n\n{python_code[idx]}")
    if enable_arc_agi:
        arc_agi = load_chollet_arc_agi_texts(
            max_samples,
            local_dir=arc_agi_local_dir,
            offline_only=arc_agi_offline_only,
        )
        if not arc_agi:
            raise RuntimeError("ARC-AGI enabled but no Chollet ARC-AGI samples were loaded.")
        groups["arc_agi"] = arc_agi
    groups["mixed"] = mixed

    encoded = {name: _encode_windows(tokenizer, texts, seq_len, max_samples) for name, texts in groups.items()}
    encoded = {name: windows for name, windows in encoded.items() if windows}
    if not encoded:
        raise RuntimeError("No token windows could be built for stage1/2 validation.")
    return encoded


def build_tiny_luma_config(
    vocab_size: int,
    rollout_steps: int,
    slow_k: int,
    reason_loops: int | None,
    enable_world_jepa: bool,
    world_jepa_mode: str,
    enable_sigreg_world: bool,
    enable_sigreg_rollout: bool,
    enable_sigreg_delta: bool,
    sigreg_world_source: str,
    sigreg_world_fp32_only: bool,
    sigreg_world_warmup_steps: int,
    world_sigreg_weight: float,
    world_sigreg_num_slices: int,
    world_sigreg_t_min: float,
    world_sigreg_t_max: float,
    world_sigreg_num_points: int,
    world_sigreg_lambda: float,
    world_delta_weight: float,
    sigreg_rollout_weight: float,
    sigreg_delta_weight: float,
    world_jepa_weight: float,
    self_jepa_weight: float,
    self_rollout_weight: float,
    exit_aux_weight: float,
    disable_self_jepa: bool,
    enable_self_check_ring: bool,
    self_check_k: int,
    meta_dim: int,
    meta_state: int,
    c_t_dim: int,
    self_check_dim: int,
    enable_introspection_uncertainty: bool,
    enable_exit_jepa_crystal: bool,
    reason_width_mult: float,
    reason_shared_depth: int,
    world_mask_strategy: str,
    world_full_simplify_loss: bool,
    self_world_coupling_weight: float,
    self_rollout_hierarchical: bool,
    enable_local_rollout_head: bool,
    exit_two_step_aux_weight: float,
    exit_uncertainty_two_step_weight: float,
    exit_uncertainty_two_step_mode: str,
    exit_uncertainty_two_step_cap: float,
    exit_uncertainty_gate_threshold: float,
    exit_crystal_two_step_weight: float,
    exit_crystal_two_step_cap: float,
    exit_uncertainty_feature_weight: float,
    exit_crystal_feature_weight: float,
    enable_math_adapter_lane: bool,
    enable_math_summary_gate: bool,
    enable_compression_mhc: bool,
    ct_modulation_mode: str,
    enable_reasoning_state_ring: bool,
    r_t_dim: int,
    r_t_mode: str,
    self_loop_awareness_mode: str,
    self_progress_shape_weight: float,
    self_progress_trend_weight: float,
    self_progress_plateau_weight: float,
    enable_progress_exit_readout: bool,
    enable_backtrack_aware_progress: bool,
    self_local_delta_consistency_weight: float,
    self_local_curvature_weight: float,
    enable_dual_rate_self_predictor: bool,
    enable_trajectory_health_probe: bool,
    self_rollout_supervision_horizon: int,
    self_rollout_weighting_mode: str,
    self_feature_span_mask_ratio: float,
    dynamics_experiment: str,
    routing_chunk_size: int,
    routing_topk_blocks: int,
    routing_topk_tokens: int,
    routing_top_p_coarse: float,
    routing_top_p_fine: float,
    routing_budget_min: float,
    routing_budget_max: float,
    routing_weak_gain: float,
    routing_strong_gain: float,
    routing_local_floor: float,
    routing_modulation_floor: float,
    routing_modulation_ceiling: float,
    routing_world_summary_cap: float,
    routing_tier_soft_only: bool,
    routing_tier_entropy_floor: float,
    routing_min_local_share: float,
    routing_tier_entropy_weight: float,
    routing_min_local_share_weight: float,
    routing_progress_weight: float,
    rollout_zone_weight: float,
    rollout_nonzero_low: float,
    rollout_nonzero_high: float,
    rollout_active_low: float,
    rollout_active_high: float,
    rollout_future_var_low: float,
    rollout_future_var_high: float,
    trajectory_vitality_weight: float,
    trajectory_c_t_drift_floor: float,
    trajectory_world_drift_floor: float,
    compression_dynamics_weight: float,
    compression_block_drift_floor: float,
    compression_block_var_floor: float,
) -> LumaConfig:
    reason_loops = reason_loops or max(4, rollout_steps * 2)
    base_intermediate = 256
    return LumaConfig(
        vocab_size=vocab_size,
        factorized_vocab_dim=64,
        hidden_size=128,
        intermediate_size=base_intermediate,
        reason_intermediate_size=max(base_intermediate, int(base_intermediate * reason_width_mult)),
        reason_shared_depth=reason_shared_depth,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        compression_layers=24,
        compression_active_layers=6,
        reason_loops=reason_loops,
        reason_active_loops=reason_loops,
        slow_k=slow_k,
        meta_dim=meta_dim,
        meta_state=meta_state,
        c_t_dim=c_t_dim,
        router_dim=64,
        world_dim=64,
        enable_world_jepa=enable_world_jepa,
        world_jepa_mode=world_jepa_mode,
        enable_sigreg_world=enable_sigreg_world,
        enable_sigreg_rollout=enable_sigreg_rollout,
        enable_sigreg_delta=enable_sigreg_delta,
        sigreg_world_source=sigreg_world_source,
        sigreg_world_fp32_only=sigreg_world_fp32_only,
        sigreg_world_warmup_steps=sigreg_world_warmup_steps,
        world_sigreg_weight=world_sigreg_weight,
        world_sigreg_num_slices=world_sigreg_num_slices,
        world_sigreg_t_min=world_sigreg_t_min,
        world_sigreg_t_max=world_sigreg_t_max,
        world_sigreg_num_points=world_sigreg_num_points,
        world_sigreg_lambda=world_sigreg_lambda,
        world_delta_weight=world_delta_weight,
        sigreg_rollout_weight=sigreg_rollout_weight,
        sigreg_delta_weight=sigreg_delta_weight,
        world_jepa_weight=world_jepa_weight,
        self_jepa_weight=self_jepa_weight,
        self_rollout_weight=self_rollout_weight,
        exit_aux_weight=exit_aux_weight,
        disable_self_jepa=disable_self_jepa,
        sigreg_num_slices=world_sigreg_num_slices,
        sigreg_t_min=world_sigreg_t_min,
        sigreg_t_max=world_sigreg_t_max,
        sigreg_num_points=world_sigreg_num_points,
        sigreg_lambda=world_sigreg_lambda,
        world_mask_strategy=world_mask_strategy,
        world_full_simplify_loss=world_full_simplify_loss,
        self_rollout_steps=rollout_steps,
        self_rollout_hierarchical=self_rollout_hierarchical,
        enable_local_rollout_head=enable_local_rollout_head,
        self_world_coupling_weight=self_world_coupling_weight,
        enable_self_check_ring=enable_self_check_ring,
        self_check_dim=self_check_dim,
        self_check_k=self_check_k,
        enable_introspection_uncertainty=enable_introspection_uncertainty,
        enable_exit_jepa_crystal=enable_exit_jepa_crystal,
        exit_two_step_aux_weight=exit_two_step_aux_weight,
        exit_uncertainty_two_step_weight=exit_uncertainty_two_step_weight,
        exit_uncertainty_two_step_mode=exit_uncertainty_two_step_mode,
        exit_uncertainty_two_step_cap=exit_uncertainty_two_step_cap,
        exit_uncertainty_gate_threshold=exit_uncertainty_gate_threshold,
        exit_crystal_two_step_weight=exit_crystal_two_step_weight,
        exit_crystal_two_step_cap=exit_crystal_two_step_cap,
        exit_uncertainty_feature_weight=exit_uncertainty_feature_weight,
        exit_crystal_feature_weight=exit_crystal_feature_weight,
        enable_math_adapter_lane=enable_math_adapter_lane,
        enable_math_summary_gate=enable_math_summary_gate,
        enable_compression_mhc=enable_compression_mhc,
        ct_modulation_mode=ct_modulation_mode,
        enable_reasoning_state_ring=enable_reasoning_state_ring,
        r_t_dim=r_t_dim,
        r_t_mode=r_t_mode,
        self_loop_awareness_mode=self_loop_awareness_mode,
        self_progress_shape_weight=self_progress_shape_weight,
        self_progress_trend_weight=self_progress_trend_weight,
        self_progress_plateau_weight=self_progress_plateau_weight,
        enable_progress_exit_readout=enable_progress_exit_readout,
        enable_backtrack_aware_progress=enable_backtrack_aware_progress,
        self_local_delta_consistency_weight=self_local_delta_consistency_weight,
        self_local_curvature_weight=self_local_curvature_weight,
        enable_dual_rate_self_predictor=enable_dual_rate_self_predictor,
        enable_trajectory_health_probe=enable_trajectory_health_probe,
        self_rollout_supervision_horizon=self_rollout_supervision_horizon,
        self_rollout_weighting_mode=self_rollout_weighting_mode,
        self_feature_span_mask_ratio=self_feature_span_mask_ratio,
        dynamics_experiment=dynamics_experiment,
        routing_chunk_size=routing_chunk_size,
        routing_topk_blocks=routing_topk_blocks,
        routing_topk_tokens=routing_topk_tokens,
        routing_top_p_coarse=routing_top_p_coarse,
        routing_top_p_fine=routing_top_p_fine,
        routing_budget_min=routing_budget_min,
        routing_budget_max=routing_budget_max,
        routing_weak_gain=routing_weak_gain,
        routing_strong_gain=routing_strong_gain,
        routing_local_floor=routing_local_floor,
        routing_modulation_floor=routing_modulation_floor,
        routing_modulation_ceiling=routing_modulation_ceiling,
        routing_world_summary_cap=routing_world_summary_cap,
        routing_tier_soft_only=routing_tier_soft_only,
        routing_tier_entropy_floor=routing_tier_entropy_floor,
        routing_min_local_share=routing_min_local_share,
        routing_tier_entropy_weight=routing_tier_entropy_weight,
        routing_min_local_share_weight=routing_min_local_share_weight,
        routing_progress_weight=routing_progress_weight,
        rollout_zone_weight=rollout_zone_weight,
        rollout_nonzero_low=rollout_nonzero_low,
        rollout_nonzero_high=rollout_nonzero_high,
        rollout_active_low=rollout_active_low,
        rollout_active_high=rollout_active_high,
        rollout_future_var_low=rollout_future_var_low,
        rollout_future_var_high=rollout_future_var_high,
        trajectory_vitality_weight=trajectory_vitality_weight,
        trajectory_c_t_drift_floor=trajectory_c_t_drift_floor,
        trajectory_world_drift_floor=trajectory_world_drift_floor,
        compression_dynamics_weight=compression_dynamics_weight,
        compression_block_drift_floor=compression_block_drift_floor,
        compression_block_var_floor=compression_block_var_floor,
        mamba_d_state=32,
        mamba_expand=2,
        mamba_headdim=32,
        mamba_chunk_size=16,
        swa_window=32,
        mhc_streams=4,
        mhc_sinkhorn_iters=8,
    )


def metric(path: Path, event: str, ok: bool, value: float, note: str) -> None:
    write_metric_jsonl(
        str(path),
        Stage0MetricRecord(
            event=event,
            ok=ok,
            value=float(value),
            note=note,
            timestamp=now(),
        ),
    )


def tail_float(values: list[float]) -> float:
    if not values:
        return 0.0
    width = max(1, min(2, len(values)))
    return float(sum(values[-width:]) / width)


def stage1_validate(model: LumaForCausalLM, samples: list[torch.Tensor], device: torch.device, metrics_path: Path) -> dict:
    """Stage 1 checks that the slow ring affects the main stream and does not collapse into silence.
    阶段1检查慢环是否真的影响主流，并且没有塌缩成沉默常数。
    """

    # We keep train-mode here on purpose so the current Mamba wrapper takes the stable SISO path.
    # Validation still stays deterministic because we disable gradients and the scaffold dropout is zero.
    # 这里刻意保持 train-mode，让当前 Mamba 包装层走稳定的 SISO 路径。
    # 验证仍然是确定性的，因为我们关闭了梯度，而且骨架默认 dropout 为零。
    model.train()
    model.model.exit_controller.score_threshold = 0.70
    model.model.exit_controller.delta_threshold = 0.55
    model.model.exit_controller.self_threshold = 0.55
    model.model.exit_controller.rollout_threshold = 0.60
    model.model.exit_controller.world_threshold = 0.55
    model.model.exit_controller.improvement_margin = 0.03
    model.model.exit_controller.use_sampling = False
    kls = []
    hidden_deltas = []
    hard_loops = []
    soft_loops = []
    ct_vectors = []
    exit_scores = []
    sampled_exit_scores = []
    improvements = []
    self_check_scores = []
    jepa_crystals = []
    uncertainties = []
    surprises = []
    state_variances = []
    ct_drifts = []
    world_summary_drifts = []
    c_t_delta_norm_means = []
    c_t_delta_norm_stds = []
    pred_delta_c_cos_adjacent_means = []
    math_lane_scores = []
    math_summary_gates = []
    r_t_drifts = []
    r_t_trusts = []
    progress_next_values = []
    progress_trend_values = []
    progress_plateau_values = []
    progress_rollout_pairs = []
    progress_exit_pairs = []
    exit_invalid_counts = []
    exit_clamped_ratios = []
    bernoulli_invalid_prevented = []
    nan_to_num_triggers = []
    sigreg_source_means = []
    sigreg_source_stds = []
    modulation_stats: dict[str, list[float]] = {}
    r_t_switches = []
    with torch.no_grad():
        for sample in samples:
            batch = sample.unsqueeze(0).to(device)
            out_on = model(input_ids=batch, labels=batch)
            aux_on = model.last_aux
            logits_on = out_on.logits.detach()
            hidden_on = out_on.hidden_states.detach()

            out_off = model(input_ids=batch, labels=batch, disable_ct_injection=True)
            logits_off = out_off.logits.detach()
            hidden_off = out_off.hidden_states.detach()

            p = torch.log_softmax(logits_on.float(), dim=-1)
            q = torch.log_softmax(logits_off.float(), dim=-1)
            kl = torch.sum(torch.exp(p) * (p - q), dim=-1).mean()
            hidden_delta = (hidden_on - hidden_off).norm(dim=-1).mean()

            kls.append(kl)
            hidden_deltas.append(hidden_delta)
            hard_loops.append(float(aux_on["executed_loops"]))
            ct_vectors.append(aux_on["c_t"].float())
            if aux_on["exit_score_history"]:
                exit_scores.append(torch.stack(aux_on["exit_score_history"]).float().mean())
            if aux_on["sampled_exit_score_history"]:
                sampled_exit_scores.append(torch.stack(aux_on["sampled_exit_score_history"]).float().mean())
            if aux_on["joint_benefit_history"]:
                improvements.append(torch.stack(aux_on["joint_benefit_history"]).float().mean())
            elif aux_on["two_step_improvement_history"]:
                improvements.append(torch.stack(aux_on["two_step_improvement_history"]).float().mean())
            if aux_on["self_check_score_history"]:
                self_check_scores.append(torch.stack(aux_on["self_check_score_history"]).float().mean())
            if aux_on.get("uncertainty_history"):
                uncertainties.append(torch.stack(aux_on["uncertainty_history"]).float().mean())
            if aux_on.get("jepa_crystal_history"):
                jepa_crystals.append(torch.stack(aux_on["jepa_crystal_history"]).float().mean())
            if aux_on.get("world_surprise_history"):
                surprises.append(torch.stack(aux_on["world_surprise_history"]).float().mean())
            if aux_on["loop_history"]:
                loop_stack = torch.stack([x.float().mean(dim=1) for x in aux_on["loop_history"]], dim=0)
                state_variances.append(loop_stack.var(dim=0, unbiased=False).mean())
            if len(aux_on["c_t_history"]) > 1:
                drifts = []
                for prev_state, next_state in zip(aux_on["c_t_history"][:-1], aux_on["c_t_history"][1:]):
                    drifts.append((next_state.float() - prev_state.float()).norm(dim=-1).mean())
                ct_drifts.append(torch.stack(drifts).mean())
            if len(aux_on.get("world_summary_history", [])) > 1:
                drifts = []
                for prev_summary, next_summary in zip(aux_on["world_summary_history"][:-1], aux_on["world_summary_history"][1:]):
                    drifts.append((next_summary.float() - prev_summary.float()).norm(dim=-1).mean())
                world_summary_drifts.append(torch.stack(drifts).mean())
            if aux_on.get("c_t_delta_norm_history"):
                delta_norm_hist = torch.stack(aux_on["c_t_delta_norm_history"]).float()
                c_t_delta_norm_means.append(delta_norm_hist.mean())
                c_t_delta_norm_stds.append(delta_norm_hist.std(unbiased=False))
            if aux_on.get("pred_delta_c_cos_adjacent_history"):
                pred_cos_hist = torch.stack(aux_on["pred_delta_c_cos_adjacent_history"]).float()
                pred_delta_c_cos_adjacent_means.append(pred_cos_hist.mean())
            if aux_on.get("math_lane_score_history"):
                math_lane_scores.append(torch.stack(aux_on["math_lane_score_history"]).float().mean())
            if aux_on.get("math_summary_gate_history"):
                math_summary_gates.append(torch.stack(aux_on["math_summary_gate_history"]).float().mean())
            if aux_on.get("r_t_trust_history"):
                r_t_trusts.append(torch.stack(aux_on["r_t_trust_history"]).float().mean())
            if aux_on.get("r_t_switch_history"):
                r_t_switches.append(torch.stack(aux_on["r_t_switch_history"]).float().mean())
            if len(aux_on.get("r_t_history", [])) > 1:
                drifts = []
                for prev_r, next_r in zip(aux_on["r_t_history"][:-1], aux_on["r_t_history"][1:]):
                    drifts.append((next_r.float() - prev_r.float()).norm(dim=-1).mean())
                r_t_drifts.append(torch.stack(drifts).mean())
            if aux_on.get("progress_next_history"):
                next_hist = torch.stack(aux_on["progress_next_history"]).float()
                trend_hist = torch.stack(aux_on.get("progress_trend_history", [])).float() if aux_on.get("progress_trend_history") else torch.zeros_like(next_hist)
                plateau_hist = torch.stack(aux_on.get("progress_plateau_history", [])).float() if aux_on.get("progress_plateau_history") else torch.zeros_like(next_hist)
                progress_next_values.append(next_hist.mean())
                progress_trend_values.append(trend_hist.mean())
                progress_plateau_values.append(plateau_hist.mean())
                if aux_on.get("rollout_error_history"):
                    rollout_mean = torch.stack(aux_on["rollout_error_history"]).float().mean()
                    progress_rollout_pairs.append((next_hist.mean(), rollout_mean))
                if aux_on.get("exit_score_history"):
                    exit_mean = torch.stack(aux_on["exit_score_history"]).float().mean()
                    progress_exit_pairs.append((next_hist.mean(), exit_mean))
            if aux_on.get("exit_score_preclamp_nonfinite_history"):
                exit_invalid_counts.append(torch.stack(aux_on["exit_score_preclamp_nonfinite_history"]).float().sum())
            if aux_on.get("exit_score_postfix_clamped_ratio_history"):
                exit_clamped_ratios.append(torch.stack(aux_on["exit_score_postfix_clamped_ratio_history"]).float().mean())
            if aux_on.get("bernoulli_invalid_prevented_history"):
                bernoulli_invalid_prevented.append(torch.stack(aux_on["bernoulli_invalid_prevented_history"]).float().sum())
            if aux_on.get("nan_to_num_trigger_history"):
                nan_to_num_triggers.append(torch.stack(aux_on["nan_to_num_trigger_history"]).float().sum())
            if "world_sigreg_source_mean" in aux_on:
                sigreg_source_means.append(aux_on["world_sigreg_source_mean"].float())
            if "world_sigreg_source_std" in aux_on:
                sigreg_source_stds.append(aux_on["world_sigreg_source_std"].float())
            if aux_on.get("dynamics_modulation_summary"):
                for key, value in aux_on["dynamics_modulation_summary"].items():
                    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
                    modulation_stats.setdefault(key, []).append(float(tensor.float().mean().item()))

            model.model.exit_controller.use_sampling = True
            model.model.exit_controller.sampling_temperature = 0.75
            _ = model(input_ids=batch, labels=batch)
            soft_loops.append(float(model.last_aux["executed_loops"]))
            model.model.exit_controller.use_sampling = False

    mean_kl = torch.stack(kls).mean().item()
    mean_hidden_delta = torch.stack(hidden_deltas).mean().item()
    hard_loop_var = torch.tensor(hard_loops).var(unbiased=False).item() if len(hard_loops) > 1 else 0.0
    soft_loop_var = torch.tensor(soft_loops).var(unbiased=False).item() if len(soft_loops) > 1 else 0.0
    c_t_var = torch.stack(ct_vectors).var(dim=0, unbiased=False).mean().item()
    exit_score_var = torch.stack(exit_scores).var(unbiased=False).item() if len(exit_scores) > 1 else 0.0
    sampled_exit_score_var = torch.stack(sampled_exit_scores).var(unbiased=False).item() if len(sampled_exit_scores) > 1 else 0.0
    improvement_mean = torch.stack(improvements).mean().item() if improvements else 0.0
    self_check_mean = torch.stack(self_check_scores).mean().item() if self_check_scores else 0.0
    jepa_crystal_mean = torch.stack(jepa_crystals).mean().item() if jepa_crystals else 0.0
    uncertainty_mean = torch.stack(uncertainties).mean().item() if uncertainties else 0.0
    surprise_mean = torch.stack(surprises).mean().item() if surprises else 0.0
    intermediate_state_variance = torch.stack(state_variances).mean().item() if state_variances else 0.0
    c_t_drift_mean = torch.stack(ct_drifts).mean().item() if ct_drifts else 0.0
    world_summary_drift_mean = torch.stack(world_summary_drifts).mean().item() if world_summary_drifts else 0.0
    world_summary_drift_std = torch.stack(world_summary_drifts).std(unbiased=False).item() if len(world_summary_drifts) > 1 else 0.0
    c_t_delta_norm_mean = torch.stack(c_t_delta_norm_means).mean().item() if c_t_delta_norm_means else 0.0
    c_t_delta_norm_std = torch.stack(c_t_delta_norm_stds).mean().item() if c_t_delta_norm_stds else 0.0
    pred_delta_c_cos_adjacent = torch.stack(pred_delta_c_cos_adjacent_means).mean().item() if pred_delta_c_cos_adjacent_means else 0.0
    math_lane_score_mean = torch.stack(math_lane_scores).mean().item() if math_lane_scores else 0.0
    math_summary_gate_mean = torch.stack(math_summary_gates).mean().item() if math_summary_gates else 0.0
    r_t_drift_mean = torch.stack(r_t_drifts).mean().item() if r_t_drifts else 0.0
    r_t_trust_mean = torch.stack(r_t_trusts).mean().item() if r_t_trusts else 0.0
    r_t_switch_mean = torch.stack(r_t_switches).mean().item() if r_t_switches else 0.0
    progress_next_mean = torch.stack(progress_next_values).mean().item() if progress_next_values else 0.0
    progress_next_std = torch.stack(progress_next_values).std(unbiased=False).item() if len(progress_next_values) > 1 else 0.0
    progress_trend_mean = torch.stack(progress_trend_values).mean().item() if progress_trend_values else 0.0
    progress_trend_std = torch.stack(progress_trend_values).std(unbiased=False).item() if len(progress_trend_values) > 1 else 0.0
    progress_plateau_mean = torch.stack(progress_plateau_values).mean().item() if progress_plateau_values else 0.0
    progress_plateau_std = torch.stack(progress_plateau_values).std(unbiased=False).item() if len(progress_plateau_values) > 1 else 0.0
    exit_invalid_count = torch.stack(exit_invalid_counts).sum().item() if exit_invalid_counts else 0.0
    exit_clamped_ratio = torch.stack(exit_clamped_ratios).mean().item() if exit_clamped_ratios else 0.0
    bernoulli_invalid_count = torch.stack(bernoulli_invalid_prevented).sum().item() if bernoulli_invalid_prevented else 0.0
    nan_to_num_count = torch.stack(nan_to_num_triggers).sum().item() if nan_to_num_triggers else 0.0
    sigreg_source_mean = torch.stack(sigreg_source_means).mean().item() if sigreg_source_means else 0.0
    sigreg_source_std = torch.stack(sigreg_source_stds).mean().item() if sigreg_source_stds else 0.0

    def _corr(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        if len(pairs) < 2:
            return 0.0
        x = torch.stack([item[0].float() for item in pairs])
        y = torch.stack([item[1].float() for item in pairs])
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.sqrt((x.pow(2).mean() * y.pow(2).mean()).clamp_min(1e-8))
        return float((x.mul(y).mean() / denom).item())

    progress_vs_rollout_corr = _corr(progress_rollout_pairs)
    progress_vs_exit_corr = _corr(progress_exit_pairs)
    modulation_summary = {
        key: {
            "mean": float(sum(values) / len(values)),
            "std": float(torch.tensor(values).std(unbiased=False).item()) if len(values) > 1 else 0.0,
        }
        for key, values in modulation_stats.items()
        if values
    }

    metric(metrics_path, "stage1_ct_kl", mean_kl > 1e-6, mean_kl, "KL between c_t-on and c_t-off logits / 打开关闭 c_t 注入后的 logits KL")
    metric(metrics_path, "stage1_ct_hidden_delta", mean_hidden_delta > 1e-5, mean_hidden_delta, "Hidden drift caused by c_t injection / c_t 注入导致的隐状态漂移")
    metric(metrics_path, "stage1_hard_loop_variance", hard_loop_var > 0.0, hard_loop_var, "Executed loop variance with hard exit / 硬退出下的循环步数方差")
    metric(metrics_path, "stage1_soft_loop_variance", soft_loop_var > 0.0, soft_loop_var, "Executed loop variance with sampled exit / 采样退出下的循环步数方差")
    metric(metrics_path, "stage1_ct_variance", c_t_var > 1e-6, c_t_var, "c_t variance should stay above collapse floor / c_t 方差不能塌到近零")
    metric(metrics_path, "stage1_exit_score_variance", exit_score_var > 0.0, exit_score_var, "Exit score variance should be non-zero / 退出分数方差应为非零")
    metric(metrics_path, "stage1_sampled_exit_score_variance", sampled_exit_score_var > 0.0, sampled_exit_score_var, "Sampled exit score variance should be non-zero / 采样退出分数方差应为非零")
    metric(metrics_path, "stage1_two_step_improvement", True, improvement_mean, "Mean joint one-step benefit signal / 联合一步收益均值")
    metric(metrics_path, "stage1_self_check_mean", True, self_check_mean, "Mean self-check score / 自检分数均值")
    metric(metrics_path, "stage1_jepa_crystal_mean", True, jepa_crystal_mean, "Mean JEPA-guided entropy crystal signal / JEPA 引导熵结晶信号均值")
    metric(metrics_path, "stage1_uncertainty_mean", True, uncertainty_mean, "Mean introspection uncertainty signal / 自省疑惑度均值")
    metric(metrics_path, "stage1_world_surprise_mean", True, surprise_mean, "Mean world surprise signal / world surprise 信号均值")
    metric(metrics_path, "stage1_intermediate_state_variance", True, intermediate_state_variance, "Intermediate state variance / 中间状态方差")
    metric(metrics_path, "stage1_c_t_drift_mean", True, c_t_drift_mean, "Mean c_t drift / c_t 漂移均值")
    metric(metrics_path, "stage1_world_summary_drift_mean", True, world_summary_drift_mean, "Mean world summary drift / world summary 漂移均值")
    metric(metrics_path, "stage1_c_t_delta_norm_mean", True, c_t_delta_norm_mean, "Mean c_t delta norm / c_t 相邻步差分范数均值")
    metric(metrics_path, "stage1_pred_delta_c_cos_adjacent", True, pred_delta_c_cos_adjacent, "Mean adjacent cosine of predicted delta_c / 相邻预测 delta_c 余弦均值")
    metric(metrics_path, "stage1_math_lane_score_mean", True, math_lane_score_mean, "Mean math adapter lane score / math adapter lane 分数均值")
    metric(metrics_path, "stage1_math_summary_gate_mean", True, math_summary_gate_mean, "Mean math summary gate / math summary gate 均值")
    metric(metrics_path, "stage1_r_t_drift_mean", True, r_t_drift_mean, "Mean r_t drift / r_t 漂移均值")
    metric(metrics_path, "stage1_r_t_trust_mean", True, r_t_trust_mean, "Mean r_t trust / r_t 信任均值")
    metric(metrics_path, "stage1_r_t_switch_mean", True, r_t_switch_mean, "Mean c_t/r_t switch gate / c_t/r_t 切换门均值")
    metric(metrics_path, "stage1_progress_vs_rollout_corr", True, progress_vs_rollout_corr, "Correlation between progress_next and rollout error / progress_next 与 rollout 误差相关性")
    metric(metrics_path, "stage1_progress_vs_exit_corr", True, progress_vs_exit_corr, "Correlation between progress_next and exit score / progress_next 与 exit 分数相关性")
    metric(metrics_path, "stage1_exit_invalid_count", exit_invalid_count <= 0.0, exit_invalid_count, "Invalid exit-score count should stay zero / exit 非有限计数应为零")
    metric(metrics_path, "stage1_nan_to_num_trigger_count", True, nan_to_num_count, "nan_to_num trigger count / nan_to_num 触发次数")
    metric(metrics_path, "stage1_sigreg_source_mean", True, sigreg_source_mean, "Mean SIGReg source activation / SIGReg 输入均值")
    metric(metrics_path, "stage1_sigreg_source_std", True, sigreg_source_std, "Mean SIGReg source std / SIGReg 输入标准差")

    return {
        "mean_kl": mean_kl,
        "mean_hidden_delta": mean_hidden_delta,
        "hard_loop_var": hard_loop_var,
        "soft_loop_var": soft_loop_var,
        "c_t_var": c_t_var,
        "exit_score_var": exit_score_var,
        "sampled_exit_score_var": sampled_exit_score_var,
        "two_step_improvement_mean": improvement_mean,
        "self_check_mean": self_check_mean,
        "jepa_crystal_mean": jepa_crystal_mean,
        "uncertainty_mean": uncertainty_mean,
        "world_surprise_mean": surprise_mean,
        "intermediate_state_variance": intermediate_state_variance,
        "c_t_drift_mean": c_t_drift_mean,
        "world_summary_drift_mean": world_summary_drift_mean,
        "world_summary_drift_std": world_summary_drift_std,
        "c_t_delta_norm_mean": c_t_delta_norm_mean,
        "c_t_delta_norm_std": c_t_delta_norm_std,
        "pred_delta_c_cos_adjacent": pred_delta_c_cos_adjacent,
        "math_lane_score_mean": math_lane_score_mean,
        "math_summary_gate_mean": math_summary_gate_mean,
        "r_t_drift_mean": r_t_drift_mean,
        "r_t_trust_mean": r_t_trust_mean,
        "r_t_switch_mean": r_t_switch_mean,
        "progress_next_mean": progress_next_mean,
        "progress_next_std": progress_next_std,
        "progress_trend_mean": progress_trend_mean,
        "progress_trend_std": progress_trend_std,
        "progress_plateau_mean": progress_plateau_mean,
        "progress_plateau_std": progress_plateau_std,
        "progress_vs_rollout_corr": progress_vs_rollout_corr,
        "progress_vs_exit_corr": progress_vs_exit_corr,
        "exit_invalid_count": exit_invalid_count,
        "exit_score_postfix_clamped_ratio": exit_clamped_ratio,
        "bernoulli_invalid_prevented_count": bernoulli_invalid_count,
        "nan_to_num_trigger_count": nan_to_num_count,
        "sigreg_source_mean": sigreg_source_mean,
        "sigreg_source_std": sigreg_source_std,
        "modulation_summary": modulation_summary,
    }


def stage2_validate(model: LumaForCausalLM, samples: list[torch.Tensor], device: torch.device, metrics_path: Path, steps: int) -> dict:
    """Stage 2 applies short training pressure and checks whether residual Self-JEPA starts behaving like a learnable signal.
    阶段2施加短程训练压力，检查残差 Self-JEPA 是否开始表现出可学习信号。
    """

    model.train()
    optimizer = LumaMuonAdamWOptimizer(
        model,
        LumaOptimizerConfig(
            matrix_lr=0.02,
            scalar_lr=2e-4,
            muon_momentum=0.95,
            weight_decay=0.01,
            muon_clip_factor=1.5,
            modular_norm_power=0.5,
        ),
    )
    losses = []
    self_losses = []
    rollout_losses = []
    rollout_active_flags = []
    rollout_nonzero_flags = []
    delta_norms = []
    rollout_zone_losses = []
    routing_entropy_losses = []
    trajectory_vitality_losses = []
    compression_dynamics_losses = []
    compression_block_drifts = []
    compression_block_vars = []
    sigreg_world_losses = []
    sigreg_rollout_losses = []
    sigreg_delta_losses = []
    sigreg_world_steps = []
    sigreg_source_means = []
    sigreg_source_stds = []
    sigreg_world_steps = []
    sigreg_source_means = []
    sigreg_source_stds = []
    grad_norm_totals = []
    grad_norm_world_encoders = []
    aborted_on_nonfinite = False
    first_nonfinite_step: int | None = None

    def _grad_norm(params) -> float:
        sq_sum = 0.0
        has_grad = False
        for param in params:
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            sq_sum += float(grad.pow(2).sum().item())
            has_grad = True
        if not has_grad:
            return 0.0
        return math.sqrt(max(0.0, sq_sum))

    for step in range(steps):
        batch = samples[step % len(samples)].unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        aux = model.last_aux
        self_tensor = aux["self_jepa_loss"].detach()
        rollout_tensor = aux["self_rollout_loss"].detach()
        target_delta = aux["target_delta_c"].detach()
        rollout_zone_tensor = aux.get("rollout_activity_zone_loss", torch.zeros_like(loss.detach())).detach()
        routing_entropy_tensor = aux.get("routing_entropy_loss", torch.zeros_like(loss.detach())).detach()
        trajectory_vitality_tensor = aux.get("trajectory_vitality_loss", torch.zeros_like(loss.detach())).detach()
        compression_dynamics_tensor = aux.get("compression_dynamics_loss", torch.zeros_like(loss.detach())).detach()
        compression_block_drift_tensor = aux.get(
            "compression_block_drift_mean_internal",
            aux.get("compression_block_drift_mean", torch.zeros_like(loss.detach())),
        ).detach()
        compression_block_var_tensor = aux.get(
            "compression_block_var_mean_internal",
            aux.get("compression_block_var_mean", torch.zeros_like(loss.detach())),
        ).detach()
        sigreg_world_tensor = aux.get("world_sigreg_loss", torch.zeros_like(loss.detach())).detach()
        sigreg_rollout_tensor = aux.get("sigreg_rollout_loss", torch.zeros_like(loss.detach())).detach()
        sigreg_delta_tensor = aux.get("sigreg_delta_loss", torch.zeros_like(loss.detach())).detach()
        sigreg_step_tensor = aux.get("world_sigreg_loss_step", torch.full_like(loss.detach(), -1.0)).detach()
        sigreg_source_mean_tensor = aux.get("world_sigreg_source_mean", torch.zeros_like(loss.detach())).detach()
        sigreg_source_std_tensor = aux.get("world_sigreg_source_std", torch.zeros_like(loss.detach())).detach()
        finite_ok = bool(
            torch.isfinite(loss.detach()).item()
            and torch.isfinite(self_tensor).item()
            and torch.isfinite(rollout_tensor).item()
            and torch.isfinite(target_delta).all().item()
            and torch.isfinite(compression_dynamics_tensor).item()
            and torch.isfinite(compression_block_drift_tensor).item()
            and torch.isfinite(compression_block_var_tensor).item()
        )
        if not finite_ok:
            losses.append(float("nan"))
            self_losses.append(float("nan"))
            rollout_losses.append(float("nan"))
            rollout_active_flags.append(1.0 if aux.get("rollout_error_history") else 0.0)
            rollout_nonzero_flags.append(0.0)
            delta_norms.append(float("nan"))
            rollout_zone_losses.append(float("nan"))
            routing_entropy_losses.append(float("nan"))
            trajectory_vitality_losses.append(float("nan"))
            compression_dynamics_losses.append(float("nan"))
            compression_block_drifts.append(float("nan"))
            compression_block_vars.append(float("nan"))
            sigreg_world_losses.append(float("nan"))
            sigreg_rollout_losses.append(float("nan"))
            sigreg_delta_losses.append(float("nan"))
            sigreg_world_steps.append(float("nan"))
            sigreg_source_means.append(float("nan"))
            sigreg_source_stds.append(float("nan"))
            grad_norm_totals.append(float("nan"))
            grad_norm_world_encoders.append(float("nan"))
            aborted_on_nonfinite = True
            if first_nonfinite_step is None:
                first_nonfinite_step = step
            break
        loss.backward()
        grad_norm_totals.append(_grad_norm(model.parameters()))
        world_module = model.model.world_latent_jepa
        if hasattr(world_module, "online_encoder"):
            grad_norm_world_encoders.append(_grad_norm(world_module.online_encoder.parameters()))
        elif hasattr(world_module, "online_observer"):
            grad_norm_world_encoders.append(_grad_norm([world_module.online_observer.weight]))
        else:
            grad_norm_world_encoders.append(0.0)
        optimizer.step()
        model.model.update_world_jepa_ema()

        losses.append(float(loss.detach()))
        self_losses.append(float(self_tensor))
        rollout_value = float(rollout_tensor)
        rollout_losses.append(rollout_value)
        rollout_active_flags.append(1.0 if aux.get("rollout_error_history") else 0.0)
        rollout_nonzero_flags.append(1.0 if abs(rollout_value) > 1e-12 else 0.0)
        delta_norms.append(float(target_delta.norm(dim=-1).mean().detach()))
        rollout_zone_losses.append(float(rollout_zone_tensor))
        routing_entropy_losses.append(float(routing_entropy_tensor))
        trajectory_vitality_losses.append(float(trajectory_vitality_tensor))
        compression_dynamics_losses.append(float(compression_dynamics_tensor))
        compression_block_drifts.append(float(compression_block_drift_tensor))
        compression_block_vars.append(float(compression_block_var_tensor))
        sigreg_world_losses.append(float(sigreg_world_tensor))
        sigreg_rollout_losses.append(float(sigreg_rollout_tensor))
        sigreg_delta_losses.append(float(sigreg_delta_tensor))
        sigreg_world_steps.append(float(sigreg_step_tensor))
        sigreg_source_means.append(float(sigreg_source_mean_tensor))
        sigreg_source_stds.append(float(sigreg_source_std_tensor))

    head = sum(self_losses[: max(1, min(2, len(self_losses)))]) / max(1, min(2, len(self_losses)))
    tail = sum(self_losses[-max(1, min(2, len(self_losses))):]) / max(1, min(2, len(self_losses)))
    rollout_head = sum(rollout_losses[: max(1, min(2, len(rollout_losses)))]) / max(1, min(2, len(rollout_losses)))
    rollout_tail = sum(rollout_losses[-max(1, min(2, len(rollout_losses))):]) / max(1, min(2, len(rollout_losses)))
    finite_delta_norms = [value for value in delta_norms if math.isfinite(value)]
    mean_delta_norm = (sum(finite_delta_norms) / len(finite_delta_norms)) if finite_delta_norms else float("nan")
    rollout_active_ratio = sum(rollout_active_flags) / len(rollout_active_flags) if rollout_active_flags else 0.0
    rollout_nonzero_ratio = sum(rollout_nonzero_flags) / len(rollout_nonzero_flags) if rollout_nonzero_flags else 0.0

    metric(metrics_path, "stage2_self_jepa_start", True, head, "Initial short-run Self-JEPA loss / 初始短程 Self-JEPA 损失")
    metric(metrics_path, "stage2_self_jepa_end", tail <= head * 1.10, tail, "Ending Self-JEPA loss should not explode / 结束时 Self-JEPA 损失不应爆炸")
    metric(metrics_path, "stage2_self_rollout_end", rollout_tail <= rollout_head * 1.25 if rollout_head > 0 else True, rollout_tail, "Self-rollout loss should stay bounded / Self-rollout 损失应保持有界")
    metric(metrics_path, "stage2_rollout_active_ratio", rollout_active_ratio > 0.0, rollout_active_ratio, "Rollout supervision should activate at least sometimes / rollout 监督至少应在部分 step 上激活")
    metric(metrics_path, "stage2_rollout_nonzero_ratio", True, rollout_nonzero_ratio, "Fraction of non-zero rollout losses / 非零 rollout 损失占比")
    metric(metrics_path, "stage2_delta_norm", mean_delta_norm > 1e-5, mean_delta_norm, "Target delta_c norm should stay non-zero / 目标 delta_c 范数不能接近零")
    metric(metrics_path, "stage2_nonfinite_abort", not aborted_on_nonfinite, 1.0 if aborted_on_nonfinite else 0.0, "Stage2 should avoid non-finite aborts / Stage2 不应因非有限值中止")
    finite_zone = [value for value in rollout_zone_losses if math.isfinite(value)]
    finite_entropy = [value for value in routing_entropy_losses if math.isfinite(value)]
    finite_vitality = [value for value in trajectory_vitality_losses if math.isfinite(value)]
    finite_compression = [value for value in compression_dynamics_losses if math.isfinite(value)]
    finite_compression_drift = [value for value in compression_block_drifts if math.isfinite(value)]
    finite_compression_var = [value for value in compression_block_vars if math.isfinite(value)]
    zone_tail = tail_float(finite_zone)
    entropy_tail = tail_float(finite_entropy)
    vitality_tail = tail_float(finite_vitality)
    compression_tail = tail_float(finite_compression)
    compression_drift_tail = tail_float(finite_compression_drift)
    compression_var_tail = tail_float(finite_compression_var)
    world_sigreg_tail = tail_float([value for value in sigreg_world_losses if math.isfinite(value)])
    rollout_sigreg_tail = tail_float([value for value in sigreg_rollout_losses if math.isfinite(value)])
    delta_sigreg_tail = tail_float([value for value in sigreg_delta_losses if math.isfinite(value)])
    finite_world_sigreg = [value for value in sigreg_world_losses if math.isfinite(value)]
    world_sigreg_head = tail_float(finite_world_sigreg[: max(1, min(2, len(finite_world_sigreg)))]) if finite_world_sigreg else 0.0
    world_sigreg_max = max(finite_world_sigreg) if finite_world_sigreg else 0.0
    sigreg_source_mean = tail_float([value for value in sigreg_source_means if math.isfinite(value)])
    sigreg_source_std = tail_float([value for value in sigreg_source_stds if math.isfinite(value)])
    grad_norm_total_tail = tail_float([value for value in grad_norm_totals if math.isfinite(value)])
    grad_norm_world_encoder_tail = tail_float([value for value in grad_norm_world_encoders if math.isfinite(value)])
    valid_sigreg_steps = [int(value) for value in sigreg_world_steps if math.isfinite(value) and value >= 0]
    world_sigreg_loss_step = min(valid_sigreg_steps) if valid_sigreg_steps else -1
    metric(metrics_path, "stage2_rollout_zone_loss_tail", True, zone_tail, "Tail rollout zone loss / rollout 活性区间损失尾值")
    metric(metrics_path, "stage2_routing_entropy_loss_tail", True, entropy_tail, "Tail routing entropy loss / routing 熵损失尾值")
    metric(metrics_path, "stage2_trajectory_vitality_loss_tail", True, vitality_tail, "Tail trajectory vitality loss / 轨迹活性损失尾值")
    metric(metrics_path, "stage2_compression_dynamics_loss_tail", True, compression_tail, "Tail compression dynamics loss / 压缩区动力学约束损失尾值")
    metric(metrics_path, "stage2_compression_block_drift_tail", True, compression_drift_tail, "Tail compression block drift / 压缩区块漂移尾值")
    metric(metrics_path, "stage2_compression_block_var_tail", True, compression_var_tail, "Tail compression block variance / 压缩区块方差尾值")
    metric(metrics_path, "stage2_world_sigreg_loss_tail", True, world_sigreg_tail, "Tail world SIGReg loss / world SIGReg 损失尾值")
    metric(metrics_path, "stage2_world_sigreg_loss_head", True, world_sigreg_head, "Head world SIGReg loss / world SIGReg 损失头值")
    metric(metrics_path, "stage2_world_sigreg_loss_max", True, world_sigreg_max, "Max world SIGReg loss / world SIGReg 损失最大值")
    metric(metrics_path, "stage2_rollout_sigreg_loss_tail", True, rollout_sigreg_tail, "Tail rollout SIGReg loss / rollout SIGReg 损失尾值")
    metric(metrics_path, "stage2_delta_sigreg_loss_tail", True, delta_sigreg_tail, "Tail delta SIGReg loss / delta SIGReg 损失尾值")
    metric(metrics_path, "stage2_sigreg_source_mean", True, sigreg_source_mean, "Tail SIGReg source mean / SIGReg 输入均值尾值")
    metric(metrics_path, "stage2_sigreg_source_std", True, sigreg_source_std, "Tail SIGReg source std / SIGReg 输入标准差尾值")
    metric(metrics_path, "stage2_grad_norm_total_tail", True, grad_norm_total_tail, "Tail total grad norm / 总梯度范数尾值")
    metric(metrics_path, "stage2_grad_norm_world_encoder_tail", True, grad_norm_world_encoder_tail, "Tail world encoder grad norm / world encoder 梯度范数尾值")
    metric(metrics_path, "stage2_first_nonfinite_step", first_nonfinite_step is None, -1.0 if first_nonfinite_step is None else float(first_nonfinite_step), "First non-finite step index / 首次非有限值 step")

    return {
        "losses": losses,
        "self_losses": self_losses,
        "rollout_losses": rollout_losses,
        "mean_delta_norm": mean_delta_norm,
        "rollout_active_ratio": rollout_active_ratio,
        "rollout_nonzero_ratio": rollout_nonzero_ratio,
        "self_loss_head": head,
        "self_loss_tail": tail,
        "self_rollout_head": rollout_head,
        "self_rollout_tail": rollout_tail,
        "rollout_zone_loss_tail": zone_tail,
        "routing_entropy_loss_tail": entropy_tail,
        "trajectory_vitality_loss_tail": vitality_tail,
        "compression_dynamics_loss_tail": compression_tail,
        "compression_block_drift_tail": compression_drift_tail,
        "compression_block_var_tail": compression_var_tail,
        "world_sigreg_loss_tail": world_sigreg_tail,
        "world_sigreg_loss_head": world_sigreg_head,
        "world_sigreg_loss_max": world_sigreg_max,
        "world_sigreg_loss_step": world_sigreg_loss_step,
        "rollout_sigreg_loss_tail": rollout_sigreg_tail,
        "delta_sigreg_loss_tail": delta_sigreg_tail,
        "world_sigreg_source_mean": sigreg_source_mean,
        "world_sigreg_source_std": sigreg_source_std,
        "grad_norm_total_tail": grad_norm_total_tail,
        "grad_norm_world_encoder_tail": grad_norm_world_encoder_tail,
        "first_nonfinite_step": first_nonfinite_step,
        "nonfinite_abort": aborted_on_nonfinite,
    }


def bucket_probe_from_mixed_model(
    model: LumaForCausalLM,
    samples: list[torch.Tensor],
    device: torch.device,
    metrics_path: Path,
    bucket_name: str = "mixed",
    math_probe_rollout_steps: int = 0,
    math_probe_reason_loops: int = 0,
    math_probe_score_threshold: float = 0.0,
    math_probe_min_loops: int = 0,
) -> dict:
    """Luma reads each bucket through the same mixed-trained model so bucket scores reflect one shared training world.
    Luma 用同一个 mixed 训练后的模型去读每个桶，这样单桶分数才真正来自同一条 mixed 主线。
    """

    original_rollout = model.model.config.self_rollout_steps
    original_loops = model.model.config.reason_active_loops
    original_reason_loops = model.model.config.reason_loops
    original_score_threshold = model.model.exit_controller.score_threshold
    original_min_loops = model.model.exit_controller.min_loops
    if bucket_name == "math":
        if math_probe_rollout_steps > 0:
            model.model.config.self_rollout_steps = math_probe_rollout_steps
        if math_probe_reason_loops > 0:
            model.model.config.reason_active_loops = math_probe_reason_loops
            model.model.config.reason_loops = math_probe_reason_loops
        if math_probe_score_threshold > 0:
            model.model.exit_controller.score_threshold = math_probe_score_threshold
        if math_probe_min_loops > 0:
            model.model.exit_controller.min_loops = math_probe_min_loops

    stage1 = stage1_validate(model, samples, device, metrics_path)

    model.train()
    losses = []
    self_losses = []
    rollout_losses = []
    rollout_active_flags = []
    rollout_nonzero_flags = []
    delta_norms = []
    rollout_zone_losses = []
    routing_entropy_losses = []
    trajectory_vitality_losses = []
    sigreg_world_losses = []
    sigreg_rollout_losses = []
    sigreg_delta_losses = []
    sigreg_world_steps = []
    sigreg_source_means = []
    sigreg_source_stds = []
    surprises = []
    state_variances = []
    ct_drifts = []
    world_summary_drifts = []
    math_lane_scores = []
    math_summary_gates = []
    r_t_drifts = []
    r_t_trusts = []
    uncertainties = []
    progress_next_values = []
    progress_trend_values = []
    progress_plateau_values = []
    exit_invalid_counts = []
    nan_to_num_triggers = []
    modulation_stats: dict[str, list[float]] = {}
    with torch.no_grad():
        for sample in samples:
            batch = sample.unsqueeze(0).to(device)
            outputs = model(input_ids=batch, labels=batch)
            aux = model.last_aux
            losses.append(float(outputs.loss.detach()))
            self_losses.append(float(aux["self_jepa_loss"].detach()))
            rollout_value = float(aux["self_rollout_loss"].detach())
            rollout_losses.append(rollout_value)
            rollout_active_flags.append(1.0 if aux.get("rollout_error_history") else 0.0)
            rollout_nonzero_flags.append(1.0 if abs(rollout_value) > 1e-12 else 0.0)
            delta_norms.append(float(aux["target_delta_c"].norm(dim=-1).mean().detach()))
            rollout_zone_losses.append(float(aux.get("rollout_activity_zone_loss", torch.zeros((), device=device)).detach()))
            routing_entropy_losses.append(float(aux.get("routing_entropy_loss", torch.zeros((), device=device)).detach()))
            trajectory_vitality_losses.append(float(aux.get("trajectory_vitality_loss", torch.zeros((), device=device)).detach()))
            sigreg_world_losses.append(float(aux.get("world_sigreg_loss", torch.zeros((), device=device)).detach()))
            sigreg_rollout_losses.append(float(aux.get("sigreg_rollout_loss", torch.zeros((), device=device)).detach()))
            sigreg_delta_losses.append(float(aux.get("sigreg_delta_loss", torch.zeros((), device=device)).detach()))
            sigreg_world_steps.append(float(aux.get("world_sigreg_loss_step", torch.full((1,), -1.0, device=device)).detach().mean()))
            sigreg_source_means.append(float(aux.get("world_sigreg_source_mean", torch.zeros((), device=device)).detach()))
            sigreg_source_stds.append(float(aux.get("world_sigreg_source_std", torch.zeros((), device=device)).detach()))
            if aux.get("world_surprise_history"):
                surprises.append(float(torch.stack(aux["world_surprise_history"]).float().mean().detach()))
            if aux.get("uncertainty_history"):
                uncertainties.append(float(torch.stack(aux["uncertainty_history"]).float().mean().detach()))
            if aux["loop_history"]:
                loop_stack = torch.stack([x.float().mean(dim=1) for x in aux["loop_history"]], dim=0)
                state_variances.append(float(loop_stack.var(dim=0, unbiased=False).mean().detach()))
            if len(aux["c_t_history"]) > 1:
                drifts = []
                for prev_state, next_state in zip(aux["c_t_history"][:-1], aux["c_t_history"][1:]):
                    drifts.append((next_state.float() - prev_state.float()).norm(dim=-1).mean())
                ct_drifts.append(float(torch.stack(drifts).mean().detach()))
            if len(aux.get("world_summary_history", [])) > 1:
                drifts = []
                for prev_summary, next_summary in zip(aux["world_summary_history"][:-1], aux["world_summary_history"][1:]):
                    drifts.append((next_summary.float() - prev_summary.float()).norm(dim=-1).mean())
                world_summary_drifts.append(float(torch.stack(drifts).mean().detach()))
            if aux.get("progress_next_history"):
                progress_next_values.append(float(torch.stack(aux["progress_next_history"]).float().mean().detach()))
            if aux.get("progress_trend_history"):
                progress_trend_values.append(float(torch.stack(aux["progress_trend_history"]).float().mean().detach()))
            if aux.get("progress_plateau_history"):
                progress_plateau_values.append(float(torch.stack(aux["progress_plateau_history"]).float().mean().detach()))
            if aux.get("exit_score_preclamp_nonfinite_history"):
                exit_invalid_counts.append(float(torch.stack(aux["exit_score_preclamp_nonfinite_history"]).float().sum().detach()))
            if aux.get("nan_to_num_trigger_history"):
                nan_to_num_triggers.append(float(torch.stack(aux["nan_to_num_trigger_history"]).float().sum().detach()))
            if aux.get("dynamics_modulation_summary"):
                for key, value in aux["dynamics_modulation_summary"].items():
                    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
                    modulation_stats.setdefault(key, []).append(float(tensor.float().mean().item()))

    def _edge_mean(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        width = max(1, min(2, len(values)))
        head = sum(values[:width]) / width
        tail = sum(values[-width:]) / width
        return head, tail

    self_head, self_tail = _edge_mean(self_losses)
    rollout_head, rollout_tail = _edge_mean(rollout_losses)
    mean_delta_norm = sum(delta_norms) / len(delta_norms) if delta_norms else 0.0
    rollout_active_ratio = sum(rollout_active_flags) / len(rollout_active_flags) if rollout_active_flags else 0.0
    rollout_nonzero_ratio = sum(rollout_nonzero_flags) / len(rollout_nonzero_flags) if rollout_nonzero_flags else 0.0
    mean_surprise = sum(surprises) / len(surprises) if surprises else 0.0
    mean_state_variance = sum(state_variances) / len(state_variances) if state_variances else 0.0
    mean_ct_drift = sum(ct_drifts) / len(ct_drifts) if ct_drifts else 0.0
    mean_world_summary_drift = sum(world_summary_drifts) / len(world_summary_drifts) if world_summary_drifts else 0.0
    mean_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
    progress_next_mean = sum(progress_next_values) / len(progress_next_values) if progress_next_values else 0.0
    progress_trend_mean = sum(progress_trend_values) / len(progress_trend_values) if progress_trend_values else 0.0
    progress_plateau_mean = sum(progress_plateau_values) / len(progress_plateau_values) if progress_plateau_values else 0.0
    exit_invalid_count = sum(exit_invalid_counts) if exit_invalid_counts else 0.0
    nan_to_num_count = sum(nan_to_num_triggers) if nan_to_num_triggers else 0.0
    modulation_summary = {
        key: {
            "mean": float(sum(values) / len(values)),
            "std": float(torch.tensor(values).std(unbiased=False).item()) if len(values) > 1 else 0.0,
        }
        for key, values in modulation_stats.items()
        if values
    }

    stage2_probe = {
        "losses": losses,
        "self_losses": self_losses,
        "rollout_losses": rollout_losses,
        "mean_delta_norm": mean_delta_norm,
        "rollout_active_ratio": rollout_active_ratio,
        "rollout_nonzero_ratio": rollout_nonzero_ratio,
        "self_loss_head": self_head,
        "self_loss_tail": self_tail,
        "self_rollout_head": rollout_head,
        "self_rollout_tail": rollout_tail,
        "rollout_zone_loss_tail": tail_float([x for x in rollout_zone_losses if math.isfinite(x)]),
        "routing_entropy_loss_tail": tail_float([x for x in routing_entropy_losses if math.isfinite(x)]),
        "trajectory_vitality_loss_tail": tail_float([x for x in trajectory_vitality_losses if math.isfinite(x)]),
        "world_sigreg_loss_tail": tail_float([x for x in sigreg_world_losses if math.isfinite(x)]),
        "world_sigreg_loss_head": tail_float([x for x in sigreg_world_losses[: max(1, min(2, len(sigreg_world_losses)))] if math.isfinite(x)]),
        "world_sigreg_loss_max": max([x for x in sigreg_world_losses if math.isfinite(x)], default=0.0),
        "world_sigreg_loss_step": min([int(x) for x in sigreg_world_steps if math.isfinite(x) and x >= 0], default=-1),
        "rollout_sigreg_loss_tail": tail_float([x for x in sigreg_rollout_losses if math.isfinite(x)]),
        "delta_sigreg_loss_tail": tail_float([x for x in sigreg_delta_losses if math.isfinite(x)]),
        "world_sigreg_source_mean": tail_float([x for x in sigreg_source_means if math.isfinite(x)]),
        "world_sigreg_source_std": tail_float([x for x in sigreg_source_stds if math.isfinite(x)]),
        "world_surprise_mean": mean_surprise,
        "intermediate_state_variance": mean_state_variance,
        "c_t_drift_mean": mean_ct_drift,
        "world_summary_drift_mean": mean_world_summary_drift,
        "uncertainty_mean": mean_uncertainty,
        "progress_next_mean": progress_next_mean,
        "progress_trend_mean": progress_trend_mean,
        "progress_plateau_mean": progress_plateau_mean,
        "exit_invalid_count": exit_invalid_count,
        "nan_to_num_trigger_count": nan_to_num_count,
        "modulation_summary": modulation_summary,
        "probe_from_mixed_model": True,
    }

    model.model.config.self_rollout_steps = original_rollout
    model.model.config.reason_active_loops = original_loops
    model.model.config.reason_loops = original_reason_loops
    model.model.exit_controller.score_threshold = original_score_threshold
    model.model.exit_controller.min_loops = original_min_loops

    return {"stage1": stage1, "stage2": stage2_probe}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Luma stage1/stage2 validation with a tiny public text fixture.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Force model parameters and activations to float32 on CUDA for strict numeric stability checks.",
    )
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--stage2-steps", type=int, default=8)
    parser.add_argument("--metrics-out", type=str, default=str(ROOT / "artifacts" / "stage0_metrics.jsonl"))
    parser.add_argument("--json-out", type=str, default=str(ROOT / "artifacts" / "stage12_report.json"))
    parser.add_argument("--candidate-name", type=str, default="")
    parser.add_argument("--load-checkpoint", type=str, default="")
    parser.add_argument("--save-checkpoint", type=str, default="")
    parser.add_argument(
        "--fixture-mode",
        choices=[
            "math_dialogue",
            "hard_math_dialogue",
            "math_dialogue_emotion",
            "hard_math_dialogue_emotion",
            "competition_math_dialogue",
            "competition_math_dialogue_emotion",
        ],
        default="math_dialogue",
    )
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--slow-k", type=int, default=2)
    parser.add_argument("--reason-loops", type=int, default=0)
    parser.add_argument("--disable-world-jepa", action="store_true")
    parser.add_argument("--world-jepa-mode", choices=["scaffold", "full"], default="scaffold")
    parser.add_argument("--enable-sigreg-world", action="store_true")
    parser.add_argument("--enable-sigreg-rollout", action="store_true")
    parser.add_argument("--enable-sigreg-delta", action="store_true")
    parser.add_argument("--sigreg-world-source", choices=["sigreg_on_online", "sigreg_on_encoder_latent"], default="sigreg_on_online")
    parser.add_argument("--sigreg-world-fp32-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sigreg-world-warmup-steps", type=int, default=0)
    parser.add_argument("--world-sigreg-weight", type=float, default=0.05)
    parser.add_argument("--world-sigreg-num-slices", type=int, default=128)
    parser.add_argument("--world-sigreg-t-min", type=float, default=0.2)
    parser.add_argument("--world-sigreg-t-max", type=float, default=4.0)
    parser.add_argument("--world-sigreg-num-points", type=int, default=17)
    parser.add_argument("--world-sigreg-lambda", type=float, default=1.0)
    parser.add_argument("--world-delta-weight", type=float, default=0.10)
    parser.add_argument("--sigreg-rollout-weight", type=float, default=0.05)
    parser.add_argument("--sigreg-delta-weight", type=float, default=0.05)
    parser.add_argument("--world-jepa-weight", type=float, default=1.0)
    parser.add_argument("--self-jepa-weight", type=float, default=1.0)
    parser.add_argument("--self-rollout-weight", type=float, default=0.5)
    parser.add_argument("--exit-aux-weight", type=float, default=0.01)
    parser.add_argument("--disable-self-jepa", action="store_true")
    parser.add_argument("--enable-self-check-ring", action="store_true")
    parser.add_argument("--self-check-k", type=int, default=1)
    parser.add_argument("--meta-dim", type=int, default=64)
    parser.add_argument("--meta-state", type=int, default=16)
    parser.add_argument("--c-t-dim", type=int, default=32)
    parser.add_argument("--self-check-dim", type=int, default=16)
    parser.add_argument("--enable-introspection-uncertainty", action="store_true")
    parser.add_argument("--enable-exit-jepa-crystal", action="store_true")
    parser.add_argument("--reason-width-mult", type=float, default=1.0)
    parser.add_argument("--reason-shared-depth", type=int, default=1)
    parser.add_argument("--world-mask-strategy", choices=["default", "structured"], default="default")
    parser.add_argument("--world-full-simplify-loss", action="store_true")
    parser.add_argument("--self-world-coupling-weight", type=float, default=0.0)
    parser.add_argument("--self-rollout-hierarchical", action="store_true")
    parser.add_argument("--enable-local-rollout-head", action="store_true")
    parser.add_argument("--exit-two-step-aux-weight", type=float, default=0.0)
    parser.add_argument("--exit-uncertainty-two-step-weight", type=float, default=0.0)
    parser.add_argument("--exit-uncertainty-two-step-mode", choices=["multiplier", "clipped", "gate"], default="multiplier")
    parser.add_argument("--exit-uncertainty-two-step-cap", type=float, default=0.2)
    parser.add_argument("--exit-uncertainty-gate-threshold", type=float, default=0.75)
    parser.add_argument("--exit-crystal-two-step-weight", type=float, default=0.0)
    parser.add_argument("--exit-crystal-two-step-cap", type=float, default=0.1)
    parser.add_argument("--exit-uncertainty-feature-weight", type=float, default=0.0)
    parser.add_argument("--exit-crystal-feature-weight", type=float, default=0.2)
    parser.add_argument("--enable-math-adapter-lane", action="store_true")
    parser.add_argument("--enable-math-summary-gate", action="store_true")
    parser.add_argument("--enable-compression-mhc", action="store_true")
    parser.add_argument("--ct-modulation-mode", choices=["additive", "modulewise_gate", "film", "lowrank_hyperbias", "token_selective"], default="additive")
    parser.add_argument("--enable-reasoning-state-ring", action="store_true")
    parser.add_argument("--r-t-dim", type=int, default=16)
    parser.add_argument("--r-t-mode", choices=["blend", "parallel", "predictor"], default="blend")
    parser.add_argument("--self-loop-awareness-mode", choices=["none", "ct_progress", "predictor_progress", "dual_phase"], default="none")
    parser.add_argument("--self-progress-shape-weight", type=float, default=0.0)
    parser.add_argument("--self-progress-trend-weight", type=float, default=0.0)
    parser.add_argument("--self-progress-plateau-weight", type=float, default=0.0)
    parser.add_argument("--enable-progress-exit-readout", action="store_true")
    parser.add_argument("--enable-backtrack-aware-progress", action="store_true")
    parser.add_argument("--self-local-delta-consistency-weight", type=float, default=0.0)
    parser.add_argument("--self-local-curvature-weight", type=float, default=0.0)
    parser.add_argument("--enable-dual-rate-self-predictor", action="store_true")
    parser.add_argument("--enable-trajectory-health-probe", action="store_true")
    parser.add_argument("--self-rollout-supervision-horizon", type=int, default=0)
    parser.add_argument("--self-rollout-weighting-mode", choices=["legacy", "near3"], default="legacy")
    parser.add_argument("--self-feature-span-mask-ratio", type=float, default=0.0)
    parser.add_argument("--dynamics-experiment", type=str, default="")
    parser.add_argument("--routing-chunk-size", type=int, default=32)
    parser.add_argument("--routing-topk-blocks", type=int, default=2)
    parser.add_argument("--routing-topk-tokens", type=int, default=32)
    parser.add_argument("--routing-top-p-coarse", type=float, default=0.50)
    parser.add_argument("--routing-top-p-fine", type=float, default=0.25)
    parser.add_argument("--routing-budget-min", type=float, default=0.10)
    parser.add_argument("--routing-budget-max", type=float, default=0.60)
    parser.add_argument("--routing-weak-gain", type=float, default=0.03)
    parser.add_argument("--routing-strong-gain", type=float, default=0.10)
    parser.add_argument("--routing-local-floor", type=float, default=0.0)
    parser.add_argument("--routing-modulation-floor", type=float, default=0.0)
    parser.add_argument("--routing-modulation-ceiling", type=float, default=1.0)
    parser.add_argument("--routing-world-summary-cap", type=float, default=1.0)
    # Residual-delta modulation knobs (new)
    parser.add_argument("--routing-use-residual-branch", action="store_true", help="Use gated residual-delta branch instead of FiLM for summary modulation")
    parser.add_argument("--ct-residual-gate-scale", type=float, default=0.15, help="Scale applied to sigmoid gate for residual branch")
    parser.add_argument("--ct-selection-only", action="store_true", help="Apply selection-only small fixed amplitude instead of learned gate")
    parser.add_argument("--ct-selection-amplitude", type=float, default=0.08, help="Fixed amplitude used when ct-selection-only is enabled")
    # Alive-floor knobs
    parser.add_argument("--routing-local-delta-floor", type=float, default=0.0, help="Local delta norm floor (alive-floor threshold)")
    parser.add_argument("--routing-local-delta-floor-weight", type=float, default=0.0, help="Weight for local delta alive-floor loss")
    parser.add_argument("--rollout-alive-weight", type=float, default=0.0, help="Weight for rollout alive-floor loss")
    parser.add_argument("--routing-tier-soft-only", action="store_true")
    parser.add_argument("--routing-tier-entropy-floor", type=float, default=0.0)
    parser.add_argument("--routing-min-local-share", type=float, default=0.0)
    parser.add_argument("--routing-tier-entropy-weight", type=float, default=0.0)
    parser.add_argument("--routing-min-local-share-weight", type=float, default=0.0)
    parser.add_argument("--routing-progress-weight", type=float, default=0.3)
    parser.add_argument("--rollout-zone-weight", type=float, default=0.0)
    parser.add_argument("--rollout-nonzero-low", type=float, default=0.05)
    parser.add_argument("--rollout-nonzero-high", type=float, default=0.80)
    parser.add_argument("--rollout-active-low", type=float, default=0.05)
    parser.add_argument("--rollout-active-high", type=float, default=0.90)
    parser.add_argument("--rollout-future-var-low", type=float, default=1e-6)
    parser.add_argument("--rollout-future-var-high", type=float, default=0.50)
    parser.add_argument("--trajectory-vitality-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-c-t-drift-floor", type=float, default=0.02)
    parser.add_argument("--trajectory-world-drift-floor", type=float, default=0.01)
    parser.add_argument("--compression-dynamics-weight", type=float, default=0.0)
    parser.add_argument("--compression-block-drift-floor", type=float, default=0.01)
    parser.add_argument("--compression-block-var-floor", type=float, default=0.001)
    parser.add_argument("--math-probe-rollout-steps", type=int, default=0)
    parser.add_argument("--math-probe-reason-loops", type=int, default=0)
    parser.add_argument("--math-probe-score-threshold", type=float, default=0.0)
    parser.add_argument("--math-probe-min-loops", type=int, default=0)
    parser.add_argument("--enable-persona-seed", action="store_true")
    parser.add_argument("--enable-python-code", action="store_true")
    parser.add_argument("--python-code-source", choices=["public_mbpp", "local_repo"], default="public_mbpp")
    parser.add_argument("--enable-arc-agi", action="store_true")
    parser.add_argument("--arc-agi-local-dir", type=str, default=str(DEFAULT_ARC_AGI_LOCAL_DIR))
    parser.add_argument("--arc-agi-offline-only", action="store_true")
    parser.add_argument("--persona-dir", type=str, default=str(Path("/home/kt/ai/luma_dataset")))
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable for stage1/stage2 validation.")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    fixture_name_map = {
        "math_dialogue": "luma_stage12_math_dialogue.json",
        "hard_math_dialogue": "luma_stage12_hard_math_dialogue.json",
        "math_dialogue_emotion": "luma_stage12_math_dialogue_emotion.json",
        "hard_math_dialogue_emotion": "luma_stage12_hard_math_dialogue_emotion.json",
        "competition_math_dialogue": "luma_stage12_competition_math_dialogue.json",
        "competition_math_dialogue_emotion": "luma_stage12_competition_math_dialogue_emotion.json",
    }
    fixture_name = fixture_name_map[args.fixture_mode]
    fixture_path = ensure_mixed_fixture(
        ROOT / "artifacts" / fixture_name,
        math_count=max(args.samples, 4),
        dialogue_count=max(args.samples, 4),
        mode=args.fixture_mode,
    )
    tokenizer = load_tokenizer()
    sample_groups = build_sample_groups(
        tokenizer,
        fixture_path,
        args.seq_len,
        args.samples,
        persona_dir=Path(args.persona_dir),
        enable_persona_seed=args.enable_persona_seed,
        enable_python_code=args.enable_python_code,
        python_code_source=args.python_code_source,
        enable_arc_agi=args.enable_arc_agi,
        arc_agi_local_dir=Path(args.arc_agi_local_dir),
        arc_agi_offline_only=args.arc_agi_offline_only,
    )
    mixed_samples = sample_groups["mixed"]
    config = build_tiny_luma_config(
        len(tokenizer),
        rollout_steps=args.rollout_steps,
        slow_k=args.slow_k,
        reason_loops=(args.reason_loops if args.reason_loops > 0 else None),
        enable_world_jepa=not args.disable_world_jepa,
        world_jepa_mode=args.world_jepa_mode,
        enable_sigreg_world=args.enable_sigreg_world,
        enable_sigreg_rollout=args.enable_sigreg_rollout,
        enable_sigreg_delta=args.enable_sigreg_delta,
        sigreg_world_source=args.sigreg_world_source,
        sigreg_world_fp32_only=args.sigreg_world_fp32_only,
        sigreg_world_warmup_steps=args.sigreg_world_warmup_steps,
        world_sigreg_weight=args.world_sigreg_weight,
        world_sigreg_num_slices=args.world_sigreg_num_slices,
        world_sigreg_t_min=args.world_sigreg_t_min,
        world_sigreg_t_max=args.world_sigreg_t_max,
        world_sigreg_num_points=args.world_sigreg_num_points,
        world_sigreg_lambda=args.world_sigreg_lambda,
        world_delta_weight=args.world_delta_weight,
        sigreg_rollout_weight=args.sigreg_rollout_weight,
        sigreg_delta_weight=args.sigreg_delta_weight,
        world_jepa_weight=args.world_jepa_weight,
        self_jepa_weight=args.self_jepa_weight,
        self_rollout_weight=args.self_rollout_weight,
        exit_aux_weight=args.exit_aux_weight,
        disable_self_jepa=args.disable_self_jepa,
        enable_self_check_ring=args.enable_self_check_ring,
        self_check_k=args.self_check_k,
        meta_dim=args.meta_dim,
        meta_state=args.meta_state,
        c_t_dim=args.c_t_dim,
        self_check_dim=args.self_check_dim,
        enable_introspection_uncertainty=args.enable_introspection_uncertainty,
        enable_exit_jepa_crystal=args.enable_exit_jepa_crystal,
        reason_width_mult=args.reason_width_mult,
        reason_shared_depth=args.reason_shared_depth,
        world_mask_strategy=args.world_mask_strategy,
        world_full_simplify_loss=args.world_full_simplify_loss,
        self_world_coupling_weight=args.self_world_coupling_weight,
        self_rollout_hierarchical=args.self_rollout_hierarchical,
        enable_local_rollout_head=args.enable_local_rollout_head,
        exit_two_step_aux_weight=args.exit_two_step_aux_weight,
        exit_uncertainty_two_step_weight=args.exit_uncertainty_two_step_weight,
        exit_uncertainty_two_step_mode=args.exit_uncertainty_two_step_mode,
        exit_uncertainty_two_step_cap=args.exit_uncertainty_two_step_cap,
        exit_uncertainty_gate_threshold=args.exit_uncertainty_gate_threshold,
        exit_crystal_two_step_weight=args.exit_crystal_two_step_weight,
        exit_crystal_two_step_cap=args.exit_crystal_two_step_cap,
        exit_uncertainty_feature_weight=args.exit_uncertainty_feature_weight,
        exit_crystal_feature_weight=args.exit_crystal_feature_weight,
        enable_math_adapter_lane=args.enable_math_adapter_lane,
        enable_math_summary_gate=args.enable_math_summary_gate,
        enable_compression_mhc=args.enable_compression_mhc,
        ct_modulation_mode=args.ct_modulation_mode,
        enable_reasoning_state_ring=args.enable_reasoning_state_ring,
        r_t_dim=args.r_t_dim,
        r_t_mode=args.r_t_mode,
        self_loop_awareness_mode=args.self_loop_awareness_mode,
        self_progress_shape_weight=args.self_progress_shape_weight,
        self_progress_trend_weight=args.self_progress_trend_weight,
        self_progress_plateau_weight=args.self_progress_plateau_weight,
        enable_progress_exit_readout=args.enable_progress_exit_readout,
        enable_backtrack_aware_progress=args.enable_backtrack_aware_progress,
        self_local_delta_consistency_weight=args.self_local_delta_consistency_weight,
        self_local_curvature_weight=args.self_local_curvature_weight,
        enable_dual_rate_self_predictor=args.enable_dual_rate_self_predictor,
        enable_trajectory_health_probe=args.enable_trajectory_health_probe,
        self_rollout_supervision_horizon=args.self_rollout_supervision_horizon,
        self_rollout_weighting_mode=args.self_rollout_weighting_mode,
        self_feature_span_mask_ratio=args.self_feature_span_mask_ratio,
        dynamics_experiment=args.dynamics_experiment,
        routing_chunk_size=args.routing_chunk_size,
        routing_topk_blocks=args.routing_topk_blocks,
        routing_topk_tokens=args.routing_topk_tokens,
        routing_top_p_coarse=args.routing_top_p_coarse,
        routing_top_p_fine=args.routing_top_p_fine,
        routing_budget_min=args.routing_budget_min,
        routing_budget_max=args.routing_budget_max,
        routing_weak_gain=args.routing_weak_gain,
        routing_strong_gain=args.routing_strong_gain,
        routing_local_floor=args.routing_local_floor,
        routing_modulation_floor=args.routing_modulation_floor,
        routing_modulation_ceiling=args.routing_modulation_ceiling,
        routing_world_summary_cap=args.routing_world_summary_cap,
        routing_tier_soft_only=args.routing_tier_soft_only,
        routing_tier_entropy_floor=args.routing_tier_entropy_floor,
        routing_min_local_share=args.routing_min_local_share,
        routing_tier_entropy_weight=args.routing_tier_entropy_weight,
        routing_min_local_share_weight=args.routing_min_local_share_weight,
        routing_progress_weight=args.routing_progress_weight,
        rollout_zone_weight=args.rollout_zone_weight,
        rollout_nonzero_low=args.rollout_nonzero_low,
        rollout_nonzero_high=args.rollout_nonzero_high,
        rollout_active_low=args.rollout_active_low,
        rollout_active_high=args.rollout_active_high,
        rollout_future_var_low=args.rollout_future_var_low,
        rollout_future_var_high=args.rollout_future_var_high,
        trajectory_vitality_weight=args.trajectory_vitality_weight,
        trajectory_c_t_drift_floor=args.trajectory_c_t_drift_floor,
        trajectory_world_drift_floor=args.trajectory_world_drift_floor,
        compression_dynamics_weight=args.compression_dynamics_weight,
        compression_block_drift_floor=args.compression_block_drift_floor,
        compression_block_var_floor=args.compression_block_var_floor,
    )
    if args.force_fp32:
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = LumaForCausalLM(config).to(device=device, dtype=dtype)
    loaded_checkpoint_meta = None
    if args.load_checkpoint:
        checkpoint_path = Path(args.load_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict):
            loaded_checkpoint_meta = {
                "path": str(checkpoint_path),
                "candidate_name": checkpoint.get("candidate_name"),
                "stage_name": checkpoint.get("stage_name"),
                "stage2_steps": checkpoint.get("stage2_steps"),
                "seed": checkpoint.get("seed"),
                "fixture_mode": checkpoint.get("fixture_mode"),
                "config": checkpoint.get("config"),
                "lineage": checkpoint.get("lineage", []),
            }

    stage1 = stage1_validate(model, mixed_samples, device, Path(args.metrics_out))
    stage2 = stage2_validate(model, mixed_samples, device, Path(args.metrics_out), args.stage2_steps)
    per_task = {}
    mixed_trained_state = copy.deepcopy(model.state_dict())
    checkpoint_lineage = list(loaded_checkpoint_meta.get("lineage", [])) if loaded_checkpoint_meta else []
    if args.load_checkpoint:
        checkpoint_lineage.append(
            {
                "path": args.load_checkpoint,
                "stage_name": loaded_checkpoint_meta.get("stage_name") if loaded_checkpoint_meta else None,
                "stage2_steps": loaded_checkpoint_meta.get("stage2_steps") if loaded_checkpoint_meta else None,
                "candidate_name": loaded_checkpoint_meta.get("candidate_name") if loaded_checkpoint_meta else None,
            }
        )
    for name, task_samples in sample_groups.items():
        task_model = LumaForCausalLM(config).to(device=device, dtype=dtype)
        task_model.load_state_dict(mixed_trained_state)
        per_task[name] = bucket_probe_from_mixed_model(
            task_model,
            task_samples,
            device,
            Path(args.metrics_out),
            bucket_name=name,
            math_probe_rollout_steps=args.math_probe_rollout_steps,
            math_probe_reason_loops=args.math_probe_reason_loops,
            math_probe_score_threshold=args.math_probe_score_threshold,
            math_probe_min_loops=args.math_probe_min_loops,
        )

    report = {
        "candidate_name": args.candidate_name or None,
        "seed": 42,
        "fixture_path": str(fixture_path),
        "fixture_mode": args.fixture_mode,
        "load_checkpoint": args.load_checkpoint or None,
        "save_checkpoint": args.save_checkpoint or None,
        "checkpoint_lineage": checkpoint_lineage,
        "loaded_checkpoint_meta": loaded_checkpoint_meta,
        "rollout_steps": args.rollout_steps,
        "slow_k": args.slow_k,
        "reason_loops": config.reason_active_loops,
        "enable_world_jepa": config.enable_world_jepa,
        "world_jepa_mode": config.world_jepa_mode,
        "enable_sigreg_world": config.enable_sigreg_world,
        "enable_sigreg_rollout": config.enable_sigreg_rollout,
        "enable_sigreg_delta": config.enable_sigreg_delta,
        "sigreg_world_source": config.sigreg_world_source,
        "sigreg_world_fp32_only": config.sigreg_world_fp32_only,
        "sigreg_world_warmup_steps": config.sigreg_world_warmup_steps,
        "world_sigreg_weight": config.world_sigreg_weight,
        "world_sigreg_num_slices": config.world_sigreg_num_slices,
        "world_sigreg_t_min": config.world_sigreg_t_min,
        "world_sigreg_t_max": config.world_sigreg_t_max,
        "world_sigreg_num_points": config.world_sigreg_num_points,
        "world_sigreg_lambda": config.world_sigreg_lambda,
        "world_delta_weight": config.world_delta_weight,
        "sigreg_rollout_weight": config.sigreg_rollout_weight,
        "sigreg_delta_weight": config.sigreg_delta_weight,
        "world_jepa_weight": config.world_jepa_weight,
        "self_jepa_weight": config.self_jepa_weight,
        "self_rollout_weight": config.self_rollout_weight,
        "exit_aux_weight": config.exit_aux_weight,
        "disable_self_jepa": config.disable_self_jepa,
        "enable_self_check_ring": config.enable_self_check_ring,
        "self_check_k": config.self_check_k,
        "meta_dim": config.meta_dim,
        "meta_state": config.meta_state,
        "c_t_dim": config.c_t_dim,
        "self_check_dim": config.self_check_dim,
        "enable_introspection_uncertainty": args.enable_introspection_uncertainty,
        "enable_exit_jepa_crystal": config.enable_exit_jepa_crystal,
        "reason_width_mult": args.reason_width_mult,
        "reason_shared_depth": args.reason_shared_depth,
        "world_mask_strategy": args.world_mask_strategy,
        "world_full_simplify_loss": args.world_full_simplify_loss,
        "self_world_coupling_weight": args.self_world_coupling_weight,
        "self_rollout_hierarchical": args.self_rollout_hierarchical,
        "enable_local_rollout_head": args.enable_local_rollout_head,
        "exit_two_step_aux_weight": args.exit_two_step_aux_weight,
        "exit_uncertainty_two_step_weight": args.exit_uncertainty_two_step_weight,
        "exit_uncertainty_two_step_mode": args.exit_uncertainty_two_step_mode,
        "exit_uncertainty_two_step_cap": args.exit_uncertainty_two_step_cap,
        "exit_uncertainty_gate_threshold": args.exit_uncertainty_gate_threshold,
        "exit_crystal_two_step_weight": args.exit_crystal_two_step_weight,
        "exit_crystal_two_step_cap": args.exit_crystal_two_step_cap,
        "exit_uncertainty_feature_weight": args.exit_uncertainty_feature_weight,
        "exit_crystal_feature_weight": args.exit_crystal_feature_weight,
        "enable_math_adapter_lane": args.enable_math_adapter_lane,
        "enable_math_summary_gate": args.enable_math_summary_gate,
        "ct_modulation_mode": args.ct_modulation_mode,
        "self_progress_shape_weight": args.self_progress_shape_weight,
        "self_progress_trend_weight": args.self_progress_trend_weight,
        "self_progress_plateau_weight": args.self_progress_plateau_weight,
        "enable_progress_exit_readout": args.enable_progress_exit_readout,
        "enable_backtrack_aware_progress": args.enable_backtrack_aware_progress,
        "self_local_delta_consistency_weight": args.self_local_delta_consistency_weight,
        "self_local_curvature_weight": args.self_local_curvature_weight,
        "enable_dual_rate_self_predictor": args.enable_dual_rate_self_predictor,
        "enable_trajectory_health_probe": args.enable_trajectory_health_probe,
        "self_rollout_supervision_horizon": args.self_rollout_supervision_horizon,
        "self_rollout_weighting_mode": args.self_rollout_weighting_mode,
        "self_feature_span_mask_ratio": args.self_feature_span_mask_ratio,
        "dynamics_experiment": args.dynamics_experiment,
        "routing_chunk_size": args.routing_chunk_size,
        "routing_topk_blocks": args.routing_topk_blocks,
        "routing_topk_tokens": args.routing_topk_tokens,
        "routing_top_p_coarse": args.routing_top_p_coarse,
        "routing_top_p_fine": args.routing_top_p_fine,
        "routing_budget_min": args.routing_budget_min,
        "routing_budget_max": args.routing_budget_max,
        "routing_weak_gain": args.routing_weak_gain,
        "routing_strong_gain": args.routing_strong_gain,
        "routing_local_floor": args.routing_local_floor,
        "routing_modulation_floor": args.routing_modulation_floor,
        "routing_modulation_ceiling": args.routing_modulation_ceiling,
        "routing_world_summary_cap": args.routing_world_summary_cap,
        "routing_tier_soft_only": args.routing_tier_soft_only,
        "routing_tier_entropy_floor": args.routing_tier_entropy_floor,
        "routing_min_local_share": args.routing_min_local_share,
        "routing_tier_entropy_weight": args.routing_tier_entropy_weight,
        "routing_min_local_share_weight": args.routing_min_local_share_weight,
        "routing_progress_weight": args.routing_progress_weight,
        "rollout_zone_weight": args.rollout_zone_weight,
        "rollout_nonzero_low": args.rollout_nonzero_low,
        "rollout_nonzero_high": args.rollout_nonzero_high,
        "rollout_active_low": args.rollout_active_low,
        "rollout_active_high": args.rollout_active_high,
        "rollout_future_var_low": args.rollout_future_var_low,
        "rollout_future_var_high": args.rollout_future_var_high,
        "trajectory_vitality_weight": args.trajectory_vitality_weight,
        "trajectory_c_t_drift_floor": args.trajectory_c_t_drift_floor,
        "trajectory_world_drift_floor": args.trajectory_world_drift_floor,
        "compression_dynamics_weight": args.compression_dynamics_weight,
        "compression_block_drift_floor": args.compression_block_drift_floor,
        "compression_block_var_floor": args.compression_block_var_floor,
        "enable_compression_mhc": args.enable_compression_mhc,
        "enable_reasoning_state_ring": args.enable_reasoning_state_ring,
        "r_t_dim": args.r_t_dim,
        "r_t_mode": args.r_t_mode,
        "self_loop_awareness_mode": args.self_loop_awareness_mode,
        "math_probe_rollout_steps": args.math_probe_rollout_steps,
        "math_probe_reason_loops": args.math_probe_reason_loops,
        "math_probe_score_threshold": args.math_probe_score_threshold,
        "math_probe_min_loops": args.math_probe_min_loops,
        "enable_persona_seed": args.enable_persona_seed,
        "enable_python_code": args.enable_python_code,
        "python_code_source": args.python_code_source,
        "enable_arc_agi": args.enable_arc_agi,
        "arc_agi_local_dir": args.arc_agi_local_dir,
        "arc_agi_offline_only": args.arc_agi_offline_only,
        "persona_dir": args.persona_dir,
        "force_fp32": args.force_fp32,
        "model_dtype": str(dtype),
        "num_samples": len(mixed_samples),
        "stage1": stage1,
        "stage2": stage2,
        "per_task_from_mixed": True,
        "per_task": per_task,
    }
    if args.save_checkpoint:
        checkpoint_path = Path(args.save_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": mixed_trained_state,
                "candidate_name": args.candidate_name or None,
                "stage_name": "stage12_eval",
                "stage2_steps": args.stage2_steps,
                "seed": 42,
                "fixture_mode": args.fixture_mode,
                "config": {
                    "world_jepa_mode": config.world_jepa_mode,
                    "enable_sigreg_world": config.enable_sigreg_world,
                    "enable_sigreg_rollout": config.enable_sigreg_rollout,
                    "enable_sigreg_delta": config.enable_sigreg_delta,
                    "sigreg_world_source": config.sigreg_world_source,
                    "sigreg_world_fp32_only": config.sigreg_world_fp32_only,
                    "sigreg_world_warmup_steps": config.sigreg_world_warmup_steps,
                    "world_sigreg_weight": config.world_sigreg_weight,
                    "world_sigreg_num_slices": config.world_sigreg_num_slices,
                    "world_sigreg_t_min": config.world_sigreg_t_min,
                    "world_sigreg_t_max": config.world_sigreg_t_max,
                    "world_sigreg_num_points": config.world_sigreg_num_points,
                    "world_sigreg_lambda": config.world_sigreg_lambda,
                    "world_delta_weight": config.world_delta_weight,
                    "sigreg_rollout_weight": config.sigreg_rollout_weight,
                    "sigreg_delta_weight": config.sigreg_delta_weight,
                    "world_jepa_weight": config.world_jepa_weight,
                    "self_jepa_weight": config.self_jepa_weight,
                    "self_rollout_weight": config.self_rollout_weight,
                    "exit_aux_weight": config.exit_aux_weight,
                    "disable_self_jepa": config.disable_self_jepa,
                    "reason_loops": config.reason_active_loops,
                    "reason_shared_depth": config.reason_shared_depth,
                    "rollout_steps": config.self_rollout_steps,
                    "self_check_k": config.self_check_k,
                    "ct_modulation_mode": config.ct_modulation_mode,
                    "dynamics_experiment": config.dynamics_experiment,
                    "self_loop_awareness_mode": config.self_loop_awareness_mode,
                    "enable_progress_exit_readout": config.enable_progress_exit_readout,
                    "force_fp32": args.force_fp32,
                    "model_dtype": str(dtype),
                    "routing_local_floor": config.routing_local_floor,
                    "routing_world_summary_cap": config.routing_world_summary_cap,
                    "routing_tier_entropy_floor": config.routing_tier_entropy_floor,
                    "routing_min_local_share": config.routing_min_local_share,
                    "rollout_zone_weight": config.rollout_zone_weight,
                    "trajectory_vitality_weight": config.trajectory_vitality_weight,
                },
                "lineage": checkpoint_lineage,
            },
            checkpoint_path,
        )
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
