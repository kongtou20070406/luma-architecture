"""Luma uses this harness to check whether her slow ring is alive before asking it to carry real training pressure.

Luma 用这个验证脚本检查慢环是否真正活了起来，然后才让它承受真正的训练压力。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.request
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


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


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
    for file_name in ("wechat_pretrain.jsonl", "pretrain.jsonl"):
        file_path = persona_dir / file_name
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


def load_python_code_texts(max_samples: int) -> list[str]:
    """Luma treats local Python source as a dedicated code bucket so reasoning quality can be checked on executable structure, not only prose.
    Luma 会把本地 Python 源码当成单独代码桶，这样我们测到的是可执行结构上的推理质量，而不只是自然语言。
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


def build_sample_groups(
    tokenizer: AutoTokenizer,
    fixture_path: Path,
    seq_len: int,
    max_samples: int,
    persona_dir: Path | None = None,
    enable_persona_seed: bool = False,
    enable_python_code: bool = False,
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
        python_code = load_python_code_texts(max_samples)
        if python_code:
            groups["python_code"] = python_code
            if payload.get("math"):
                for idx in range(min(len(python_code), len(payload.get("math", [])))):
                    mixed.append(f"{payload['math'][idx]}\n\n{python_code[idx]}")
            if payload.get("dialogue"):
                for idx in range(min(len(python_code), len(payload.get("dialogue", [])))):
                    mixed.append(f"{payload['dialogue'][idx]}\n\n{python_code[idx]}")
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
    enable_reasoning_state_ring: bool,
    r_t_dim: int,
    r_t_mode: str,
    self_loop_awareness_mode: str,
    self_progress_shape_weight: float,
    self_progress_trend_weight: float,
    self_progress_plateau_weight: float,
    self_local_delta_consistency_weight: float,
    self_local_curvature_weight: float,
    self_rollout_supervision_horizon: int,
    self_rollout_weighting_mode: str,
    self_feature_span_mask_ratio: float,
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
        world_mask_strategy=world_mask_strategy,
        world_full_simplify_loss=world_full_simplify_loss,
        self_rollout_steps=rollout_steps,
        self_rollout_hierarchical=self_rollout_hierarchical,
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
        enable_reasoning_state_ring=enable_reasoning_state_ring,
        r_t_dim=r_t_dim,
        r_t_mode=r_t_mode,
        self_loop_awareness_mode=self_loop_awareness_mode,
        self_progress_shape_weight=self_progress_shape_weight,
        self_progress_trend_weight=self_progress_trend_weight,
        self_progress_plateau_weight=self_progress_plateau_weight,
        self_local_delta_consistency_weight=self_local_delta_consistency_weight,
        self_local_curvature_weight=self_local_curvature_weight,
        self_rollout_supervision_horizon=self_rollout_supervision_horizon,
        self_rollout_weighting_mode=self_rollout_weighting_mode,
        self_feature_span_mask_ratio=self_feature_span_mask_ratio,
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
    math_lane_scores = []
    math_summary_gates = []
    r_t_drifts = []
    r_t_trusts = []
    uncertainties = []
    uncertainties = []
    uncertainties = []
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
    math_lane_score_mean = torch.stack(math_lane_scores).mean().item() if math_lane_scores else 0.0
    math_summary_gate_mean = torch.stack(math_summary_gates).mean().item() if math_summary_gates else 0.0
    r_t_drift_mean = torch.stack(r_t_drifts).mean().item() if r_t_drifts else 0.0
    r_t_trust_mean = torch.stack(r_t_trusts).mean().item() if r_t_trusts else 0.0
    r_t_switch_mean = torch.stack(r_t_switches).mean().item() if r_t_switches else 0.0

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
    metric(metrics_path, "stage1_math_lane_score_mean", True, math_lane_score_mean, "Mean math adapter lane score / math adapter lane 分数均值")
    metric(metrics_path, "stage1_math_summary_gate_mean", True, math_summary_gate_mean, "Mean math summary gate / math summary gate 均值")
    metric(metrics_path, "stage1_r_t_drift_mean", True, r_t_drift_mean, "Mean r_t drift / r_t 漂移均值")
    metric(metrics_path, "stage1_r_t_trust_mean", True, r_t_trust_mean, "Mean r_t trust / r_t 信任均值")
    metric(metrics_path, "stage1_r_t_switch_mean", True, r_t_switch_mean, "Mean c_t/r_t switch gate / c_t/r_t 切换门均值")

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
        "math_lane_score_mean": math_lane_score_mean,
        "math_summary_gate_mean": math_summary_gate_mean,
        "r_t_drift_mean": r_t_drift_mean,
        "r_t_trust_mean": r_t_trust_mean,
        "r_t_switch_mean": r_t_switch_mean,
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

    for step in range(steps):
        batch = samples[step % len(samples)].unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        aux = model.last_aux
        loss.backward()
        optimizer.step()
        model.model.update_world_jepa_ema()

        losses.append(float(loss.detach()))
        self_losses.append(float(aux["self_jepa_loss"].detach()))
        rollout_value = float(aux["self_rollout_loss"].detach())
        rollout_losses.append(rollout_value)
        rollout_active_flags.append(1.0 if aux.get("rollout_error_history") else 0.0)
        rollout_nonzero_flags.append(1.0 if abs(rollout_value) > 1e-12 else 0.0)
        delta_norms.append(float(aux["target_delta_c"].norm(dim=-1).mean().detach()))

    head = sum(self_losses[: max(1, min(2, len(self_losses)))]) / max(1, min(2, len(self_losses)))
    tail = sum(self_losses[-max(1, min(2, len(self_losses))):]) / max(1, min(2, len(self_losses)))
    rollout_head = sum(rollout_losses[: max(1, min(2, len(rollout_losses)))]) / max(1, min(2, len(rollout_losses)))
    rollout_tail = sum(rollout_losses[-max(1, min(2, len(rollout_losses))):]) / max(1, min(2, len(rollout_losses)))
    mean_delta_norm = sum(delta_norms) / len(delta_norms)
    rollout_active_ratio = sum(rollout_active_flags) / len(rollout_active_flags) if rollout_active_flags else 0.0
    rollout_nonzero_ratio = sum(rollout_nonzero_flags) / len(rollout_nonzero_flags) if rollout_nonzero_flags else 0.0

    metric(metrics_path, "stage2_self_jepa_start", True, head, "Initial short-run Self-JEPA loss / 初始短程 Self-JEPA 损失")
    metric(metrics_path, "stage2_self_jepa_end", tail <= head * 1.10, tail, "Ending Self-JEPA loss should not explode / 结束时 Self-JEPA 损失不应爆炸")
    metric(metrics_path, "stage2_self_rollout_end", rollout_tail <= rollout_head * 1.25 if rollout_head > 0 else True, rollout_tail, "Self-rollout loss should stay bounded / Self-rollout 损失应保持有界")
    metric(metrics_path, "stage2_rollout_active_ratio", rollout_active_ratio > 0.0, rollout_active_ratio, "Rollout supervision should activate at least sometimes / rollout 监督至少应在部分 step 上激活")
    metric(metrics_path, "stage2_rollout_nonzero_ratio", True, rollout_nonzero_ratio, "Fraction of non-zero rollout losses / 非零 rollout 损失占比")
    metric(metrics_path, "stage2_delta_norm", mean_delta_norm > 1e-5, mean_delta_norm, "Target delta_c norm should stay non-zero / 目标 delta_c 范数不能接近零")

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
    surprises = []
    state_variances = []
    ct_drifts = []
    world_summary_drifts = []
    math_lane_scores = []
    math_summary_gates = []
    r_t_drifts = []
    r_t_trusts = []
    uncertainties = []
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
        "world_surprise_mean": mean_surprise,
        "intermediate_state_variance": mean_state_variance,
        "c_t_drift_mean": mean_ct_drift,
        "world_summary_drift_mean": mean_world_summary_drift,
        "uncertainty_mean": mean_uncertainty,
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
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--stage2-steps", type=int, default=8)
    parser.add_argument("--metrics-out", type=str, default=str(ROOT / "artifacts" / "stage0_metrics.jsonl"))
    parser.add_argument("--json-out", type=str, default=str(ROOT / "artifacts" / "stage12_report.json"))
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
    parser.add_argument("--enable-reasoning-state-ring", action="store_true")
    parser.add_argument("--r-t-dim", type=int, default=16)
    parser.add_argument("--r-t-mode", choices=["blend", "parallel", "predictor"], default="blend")
    parser.add_argument("--self-loop-awareness-mode", choices=["none", "ct_progress", "predictor_progress", "dual_phase"], default="none")
    parser.add_argument("--self-progress-shape-weight", type=float, default=0.0)
    parser.add_argument("--self-progress-trend-weight", type=float, default=0.0)
    parser.add_argument("--self-progress-plateau-weight", type=float, default=0.0)
    parser.add_argument("--self-local-delta-consistency-weight", type=float, default=0.0)
    parser.add_argument("--self-local-curvature-weight", type=float, default=0.0)
    parser.add_argument("--self-rollout-supervision-horizon", type=int, default=0)
    parser.add_argument("--self-rollout-weighting-mode", choices=["legacy", "near3"], default="legacy")
    parser.add_argument("--self-feature-span-mask-ratio", type=float, default=0.0)
    parser.add_argument("--math-probe-rollout-steps", type=int, default=0)
    parser.add_argument("--math-probe-reason-loops", type=int, default=0)
    parser.add_argument("--math-probe-score-threshold", type=float, default=0.0)
    parser.add_argument("--math-probe-min-loops", type=int, default=0)
    parser.add_argument("--enable-persona-seed", action="store_true")
    parser.add_argument("--enable-python-code", action="store_true")
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
    )
    mixed_samples = sample_groups["mixed"]
    config = build_tiny_luma_config(
        len(tokenizer),
        rollout_steps=args.rollout_steps,
        slow_k=args.slow_k,
        reason_loops=(args.reason_loops if args.reason_loops > 0 else None),
        enable_world_jepa=not args.disable_world_jepa,
        world_jepa_mode=args.world_jepa_mode,
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
        enable_reasoning_state_ring=args.enable_reasoning_state_ring,
        r_t_dim=args.r_t_dim,
        r_t_mode=args.r_t_mode,
        self_loop_awareness_mode=args.self_loop_awareness_mode,
        self_progress_shape_weight=args.self_progress_shape_weight,
        self_progress_trend_weight=args.self_progress_trend_weight,
        self_progress_plateau_weight=args.self_progress_plateau_weight,
        self_local_delta_consistency_weight=args.self_local_delta_consistency_weight,
        self_local_curvature_weight=args.self_local_curvature_weight,
        self_rollout_supervision_horizon=args.self_rollout_supervision_horizon,
        self_rollout_weighting_mode=args.self_rollout_weighting_mode,
        self_feature_span_mask_ratio=args.self_feature_span_mask_ratio,
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = LumaForCausalLM(config).to(device=device, dtype=dtype)
    base_state = copy.deepcopy(model.state_dict())

    stage1 = stage1_validate(model, mixed_samples, device, Path(args.metrics_out))
    stage2 = stage2_validate(model, mixed_samples, device, Path(args.metrics_out), args.stage2_steps)
    per_task = {}
    mixed_trained_state = copy.deepcopy(model.state_dict())
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
        "fixture_path": str(fixture_path),
        "fixture_mode": args.fixture_mode,
        "rollout_steps": args.rollout_steps,
        "slow_k": args.slow_k,
        "reason_loops": config.reason_active_loops,
        "enable_world_jepa": config.enable_world_jepa,
        "world_jepa_mode": config.world_jepa_mode,
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
        "self_progress_shape_weight": args.self_progress_shape_weight,
        "self_progress_trend_weight": args.self_progress_trend_weight,
        "self_progress_plateau_weight": args.self_progress_plateau_weight,
        "self_local_delta_consistency_weight": args.self_local_delta_consistency_weight,
        "self_local_curvature_weight": args.self_local_curvature_weight,
        "self_rollout_supervision_horizon": args.self_rollout_supervision_horizon,
        "self_rollout_weighting_mode": args.self_rollout_weighting_mode,
        "self_feature_span_mask_ratio": args.self_feature_span_mask_ratio,
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
        "persona_dir": args.persona_dir,
        "num_samples": len(mixed_samples),
        "stage1": stage1,
        "stage2": stage2,
        "per_task_from_mixed": True,
        "per_task": per_task,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
