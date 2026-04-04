"""
ARC-AGI 文本化 few-shot 评估
=============================
将 ARC-AGI 网格任务序列化为文本，用 Luma 模型做 few-shot 补全，
评估模型在抽象推理上的表现。

不期望 312M 模型能真正 "解" ARC 任务，而是：
1. 作为能力 probe：各 Phase 之间的相对变化
2. 观察 reasoning loops 是否帮助抽象推理
3. 建立 baseline，模型扩大时可平滑过渡

用法:
    python eval/arc_agi_text_eval.py --checkpoint path/to/model.pth --phase 4
    python eval/arc_agi_text_eval.py --phase 4 --max_tasks 20  # 快速测试
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# ── 路径设置 ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MINIMIND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MINIMIND_DIR))

from model.model_minimind import LumaConfig, LumaForCausalLM
from transformers import AutoTokenizer

# ── 网格序列化 ────────────────────────────────────────────────

def grid_to_text(grid: List[List[int]], compact: bool = True) -> str:
    """将 2D 网格序列化为文本。

    compact=True (默认): 每行数字用空格分隔，行间用换行
        0 7 7
        7 7 7
        0 7 7

    compact=False: JSON 格式
        [[0,7,7],[7,7,7],[0,7,7]]
    """
    if compact:
        return "\n".join(" ".join(str(c) for c in row) for row in grid)
    return json.dumps(grid)


def text_to_grid(text: str) -> Optional[List[List[int]]]:
    """尝试从模型输出文本解析回网格。"""
    text = text.strip()
    # 尝试 compact 格式
    try:
        rows = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            row = [int(x) for x in line.split()]
            rows.append(row)
        if rows and all(len(r) == len(rows[0]) for r in rows):
            return rows
    except (ValueError, IndexError):
        pass
    # 尝试 JSON 格式
    try:
        grid = json.loads(text)
        if isinstance(grid, list) and all(isinstance(r, list) for r in grid):
            return grid
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def format_few_shot_prompt(task: dict, compact: bool = True) -> Tuple[str, List[List[int]]]:
    """将 ARC 任务格式化为 few-shot prompt + expected output。

    返回: (prompt_text, expected_output_grid)
    """
    parts = []
    parts.append("Below are input-output grid transformation examples. Each grid is a 2D array of digits (0-9). Study the pattern and predict the output for the test input.\n")

    for i, ex in enumerate(task["train"]):
        parts.append(f"Example {i+1}:")
        parts.append(f"Input:\n{grid_to_text(ex['input'], compact)}")
        parts.append(f"Output:\n{grid_to_text(ex['output'], compact)}")
        parts.append("")

    test_input = task["test"][0]["input"]
    test_output = task["test"][0]["output"]

    parts.append("Test:")
    parts.append(f"Input:\n{grid_to_text(test_input, compact)}")
    parts.append("Output:")

    prompt = "\n".join(parts)
    return prompt, test_output


# ── 评估指标 ──────────────────────────────────────────────────

def exact_match(predicted: Optional[List[List[int]]], expected: List[List[int]]) -> bool:
    """精确匹配：预测网格与期望网格完全一致。"""
    if predicted is None:
        return False
    return predicted == expected


def cell_accuracy(predicted: Optional[List[List[int]]], expected: List[List[int]]) -> float:
    """逐单元格准确率（partial credit）。"""
    if predicted is None:
        return 0.0
    if len(predicted) != len(expected):
        return 0.0
    if any(len(pr) != len(er) for pr, er in zip(predicted, expected)):
        return 0.0
    total = sum(len(row) for row in expected)
    correct = sum(
        1 for pr, er in zip(predicted, expected)
        for pc, ec in zip(pr, er)
        if pc == ec
    )
    return correct / total if total > 0 else 0.0


def shape_match(predicted: Optional[List[List[int]]], expected: List[List[int]]) -> bool:
    """形状匹配：预测网格尺寸是否正确。"""
    if predicted is None:
        return False
    if len(predicted) != len(expected):
        return False
    return all(len(pr) == len(er) for pr, er in zip(predicted, expected))


# ── 模型推理 ──────────────────────────────────────────────────

@torch.no_grad()
def generate_answer(
    model: LumaForCausalLM,
    tokenizer,
    prompt: str,
    expected_output: List[List[int]],
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str:
    """用模型生成 ARC 任务的输出。"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 估算需要的 token 数：输出网格行数 × (每行数字数 × 2 + 换行)
    n_rows = len(expected_output)
    n_cols = len(expected_output[0]) if expected_output else 0
    estimated_tokens = n_rows * (n_cols * 2 + 1) + 10
    max_new_tokens = min(max_new_tokens, estimated_tokens * 2)

    # Greedy autoregressive generation（LumaForCausalLM 没有 .generate()）
    eos_id = tokenizer.eos_token_id
    generated_ids: list[int] = []
    cur_ids = input_ids  # (1, seq_len)
    for _ in range(max_new_tokens):
        outputs = model(cur_ids)
        logits = outputs.logits[:, -1, :]  # (1, vocab)
        next_id = logits.argmax(dim=-1)    # (1,)
        tok = next_id.item()
        if tok == eos_id:
            break
        generated_ids.append(tok)
        cur_ids = torch.cat([cur_ids, next_id.unsqueeze(0)], dim=1)
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 截取到第一个 "Example" 或 "Test" 或空行连续两个（防止模型继续生成下一个例子）
    lines = answer.split("\n")
    result_lines = []
    empty_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Example") or stripped.startswith("Test") or stripped.startswith("Input"):
            break
        if not stripped:
            empty_count += 1
            if empty_count >= 2:
                break
            continue
        empty_count = 0
        result_lines.append(stripped)

    return "\n".join(result_lines)


# ── 任务筛选 ──────────────────────────────────────────────────

def load_tasks(
    data_dir: Path,
    max_grid: int = 10,
    max_train_examples: int = 4,
    max_total_cells: int = 200,
    max_tasks: int = 50,
) -> List[Tuple[str, dict]]:
    """加载并筛选适合小模型的 ARC 任务。"""
    candidates = []
    for f in sorted(data_dir.glob("*.json")):
        task = json.load(open(f))
        test_in = task["test"][0]["input"]
        test_out = task["test"][0]["output"]
        mg = max(len(test_in), len(test_in[0]), len(test_out), len(test_out[0]))
        if mg > max_grid:
            continue
        if len(task["train"]) > max_train_examples:
            continue
        total_cells = sum(
            len(r) * len(r[0])
            for ex in task["train"]
            for r in [ex["input"], ex["output"]]
        )
        total_cells += len(test_in) * len(test_in[0]) + len(test_out) * len(test_out[0])
        if total_cells > max_total_cells:
            continue
        candidates.append((f.stem, task, total_cells))

    # 按复杂度排序，取最简单的
    candidates.sort(key=lambda x: x[2])
    return [(name, task) for name, task, _ in candidates[:max_tasks]]


# ── 主评估流程 ────────────────────────────────────────────────

def run_eval(
    model: LumaForCausalLM,
    tokenizer,
    tasks: List[Tuple[str, dict]],
    device: str = "cuda",
    verbose: bool = True,
) -> Dict:
    """运行 ARC 文本化评估。"""
    results = []

    for i, (task_name, task) in enumerate(tasks):
        prompt, expected = format_few_shot_prompt(task)

        # 检查 prompt 长度
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens > 450:  # 留空间给输出（seq_len=512）
            if verbose:
                print(f"  [{i+1}/{len(tasks)}] {task_name}: SKIP (prompt too long: {prompt_tokens} tokens)")
            continue

        t0 = time.time()
        answer = generate_answer(model, tokenizer, prompt, expected, device=device)
        dt = time.time() - t0

        predicted = text_to_grid(answer)
        em = exact_match(predicted, expected)
        ca = cell_accuracy(predicted, expected)
        sm = shape_match(predicted, expected)

        result = {
            "task": task_name,
            "exact_match": em,
            "cell_accuracy": ca,
            "shape_match": sm,
            "prompt_tokens": prompt_tokens,
            "time_s": round(dt, 2),
            "predicted_raw": answer[:200],  # 截断存储
        }
        results.append(result)

        if verbose:
            status = "✓" if em else ("△" if sm else "✗")
            print(f"  [{i+1}/{len(tasks)}] {task_name}: {status}  cell_acc={ca:.1%}  shape={'ok' if sm else 'wrong'}  ({dt:.1f}s)")

    # 汇总
    n = len(results)
    if n == 0:
        return {"n_tasks": 0, "error": "no valid tasks"}

    summary = {
        "n_tasks": n,
        "exact_match_rate": sum(r["exact_match"] for r in results) / n,
        "mean_cell_accuracy": sum(r["cell_accuracy"] for r in results) / n,
        "shape_match_rate": sum(r["shape_match"] for r in results) / n,
        "mean_time_s": sum(r["time_s"] for r in results) / n,
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI 文本化 few-shot 评估")
    parser.add_argument("--arc_data", type=str,
                        default=str(MINIMIND_DIR.parent / "data" / "ARC-AGI" / "data" / "training"))
    parser.add_argument("--tokenizer_path", type=str,
                        default=str(MINIMIND_DIR / "model" / "qwen3_5_tokenizer"))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型 checkpoint 路径（.pth）。不指定则用随机初始化模型")
    parser.add_argument("--phase", type=int, default=4,
                        help="Phase 编号（用于构建对应的 LumaConfig）")
    parser.add_argument("--max_tasks", type=int, default=30,
                        help="最多评估多少个任务")
    parser.add_argument("--max_grid", type=int, default=8,
                        help="网格最大尺寸")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--reason_active_loops", type=int, default=None,
                        help="覆盖 reason_active_loops（eval 时可用更多 loops）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出 JSON 路径")
    args = parser.parse_args()

    print(f"ARC-AGI Text Eval | phase={args.phase} | max_tasks={args.max_tasks} | max_grid={args.max_grid}")

    # 加载任务
    tasks = load_tasks(
        Path(args.arc_data),
        max_grid=args.max_grid,
        max_train_examples=4,
        max_total_cells=200,
        max_tasks=args.max_tasks,
    )
    print(f"Loaded {len(tasks)} tasks")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建模型（复用 trainer 的 config builder）
    # 这里简化：直接用默认参数构建
    from trainer.train_luma_refactor import (
        build_phase0_config, build_phase3_config, build_phase35_config,
        build_phase4_config,
    )

    # 用 argparse namespace 模拟 trainer args
    class FakeArgs:
        pass

    fake = FakeArgs()
    # 复制所有 build config 需要的默认参数
    defaults = {
        "vocab_size": 151936, "factorized_vocab_dim": 192,
        "hidden_size": 768, "intermediate_size": 3072,
        "reason_intermediate_size": None, "reason_shared_depth": 1,
        "num_attention_heads": 12, "num_key_value_heads": 3,
        "compression_layers": 24, "compression_active_layers": 0,
        "reason_loops": 15, "reason_loops_max": 20, "reason_active_loops": 0,
        "slow_k": 1, "c_t_dim": 64, "meta_dim": 96, "meta_state": 32,
        "mamba_d_state": 192, "mamba_expand": 2, "mamba_headdim": 64,
        "mamba_chunk_size": 16, "max_seq_len": args.max_seq_len,
        "bos_token_id": 1, "eos_token_id": 2,
        "world_jepa_mode": "full", "world_mask_ratio": 0.25,
        "world_ema_decay": 0.99, "self_rollout_steps": 10,
        "self_check_dim": 16, "self_check_k": 2,
        "exit_train_use_sampling": 1, "exit_eval_use_sampling": 0,
        "exit_sampling_temperature": 1.0, "use_gradient_checkpointing": 0,
        "self_jepa_weight": 0.1, "sigreg_delta_weight": 0.05,
        "self_check_loss_weight": 0.1, "compress_weight": 0.2,
        "sigreg_ct_weight": 0.05, "enable_sigreg_ct": 0,
    }
    for k, v in defaults.items():
        setattr(fake, k, v)

    if args.phase >= 4:
        luma_config = build_phase4_config(fake)
    elif args.phase == 35:
        luma_config = build_phase35_config(fake)
    elif args.phase >= 3:
        luma_config = build_phase3_config(fake)
    else:
        luma_config = build_phase0_config(fake)

    if args.reason_active_loops is not None:
        luma_config.reason_active_loops = args.reason_active_loops
    if luma_config.vocab_size < len(tokenizer):
        luma_config.vocab_size = len(tokenizer)

    print(f"Building model...")
    model = LumaForCausalLM(luma_config).to(args.device, dtype=torch.bfloat16)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state, strict=False)
    else:
        print("WARNING: No checkpoint specified, using random weights (baseline)")

    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {params_m:.1f}M params")

    # 运行评估
    print(f"\n{'='*60}")
    print("Running ARC-AGI Text Eval...")
    print(f"{'='*60}")
    summary = run_eval(model, tokenizer, tasks, device=args.device)

    print(f"\n{'='*60}")
    print(f"Results: {summary['n_tasks']} tasks")
    print(f"  Exact Match:    {summary['exact_match_rate']:.1%}")
    print(f"  Cell Accuracy:  {summary['mean_cell_accuracy']:.1%}")
    print(f"  Shape Match:    {summary['shape_match_rate']:.1%}")
    print(f"  Avg Time:       {summary['mean_time_s']:.1f}s/task")
    print(f"{'='*60}")

    # 保存
    out_path = args.output or str(
        MINIMIND_DIR / "artifacts" / "refactor" / f"arc_eval_phase{args.phase}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # results 里不存大文本
    save_summary = {k: v for k, v in summary.items() if k != "results"}
    save_summary["per_task"] = [
        {k: v for k, v in r.items() if k != "predicted_raw"}
        for r in summary.get("results", [])
    ]
    with open(out_path, "w") as f:
        json.dump(save_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
