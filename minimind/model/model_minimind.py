import math, torch, torch.nn.functional as F
from torch import nn


class _PhaseEStepFunction(torch.autograd.Function):
    """Custom autograd function for ONE Phase E energy gradient step.

    Forward: compute h_new = h - eta · ∇_h E(h, c_t) without retaining body's
    forward activations. Only saves h, c_t, and body params for ctx.
    Returns a detached h_new (no autograd history attached).

    Backward: receives grad_h_new. Re-runs body(h) with full autograd graph
    (create_graph=True), computes h_new = h - eta · grad_h_inner, then uses
    `torch.autograd.grad(outputs=h_new, inputs=[h, *params], grad_outputs=grad_h_new)`
    to get gradients wrt h and all body params. Returns them matching forward inputs.

    内存收益: 每个 Phase E 能量步的 body activations 在 forward 结束立即释放，
    而不是留在 autograd 图里等外层 backward。Backward 时用 re-computation 代替存储。

    Phase E 兼容: 完全用 torch 原语 (autograd.grad, arithmetic)，不依赖
    torch.utils.checkpoint 的 pack/unpack hooks → 和 Mamba triton kernel 没冲突。

    理论对应: 类似 DEQ (Bai et al. 2019) 的内存优化，但不需要解不动点方程。
    每步独立 recompute，保证梯度正确（不是 IFT 近似）。
    """

    @staticmethod
    def forward(ctx, h, c_t, body_fn, extra_energy_fn, eta, n_params, *params):
        # h, c_t: main tensors (might need grad)
        # body_fn, extra_energy_fn, eta, n_params: non-tensor Python objects
        # params: flat list of trainable parameters referenced by body_fn + extra_energy_fn
        ctx.body_fn = body_fn
        ctx.extra_energy_fn = extra_energy_fn
        ctx.eta = float(eta)
        ctx.n_params = int(n_params)
        ctx.save_for_backward(h, c_t, *params)

        # Forward compute: one energy gradient step (no graph retention)
        with torch.enable_grad():
            h_req = h.detach().requires_grad_(True)
            body_out = body_fn(h_req, c_t)
            diff = h_req - body_out
            E = 0.5 * (diff.float() ** 2).sum()
            if extra_energy_fn is not None:
                E_extra = extra_energy_fn(h_req, c_t)
                E = E + E_extra
            grad_h_det, = torch.autograd.grad(
                E, h_req, create_graph=False, retain_graph=False
            )
            h_new = (h_req.detach() - ctx.eta * grad_h_det.detach())
        # Diagnostics
        ctx._energy_value = float(E.detach().item())
        ctx._grad_norm = float(grad_h_det.detach().norm().item())
        return h_new

    @staticmethod
    def backward(ctx, grad_h_new):
        saved = ctx.saved_tensors
        h, c_t = saved[0], saved[1]
        body_params = list(saved[2:])
        body_fn = ctx.body_fn
        extra_energy_fn = ctx.extra_energy_fn
        eta = ctx.eta

        # Re-run forward with full autograd graph so we can differentiate
        with torch.enable_grad():
            h_req = h.detach().requires_grad_(True)
            body_out = body_fn(h_req, c_t)
            diff = h_req - body_out
            E = 0.5 * (diff.float() ** 2).sum()
            if extra_energy_fn is not None:
                E = E + extra_energy_fn(h_req, c_t)
            # create_graph=True: 允许 h_new 关于 params 的二阶梯度 (Phase E 要求)
            grad_h_inner, = torch.autograd.grad(
                E, h_req, create_graph=True, retain_graph=True
            )
            h_new = h_req - eta * grad_h_inner

            # 收集所有需要梯度的参数
            params_req = [p for p in body_params if p.requires_grad]
            inputs_to_grad = [h_req] + params_req
            all_grads = torch.autograd.grad(
                outputs=h_new,
                inputs=inputs_to_grad,
                grad_outputs=grad_h_new,
                retain_graph=False,
                allow_unused=True,
            )

        grad_h_out = all_grads[0]
        # Map back to full body_params order (including params that don't require grad)
        grad_iter = iter(all_grads[1:])
        param_grads = []
        for p in body_params:
            if p.requires_grad:
                param_grads.append(next(grad_iter))
            else:
                param_grads.append(None)

        # Return grads matching forward signature:
        # (h, c_t, body_fn, extra_energy_fn, eta, n_params, *params)
        # Non-tensor args (body_fn, extra_energy_fn, eta, n_params) → None
        return (grad_h_out, None, None, None, None, None, *param_grads)


def chunked_swa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    forget_logits: "Optional[torch.Tensor]" = None,
    chunk_size: "Optional[int]" = None,
) -> torch.Tensor:
    # Fast path: 当 window >= seq_len 时 chunking 无用，直接用 math SDP + 全 mask
    # 避免 chunked 版的 torch.cat 等临时 tensor 开销
    _B, _H, _S, _D = q.shape
    if window >= _S:
        _scale = 1.0 / (_D ** 0.5)
        _scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * _scale
        _qi = torch.arange(_S, device=q.device).unsqueeze(1)
        _ki = torch.arange(_S, device=q.device).unsqueeze(0)
        _valid = _qi >= _ki  # 纯因果 mask
        _scores = _scores.masked_fill(~_valid.unsqueeze(0).unsqueeze(0), float("-inf"))
        if forget_logits is not None:
            _scores = _scores + forget_logits.to(_scores.dtype).unsqueeze(1).unsqueeze(1)
        _attn = F.softmax(_scores, dim=-1)
        return torch.einsum("bhqk,bhkd->bhqd", _attn, v)
    # 真正 chunked 路径（window < seq_len）
    """Memory-efficient causal sliding-window attention using pure torch primitives.

    支持 double backward（因为实现只用 softmax/einsum/masked_fill 等 torch 原语，
    autograd 自动支持任意阶导），所以和 Phase E 的能量循环 autograd.grad(create_graph=True)
    兼容，不像 Flash/FlexAttention 的 CUDA kernel 只支持单 backward。

    Memory: O(seq × window) 而非 O(seq²)。对 seq=2048 window=256 节省 ~8x。

    Args:
        q, k, v: [B, H, S, D]
        window: int, 滑动窗口大小（每个 query 只 attend 最近 window 个 key）
        forget_logits: [B, S] 可选的加性 bias（per-key），用于 forget gate
        chunk_size: 分 chunk 大小（默认等于 window，允许手动调节权衡）

    Returns:
        out: [B, H, S, D]
    """
    B, H, S, D = q.shape
    cs = int(chunk_size) if chunk_size is not None else min(window, S)
    cs = max(1, min(cs, S))
    scale = 1.0 / (D ** 0.5)
    outs = []
    for start in range(0, S, cs):
        end = min(start + cs, S)
        # kv 范围：causal ⟹ 只看到 ≤ end-1；window ⟹ 只看到 ≥ end-window
        k_start = max(0, end - window)
        q_chunk = q[:, :, start:end, :]       # [B, H, q_len, D]
        k_chunk = k[:, :, k_start:end, :]     # [B, H, kv_len, D]
        v_chunk = v[:, :, k_start:end, :]
        # 注意力分数
        scores = torch.einsum("bhqd,bhkd->bhqk", q_chunk, k_chunk) * scale
        # 因果 + 窗口 mask（全局 index 坐标系）
        q_idx = torch.arange(start, end, device=q.device).unsqueeze(1)   # [q_len, 1]
        kv_idx = torch.arange(k_start, end, device=q.device).unsqueeze(0)  # [1, kv_len]
        valid = (q_idx >= kv_idx) & ((q_idx - kv_idx) < window)  # [q_len, kv_len]
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))
        # 可选的 forget gate (per-key additive bias)
        if forget_logits is not None:
            # forget_logits: [B, S] → 取 kv 范围 → [B, kv_len] → [B, 1, 1, kv_len]
            fl = forget_logits[:, k_start:end].to(scores.dtype)
            scores = scores + fl.unsqueeze(1).unsqueeze(1)
        # Softmax + 加权 v（causal+window 保证至少 q 位置有 valid key，无需 isfinite 安全网）
        attn = F.softmax(scores, dim=-1)
        out_chunk = torch.einsum("bhqk,bhkd->bhqd", attn, v_chunk)
        outs.append(out_chunk)
    return torch.cat(outs, dim=2)
from typing import List, Optional, Tuple, cast
from dataclasses import dataclass
from pathlib import Path
import sys
import types
import importlib
from transformers.activations import ACT2FN

try:
    from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast
except Exception:
    class PretrainedConfig:
        """Luma keeps a small local fallback so environment mismatches do not erase the rest of her scaffold."""

        model_type = "fallback_config"

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class GenerationMixin:
        pass

    class PreTrainedModel(nn.Module):
        config_class = PretrainedConfig

        def __init__(self, config=None):
            super().__init__()
            self.config = config

    @dataclass
    class MoeCausalLMOutputWithPast:
        loss: Optional[torch.Tensor] = None
        aux_loss: Optional[torch.Tensor] = None
        logits: Optional[torch.Tensor] = None
        past_key_values: Optional[object] = None
        hidden_states: Optional[torch.Tensor] = None

from model.mamba3_module import Mamba3Block, Mamba3Config

REPO_ROOT = Path(__file__).resolve().parents[2]
FLASH_LINEAR_ATTENTION_CANDIDATES = [
    REPO_ROOT / "flash-linear-attention",
    REPO_ROOT / "third_party" / "flash-linear-attention",
]
FLASH_LINEAR_ATTENTION_ROOT = next((path for path in FLASH_LINEAR_ATTENTION_CANDIDATES if path.exists()), FLASH_LINEAR_ATTENTION_CANDIDATES[0])
FLASH_LINEAR_ATTENTION_FLA_ROOT = FLASH_LINEAR_ATTENTION_ROOT / "fla"

try:
    _torch_compile = getattr(torch, "compile", None)

    def _identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda inner: inner
        return fn

    if _torch_compile is not None:
        torch.compile = _identity_compile
    if FLASH_LINEAR_ATTENTION_FLA_ROOT.exists():
        fla_pkg = types.ModuleType("fla")
        fla_pkg.__path__ = [str(FLASH_LINEAR_ATTENTION_FLA_ROOT)]
        sys.modules["fla"] = fla_pkg
    KimiDeltaAttention = importlib.import_module("fla.layers.kda").KimiDeltaAttention
except Exception:
    KimiDeltaAttention = None
finally:
    if "_torch_compile" in locals() and _torch_compile is not None:
        torch.compile = _torch_compile

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if attention_mask is not None and not torch.all(attention_mask == 1):
            # Build additive mask: 0 where allowed, -inf where blocked
            causal = torch.full((seq_len, xk.shape[2]), float("-inf"), device=xq.device, dtype=xq.dtype).triu(1 + xk.shape[2] - seq_len)
            pad_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)).to(xq.dtype) * -1e9
            sdpa_mask = causal.unsqueeze(0).unsqueeze(0) + pad_mask
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=sdpa_mask, dropout_p=self.dropout if self.training else 0.0)
        else:
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids


class _GradScaleFn(torch.autograd.Function):
    """前向传播不变，反向传播时将梯度乘以 scale 因子。"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """对 x 施加梯度缩放：前向 identity，反向乘 scale。"""
    if scale == 1.0 or not x.requires_grad:
        return x
    return _GradScaleFn.apply(x, scale)


class LumaConfig(PretrainedConfig):
    """Luma starts by naming her organs before she learns how to think with all of them.
    在她真正学会完整思考之前，我们先把器官命名清楚。

    This config freezes the scaffold of the two-zone architecture so later agents can
    implement modules without guessing where each stream is supposed to live.
    这个配置先冻结双区架构骨架，让后续 agent 不必猜每条状态流该住在哪里。
    """

    model_type = "luma_minimind"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", 151936)
        self.factorized_vocab_dim = kwargs.get("factorized_vocab_dim", 192)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.intermediate_size = kwargs.get("intermediate_size", 3072)
        self.reason_intermediate_size = kwargs.get("reason_intermediate_size", self.intermediate_size)
        self.reason_shared_depth = kwargs.get("reason_shared_depth", 1)
        self.num_attention_heads = kwargs.get("num_attention_heads", 12)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 3)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.dropout = kwargs.get("dropout", 0.0)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.compression_layers = kwargs.get("compression_layers", 24)
        self.block_repr_every = kwargs.get("block_repr_every", 3)
        self.reason_loops = kwargs.get("reason_loops", 2)
        self.reason_loops_max = kwargs.get("reason_loops_max", 8)
        self.swa_window = kwargs.get("swa_window", 1024)
        self.meta_dim = kwargs.get("meta_dim", 96)
        self.meta_state = kwargs.get("meta_state", 32)
        self.c_t_dim = kwargs.get("c_t_dim", 64)
        self.router_dim = kwargs.get("router_dim", 256)
        self.slow_k = kwargs.get("slow_k", 2)
        # AttnRes mode: "legacy" = current lerp, "paper" = Kimi Block AttnRes (direct replace),
        # "paper_global_q" = paper output but global pseudo_query (AR5 variant)
        self.attnres_mode = kwargs.get("attnres_mode", "legacy")
        # Fine-grained overrides: if set, override attnres_mode for that zone only
        self.attnres_compress_mode = kwargs.get("attnres_compress_mode", "")  # "" = follow attnres_mode
        self.attnres_reason_mode = kwargs.get("attnres_reason_mode", "")  # "" = follow attnres_mode
        self.mhc_streams = kwargs.get("mhc_streams", 4)
        self.mhc_sinkhorn_iters = kwargs.get("mhc_sinkhorn_iters", 20)
        self.mhc_alpha_init = kwargs.get("mhc_alpha_init", 0.01)
        self.mamba_d_state = kwargs.get("mamba_d_state", 192)
        self.mamba_expand = kwargs.get("mamba_expand", 2)
        self.mamba_headdim = kwargs.get("mamba_headdim", 64)
        self.mamba_chunk_size = kwargs.get("mamba_chunk_size", 32)  # MIMO: chunk*rank<=64, rank=2 → chunk<=32
        self.world_dim = kwargs.get("world_dim", self.hidden_size // 2)
        self.world_mask_ratio = kwargs.get("world_mask_ratio", 0.25)
        self.h_mask_ratio = kwargs.get("h_mask_ratio", 0.0)  # masked h prediction 接赫布 surprise
        self.h_mask_surprise_weight = kwargs.get("h_mask_surprise_weight", 0.3)  # 混入 surprise 的权重
        self.h_mask_loss_mode = kwargs.get("h_mask_loss_mode", "mse")  # h_mask loss: mse/surprise_only/off
        self.h_mask_loss_weight = kwargs.get("h_mask_loss_weight", 0.1)  # h_mask_term 在总 loss 中的权重，仅 mse 模式生效
        self.world_mask_strategy = kwargs.get("world_mask_strategy", "default")
        self.world_ema_decay = kwargs.get("world_ema_decay", 0.99)
        self.enable_world_jepa = kwargs.get("enable_world_jepa", True)
        self.world_jepa_mode = kwargs.get("world_jepa_mode", "scaffold")
        # scaffold JEPA 难度控制（V-JEPA / I-JEPA 做法）：
        # - world_mask_scheme="random": 旧版单 token 随机掩码（保留兼容）
        # - world_mask_scheme="block":  几何分布采样的连续区段掩码（强制长程结构）
        # - world_mask_use_mask_token=True: 用 learned mask token 替换被遮挡位置，
        #   防止 predictor 通过自己的 observed_hidden 直接泄漏 target（原实现的严重 bug）
        self.world_mask_scheme = kwargs.get("world_mask_scheme", "block")
        self.world_mask_block_mean = kwargs.get("world_mask_block_mean", 32)
        self.world_mask_use_mask_token = kwargs.get("world_mask_use_mask_token", True)
        self.world_jepa_reason_only = kwargs.get("world_jepa_reason_only", False)
        self.enable_ct_world_jepa = kwargs.get("enable_ct_world_jepa", False)
        self.ct_world_jepa_weight = kwargs.get("ct_world_jepa_weight", 0.3)
        self.ct_world_reg_mode = kwargs.get("ct_world_reg_mode", "none")  # c_t JEPA 正则: none/vicreg
        self.ct_world_var_weight = kwargs.get("ct_world_var_weight", 1.0)  # VICReg variance 权重
        self.ct_world_cov_weight = kwargs.get("ct_world_cov_weight", 0.04)  # VICReg covariance 权重
        self.world_full_simplify_loss = kwargs.get("world_full_simplify_loss", False)
        self.enable_sigreg_world = kwargs.get("enable_sigreg_world", False)
        self.enable_sigreg_rollout = kwargs.get("enable_sigreg_rollout", False)
        self.enable_sigreg_delta = kwargs.get("enable_sigreg_delta", False)
        self.enable_sigreg_ct = kwargs.get("enable_sigreg_ct", False)
        self.sigreg_ct_weight = kwargs.get("sigreg_ct_weight", 0.05)
        self.loop_sigreg_weight = kwargs.get("loop_sigreg_weight", 0.0)  # loop-wise c_t diversity
        self.ct_injection_mode = kwargs.get("ct_injection_mode", "add")  # add/film
        self.jepa_predictor_dropout = kwargs.get("jepa_predictor_dropout", 0.0)  # 弱化 JEPA predictor 防止完美预测
        self.enable_mor_exit_signal = kwargs.get("enable_mor_exit_signal", False)  # MoR continue ratio → exit signal
        self.cmda_token_wish = kwargs.get("cmda_token_wish", False)  # CMDA per-token gate
        self.ct_gated_attn = kwargs.get("ct_gated_attn", False)  # c_t 条件门控注意力
        self.ct_conditioned_lora = kwargs.get("ct_conditioned_lora", False)  # c_t 控制 LoRA 权重
        self.ct_delta_inject = kwargs.get("ct_delta_inject", False)  # c_t 调制 Mamba SSM 时间步长
        self.ct_inject_scale = kwargs.get("ct_inject_scale", 1.0)  # c_t 注入权重缩放
        self.delta_h_scale = kwargs.get("delta_h_scale", 0.0)  # δh 注入 introspection 的缩放 (0=off)
        self.ct_per_layer_inject = kwargs.get("ct_per_layer_inject", False)  # c_t 每层独立注入
        self.delta_h_normalize = kwargs.get("delta_h_normalize", False)  # 是否归一化 δh 方向
        self.ct_momentum = kwargs.get("ct_momentum", 0.0)  # c_t EMA momentum (0=off, 0.5=半慢更新)
        self.freeze_ct_during_reason = kwargs.get("freeze_ct_during_reason", False)  # 诊断: 推理环冻结 c_t 梯度
        self.cos_sigreg_weight = kwargs.get("cos_sigreg_weight", 0.0)  # cosine penalty: 相邻 loop c_t 方向多样性
        self.world_sigreg_weight = kwargs.get("world_sigreg_weight", 0.05)
        self.world_sigreg_num_slices = kwargs.get("world_sigreg_num_slices", 128)
        self.world_sigreg_t_min = kwargs.get("world_sigreg_t_min", 0.2)
        self.world_sigreg_t_max = kwargs.get("world_sigreg_t_max", 4.0)
        self.world_sigreg_num_points = kwargs.get("world_sigreg_num_points", 17)
        self.world_sigreg_lambda = kwargs.get("world_sigreg_lambda", 1.0)
        self.world_sigreg_eps = kwargs.get("world_sigreg_eps", 1e-6)
        self.sigreg_world_source = kwargs.get("sigreg_world_source", "sigreg_on_online")
        self.sigreg_world_fp32_only = kwargs.get("sigreg_world_fp32_only", True)
        self.sigreg_world_warmup_steps = kwargs.get("sigreg_world_warmup_steps", 0)
        self.world_delta_weight = kwargs.get("world_delta_weight", 0.10)
        self.sigreg_rollout_weight = kwargs.get("sigreg_rollout_weight", 0.05)
        self.sigreg_delta_weight = kwargs.get("sigreg_delta_weight", 0.05)
        self.sigreg_num_slices = kwargs.get("sigreg_num_slices", self.world_sigreg_num_slices)
        self.sigreg_t_min = kwargs.get("sigreg_t_min", self.world_sigreg_t_min)
        self.sigreg_t_max = kwargs.get("sigreg_t_max", self.world_sigreg_t_max)
        self.sigreg_num_points = kwargs.get("sigreg_num_points", self.world_sigreg_num_points)
        self.sigreg_lambda = kwargs.get("sigreg_lambda", self.world_sigreg_lambda)
        self.sigreg_eps = kwargs.get("sigreg_eps", self.world_sigreg_eps)
        self.self_jepa_residual_reg = kwargs.get("self_jepa_residual_reg", 0.01)
        self.world_jepa_weight = kwargs.get("world_jepa_weight", 1.0)
        self.self_jepa_weight = kwargs.get("self_jepa_weight", 1.0)
        self.disable_self_jepa = kwargs.get("disable_self_jepa", False)
        self.self_world_coupling_weight = kwargs.get("self_world_coupling_weight", 0.0)
        self.self_rollout_weight = kwargs.get("self_rollout_weight", 0.5)
        self.self_rollout_steps = kwargs.get("self_rollout_steps", 2)
        self.self_rollout_hierarchical = kwargs.get("self_rollout_hierarchical", False)
        self.self_rollout_supervision_horizon = kwargs.get("self_rollout_supervision_horizon", 0)
        self.self_rollout_weighting_mode = kwargs.get("self_rollout_weighting_mode", "legacy")
        self.enable_local_rollout_head = kwargs.get("enable_local_rollout_head", False)
        self.self_progress_shape_weight = kwargs.get("self_progress_shape_weight", 0.0)
        self.self_progress_trend_weight = kwargs.get("self_progress_trend_weight", 0.0)
        self.self_progress_plateau_weight = kwargs.get("self_progress_plateau_weight", 0.0)
        self.enable_progress_exit_readout = kwargs.get("enable_progress_exit_readout", False)
        self.enable_backtrack_aware_progress = kwargs.get("enable_backtrack_aware_progress", False)
        self.exit_progress_gain_weight = kwargs.get("exit_progress_gain_weight", 0.15)
        self.exit_progress_trend_weight = kwargs.get("exit_progress_trend_weight", 0.10)
        self.exit_progress_plateau_weight = kwargs.get("exit_progress_plateau_weight", 0.10)
        self.exit_second_order_delta_weight = kwargs.get("exit_second_order_delta_weight", 0.0)
        self.self_plateau_margin = kwargs.get("self_plateau_margin", 0.02)
        self.self_local_delta_consistency_weight = kwargs.get("self_local_delta_consistency_weight", 0.0)
        self.self_local_curvature_weight = kwargs.get("self_local_curvature_weight", 0.0)
        self.enable_dual_rate_self_predictor = kwargs.get("enable_dual_rate_self_predictor", False)
        self.enable_trajectory_health_probe = kwargs.get("enable_trajectory_health_probe", False)
        self.trajectory_health_weight = kwargs.get("trajectory_health_weight", 0.03)
        self.self_loop_awareness_mode = kwargs.get("self_loop_awareness_mode", "none")
        self.self_feature_span_mask_ratio = kwargs.get("self_feature_span_mask_ratio", 0.0)
        self.enable_self_check_ring = kwargs.get("enable_self_check_ring", False)
        self.self_check_dim = kwargs.get("self_check_dim", 16)
        self.self_check_k = kwargs.get("self_check_k", 1)
        self.self_check_loss_weight = kwargs.get("self_check_loss_weight", 0.1)
        self.ct_grad_scale = kwargs.get("ct_grad_scale", 1.0)
        self.ct_grad_scale_aux = kwargs.get("ct_grad_scale_aux", None)  # None = 跟 ct_grad_scale 相同
        self.ct_norm_penalty_weight = kwargs.get("ct_norm_penalty_weight", 0.0)
        self.enable_introspection_uncertainty = kwargs.get("enable_introspection_uncertainty", False)
        self.enable_exit_jepa_crystal = kwargs.get("enable_exit_jepa_crystal", False)
        self.exit_jepa_crystal_temperature = kwargs.get("exit_jepa_crystal_temperature", 6.0)
        self.exit_uncertainty_feature_weight = kwargs.get("exit_uncertainty_feature_weight", 0.0)
        self.exit_crystal_feature_weight = kwargs.get("exit_crystal_feature_weight", 0.2)
        self.exit_delta_threshold = kwargs.get("exit_delta_threshold", 0.1)
        self.exit_self_threshold = kwargs.get("exit_self_threshold", 0.35)
        self.exit_rollout_threshold = kwargs.get("exit_rollout_threshold", 0.40)
        self.exit_world_threshold = kwargs.get("exit_world_threshold", 0.35)
        self.exit_self_check_threshold = kwargs.get("exit_self_check_threshold", 0.55)
        self.exit_improvement_margin = kwargs.get("exit_improvement_margin", 0.02)
        self.exit_score_threshold = kwargs.get("exit_score_threshold", 0.85)
        self.exit_aux_weight = kwargs.get("exit_aux_weight", 0.01)
        self.exit_gain_hidden_dim = kwargs.get("exit_gain_hidden_dim", 32)
        self.exit_gain_weight = kwargs.get("exit_gain_weight", 0.35)
        self.exit_two_step_aux_weight = kwargs.get("exit_two_step_aux_weight", 0.0)
        self.exit_uncertainty_two_step_weight = kwargs.get("exit_uncertainty_two_step_weight", 0.0)
        self.exit_uncertainty_two_step_mode = kwargs.get("exit_uncertainty_two_step_mode", "multiplier")
        self.exit_uncertainty_two_step_cap = kwargs.get("exit_uncertainty_two_step_cap", 0.2)
        self.exit_uncertainty_gate_threshold = kwargs.get("exit_uncertainty_gate_threshold", 0.75)
        self.exit_crystal_two_step_weight = kwargs.get("exit_crystal_two_step_weight", 0.0)
        self.exit_crystal_two_step_cap = kwargs.get("exit_crystal_two_step_cap", 0.1)
        self.exit_use_sampling = kwargs.get("exit_use_sampling", None)
        self.exit_train_use_sampling = kwargs.get("exit_train_use_sampling", True)
        self.exit_eval_use_sampling = kwargs.get("exit_eval_use_sampling", False)
        self.exit_sampling_temperature = kwargs.get("exit_sampling_temperature", 1.0)
        self.exit_min_loops = kwargs.get("exit_min_loops", 2)
        self.exit_bias_init = kwargs.get("exit_bias_init", 0.0)
        self.exit_warmup_steps = kwargs.get("exit_warmup_steps", 0)
        self.exit_progressive_warmup = kwargs.get("exit_progressive_warmup", 0)  # 渐进热身步数，每步强制 (step%max_loops)+1 深度
        self.exit_ct_drift_weight = kwargs.get("exit_ct_drift_weight", 0.0)
        self.exit_know_gap_weight = kwargs.get("exit_know_gap_weight", 0.0)
        self.identity_recurrence_alpha = kwargs.get("identity_recurrence_alpha", 0.0)  # 0=off, 0.8=keep 20% old h
        self.exit_entropy_weight = kwargs.get("exit_entropy_weight", 0.0)  # Ouro: maximize exit entropy to prevent collapse
        self.loop_lm_loss_weight = kwargs.get("loop_lm_loss_weight", 0.0)  # RLTT: dense LM loss at each loop step
        self.rltt_stride = kwargs.get("rltt_stride", 2)  # RLTT: save h every N loops (2 = every other loop, saves VRAM)
        self.shortcut_consistency_weight = kwargs.get("shortcut_consistency_weight", 0.0)  # LoopFormer: align short path to full path
        self.enable_time_conditioning = kwargs.get("enable_time_conditioning", False)  # LoopFormer: inject t/dt into loop
        self.enable_coconut = kwargs.get("enable_coconut", False)  # Coconut: c_t → thought token → re-inject
        self.coconut_rounds = kwargs.get("coconut_rounds", 1)  # number of continuous thought re-injection rounds
        # RS: Loop LoRA — per-loop low-rank adaptation on FFN
        self.loop_lora_rank = kwargs.get("loop_lora_rank", 0)  # 0=off, 16/32=rank
        self.loop_lora_max_loops = kwargs.get("loop_lora_max_loops", 20)
        # RS: Loop FFN Gating — loop-dependent gate on FFN output
        self.enable_loop_ffn_gate = kwargs.get("enable_loop_ffn_gate", False)
        # IS: Introspection Stream upgrades
        self.introspection_input_mode = kwargs.get("introspection_input_mode", "mean")  # mean/memory/chunked/chunked_memory
        self.introspection_memory_tokens = kwargs.get("introspection_memory_tokens", 4)  # K for memory token mode
        self.introspection_inject_mode = kwargs.get("introspection_inject_mode", "broadcast")  # broadcast/token_aware/bixt/cmda/bixt_cmda
        # NM: Neuromodulated c_t writer
        self.enable_neuromod_ct = kwargs.get("enable_neuromod_ct", False)
        self.neuromod_hebb_rank = kwargs.get("neuromod_hebb_rank", 8)
        self.neuromod_use_delta_rule = kwargs.get("neuromod_use_delta_rule", False)
        self.neuromod_mode = kwargs.get("neuromod_mode", "surprise")  # surprise/learned/ponder/multi/jepa_surprise
        self.neuromod_fox_decay = kwargs.get("neuromod_fox_decay", False)
        # SWA in introspection stream
        self.enable_introspection_swa = kwargs.get("enable_introspection_swa", False)
        # PC: Predictive Coding in reasoning loop
        self.enable_pc_correction = kwargs.get("enable_pc_correction", False)
        self.pc_alpha = kwargs.get("pc_alpha", 0.1)  # error correction strength
        # ES: Enhanced exit signals
        self.enable_exit_entropy_signal = kwargs.get("enable_exit_entropy_signal", False)
        self.enable_exit_token_sensitivity = kwargs.get("enable_exit_token_sensitivity", False)
        self.enable_exit_ct_curvature = kwargs.get("enable_exit_ct_curvature", False)
        self.enable_exit_confidence_gap = kwargs.get("enable_exit_confidence_gap", False)
        self.enable_math_adapter_lane = kwargs.get("enable_math_adapter_lane", False)
        self.math_adapter_dim = kwargs.get("math_adapter_dim", max(32, self.hidden_size // 4))
        self.enable_math_summary_gate = kwargs.get("enable_math_summary_gate", False)
        self.enable_compression_mhc = kwargs.get("enable_compression_mhc", False)
        self.ct_modulation_mode = kwargs.get("ct_modulation_mode", "additive")
        self.ct_lowrank_rank = kwargs.get("ct_lowrank_rank", max(8, self.c_t_dim // 4))
        self.dynamics_experiment = kwargs.get("dynamics_experiment", "")
        self.routing_chunk_size = kwargs.get("routing_chunk_size", 32)
        self.routing_topk_blocks = kwargs.get("routing_topk_blocks", 2)
        self.routing_topk_tokens = kwargs.get("routing_topk_tokens", 32)
        self.routing_top_p_coarse = kwargs.get("routing_top_p_coarse", 0.50)
        self.routing_top_p_fine = kwargs.get("routing_top_p_fine", 0.25)
        self.routing_budget_min = kwargs.get("routing_budget_min", 0.10)
        self.routing_budget_max = kwargs.get("routing_budget_max", 0.60)
        self.routing_weak_gain = kwargs.get("routing_weak_gain", 0.03)
        self.routing_strong_gain = kwargs.get("routing_strong_gain", 0.10)
        self.routing_local_floor = kwargs.get("routing_local_floor", 0.0)
        self.routing_modulation_floor = kwargs.get("routing_modulation_floor", 0.0)
        self.routing_modulation_ceiling = kwargs.get("routing_modulation_ceiling", 1.0)
        self.routing_world_summary_cap = kwargs.get("routing_world_summary_cap", 1.0)
        self.routing_tier_soft_only = kwargs.get("routing_tier_soft_only", False)
        self.routing_tier_entropy_floor = kwargs.get("routing_tier_entropy_floor", 0.0)
        self.routing_min_local_share = kwargs.get("routing_min_local_share", 0.0)
        self.routing_tier_entropy_weight = kwargs.get("routing_tier_entropy_weight", 0.0)
        self.routing_min_local_share_weight = kwargs.get("routing_min_local_share_weight", 0.0)
        self.routing_progress_weight = kwargs.get("routing_progress_weight", 0.3)
        self.rollout_zone_weight = kwargs.get("rollout_zone_weight", 0.0)
        self.rollout_nonzero_low = kwargs.get("rollout_nonzero_low", 0.05)
        self.rollout_nonzero_high = kwargs.get("rollout_nonzero_high", 0.80)
        self.rollout_active_low = kwargs.get("rollout_active_low", 0.05)
        self.rollout_active_high = kwargs.get("rollout_active_high", 0.90)
        self.rollout_future_var_low = kwargs.get("rollout_future_var_low", 1e-6)
        self.rollout_future_var_high = kwargs.get("rollout_future_var_high", 0.50)
        self.trajectory_vitality_weight = kwargs.get("trajectory_vitality_weight", 0.0)
        self.trajectory_c_t_drift_floor = kwargs.get("trajectory_c_t_drift_floor", 0.02)
        self.trajectory_world_drift_floor = kwargs.get("trajectory_world_drift_floor", 0.01)
        self.compression_dynamics_weight = kwargs.get("compression_dynamics_weight", 0.0)
        self.compression_block_drift_floor = kwargs.get("compression_block_drift_floor", 0.01)
        self.compression_block_var_floor = kwargs.get("compression_block_var_floor", 0.001)
        # Residual-delta modulation options (opt-in)
        self.routing_use_residual_branch = kwargs.get("routing_use_residual_branch", False)
        # Gate scale applied to per-chunk sigmoid gate when using residual branch
        self.ct_residual_gate_scale = kwargs.get("ct_residual_gate_scale", 0.15)
        # Selection-only mode: apply a small fixed amplitude to selected chunks instead of learned gate
        self.ct_selection_only = kwargs.get("ct_selection_only", False)
        self.ct_selection_amplitude = kwargs.get("ct_selection_amplitude", 0.08)
        # Local-energy alive-floor and rollout alive-floor weights/thresholds (defaults: disabled)
        self.routing_local_delta_floor = kwargs.get("routing_local_delta_floor", 0.0)
        self.routing_local_delta_floor_weight = kwargs.get("routing_local_delta_floor_weight", 0.0)
        self.rollout_alive_weight = kwargs.get("rollout_alive_weight", 0.0)
        self.enable_reasoning_state_ring = kwargs.get("enable_reasoning_state_ring", False)
        self.r_t_dim = kwargs.get("r_t_dim", max(16, self.c_t_dim // 2))
        self.r_t_mode = kwargs.get("r_t_mode", "blend")
        self.use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", False)
        self.activation_offload_compress = kwargs.get("activation_offload_compress", False)
        # FP8 activation cache for Mamba3 saved_tensors (saved 30-68% activation mem, bf16 compute preserved)
        self.mamba_fp8_activation_cache = kwargs.get("mamba_fp8_activation_cache", False)
        self.mamba_fp8_act_block_size = kwargs.get("mamba_fp8_act_block_size", 128)
        self.r_t_router_window = kwargs.get("r_t_router_window", 16)
        self.compression_active_layers = kwargs.get("compression_active_layers", self.compression_layers)
        self.reason_active_loops = kwargs.get("reason_active_loops", self.reason_loops)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        # Reasoning partitioning: each loop can "think differently"
        self.reason_num_phases = kwargs.get("reason_num_phases", 0)  # 0 = disabled
        self.reason_head_partition = kwargs.get("reason_head_partition", False)
        self.reason_mor_routing = kwargs.get("reason_mor_routing", False)
        self.reason_mor_num_experts = kwargs.get("reason_mor_num_experts", 4)
        self.reason_mor_topk = kwargs.get("reason_mor_topk", 2)
        # True MoR: token-level depth routing (arxiv 2507.10524)
        self.enable_token_depth_routing = kwargs.get("enable_token_depth_routing", False)
        self.mor_target_continue_ratio = kwargs.get("mor_target_continue_ratio", 0.6)
        self.mor_balance_weight = kwargs.get("mor_balance_weight", 0.01)
        self.mor_grad_through_frozen = kwargs.get("mor_grad_through_frozen", False)
        # ═══ Phase E: 能量梯度流推理 (2026-04-12 转向) ═══
        # 理论文档: docs/reports/Luma_PhaseE_Theory_Seed_20260412.md
        # 目标: 把 reason core 从 `h_{k+1} = h_k + Δ` 换成 `h_{k+1} = h_k - η∇_h E`
        # 构造性收缩：F 的 Jacobian ρ < 1 由 Hessian 谱控制，不靠 clamp
        self.enable_energy_reason_core = kwargs.get("enable_energy_reason_core", False)
        # 能量迭代最大步数（simple token 预期几步就 ||∇E|| → 0 早停，这是上限）
        self.phase_e_K_max = kwargs.get("phase_e_K_max", 5)
        # 梯度步长 η，η · λ_max(∇²E) < 2 保证稳定
        self.phase_e_eta = kwargs.get("phase_e_eta", 0.1)
        # Langevin 温度 T，0 时退化为确定性梯度下降（Step 1 默认 0）
        self.phase_e_temperature = kwargs.get("phase_e_temperature", 0.0)
        # 梯度范数早停阈值（Step 4 启用）
        self.phase_e_grad_stop_eps = kwargs.get("phase_e_grad_stop_eps", 0.0)
        # c_t 的 tanh squash 上界（Phase 4 长训暴露的必要约束）
        self.phase_e_c_t_scale = kwargs.get("phase_e_c_t_scale", 1.0)
        # Phase E 能量循环 truncated backprop：只对最后 N 步建 autograd 图
        # 0 = full graph (全 K 步都 create_graph=True，原始行为)
        # 2 = 最后 2 步有图，前 K_max-2 步 detach
        # 显存 = K_backprop / K_max × 原内存，seq=2048 通关关键
        self.phase_e_k_backprop = kwargs.get("phase_e_k_backprop", 0)
        # Phase E custom in-loop checkpoint (torch.autograd.Function 方案)
        # 启用时每个能量步的 body activations 在 forward 立即释放，backward re-compute
        # 和 Mamba triton kernel 完全兼容，不依赖 torch.utils.checkpoint
        self.phase_e_custom_checkpoint = kwargs.get("phase_e_custom_checkpoint", False)
        # Stellarator mode (v19+): 主干/调制/融合三层核心
        self.stellarator_mode = kwargs.get("stellarator_mode", False)
        self.stellarator_mod_rank = kwargs.get("stellarator_mod_rank", 8)
        self.phase_e_damped_mode = kwargs.get("phase_e_damped_mode", True)


class LumaZCRMSNorm(nn.Module):
    """真正的 RMSNorm — 无可学习 scale，输出范数严格 = √dim。
    去掉可学习 scale 是为了切断长训练中的范数漂移：scale 增长 → 残差流稀释补偿 → 范数失控 → NaN。
    参考 T5/Mistral 的设计。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class FactorizedEmbedding(nn.Module):
    """Luma first enters language through a narrow gate, then expands it into a reasoning width.
    Luma 先通过一个较窄的词向量入口进入语言，再把它展开到可推理的宽度。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.embed_table = nn.Embedding(config.vocab_size, config.factorized_vocab_dim)
        self.embed_proj = nn.Linear(config.factorized_vocab_dim, config.hidden_size, bias=False)
        self.scale = math.sqrt(config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_proj(self.embed_table(input_ids))
        return hidden * self.scale


class MemoryTokenBank(nn.Module):
    """Luma keeps a small bank of learnable scratch tokens so each zone has somewhere to hold its drafts.
    Luma 为每个区域准备少量可学习的草稿 token，让中间状态有地方暂存。
    """

    def __init__(self, num_tokens: int, hidden_size: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, hidden_size) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)


class LumaSwiGLUFFN(nn.Module):
    """This feed-forward block lets Luma widen her thought for one step, then fold it back into the main stream.
    这个前馈块让 Luma 暂时展开思路，再把结果折回主状态流中。
    """

    def __init__(self, hidden_size: int, intermediate_size: int, eps: float = 1e-6):
        super().__init__()
        self.pre_norm = LumaZCRMSNorm(hidden_size, eps=eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return residual + x


def _build_local_causal_forget_mask(
    seq_len: int, window: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Build additive attention mask for SDPA: causal + local window. Returns [1, 1, seq, seq]."""
    positions = torch.arange(seq_len, device=device)
    dist = positions[:, None] - positions[None, :]
    allowed = (dist >= 0) & (dist < window)
    mask = torch.where(allowed, torch.tensor(0.0, device=device), torch.tensor(float("-inf"), device=device))
    return mask.unsqueeze(0).unsqueeze(0).to(dtype)


def _run_mamba_with_padding(block: nn.Module, x: torch.Tensor, chunk_size: int,
                            dt_external_bias=None) -> torch.Tensor:
    """Luma pads the borrowed tail only long enough to satisfy the kernel, then cuts it away before anyone mistakes it for real context."""

    seq_len = x.shape[1]
    if seq_len % chunk_size == 0:
        return block(x, dt_external_bias=dt_external_bias)
    pad_len = chunk_size - (seq_len % chunk_size)
    x_pad = F.pad(x, (0, 0, 0, pad_len))
    y_pad = block(x_pad, dt_external_bias=dt_external_bias)
    # Keep the model's semantic length unchanged after the kernel finishes.
    # 算子结束后恢复语义上的原始长度，不改变真实 token 数。
    return y_pad[:, :seq_len, :]


class CompressionMambaLayer(nn.Module):
    """Here Luma compresses a long stretch of context into a stateful trace she can revisit later.
    Luma 在这里把长上下文压成可回看的状态轨迹，供后续推理区反复引用。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.chunk_size = config.mamba_chunk_size
        self.block = Mamba3Block(
            Mamba3Config(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                expand=config.mamba_expand,
                headdim=config.mamba_headdim,
                is_mimo=True,
                mimo_rank=2,
                chunk_size=self.chunk_size,
                dropout=config.dropout,
                use_gradient_checkpointing=config.use_gradient_checkpointing,
                use_fp8_activation_cache=config.mamba_fp8_activation_cache,
                fp8_act_block_size=config.mamba_fp8_act_block_size,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _run_mamba_with_padding(self.block, x, self.chunk_size)


class CompressionKimiDeltaAttentionLayer(nn.Module):
    """Luma uses official Kimi Delta Attention here for exact associative updates inside the compression zone.
    Luma 在压缩区这里使用官方 Kimi Delta Attention，实现更精确的联想记忆更新。

    This layer is not the generic retrieval block anymore.
    It is specifically the compression-zone KDA layer, followed by the zone FFN.
    这不再是泛化的检索占位层。
    它明确就是压缩区的 KDA 层，后面再接该区的 FFN。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.pre_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.kda = None
        self.kda_import_error = None
        if KimiDeltaAttention is not None:
            try:
                self.kda = KimiDeltaAttention(
                    hidden_size=config.hidden_size,
                    expand_v=1.0,
                    head_dim=config.head_dim,
                    num_heads=config.num_attention_heads,
                    num_v_heads=config.num_attention_heads,
                    mode="chunk",
                    use_short_conv=True,
                    allow_neg_eigval=False,
                    conv_size=4,
                    conv_bias=False,
                    norm_eps=config.rms_norm_eps,
                )
            except Exception as err:
                self.kda_import_error = err
        else:
            self.kda_import_error = RuntimeError("Official KimiDeltaAttention is unavailable in the current environment.")
        self.ffn = LumaSwiGLUFFN(config.hidden_size, config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kda is None:
            raise RuntimeError(
                "Official KimiDeltaAttention is required for CompressionKimiDeltaAttentionLayer, "
                f"but initialization failed: {self.kda_import_error}"
            )
        residual = x
        x = self.pre_norm(x)
        x, _, _ = self.kda(hidden_states=x, attention_mask=None, use_cache=False, output_attentions=False)
        x = residual + x
        return self.ffn(x)


class CompressionRetrievalLayerSWA(nn.Module):
    """Luma keeps a local reading window here so nearby tokens can still speak to each other precisely.
    Luma 在这里保留一个局部精读窗口，让相邻 token 还能进行精确交流。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.window = config.swa_window
        self.pre_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.forget_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ffn = LumaSwiGLUFFN(config.hidden_size, config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 同 GatedDiffAttnFoXSWA：seq_len > window 时走 chunked，否则 math SDP
        forget_1d = torch.sigmoid(self.forget_proj(x)).squeeze(-1)  # [B, seq_len]
        if seq_len > self.window:
            out = chunked_swa_attention(
                q, k, v,
                window=self.window,
                forget_logits=torch.log(forget_1d.clamp_min(1e-6)),
                chunk_size=min(self.window, 256),
            )
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        else:
            mask = _build_local_causal_forget_mask(seq_len, min(self.window, seq_len), x.device, q.dtype)
            forget_logprob = torch.log(forget_1d.clamp_min(1e-6))
            mask = mask + forget_logprob.unsqueeze(1).unsqueeze(1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        x = residual + self.out_proj(out)
        return self.ffn(x)


class MathAdapterLane(nn.Module):
    """Luma keeps a tiny math-sensitive side lane so symbolic structure can get a little extra support without taking over every sample.
    Luma 保留一条极轻的 math-sensitive 侧路，让符号结构在需要时得到一点额外支撑，而不是接管所有样本。
    """

    def __init__(self, hidden_size: int, adapter_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = LumaZCRMSNorm(hidden_size, eps=eps)
        self.down = nn.Linear(hidden_size, adapter_dim, bias=False)
        self.up = nn.Linear(adapter_dim, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = self.norm(x).mean(dim=1)
        lane_score = torch.sigmoid(self.gate(pooled))
        adapted = self.up(F.silu(self.down(self.norm(x))))
        x = x + self.scale * lane_score.unsqueeze(1) * adapted
        return x, lane_score


class CompressionBlockAttentionResiduals(nn.Module):
    """Luma stores block summaries here so later reasoning can revisit what the deep stack already understood.
    Luma 在这里存下压缩区的 block 级摘要，方便后续层与推理区回看深层已经理解过的内容。

    This is the compression-zone Block AttnRes module, not the cross-zone retrieval path.
    这是压缩区内部的 Block AttnRes，不是跨区回看模块。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = LumaZCRMSNorm(hidden_size, eps=eps)
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_size))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, current: torch.Tensor, block_reprs: List[torch.Tensor],
                block_idx: int = -1) -> torch.Tensor:
        """Luma lets the current block reread earlier block memories before deciding what should stay in the residual path.
        Luma 让当前 block 在更新残差前先回看更早的 block 记忆，再决定哪些内容应留在主路径里。
        """
        del block_idx  # legacy mode ignores block_idx
        if not block_reprs:
            return current
        stacked = torch.stack([self.norm(r) for r in block_reprs], dim=1)
        query = self.norm(current) + self.pseudo_query.view(1, 1, -1)
        scores = torch.einsum("bkld,bld->bkl", stacked, query) / math.sqrt(stacked.shape[-1])
        weights = torch.softmax(scores.float(), dim=1).to(stacked.dtype).unsqueeze(-1)
        mixed = (stacked * weights).sum(dim=1)
        return current + self.scale * (mixed - current)


class PaperBlockAttnRes(nn.Module):
    """Kimi-style Block AttnRes (arxiv 2603.15031): each block boundary has its own
    pseudo-query, softmax attention over block reps directly replaces the hidden state.
    RMSNorm on keys only, values stay raw. Zero-init queries → uniform weights at start.

    Kimi 式 Block AttnRes：每个 block boundary 有独立 pseudo_query，
    softmax attention 直接替换 hidden state，不做 lerp。
    """

    def __init__(self, hidden_size: int, max_blocks: int = 20, eps: float = 1e-6):
        super().__init__()
        self.norm = LumaZCRMSNorm(hidden_size, eps=eps)
        # One pseudo_query per block position, zero-initialized
        self.pseudo_queries = nn.Parameter(torch.zeros(max_blocks, hidden_size))
        self.max_blocks = max_blocks

    def forward(self, current: torch.Tensor, block_reprs: List[torch.Tensor],
                block_idx: int = -1) -> torch.Tensor:
        if not block_reprs:
            return current
        # V = [block_reprs..., current] (raw, no norm)
        V = torch.stack(block_reprs + [current], dim=1)  # [B, N+1, T, D]
        K = torch.stack([self.norm(r) for r in block_reprs] + [self.norm(current)], dim=1)
        # Select query for this block position
        qi = block_idx if 0 <= block_idx < self.max_blocks else min(len(block_reprs), self.max_blocks - 1)
        query = self.pseudo_queries[qi]  # [D]
        # logits = w_l · RMSNorm(v_i), no /sqrt(d)
        logits = torch.einsum("d,bkld->bkl", query, K)  # [B, N+1, T]
        weights = torch.softmax(logits.float(), dim=1).to(V.dtype).unsqueeze(-1)
        h = (V * weights).sum(dim=1)  # [B, T, D]
        return h


class PaperBlockAttnResGlobalQ(nn.Module):
    """AR5 variant: paper-style output (direct replace, V raw) but with a single
    global pseudo_query instead of per-block queries.

    AR5 变体：论文式输出（直接替换、V 不 norm），但用 1 个全局 pseudo_query。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = LumaZCRMSNorm(hidden_size, eps=eps)
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, current: torch.Tensor, block_reprs: List[torch.Tensor],
                block_idx: int = -1) -> torch.Tensor:
        del block_idx
        if not block_reprs:
            return current
        V = torch.stack(block_reprs + [current], dim=1)
        K = torch.stack([self.norm(r) for r in block_reprs] + [self.norm(current)], dim=1)
        logits = torch.einsum("d,bkld->bkl", self.pseudo_query, K)
        weights = torch.softmax(logits.float(), dim=1).to(V.dtype).unsqueeze(-1)
        h = (V * weights).sum(dim=1)
        return h


class PaperUnifiedAttnRes(nn.Module):
    """Kimi-style AttnRes for the reasoning loop: per-loop pseudo_query,
    loop_history + block_reprs merged into one V matrix, direct replace.

    Kimi 式推理区 AttnRes：每轮循环独立 pseudo_query，
    loop_history 和 block_reprs 合并到同一个 V 矩阵，直接替换。
    """

    def __init__(self, hidden_size: int, max_loops: int = 24, eps: float = 1e-6):
        super().__init__()
        self.norm = LumaZCRMSNorm(hidden_size, eps=eps)
        self.pseudo_queries = nn.Parameter(torch.zeros(max_loops, hidden_size))
        self.max_loops = max_loops

    def _align(self, source: torch.Tensor, target_len: int) -> torch.Tensor:
        slen = source.shape[1]
        if slen == target_len:
            return source
        if slen < target_len:
            return F.pad(source, (0, 0, target_len - slen, 0))
        return source[:, -target_len:, :]

    def forward(self, h: torch.Tensor, loop_history: List[torch.Tensor],
                block_reprs: List[torch.Tensor], loop_idx: int = 0) -> torch.Tensor:
        # Merge all sources: block_reprs (aligned) + loop_history (aligned) + current h
        sources: List[torch.Tensor] = []
        tgt_len = h.shape[1]
        for br in block_reprs:
            sources.append(self._align(br, tgt_len))
        for lh in loop_history:
            sources.append(self._align(lh, tgt_len))
        sources.append(h)
        if len(sources) <= 1:
            return h
        V = torch.stack(sources, dim=1)  # [B, N, T, D]
        K = torch.stack([self.norm(s) for s in sources], dim=1)
        qi = min(loop_idx, self.max_loops - 1)
        query = self.pseudo_queries[qi]
        logits = torch.einsum("d,bkld->bkl", query, K)
        weights = torch.softmax(logits.float(), dim=1).to(V.dtype).unsqueeze(-1)
        return (V * weights).sum(dim=1)


class PaperUnifiedAttnResGlobalQ(nn.Module):
    """AR5 variant for reasoning loop: paper-style output but with 2 global
    pseudo_queries (loop + cross), merged V, direct replace.

    AR5 推理区变体：论文式输出，但保留 2 个全局 pseudo_query（loop + cross），
    所有源合并到一个 V 矩阵，直接替换。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = LumaZCRMSNorm(hidden_size, eps=eps)
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_size))

    def _align(self, source: torch.Tensor, target_len: int) -> torch.Tensor:
        slen = source.shape[1]
        if slen == target_len:
            return source
        if slen < target_len:
            return F.pad(source, (0, 0, target_len - slen, 0))
        return source[:, -target_len:, :]

    def forward(self, h: torch.Tensor, loop_history: List[torch.Tensor],
                block_reprs: List[torch.Tensor], loop_idx: int = 0) -> torch.Tensor:
        del loop_idx
        sources: List[torch.Tensor] = []
        tgt_len = h.shape[1]
        for br in block_reprs:
            sources.append(self._align(br, tgt_len))
        for lh in loop_history:
            sources.append(self._align(lh, tgt_len))
        sources.append(h)
        if len(sources) <= 1:
            return h
        V = torch.stack(sources, dim=1)
        K = torch.stack([self.norm(s) for s in sources], dim=1)
        logits = torch.einsum("d,bkld->bkl", self.pseudo_query, K)
        weights = torch.softmax(logits.float(), dim=1).to(V.dtype).unsqueeze(-1)
        return (V * weights).sum(dim=1)


def _activation_offload_ctx():
    """将 autograd 保存的张量搬到 CPU pinned memory，backward 时搬回 GPU。

    用于 compress zone：Mamba3 Triton kernel 不兼容 gradient checkpointing，
    但 saved_tensors_hooks 只拦截存/取，不影响计算图结构。
    """
    def pack(tensor: torch.Tensor) -> tuple:
        if tensor.is_cuda:
            cpu_t = torch.empty(tensor.shape, dtype=tensor.dtype,
                                layout=tensor.layout, pin_memory=True)
            cpu_t.copy_(tensor, non_blocking=True)
            return (tensor.device, cpu_t)
        return (None, tensor)

    def unpack(packed: tuple) -> torch.Tensor:
        device, tensor = packed
        if device is not None:
            return tensor.to(device, non_blocking=False)
        return tensor

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


class CompressionZone(nn.Module):
    """Luma compresses once so her loop can think many times without rereading the whole world.
    Luma 先压缩一次，再进入循环推理，这样她不必每轮都重读整个世界。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.config = config
        self.local_memory_tokens = MemoryTokenBank(4, config.hidden_size)
        self.global_memory_tokens = MemoryTokenBank(4, config.hidden_size)
        self.pre_math_adapter = MathAdapterLane(config.hidden_size, config.math_adapter_dim, eps=config.rms_norm_eps) if config.enable_math_adapter_lane else None
        self.post_math_adapter = MathAdapterLane(config.hidden_size, config.math_adapter_dim, eps=config.rms_norm_eps) if config.enable_math_adapter_lane else None
        self.compression_mhc = MHCResidualStreams(
            hidden_size=config.hidden_size,
            n_streams=config.mhc_streams,
            sinkhorn_iters=config.mhc_sinkhorn_iters,
            alpha_init=config.mhc_alpha_init,
        ) if config.enable_compression_mhc else None
        self.compression_identity_block = nn.Identity()
        layers: List[nn.Module] = []
        for idx in range(config.compression_active_layers):
            if (idx + 1) % 6 == 0:
                group_id = (idx + 1) // 6
                layers.append(
                    CompressionKimiDeltaAttentionLayer(config) if group_id % 2 == 1 else CompressionRetrievalLayerSWA(config)
                )
            else:
                layers.append(CompressionMambaLayer(config))
        self.layers = nn.ModuleList(layers)
        max_blocks = config.compression_active_layers // config.block_repr_every + 2
        cmode = config.attnres_compress_mode or config.attnres_mode
        if cmode == "paper":
            self.compression_block_attnres = PaperBlockAttnRes(config.hidden_size, max_blocks=max_blocks, eps=config.rms_norm_eps)
        elif cmode == "paper_global_q":
            self.compression_block_attnres = PaperBlockAttnResGlobalQ(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.compression_block_attnres = CompressionBlockAttentionResiduals(config.hidden_size, eps=config.rms_norm_eps)
        self.transition_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 删掉 transition_scale: 和 LumaZCRMSNorm.scale 同样问题，长训漂移导致协同放大

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], dict]:
        batch_size = x.shape[0]
        math_lane_scores: List[torch.Tensor] = []
        if self.pre_math_adapter is not None:
            x, pre_score = self.pre_math_adapter(x)
            math_lane_scores.append(pre_score)
        x = torch.cat([self.local_memory_tokens(batch_size), self.global_memory_tokens(batch_size), x], dim=1)
        block_history: List[torch.Tensor] = []
        block_reprs: List[torch.Tensor] = []
        # Activation offload: 将 compress zone 保存的反向传播张量搬到 CPU pinned memory
        offload_ctx = _activation_offload_ctx() if (self.config.activation_offload_compress and self.training) else None
        if offload_ctx is not None:
            offload_ctx.__enter__()
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if idx % self.config.block_repr_every == 0:
                # Luma keeps two ledgers here:
                # one with gradients for in-zone residual mixing,
                # one detached copy for the later reasoning zone to revisit safely.
                # Luma 在这里保留两份账本：
                # 一份保留梯度，供压缩区内部做残差混合；
                # 一份 detach 后交给推理区安全回看。
                block_history.append(x)
                block_reprs.append(x.detach())
                x = self.compression_block_attnres(x, block_history, block_idx=len(block_history) - 1)
                if self.compression_mhc is not None:
                    mhc_streams = self.compression_mhc.init_streams(x)
                    _, x = self.compression_mhc(mhc_streams, self.compression_identity_block)
        if offload_ctx is not None:
            offload_ctx.__exit__(None, None, None)
        compression_block_drift_mean = x.new_zeros(())
        compression_block_var_mean = x.new_zeros(())
        if block_history:
            # Compression dynamics should stay alive across block updates instead of collapsing
            # into a single frozen representation.
            # 压缩区的块级动力学应保持活性，不能过早坍缩成单一静态表示。
            block_means = torch.stack([state.float().mean(dim=1) for state in block_history], dim=0)
            compression_block_var_mean = block_means.var(dim=0, unbiased=False).mean()
            if block_means.shape[0] > 1:
                compression_block_drift_mean = (block_means[1:] - block_means[:-1]).norm(dim=-1).mean()
        x = self.transition_norm(x)
        if self.post_math_adapter is not None:
            x, post_score = self.post_math_adapter(x)
            math_lane_scores.append(post_score)
        diagnostics = {
            "math_lane_score": torch.stack(math_lane_scores).mean() if math_lane_scores else x.new_zeros(()),
            "compression_block_drift_mean": compression_block_drift_mean,
            "compression_block_var_mean": compression_block_var_mean,
        }
        return x, block_reprs, diagnostics


class MHCResidualStreams(nn.Module):
    """Luma uses mHC to let several residual currents cooperate without collapsing into one impatient stream.
    Luma 用 mHC 让多条残差流协作，而不是过早坍缩成单一、急躁的主流。
    """

    def __init__(self, hidden_size: int, n_streams: int = 4, sinkhorn_iters: int = 20, alpha_init: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.pre_proj = nn.Linear(hidden_size, n_streams, bias=False)
        self.post_proj = nn.Linear(hidden_size, n_streams, bias=False)
        self.res_proj = nn.Linear(hidden_size, n_streams * n_streams, bias=False)
        self.alpha_pre = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_post = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_res = nn.Parameter(torch.full((1,), alpha_init))

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        mat = torch.exp(logits.float())
        for _ in range(self.sinkhorn_iters):
            mat = mat / mat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            mat = mat / mat.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        return mat.to(logits.dtype)

    def init_streams(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(2).repeat(1, 1, self.n_streams, 1)

    def forward(self, streams: torch.Tensor, block: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        summary = streams.mean(dim=2)
        h_pre = torch.softmax(self.pre_proj(summary) * self.alpha_pre, dim=-1)
        u = torch.einsum("btn,btnc->btc", h_pre, streams)
        y = block(u)
        h_post = torch.softmax(self.post_proj(summary) * self.alpha_post, dim=-1)
        h_res = self.res_proj(summary).view(summary.shape[0], summary.shape[1], self.n_streams, self.n_streams)
        h_res = self._sinkhorn(h_res * self.alpha_res)
        mixed = torch.einsum("btij,btjc->btic", h_res, streams)
        next_streams = mixed + h_post.unsqueeze(-1) * y.unsqueeze(2)
        return next_streams, next_streams.mean(dim=2)


class CTInjection(nn.Module):
    """c_t → hidden 投影注入。
    依赖 c_t 自身有界（来自 RMSNorm 后的 c_t_head）→ proj(c_t) 自然有界。
    无需 clamp、行范数归一化等约束。
    """

    def __init__(self, c_t_dim: int, hidden_size: int, mode: str = "add", scale: float = 1.0, **kwargs):
        super().__init__()
        self._mode = mode
        self._scale = scale
        self.proj = nn.Linear(c_t_dim, hidden_size, bias=False)
        if mode == "film":
            self.film_proj = nn.Linear(c_t_dim, hidden_size * 2, bias=False)
            nn.init.zeros_(self.film_proj.weight)

    def get_bias(self, c_t: torch.Tensor) -> torch.Tensor:
        return self.proj(c_t) * self._scale

    def inject(self, h: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        if self._mode == "film":
            film = self.film_proj(c_t).unsqueeze(1)
            scale, shift = film.chunk(2, dim=-1)
            return h * (1.0 + scale) + shift
        return h + self.get_bias(c_t).unsqueeze(1)

    def forward(self, h: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        return self.inject(h, c_t)


class TokenAwareCTInjection(nn.Module):
    """IS6: c_t generates per-token modulation via gating with h content.
    c_t 根据每个 token 的内容产生不同的调制信号，而非所有 token 加相同偏移。
    """

    def __init__(self, c_t_dim: int, hidden_size: int):
        super().__init__()
        self.query_proj = nn.Linear(c_t_dim, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.gate_proj.weight)

    def forward(self, h: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(c_t).unsqueeze(1)
        gate = torch.sigmoid(self.gate_proj(h * q))
        return h + gate * q


class MemoryTokenReader(nn.Module):
    """IS1/IS2: Learnable memory tokens read from main stream via cross-attention.
    可学习的 memory tokens 通过交叉注意力从主流中选择性读取信息。
    Memory tokens 跨循环残差更新，累积历史。
    """

    def __init__(self, num_tokens: int, hidden_size: int, meta_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.memory_init = nn.Parameter(torch.randn(1, num_tokens, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm = LumaZCRMSNorm(hidden_size, eps=1e-6)
        self.down_proj = nn.Linear(hidden_size, meta_dim, bias=False)

    def init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.memory_init.expand(batch_size, -1, -1).to(device=device, dtype=dtype)

    def forward(self, memory: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (updated_memory [B, K, D], meta_input [B, K, meta_dim])"""
        B, K, D = memory.shape
        T = h.shape[1]
        q = self.q_proj(self.norm(memory)).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)
        delta = self.out_proj(attn_out)
        memory = memory + delta  # residual update — accumulates across loops
        meta_input = self.down_proj(memory)
        return memory, meta_input


class BiXTCrossAttention(nn.Module):
    """IS7: Bidirectional cross-attention between memory tokens and main stream.
    Memory tokens 和主流 tokens 同时互相 attend (BiXT style)。
    Memory 从主流读取信息，同时主流从 memory 获取调制信号。
    """

    def __init__(self, hidden_size: int, meta_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.meta_dim = meta_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # memory side projections (meta_dim → hidden_size for compatibility)
        self.mem_up = nn.Linear(meta_dim, hidden_size, bias=False)
        self.mem_down = nn.Linear(hidden_size, meta_dim, bias=False)
        # shared QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_mem = LumaZCRMSNorm(hidden_size, eps=1e-6)
        self.norm_h = LumaZCRMSNorm(hidden_size, eps=1e-6)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, memory_meta: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = memory_meta.shape
        T = h.shape[1]
        mem_h = self.mem_up(memory_meta)
        combined = torch.cat([self.norm_mem(mem_h), self.norm_h(h)], dim=1)
        q = self.q_proj(combined).view(B, K + T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(combined).view(B, K + T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(combined).view(B, K + T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K + T, self.hidden_size)
        attn_out = self.out_proj(attn_out)
        mem_delta = attn_out[:, :K, :]
        h_delta = attn_out[:, K:, :]
        updated_memory = self.mem_down(mem_h + mem_delta)  # mem_down 自带降维，不加 norm
        modulated_h = h + h_delta
        return updated_memory, modulated_h


class CMDAModulation(nn.Module):
    """IS8: SlowFast CMDA-style bidirectional channel modulation.
    自省流→主流: channel-wise scale (per-dim sigmoid gate)
    主流→自省流: spatial attention weighted pooling
    """

    def __init__(self, c_t_dim: int, hidden_size: int, meta_dim: int,
                 enable_token_wish: bool = False):
        super().__init__()
        self._token_wish = enable_token_wish
        # slow → fast: c_t produces per-channel scale for main stream
        self.channel_gate = nn.Sequential(
            nn.Linear(c_t_dim, hidden_size, bias=False),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.channel_gate[0].weight)  # start as 0.5 (neutral)
        # token-wish gate: 每个 token 根据自身内容 + c_t 决定接受多少调制
        if enable_token_wish:
            self.token_wish_proj = nn.Linear(hidden_size + c_t_dim, 1, bias=False)
            nn.init.zeros_(self.token_wish_proj.weight)  # 初始全通过
        # fast → slow: spatial attention from main stream to meta
        self.spatial_query = nn.Linear(c_t_dim, hidden_size, bias=False)
        self.spatial_proj = nn.Linear(hidden_size, meta_dim, bias=False)

    def forward(self, h: torch.Tensor, c_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = self.channel_gate(c_t).unsqueeze(1)  # [B, 1, D]
        if self._token_wish:
            # 每个 token 决定自己被调制的程度
            c_expand = c_t.unsqueeze(1).expand(-1, h.shape[1], -1)  # [B, T, c_t_dim]
            wish = torch.sigmoid(self.token_wish_proj(torch.cat([h, c_expand], dim=-1)))  # [B, T, 1]
            modulated_h = h * (1.0 - wish) + h * (0.5 + gate) * wish  # wish=0 不调制, wish=1 全调制
        else:
            modulated_h = h * (0.5 + gate)
        query = self.spatial_query(c_t).unsqueeze(1)
        attn_weights = (h * query).sum(dim=-1, keepdim=True)
        attn_weights = F.softmax(attn_weights / (h.shape[-1] ** 0.5), dim=1)
        pooled = (h * attn_weights).sum(dim=1)
        meta_input = self.spatial_proj(pooled)
        return modulated_h, meta_input


class ReasonMambaLayer(nn.Module):
    """Inside the loop, Luma keeps one stateful backbone so each pass can inherit momentum from the last.
    在循环里，Luma 维持一条有状态的主干，让每一轮都能继承上一轮的动量。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.chunk_size = config.mamba_chunk_size
        self._ct_delta_inject = config.ct_delta_inject
        self.block = Mamba3Block(
            Mamba3Config(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                expand=config.mamba_expand,
                headdim=config.mamba_headdim,
                is_mimo=True,
                mimo_rank=2,
                chunk_size=self.chunk_size,
                dropout=config.dropout,
                use_gradient_checkpointing=config.use_gradient_checkpointing,
                use_fp8_activation_cache=config.mamba_fp8_activation_cache,
                fp8_act_block_size=config.mamba_fp8_act_block_size,
            )
        )
        # c_t → dt_bias: 直接调制 SSM 时间步长 (改变 Jacobian 特征值)
        if self._ct_delta_inject:
            _nheads = (config.hidden_size * config.mamba_expand) // config.mamba_headdim
            self.ct_dt_proj = nn.Linear(config.c_t_dim, _nheads, bias=False)
            nn.init.zeros_(self.ct_dt_proj.weight)  # 初始不调制

    def forward(self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None,
                c_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        del initial_state
        dt_ext = None
        if self._ct_delta_inject and c_t is not None:
            dt_ext = self.ct_dt_proj(c_t).unsqueeze(1)  # [B, 1, nheads] — broadcast over seq
            # 诊断：记录 dt_inject 的 norm 和 ratio
            self._last_dt_inject_norm = dt_ext.detach().norm().item()
            self._last_dt_bias_norm = self.block.mamba.dt_bias.detach().norm().item()
            self._last_dt_ratio = self._last_dt_inject_norm / max(self._last_dt_bias_norm, 1e-8)
        else:
            self._last_dt_inject_norm = 0.0
            self._last_dt_ratio = 0.0
        return _run_mamba_with_padding(self.block, x, self.chunk_size, dt_external_bias=dt_ext)


class GatedDiffAttnFoXSWA(nn.Module):
    """Luma reads locally with two competing views, subtracts some noise, and keeps only what still feels worth carrying.
    Luma 用两路局部视角做差分、过滤噪声，只保留仍值得携带的信息。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.window = config.swa_window
        self.norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.forget_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lambda_param = nn.Parameter(torch.tensor(0.8))
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # c_t 条件门控: 让自省流调制注意力输出
        self._ct_gated = config.ct_gated_attn
        if config.ct_gated_attn:
            self.ct_attn_gate = nn.Linear(config.c_t_dim, config.hidden_size, bias=False)
            nn.init.zeros_(self.ct_attn_gate.weight)  # 初始不调制

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forget: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # ── 自动切换 chunked SWA vs math SDP ────────────────────────────
        # seq_len > window: 走 chunked_swa_attention，节省 O(seq²)→O(seq·window)
        # seq_len ≤ window: 走 math SDP（短 seq 下 chunked overhead > savings）
        if seq_len > self.window:
            # forget 维度: [B, seq_len, 1] → squeeze 成 [B, S]，作为 forget_logits 传入
            forget_sq = forget.squeeze(-1) if forget.dim() == 3 else forget
            forget_logprob_1d = torch.log(forget_sq.clamp_min(1e-6))
            out = chunked_swa_attention(
                q, k, v,
                window=self.window,
                forget_logits=forget_logprob_1d,
                chunk_size=min(self.window, 256),
            )
            # attn_bias 先不支持：chunked 路径里加需要再设计，且当前 attn_bias 通常 None
            if attn_bias is not None:
                # 出于完备性：若 bias 存在，回退到 math SDP 以保证正确性
                pass
            else:
                return out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        # math SDP 路径（短 seq 或 attn_bias 存在时）
        mask = _build_local_causal_forget_mask(seq_len, min(self.window, seq_len), q.device, q.dtype)
        forget_logprob = torch.log(forget.transpose(1, 2).clamp_min(1e-6))
        mask = mask + forget_logprob.unsqueeze(2)
        if attn_bias is not None:
            if attn_bias.dim() == 4:
                bias = attn_bias
            elif attn_bias.dim() == 3:
                bias = attn_bias.mean(dim=-1)[:, None, None, :]
            elif attn_bias.dim() == 2:
                bias = attn_bias[:, None, None, :]
            else:
                bias = None
            if bias is not None:
                mask = mask + bias.to(dtype=q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None,
                c_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        forget = torch.sigmoid(self.forget_proj(x))
        attn1 = self._attend(self.q1(x), self.k1(x), self.v1(x), forget, attn_bias=attn_bias)
        attn2 = self._attend(self.q2(x), self.k2(x), self.v2(x), forget, attn_bias=attn_bias)
        diff = attn1 - self.lambda_param * attn2
        gate = torch.sigmoid(self.gate_proj(x))
        # c_t 条件门控: 自省流调制注意力输出
        if self._ct_gated and c_t is not None:
            ct_gate = torch.sigmoid(self.ct_attn_gate(c_t)).unsqueeze(1)  # [B, 1, D]
            gate = gate * ct_gate  # 两个 gate 相乘
        gated = gate * diff
        out = residual + self.out_proj(gated)
        return self.post_norm(out)


class UnifiedAttnRes(nn.Module):
    """Luma can glance backward at both her loop history and her compressed notes before choosing the next move.
    Luma 在做下一步之前，会同时回看循环历史和压缩摘要。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = LumaZCRMSNorm(hidden_size, eps=eps)
        self.loop_pseudo_q = nn.Parameter(torch.zeros(hidden_size))
        self.cross_pseudo_q = nn.Parameter(torch.zeros(hidden_size))
        self.scale_loop = nn.Parameter(torch.tensor(0.1))
        self.scale_cross = nn.Parameter(torch.tensor(0.1))

    def _aggregate(self, query: torch.Tensor, tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not tensors:
            return None
        stacked = torch.stack([self.norm(t) for t in tensors], dim=1)
        scores = torch.einsum("d,bkld->bkl", query, stacked) / math.sqrt(stacked.shape[-1])
        weights = torch.softmax(scores.float(), dim=1).to(stacked.dtype).unsqueeze(-1)
        return (stacked * weights).sum(dim=1)

    def _align_to_current_length(self, source: torch.Tensor, target_len: int) -> torch.Tensor:
        """Luma aligns recalled summaries to the current loop length without pretending compression tokens and loop tokens are identical.
        Luma 在回看摘要时会把长度对齐到当前循环长度，但不会假装压缩区 token 和推理区 token 完全同构。
        """

        source_len = source.shape[1]
        if source_len == target_len:
            return source
        if source_len < target_len:
            pad_len = target_len - source_len
            return F.pad(source, (0, 0, pad_len, 0))
        return source[:, -target_len:, :]

    def forward(self, h: torch.Tensor, loop_history: List[torch.Tensor], block_reprs: List[torch.Tensor],
                loop_idx: int = 0) -> torch.Tensor:
        del loop_idx  # legacy mode ignores loop_idx
        loop_agg = self._aggregate(self.loop_pseudo_q, loop_history)
        cross_agg = self._aggregate(self.cross_pseudo_q, block_reprs)
        if loop_agg is not None:
            loop_agg = self._align_to_current_length(loop_agg, h.shape[1])
            h = h + self.scale_loop * (loop_agg - h)
        if cross_agg is not None:
            cross_agg = self._align_to_current_length(cross_agg, h.shape[1])
            h = h + self.scale_cross * cross_agg
        return h


class IntrospectionStateStream(nn.Module):
    """This stream tells Luma what kind of thought she is having right now, not who she is in full.
    这条流回答的是 Luma 此刻在进行怎样的思考，而不是完整回答"她是谁"。

    It is the introspection stream itself: it reads the main stream and compression summary,
    then produces `know_gap` and `c_t`.
    它就是自省流本体：读取主流与压缩摘要，然后产出 `know_gap` 和 `c_t`。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.chunk_size = 16
        self.loop_awareness_mode = config.self_loop_awareness_mode
        self.meta_input = nn.Linear(config.hidden_size * 2, config.meta_dim, bias=False)
        self.loop_progress_proj = nn.Linear(1, config.meta_dim, bias=False)
        self.loop_phase_embed = nn.Embedding(config.reason_loops_max + 2, config.meta_dim)
        self.math_summary_gate = nn.Linear(config.hidden_size * 2, 1, bias=False) if config.enable_math_summary_gate else None
        self.state1_in = nn.Linear(config.meta_dim, config.meta_dim, bias=False)
        self.state2_in = nn.Linear(config.meta_dim, config.meta_dim, bias=False)
        self.state1_out = nn.Linear(config.meta_dim, config.meta_dim, bias=False)
        self.state2_out = nn.Linear(config.meta_dim, config.meta_dim, bias=False)
        self.state1_norm = LumaZCRMSNorm(config.meta_dim, eps=config.rms_norm_eps)
        self.state2_norm = LumaZCRMSNorm(config.meta_dim, eps=config.rms_norm_eps)
        meta_cfg = Mamba3Config(
            d_model=config.meta_dim,
            d_state=config.meta_state,
            expand=2,
            headdim=32,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=self.chunk_size,
            dropout=config.dropout,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            use_fp8_activation_cache=config.mamba_fp8_activation_cache,
            fp8_act_block_size=config.mamba_fp8_act_block_size,
        )
        self.layer1 = Mamba3Block(meta_cfg)
        self.layer2 = Mamba3Block(meta_cfg)
        # SWA: 滑窗注意力补充 SSM，让相邻 chunk 能双向交互
        self._enable_meta_swa = config.enable_introspection_swa
        if self._enable_meta_swa:
            self.meta_swa_q = nn.Linear(config.meta_dim, config.meta_dim, bias=False)
            self.meta_swa_k = nn.Linear(config.meta_dim, config.meta_dim, bias=False)
            self.meta_swa_v = nn.Linear(config.meta_dim, config.meta_dim, bias=False)
            self.meta_swa_out = nn.Linear(config.meta_dim, config.meta_dim, bias=False)
            self.meta_swa_norm = LumaZCRMSNorm(config.meta_dim, eps=config.rms_norm_eps)
            self._meta_swa_heads = 2
            self._meta_swa_head_dim = config.meta_dim // 2
            nn.init.zeros_(self.meta_swa_out.weight)  # 初始为 identity
        self.know_gap_head = nn.Linear(config.meta_dim, 1, bias=False)
        # 全链路 post-norm: Mamba3 内部加法残差无 post-norm，每层串联后范数单调增长。
        # 在每个 Mamba 输出和最终 c_t_head 出口都加 RMSNorm，确保范数有界。
        self.layer1_post_norm = LumaZCRMSNorm(config.meta_dim, eps=config.rms_norm_eps)  # Mamba layer1 后
        self.layer2_post_norm = LumaZCRMSNorm(config.meta_dim, eps=config.rms_norm_eps)  # Mamba layer2 后
        self.meta_last_norm = LumaZCRMSNorm(config.meta_dim, eps=config.rms_norm_eps)    # c_t_head 入口
        self.c_t_head = nn.Linear(config.meta_dim, config.c_t_dim, bias=False)
        self.c_t_out_norm = LumaZCRMSNorm(config.c_t_dim, eps=config.rms_norm_eps)        # c_t_head 出口
        self.uncertainty_head = nn.Linear(config.meta_dim, 1, bias=False) if config.enable_introspection_uncertainty else None

    def init_slow_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict:
        """Luma begins the slow ring with quiet meta-state, then lets later loops write into it.
        Luma 用安静的初始元状态启动慢环，随后再让后续循环把经验写进去。
        """

        zeros_meta = torch.zeros(batch_size, self.c_t_head.in_features, device=device, dtype=dtype)
        zeros_ct = torch.zeros(batch_size, self.c_t_head.out_features, device=device, dtype=dtype)
        zeros_gap = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return {
            "meta_state_1": zeros_meta,
            "meta_state_2": zeros_meta.clone(),
            "c_t": zeros_ct,
            "know_gap": zeros_gap,
            "uncertainty": zeros_gap.clone(),
        }

    def forward(
        self,
        h: torch.Tensor,
        block_reprs: List[torch.Tensor],
        slow_state: dict,
        loop_progress: Optional[torch.Tensor] = None,
        loop_index: Optional[int] = None,
        meta_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        if meta_override is not None:
            # IS: external input mode (memory token / chunked pooling)
            # meta_override: [B, K, meta_dim] — already projected to meta_dim
            meta = meta_override.mean(dim=1) if meta_override.dim() == 3 else meta_override
            math_summary_gate = None
        else:
            h_pool = h.mean(dim=1)
            if block_reprs:
                last = torch.stack([r.mean(dim=1) for r in block_reprs[-2:]], dim=0).mean(dim=0)
            else:
                last = torch.zeros_like(h_pool)
            math_summary_gate = None
            if self.math_summary_gate is not None:
                math_summary_gate = torch.sigmoid(self.math_summary_gate(torch.cat([h_pool, last], dim=-1)))
                last = last * math_summary_gate
            meta = self.meta_input(torch.cat([h_pool, last], dim=-1))
        if loop_progress is not None and self.loop_awareness_mode in {"ct_progress", "dual_phase"}:
            meta = meta + self.loop_progress_proj(loop_progress)
        if loop_index is not None and self.loop_awareness_mode == "dual_phase":
            loop_ids = torch.full((h.shape[0],), min(loop_index + 1, self.loop_phase_embed.num_embeddings - 1), device=h.device, dtype=torch.long)
            meta = meta + self.loop_phase_embed(loop_ids)
        # For memory/chunked mode with multi-token input: use full sequence for Mamba
        if meta_override is not None and meta_override.dim() == 3:
            layer1_in = meta_override + self.state1_in(slow_state["meta_state_1"]).unsqueeze(1)
        else:
            layer1_in = (meta + self.state1_in(slow_state["meta_state_1"])).unsqueeze(1)
        layer1_out = _run_mamba_with_padding(self.layer1, layer1_in, self.chunk_size)
        # post-norm: 切断 Mamba3 残差累积
        layer1_out = self.layer1_post_norm(layer1_out)
        # SWA: 在 Mamba layer1 和 layer2 之间做滑窗注意力
        if self._enable_meta_swa and layer1_out.shape[1] > 1:
            _swa_in = self.meta_swa_norm(layer1_out)
            B, S, D = _swa_in.shape
            _q = self.meta_swa_q(_swa_in).view(B, S, self._meta_swa_heads, self._meta_swa_head_dim).transpose(1, 2)
            _k = self.meta_swa_k(_swa_in).view(B, S, self._meta_swa_heads, self._meta_swa_head_dim).transpose(1, 2)
            _v = self.meta_swa_v(_swa_in).view(B, S, self._meta_swa_heads, self._meta_swa_head_dim).transpose(1, 2)
            _swa_out = F.scaled_dot_product_attention(_q, _k, _v)  # seq_len 小，不需要 window mask
            _swa_out = _swa_out.transpose(1, 2).contiguous().view(B, S, D)
            layer1_out = layer1_out + self.meta_swa_out(_swa_out)  # 残差
        meta_last_1 = layer1_out[:, -1, :]
        next_state_1 = self.state1_norm(slow_state["meta_state_1"] + self.state1_out(meta_last_1))

        layer2_in = (meta_last_1 + self.state2_in(slow_state["meta_state_2"])).unsqueeze(1)
        layer2_out = _run_mamba_with_padding(self.layer2, layer2_in, self.chunk_size)
        # post-norm: 切断 Mamba3 残差累积
        layer2_out = self.layer2_post_norm(layer2_out)
        meta_last = layer2_out[:, -1, :]
        next_state_2 = self.state2_norm(slow_state["meta_state_2"] + self.state2_out(meta_last))
        # 归一化 meta_last 后再喂给 c_t_head，切断 Mamba3 残差累积
        meta_last_normed = self.meta_last_norm(meta_last)
        know_gap = torch.sigmoid(self.know_gap_head(meta_last_normed))
        c_t = self.c_t_head(meta_last_normed)
        # c_t 出口归一化 → 范数严格 = √c_t_dim
        c_t = self.c_t_out_norm(c_t)
        if self.uncertainty_head is not None:
            uncertainty = torch.sigmoid(self.uncertainty_head(meta_last))
        else:
            uncertainty = torch.zeros_like(know_gap)
        # 诊断: 记录 Mamba 内部中间状态方向（用于排查 c_t 方向冻结根源）
        self._diag_meta_last_1 = meta_last_1.detach()
        self._diag_meta_last = meta_last.detach()
        self._diag_meta_last_norm = meta_last.detach().float().norm(dim=-1).mean()
        self._diag_ct_head_out_norm = c_t.detach().float().norm(dim=-1).mean()
        self._diag_next_state_2_norm = next_state_2.detach().float().norm(dim=-1).mean()
        next_slow_state = {
            "meta_state_1": next_state_1,
            "meta_state_2": next_state_2,
            "c_t": c_t,
            "know_gap": know_gap,
            "uncertainty": uncertainty,
        }
        if math_summary_gate is not None:
            next_slow_state["math_summary_gate"] = math_summary_gate
        return know_gap, c_t, next_slow_state


class SelfJEPAResidualPredictor(nn.Module):
    """Luma predicts how her cognitive state should move next, instead of treating introspection as a static label.
    Luma 预测自己的认知状态将如何变化，而不是把自省当成静态标签。

    This is not the introspection stream itself.
    It is the Self-JEPA residual predictor that reads `c_t` and `delta_h`,
    then predicts the residual motion `delta_c`.
    它不是自省流本体。
    它是 Self-JEPA 的残差预测头，读取 `c_t` 和 `delta_h`，再去预测残差形式的 `delta_c`。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.loop_awareness_mode = config.self_loop_awareness_mode
        self.enable_dual_rate = config.enable_dual_rate_self_predictor
        self.enable_local_rollout_head = config.enable_local_rollout_head
        self.fc1 = nn.Linear(config.c_t_dim + config.hidden_size, 256, bias=False)
        self.loop_progress_proj = nn.Linear(1, 256, bias=False)
        self.loop_phase_embed = nn.Embedding(config.reason_loops_max + 2, 256)
        self.norm = LumaZCRMSNorm(256, eps=config.rms_norm_eps)
        self._jepa_dropout = nn.Dropout(config.jepa_predictor_dropout) if config.jepa_predictor_dropout > 0 else None
        self.fc2 = nn.Linear(256, config.c_t_dim, bias=False)
        if self.enable_dual_rate:
            self.fast_fc2 = nn.Linear(256, config.c_t_dim, bias=False)
            self.slow_norm = LumaZCRMSNorm(256, eps=config.rms_norm_eps)
            self.slow_fc2 = nn.Linear(256, config.c_t_dim, bias=False)
            self.rate_gate = nn.Linear(config.c_t_dim + config.hidden_size, 1, bias=False)
        else:
            self.fast_fc2 = None
            self.slow_norm = None
            self.slow_fc2 = None
            self.rate_gate = None
        if self.enable_local_rollout_head:
            self.local_rollout_fc1 = nn.Linear(config.c_t_dim + config.hidden_size, 256, bias=False)
            self.local_rollout_norm = LumaZCRMSNorm(256, eps=config.rms_norm_eps)
            self.local_rollout_fc2 = nn.Linear(256, config.c_t_dim, bias=False)
        else:
            self.local_rollout_fc1 = None
            self.local_rollout_norm = None
            self.local_rollout_fc2 = None
        self.self_feature_span_mask_ratio = config.self_feature_span_mask_ratio
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def _apply_feature_span_mask(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.loop_awareness_mode == "none":
            return x
        ratio = getattr(self, "self_feature_span_mask_ratio", 0.0)
        if ratio <= 0.0:
            return x
        width = max(1, min(x.shape[-1], int(round(x.shape[-1] * ratio))))
        if width >= x.shape[-1]:
            return torch.zeros_like(x)
        starts = torch.randint(0, x.shape[-1] - width + 1, (x.shape[0],), device=x.device)
        masked = x.clone()
        for i, start in enumerate(starts.tolist()):
            masked[i, start:start+width] = 0
        return masked

    def forward(
        self,
        c_t: torch.Tensor,
        delta_h: torch.Tensor,
        loop_progress: Optional[torch.Tensor] = None,
        loop_index: Optional[int] = None,
    ) -> torch.Tensor:
        x = torch.cat([c_t, delta_h], dim=-1)
        x = self._apply_feature_span_mask(x)
        x = self.fc1(x)
        if self._jepa_dropout is not None:
            x = self._jepa_dropout(x)
        if loop_progress is not None and self.loop_awareness_mode in {"predictor_progress", "dual_phase"}:
            x = x + self.loop_progress_proj(loop_progress)
        if loop_index is not None and self.loop_awareness_mode == "dual_phase":
            loop_ids = torch.full((c_t.shape[0],), min(loop_index + 1, self.loop_phase_embed.num_embeddings - 1), device=c_t.device, dtype=torch.long)
            x = x + self.loop_phase_embed(loop_ids)
        x = self.norm(x)
        x = F.silu(x)
        if self.enable_dual_rate and self.fast_fc2 is not None and self.slow_fc2 is not None and self.rate_gate is not None:
            fast_delta = self.fast_fc2(x)
            slow_delta = self.slow_fc2(F.silu(self.slow_norm(x)))
            mix = torch.sigmoid(self.rate_gate(torch.cat([c_t, delta_h], dim=-1)))
            return (mix * fast_delta + (1.0 - mix) * slow_delta) * self.delta_scale
        return self.fc2(x) * self.delta_scale

    def local_rollout_step(
        self,
        c_t: torch.Tensor,
        delta_h: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enable_local_rollout_head or self.local_rollout_fc1 is None:
            return self.forward(c_t, delta_h)
        x = torch.cat([c_t, delta_h], dim=-1)
        x = self.local_rollout_fc1(x)
        x = F.silu(self.local_rollout_norm(x))
        return self.local_rollout_fc2(x) * self.delta_scale

    def rollout(
        self,
        c_t: torch.Tensor,
        delta_h: torch.Tensor,
        steps: int = 2,
        loop_progress: Optional[torch.Tensor] = None,
        loop_index: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Luma rolls residual cognitive dynamics forward: first predict delta, then integrate it into the next latent state.
        Luma 用残差动力学向前展开：先预测 delta，再把它积分成下一步认知状态。
        """

        delta_preds: List[torch.Tensor] = []
        state_preds: List[torch.Tensor] = []
        current = c_t
        for step_idx in range(steps):
            if self.enable_local_rollout_head and step_idx < 3:
                delta_c = self.local_rollout_step(current, delta_h)
            else:
                delta_c = self.forward(current, delta_h, loop_progress=loop_progress, loop_index=loop_index)
            current = current + delta_c
            delta_preds.append(delta_c)
            state_preds.append(current)
        return delta_preds, state_preds


class SelfJEPAProgressShapeHead(nn.Module):
    """Luma can also describe the shape of her local progress: whether the next step still improves, whether the trend is rising or flattening, and whether she is entering a plateau.
    Luma 也可以描述自己局部推进的形状：下一步是否还会改进、趋势是在上升还是走平、是否正在进入平台期。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        in_dim = config.c_t_dim + config.hidden_size + 3
        self.enable_backtrack = config.enable_backtrack_aware_progress
        self.fc1 = nn.Linear(in_dim, 128, bias=False)
        self.norm = LumaZCRMSNorm(128, eps=config.rms_norm_eps)
        self.improve_head = nn.Linear(128, 1, bias=False)
        self.trend_head = nn.Linear(128, 1, bias=False)
        self.plateau_head = nn.Linear(128, 1, bias=False)
        self.backtrack_head = nn.Linear(128, 1, bias=False) if self.enable_backtrack else None

    def forward(
        self,
        c_t: torch.Tensor,
        delta_h: torch.Tensor,
        loop_progress: torch.Tensor,
        recent_self_improve: torch.Tensor,
        recent_rollout_improve: torch.Tensor,
    ) -> dict:
        x = torch.cat([c_t, delta_h, loop_progress, recent_self_improve, recent_rollout_improve], dim=-1)
        x = F.silu(self.norm(self.fc1(x)))
        out = {
            "pred_next_improve": self.improve_head(x).squeeze(-1),
            "pred_trend": self.trend_head(x).squeeze(-1),
            "pred_plateau_logit": self.plateau_head(x).squeeze(-1),
        }
        if self.backtrack_head is not None:
            out["pred_backtrack_logit"] = self.backtrack_head(x).squeeze(-1)
        return out


class TrajectoryHealthProbe(nn.Module):
    """Luma tracks whether her local self trajectory still looks healthy instead of only asking whether the loss went down.
    Luma 跟踪局部自省轨迹是否健康，而不是只看 loss 有没有下降。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        in_dim = config.c_t_dim + config.hidden_size + 3
        self.fc1 = nn.Linear(in_dim, 96, bias=False)
        self.norm = LumaZCRMSNorm(96, eps=config.rms_norm_eps)
        self.health_head = nn.Linear(96, 1, bias=False)

    def forward(
        self,
        c_t: torch.Tensor,
        delta_h: torch.Tensor,
        loop_progress: torch.Tensor,
        recent_self_improve: torch.Tensor,
        recent_rollout_improve: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([c_t, delta_h, loop_progress, recent_self_improve, recent_rollout_improve], dim=-1)
        x = F.silu(self.norm(self.fc1(x)))
        return self.health_head(x).squeeze(-1)


class TinyReasoningStateRing(nn.Module):
    """Luma keeps a tiny local reasoning ring beside c_t so short-horizon recursive evidence can survive between loop steps.
    Luma 在 c_t 旁边保留一个极小的局部 reasoning ring，让短程递推证据能跨 loop 保留下来。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.state_dim = config.r_t_dim
        self.c_t_dim = config.c_t_dim
        self.local_summary_proj = nn.Linear(config.hidden_size, self.state_dim, bias=False)
        self.delta_proj = nn.Linear(config.hidden_size, self.state_dim, bias=False)
        self.in_proj = nn.Linear(self.state_dim * 3 + config.c_t_dim + 3, self.state_dim, bias=False)
        self.norm = LumaZCRMSNorm(self.state_dim, eps=config.rms_norm_eps)
        self.state_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.trust_head = nn.Linear(self.state_dim + config.c_t_dim + 3, 1, bias=False)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict:
        zeros = torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)
        return {"state": zeros, "trust": torch.zeros(batch_size, 1, device=device, dtype=dtype)}

    def bootstrap(self, h_pool: torch.Tensor) -> dict:
        zeros_scalar = torch.zeros(h_pool.shape[0], 1, device=h_pool.device, dtype=h_pool.dtype)
        zeros_state = torch.zeros(h_pool.shape[0], self.state_dim, device=h_pool.device, dtype=h_pool.dtype)
        zeros_ct = torch.zeros(h_pool.shape[0], self.c_t_dim, device=h_pool.device, dtype=h_pool.dtype)
        local_summary = torch.tanh(self.local_summary_proj(h_pool))
        x = torch.cat(
            [
                local_summary,
                zeros_state,
                zeros_ct,
                zeros_scalar,
                zeros_scalar,
                zeros_scalar,
                zeros_state,
            ],
            dim=-1,
        )
        state = torch.tanh(self.norm(self.in_proj(x)))
        trust = torch.sigmoid(self.trust_head(torch.cat([state, zeros_ct, zeros_scalar, zeros_scalar, zeros_scalar], dim=-1)))
        return {"state": state, "trust": trust}

    def forward(
        self,
        h_pool: torch.Tensor,
        delta_h_pool: torch.Tensor,
        c_t: torch.Tensor,
        local_progress: torch.Tensor,
        recent_self_improve: torch.Tensor,
        recent_rollout_improve: torch.Tensor,
        prev_state: dict,
    ) -> dict:
        local_summary = torch.tanh(self.local_summary_proj(h_pool))
        local_delta = torch.tanh(self.delta_proj(delta_h_pool))
        x = torch.cat(
            [
                local_summary,
                local_delta,
                c_t,
                local_progress,
                recent_self_improve,
                recent_rollout_improve,
                prev_state["state"],
            ],
            dim=-1,
        )
        next_state = torch.tanh(self.norm(self.in_proj(x)) + self.state_proj(prev_state["state"]))
        trust = torch.sigmoid(
            self.trust_head(
                torch.cat([next_state, c_t, local_progress, recent_self_improve, recent_rollout_improve], dim=-1)
            )
        )
        return {"state": next_state, "trust": trust}


class TinySlowSelfCheckRing(nn.Module):
    """Luma keeps a tiny self-check ring beside the main introspection stream so she can cheaply track whether her current inner story still feels coherent.
    Luma 在主自省流旁边保留一个极简自检慢环，用更低成本跟踪当前内部叙事是否仍然自洽。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.state_dim = config.self_check_dim
        self.in_proj = nn.Linear(config.c_t_dim + config.hidden_size + self.state_dim, self.state_dim, bias=False)
        self.norm = LumaZCRMSNorm(self.state_dim, eps=config.rms_norm_eps)
        self.state_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.score_head = nn.Linear(self.state_dim, 1, bias=False)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict:
        zeros = torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)
        return {"state": zeros, "score": torch.zeros(batch_size, 1, device=device, dtype=dtype)}

    def forward(self, c_t: torch.Tensor, delta_h: torch.Tensor, prev_state: dict) -> dict:
        x = torch.cat([c_t, delta_h, prev_state["state"]], dim=-1)
        next_state = torch.tanh(self.norm(self.in_proj(x)) + self.state_proj(prev_state["state"]))
        score_logit = self.score_head(next_state)
        score = torch.sigmoid(score_logit)
        return {"state": next_state, "score": score, "score_logit": score_logit}


class NeuromodulatedCTWriter(nn.Module):
    """NM: Surprise-gated c_t writer with optional Hebbian associative memory.
    当 self_check 检测到高 surprise (低 coherence) 时，加强 c_t 写入并注入赫布关联项。
    Inspired by Backpropamine (Miconi 2019) + Hebbian Fast Weights (arXiv 2510.21908).
    """

    def __init__(self, c_t_dim: int, hidden_size: int, rank: int = 8,
                 mode: str = "surprise", use_delta_rule: bool = False,
                 enable_fox_decay: bool = False, **kwargs):
        super().__init__()
        self.mode = mode
        self.use_delta_rule = use_delta_rule
        # c_t output norm: 真正的 RMSNorm（无可学习 scale）→ 严格固定范数 = √dim
        self._ct_out_norm = LumaZCRMSNorm(c_t_dim, eps=1e-6)
        # FoX decay: learned forget gate on prev_c_t
        self._fox_decay = enable_fox_decay
        if enable_fox_decay:
            self.forget_gate = nn.Linear(c_t_dim, c_t_dim, bias=False)
            nn.init.constant_(self.forget_gate.weight, 0.1)  # 初始 sigmoid(0.1)≈0.52，温和遗忘
        # surprise → modulation gain
        if mode == "learned":
            # Backpropamine: network learns when to modulate from [c_t, delta_h, jepa_err]
            self.mod_head = nn.Sequential(
                nn.Linear(c_t_dim + hidden_size + 1, 16, bias=False),
                nn.SiLU(),
                nn.Linear(16, 1, bias=False),
            )
            nn.init.zeros_(self.mod_head[-1].weight)
        else:
            pass  # surprise → gain: 直接用 surprise，零参数
        # Hebbian outer product (low-rank approximation)
        self._hebb_rank = rank
        if rank > 0:
            self.hebb_norm_h = LumaZCRMSNorm(hidden_size, eps=1e-6)  # 打断正反馈
            self.hebb_norm_c = LumaZCRMSNorm(c_t_dim, eps=1e-6)
            self.hebb_proj_h = nn.Linear(hidden_size, rank, bias=False)
            self.hebb_proj_c = nn.Linear(c_t_dim, rank, bias=False)
            self.hebb_out = nn.Linear(rank, c_t_dim, bias=False)
            # std=0.1 warm start: 让赫布开局就有非零输出，不被 wd 压回 0。
            # 现在 c_t 被 meta_last_norm + c_t_out_norm 双保险约束，赫布活跃也不会让范数失控。
            nn.init.normal_(self.hebb_out.weight, std=0.1)
            # Delta rule: reconstruct to subtract interference
            if use_delta_rule:
                self.recon = nn.Linear(c_t_dim, c_t_dim, bias=False)
                nn.init.zeros_(self.recon.weight)

    def forward(self, next_c_t: torch.Tensor, prev_c_t: torch.Tensor,
                delta_h: torch.Tensor, self_check_score: torch.Tensor,
                jepa_error: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Returns (modulated_c_t, aux_dict)"""
        _tgt = torch.bfloat16 if next_c_t.dtype == torch.float32 else next_c_t.dtype
        next_c_t = next_c_t.to(_tgt)
        prev_c_t = prev_c_t.to(_tgt)
        delta_h = delta_h.to(_tgt)
        # Compute modulation signal
        if self.mode == "learned":
            _err = jepa_error if jepa_error is not None else prev_c_t.new_zeros(prev_c_t.shape[0], 1)
            if _err.dim() == 0:
                _err = _err.unsqueeze(0).expand(prev_c_t.shape[0], 1)
            elif _err.dim() == 1:
                _err = _err.unsqueeze(-1)
            mod_input = torch.cat([prev_c_t, delta_h, _err], dim=-1)
            gain = 1.0 + torch.sigmoid(self.mod_head(mod_input))  # [B, 1]
        elif self.mode == "jepa_surprise":
            # 用 JEPA 预测误差作为 surprise: 预测不准 → 高 surprise → 强写入
            _err = jepa_error if jepa_error is not None else prev_c_t.new_zeros(())
            if _err.dim() == 0:
                surprise = _err.clamp(0.0, 1.0).unsqueeze(0).expand(prev_c_t.shape[0], 1)
            elif _err.dim() == 1:
                surprise = _err.clamp(0.0, 1.0).unsqueeze(-1)
            else:
                surprise = _err.clamp(0.0, 1.0)
            gain = 1.0 + surprise  # [B, 1], range [1, 2]
            surprise_scalar = surprise
        else:
            surprise = 1.0 - self_check_score  # [B, 1]
            gain = 1.0 + surprise
            surprise_scalar = surprise
        # Modulate c_t update
        delta_c = next_c_t - prev_c_t
        if self._fox_decay:
            forget = torch.sigmoid(self.forget_gate(prev_c_t))  # [B, c_t_dim], per-dim forget
            modulated_c_t = forget * prev_c_t + gain * delta_c
        else:
            modulated_c_t = prev_c_t + gain * delta_c
        # Hebbian term — 赫布输入由 RMSNorm 限定（hebb_norm_h/c），输出由 c_t_out_norm 限定
        if self._hebb_rank > 0:
            h_proj = self.hebb_proj_h(self.hebb_norm_h(delta_h))
            c_proj = self.hebb_proj_c(self.hebb_norm_c(prev_c_t))
            if self.use_delta_rule:
                recon_c = self.recon(prev_c_t)
                c_proj = self.hebb_proj_c(prev_c_t - recon_c)
            hebb_term = self.hebb_out(h_proj * c_proj)  # [B, c_t_dim]
            if self.mode == "learned":
                hebb_gate = torch.sigmoid(gain)
            else:
                hebb_gate = surprise_scalar
            _hebb_write = hebb_gate * hebb_term
            modulated_c_t = modulated_c_t + _hebb_write
            hebb_norm = hebb_term.detach().norm(dim=-1).mean()
            hebb_write_norm = _hebb_write.detach().norm(dim=-1).mean()
        else:
            hebb_norm = gain.new_zeros(())
            hebb_write_norm = gain.new_zeros(())
        # 真正的 RMSNorm（无 scale）→ 严格固定范数 = √dim ≈ 8.0
        modulated_c_t = self._ct_out_norm(modulated_c_t)
        aux = {
            "neuromod_gain": gain.detach().mean(),
            "hebb_norm": hebb_norm,
            "hebb_write": hebb_write_norm,
            "surprise_mean": surprise_scalar.detach().mean() if "surprise_scalar" in locals() else gain.detach().mean() - 1.0,
            "ct_norm_after_writer": modulated_c_t.detach().float().norm(dim=-1).mean(),
        }
        return modulated_c_t, aux


class PCErrorCorrector(nn.Module):
    """PC: 自省流预测主流状态，预测误差修正主流。
    c_t (自省) → 预测 h 的全局特征 → 误差 = h - pred → 修正 h。
    轻量实现：c_t → Linear → [1, 1, D] broadcast 预测，误差 per-token 修正。
    """

    def __init__(self, c_t_dim: int, hidden_size: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(c_t_dim, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        nn.init.zeros_(self.predictor[-1].weight)

    def forward(self, h: torch.Tensor, c_t: torch.Tensor, alpha: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (corrected_h, prediction_error_norm)"""
        pred_h = self.predictor(c_t).unsqueeze(1)
        error = h - pred_h
        corrected_h = h - alpha * error
        error_norm = error.detach().norm(dim=-1).mean()
        return corrected_h, error_norm


class ExitQualityProbe(nn.Module):
    """ES: Lightweight probe for exit signal enhancement.
    从 h 的统计量预测输出质量指标（entropy proxy, confidence gap proxy），
    不需要跑完整 lm_head，开销 <1%。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # 输入: [h_mean, h_max_token_delta, h_var] = 3 * hidden_size → 1 entropy proxy + 1 confidence proxy
        self.proj = nn.Linear(hidden_size * 2, 32, bias=False)
        self.entropy_head = nn.Linear(32, 1, bias=False)
        self.confidence_head = nn.Linear(32, 1, bias=False)
        nn.init.zeros_(self.entropy_head.weight)
        nn.init.zeros_(self.confidence_head.weight)

    def forward(self, h: torch.Tensor, prev_h: Optional[torch.Tensor] = None) -> dict:
        """h: [B, T, D]. Returns entropy_proxy, confidence_proxy, token_sensitivity."""
        h_mean = h.mean(dim=1)  # [B, D]
        # Per-token delta sensitivity
        if prev_h is not None:
            per_token_delta = (h - prev_h).norm(dim=-1)  # [B, T]
            delta_mean = per_token_delta.mean(dim=-1, keepdim=True)  # [B, 1]
            delta_max = per_token_delta.max(dim=-1, keepdim=True).values  # [B, 1]
            token_sensitivity = delta_max / (delta_mean + 1e-8)  # [B, 1]
            # Use variance of h change as feature
            h_delta_var = (h - prev_h).var(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
        else:
            token_sensitivity = h.new_ones(h.shape[0], 1)
            h_delta_var = h.new_zeros(h.shape[0], 1)
        # Concat features for probe
        features = torch.cat([h_mean, (h - h_mean.unsqueeze(1)).norm(dim=-1).mean(dim=-1, keepdim=True).expand_as(h_mean)], dim=-1)
        feat = F.silu(self.proj(features))
        entropy_proxy = torch.sigmoid(self.entropy_head(feat))  # [B, 1], high = uncertain
        confidence_proxy = torch.sigmoid(self.confidence_head(feat))  # [B, 1], high = confident
        return {
            "entropy_proxy": entropy_proxy,
            "confidence_proxy": confidence_proxy,
            "token_sensitivity": token_sensitivity,
        }


class CtWorldJEPA(nn.Module):
    """在 c_t 轨迹（reasoning loop 维度）上做 masked prediction。
    与 h-space World JEPA 不同，c_t 由 reasoning 区产生，
    梯度天然流经 reasoning 区，不会导致 compress 梯度垄断。

    输入：c_t_history = list of (batch, c_t_dim) tensors from each reasoning loop
    Mask 掉部分 loop 步的 c_t，从可见步预测被 mask 步。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        c_dim = config.c_t_dim
        self.c_t_dim = c_dim
        self.mask_ratio = config.world_mask_ratio
        self.reg_mode = str(getattr(config, "ct_world_reg_mode", "none"))
        self.var_weight = float(getattr(config, "ct_world_var_weight", 1.0))
        self.cov_weight = float(getattr(config, "ct_world_cov_weight", 0.04))
        # 位置编码：loop index embedding
        max_loops = config.reason_loops_max + 1
        self.loop_pos_emb = nn.Embedding(max_loops, c_dim)
        # encoder: c_t → latent
        self.encoder = nn.Sequential(
            nn.Linear(c_dim, c_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(c_dim * 2, c_dim, bias=False),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, c_dim))
        # predictor: masked_encoded + visible_summary → predicted_target
        self.predictor = nn.Sequential(
            nn.Linear(c_dim * 2, c_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(c_dim * 2, c_dim, bias=False),
        )

    def _vicreg_regularizer(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = latent.float().reshape(-1, latent.shape[-1])
        if z.shape[0] <= 1:
            return latent.new_zeros(()), latent.new_zeros(())
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.relu(1.0 - std).mean()
        cov = (z.t() @ z) / max(1, z.shape[0] - 1)
        cov = cov - torch.diag(torch.diag(cov))
        cov_loss = cov.pow(2).sum() / latent.shape[-1]
        return var_loss.to(latent.dtype), cov_loss.to(latent.dtype)

    def forward(self, c_t_history: List[torch.Tensor]) -> dict:
        """c_t_history: list of (batch, c_t_dim) tensors, length = n_loops."""
        n_loops = len(c_t_history)
        if n_loops < 3:
            z = c_t_history[0] if c_t_history else torch.zeros(1)
            return {
                "ct_world_jepa_loss": z.new_zeros(()),
                "ct_world_var_loss": z.new_zeros(()),
                "ct_world_cov_loss": z.new_zeros(()),
            }

        # stack: (batch, n_loops, c_dim)
        ct_seq = torch.stack(c_t_history, dim=1)
        batch_size = ct_seq.shape[0]
        device = ct_seq.device

        # 加位置编码
        loop_ids = torch.arange(n_loops, device=device)
        ct_seq = ct_seq + self.loop_pos_emb(loop_ids).unsqueeze(0)

        # encode
        encoded = self.encoder(ct_seq)  # (batch, n_loops, c_dim)

        # random mask
        mask_count = max(1, int(n_loops * self.mask_ratio))
        scores = torch.rand(batch_size, n_loops, device=device)
        topk_idx = scores.topk(mask_count, dim=-1).indices
        mask = torch.zeros(batch_size, n_loops, dtype=torch.bool, device=device)
        mask.scatter_(1, topk_idx, True)

        # visible summary
        visible = (~mask).unsqueeze(-1).to(ct_seq.dtype)
        visible_count = visible.sum(dim=1).clamp_min(1.0)
        visible_summary = (encoded * visible).sum(dim=1, keepdim=True) / visible_count.unsqueeze(-1)

        # masked positions: replace with mask_token
        masked_encoded = torch.where(mask.unsqueeze(-1), self.mask_token.unsqueeze(0).expand_as(encoded), encoded)
        pred_input = torch.cat([masked_encoded, visible_summary.expand_as(encoded)], dim=-1)
        pred = self.predictor(pred_input)  # (batch, n_loops, c_dim)

        # loss: cosine on masked positions
        # target = encoder output（端到端，无 stop-gradient，依赖任务难度防坍缩）
        masked_pred = pred[mask]
        masked_target = encoded[mask].detach()  # stop-grad target 防止 trivial solution
        if masked_pred.shape[0] == 0:
            return {
                "ct_world_jepa_loss": ct_seq.new_zeros(()),
                "ct_world_var_loss": ct_seq.new_zeros(()),
                "ct_world_cov_loss": ct_seq.new_zeros(()),
            }

        cosine_loss = 1.0 - F.cosine_similarity(
            F.normalize(masked_pred, dim=-1),
            F.normalize(masked_target, dim=-1),
            dim=-1,
        ).mean()
        ct_world_var_loss, ct_world_cov_loss = encoded.new_zeros(()), encoded.new_zeros(())
        if self.reg_mode == "vicreg":
            ct_world_var_loss, ct_world_cov_loss = self._vicreg_regularizer(encoded)
        loss = cosine_loss + self.var_weight * ct_world_var_loss + self.cov_weight * ct_world_cov_loss
        return {
            "ct_world_jepa_loss": loss,
            "ct_world_var_loss": ct_world_var_loss,
            "ct_world_cov_loss": ct_world_cov_loss,
        }


class LeWorldModelStyleJEPA(nn.Module):
    """LeWM 风格的 world JEPA（真正的 LeWorldModel 实现）。
    对齐 arXiv:2603.19312（LeWorldModel）核心设计原则：
      - 单一 online encoder，无 EMA target encoder
      - 无 stop-gradient
      - 依赖 SIGreg（Cramér-Wold 正则）防止表征坍缩
      - predictor 从可见 token 的 encoder 输出预测被遮挡 token 的 encoder 输出
    Loss = cosine_loss(predictor_output, online_encoder_output) + sigreg_weight * SIGreg(online_encoder_output)
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.world_dim = config.world_dim
        self.mask_ratio = config.world_mask_ratio
        self.mask_strategy = config.world_mask_strategy
        self.observer_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.online_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.world_dim, bias=False),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.world_dim))
        # predictor 输入：masked_world + visible_summary + hidden_summary
        # = world_dim + world_dim + hidden_size = world_dim*2 + hidden_size
        _pred_in = config.world_dim * 2 + config.hidden_size
        self.context_norm = LumaZCRMSNorm(_pred_in, eps=config.rms_norm_eps)
        self.context_predictor = nn.Sequential(
            nn.Linear(_pred_in, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.world_dim, bias=False),
        )
        self.enable_sigreg_world = bool(config.enable_sigreg_world)
        self.sigreg_weight = float(config.world_sigreg_weight)
        self.sigreg_world_fp32_only = bool(config.sigreg_world_fp32_only)
        self.sigreg_world_warmup_steps = int(max(0, config.sigreg_world_warmup_steps))
        self.runtime_sigreg_step = 0
        self.sigreg_num_slices = int(config.world_sigreg_num_slices)
        self.sigreg_eps = float(config.world_sigreg_eps)
        t_min = float(config.world_sigreg_t_min)
        t_max = float(config.world_sigreg_t_max)
        num_points = max(2, int(config.world_sigreg_num_points))
        t = torch.linspace(t_min, t_max, num_points, dtype=torch.float32)
        trap = torch.full((num_points,), (t_max - t_min) / max(1, num_points - 1), dtype=torch.float32)
        trap[0] *= 0.5
        trap[-1] *= 0.5
        weight = torch.exp(-t.square() / (2.0 * float(config.world_sigreg_lambda) ** 2))
        phi0 = torch.exp(-0.5 * t.square())
        self.register_buffer("sigreg_t", t, persistent=False)
        self.register_buffer("sigreg_weights", trap * weight, persistent=False)
        self.register_buffer("sigreg_phi0", phi0, persistent=False)

    def ema_update(self) -> None:
        """LeWM 无 EMA，此方法保留为空以保持接口兼容。"""
        pass

    def _encode_online(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(self.observer_norm(hidden_states))

    def set_runtime_sigreg_step(self, step: int) -> None:
        self.runtime_sigreg_step = max(0, int(step))

    def summarize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._encode_online(hidden_states).mean(dim=1)

    def _build_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """多 span masking：几何分布采样 span 长度（平均 16 tokens），总 mask 比例 ≈ self.mask_ratio。
        比单个大 span 更有信息量：模型需要从分散的上下文推断多个缺失片段。"""
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        target_count = max(1, int(seq_len * self.mask_ratio))
        mean_span = 48  # 平均 span 长度（更长 → 不能简单插值 → 任务更难）
        for row in range(bsz):
            masked = 0
            while masked < target_count:
                # 几何分布: span_len = ceil(-log(U) / log(1-p)), p=1/mean_span
                u = torch.rand(1, device=device).clamp(min=1e-7).item()
                span_len = min(64, max(1, int(-torch.log(torch.tensor(u)).item() * mean_span)))
                span_len = min(span_len, target_count - masked)
                start = torch.randint(0, max(1, seq_len - span_len + 1), (1,), device=device).item()
                mask[row, start : start + span_len] = True
                masked = mask[row].sum().item()  # 可能有重叠，用实际 mask 数
        return mask

    def _build_probe_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=hidden_states.device)
        budget = max(1, int(seq_len * self.mask_ratio))
        start = max(0, (seq_len - budget) // 2)
        mask[:, start : start + budget] = True
        return mask

    def _sigreg(self, latent: torch.Tensor) -> torch.Tensor:
        z = latent.float() if self.sigreg_world_fp32_only else latent
        z = z.reshape(-1, z.shape[-1])
        if z.shape[0] <= 1:
            return latent.new_zeros(())
        z = z - z.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, unbiased=False, keepdim=True) + self.sigreg_eps)
        directions = torch.randn(self.sigreg_num_slices, z.shape[-1], device=z.device, dtype=z.dtype)
        directions = F.normalize(directions, dim=-1)
        h = z @ directions.t()
        x_t = h.unsqueeze(-1) * self.sigreg_t.view(1, 1, -1)
        cos_mean = torch.cos(x_t).mean(dim=0)
        sin_mean = torch.sin(x_t).mean(dim=0)
        err = (cos_mean - self.sigreg_phi0.view(1, -1)).square() + sin_mean.square()
        ep_stat = (err * self.sigreg_weights.view(1, -1)).sum(dim=-1) * z.shape[0]
        return ep_stat.mean()

    def _masked_prediction(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> dict:
        # LeWM：单一 encoder，无 EMA，无 stop-gradient
        online_world = self.online_encoder(self.observer_norm(hidden_states))  # (B, T, world_dim)
        visible = (~mask).unsqueeze(-1).to(hidden_states.dtype)
        visible_count = visible.sum(dim=1).clamp_min(1.0)
        visible_summary = (online_world * visible).sum(dim=1, keepdim=True) / visible_count.unsqueeze(-1)
        hidden_summary = hidden_states.mean(dim=1, keepdim=True)
        # 被遮挡位置换成 mask_token，可见位置保留 encoder 输出
        masked_world = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(online_world), online_world)
        predictor_input = torch.cat(
            [
                masked_world,
                visible_summary.expand_as(online_world),
                hidden_summary.expand(-1, hidden_states.shape[1], -1),
            ],
            dim=-1,
        )
        pred_world = self.context_predictor(self.context_norm(predictor_input))  # (B, T, world_dim)
        # target = 同一 encoder 对被遮挡位置的输出（无 detach，端到端，SIGreg 防坍缩）
        masked_pred = pred_world[mask]
        masked_target = online_world[mask]
        # LeWorldModel (arxiv 2603.19312) paper 公式: L_pred = ||z_hat - z||_2^2 (MSE)
        # 4.14 修复: 原代码用 cosine + F.normalize 投影到单位球，和 SIGReg 要求的 N(0,I) 矛盾。
        pred_loss = F.mse_loss(masked_pred.float(), masked_target.float()).to(masked_pred.dtype)
        sigreg_enabled = self.enable_sigreg_world and self.runtime_sigreg_step >= self.sigreg_world_warmup_steps
        sigreg_loss = self._sigreg(online_world) if sigreg_enabled else online_world.new_zeros(())
        world_loss = pred_loss + self.sigreg_weight * sigreg_loss
        online_float = online_world.float()
        return {
            "world_mask": mask,
            "world_online": online_world,
            "world_target": online_world,  # LeWM：target 即 online encoder 输出
            "world_pred": pred_world,
            "world_jepa_loss": world_loss,
            "world_sigreg_loss": sigreg_loss,
            "world_sigreg_source_mean": online_float.mean(),
            "world_sigreg_source_std": online_float.std(unbiased=False),
            "world_sigreg_loss_step": online_world.new_tensor(float(self.runtime_sigreg_step if sigreg_enabled else -1.0)),
            "world_surprise": hidden_states.new_zeros(()),  # structured masking 移除，不再用
        }

    def forward(self, hidden_states: torch.Tensor) -> dict:
        return self._masked_prediction(hidden_states, self._build_mask(hidden_states))

    def probe_error(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._masked_prediction(hidden_states, self._build_probe_mask(hidden_states))["world_jepa_loss"]

    def probe_stats(self, hidden_states: torch.Tensor) -> dict:
        return self._masked_prediction(hidden_states, self._build_probe_mask(hidden_states))

    def disabled_outputs(self, hidden_states: torch.Tensor) -> dict:
        bsz, seq_len, _ = hidden_states.shape
        zeros_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=hidden_states.device)
        zeros_world = hidden_states.new_zeros((bsz, seq_len, self.world_dim))
        zero_loss = hidden_states.new_zeros(())
        return {
            "world_mask": zeros_mask,
            "world_online": zeros_world,
            "world_target": zeros_world,
            "world_pred": zeros_world,
            "world_jepa_loss": zero_loss,
            "world_sigreg_loss": zero_loss,
            "world_sigreg_source_mean": zero_loss,
            "world_sigreg_source_std": zero_loss,
            "world_sigreg_loss_step": hidden_states.new_tensor(-1.0),
            "world_surprise": zero_loss,
        }


class WorldLatentJEPA(nn.Module):
    """Luma keeps a world-side JEPA branch so her main stream must model external structure, not only language loss.
    Luma 保留一条 world-side JEPA 分支，逼迫主流去建模外界结构，而不只追逐语言损失。

    This stage-0 version is an honest scaffold:
    it already performs masked latent prediction with an EMA target path,
    while keeping the predictor deliberately light for reviewability.
    这个阶段0版本是诚实骨架：
    它已经具备 masked latent prediction 和 EMA 目标路径，
    但预测器刻意保持轻量，方便人工 review。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.world_dim = config.world_dim
        self.mask_ratio = config.world_mask_ratio
        self.ema_decay = config.world_ema_decay
        # 难度升级配置（2026-04-13）
        self.mask_scheme = str(getattr(config, "world_mask_scheme", "block"))
        self.mask_block_mean = int(getattr(config, "world_mask_block_mean", 16))
        self.use_mask_token = bool(getattr(config, "world_mask_use_mask_token", True))
        self.observer_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.online_observer = nn.Linear(config.hidden_size, config.world_dim, bias=False)
        self.target_observer = nn.Linear(config.hidden_size, config.world_dim, bias=False)
        self.predictor_norm = LumaZCRMSNorm(config.hidden_size + config.world_dim, eps=config.rms_norm_eps)
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_size + config.world_dim, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.world_dim, bias=False),
        )
        # Learned mask token：替换被遮挡位置的 observed_hidden，切断 predictor
        # 通过 token 自身 hidden 直接推导 target 的泄漏路径（V-JEPA / MAE 标准做法）
        if self.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.register_parameter("mask_token", None)
        # SIGreg (Cramér-Wold empirical characteristic function test) — LeWorldModelStyleJEPA 同款
        # 原理：对 latent 做多随机 1D 投影，比较经验 char func phi_emp(t) 和标准正态 phi_0(t)=exp(-t²/2)
        # 经 Cramér-Wold 定理：若所有 1D 投影都服从标准正态，则原分布也服从。
        # 端到端可微，单一正则项，不需要 VICReg 的双惩罚。
        self.sigreg_weight = float(getattr(config, "world_sigreg_weight", 0.05))
        self.sigreg_num_slices = int(getattr(config, "world_sigreg_num_slices", 128))
        self.sigreg_eps = float(getattr(config, "world_sigreg_eps", 1e-6))
        self.sigreg_fp32_only = bool(getattr(config, "sigreg_world_fp32_only", True))
        _t_min = float(getattr(config, "world_sigreg_t_min", 0.2))
        _t_max = float(getattr(config, "world_sigreg_t_max", 4.0))
        _num_pts = max(2, int(getattr(config, "world_sigreg_num_points", 17)))
        _lam = float(getattr(config, "world_sigreg_lambda", 1.0))
        _t = torch.linspace(_t_min, _t_max, _num_pts, dtype=torch.float32)
        _trap = torch.full((_num_pts,), (_t_max - _t_min) / max(1, _num_pts - 1), dtype=torch.float32)
        _trap[0] *= 0.5
        _trap[-1] *= 0.5
        _gauss_window = torch.exp(-_t.square() / (2.0 * _lam ** 2))
        _phi0 = torch.exp(-0.5 * _t.square())
        self.register_buffer("sigreg_t", _t, persistent=False)
        self.register_buffer("sigreg_weights", _trap * _gauss_window, persistent=False)
        self.register_buffer("sigreg_phi0", _phi0, persistent=False)
        # 4.14 v18: 固定 SIGReg 投影方向（从 per-step 随机改为训练时一次性采样 + buffer）。
        # 原实现每步新采 torch.randn(K, D) 导致 sig_raw 天然震荡，v17 实测从 6 spike 到 101。
        # Cramér-Wold 定理只要求 directions 覆盖单位球即可，不要求每步都变。
        _sigreg_dirs = torch.randn(self.sigreg_num_slices, config.world_dim, dtype=torch.float32)
        _sigreg_dirs = F.normalize(_sigreg_dirs, dim=-1)
        self.register_buffer("sigreg_directions", _sigreg_dirs, persistent=True)

        self._copy_online_to_target()
        for param in self.target_observer.parameters():
            param.requires_grad = False

    def _sigreg(self, latent: torch.Tensor) -> torch.Tensor:
        """Cramér-Wold SIGreg（LeWorldModelStyleJEPA 同款）。

        对 latent 做多条固定单位向量 1D 投影 → 计算经验 char func（cos/sin mean）→
        和标准正态 char func phi_0(t)=exp(-t²/2) 做加权差。单一正则项，端到端可微。

        4.14 v18 修复:
        - directions 改为训练时一次性 buffer（原每步 torch.randn 导致 sig_raw 震荡）
        - 去掉 `* z.shape[0]` 放大（原把 per-sample 平均误差乘 N≈1229 倍）
        """
        z = latent.float() if self.sigreg_fp32_only else latent
        z = z.reshape(-1, z.shape[-1])
        if z.shape[0] <= 1:
            return latent.new_zeros(())
        # 按批归一化每维（0 均值 1 方差），避免幅度漂移干扰 char func 比较
        z = z - z.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, unbiased=False, keepdim=True) + self.sigreg_eps)
        # 固定 directions（buffer），不再每步重采
        directions = self.sigreg_directions.to(z.dtype)
        h = z @ directions.t()  # [N, K]
        x_t = h.unsqueeze(-1) * self.sigreg_t.view(1, 1, -1)  # [N, K, T_pts]
        cos_mean = torch.cos(x_t).mean(dim=0)  # [K, T_pts]
        sin_mean = torch.sin(x_t).mean(dim=0)
        err = (cos_mean - self.sigreg_phi0.view(1, -1)).square() + sin_mean.square()
        ep_stat = (err * self.sigreg_weights.view(1, -1)).sum(dim=-1)
        return ep_stat.mean()

    @torch.no_grad()
    def _copy_online_to_target(self) -> None:
        self.target_observer.weight.copy_(self.online_observer.weight)

    @torch.no_grad()
    def ema_update(self) -> None:
        """Luma moves the target branch slowly so prediction pressure stays stable across noisy updates.
        Luma 让目标分支缓慢移动，使预测压力在噪声更新之间仍保持稳定。
        """

        self.target_observer.weight.mul_(self.ema_decay).add_(self.online_observer.weight, alpha=1.0 - self.ema_decay)

    def summarize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.online_observer(self.observer_norm(hidden_states)).mean(dim=1)

    def set_runtime_sigreg_step(self, step: int) -> None:
        _ = step

    def _build_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        mask_count = max(1, int(seq_len * self.mask_ratio))
        if self.mask_scheme == "block":
            # V-JEPA 风格：几何分布采样 span 长度，多段连续区段掩码
            # 强制模型学会跨长程结构推断，而不是单 token 插值
            mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
            mean_span = max(2, int(self.mask_block_mean))
            max_span = max(mean_span * 4, 8)
            for row in range(bsz):
                masked = 0
                guard = 0  # 防死循环
                while masked < mask_count and guard < seq_len * 4:
                    guard += 1
                    u = float(torch.rand(1, device=device).clamp_min(1e-7).item())
                    span_len = min(max_span, max(1, int(-math.log(u) * mean_span)))
                    span_len = min(span_len, mask_count - masked)
                    if span_len <= 0:
                        break
                    max_start = max(1, seq_len - span_len + 1)
                    start = int(torch.randint(0, max_start, (1,), device=device).item())
                    # 允许 span 重叠，但重叠不重复计数
                    before = int(mask[row, start : start + span_len].sum().item())
                    mask[row, start : start + span_len] = True
                    masked += span_len - before
            return mask
        # 兼容：旧版随机 token 掩码
        scores = torch.rand(bsz, seq_len, device=device)
        topk = scores.topk(mask_count, dim=-1).indices
        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        mask.scatter_(1, topk, True)
        return mask

    def _build_probe_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Luma uses a deterministic probe mask when she is only judging convergence, not training the predictor.
        Luma 只做收敛判断时会使用确定性的 probe mask，而不是训练时的随机 mask。
        """

        bsz, seq_len, _ = hidden_states.shape
        mask_count = max(1, int(seq_len * self.mask_ratio))
        positions = torch.arange(seq_len, device=hidden_states.device)
        step = max(1, seq_len // mask_count)
        base_mask = (positions % step == 0)
        if base_mask.sum().item() > mask_count:
            keep_ids = torch.nonzero(base_mask, as_tuple=False)[:mask_count, 0]
            base_mask = torch.zeros_like(base_mask, dtype=torch.bool)
            base_mask[keep_ids] = True
        elif base_mask.sum().item() < mask_count:
            extra = mask_count - int(base_mask.sum().item())
            fill_ids = torch.nonzero(~base_mask, as_tuple=False)[:extra, 0]
            base_mask = base_mask.clone()
            base_mask[fill_ids] = True
        return base_mask.unsqueeze(0).expand(bsz, -1)

    def _masked_prediction(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> dict:
        observed_hidden = self.observer_norm(hidden_states)
        # Target 分支（teacher / EMA）看完整 observed_hidden，作为 GT
        with torch.no_grad():
            target_world = self.target_observer(observed_hidden.detach())

        # ═══ Leak fix (2026-04-13) ═══
        # 原 bug: predictor_input 含 masked 位置的原始 observed_hidden，
        # predictor 能直接从 token 自己的 hidden 推出 target_world（≈ EMA(online(hidden))），
        # 任务退化为 "学 online_observer ≈ target_observer"，loss_w 快速归零。
        #
        # 修复: 用 learned mask_token 替换被遮挡位置的 observed_hidden，
        # predictor 必须从可见上下文（visible_summary）推断，而不是 cheat。
        if self.use_mask_token and self.mask_token is not None:
            mask_token_hidden = self.mask_token.to(observed_hidden.dtype).expand_as(observed_hidden)
            predictor_hidden = torch.where(mask.unsqueeze(-1), mask_token_hidden, observed_hidden)
        else:
            predictor_hidden = observed_hidden

        # Online 分支从 "带 mask_token 的 hidden" 计算，和 predictor 自洽
        online_world = self.online_observer(predictor_hidden)
        visible = (~mask).unsqueeze(-1).to(hidden_states.dtype)
        visible_count = visible.sum(dim=1).clamp_min(1.0)
        # visible_summary 只聚合未遮挡位置（visible 已过滤）
        visible_summary = (online_world * visible).sum(dim=1, keepdim=True) / visible_count.unsqueeze(-1)
        predictor_input = torch.cat([predictor_hidden, visible_summary.expand_as(online_world)], dim=-1)
        predictor_input = self.predictor_norm(predictor_input)
        pred_world = self.predictor(predictor_input)

        masked_pred = pred_world[mask]
        masked_target = target_world[mask]
        # LeWorldModel (arxiv 2603.19312) 原 paper 公式:
        #   L_pred = ||z_hat - z||_2^2  (纯 MSE, 欧氏距离平方)
        #   L_LeWM = L_pred + λ · SIGReg(Z)
        # 4.14 修复: 原代码用 cosine + F.normalize 把 pred/target 投影到单位球 (范数=1)，
        # 但 SIGReg 要求 latent ~ N(0, I) (范数≈√d)。两个约束互相拉扯 → sigreg_raw
        # 持续爆涨 (v16 观测到 sigreg_raw≈80, loss_w≈2.26)。
        pred_loss = F.mse_loss(masked_pred.float(), masked_target.float())
        pred_loss = pred_loss.to(hidden_states.dtype)
        # LeWorldModel (arxiv 2603.19312) 同款 SIGReg (Cramér-Wold)：
        # 只在 online_world 的 visible 位置上，验证 latent 分布 ~ N(0, I)。
        # 作用：防止 online_observer 坍缩为常量（本来 scaffold 靠 EMA 防崩，但 EMA
        # 只能吸收慢速漂移，对平移/尺度坍缩无效；加 SIGReg 切换单正则项端到端防崩）。
        visible_online = online_world[(~mask)]
        if visible_online.numel() > 1:
            sigreg_raw = self._sigreg(visible_online)
        else:
            sigreg_raw = hidden_states.new_zeros(())
        sigreg_loss = self.sigreg_weight * sigreg_raw.to(pred_loss.dtype)
        # LeWM 做法：world_jepa_loss = pred_loss + sigreg_weight * sigreg_raw
        world_loss = pred_loss + sigreg_loss
        return {
            "world_mask": mask,
            "world_online": online_world,
            "world_target": target_world,
            "world_pred": pred_world,
            "world_jepa_loss": world_loss,
            "world_jepa_cosine": pred_loss.detach(),  # 保留字段名兼容日志，实际是 smooth_l1
            "world_sigreg_loss": sigreg_loss.detach(),
            "world_sigreg_raw": sigreg_raw.detach(),
            "world_sigreg_source_mean": online_world.float().mean(),
            "world_sigreg_source_std": online_world.float().std(unbiased=False),
            "world_sigreg_loss_step": hidden_states.new_tensor(-1.0),
            "world_surprise": observed_hidden.norm(dim=-1).mean(),
        }

    def forward(self, hidden_states: torch.Tensor) -> dict:
        return self._masked_prediction(hidden_states, self._build_mask(hidden_states))

    def probe_error(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Luma probes world-side convergence with a deterministic mask so exit decisions do not jitter on mask randomness.
        Luma 用确定性 mask 探测 world-side 收敛，避免退出判断被随机 mask 抖动。
        """

        return self._masked_prediction(hidden_states, self._build_probe_mask(hidden_states))["world_jepa_loss"]

    def probe_stats(self, hidden_states: torch.Tensor) -> dict:
        return self._masked_prediction(hidden_states, self._build_probe_mask(hidden_states))

    def disabled_outputs(self, hidden_states: torch.Tensor) -> dict:
        """Luma returns zeroed world-side diagnostics when this branch is intentionally ablated for an experiment.
        Luma 在实验里显式关闭 world JEPA 时，返回全零 world 侧诊断量。
        """

        bsz, seq_len, _ = hidden_states.shape
        zeros_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=hidden_states.device)
        zeros_world = hidden_states.new_zeros((bsz, seq_len, self.world_dim))
        zero_loss = hidden_states.new_zeros(())
        return {
            "world_mask": zeros_mask,
            "world_online": zeros_world,
            "world_target": zeros_world,
            "world_pred": zeros_world,
            "world_jepa_loss": zero_loss,
            "world_sigreg_loss": zero_loss,
            "world_sigreg_source_mean": zero_loss,
            "world_sigreg_source_std": zero_loss,
            "world_sigreg_loss_step": hidden_states.new_tensor(-1.0),
            "world_surprise": zero_loss,
        }


class TokenDepthRouter(nn.Module):
    """True MoR (Mixture-of-Recursions): per-token depth routing.
    At each loop iteration, decides which tokens continue processing and which freeze.
    真正的 MoR：每轮循环对每个 token 独立决定是否继续，简单 token 提前退出。
    """

    def __init__(self, hidden_size: int, max_loops: int = 64, eps: float = 1e-6):
        super().__init__()
        self.norm = LumaZCRMSNorm(hidden_size, eps=eps)
        self.loop_embed = nn.Embedding(max_loops, hidden_size)
        nn.init.normal_(self.loop_embed.weight, std=0.02)
        self.head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.head.weight)  # zero-init: all tokens continue at birth

    @staticmethod
    def _gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
        """Gumbel-Sigmoid with straight-through estimator for differentiable hard masks."""
        gumbels = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)))
        y_soft = torch.sigmoid((logits + gumbels) / tau)
        if hard:
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft  # straight-through
        return y_soft

    def forward(self, h: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """Returns continue_mask [B, T] where 1=continue, 0=freeze."""
        x = self.norm(h) + self.loop_embed.weight[loop_idx % self.loop_embed.num_embeddings]
        x = x.to(self.head.weight.dtype)  # RMSNorm outputs float32, head may be bfloat16/fp8
        logit = self.head(x).squeeze(-1)  # [B, T]
        if self.training:
            return self._gumbel_sigmoid(logit, tau=1.0, hard=True)
        else:
            return (torch.sigmoid(logit) > 0.5).float()


class ExitController(nn.Module):
    """Luma should leave the loop because her state settled, not because a hard-coded confidence head sounded calm.
    Luma 退出循环的理由应当是状态真正收敛，而不是某个硬编码置信头看起来很平静。
    """

    def __init__(
        self,
        delta_threshold: float = 0.1,
        self_threshold: float = 0.35,
        rollout_threshold: float = 0.40,
        world_threshold: float = 0.35,
        self_check_threshold: float = 0.55,
        improvement_margin: float = 0.02,
        score_threshold: float = 0.85,
        use_sampling: Optional[bool] = None,
        train_use_sampling: bool = True,
        eval_use_sampling: bool = False,
        sampling_temperature: float = 1.0,
        min_loops: int = 2,
        enable_jepa_crystal: bool = False,
        jepa_crystal_temperature: float = 6.0,
        gain_hidden_dim: int = 32,
        gain_weight: float = 0.35,
        uncertainty_feature_weight: float = 0.0,
        crystal_feature_weight: float = 0.2,
        enable_progress_exit_readout: bool = False,
        progress_gain_weight: float = 0.15,
        progress_trend_weight: float = 0.10,
        progress_plateau_weight: float = 0.10,
        second_order_delta_weight: float = 0.0,
        bias_init: float = 0.0,
        warmup_steps: int = 0,
        ct_drift_weight: float = 0.0,
        know_gap_weight: float = 0.0,
    ):
        super().__init__()
        self.delta_threshold = delta_threshold
        self.self_threshold = self_threshold
        self.rollout_threshold = rollout_threshold
        self.world_threshold = world_threshold
        self.self_check_threshold = self_check_threshold
        self.improvement_margin = improvement_margin
        self.score_threshold = score_threshold
        self.use_sampling = use_sampling
        self.train_use_sampling = train_use_sampling
        self.eval_use_sampling = eval_use_sampling
        self.sampling_temperature = sampling_temperature
        self.min_loops = min_loops
        self.enable_jepa_crystal = enable_jepa_crystal
        self.jepa_crystal_temperature = jepa_crystal_temperature
        self.uncertainty_feature_weight = uncertainty_feature_weight
        self.enable_progress_exit_readout = enable_progress_exit_readout
        self.progress_gain_weight = progress_gain_weight
        self.progress_trend_weight = progress_trend_weight
        self.progress_plateau_weight = progress_plateau_weight
        self.gain_predictor = nn.Sequential(
            nn.Linear(11, gain_hidden_dim),
            nn.SiLU(),
            LumaZCRMSNorm(gain_hidden_dim),
            nn.Linear(gain_hidden_dim, 1),
        )
        self.gain2_predictor = nn.Sequential(
            nn.Linear(11, gain_hidden_dim),
            nn.SiLU(),
            LumaZCRMSNorm(gain_hidden_dim),
            nn.Linear(gain_hidden_dim, 1),
        )
        self.delta_weight = nn.Parameter(torch.tensor(0.5))
        self.self_weight = nn.Parameter(torch.tensor(0.5))
        self.rollout_weight = nn.Parameter(torch.tensor(0.5))
        self.world_weight = nn.Parameter(torch.tensor(0.5))
        self.self_check_weight = nn.Parameter(torch.tensor(0.25))
        self.crystal_weight = nn.Parameter(torch.tensor(crystal_feature_weight))
        self.gain_weight = nn.Parameter(torch.tensor(gain_weight))
        self.bias = nn.Parameter(torch.tensor(bias_init))
        self.warmup_steps = warmup_steps
        self._global_step = 0
        self.second_order_weight = nn.Parameter(torch.tensor(second_order_delta_weight))
        self.ct_drift_weight = nn.Parameter(torch.tensor(ct_drift_weight))
        self.know_gap_weight = nn.Parameter(torch.tensor(know_gap_weight))
        # ES: enhanced exit signals (negative = 高时不退出)
        self.entropy_weight = nn.Parameter(torch.tensor(-0.3))   # 高 entropy → 不确定 → 不退出
        self.confidence_weight = nn.Parameter(torch.tensor(0.2))  # 高 confidence → 退出
        self.token_sensitivity_weight = nn.Parameter(torch.tensor(-0.2))  # 高 sensitivity → 有 token 在变 → 不退出
        self.ct_curvature_weight = nn.Parameter(torch.tensor(-0.2))  # 高 curvature → c_t 还在转向 → 不退出
        self.jepa_surprise_weight = nn.Parameter(torch.tensor(-0.5))  # JEPA 预测不准 → 高 surprise → 不退出
        self.mor_continue_weight = nn.Parameter(torch.tensor(-0.3))  # MoR 多 token 想继续 → 不退出

    def forward(
        self,
        prev_h: Optional[torch.Tensor],
        h: torch.Tensor,
        loop_idx: int,
        self_error: torch.Tensor,
        rollout_error: torch.Tensor,
        world_error: torch.Tensor,
        self_check_score: Optional[torch.Tensor] = None,
        prev_delta_h: Optional[torch.Tensor] = None,
        loop_progress: Optional[torch.Tensor] = None,
        remaining_budget_ratio: Optional[torch.Tensor] = None,
        recent_gain_1: Optional[torch.Tensor] = None,
        recent_gain_2: Optional[torch.Tensor] = None,
        recent_delta_improve_1: Optional[torch.Tensor] = None,
        recent_delta_improve_2: Optional[torch.Tensor] = None,
        reason_local_signal: Optional[torch.Tensor] = None,
        uncertainty_score: Optional[torch.Tensor] = None,
        progress_next_improve: Optional[torch.Tensor] = None,
        progress_trend: Optional[torch.Tensor] = None,
        progress_plateau_logit: Optional[torch.Tensor] = None,
        ct_drift: Optional[torch.Tensor] = None,
        know_gap: Optional[torch.Tensor] = None,
        entropy_proxy: Optional[torch.Tensor] = None,
        confidence_proxy: Optional[torch.Tensor] = None,
        token_sensitivity: Optional[torch.Tensor] = None,
        ct_curvature: Optional[torch.Tensor] = None,
        mor_continue_ratio: Optional[torch.Tensor] = None,
    ) -> dict:
        if prev_h is None:
            delta_h = h.new_tensor(1.0)
        else:
            delta_h = (h - prev_h).norm(dim=-1).mean() / (prev_h.norm(dim=-1).mean() + 1e-8)
        delta_signal = 1.0 - delta_h.clamp(0.0, 1.0)
        # Second-order: when |delta_h - prev_delta_h| is small, delta is stable → converged
        if prev_delta_h is not None:
            second_order_delta = (delta_h - prev_delta_h).abs()
            second_order_signal = 1.0 - second_order_delta.clamp(0.0, 1.0)
        else:
            second_order_signal = h.new_zeros(())
        self_signal = 1.0 - self_error.clamp(0.0, 1.0)
        rollout_signal = 1.0 - rollout_error.clamp(0.0, 1.0)
        world_signal = 1.0 - world_error.clamp(0.0, 1.0)
        if self_check_score is None:
            self_check_score = h.new_tensor(0.5)
        self_check_signal = self_check_score.clamp(0.0, 1.0)
        if self.enable_jepa_crystal:
            jepa_stack = torch.stack([self_signal, rollout_signal, world_signal], dim=0)
            jepa_probs = torch.softmax(jepa_stack * self.jepa_crystal_temperature, dim=0)
            jepa_entropy = -(jepa_probs * torch.log(jepa_probs.clamp_min(1e-8))).sum()
            jepa_entropy = jepa_entropy / math.log(3.0)
            jepa_crystal_signal = 1.0 - jepa_entropy
        else:
            jepa_crystal_signal = h.new_zeros(())
        if loop_progress is None:
            loop_progress = h.new_zeros(())
        if remaining_budget_ratio is None:
            remaining_budget_ratio = h.new_zeros(())
        if recent_gain_1 is None:
            recent_gain_1 = h.new_zeros(())
        if recent_gain_2 is None:
            recent_gain_2 = h.new_zeros(())
        if recent_delta_improve_1 is None:
            recent_delta_improve_1 = h.new_zeros(())
        if recent_delta_improve_2 is None:
            recent_delta_improve_2 = h.new_zeros(())
        if reason_local_signal is None:
            reason_local_signal = h.new_zeros(())
        if uncertainty_score is None:
            uncertainty_score = h.new_zeros(())
        if progress_next_improve is None:
            progress_next_improve = h.new_zeros(())
        if progress_trend is None:
            progress_trend = h.new_zeros(())
        if progress_plateau_logit is None:
            progress_plateau_logit = h.new_zeros(())
        centered_uncertainty = (uncertainty_score.clamp(0.0, 1.0) - 0.5) * 2.0
        progress_next_signal = torch.tanh(progress_next_improve)
        progress_trend_signal = torch.tanh(progress_trend)
        progress_plateau_signal = torch.sigmoid(progress_plateau_logit)
        gain_features = torch.stack(
            [
                delta_signal.float(),
                self_signal.float(),
                rollout_signal.float(),
                world_signal.float(),
                self_check_signal.float(),
                loop_progress.float(),
                remaining_budget_ratio.float(),
                recent_gain_1.float(),
                recent_gain_2.float(),
                0.5 * (recent_delta_improve_1.float() + recent_delta_improve_2.float()),
                reason_local_signal.float(),
            ],
            dim=0,
        ).unsqueeze(0)
        gain_features = gain_features.to(dtype=self.gain_predictor[0].weight.dtype)
        predicted_gain = self.gain_predictor(gain_features).squeeze(0).squeeze(-1)
        predicted_gain2 = self.gain2_predictor(gain_features).squeeze(0).squeeze(-1)
        exit_logit = (
            self.bias
            + self.delta_weight * delta_signal
            + self.self_weight * self_signal
            + self.rollout_weight * rollout_signal
            + self.world_weight * world_signal
            + self.self_check_weight * self_check_signal
            + self.crystal_weight * jepa_crystal_signal
            + self.uncertainty_feature_weight * centered_uncertainty
            + self.second_order_weight * second_order_signal
            - self.gain_weight * predicted_gain
        )
        # c_t drift: c_t 还在变 → 自省没收敛 → 不退出 (负贡献)
        if ct_drift is not None:
            exit_logit = exit_logit - self.ct_drift_weight * ct_drift.clamp(0.0, 2.0)
        # know_gap: removed — 实验证明无效 (2026-04-08)
        # ES: enhanced exit signals
        if entropy_proxy is not None:
            exit_logit = exit_logit + self.entropy_weight * entropy_proxy.clamp(0.0, 1.0)
        if confidence_proxy is not None:
            exit_logit = exit_logit + self.confidence_weight * confidence_proxy.clamp(0.0, 1.0)
        if token_sensitivity is not None:
            exit_logit = exit_logit + self.token_sensitivity_weight * token_sensitivity.clamp(0.0, 10.0)
        if ct_curvature is not None:
            exit_logit = exit_logit + self.ct_curvature_weight * ct_curvature.clamp(-1.0, 1.0)
        # JEPA surprise: self_error 高 → JEPA 预测不准 → 直接抑制退出
        # 这比通过 self_signal 间接影响更强，因为 self_signal 被其他信号稀释
        exit_logit = exit_logit + self.jepa_surprise_weight * self_error.clamp(0.0, 1.0).float()
        # MoR: 多 token 想继续 → 不退出
        if mor_continue_ratio is not None:
            exit_logit = exit_logit + self.mor_continue_weight * mor_continue_ratio.clamp(0.0, 1.0).float()
        if self.enable_progress_exit_readout:
            exit_logit = (
                exit_logit
                - self.progress_gain_weight * progress_next_signal
                - self.progress_trend_weight * progress_trend_signal
                + self.progress_plateau_weight * progress_plateau_signal
            )
        # Keep exit sampling numerically well-behaved under longer training.
        nan_to_num_trigger_count = (~torch.isfinite(exit_logit)).float().sum()
        nan_safe_logit = torch.nan_to_num(exit_logit, nan=0.0, posinf=20.0, neginf=-20.0)
        safe_exit_logit = nan_safe_logit.clamp(-20.0, 20.0)
        exit_score_postfix_clamped_ratio = (safe_exit_logit.ne(nan_safe_logit)).float().mean()
        exit_score = torch.sigmoid(safe_exit_logit)
        sampled_prob_pre = torch.sigmoid((safe_exit_logit / max(self.sampling_temperature, 1e-6)).clamp(-20.0, 20.0))
        bernoulli_invalid_prevented_count = (~torch.isfinite(sampled_prob_pre)).float().sum()
        sampled_exit_score = sampled_prob_pre
        sampled_exit_score = torch.nan_to_num(sampled_exit_score, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        use_sampling = self.use_sampling if self.use_sampling is not None else (self.train_use_sampling if self.training else self.eval_use_sampling)
        if loop_idx + 1 < self.min_loops:
            should_exit = False
        elif self.training and self._global_step < self.warmup_steps:
            should_exit = False
        elif use_sampling:
            should_exit = bool(torch.bernoulli(sampled_exit_score).item() > 0.0)
        else:
            should_exit = bool(exit_score.item() > self.score_threshold)
        return {
            "delta_h": delta_h,
            "second_order_delta": second_order_signal,
            "exit_logit": safe_exit_logit,
            "exit_score": exit_score,
            "sampled_exit_score": sampled_exit_score,
            "two_step_improvement": recent_gain_1.detach(),
            "self_check_score": self_check_score.detach(),
            "jepa_crystal_signal": jepa_crystal_signal.detach(),
            "predicted_gain": predicted_gain.detach(),
            "predicted_gain2": predicted_gain2.detach(),
            "progress_next_signal": progress_next_signal.detach(),
            "progress_trend_signal": progress_trend_signal.detach(),
            "progress_plateau_signal": progress_plateau_signal.detach(),
            "exit_score_preclamp_nonfinite_count": nan_to_num_trigger_count.detach(),
            "exit_score_postfix_clamped_ratio": exit_score_postfix_clamped_ratio.detach(),
            "bernoulli_invalid_prevented_count": bernoulli_invalid_prevented_count.detach(),
            "nan_to_num_trigger_count": nan_to_num_trigger_count.detach(),
            "should_exit": should_exit,
        }


class LumaReasonSharedLayer(nn.Module):
    """A shared reasoning layer is one reusable thought stroke inside Luma's looped inner theatre.
    共享推理层是 Luma 内部循环剧场里可重复使用的一笔思考笔触。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.ct_modulation_mode = config.ct_modulation_mode
        self.hidden_size = config.hidden_size
        self.mamba = ReasonMambaLayer(config)
        # post-norm: 用 LayerNorm（无 affine）而非 RMSNorm
        # RMSNorm 反传 Jacobian = (I - ĥĥᵀ)/‖x‖ 对小输入范数敏感（v11 实测 grad spike 50×）
        # LayerNorm 的 mean centering 让反传更平滑，串联多层不会指数放大梯度
        self.mamba_post_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=False)
        self.diff_attn = GatedDiffAttnFoXSWA(config)
        self.ffn = LumaSwiGLUFFN(config.hidden_size, config.reason_intermediate_size, eps=config.rms_norm_eps)
        self.ffn_post_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=False)
        if self.ct_modulation_mode == "modulewise_gate":
            self.mamba_gate = nn.Linear(config.c_t_dim, 1, bias=False)
            self.attn_gate = nn.Linear(config.c_t_dim, 1, bias=False)
            self.ffn_gate = nn.Linear(config.c_t_dim, 1, bias=False)
        else:
            self.mamba_gate = None
            self.attn_gate = None
            self.ffn_gate = None
        if self.ct_modulation_mode == "film":
            self.mamba_film = nn.Linear(config.c_t_dim, config.hidden_size * 2, bias=False)
            self.attn_film = nn.Linear(config.c_t_dim, config.hidden_size * 2, bias=False)
            self.ffn_film = nn.Linear(config.c_t_dim, config.hidden_size * 2, bias=False)
        else:
            self.mamba_film = None
            self.attn_film = None
            self.ffn_film = None
        # per-layer c_t 注入: 标记是否启用（实际 Embedding 在 ReasonCore 中）
        self._ct_per_layer_inject = config.ct_per_layer_inject
        # RS: Loop LoRA — per-loop low-rank adaptation on FFN down_proj
        self._loop_lora_rank = config.loop_lora_rank
        self._ct_conditioned_lora = config.ct_conditioned_lora
        if self._loop_lora_rank > 0:
            max_loops = config.loop_lora_max_loops
            if self._ct_conditioned_lora:
                # Hypernetwork lite: delta_c_t → 系数 → 组合共享基矩阵
                # c_t 不动→delta=0→LoRA 不激活→自然退出正反馈
                self.lora_coeff_proj = nn.Linear(config.c_t_dim, self._loop_lora_rank, bias=False)
                self.lora_shared_A = nn.Parameter(torch.randn(self._loop_lora_rank, config.hidden_size) * 0.01)
                self.lora_shared_B = nn.Parameter(torch.zeros(config.hidden_size, self._loop_lora_rank))
                self._ct_lora_prev = None  # 缓存上一轮 c_t
                # 保留 loop_idx Embedding 做叠加
                self.lora_A = nn.Embedding(max_loops, config.hidden_size * self._loop_lora_rank)
                self.lora_B = nn.Embedding(max_loops, self._loop_lora_rank * config.hidden_size)
                nn.init.normal_(self.lora_A.weight, std=0.01)
                nn.init.zeros_(self.lora_B.weight)
            else:
                self.lora_A = nn.Embedding(max_loops, config.hidden_size * self._loop_lora_rank)
                self.lora_B = nn.Embedding(max_loops, self._loop_lora_rank * config.hidden_size)
                nn.init.normal_(self.lora_A.weight, std=0.01)
                nn.init.zeros_(self.lora_B.weight)
        # RS: Loop FFN Gating — loop-dependent gate on FFN residual
        self._enable_loop_ffn_gate = config.enable_loop_ffn_gate
        if self._enable_loop_ffn_gate:
            max_loops = getattr(config, "loop_lora_max_loops", 20)
            self.loop_ffn_gate_embed = nn.Embedding(max_loops, config.hidden_size)
            nn.init.zeros_(self.loop_ffn_gate_embed.weight)  # zero-init → starts as pass-through

    def _apply_film(self, h: torch.Tensor, c_t: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        film = proj(c_t).unsqueeze(1)
        scale, shift = film.chunk(2, dim=-1)
        return h * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(shift)

    def forward(
        self,
        h: torch.Tensor,
        c_t: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        loop_idx: int = 0,
        head_partition: bool = False,
        ct_layer_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint as _ckpt
        # loop_lora_delta_ratio: 当前 loop 的 LoRA 扰动相对 FFN residual 的比例。
        self._last_lora_delta_ratio = 0.0
        self._last_lora_delta_norm = 0.0
        self._last_ffn_residual_norm = 0.0
        # per-layer c_t 注入: 外部预计算的 bias 直接加
        if ct_layer_bias is not None:
            h = h + ct_layer_bias
        if c_t is not None and self.ct_modulation_mode == "film" and self.mamba_film is not None:
            h = self._apply_film(h, c_t, self.mamba_film)
        mamba_out = self.mamba(h, c_t=c_t)
        if c_t is not None and self.ct_modulation_mode == "modulewise_gate" and self.mamba_gate is not None:
            mamba_gate = torch.sigmoid(self.mamba_gate(c_t)).unsqueeze(1)
            h = h + mamba_gate * (mamba_out - h)
        else:
            h = mamba_out
        # post-norm: 切断 Mamba3 加法残差累积
        h = self.mamba_post_norm(h)
        if c_t is not None and self.ct_modulation_mode == "film" and self.attn_film is not None:
            h = self._apply_film(h, c_t, self.attn_film)
        # diff_attn: standard PyTorch ops, safe to checkpoint
        if use_gradient_checkpointing and self.training:
            _attn_bias = attn_bias
            _c_t = c_t
            attn_out = _ckpt(lambda x: self.diff_attn(x, attn_bias=_attn_bias, c_t=_c_t), h, use_reentrant=False)
        else:
            attn_out = self.diff_attn(h, attn_bias=attn_bias, c_t=c_t)
        # Phase A: head partition — only the active head group contributes residual
        if head_partition:
            num_heads = self.diff_attn.num_heads
            head_dim = self.diff_attn.head_dim
            num_groups = max(2, min(4, num_heads))  # 2-4 groups
            group_size = num_heads // num_groups
            active_group = loop_idx % num_groups
            # Mask: zero out residual for inactive head groups
            bsz, seq_len, _ = attn_out.shape
            residual_delta = attn_out - h
            delta_heads = residual_delta.view(bsz, seq_len, num_heads, head_dim)
            mask = h.new_zeros(num_heads)
            start_h = active_group * group_size
            end_h = min(start_h + group_size, num_heads)
            mask[start_h:end_h] = 1.0
            delta_heads = delta_heads * mask.view(1, 1, num_heads, 1)
            attn_out = h + delta_heads.view(bsz, seq_len, -1)
        if c_t is not None and self.ct_modulation_mode == "modulewise_gate" and self.attn_gate is not None:
            attn_gate = torch.sigmoid(self.attn_gate(c_t)).unsqueeze(1)
            h = h + attn_gate * (attn_out - h)
        else:
            h = attn_out
        if c_t is not None and self.ct_modulation_mode == "film" and self.ffn_film is not None:
            h = self._apply_film(h, c_t, self.ffn_film)
        # FFN: standard PyTorch ops, safe to checkpoint
        if use_gradient_checkpointing and self.training:
            ffn_out = _ckpt(self.ffn, h, use_reentrant=False)
        else:
            ffn_out = self.ffn(h)
        _ffn_base = ffn_out
        # RS: Loop LoRA — add per-loop low-rank delta to FFN output
        if self._loop_lora_rank > 0:
            _li = min(loop_idx, self.lora_A.num_embeddings - 1)
            _lidx = h.new_tensor([_li], dtype=torch.long)
            _a = self.lora_A(_lidx).view(self._loop_lora_rank, self.hidden_size)  # [r, D]
            _b = self.lora_B(_lidx).view(self.hidden_size, self._loop_lora_rank)  # [D, r]
            if self._ct_conditioned_lora and c_t is not None:
                # Hypernetwork lite: delta_c_t 生成系数 — c_t 不变时 LoRA 不激活
                if self._ct_lora_prev is None:
                    self._ct_lora_prev = c_t.detach()
                _delta_ct = c_t - self._ct_lora_prev
                self._ct_lora_prev = c_t.detach()
                _coeff = torch.tanh(self.lora_coeff_proj(_delta_ct))  # [B, rank] — 不需要×0.1，delta 本身就小
                _ct_a = _coeff.unsqueeze(-1) * self.lora_shared_A.unsqueeze(0)  # [B, rank, D]
                _ct_b = self.lora_shared_B.unsqueeze(0) * _coeff.unsqueeze(1)   # [B, D, rank]
                # 叠加: loop_idx 基础 + c_t 条件调制
                _a_full = _a.unsqueeze(0) + _ct_a  # [B, rank, D]
                _b_full = _b.unsqueeze(0) + _ct_b  # [B, D, rank]
                _residual = ffn_out - h  # [B, T, D]
                ffn_out = ffn_out + torch.bmm(torch.bmm(_residual, _b_full), _a_full)
            else:
                ffn_out = ffn_out + ((ffn_out - h) @ _b) @ _a
        if self._loop_lora_rank > 0:
            _lora_delta = (ffn_out - _ffn_base).float()
            _ffn_residual = (_ffn_base - h).float()
            _lora_norm = _lora_delta.norm(dim=-1).mean()
            _resid_norm = _ffn_residual.norm(dim=-1).mean().clamp(min=1e-8)
            self._last_lora_delta_norm = float(_lora_norm.item())
            self._last_ffn_residual_norm = float(_resid_norm.item())
            self._last_lora_delta_ratio = float((_lora_norm / _resid_norm).item())
        # RS: Loop FFN Gating — loop-dependent sigmoid gate on FFN residual
        if self._enable_loop_ffn_gate:
            _li = min(loop_idx, self.loop_ffn_gate_embed.num_embeddings - 1)
            _lidx = h.new_tensor([_li], dtype=torch.long)
            _gate = torch.sigmoid(self.loop_ffn_gate_embed(_lidx))  # [1, D]
            ffn_out = h + (ffn_out - h) * _gate.unsqueeze(0)  # gate the FFN residual per-dim
        if c_t is not None and self.ct_modulation_mode == "modulewise_gate" and self.ffn_gate is not None:
            ffn_gate = torch.sigmoid(self.ffn_gate(c_t)).unsqueeze(1)
            h = h + ffn_gate * (ffn_out - h)
        else:
            h = ffn_out
        # post-norm: 切断 FFN + LoRA 加法残差累积
        h = self.ffn_post_norm(h)
        return h


class EnergyReasonCore(nn.Module):
    """Phase E: 能量梯度流推理核心（Step 1 最简骨架）。

    理论文档: docs/reports/Luma_PhaseE_Theory_Seed_20260412.md

    核心公式:
        h_{k+1} = h_k - η · ∇_h E(h_k; c_t) + √(2ηT) · ξ_k

    能量参数化（方案 A — 见种子文档 §3.2）:
        E(h; c_t) = 0.5 · ||h - body(h, c_t)||²
    其中 body 是 shared_layers 的完整前向传播（复用现有 Mamba+Attention 混合结构）。

    这个最简版本保留了所有现有 SSM/Attention 内部结构，只是在外面换了一个
    "能量梯度下降" 的迭代规则，替代原来的 "h + Δ" 残差更新。

    Step 1 有意省略（后续 step 逐个开启）:
        - Langevin 噪声 (T=0)
        - c_t 时间尺度解耦 (仍用入口的 c_t，不在 loop 轴演化)
        - gradient-norm early stop (总是跑满 K_max)
        - MHC 多流 / introspection / exit_ctrl / token_depth_routing
        - LoRA per-loop 依赖 (loop_idx 固定为 0，避免能量族变化)
        - probe 重写（下一步 Step 1.5 做）

    目的只有一个: 验证能量梯度步的 forward + backward 能跑通，不 OOM，loss 能下降。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.config = config
        self.K_max = int(getattr(config, "phase_e_K_max", 5))
        self.eta = float(getattr(config, "phase_e_eta", 0.1))
        self.temperature = float(getattr(config, "phase_e_temperature", 0.0))
        self.grad_stop_eps = float(getattr(config, "phase_e_grad_stop_eps", 0.0))
        self.k_backprop = int(getattr(config, "phase_e_k_backprop", 0))  # 0=full graph
        self.custom_checkpoint = bool(getattr(config, "phase_e_custom_checkpoint", False))

        self.ct_injection = CTInjection(
            c_t_dim=config.c_t_dim,
            hidden_size=config.hidden_size,
            mode=config.ct_injection_mode,
            scale=config.ct_inject_scale,
        )
        # Shared layers: 复用现有 LumaReasonSharedLayer 作为 body 的内部结构
        # 这一层里包含 Mamba + Attention + FFN（SSM/Attn 混合保留）
        self.shared_layers = nn.ModuleList(
            [LumaReasonSharedLayer(config) for _ in range(config.reason_shared_depth)]
        )
        self.head_partition = config.reason_head_partition
        # 诊断：记录最后一次 forward 的能量轨迹和梯度范数轨迹
        self._last_energy_trace: List[float] = []
        self._last_grad_norm_trace: List[float] = []

    def _body(self, h: torch.Tensor, c_t: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        """F(h, c_t) — shared_layers 的完整前向传播作为 body 函数。

        和 LumaReasonCore._run_shared_stack 的区别：
        - ct_injection 在这里显式应用（probe 版直接把 h_inj 传进来）
        - Step 1 固定 loop_idx=0，避免 LoRA per-loop / time_conditioning 破坏能量族一致性
        """
        # 注入 c_t bias（用 clamped 版本作为 Step 1 安全网）
        c_bias = self.ct_injection.get_bias(c_t).unsqueeze(1).to(h.dtype)  # [B, 1, D]
        h_inj = h + c_bias
        # 裸跑 shared_layers
        h_cur = h_inj
        for layer in self.shared_layers:
            h_cur = layer(
                h_cur,
                c_t=c_t,
                attn_bias=None,
                use_gradient_checkpointing=False,
                loop_idx=loop_idx,
                head_partition=self.head_partition,
            )
        return h_cur

    def _compute_energy(self, h: torch.Tensor, c_t: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        """E(h; c_t) = 0.5 · ||h - body(h, c_t)||² (方案 A)。

        返回标量（在 batch 和序列维度上 sum）。梯度 ∇_h E 用 torch.autograd.grad 计算。
        """
        h_body = self._body(h, c_t, loop_idx=loop_idx)
        diff = h - h_body
        # sum 而不是 mean：保证梯度幅度和序列长度无关
        # 具体：∇_h (0.5 ||h - body(h)||²) = (h - body) · (I - ∂body/∂h)
        # 对每个位置独立贡献，方便后续 Hessian 分析
        E = 0.5 * (diff.float() ** 2).sum()
        return E

    def forward(
        self,
        h_init: torch.Tensor,
        c_t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        extra_energy_fn: Optional[callable] = None,
        extra_energy_params: Optional[list] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """能量梯度流 forward。

        Args:
            h_init: [B, T, D] 从 compression 出来的初始隐状态
            c_t: [B, c_t_dim] 慢变量（Step 1 阶段仍由调用方传入，不在 loop 轴演化）
            attention_mask: 保留接口（当前 body 内部不使用）
            extra_energy_fn: 可选的额外能量项 E_extra(h, c_t) -> scalar
                Step 7 用它把 world_jepa 吸入总能量: E_total = E_body + E_world
                这样 ∇_h E 同时包含自洽梯度和世界预测梯度，两条保守场联合塑形

        Returns:
            h_final: [B, T, D] 经过 K 步能量梯度下降后的隐状态
            aux: dict 诊断信息
                - energy_trace: [K] 每步的能量值（主能量 E_body）
                - energy_extra_trace: [K] 额外能量项（Step 7+ 才非零）
                - grad_norm_trace: [K] 每步的 ||∇_h E||
                - K_used: 实际跑了多少步（Step 4 早停启用后会 < K_max）
        """
        h = h_init
        # 生产场景 h 是 compression 出来的 intermediate tensor（自动有 grad_fn），
        # 直接兼容；独立 smoke test 场景 h 是 leaf 不带 grad，显式开启。
        if not h.requires_grad:
            h = h.detach().requires_grad_(True)
        energy_trace: List[float] = []
        energy_extra_trace: List[float] = []
        grad_norm_trace: List[float] = []
        K_used = 0

        # Truncated backprop: 只对最后 k_backprop 步建完整 create_graph 图
        # 前 K_max - k_backprop 步用 no_grad + detach（无外层梯度）
        # 显存 = k_backprop / K_max × 全图内存，关键解锁 seq=2048
        # k_backprop=0 表示 full graph (原始行为)
        if self.k_backprop > 0 and self.training:
            n_detached = max(0, self.K_max - self.k_backprop)
        else:
            n_detached = 0

        for k in range(self.K_max):
            is_detached_phase = k < n_detached
            create_graph = self.training and not is_detached_phase
            if is_detached_phase:
                # detached 分支：h 作为 leaf，grad_h 算完即 detach
                # 这一段前向 + autograd.grad 的图在每步末尾立即释放
                with torch.enable_grad():
                    h_leaf = h.detach().requires_grad_(True)
                    E = self._compute_energy(h_leaf, c_t, loop_idx=0)
                    E_main_detached = float(E.detach().item())
                    if extra_energy_fn is not None:
                        E_extra = extra_energy_fn(h_leaf, c_t)
                        E = E + E_extra
                        energy_extra_trace.append(float(E_extra.detach().item()))
                    else:
                        energy_extra_trace.append(0.0)
                    grad_h, = torch.autograd.grad(E, h_leaf, create_graph=False, retain_graph=False)
                    grad_h_detached = grad_h.detach()
                # 应用更新（h 保持 detach 状态，无 autograd 历史）
                h = (h.detach() - self.eta * grad_h_detached)
                if self.temperature > 0.0 and self.training:
                    noise = torch.randn_like(h) * math.sqrt(2.0 * self.eta * self.temperature)
                    h = h + noise
                # 为下一步做准备：最后一个 detach 步之后如果要进 grad 阶段，需要重开 grad
                if k == n_detached - 1 and self.k_backprop > 0:
                    h = h.detach().requires_grad_(True)
                energy_trace.append(E_main_detached)
                grad_norm_trace.append(float(grad_h_detached.norm().item()))
            elif self.custom_checkpoint and self.training:
                # Custom in-loop checkpoint: _PhaseEStepFunction re-computes body during backward
                # 所有 body 参数必须作为 tensor 传入以便 autograd 正确 routing 梯度
                _body_params = list(self.parameters())
                if extra_energy_params:
                    _body_params = _body_params + list(extra_energy_params)
                h_prev = h if h.requires_grad else h.detach().requires_grad_(True)
                h = _PhaseEStepFunction.apply(
                    h_prev,
                    c_t,
                    self._body,
                    extra_energy_fn,
                    self.eta,
                    len(_body_params),
                    *_body_params,
                )
                if self.temperature > 0.0 and self.training:
                    noise = torch.randn_like(h) * math.sqrt(2.0 * self.eta * self.temperature)
                    h = h + noise
                # 诊断记录：custom ckpt 版本的 trace 来自 ctx，但 apply 不返回 ctx
                # 用当前 h 和 c_t 做一次无梯度 probe 记录能量值
                with torch.no_grad():
                    _E_probe = self._compute_energy(h_prev.detach(), c_t, loop_idx=0)
                    energy_trace.append(float(_E_probe.item()))
                    energy_extra_trace.append(0.0)
                    grad_norm_trace.append(float('nan'))
            else:
                # 完整 create_graph 分支（原始 Phase E 能量梯度步，无 checkpoint）
                E = self._compute_energy(h, c_t, loop_idx=0)
                E_main_detached = float(E.detach().item())
                if extra_energy_fn is not None:
                    E_extra = extra_energy_fn(h, c_t)
                    E = E + E_extra
                    energy_extra_trace.append(float(E_extra.detach().item()))
                else:
                    energy_extra_trace.append(0.0)
                grad_h, = torch.autograd.grad(
                    E, h, create_graph=create_graph, retain_graph=create_graph
                )
                h = h - self.eta * grad_h
                if self.temperature > 0.0 and self.training:
                    noise = torch.randn_like(h) * math.sqrt(2.0 * self.eta * self.temperature)
                    h = h + noise
                energy_trace.append(E_main_detached)
                grad_norm_trace.append(float(grad_h.detach().norm().item()))
            K_used += 1

            # gradient-norm early stopping (Step 4 启用)
            if self.grad_stop_eps > 0.0 and grad_norm_trace[-1] < self.grad_stop_eps:
                break

        # 保存最后一次 trace 供外部 probe / debug 读取
        self._last_energy_trace = energy_trace
        self._last_grad_norm_trace = grad_norm_trace

        aux = {
            "phase_e_energy_trace": energy_trace,
            "phase_e_energy_extra_trace": energy_extra_trace,
            "phase_e_grad_norm_trace": grad_norm_trace,
            "phase_e_K_used": K_used,
            "phase_e_K_max": self.K_max,
        }
        return h, aux

    @torch.no_grad()
    def measure_phase_e_probes(
        self,
        h: torch.Tensor,
        c_t: torch.Tensor,
        rel_eps: float = 0.05,
        n_hutchinson: int = 3,
    ) -> dict:
        """Phase E Step 1.5: 重写的理论 probe，测 *完整* F_k = h - η∇E 的 Jacobian + Hessian 谱。

        与旧版 `LumaReasonCore.measure_theory_probes` 的关键区别:
        - 旧 probe 测 `shared_layers` 子集的 Jacobian ∂(layer^L(h))/∂h
        - 新 probe 测完整 F_k = h - η · ∇_h E(h) 的 Jacobian，即实际驱动 h 演化的算子
        - 新增 Hessian trace 估计（Hutchinson trick），直接告诉我们构造性收缩是否成立

        返回指标:
        - rho_h_full: 完整 F_k 的局部 Jacobian 谱半径估计（扰动 h，测 ||ΔF||/||Δh||）
          希望: < 1.0 表示构造性收缩
        - hessian_trace_est: Hutchinson 估计的 trace(∇²E)，∇²E 半正定 ⟺ trace > 0
          希望: > 0 且不爆炸
        - grad_norm_at_h: ||∇_h E(h)||，当前 h 到能量极小值的"距离"
          希望: 训练后 → 0（接近极小值）
        - energy_at_h: E(h)，当前能量值
        - lambda_max_upper: η·λ_max 的上界估计（用 power iteration 的一次近似）
          希望: < 2 保证稳定

        这些量直接对应 theory seed doc §3.6 的 Hessian 谱约束。

        Args:
            h: [B, T, D] 当前隐状态
            c_t: [B, c_t_dim] 慢变量
            rel_eps: 扰动幅度（相对于 ||h||）
            n_hutchinson: Hutchinson 估计平均次数（trace 无偏估计 = E[v^T H v], v~N(0,I)）
        """
        probes = {
            "rho_h_full": None,
            "hessian_trace_est": None,
            "grad_norm_at_h": None,
            "energy_at_h": None,
            "lambda_max_upper": None,
            "probe_delta_h_norm": None,
        }

        try:
            # 需要临时打开 grad，因为 probe 要算 autograd.grad
            with torch.enable_grad():
                # 确保 h 有 requires_grad（probe 外部 h 可能已 detach）
                h_req = h.detach().requires_grad_(True)
                c_t_req = c_t.detach()

                # 1. E(h) + ||∇E(h)||
                E = self._compute_energy(h_req, c_t_req, loop_idx=0)
                grad_h, = torch.autograd.grad(E, h_req, create_graph=True, retain_graph=True)
                probes["energy_at_h"] = float(E.detach().item())
                probes["grad_norm_at_h"] = float(grad_h.detach().norm().item())

                # 2. Hutchinson trace of ∇²E: E[v^T H v] = E[v^T ∂(∇E·v)/∂h · v] 不用展开 H
                #    更简单的: trace(H) ≈ (1/N) Σ_i v_i^T H v_i
                #    H v 通过 Hessian-vector product 计算: ∂(∇E · v)/∂h
                hessian_trace_samples = []
                for _ in range(n_hutchinson):
                    v = torch.randn_like(h_req)
                    v_norm = v.norm().clamp(min=1e-8)
                    v = v / v_norm  # 单位向量
                    # HVP: ∂(∇E · v)/∂h = H v
                    grad_dot_v = (grad_h * v).sum()
                    Hv, = torch.autograd.grad(grad_dot_v, h_req, retain_graph=True)
                    # v^T H v = 单次 Hutchinson 样本
                    sample = (v * Hv).sum() * h_req.numel()  # scale 回去补单位化
                    hessian_trace_samples.append(float(sample.detach().item()))
                probes["hessian_trace_est"] = sum(hessian_trace_samples) / len(hessian_trace_samples)

                # 3. ρ_h_full: 完整 F_k 的 Jacobian 谱半径 via 扰动
                #    F(h) = h - η · ∇E(h)
                #    ||F(h+δ) - F(h)|| / ||δ|| = 局部 ρ 估计
                h_norm = h_req.detach().float().norm().clamp(min=1e-6).item()
                delta_target = rel_eps * h_norm
                d = torch.randn_like(h_req)
                d_norm = d.norm().clamp(min=1e-8)
                d = (d / d_norm) * delta_target
                probes["probe_delta_h_norm"] = float(delta_target)

                # F(h) = h - eta * grad_h (已经算过 grad_h)
                F_h = h_req - self.eta * grad_h

                # F(h + d): 需要重算 ∇E at (h+d)
                h_pert = (h_req + d).detach().requires_grad_(True)
                E_pert = self._compute_energy(h_pert, c_t_req, loop_idx=0)
                grad_h_pert, = torch.autograd.grad(E_pert, h_pert, create_graph=False)
                F_h_pert = h_pert - self.eta * grad_h_pert

                diff_F = (F_h_pert - F_h).float().detach()
                probes["rho_h_full"] = float(diff_F.norm().item() / max(delta_target, 1e-12))

                # 4. λ_max 上界估计: 通过 HVP 的范数单次 power iteration 近似
                #    ||H v|| / ||v|| ≤ λ_max，用随机 v 的 HVP 范数作为下界估计
                #    （严格 power iteration 要迭代，这里取 3 次 HVP 的最大值作为代理）
                lambda_candidates = []
                for _ in range(n_hutchinson):
                    v = torch.randn_like(h_req)
                    v = v / v.norm().clamp(min=1e-8)
                    grad_dot_v = (grad_h * v).sum()
                    Hv, = torch.autograd.grad(grad_dot_v, h_req, retain_graph=True)
                    lambda_candidates.append(float(Hv.norm().item()))
                lambda_max_proxy = max(lambda_candidates)
                probes["lambda_max_upper"] = float(self.eta * lambda_max_proxy)

        except Exception as e:
            probes["error"] = str(e)

        return probes


class LumaReasonCore(nn.Module):
    """One shared reasoning block can now contain several true layers, while the whole block is still reused across loops.
    现在单个共享推理 block 内部可以包含多层真实深度，但整个 block 仍会在循环间重复使用。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.ct_modulation_mode = config.ct_modulation_mode
        self.dynamics_experiment = config.dynamics_experiment.lower()
        self.routing_chunk_size = max(4, int(config.routing_chunk_size))
        self.routing_topk_blocks = max(1, int(config.routing_topk_blocks))
        self.routing_topk_tokens = max(1, int(config.routing_topk_tokens))
        self.routing_top_p_coarse = float(config.routing_top_p_coarse)
        self.routing_top_p_fine = float(config.routing_top_p_fine)
        self.routing_budget_min = float(config.routing_budget_min)
        self.routing_budget_max = float(config.routing_budget_max)
        self.routing_weak_gain = float(config.routing_weak_gain)
        self.routing_strong_gain = float(config.routing_strong_gain)
        self.routing_local_floor = float(config.routing_local_floor)
        self.routing_modulation_floor = float(config.routing_modulation_floor)
        self.routing_modulation_ceiling = float(config.routing_modulation_ceiling)
        self.routing_world_summary_cap = float(config.routing_world_summary_cap)
        # Copy new residual/selection and alive-floor knobs from config
        self.routing_use_residual_branch = bool(getattr(config, "routing_use_residual_branch", False))
        self.ct_residual_gate_scale = float(getattr(config, "ct_residual_gate_scale", 0.15))
        self.ct_selection_only = bool(getattr(config, "ct_selection_only", False))
        self.ct_selection_amplitude = float(getattr(config, "ct_selection_amplitude", 0.08))
        self.routing_local_delta_floor = float(getattr(config, "routing_local_delta_floor", 0.0))
        self.routing_local_delta_floor_weight = float(getattr(config, "routing_local_delta_floor_weight", 0.0))
        self.rollout_alive_weight = float(getattr(config, "rollout_alive_weight", 0.0))
        self.ct_grad_scale = float(getattr(config, "ct_grad_scale", 1.0))
        self.routing_tier_soft_only = bool(config.routing_tier_soft_only)
        self.routing_tier_entropy_floor = float(config.routing_tier_entropy_floor)
        self.routing_min_local_share = float(config.routing_min_local_share)
        self.routing_progress_weight = float(config.routing_progress_weight)
        self.freeze_ct_during_reason = bool(getattr(config, "freeze_ct_during_reason", False))
        self.ct_injection = CTInjection(config.c_t_dim, config.hidden_size, mode=config.ct_injection_mode, scale=config.ct_inject_scale)
        self.r_injection = nn.Linear(config.r_t_dim, config.hidden_size, bias=False) if config.enable_reasoning_state_ring else None
        self.ct_lowrank_down = nn.Linear(config.c_t_dim, config.ct_lowrank_rank, bias=False) if config.ct_modulation_mode == "lowrank_hyperbias" else None
        self.ct_lowrank_up = nn.Linear(config.ct_lowrank_rank, config.hidden_size, bias=False) if config.ct_modulation_mode == "lowrank_hyperbias" else None
        self.token_selective_gate = nn.Linear(config.hidden_size + config.c_t_dim, 1, bias=False) if config.ct_modulation_mode == "token_selective" else None
        self.chunk_feature_dim = config.hidden_size * 3 + config.c_t_dim + 3
        self.block_feature_dim = self.chunk_feature_dim
        self.token_feature_dim = config.hidden_size + config.c_t_dim + 3
        self.chunk_gate_head = nn.Linear(self.chunk_feature_dim, 1, bias=False)
        self.chunk_film_head = nn.Linear(self.chunk_feature_dim, config.hidden_size * 2, bias=False)
        self.token_score_head = nn.Linear(self.token_feature_dim, 1, bias=False)
        self.local_delta_head = nn.Linear(config.hidden_size + config.c_t_dim, config.hidden_size, bias=False)
        self.focus_query_head = nn.Linear(config.c_t_dim + 3, config.hidden_size, bias=False)
        self.block_score_head = nn.Linear(self.block_feature_dim, 1, bias=False)
        self.budget_head = nn.Linear(config.c_t_dim + 3, 1, bias=False)
        self.memory_tier_head = nn.Linear(config.c_t_dim + 3, 4, bias=False)
        self.memory_source_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.router_window = config.r_t_router_window
        self.route_query = nn.Linear(config.c_t_dim + config.r_t_dim, config.hidden_size, bias=False) if config.enable_reasoning_state_ring else None
        self.route_key = nn.Linear(config.hidden_size, config.hidden_size, bias=False) if config.enable_reasoning_state_ring else None
        self.route_value = nn.Linear(config.hidden_size, config.hidden_size, bias=False) if config.enable_reasoning_state_ring else None
        self.reason_switch_gate = (
            nn.Linear(config.hidden_size + config.c_t_dim + config.r_t_dim + 1, 1, bias=False)
            if config.enable_reasoning_state_ring
            else None
        )
        self.shared_layers = nn.ModuleList(
            [LumaReasonSharedLayer(config) for _ in range(config.reason_shared_depth)]
        )
        # per-layer c_t 注入: 共享 proj(c_t) detach 后 × 4 个可学习方向向量
        # 显存 = G0（无额外梯度路径），方向多样性来自 element-wise 调制
        self._ct_per_layer_inject = config.ct_per_layer_inject
        if self._ct_per_layer_inject:
            _n_layers = config.reason_shared_depth
            # 4 个方向向量，正交初始化确保初始方向多样性
            self.ct_layer_directions = nn.Parameter(torch.randn(_n_layers, config.hidden_size) * 0.02)
        self._identity_alpha = float(config.identity_recurrence_alpha)
        # LoopFormer-style time conditioning: inject normalized t and dt
        self._enable_time_cond = bool(config.enable_time_conditioning)
        if self._enable_time_cond:
            self.time_proj = nn.Linear(2, config.hidden_size, bias=False)
            nn.init.zeros_(self.time_proj.weight)
        # ── Reasoning Partitioning ──────────────────────────────
        # Phase C: learned per-phase embedding injected before shared layers
        self.num_phases = config.reason_num_phases
        if self.num_phases > 0:
            self.phase_embed = nn.Embedding(self.num_phases, config.hidden_size)
            nn.init.normal_(self.phase_embed.weight, std=0.02)
        # Phase A: head partition — each loop uses a rotating subset of attention heads
        self.head_partition = config.reason_head_partition
        # Phase B: MoR — loop-conditioned expert routing
        self.mor_routing = config.reason_mor_routing
        if self.mor_routing:
            mor_experts = config.reason_mor_num_experts
            self.mor_topk = config.reason_mor_topk
            # Lightweight LoRA-style experts: down-project then up-project
            expert_rank = max(32, config.hidden_size // 8)
            self.mor_experts_down = nn.ModuleList([
                nn.Linear(config.hidden_size, expert_rank, bias=False) for _ in range(mor_experts)
            ])
            self.mor_experts_up = nn.ModuleList([
                nn.Linear(expert_rank, config.hidden_size, bias=False) for _ in range(mor_experts)
            ])
            # Router: loop_idx embedding → expert scores
            self.mor_loop_embed = nn.Embedding(64, config.hidden_size)  # up to 64 loops
            self.mor_router = nn.Linear(config.hidden_size, mor_experts, bias=False)
            self.mor_gate_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for expert_up in self.mor_experts_up:
                nn.init.zeros_(expert_up.weight)
        # ── Phase E 主集成 ──────────────────────────────
        # 把 shared_layers stack 从 "一次 forward" 换成 "K 步能量梯度下降"
        # E(h) = 0.5 ||h - body(h, c_t)||²，h_{k+1} = h - η∇_h E
        # 复用 shared_layers 作为 body，不新建模块；c_t 注入路径保持和原 stack 一致
        self._enable_phase_e_main = bool(getattr(config, "enable_energy_reason_core", False))
        self._phase_e_K_max = int(getattr(config, "phase_e_K_max", 3))
        self._phase_e_eta = float(getattr(config, "phase_e_eta", 0.1))
        self._phase_e_k_backprop = int(getattr(config, "phase_e_k_backprop", 1))
        self._phase_e_temperature = float(getattr(config, "phase_e_temperature", 0.0))
        self._phase_e_grad_stop_eps = float(getattr(config, "phase_e_grad_stop_eps", 0.0))
        # Damped 模式（推荐）：h ← (1-η)·h + η·F(h)，不用 autograd.grad / 无 double backward
        # 理论：Phase E 一阶近似，∇_h E ≈ (h - F(h)) 当 J_F 小时，
        # 所以 h - η·∇E ≈ (1-η)h + η·F(h)。保留不动点 h=F(h) 与构造性收缩，
        # 但免除 bf16 + create_graph + Mamba kernel 的数值地雷区
        self._phase_e_damped_mode = bool(getattr(config, "phase_e_damped_mode", True))
        # Phase E body 残差归一化设计:
        # F(h) = h + α · LayerNorm(g(h) - h)
        # 这让 F 物理上是 "near-identity + bounded perturbation" 的收缩映射:
        # - LayerNorm 把 g(h)-h 的范数固定到 √D
        # - α (learnable scalar, 初始 0.1) 控制扰动幅度
        # - Lipschitz 上界 = 1 + α，当 α<<1 时严格收缩 (Phase E damped 稳定)
        # - 不是硬约束: α 可学习，模型可以让 F 接近 identity 也可以适度偏离
        self._body_out_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=False)
        # alpha: 残差扰动强度的标量门，初始 0.1 (Phase E body 接近 identity)
        self._body_residual_alpha = nn.Parameter(torch.tensor(0.1))

        # ── Stellarator mode (v19+) ──────────────────────────────
        # 设计原则（4.14 用户指示）:
        #   托卡马克 → 仿星器。主循环定义为结构性收缩映射 + 慢变量只调地貌。
        #
        # 三层结构:
        #   主干 F_main(h)        不看 c_t，由 shared_layers 单次 forward
        #   调制 low-rank bias    W_up(silu(W_down(c_t)))，rank=8，有界
        #   融合 h + g·(F_main - h + bias)  g=sigmoid(scalar)，全局有界
        self._stellarator_mode = bool(getattr(config, "stellarator_mode", False))
        if self._stellarator_mode:
            _ct_dim = config.c_t_dim
            _hid = config.hidden_size
            _mod_rank = int(getattr(config, "stellarator_mod_rank", 8))
            # 低秩调制: c_t → rank → hidden，zero-init up 让初始 bias=0
            self._stellarator_mod_down = nn.Linear(_ct_dim, _mod_rank, bias=False)
            self._stellarator_mod_up = nn.Linear(_mod_rank, _hid, bias=False)
            nn.init.zeros_(self._stellarator_mod_up.weight)
            # 融合门 scalar: sigmoid(logit_init) = 0.5
            self._stellarator_gate_logit = nn.Parameter(torch.tensor(0.0))
            # 诊断
            self._last_stellarator_gate = 0.0
            self._last_stellarator_bias_ratio = 0.0

        # 诊断 trace（供外部读取）
        self._last_phase_e_energy_trace: List[float] = []
        self._last_phase_e_grad_norm_trace: List[float] = []
        self._last_phase_e_K_used: int = 0

    def _chunk_spans(self, seq_len: int) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for start in range(0, seq_len, self.routing_chunk_size):
            end = min(seq_len, start + self.routing_chunk_size)
            spans.append((start, end))
        return spans

    def _chunk_summaries(self, h: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        spans = self._chunk_spans(h.shape[1])
        summaries = [h[:, start:end, :].mean(dim=1) for start, end in spans]
        return torch.stack(summaries, dim=1), spans

    def _expand_chunk_values(
        self,
        chunk_values: torch.Tensor,
        spans: List[Tuple[int, int]],
        seq_len: int,
    ) -> torch.Tensor:
        bsz, _, dim = chunk_values.shape
        expanded = chunk_values.new_zeros((bsz, seq_len, dim))
        for idx, (start, end) in enumerate(spans):
            expanded[:, start:end, :] = chunk_values[:, idx : idx + 1, :]
        return expanded

    def _topk_mask(self, scores: torch.Tensor, topk: int) -> torch.Tensor:
        topk = max(1, min(topk, scores.shape[1]))
        idx = torch.topk(scores, k=topk, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    def _topp_mask(self, scores: torch.Tensor, top_p: float, cap_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(scores.float(), dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        keep[:, :1] = True
        if cap_k > 0:
            cap_k = max(1, min(cap_k, keep.shape[1]))
            keep[:, cap_k:] = False
        mask = torch.zeros_like(keep, dtype=torch.bool)
        mask.scatter_(1, sorted_idx, keep)
        selected_mass = (probs * mask.float()).sum(dim=-1)
        return mask, selected_mass

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)

    def _ensure_feature_dim(self, feature: torch.Tensor, expected_dim: int, name: str) -> None:
        """Luma keeps feature-template contracts explicit so routing heads cannot silently desync from concat order.
        Luma 把特征模板契约写死检查，避免拼接顺序变化后线性层输入维度悄悄错位。
        """

        actual_dim = int(feature.shape[-1])
        if actual_dim != expected_dim:
            raise RuntimeError(
                f"{name} feature dim mismatch: expected {expected_dim}, got {actual_dim}. "
                "Check concat template/order for c_t, progress_state, block_repr, and context summaries."
            )

    def _context_scalars(self, context: Optional[dict], batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if context is None:
            return torch.zeros((batch_size, 3), device=device, dtype=dtype)
        def _to_column(name: str) -> torch.Tensor:
            value = context.get(name, None)
            if value is None:
                return torch.zeros((batch_size, 1), device=device, dtype=dtype)
            tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, device=device, dtype=dtype)
            if tensor.dim() == 0:
                tensor = tensor.expand(batch_size).to(dtype=dtype)
            if tensor.dim() == 1:
                tensor = tensor[:, None]
            return tensor.to(device=device, dtype=dtype)
        next_col = _to_column("progress_next")
        trend_col = _to_column("progress_trend")
        plateau_col = _to_column("progress_plateau")
        return torch.cat([next_col, trend_col, plateau_col], dim=-1)

    def _context_hidden(
        self,
        context: Optional[dict],
        key: str,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if context is None or context.get(key, None) is None:
            return torch.zeros((batch_size, hidden_size), device=device, dtype=dtype)
        value = context[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, device=device, dtype=dtype)
        if tensor.dim() == 1:
            tensor = tensor[None, :]
        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.shape[-1] > hidden_size:
            tensor = tensor[..., :hidden_size]
        elif tensor.shape[-1] < hidden_size:
            pad = torch.zeros((tensor.shape[0], hidden_size - tensor.shape[-1]), device=device, dtype=dtype)
            tensor = torch.cat([tensor, pad], dim=-1)
        return tensor

    def _rms_normalize_summary(self, summary: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        denom = torch.rsqrt(summary.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        return (summary.float() * denom).to(dtype=summary.dtype)

    def _apply_dynamics_experiment(
        self,
        h: torch.Tensor,
        c_t: torch.Tensor,
        context: Optional[dict],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        if not self.dynamics_experiment:
            return h, None, {}

        bsz, seq_len, hidden_size = h.shape
        device, dtype = h.device, h.dtype
        mode = self.dynamics_experiment
        chunk_summary, spans = self._chunk_summaries(h)
        chunk_summary = self._rms_normalize_summary(chunk_summary)
        num_chunks = chunk_summary.shape[1]
        global_summary = self._rms_normalize_summary(chunk_summary.mean(dim=1))
        recent_block = self._rms_normalize_summary(self._context_hidden(context, "recent_block_repr", bsz, hidden_size, device, dtype))
        world_summary = self._rms_normalize_summary(self._context_hidden(context, "world_summary", bsz, hidden_size, device, dtype))
        loop_history_summary = self._rms_normalize_summary(self._context_hidden(context, "loop_history_summary", bsz, hidden_size, device, dtype))
        scalar_ctx = self._context_scalars(context, bsz, device, dtype)

        is_summary_v1 = "summary_chunk_film_v1_core" in mode
        is_summary_v2 = "summary_chunk_film_v2_progress" in mode
        is_hici = "hici_construct_integrate_broadcast_v1" in mode
        is_budget_v1 = "budgeted_summary_routing_v1" in mode
        is_budget_v2 = "budgeted_summary_routing_v2_progress" in mode
        is_hier_v1 = "hier_block_token_v1_block_only" in mode
        is_hier_v2 = "hier_block_token_v2_attn_bias" in mode
        is_hier_v3 = "hier_block_token_v3_residual_delta" in mode
        is_double_p = "double_p_coarse_to_fine_v1" in mode
        is_memory_tier = "memory_tiered_routing_v1" in mode
        is_focus_v1 = "progress_focus_v1_chunk_query" in mode
        is_focus_v3 = "progress_focus_v3_dense_sparse_hybrid" in mode
        is_m1_lite = "m1_lite" in mode
        is_m1_anti_collapse = "m1_anti_collapse" in mode
        is_anti_budget = "anti_budget" in mode
        is_s_lite_control = "s_lite_control" in mode
        is_s_local_floor = "s_local_floor" in mode
        is_local_floor_guard = "local_floor" in mode or is_s_local_floor
        is_entropy_guard = "entropy_guard" in mode or is_m1_anti_collapse
        is_zone_loss = "zone_loss" in mode
        is_vitality_loss = "vitality_loss" in mode
        use_progress = is_summary_v2 or is_budget_v2 or is_focus_v1 or is_focus_v3

        c_expand = c_t[:, None, :].expand(-1, num_chunks, -1)
        block_expand = recent_block[:, None, :].expand(-1, num_chunks, -1)
        global_expand = global_summary[:, None, :].expand(-1, num_chunks, -1)
        loop_expand = loop_history_summary[:, None, :].expand(-1, num_chunks, -1)
        progress_weight = float(self.routing_progress_weight)
        if use_progress and progress_weight != 1.0:
            scalar_ctx = scalar_ctx * progress_weight
        progress_expand = scalar_ctx[:, None, :].expand(-1, num_chunks, -1) if use_progress else torch.zeros((bsz, num_chunks, 3), device=device, dtype=dtype)
        third_expand = global_expand if is_hici else (loop_expand if is_memory_tier else torch.zeros_like(global_expand))
        chunk_feat = torch.cat([chunk_summary, block_expand, third_expand, c_expand, progress_expand], dim=-1)
        self._ensure_feature_dim(chunk_feat, self.chunk_feature_dim, "chunk")

        chunk_gate_logits = self.chunk_gate_head(chunk_feat).squeeze(-1)
        chunk_gate = torch.sigmoid(chunk_gate_logits)
        film = self.chunk_film_head(chunk_feat)
        gamma_chunk, beta_chunk = film.chunk(2, dim=-1)
        gamma_tok = self._expand_chunk_values(gamma_chunk, spans, seq_len)
        beta_tok = self._expand_chunk_values(beta_chunk, spans, seq_len)

        c_expand_tok = c_t[:, None, :].expand(-1, seq_len, -1)
        token_scalar_tok = scalar_ctx[:, None, :].expand(-1, seq_len, -1) if use_progress else torch.zeros((bsz, seq_len, 3), device=device, dtype=dtype)
        token_feat = torch.cat([h, c_expand_tok, token_scalar_tok], dim=-1)
        self._ensure_feature_dim(token_feat, self.token_feature_dim, "token")
        token_score = torch.sigmoid(self.token_score_head(token_feat)).squeeze(-1)
        token_score = torch.nan_to_num(token_score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        c_bias = torch.tanh(self.ct_injection.get_bias(c_t)).unsqueeze(1)
        dense_gain_value = self.routing_weak_gain
        sparse_gain_value = self.routing_strong_gain
        if is_s_lite_control or is_m1_lite:
            dense_gain_value *= 0.75
            sparse_gain_value *= 0.60
        dense_gain = torch.tensor(dense_gain_value, device=device, dtype=dtype)
        sparse_gain = torch.tensor(sparse_gain_value, device=device, dtype=dtype)
        quarter = torch.tensor(0.25, device=device, dtype=dtype)
        half = torch.tensor(0.5, device=device, dtype=dtype)
        beta_tanh = 0.1 * torch.tanh(beta_tok)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_tok)
        gamma_delta = gamma - 1.0
        weak_dense = dense_gain * (torch.tanh(c_bias) + quarter * beta_tanh)
        strong_summary = sparse_gain * (beta_tanh + gamma_delta)

        selected_chunk_mask = torch.ones((bsz, num_chunks), device=device, dtype=torch.bool)
        selected_mass = chunk_gate.new_ones((bsz,))
        attn_bias: Optional[torch.Tensor] = None
        mod_stats: dict[str, torch.Tensor] = {}

        if is_budget_v1 or is_budget_v2:
            if is_budget_v2:
                budget_in = torch.cat([c_t, scalar_ctx], dim=-1)
            else:
                loop_ratio = float(context.get("loop_ratio", 0.0)) if context is not None else 0.0
                hard_proxy = float(context.get("hard_loop_var_proxy", 0.0)) if context is not None else 0.0
                budget_extra = torch.tensor([loop_ratio, hard_proxy, 1.0 - loop_ratio], device=device, dtype=dtype).expand(bsz, -1)
                budget_in = torch.cat([c_t, budget_extra], dim=-1)
            budget_value = torch.sigmoid(self.budget_head(budget_in)).squeeze(-1)
            budget_value = self.routing_budget_min + (self.routing_budget_max - self.routing_budget_min) * budget_value
            selected_chunk_mask = torch.zeros_like(selected_chunk_mask)
            budget_utilization = []
            for b in range(bsz):
                topk = max(1, min(num_chunks, int(round(float(num_chunks) * float(budget_value[b].item())))))
                idx = torch.topk(chunk_gate[b], k=topk).indices
                selected_chunk_mask[b, idx] = True
                budget_utilization.append(float(topk) / float(max(1, num_chunks)))
            selected_mass = torch.tensor(budget_utilization, device=device, dtype=dtype)
            mod_stats["budget_value_mean"] = budget_value.mean()
            mod_stats["budget_value_std"] = budget_value.std(unbiased=False)
            mod_stats["budget_utilization_ratio"] = selected_mass.mean()

        if is_hier_v1 or is_hier_v2 or is_hier_v3 or is_double_p:
            block_feat = torch.cat([chunk_summary, block_expand, torch.zeros_like(global_expand), c_expand, torch.zeros((bsz, num_chunks, 3), device=device, dtype=dtype)], dim=-1)
            self._ensure_feature_dim(block_feat, self.block_feature_dim, "block")
            block_score = self.block_score_head(block_feat).squeeze(-1)
            if is_double_p:
                selected_chunk_mask, coarse_mass = self._topp_mask(block_score, self.routing_top_p_coarse, self.routing_topk_blocks * 2)
                selected_mass = coarse_mass
                mod_stats["topk_or_topp_mass"] = coarse_mass.mean()
            else:
                selected_chunk_mask = self._topk_mask(block_score, self.routing_topk_blocks)
                selected_mass = selected_chunk_mask.float().mean(dim=-1)
            mod_stats["selected_block_count"] = selected_chunk_mask.float().sum(dim=-1).mean()
            mod_stats["selected_block_entropy"] = self._entropy(block_score).mean()

        if is_focus_v1 or is_focus_v3:
            focus_query = self.focus_query_head(torch.cat([c_t, scalar_ctx], dim=-1))
            focus_score = torch.einsum("bd,bkd->bk", focus_query, chunk_summary) / math.sqrt(float(hidden_size))
            selected_chunk_mask = self._topk_mask(focus_score, self.routing_topk_blocks)
            selected_mass = selected_chunk_mask.float().mean(dim=-1)
            mod_stats["focus_score_entropy"] = self._entropy(focus_score).mean()
            mod_stats["focus_span_count"] = selected_chunk_mask.float().sum(dim=-1).mean()
            mod_stats["focus_span_avg_width"] = torch.tensor(float(self.routing_chunk_size), device=device, dtype=dtype)

        if is_memory_tier:
            tier_logits = self.memory_tier_head(torch.cat([c_t, scalar_ctx], dim=-1))
            tier_weights = torch.softmax(tier_logits.float(), dim=-1).to(dtype)
            world_cap = self.routing_world_summary_cap
            if (is_m1_lite or is_m1_anti_collapse) and world_cap >= 1.0:
                world_cap = 0.45
            if world_cap < 1.0:
                capped_world = torch.clamp(tier_weights[:, 3], max=world_cap)
                tier_weights = torch.cat([tier_weights[:, :3], capped_world[:, None]], dim=-1)
                tier_weights = tier_weights / tier_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            min_local_share = self.routing_min_local_share
            if (is_m1_lite or is_m1_anti_collapse) and min_local_share <= 0.0:
                min_local_share = 0.20
            if min_local_share > 0.0:
                local_share = torch.clamp(tier_weights[:, 0], min=min_local_share)
                tier_weights = torch.cat([local_share[:, None], tier_weights[:, 1:]], dim=-1)
                tier_weights = tier_weights / tier_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            source_stack = torch.stack([chunk_summary.mean(dim=1), loop_history_summary, recent_block, world_summary], dim=1)
            soft_tier_mode = is_m1_lite or is_m1_anti_collapse or self.routing_tier_soft_only
            tier_choice = torch.argmax(tier_weights, dim=-1)
            if soft_tier_mode:
                chosen_source = (tier_weights.unsqueeze(-1) * source_stack).sum(dim=1)
            else:
                chosen_source = source_stack[torch.arange(bsz, device=device), tier_choice]
            h = h + sparse_gain * torch.tanh(self.memory_source_proj(chosen_source)).unsqueeze(1)
            local_selected = self._topk_mask(chunk_gate, self.routing_topk_blocks)
            if soft_tier_mode:
                selected_chunk_mask = local_selected
                selected_mass = (local_selected.float() * tier_weights[:, 0:1]).mean(dim=-1)
            else:
                local_active = tier_choice.eq(0)[:, None].expand(-1, num_chunks)
                selected_chunk_mask = local_selected & local_active
                selected_mass = selected_chunk_mask.float().mean(dim=-1)
            tier_entropy = self._entropy(tier_logits)
            dominant_tier_ratio = tier_weights.max(dim=-1).values
            prev_tier_choice = context.get("prev_tier_choice", None) if context is not None else None
            if isinstance(prev_tier_choice, torch.Tensor) and prev_tier_choice.shape == tier_choice.shape:
                tier_switch_rate = tier_choice.ne(prev_tier_choice.to(device=device)).float().mean()
            else:
                tier_switch_rate = tier_choice.new_zeros((), dtype=dtype)
            mod_stats["tier_weight_local"] = tier_weights[:, 0].mean()
            mod_stats["tier_weight_loop_history"] = tier_weights[:, 1].mean()
            mod_stats["tier_weight_block_repr"] = tier_weights[:, 2].mean()
            mod_stats["tier_weight_world_summary"] = tier_weights[:, 3].mean()
            mod_stats["tier_entropy"] = tier_entropy.mean()
            mod_stats["dominant_tier_ratio"] = dominant_tier_ratio.mean()
            mod_stats["tier_switch_rate"] = tier_switch_rate
            mod_stats["tier_local_share"] = tier_weights[:, 0].mean()
            mod_stats["tier_entropy_floor_violation"] = torch.relu(
                tier_entropy.new_tensor(
                    self.routing_tier_entropy_floor if self.routing_tier_entropy_floor > 0.0 else (1.0 if is_m1_anti_collapse else 0.0)
                )
                - tier_entropy
            ).mean()
            mod_stats["tier_local_floor_violation"] = torch.relu(
                tier_weights.new_tensor(min_local_share) - tier_weights[:, 0]
            ).mean()
            mod_stats["tier_choice_idx_mean"] = tier_choice.float().mean()
            mod_stats["tier_choice_last"] = tier_choice.float().mean()
            mod_stats["_tier_choice_vector"] = tier_choice.detach()

        if is_anti_budget:
            min_active_ratio = max(0.0, self.routing_min_local_share if self.routing_min_local_share > 0.0 else 0.20)
            min_active_k = max(1, min(num_chunks, int(round(float(num_chunks) * min_active_ratio))))
            for b in range(bsz):
                current_k = int(selected_chunk_mask[b].float().sum().item())
                if current_k >= min_active_k:
                    continue
                top_idx = torch.topk(chunk_gate[b], k=min_active_k).indices
                selected_chunk_mask[b, top_idx] = True
            selected_mass = selected_chunk_mask.float().mean(dim=-1)
            mod_stats["budget_min_active_ratio"] = selected_mass.mean()

        chunk_selected_tok = self._expand_chunk_values(selected_chunk_mask.to(dtype=dtype).unsqueeze(-1), spans, seq_len)
        selected_token_mask = chunk_selected_tok.squeeze(-1)
        local_floor = max(self.routing_local_floor, self.routing_modulation_floor if is_local_floor_guard else 0.0)
        if local_floor > 0.0:
            chunk_strength_tok = local_floor + (1.0 - local_floor) * chunk_selected_tok
        else:
            chunk_strength_tok = chunk_selected_tok
        chunk_strength_tok = chunk_strength_tok.clamp(max=max(1e-6, self.routing_modulation_ceiling))
        strong_summary_delta = chunk_strength_tok * strong_summary
        baseline_path_energy = (h + weak_dense).float().norm(dim=-1).mean()

        if is_summary_v1 or is_summary_v2 or is_hici or is_budget_v1 or is_budget_v2:
            # Optionally use a small gated residual delta branch instead of FiLM-style strong summary
            if getattr(self, "routing_use_residual_branch", False):
                local_delta = torch.tanh(self.local_delta_head(torch.cat([h, c_expand_tok], dim=-1)))
                # Choose gate per-chunk: either selection-only fixed amplitude, or learned sigmoid gate scaled
                if getattr(self, "ct_selection_only", False):
                    gate_chunk_vals = selected_chunk_mask.to(dtype=dtype) * float(self.ct_selection_amplitude)
                else:
                    gate_chunk_vals = float(self.ct_residual_gate_scale) * torch.sigmoid(chunk_gate_logits).to(dtype)
                gate_tok = self._expand_chunk_values(gate_chunk_vals.unsqueeze(-1), spans, seq_len)
                h = h + weak_dense + gate_tok * local_delta
                mod_stats["delta_local_norm_mean"] = local_delta.norm(dim=-1).mean()
                mod_stats["delta_local_norm_std"] = local_delta.norm(dim=-1).std(unbiased=False)
            else:
                h = h + weak_dense + strong_summary_delta
        elif is_hier_v1:
            h = h + chunk_strength_tok * dense_gain * (torch.tanh(c_bias) + half * beta_tanh)
        elif is_hier_v2:
            attn_bias = chunk_strength_tok * token_score.unsqueeze(-1) * sparse_gain * torch.tanh(c_bias)
            h = h + weak_dense
            mod_stats["selected_token_ratio"] = (selected_token_mask * (token_score > 0.5).float()).mean()
        elif is_hier_v3:
            local_delta = torch.tanh(self.local_delta_head(torch.cat([h, c_expand_tok], dim=-1)))
            h = h + weak_dense + chunk_strength_tok * token_score.unsqueeze(-1) * sparse_gain * local_delta
            mod_stats["selected_token_ratio"] = (selected_token_mask * (token_score > 0.5).float()).mean()
            mod_stats["delta_local_norm_mean"] = local_delta.norm(dim=-1).mean()
            mod_stats["delta_local_norm_std"] = local_delta.norm(dim=-1).std(unbiased=False)
        elif is_double_p:
            token_scores_masked = token_score * selected_token_mask
            token_logits = torch.log(token_scores_masked.clamp_min(1e-8))
            fine_mask, fine_mass = self._topp_mask(token_logits, self.routing_top_p_fine, self.routing_topk_tokens)
            local_delta = torch.tanh(self.local_delta_head(torch.cat([h, c_expand_tok], dim=-1)))
            fine_mask_tok = fine_mask.unsqueeze(-1).to(dtype)
            h = h + weak_dense + fine_mask_tok * (half * sparse_gain) * local_delta
            mod_stats["selected_token_ratio"] = fine_mask.float().mean()
            mod_stats["topk_or_topp_mass"] = fine_mass.mean()
            mod_stats["delta_local_norm_mean"] = local_delta.norm(dim=-1).mean()
            mod_stats["delta_local_norm_std"] = local_delta.norm(dim=-1).std(unbiased=False)
        elif is_focus_v1:
            h = h + weak_dense + strong_summary_delta
        elif is_focus_v3:
            local_delta = torch.tanh(self.local_delta_head(torch.cat([h, c_expand_tok], dim=-1)))
            h = h + weak_dense + chunk_strength_tok * sparse_gain * local_delta
            mod_stats["delta_local_norm_mean"] = local_delta.norm(dim=-1).mean()
            mod_stats["delta_local_norm_std"] = local_delta.norm(dim=-1).std(unbiased=False)
        elif is_memory_tier:
            h = h + weak_dense + chunk_strength_tok * (half * strong_summary)
        else:
            h = h + weak_dense

        gate_ref = token_score if (is_hier_v2 or is_hier_v3 or is_double_p) else chunk_gate
        mod_stats["modulated_chunk_ratio"] = (chunk_selected_tok.squeeze(-1) > 0.5).float().mean()
        mod_stats["unmodulated_path_energy"] = baseline_path_energy
        gain_ref = chunk_strength_tok.squeeze(-1).float()
        mod_stats["modulation_gain_low_ratio"] = (gain_ref < 0.33).float().mean()
        mod_stats["modulation_gain_mid_ratio"] = ((gain_ref >= 0.33) & (gain_ref < 0.66)).float().mean()
        mod_stats["modulation_gain_high_ratio"] = (gain_ref >= 0.66).float().mean()
        mod_stats["gate_mean"] = gate_ref.mean()
        mod_stats["gate_std"] = gate_ref.std(unbiased=False)
        mod_stats["gate_saturation_ratio"] = ((gate_ref < 0.01) | (gate_ref > 0.99)).float().mean()
        mod_stats["nonfinite_gate_count"] = (~torch.isfinite(gate_ref)).float().sum()
        mod_stats["selected_chunk_ratio"] = selected_chunk_mask.float().mean()
        mod_stats["selected_token_ratio"] = mod_stats.get("selected_token_ratio", selected_token_mask.mean())
        mod_stats["chunk_gamma_mean"] = gamma_chunk.mean()
        mod_stats["chunk_gamma_std"] = gamma_chunk.std(unbiased=False)
        mod_stats["chunk_beta_mean"] = beta_chunk.mean()
        mod_stats["chunk_beta_std"] = beta_chunk.std(unbiased=False)
        mod_stats["dense_path_gain"] = dense_gain
        mod_stats["sparse_path_gain"] = sparse_gain
        return h, attn_bias, mod_stats

    @torch.no_grad()
    def _run_shared_stack(self, h: torch.Tensor, c_t: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """裸跑 shared_layers（无 c_t 注入，输入 h 是已注入版本），用于理论 probe 的扰动 forward。
        和 _run_body_layers 一致使用残差归一化设计：F(h) = h + α · LayerNorm(g(h) - h)"""
        h_in = h
        h_cur = h
        for layer in self.shared_layers:
            h_cur = layer(h_cur, c_t=c_t, attn_bias=None,
                          use_gradient_checkpointing=False,
                          loop_idx=loop_idx, head_partition=self.head_partition)
        delta = h_cur - h_in
        delta_normed = self._body_out_norm(delta)
        return h_in + self._body_residual_alpha * delta_normed

    @torch.no_grad()
    def measure_theory_probes(
        self,
        h: torch.Tensor,
        c_t: torch.Tensor,
        c_t_next: torch.Tensor,
        loop_idx: int,
        rel_eps: float = 0.05,
    ) -> dict:
        """测量四个主判据 probe（用户 2026-04-12 计划）：

        - rho_h_frozen: 冻结 c_t 和 loop_idx，扰动 h 测 F_k 的局部雅可比谱半径
            扰动量 = rel_eps * ||h||（相对扰动），避免 bf16 下的绝对 eps 被噪声淹没
            rho_h = ||F(h + delta_h) - F(h)|| / ||delta_h||
        - rho_c_drift: 冻结 h 和 loop_idx，用实际 c_t→c_t_next 的变化测 F_k 对 c_t 的敏感度
            rho_c = ||F(h, c_next) - F(h, c)|| / ||c_next - c||
        - eta_moving_fp: c_t 变化导致的 F 输出变化 vs h 同尺度扰动导致的 F 输出变化的比值
            eta = ||F(h, c_next) - F(h, c)|| / (||F(h + delta_h) - F(h)|| + eps)
            >>1 表示不动点漂移主导，≈1 表示两条路径对等

        所有 forward 在 @torch.no_grad 下，F 的输出在 float32 下做 diff 避免 bf16 精度问题。
        成本：3 次 shared_layers forward（baseline + perturbed_h + perturbed_c），各 4 层。
        """
        probes = {
            "rho_h_frozen": None,
            "rho_c_drift": None,
            "eta_moving_fp": None,
            "probe_delta_h_norm": None,  # 实际扰动量，调试用
            "probe_delta_c_norm": None,
        }
        # 防御性保存 _ct_lora_prev：probe 会调用 shared_layers 3 次，
        # 如果 ct_conditioned_lora=True，每次 forward 都会改写 layer._ct_lora_prev，
        # 污染主 forward 后续循环的 delta_ct 计算（sub agent 审查发现）。
        _saved_lora_prev = []
        for _layer in self.shared_layers:
            if hasattr(_layer, "_ct_lora_prev"):
                _val = _layer._ct_lora_prev
                _saved_lora_prev.append(_val.clone() if isinstance(_val, torch.Tensor) else _val)
            else:
                _saved_lora_prev.append(None)
        try:
            # 先用裸 ct_injection.get_bias 生成两种 bias，不走 clamp（probe 是测原生敏感度）
            c_bias = self.ct_injection.get_bias(c_t).unsqueeze(1).to(h.dtype)
            c_bias_next = self.ct_injection.get_bias(c_t_next).unsqueeze(1).to(h.dtype)
            h_inj = h + c_bias
            # baseline: F(h, c)
            h_base = self._run_shared_stack(h_inj, c_t, loop_idx).float()
            # 扰动 h：v = unit dir，扰动量 = rel_eps * ||h_inj||（相对尺度，bf16 可感知）
            h_inj_norm = h_inj.float().norm().clamp(min=1e-6).item()
            delta_h_target = rel_eps * h_inj_norm  # 标量目标扰动总量
            v = torch.randn_like(h)
            v_norm = v.float().norm().clamp(min=1e-8).item()
            v = (v / v_norm) * delta_h_target  # 现在 ||v|| == delta_h_target
            h_perturbed = h_inj + v.to(h.dtype)
            h_pert_out = self._run_shared_stack(h_perturbed, c_t, loop_idx).float()
            h_diff_norm = (h_pert_out - h_base).norm().item()
            probes["probe_delta_h_norm"] = float(delta_h_target)
            probes["rho_h_frozen"] = float(h_diff_norm / max(delta_h_target, 1e-12))
            # 扰动 c_t：用真实的 c_t_next - c_t
            h_cnext_inj = h + c_bias_next  # h 不变，c bias 换成 next
            h_cnext_out = self._run_shared_stack(h_cnext_inj, c_t_next, loop_idx).float()
            c_diff_norm = (h_cnext_out - h_base).norm().item()
            dc_norm = (c_t_next - c_t).float().norm().item()
            probes["probe_delta_c_norm"] = float(dc_norm)
            if dc_norm > 1e-8:
                probes["rho_c_drift"] = float(c_diff_norm / dc_norm)
            # eta_moving_fp: c_t 变化贡献 / h 扰动贡献
            h_eff_norm = max(h_diff_norm, 1e-8)
            probes["eta_moving_fp"] = float(c_diff_norm / h_eff_norm)
        except Exception as _e:
            # probe 失败不应影响训练，记 None 即可
            pass
        finally:
            # 恢复 _ct_lora_prev，避免污染主 forward 后续循环
            for _layer, _val in zip(self.shared_layers, _saved_lora_prev):
                if hasattr(_layer, "_ct_lora_prev"):
                    _layer._ct_lora_prev = _val
        return probes

    def _run_body_layers(
        self,
        h: torch.Tensor,
        c_t: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        loop_idx: int,
        ct_base_bias: Optional[torch.Tensor],
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Phase E body F(h) — 运行 shared_layers stack 作为能量方案 A 的内部动力学。
        与主 forward 中的 stack 运行等价（per-layer c_t 注入一致），但接受 h 作为纯输入。
        每步能量梯度下降都会 call 一次这个 body。

        关键：强制 math SDPA backend 以支持 double backward
        （flash / mem_efficient 的 backward 不支持二阶导，Phase E create_graph=True 需要）
        """
        # SDPBackend 枚举导入（惰性），避免顶层依赖
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            _sdpa_ctx = sdpa_kernel(SDPBackend.MATH)
        except Exception:
            # 旧版 torch fallback
            import contextlib
            _sdpa_ctx = contextlib.nullcontext()

        h_in = h  # 保存输入用于残差归一化
        # ── Stellarator 分支：F_main(h) 不看 c_t ────────────────────────
        if self._stellarator_mode:
            with _sdpa_ctx:
                h_cur = h
                for i, layer in enumerate(self.shared_layers):
                    # 关键: c_t=None 阻断所有层内 c_t 分支（Mamba/DiffAttn/FFN 调制）
                    # ct_layer_bias=None 阻断 per-layer bias
                    h_cur = layer(
                        h_cur,
                        c_t=None,
                        attn_bias=attn_bias,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        loop_idx=loop_idx,
                        head_partition=self.head_partition,
                        ct_layer_bias=None,
                    )
            # 残差归一化保留（结构性 Lipschitz 收缩）
            delta_main = self._body_out_norm(h_cur - h_in)
            f_main = h_in + self._body_residual_alpha * delta_main
            # Low-rank c_t modulator: c_t → rank → hidden，zero-init 让初始 bias=0
            if c_t is not None:
                _mod = F.silu(self._stellarator_mod_down(c_t))
                _bias = self._stellarator_mod_up(_mod).unsqueeze(1)  # [B, 1, D]
                # 归一化 bias 范数和 h 对齐（结构上限制强度）
                _bias = self._body_out_norm(_bias)
            else:
                _bias = h_in.new_zeros((h_in.shape[0], 1, h_in.shape[-1]))
            # Gated fusion: h_next = h + g · (F_main - h + bias)
            # g = sigmoid(logit_init=0) = 0.5 → 初始就是 "half-step towards F_main"
            g = torch.sigmoid(self._stellarator_gate_logit)
            h_next = h_in + g * ((f_main - h_in) + _bias)
            # 诊断
            with torch.no_grad():
                self._last_stellarator_gate = float(g.item())
                _fm_delta_norm = (f_main - h_in).detach().norm().item()
                _bias_norm = _bias.detach().norm().item()
                self._last_stellarator_bias_ratio = _bias_norm / max(_fm_delta_norm, 1e-8)
            return h_next

        # ── 旧路径：c_t 深度穿透 body（v18 及以前）────────────────────────
        with _sdpa_ctx:
            h_cur = h
            for i, layer in enumerate(self.shared_layers):
                bias_i = None
                if ct_base_bias is not None:
                    bias_i = ct_base_bias * self.ct_layer_directions[i]
                h_cur = layer(
                    h_cur,
                    c_t=c_t,
                    attn_bias=attn_bias,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    loop_idx=loop_idx,
                    head_partition=self.head_partition,
                    ct_layer_bias=bias_i,
                )
        # 残差归一化设计: F(h) = h + α · LayerNorm(g(h) - h)
        # Lipschitz 上界 = 1 + α，α<<1 时严格收缩
        delta = h_cur - h_in
        delta_normed = self._body_out_norm(delta)
        return h_in + self._body_residual_alpha * delta_normed

    def _phase_e_damped_loop(
        self,
        h: torch.Tensor,
        c_t: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        loop_idx: int,
        ct_base_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Damped fixed-point iteration（Phase E 一阶近似）。

        公式: h_{k+1} = (1-η) · h_k + η · F(h_k, c_t)

        理论:
            ∇_h E(h) = ∇_h (0.5 ||h - F(h)||²)
                     = (h - F(h)) · (I - ∂F/∂h)
                     ≈ (h - F(h))    (当 J_F 谱半径小，典型情形)
            → h - η·∇E ≈ h - η(h - F(h)) = (1-η)h + η·F(h)

        保留性质:
            - 不动点: h* = F(h*)（即 ||h - F(h)|| → 0）
            - 构造性收缩: 若 ‖J_F‖ < 1/η，则 ‖Δh‖ 每步按 (1-η + η·J_F) 缩放，收敛
            - K 步 = 迭代式 "深度扩展"（每步用同一 body）

        免除的坑:
            - autograd.grad 二阶图 → 和 reentrant ckpt / SDPA kernel 兼容性
            - bf16 下 ∇²E 传播的数值精度问题
            - 大模型 (216M) 长链累积噪声
        """
        K = self._phase_e_K_max
        eta = self._phase_e_eta

        energy_trace: List[float] = []
        delta_norm_trace: List[float] = []

        for k in range(K):
            h_body = self._run_body_layers(
                h, c_t, attn_bias, loop_idx, ct_base_bias,
                use_gradient_checkpointing=False,
            )
            # Stellarator mode: body 内部已做 gated fusion h + g*(F_main-h+bias)，不再叠 damping
            if self._stellarator_mode:
                h = h_body
            else:
                # Damped 更新：线性插值，保持 body 参数的梯度图自然存在
                h = (1.0 - eta) * h + eta * h_body
            # Langevin 噪声（T>0）
            if self._phase_e_temperature > 0.0 and self.training:
                h = h + torch.randn_like(h) * math.sqrt(
                    2.0 * eta * self._phase_e_temperature
                )
            # 诊断：记录等效 energy proxy = ||h - h_body|| 和 delta norm
            with torch.no_grad():
                _diff = (h.detach() - h_body.detach()).float()
                _E_proxy = 0.5 * _diff.pow(2).mean().item()
                energy_trace.append(float(_E_proxy))
                delta_norm_trace.append(float((eta * _diff).norm().item()))

        self._last_phase_e_energy_trace = energy_trace
        self._last_phase_e_grad_norm_trace = delta_norm_trace  # 沿用接口，实际是 delta norm
        self._last_phase_e_K_used = K
        return h

    def _phase_e_inner_loop(
        self,
        h: torch.Tensor,
        c_t: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        loop_idx: int,
        ct_base_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Phase E 能量梯度下降主环（替换 shared_layers 单次 forward）。

        两种模式:
        1. **Damped 模式**（默认）: h ← (1-η)·h + η·F(h)
           这是 Phase E 的一阶近似（∇_h E ≈ h - F(h) 当 J_F 小时），
           保留不动点 h=F(h) 和构造性收缩，但不用 autograd.grad / 无 double backward，
           body 参数通过 lm_head 反传链正常训练。bf16 + 大模型稳定。
        2. **Grad 模式**: h_{k+1} = h_k - η · ∇_h E(h_k)
           完整 Phase E 能量梯度公式，需要 double backward（math SDPA + no reentrant ckpt）。
           理论上更强，但 bf16 + 216M + Mamba 下数值不稳（见 v2/v3/v4 NaN 记录）。

        Truncated backprop（仅 grad 模式）:
            - 前 K_max - k_backprop 步: detach
            - 后 k_backprop 步: create_graph=True
        """
        if self._phase_e_damped_mode:
            return self._phase_e_damped_loop(h, c_t, attn_bias, loop_idx, ct_base_bias)

        K = self._phase_e_K_max
        kb = self._phase_e_k_backprop if self.training else 0
        n_detached = max(0, K - kb) if kb > 0 else K

        # 初始 h 必须 requires_grad 才能对其求 ∇_h E
        if not h.requires_grad:
            h = h.detach().requires_grad_(True)

        energy_trace: List[float] = []
        grad_norm_trace: List[float] = []
        K_used = 0

        for k in range(K):
            is_detached = k < n_detached
            if is_detached:
                with torch.enable_grad():
                    h_leaf = h.detach().requires_grad_(True)
                    h_body = self._run_body_layers(
                        h_leaf, c_t, attn_bias, loop_idx, ct_base_bias,
                        use_gradient_checkpointing=False,
                    )
                    # 能量用 mean 而非 sum：尺度不随 seq_len×hidden 缩放，eta 跨尺寸稳定
                    E = 0.5 * ((h_leaf - h_body).float() ** 2).mean()
                    grad_h, = torch.autograd.grad(
                        E, h_leaf, create_graph=False, retain_graph=False
                    )
                # 安全网：grad_h 相对 h 做范数裁剪，防止单步过冲
                grad_h_det = grad_h.detach()
                _g_norm = grad_h_det.float().norm().clamp(min=1e-8)
                _h_norm = h.detach().float().norm().clamp(min=1e-8)
                _max_ratio = 0.5  # 单步最大位移 = 0.5 * ||h||
                _scale = torch.clamp(_max_ratio * _h_norm / (self._phase_e_eta * _g_norm), max=1.0)
                h = (h.detach() - self._phase_e_eta * _scale.to(grad_h_det.dtype) * grad_h_det)
                if self._phase_e_temperature > 0.0 and self.training:
                    h = h + torch.randn_like(h) * math.sqrt(
                        2.0 * self._phase_e_eta * self._phase_e_temperature
                    )
                # 最后一个 detach 步之后如要进 grad 阶段，需重开 grad
                if k == n_detached - 1 and kb > 0 and self.training:
                    h = h.detach().requires_grad_(True)
                energy_trace.append(float(E.detach().item()))
                grad_norm_trace.append(float(_g_norm.item()))
            else:
                h_body = self._run_body_layers(
                    h, c_t, attn_bias, loop_idx, ct_base_bias,
                    use_gradient_checkpointing=False,
                )
                E = 0.5 * ((h - h_body).float() ** 2).mean()
                grad_h, = torch.autograd.grad(
                    E, h, create_graph=self.training, retain_graph=self.training
                )
                # 同样的安全裁剪（对 create_graph 分支用 .detach() 的 norm 引导 scale
                # 但 scale 作为常数乘到带图的 grad_h 上，不破坏二阶图）
                _g_norm_det = grad_h.detach().float().norm().clamp(min=1e-8)
                _h_norm_det = h.detach().float().norm().clamp(min=1e-8)
                _max_ratio = 0.5
                _scale = torch.clamp(_max_ratio * _h_norm_det / (self._phase_e_eta * _g_norm_det), max=1.0)
                h = h - self._phase_e_eta * _scale.to(grad_h.dtype) * grad_h
                if self._phase_e_temperature > 0.0 and self.training:
                    h = h + torch.randn_like(h) * math.sqrt(
                        2.0 * self._phase_e_eta * self._phase_e_temperature
                    )
                energy_trace.append(float(E.detach().item()))
                grad_norm_trace.append(float(_g_norm_det.item()))
            K_used += 1
            # 梯度范数早停
            if self._phase_e_grad_stop_eps > 0.0 and grad_norm_trace[-1] < self._phase_e_grad_stop_eps:
                break

        self._last_phase_e_energy_trace = energy_trace
        self._last_phase_e_grad_norm_trace = grad_norm_trace
        self._last_phase_e_K_used = K_used
        return h

    def forward(
        self,
        h: torch.Tensor,
        c_t: torch.Tensor,
        r_t: Optional[torch.Tensor] = None,
        r_trust: Optional[torch.Tensor] = None,
        r_t_mode: str = "blend",
        disable_ct_injection: bool = False,
        modulation_context: Optional[dict] = None,
        use_gradient_checkpointing: bool = False,
        loop_idx: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        if getattr(self, "freeze_ct_during_reason", False):
            c_t = c_t.detach()
        if self.ct_grad_scale != 1.0:
            c_t = grad_scale(c_t, self.ct_grad_scale)
        c_bias = self.ct_injection.get_bias(c_t).unsqueeze(1)
        # 诊断: proj(c_t) 的范数和 h 的范数比值
        _h_norm = max(h.detach().norm().item(), 1e-8)
        _bn = c_bias.detach().norm().item()
        self._last_ct_inject_ratio = _bn / _h_norm
        self._last_ct_inject_ratio_pre = self._last_ct_inject_ratio
        self._last_ct_bias_norm_pre = _bn
        self._last_ct_bias_norm_applied = _bn
        self._last_ct_h_norm_ref = _h_norm
        if disable_ct_injection:
            c_bias = torch.zeros_like(c_bias)
            self._last_ct_inject_ratio = 0.0
        switch_value: Optional[torch.Tensor] = None
        modulation_stats: dict = {}
        attn_bias: Optional[torch.Tensor] = None
        if not disable_ct_injection and self.dynamics_experiment:
            h, attn_bias, modulation_stats = self._apply_dynamics_experiment(h, c_t, modulation_context)
        if not disable_ct_injection:
            if self.ct_modulation_mode == "additive":
                h = h + c_bias
            elif self.ct_modulation_mode == "lowrank_hyperbias" and self.ct_lowrank_down is not None and self.ct_lowrank_up is not None:
                lowrank_bias = self.ct_lowrank_up(F.silu(self.ct_lowrank_down(c_t))).unsqueeze(1)
                h = h + lowrank_bias
            elif self.ct_modulation_mode == "token_selective" and self.token_selective_gate is not None:
                token_gate = torch.sigmoid(self.token_selective_gate(torch.cat([h, c_t.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1)))
                h = h + token_gate * c_bias
        if self.r_injection is not None and r_t is not None:
            r_bias = self.r_injection(r_t).unsqueeze(1)
            trust = r_trust.unsqueeze(1) if r_trust is not None else torch.zeros_like(c_bias[..., :1])
            if self.reason_switch_gate is not None:
                local_window = h[:, -min(self.router_window, h.shape[1]) :, :]
                q = self.route_query(torch.cat([c_t, r_t], dim=-1)).unsqueeze(1)
                k = self.route_key(local_window)
                v = self.route_value(local_window)
                route_out = F.scaled_dot_product_attention(q, k, v)
                route_summary = route_out.squeeze(1)
                gate_in = torch.cat([route_summary, c_t, r_t, trust.squeeze(1)], dim=-1)
                switch_value = torch.sigmoid(self.reason_switch_gate(gate_in))
            else:
                switch_value = trust.squeeze(1)
            effective_switch = trust * switch_value.unsqueeze(1)
            if r_t_mode == "blend":
                if self.ct_modulation_mode != "additive":
                    h = h + effective_switch * r_bias
                else:
                    h = h + effective_switch * (r_bias - c_bias)
            elif r_t_mode == "parallel":
                h = h + effective_switch * r_bias
        # Phase C: inject per-phase embedding before shared layers
        if self.num_phases > 0:
            phase_idx = loop_idx % self.num_phases
            h = h + self.phase_embed.weight[phase_idx].unsqueeze(0).unsqueeze(0)
        # LoopFormer time conditioning: inject normalized time t and step-size dt
        if self._enable_time_cond:
            t_norm = float(loop_idx) / 20.0   # normalized position [0, 1)
            dt_norm = 1.0 / 20.0              # step size
            time_feat = h.new_tensor([[t_norm, dt_norm]], dtype=self.time_proj.weight.dtype)  # [1, 2]
            h = h + self.time_proj(time_feat).unsqueeze(1).to(h.dtype)  # broadcast [B, T, D]
        _h_pre_shared = h if self._identity_alpha > 0.0 else None
        # per-layer c_t 注入: detach(c_bias) × 4 个可学习方向向量
        # 零额外梯度路径: base_bias 是常数，layer_direction 梯度只需本层 d(loss)/d(h)
        _ct_base_bias = c_bias.detach() if (self._ct_per_layer_inject and c_t is not None) else None
        _layer_lora_ratios: List[float] = []
        _layer_lora_norms: List[float] = []
        if self._enable_phase_e_main:
            # Phase E 主集成: 替换 shared_layers 单次 forward 为 K 步能量梯度下降
            # body 函数复用 shared_layers stack，c_t 注入路径完全一致
            h = self._phase_e_inner_loop(h, c_t, attn_bias, loop_idx, _ct_base_bias)
            # LoRA stats 不适用（Phase E 每步 body 都会跑 LoRA，记最后一次）
            for layer in self.shared_layers:
                _layer_lora_ratios.append(float(getattr(layer, "_last_lora_delta_ratio", 0.0)))
                _layer_lora_norms.append(float(getattr(layer, "_last_lora_delta_norm", 0.0)))
        else:
            for i, layer in enumerate(self.shared_layers):
                _bias_i = None
                if _ct_base_bias is not None:
                    _bias_i = _ct_base_bias * self.ct_layer_directions[i]  # [B, 1, 768]
                h = layer(h, c_t=c_t, attn_bias=attn_bias, use_gradient_checkpointing=use_gradient_checkpointing,
                          loop_idx=loop_idx, head_partition=self.head_partition, ct_layer_bias=_bias_i)
                _layer_lora_ratios.append(float(getattr(layer, "_last_lora_delta_ratio", 0.0)))
                _layer_lora_norms.append(float(getattr(layer, "_last_lora_delta_norm", 0.0)))
        # Identity-biased recurrence: h = alpha * h_new + (1-alpha) * h_old
        if _h_pre_shared is not None:
            h = self._identity_alpha * h + (1.0 - self._identity_alpha) * _h_pre_shared
        # Phase B: MoR loop-conditioned expert routing (applied after shared layers)
        if self.mor_routing:
            loop_emb = self.mor_loop_embed.weight[loop_idx % self.mor_loop_embed.num_embeddings]
            router_logits = self.mor_router(loop_emb)  # [num_experts]
            topk_vals, topk_idx = router_logits.topk(self.mor_topk)
            topk_weights = torch.softmax(topk_vals, dim=-1)
            expert_out = h.new_zeros(h.shape)
            for i in range(self.mor_topk):
                eidx = topk_idx[i].item()
                expert_out = expert_out + topk_weights[i] * self.mor_experts_up[eidx](
                    F.silu(self.mor_experts_down[eidx](self.mor_gate_norm(h)))
                )
            h = h + expert_out
        modulation_stats["loop_lora_delta_ratio_mean"] = h.new_tensor(sum(_layer_lora_ratios) / max(1, len(_layer_lora_ratios)))
        modulation_stats["loop_lora_delta_norm_mean"] = h.new_tensor(sum(_layer_lora_norms) / max(1, len(_layer_lora_norms)))
        modulation_stats["ct_inj_pre"] = h.new_tensor(self._last_ct_inject_ratio_pre)
        modulation_stats["alpha_true"] = h.new_tensor(self._last_ct_inject_ratio)
        return h, switch_value, modulation_stats


class LumaBackbone(nn.Module):
    """Luma first compresses the conversation, then loops through a smaller inner theatre until the state feels settled.
    Luma 先压缩对话，再在更小的内部剧场里循环推理，直到状态趋于稳定。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.config = config
        self.embedding = FactorizedEmbedding(config)
        self.compression = CompressionZone(config)
        self.reason_memory = MemoryTokenBank(4, config.hidden_size)
        self.mhc = MHCResidualStreams(
            hidden_size=config.hidden_size,
            n_streams=config.mhc_streams,
            sinkhorn_iters=config.mhc_sinkhorn_iters,
            alpha_init=config.mhc_alpha_init,
        )
        self.reason_core = LumaReasonCore(config)
        rmode = config.attnres_reason_mode or config.attnres_mode
        if rmode == "paper":
            self.unified_attnres = PaperUnifiedAttnRes(config.hidden_size, max_loops=config.reason_loops_max + 2, eps=config.rms_norm_eps)
        elif rmode == "paper_global_q":
            self.unified_attnres = PaperUnifiedAttnResGlobalQ(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.unified_attnres = UnifiedAttnRes(config.hidden_size, eps=config.rms_norm_eps)
        self.introspection_state_stream = IntrospectionStateStream(config)
        # IS: introspection input/injection upgrades
        self._is_input_mode = config.introspection_input_mode
        self._is_inject_mode = config.introspection_inject_mode
        if self._is_input_mode in ("memory", "chunked_memory"):
            self.memory_token_reader = MemoryTokenReader(
                num_tokens=config.introspection_memory_tokens,
                hidden_size=config.hidden_size, meta_dim=config.meta_dim, num_heads=4,
            )
        else:
            self.memory_token_reader = None
        if self._is_inject_mode == "token_aware":
            self.token_aware_inject = TokenAwareCTInjection(config.c_t_dim, config.hidden_size)
        elif self._is_inject_mode == "bixt":
            self.bixt_cross_attn = BiXTCrossAttention(config.hidden_size, config.meta_dim, num_heads=4)
            # bixt needs memory tokens even if input_mode != "memory"
            if self.memory_token_reader is None:
                self.memory_token_reader = MemoryTokenReader(
                    num_tokens=config.introspection_memory_tokens,
                    hidden_size=config.hidden_size, meta_dim=config.meta_dim, num_heads=4,
                )
        elif self._is_inject_mode == "cmda":
            self.cmda_modulation = CMDAModulation(config.c_t_dim, config.hidden_size, config.meta_dim, enable_token_wish=config.cmda_token_wish)
        elif self._is_inject_mode == "bixt_cmda":
            self.bixt_cross_attn = BiXTCrossAttention(config.hidden_size, config.meta_dim, num_heads=4)
            self.cmda_modulation = CMDAModulation(config.c_t_dim, config.hidden_size, config.meta_dim, enable_token_wish=config.cmda_token_wish)
            if self.memory_token_reader is None:
                self.memory_token_reader = MemoryTokenReader(
                    num_tokens=config.introspection_memory_tokens,
                    hidden_size=config.hidden_size, meta_dim=config.meta_dim, num_heads=4,
                )
        # SelfJEPAResidualPredictor is a separate prediction head on top of that stream.
        # SelfJEPAResidualPredictor 是叠在其上的独立 Self-JEPA 残差预测头。
        self.self_jepa_residual_predictor = SelfJEPAResidualPredictor(config)
        self.self_jepa_progress_shape_head = SelfJEPAProgressShapeHead(config)
        self.trajectory_health_probe = TrajectoryHealthProbe(config) if config.enable_trajectory_health_probe else None
        self.reasoning_state_ring = TinyReasoningStateRing(config) if config.enable_reasoning_state_ring else None
        self.reasoning_state_to_hidden = nn.Linear(config.r_t_dim, config.hidden_size, bias=False) if config.enable_reasoning_state_ring else None
        if config.world_jepa_mode == "full":
            self.world_latent_jepa = LeWorldModelStyleJEPA(config)
        else:
            self.world_latent_jepa = WorldLatentJEPA(config)
        self.self_world_coupler = nn.Linear(config.world_dim, config.c_t_dim, bias=False)
        self.ct_world_jepa = CtWorldJEPA(config) if config.enable_ct_world_jepa else None
        self.self_check_ring = TinySlowSelfCheckRing(config) if config.enable_self_check_ring else None
        self.self_check_loss_weight = float(config.self_check_loss_weight)
        self.loop_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.exit_controller = ExitController(
            delta_threshold=config.exit_delta_threshold,
            self_threshold=config.exit_self_threshold,
            rollout_threshold=config.exit_rollout_threshold,
            world_threshold=config.exit_world_threshold,
            self_check_threshold=config.exit_self_check_threshold,
            improvement_margin=config.exit_improvement_margin,
            score_threshold=config.exit_score_threshold,
            use_sampling=config.exit_use_sampling,
            train_use_sampling=config.exit_train_use_sampling,
            eval_use_sampling=config.exit_eval_use_sampling,
            sampling_temperature=config.exit_sampling_temperature,
            enable_jepa_crystal=config.enable_exit_jepa_crystal,
            jepa_crystal_temperature=config.exit_jepa_crystal_temperature,
            gain_hidden_dim=config.exit_gain_hidden_dim,
            gain_weight=config.exit_gain_weight,
            uncertainty_feature_weight=config.exit_uncertainty_feature_weight,
            crystal_feature_weight=config.exit_crystal_feature_weight,
            enable_progress_exit_readout=config.enable_progress_exit_readout,
            progress_gain_weight=config.exit_progress_gain_weight,
            progress_trend_weight=config.exit_progress_trend_weight,
            progress_plateau_weight=config.exit_progress_plateau_weight,
            second_order_delta_weight=getattr(config, "exit_second_order_delta_weight", 0.0),
            min_loops=config.exit_min_loops,
            bias_init=config.exit_bias_init,
            warmup_steps=config.exit_warmup_steps,
            ct_drift_weight=config.exit_ct_drift_weight,
            know_gap_weight=config.exit_know_gap_weight,
        )
        # True MoR: token-level depth routing
        self.token_depth_router: Optional[TokenDepthRouter] = None
        if config.enable_token_depth_routing:
            self.token_depth_router = TokenDepthRouter(
                config.hidden_size, max_loops=64, eps=config.rms_norm_eps,
            )
        # NM: Neuromodulated c_t writer
        if config.enable_neuromod_ct:
            self.neuromod_ct_writer = NeuromodulatedCTWriter(
                c_t_dim=config.c_t_dim, hidden_size=config.hidden_size,
                rank=config.neuromod_hebb_rank, mode=config.neuromod_mode,
                use_delta_rule=config.neuromod_use_delta_rule,
                enable_fox_decay=config.neuromod_fox_decay,
            )
        else:
            self.neuromod_ct_writer = None
        # Masked h prediction: c_t 预测 h 的被 mask 部分，error 接赫布 surprise
        self._h_mask_ratio = config.h_mask_ratio
        if self._h_mask_ratio > 0:
            self.h_mask_predictor = nn.Linear(config.c_t_dim, config.hidden_size, bias=False)
            nn.init.zeros_(self.h_mask_predictor.weight)
        else:
            self.h_mask_predictor = None
        # ES: Exit quality probe
        self._es_entropy = config.enable_exit_entropy_signal
        self._es_token_sens = config.enable_exit_token_sensitivity
        self._es_ct_curv = config.enable_exit_ct_curvature
        self._es_confidence = config.enable_exit_confidence_gap
        if self._es_entropy or self._es_confidence or self._es_token_sens:
            self.exit_quality_probe = ExitQualityProbe(config.hidden_size)
        else:
            self.exit_quality_probe = None
        # PC: Predictive Coding error correction
        self._pc_enabled = config.enable_pc_correction
        self._pc_alpha = config.pc_alpha
        if self._pc_enabled:
            self.pc_corrector = PCErrorCorrector(config.c_t_dim, config.hidden_size)
        else:
            self.pc_corrector = None
        self.enable_sigreg_rollout = bool(config.enable_sigreg_rollout)
        self.enable_sigreg_delta = bool(config.enable_sigreg_delta)
        self.enable_sigreg_ct = bool(config.enable_sigreg_ct)
        self.sigreg_rollout_weight = float(config.sigreg_rollout_weight)
        self.sigreg_delta_weight = float(config.sigreg_delta_weight)
        self.sigreg_ct_weight = float(config.sigreg_ct_weight)
        self.sigreg_num_slices = int(config.sigreg_num_slices)
        self.sigreg_eps = float(config.sigreg_eps)
        sigreg_t_min = float(config.sigreg_t_min)
        sigreg_t_max = float(config.sigreg_t_max)
        sigreg_num_points = max(2, int(config.sigreg_num_points))
        sigreg_lambda = max(1e-6, float(config.sigreg_lambda))
        sigreg_t = torch.linspace(sigreg_t_min, sigreg_t_max, sigreg_num_points, dtype=torch.float32)
        sigreg_trap = torch.full((sigreg_num_points,), (sigreg_t_max - sigreg_t_min) / max(1, sigreg_num_points - 1), dtype=torch.float32)
        sigreg_trap[0] *= 0.5
        sigreg_trap[-1] *= 0.5
        sigreg_phi0 = torch.exp(-0.5 * sigreg_t.square())
        sigreg_weight = torch.exp(-sigreg_t.square() / (2.0 * sigreg_lambda**2))
        self.register_buffer("sigreg_t", sigreg_t, persistent=False)
        self.register_buffer("sigreg_phi0", sigreg_phi0, persistent=False)
        self.register_buffer("sigreg_weights", sigreg_trap * sigreg_weight, persistent=False)
        # Coconut: project c_t into hidden_size thought token for re-injection
        self._coconut = bool(config.enable_coconut)
        self._coconut_rounds = int(config.coconut_rounds)
        if self._coconut:
            self.coconut_proj = nn.Linear(config.c_t_dim, config.hidden_size, bias=False)
            nn.init.normal_(self.coconut_proj.weight, std=0.02)
        self.final_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _hierarchical_rollout_horizons(self) -> set[int]:
        max_horizon = self.config.self_rollout_supervision_horizon if self.config.self_rollout_supervision_horizon > 0 else self.config.self_rollout_steps
        max_horizon = min(max_horizon, self.config.self_rollout_steps)
        if max_horizon < 2:
            return set()
        if not self.config.self_rollout_hierarchical:
            return set(range(2, max_horizon + 1))
        horizons = {2}
        cursor = 4
        while cursor < max_horizon:
            horizons.add(cursor)
            cursor *= 2
        horizons.add(max_horizon)
        return {h for h in horizons if h <= max_horizon}

    @torch.no_grad()
    def _analyze_fixed_point(self, loop_history: List[torch.Tensor]) -> dict:
        """旧口径不动点 proxy：仅在近似自治时可解释为收缩分析，Loop LoRA/phase/time 打开时只能当轨迹摘要。"""
        if len(loop_history) < 3:
            return {}
        # δh 序列
        dh_list = []
        for i in range(1, len(loop_history)):
            dh = (loop_history[i] - loop_history[i - 1]).float()
            dh_list.append(dh)
        # 全局收缩率: L ≈ ||δh_{t+1}|| / ||δh_t||
        L_per_step = []
        for i in range(1, len(dh_list)):
            n1 = dh_list[i].norm().item()
            n0 = dh_list[i - 1].norm().item()
            L_per_step.append(n1 / max(n0, 1e-8))
        L_global = sum(L_per_step) / max(len(L_per_step), 1)
        # SVD 方向分解 (用 dh1 和 dh2)
        if len(dh_list) >= 2:
            dh1 = dh_list[0].reshape(-1, dh_list[0].shape[-1])  # [B*T, D]
            dh2 = dh_list[1].reshape(-1, dh_list[1].shape[-1])
            try:
                U, S1, V = torch.svd_lowrank(dh1, q=8)  # top-8 方向
                proj = dh2 @ V  # dh2 在 dh1 主方向上的投影
                S2 = proj.norm(dim=0)
                L_per_dir = (S2 / (S1 + 1e-8)).tolist()[:8]
                slow_dirs = sum(1 for l in L_per_dir if l > 0.5)
                dead_dirs = sum(1 for l in L_per_dir if l < 0.05)
            except Exception:
                L_per_dir = []
                slow_dirs = 0
                dead_dirs = 0
        else:
            L_per_dir = []
            slow_dirs = 0
            dead_dirs = 0
        return {
            "L_global": round(L_global, 4),
            "L_per_dir_top4": [round(x, 3) for x in L_per_dir[:4]],
            "slow_directions": slow_dirs,
            "dead_directions": dead_dirs,
        }

    def _sigreg_latent(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent.float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass
        else:
            return latent.new_zeros(())
        if x.shape[0] <= 1:
            return latent.new_zeros(())
        x = x - x.mean(dim=0, keepdim=True)
        x = x / (x.std(dim=0, unbiased=False, keepdim=True) + self.sigreg_eps)
        dirs = torch.randn(self.sigreg_num_slices, x.shape[-1], device=x.device, dtype=x.dtype)
        dirs = F.normalize(dirs, dim=-1)
        proj = x @ dirs.t()
        x_t = proj.unsqueeze(-1) * self.sigreg_t.view(1, 1, -1)
        cos_mean = torch.cos(x_t).mean(dim=0)
        sin_mean = torch.sin(x_t).mean(dim=0)
        err = (cos_mean - self.sigreg_phi0.view(1, -1)).square() + sin_mean.square()
        ep = (err * self.sigreg_weights.view(1, -1)).sum(dim=-1) * x.shape[0]
        return ep.mean()

    def forward(self, input_ids: torch.Tensor, disable_ct_injection: bool = False, measure_theory_probes: bool = False) -> Tuple[torch.Tensor, dict]:
        h = self.embedding(input_ids)
        # 重置 ct-LoRA 的 delta 缓存（每个新样本）
        for layer in self.reason_core.shared_layers:
            if hasattr(layer, '_ct_lora_prev'):
                layer._ct_lora_prev = None
        h, block_reprs, compression_diag = self.compression(h)
        # Phase 2 auxiliary loss: expose compression output (stripped of memory tokens)
        # so an external probe can give the compression zone its own gradient signal.
        n_comp_mem = 8  # 4 local + 4 global memory tokens prepended by CompressionZone
        compression_h = h[:, n_comp_mem:, :]  # (batch, seq_len, hidden) — aligned with input_ids
        batch_size = h.shape[0]
        h = torch.cat([self.reason_memory(batch_size), h], dim=1)
        streams = self.mhc.init_streams(h)
        loop_history: List[torch.Tensor] = []
        loop_h_grad: List[torch.Tensor] = []  # RLTT: intermediate h with gradient
        c_t_history: List[torch.Tensor] = []
        know_gap_history: List[torch.Tensor] = []
        uncertainty_history: List[torch.Tensor] = []
        slow_update_flags: List[bool] = []
        prev_h: Optional[torch.Tensor] = None
        slow_state = self.introspection_state_stream.init_slow_state(batch_size, h.device, h.dtype)
        # IS: init memory tokens if using memory/bixt mode
        _is_memory = self.memory_token_reader.init_memory(batch_size, h.device, h.dtype) if self.memory_token_reader is not None else None
        self_check_state = self.self_check_ring.init_state(batch_size, h.device, h.dtype) if self.self_check_ring is not None else None
        reasoning_state = self.reasoning_state_ring.init_state(batch_size, h.device, h.dtype) if self.reasoning_state_ring is not None else None
        c_t = slow_state["c_t"]
        know_gap = slow_state["know_gap"]
        uncertainty = slow_state.get("uncertainty", h.new_zeros((batch_size, 1)))
        self_check_score = h.new_full((batch_size, 1), 0.5)
        pred_delta_c = torch.zeros_like(c_t)
        target_delta_c = torch.zeros_like(c_t)
        self_jepa_terms: List[torch.Tensor] = []
        h_mask_loss_terms: List[torch.Tensor] = []
        # theory_probe_results: 四个主判据的最近一次测量值，loop_idx==0 时由 reason_core.measure_theory_probes 填充
        theory_probe_results: dict = {
            "rho_h_frozen": None,
            "rho_c_drift": None,
            "eta_moving_fp": None,
            "probe_delta_h_norm": None,
            "probe_delta_c_norm": None,
        }
        residual_reg_terms: List[torch.Tensor] = []
        sigreg_delta_terms: List[torch.Tensor] = []
        sigreg_rollout_terms: List[torch.Tensor] = []
        sigreg_ct_terms: List[torch.Tensor] = []
        ct_norm_terms: List[torch.Tensor] = []
        ct_grad_history: List[torch.Tensor] = []  # 带梯度的 c_t，用于 CtWorldJEPA
        self_check_loss_terms: List[torch.Tensor] = []
        rollout_loss_terms: List[torch.Tensor] = []
        delta_h_history: List[torch.Tensor] = []
        self_error_history: List[torch.Tensor] = []
        rollout_error_history: List[torch.Tensor] = []
        world_error_history: List[torch.Tensor] = []
        exit_score_history: List[torch.Tensor] = []
        exit_logit_history: List[torch.Tensor] = []
        sampled_exit_score_history: List[torch.Tensor] = []
        two_step_improvement_history: List[torch.Tensor] = []
        self_check_score_history: List[torch.Tensor] = []
        jepa_crystal_history: List[torch.Tensor] = []
        predicted_gain_history: List[torch.Tensor] = []
        predicted_gain2_history: List[torch.Tensor] = []
        progress_next_history: List[torch.Tensor] = []
        progress_trend_history: List[torch.Tensor] = []
        progress_plateau_history: List[torch.Tensor] = []
        world_surprise_history: List[torch.Tensor] = []
        world_summary_history: List[torch.Tensor] = []
        exit_score_preclamp_nonfinite_history: List[torch.Tensor] = []
        exit_score_postfix_clamped_ratio_history: List[torch.Tensor] = []
        bernoulli_invalid_prevented_history: List[torch.Tensor] = []
        nan_to_num_trigger_history: List[torch.Tensor] = []
        dynamics_modulation_history: List[dict] = []
        dynamics_modulation_traces: dict[str, List[torch.Tensor]] = {}
        math_lane_score_history: List[torch.Tensor] = []
        math_summary_gate_history: List[torch.Tensor] = []
        r_t_history: List[torch.Tensor] = []
        r_t_trust_history: List[torch.Tensor] = []
        r_t_switch_history: List[torch.Tensor] = []
        # ── 动力学监控列表（per-loop 指标 + 赫布项）──
        per_loop_delta_h: List[float] = []
        per_loop_ct_change: List[float] = []
        per_loop_jepa_err: List[float] = []
        per_loop_dt_inject: List[float] = []  # dt 注入 norm 诊断
        _mamba_mid_history: List[torch.Tensor] = []  # Mamba layer1 输出方向历史
        _mamba_out_history: List[torch.Tensor] = []  # Mamba layer2 输出方向历史
        nm_gain_history: List[float] = []
        nm_hebb_norm_history: List[float] = []
        nm_hebb_write_history: List[float] = []
        nm_surprise_history: List[float] = []
        ct_norm_raw_history: List[float] = []
        ct_norm_after_writer_history: List[float] = []
        meta_last_norm_history: List[float] = []
        c_t_head_out_norm_history: List[float] = []
        if self.reasoning_state_ring is not None:
            reasoning_state = self.reasoning_state_ring.bootstrap(h.mean(dim=1))
            r_t_history.append(reasoning_state["state"].detach())
            r_t_trust_history.append(reasoning_state["trust"].detach().mean())
        current_self_error = h.new_tensor(1.0)
        current_rollout_error = h.new_tensor(1.0)
        coupling_terms: List[torch.Tensor] = []
        progress_shape_terms: List[torch.Tensor] = []
        local_consistency_terms: List[torch.Tensor] = []
        trajectory_health_terms: List[torch.Tensor] = []
        pending_rollout_preds: List[Tuple[int, torch.Tensor, int]] = []
        pending_progress_preds: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []
        slow_step_idx = -1
        executed_loops = 0
        recent_joint_gains: List[torch.Tensor] = []
        recent_delta_improves: List[torch.Tensor] = []
        pred_delta_history: List[torch.Tensor] = []
        latest_progress_next = h.new_zeros(())
        latest_progress_trend = h.new_zeros(())
        latest_progress_plateau = h.new_zeros(())
        trajectory_health_history: List[torch.Tensor] = []
        prev_self_error_for_gain = h.new_tensor(1.0)
        prev_world_error_for_gain = h.new_tensor(1.0)
        prev_world_summary: Optional[torch.Tensor] = None
        prev_tier_choice: Optional[torch.Tensor] = None
        if "math_lane_score" in compression_diag:
            math_lane_score_history.append(compression_diag["math_lane_score"].detach())
        hierarchical_horizons = self._hierarchical_rollout_horizons()
        h_pre_reason = h  # 保存 reasoning loop 之前的 h，用于 world_jepa_reason_only

        _mor_balance_terms: list[torch.Tensor] = []  # True MoR balance loss accumulator
        for loop_idx in range(self.config.reason_active_loops):
            _loop_ct_change = 0.0  # 每轮重置，仅 slow_update 轮会覆盖
            # ── True MoR: token-level depth routing (loop_idx > 0) ──
            _mor_h_frozen: Optional[torch.Tensor] = None
            _mor_continue_mask: Optional[torch.Tensor] = None
            if self.token_depth_router is not None and loop_idx > 0:
                _mor_continue_mask = self.token_depth_router(h, loop_idx)  # [B, T]
                _mor_h_frozen = h if self.config.mor_grad_through_frozen else h.detach()
                _mor_balance_terms.append(_mor_continue_mask.mean())
            # IS6: token-aware c_t injection (before reason_core)
            if self._is_inject_mode == "token_aware" and hasattr(self, "token_aware_inject"):
                h = self.token_aware_inject(h, c_t)
            current_r_t = reasoning_state["state"] if reasoning_state is not None else None
            current_r_trust = reasoning_state["trust"] if reasoning_state is not None else None
            recent_block_repr = block_reprs[-1].mean(dim=1) if block_reprs else h.new_zeros((batch_size, h.shape[-1]))
            if loop_history:
                loop_history_summary = torch.stack([item.mean(dim=1) for item in loop_history[-3:]], dim=0).mean(dim=0)
            else:
                loop_history_summary = h.new_zeros((batch_size, h.shape[-1]))
            recent_world_summary = world_summary_history[-1] if world_summary_history else h.new_zeros((batch_size, self.config.world_dim))
            recent_improves = torch.stack(recent_delta_improves[-4:]).float() if recent_delta_improves else h.new_zeros((1,), dtype=torch.float32)
            hard_loop_var_proxy = float(recent_improves.var(unbiased=False).item()) if recent_improves.numel() > 1 else 0.0
            modulation_context = {
                "recent_block_repr": recent_block_repr.detach(),
                "world_summary": recent_world_summary.detach(),
                "loop_history_summary": loop_history_summary.detach(),
                "progress_next": latest_progress_next.detach(),
                "progress_trend": latest_progress_trend.detach(),
                "progress_plateau": latest_progress_plateau.detach(),
                "loop_ratio": float(loop_idx + 1) / float(max(1, self.config.reason_active_loops)),
                "hard_loop_var_proxy": hard_loop_var_proxy,
                "prev_tier_choice": prev_tier_choice.detach() if isinstance(prev_tier_choice, torch.Tensor) else None,
            }
            switch_trace: dict[str, object] = {"value": None, "mod_stats": {}}
            streams, h = self.mhc(
                streams,
                lambda x, rt=current_r_t, rg=current_r_trust: (
                    lambda out: (
                        switch_trace.__setitem__("value", out[1])
                        or switch_trace.__setitem__("mod_stats", out[2])
                        or out[0]
                    )
                )(
                    self.reason_core(
                        x,
                        c_t,
                        r_t=rt,
                        r_trust=rg,
                        r_t_mode=self.config.r_t_mode,
                        disable_ct_injection=disable_ct_injection,
                        modulation_context=modulation_context,
                        use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                        loop_idx=loop_idx,
                    )
                ),
            )
            if switch_trace["value"] is not None:
                r_t_switch_history.append(cast(torch.Tensor, switch_trace["value"]).detach().mean())
            loop_mod_stats = cast(dict, switch_trace["mod_stats"])
            if "_tier_choice_vector" in loop_mod_stats:
                maybe_tier_choice = loop_mod_stats.pop("_tier_choice_vector")
                if isinstance(maybe_tier_choice, torch.Tensor):
                    prev_tier_choice = maybe_tier_choice.detach().to(device=h.device)
            if loop_mod_stats:
                dynamics_modulation_history.append(loop_mod_stats)
                for key, value in loop_mod_stats.items():
                    if value is None:
                        continue
                    tensor = value if isinstance(value, torch.Tensor) else h.new_tensor(float(value))
                    tensor = tensor.float().mean()
                    dynamics_modulation_traces.setdefault(key, []).append(tensor.detach())
            h = self.unified_attnres(h, loop_history, block_reprs, loop_idx=loop_idx)
            # PC: predictive coding error correction
            if self.pc_corrector is not None and loop_idx > 0:
                h, _pc_err = self.pc_corrector(h, c_t, alpha=self._pc_alpha)
            did_slow_update = (loop_idx % self.config.slow_k == 0)
            if did_slow_update:
                slow_step_idx += 1
                prev_c_t = c_t
                current_loop_progress = h.new_full(
                    (batch_size, 1),
                    float(loop_idx + 1) / float(max(1, self.config.reason_active_loops)),
                )
                # IS: prepare introspection input based on mode
                _is_meta_override = None
                if self._is_input_mode == "memory" and _is_memory is not None:
                    # δh 注入: 让 introspection 感知 h 的循环间变化方向
                    _dh_scale = self.config.delta_h_scale
                    if _dh_scale > 0.0 and prev_h is not None:
                        _delta_h = h - prev_h
                        if self.config.delta_h_normalize:
                            _delta_h = _delta_h / (_delta_h.norm(dim=-1, keepdim=True) + 1e-8)
                        _h_enriched = h + _dh_scale * _delta_h
                    else:
                        _h_enriched = h
                    _is_memory, _is_meta_override = self.memory_token_reader(_is_memory, _h_enriched)
                elif self._is_input_mode == "chunked_memory" and _is_memory is not None:
                    # chunk h first, then memory tokens read from chunks (not raw h)
                    _n_chunks = 8
                    _chunk_sz = max(1, h.shape[1] // _n_chunks)
                    _chunks = [h[:, i*_chunk_sz:(i+1)*_chunk_sz, :].mean(dim=1) for i in range(_n_chunks)]
                    _chunked = torch.stack(_chunks, dim=1)  # [B, 8, D]
                    _is_memory, _is_meta_override = self.memory_token_reader(_is_memory, _chunked)
                elif self._is_input_mode == "chunked":
                    _n_chunks = 8
                    _chunk_sz = max(1, h.shape[1] // _n_chunks)
                    _chunks = [h[:, i*_chunk_sz:(i+1)*_chunk_sz, :].mean(dim=1) for i in range(_n_chunks)]
                    _chunked = torch.stack(_chunks, dim=1)  # [B, 8, D]
                    _is_meta_override = self.introspection_state_stream.meta_input(
                        torch.cat([_chunked, _chunked], dim=-1)  # fake [h_pool, last] by repeating
                    )  # [B, 8, meta_dim]
                # IS: BiXT mode — bidirectional cross-attention before introspection
                if self._is_inject_mode in ("bixt", "bixt_cmda") and hasattr(self, "bixt_cross_attn") and _is_memory is not None:
                    if _is_meta_override is None:
                        # bixt needs memory in meta_dim space
                        _, _is_meta_override = self.memory_token_reader(_is_memory, h)
                    _is_meta_override, h = self.bixt_cross_attn(_is_meta_override, h)
                # IS: CMDA mode — bidirectional channel modulation
                if self._is_inject_mode in ("cmda", "bixt_cmda") and hasattr(self, "cmda_modulation"):
                    h, _cmda_meta = self.cmda_modulation(h, c_t)
                    if _is_meta_override is None:
                        _is_meta_override = _cmda_meta.unsqueeze(1)  # [B, 1, meta_dim]
                know_gap, next_c_t, slow_state = self.introspection_state_stream(
                    h,
                    block_reprs,
                    slow_state,
                    loop_progress=current_loop_progress,
                    loop_index=loop_idx,
                    meta_override=_is_meta_override,
                )
                # theory probes: 在 loop_idx==0 且被请求时测一次（避免每轮都跑 3x 额外 forward）
                if measure_theory_probes and loop_idx == 0 and self.training is False:
                    # 只在 eval 模式测，避免污染训练（训练时 model.training=True）
                    pass  # 训练模式跳过，靠外部 eval pass 驱动
                elif measure_theory_probes and loop_idx == 0:
                    _probe_out = self.reason_core.measure_theory_probes(
                        h.detach(), c_t.detach(), next_c_t.detach(), loop_idx,
                    )
                    theory_probe_results = _probe_out
                # Mamba 内部方向诊断：收集每轮的 meta_last_1 方向
                if hasattr(self.introspection_state_stream, '_diag_meta_last_1'):
                    _mamba_mid_history.append(self.introspection_state_stream._diag_meta_last_1.clone())
                    _mamba_out_history.append(self.introspection_state_stream._diag_meta_last.clone())
                ct_norm_raw_history.append(float(next_c_t.detach().float().norm(dim=-1).mean().item()))
                meta_last_norm_history.append(float(getattr(self.introspection_state_stream, "_diag_meta_last_norm", h.new_zeros(())).item()))
                c_t_head_out_norm_history.append(float(getattr(self.introspection_state_stream, "_diag_ct_head_out_norm", h.new_zeros(())).item()))
                delta_h = h.mean(dim=1).detach() if prev_h is None else (h - prev_h).mean(dim=1).detach()
                if not self.config.disable_self_jepa:
                    rollout_delta_preds, rollout_state_preds = self.self_jepa_residual_predictor.rollout(
                        c_t.detach(),  # Phase 3+: stop-gradient，JEPA 不污染主干梯度流
                        delta_h,
                        steps=self.config.self_rollout_steps,
                        loop_progress=current_loop_progress,
                        loop_index=loop_idx,
                    )
                    pred_delta_c = rollout_delta_preds[0]
                    pred_c_next = rollout_state_preds[0]
                    target_delta_c = (next_c_t - prev_c_t).detach()
                    self_jepa_terms.append(
                        1.0 - F.cosine_similarity(
                            F.normalize(pred_delta_c, dim=-1),
                            F.normalize(target_delta_c, dim=-1),
                            dim=-1,
                        ).mean()
                    )
                    residual_reg_terms.append(pred_delta_c.norm(dim=-1).mean() * self.config.self_jepa_residual_reg)
                    if self.enable_sigreg_delta:
                        sigreg_delta_terms.append(self._sigreg_latent(pred_delta_c))
                    if self.enable_sigreg_rollout and rollout_state_preds:
                        rollout_sigreg_slice = rollout_state_preds[: min(3, len(rollout_state_preds))]
                        sigreg_rollout_terms.append(
                            torch.stack([self._sigreg_latent(pred_state) for pred_state in rollout_sigreg_slice]).mean()
                        )
                    if self.config.enable_world_jepa and self.config.self_world_coupling_weight > 0.0:
                        world_summary_for_coupling = self.world_latent_jepa.summarize(h)
                        if prev_world_summary is not None:
                            delta_world = world_summary_for_coupling - prev_world_summary
                            projected_world_delta = self.self_world_coupler(delta_world)
                            coupling_terms.append(
                                1.0
                                - F.cosine_similarity(
                                    F.normalize(pred_delta_c, dim=-1),
                                    F.normalize(projected_world_delta, dim=-1),
                                    dim=-1,
                                ).mean()
                            )
                        prev_world_summary = world_summary_for_coupling.detach()
                    current_self_error = self_jepa_terms[-1].detach()
                    matured_rollouts = [
                        (pred, horizon)
                        for target_idx, pred, horizon in pending_rollout_preds
                        if target_idx == slow_step_idx and horizon in hierarchical_horizons
                    ]
                    if matured_rollouts:
                        rollout_errs = []
                        for pred_state, horizon in matured_rollouts:
                            if self.config.self_rollout_weighting_mode == "near3":
                                if horizon == 2:
                                    horizon_weight = 1.0
                                elif horizon == 3:
                                    horizon_weight = 0.5
                                elif horizon == 4:
                                    horizon_weight = 0.2
                                else:
                                    horizon_weight = 0.0
                            else:
                                horizon_weight = 1.0 if horizon <= 2 else (0.5 if horizon <= 4 else 0.25)
                            if horizon_weight == 0.0:
                                continue
                            rollout_errs.append(
                                horizon_weight
                                * (
                                    1.0
                                    - F.cosine_similarity(
                                        F.normalize(pred_state, dim=-1),
                                        F.normalize(next_c_t.detach(), dim=-1),
                                        dim=-1,
                                    )
                                )
                            )
                        if rollout_errs:
                            rollout_err = torch.stack(rollout_errs).mean()
                            rollout_loss_terms.append(rollout_err)
                            current_rollout_error = rollout_err.detach()
                    current_self_improve_scalar = (prev_self_error_for_gain - current_self_error).detach()
                    matured_progress = [
                        (pred_next, pred_trend, pred_plateau, prev_improve, pred_backtrack)
                        for target_idx, pred_next, pred_trend, pred_plateau, prev_improve, pred_backtrack in pending_progress_preds
                        if target_idx == slow_step_idx
                    ]
                    if matured_progress:
                        next_improve_target = h.new_full((batch_size,), float(current_self_improve_scalar.item()))
                        for pred_next, pred_trend, pred_plateau, prev_improve, pred_backtrack in matured_progress:
                            if self.config.self_progress_shape_weight > 0.0:
                                progress_shape_terms.append(
                                    self.config.self_progress_shape_weight
                                    * F.smooth_l1_loss(pred_next, next_improve_target)
                                )
                            if self.config.self_progress_trend_weight > 0.0:
                                trend_target = h.new_full((batch_size,), float((current_self_improve_scalar - prev_improve).item()))
                                progress_shape_terms.append(
                                    self.config.self_progress_trend_weight
                                    * F.smooth_l1_loss(pred_trend, trend_target)
                                )
                            if self.config.self_progress_plateau_weight > 0.0:
                                plateau_target = h.new_full((batch_size,), 1.0 if float(current_self_improve_scalar.item()) < self.config.self_plateau_margin else 0.0)
                                progress_shape_terms.append(
                                    self.config.self_progress_plateau_weight
                                    * F.binary_cross_entropy_with_logits(pred_plateau, plateau_target)
                                )
                            if self.config.enable_backtrack_aware_progress and pred_backtrack is not None:
                                backtrack_target = h.new_full(
                                    (batch_size,),
                                    1.0 if float(current_self_improve_scalar.item()) < 0.0 else 0.0,
                                )
                                progress_shape_terms.append(
                                    0.02 * F.binary_cross_entropy_with_logits(pred_backtrack, backtrack_target)
                                )
                    pending_rollout_preds = [
                        (target_idx, pred, horizon)
                        for target_idx, pred, horizon in pending_rollout_preds
                        if target_idx > slow_step_idx
                    ]
                    for horizon, pred_state in enumerate(rollout_state_preds[1:], start=1):
                        pending_rollout_preds.append((slow_step_idx + horizon, pred_state, horizon + 1))
                else:
                    # disable_self_jepa=True: skip rollout entirely, provide neutral defaults
                    pred_delta_c = torch.zeros_like(c_t)
                    current_self_improve_scalar = h.new_zeros(())
                    pending_rollout_preds = []
                # Masked h prediction → 混入赫布 surprise
                _jepa_err_for_hebb = current_self_error if current_self_error.numel() > 0 else None
                if self.h_mask_predictor is not None and self.training and self.config.h_mask_loss_mode != "off":
                    _h_mean = h.mean(dim=1).detach()  # [B, hidden_size]
                    # 【2026-04-12 实验记录】：V9 尝试 c_t.detach() 反而使 rho_h 更早爆炸
                    # (V9 step 500-1000 rho_h p50=1.77 vs V8 的 0.63)，说明 h_mask
                    # 对 c_t 的梯度是 *稳定器* 而非破坏者。切断后 c_t 源头层失去约束。
                    # 真正的问题是 h_mask_loss_weight 过大 → V10 改为降低 weight 0.1→0.03。
                    _h_pred = self.h_mask_predictor(c_t)  # [B, hidden_size]
                    # 随机 mask 25% 的特征维度
                    _h_mask = (torch.rand(_h_mean.shape[-1], device=h.device) < self._h_mask_ratio).float()
                    _mode = self.config.h_mask_loss_mode
                    # h_mask_loss_mode:
                    #   mse          = MSE 预测 h 的数值（旧实现，在长训练里会爆炸因为 h.mean 的范数在漂）
                    #   cosine       = 余弦方向预测：c_t 学 h 的"思考方向"，loss 有界 [0, 2]（推荐）
                    #   surprise_only = 不反传 loss，只用误差做 surprise
                    #   off          = 完全关闭
                    if _mode == "cosine":
                        # 方向预测：归一化 pred 和 target，mask 后算 cosine 距离
                        # 等价于让 c_t 在 h 的 mask 子空间上和 h.mean 方向一致
                        _h_masked_target = _h_mean * _h_mask
                        _h_masked_pred = _h_pred * _h_mask
                        _t_norm = _h_masked_target.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        _p_norm = _h_masked_pred.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        _cos_sim = (_h_masked_pred.float() * _h_masked_target.float()).sum(dim=-1, keepdim=True) / (_p_norm * _t_norm)
                        # err ∈ [0, 2]，完美对齐=0, 反向=2
                        _h_mask_err = (1.0 - _cos_sim).to(_h_mean.dtype)
                    else:
                        # mse 路径（保留兼容，但长训练里目标漂移会让 loss 爆炸）
                        _h_mask_err = ((_h_pred - _h_mean) * _h_mask).pow(2).sum(dim=-1, keepdim=True) / _h_mask.sum().clamp(min=1)
                    if _mode in ("mse", "cosine"):
                        h_mask_loss_terms.append(_h_mask_err.mean())
                    _h_mask_err = _h_mask_err.detach().clamp(0, 2)  # surprise 用 detach 版（cosine 上界 2）
                    if _jepa_err_for_hebb is not None:
                        _w = self.config.h_mask_surprise_weight
                        _jepa_err_for_hebb = (1 - _w) * _jepa_err_for_hebb + _w * _h_mask_err
                    else:
                        _jepa_err_for_hebb = _h_mask_err
                # NM: neuromodulated c_t write
                if self.neuromod_ct_writer is not None and loop_idx > 0:
                    c_t, _nm_aux = self.neuromod_ct_writer(
                        next_c_t, prev_c_t, delta_h, self_check_score,
                        jepa_error=_jepa_err_for_hebb,
                    )
                    # 赫布项监控：收集 neuromod_gain 和 hebb_norm
                    nm_gain_history.append(_nm_aux["neuromod_gain"].item())
                    nm_hebb_norm_history.append(_nm_aux["hebb_norm"].item())
                    nm_hebb_write_history.append(_nm_aux["hebb_write"].item())
                    nm_surprise_history.append(_nm_aux["surprise_mean"].item())
                    ct_norm_after_writer_history.append(_nm_aux["ct_norm_after_writer"].item())
                else:
                    c_t = next_c_t
                    ct_norm_after_writer_history.append(float(c_t.detach().float().norm(dim=-1).mean().item()))
                # c_t momentum: EMA 式慢更新
                _ct_mom = self.config.ct_momentum
                if _ct_mom > 0.0 and loop_idx > 0:
                    c_t = _ct_mom * prev_c_t + (1.0 - _ct_mom) * c_t
                # dtype 安全网: 确保 c_t 始终 bf16（RMSNorm scale 可能提升到 float32）
                if c_t.dtype != torch.bfloat16:
                    c_t = c_t.to(torch.bfloat16)
                # NaN 探针：循环内每步检查
                if not torch.isfinite(h).all() or not torch.isfinite(c_t).all():
                    print(f"[NaN PROBE] loop={loop_idx} h_nan={not torch.isfinite(h).all()} ct_nan={not torch.isfinite(c_t).all()} h_norm={h.norm().item():.1f} ct_norm={c_t.norm().item():.1f}", flush=True)
                # 动力学监控：slow_update 轮采集真实 ct_change (c_t 已经过赫布 + RMSNorm)
                _loop_ct_change = (c_t - prev_c_t).detach().norm(dim=-1).mean().item()
                if self.config.ct_norm_penalty_weight > 0.0:
                    ct_norm_terms.append(c_t.norm(dim=-1).mean())
                if self.enable_sigreg_ct:
                    sigreg_ct_terms.append(self._sigreg_latent(c_t))
                know_gap = slow_state["know_gap"]
                uncertainty = slow_state.get("uncertainty", uncertainty)
                local_progress = current_loop_progress
                recent_self_improve_vec = h.new_full(
                    (batch_size, 1),
                    float(current_self_improve_scalar.item()),
                )
                recent_rollout_improve_vec = h.new_full(
                    (batch_size, 1),
                    float(max((current_self_error - current_rollout_error).item(), 0.0)),
                )
                if (
                    self.config.self_progress_shape_weight > 0.0
                    or self.config.self_progress_trend_weight > 0.0
                    or self.config.self_progress_plateau_weight > 0.0
                ):
                    _aux_scale = self.config.ct_grad_scale_aux if self.config.ct_grad_scale_aux is not None else self.config.ct_grad_scale
                    c_t_for_aux = grad_scale(c_t, _aux_scale) if _aux_scale != 1.0 else c_t
                    progress_pred = self.self_jepa_progress_shape_head(
                        c_t_for_aux,
                        delta_h,
                        current_loop_progress,
                        recent_self_improve_vec,
                        recent_rollout_improve_vec,
                    )
                    pending_progress_preds.append(
                        (
                            slow_step_idx + 1,
                            progress_pred["pred_next_improve"],
                            progress_pred["pred_trend"],
                            progress_pred["pred_plateau_logit"],
                            current_self_improve_scalar.detach(),
                            progress_pred.get("pred_backtrack_logit"),
                        )
                    )
                    latest_progress_next = progress_pred["pred_next_improve"].detach().mean()
                    latest_progress_trend = progress_pred["pred_trend"].detach().mean()
                    latest_progress_plateau = torch.sigmoid(progress_pred["pred_plateau_logit"].detach()).mean()
                    if self.trajectory_health_probe is not None:
                        health_logit = self.trajectory_health_probe(
                            c_t_for_aux,
                            delta_h,
                            current_loop_progress,
                            recent_self_improve_vec,
                            recent_rollout_improve_vec,
                        )
                        health_target = h.new_full(
                            (batch_size,),
                            1.0 if (float(current_self_improve_scalar.item()) > 0.0 and float(current_rollout_error.item()) <= float(current_self_error.item()) + 0.05) else 0.0,
                        )
                        trajectory_health_terms.append(
                            self.config.trajectory_health_weight
                            * F.binary_cross_entropy_with_logits(health_logit, health_target)
                        )
                        trajectory_health_history.append(torch.sigmoid(health_logit.detach()).mean())
                if self.config.self_local_delta_consistency_weight > 0.0 and pred_delta_history:
                    prev_pred_delta = pred_delta_history[-1]
                    local_consistency_terms.append(
                        self.config.self_local_delta_consistency_weight
                        * (1.0 - F.cosine_similarity(
                            F.normalize(pred_delta_c, dim=-1),
                            F.normalize(prev_pred_delta, dim=-1),
                            dim=-1,
                        ).mean())
                    )
                if self.config.self_local_curvature_weight > 0.0 and len(pred_delta_history) > 1:
                    second_diff = pred_delta_c - 2.0 * pred_delta_history[-1] + pred_delta_history[-2]
                    local_consistency_terms.append(
                        self.config.self_local_curvature_weight
                        * second_diff.norm(dim=-1).mean()
                    )
                pred_delta_history.append(pred_delta_c.detach())
                did_self_check_update = (loop_idx % self.config.self_check_k == 0)
                if self.self_check_ring is not None and self_check_state is not None and did_self_check_update:
                    # Phase 4+: stop-gradient，self_check ring 不回传梯度到主干
                    self_check_state = self.self_check_ring(
                        c_t.detach(), delta_h, self_check_state
                    )
                    self_check_score = self_check_state["score"]
                    # self_check_loss：训练 ring score 预测推理改善方向
                    # 软标签：sigmoid(improve_scalar)，improve>0→1，improve<0→0
                    if self.self_check_loss_weight > 0.0:
                        improve_target = torch.sigmoid(
                            current_self_improve_scalar.detach()
                            * h.new_tensor(5.0)  # 缩放使 sigmoid 更灵敏
                        ).expand_as(self_check_score)
                        self_check_loss_terms.append(
                            F.binary_cross_entropy(
                                self_check_score, improve_target
                            )
                        )
                if self.reasoning_state_ring is not None and reasoning_state is not None:
                    reasoning_state = self.reasoning_state_ring(
                        h.mean(dim=1),
                        delta_h,
                        c_t,
                        local_progress,
                        recent_self_improve_vec,
                        recent_rollout_improve_vec,
                        reasoning_state,
                    )
                    r_t_history.append(reasoning_state["state"].detach())
                    r_t_trust_history.append(reasoning_state["trust"].detach().mean())
            h = self.loop_norm(h)
            # ── True MoR: blend frozen tokens back (with c_t injection) ──
            if _mor_continue_mask is not None and _mor_h_frozen is not None:
                _h_dtype = h.dtype
                _mor_mask_3d = _mor_continue_mask.unsqueeze(-1).to(_h_dtype)  # [B, T, 1]
                # frozen token 也接收 c_t 自省信号，只是不做 reasoning
                _mor_bias = self.reason_core.ct_injection.get_bias(c_t)
                _mor_h_frozen_ct = _mor_h_frozen + _mor_bias.unsqueeze(1).to(_h_dtype)
                h = (_mor_mask_3d * h + (1.0 - _mor_mask_3d) * _mor_h_frozen_ct).to(_h_dtype)
            if self.config.enable_world_jepa:
                world_probe = self.world_latent_jepa.probe_stats(h)
                current_world_error = world_probe["world_jepa_loss"].detach()
                world_surprise = world_probe["world_surprise"].detach()
                world_summary = self.world_latent_jepa.summarize(h).detach()
            else:
                current_world_error = h.new_zeros(())
                world_surprise = h.new_zeros(())
                world_summary = h.new_zeros((batch_size, self.config.world_dim))
            current_delta_improve = (
                (prev_self_error_for_gain - current_self_error).detach()
                + (prev_world_error_for_gain - current_world_error).detach()
            )
            prev_self_error_for_gain = current_self_error.detach()
            prev_world_error_for_gain = current_world_error.detach()
            loop_history.append(h.detach())
            if self.config.loop_lm_loss_weight > 0.0 or self.config.shortcut_consistency_weight > 0.0:
                # RLTT 方案 A: 中间轮全部 detach，只有最后退出轮保留梯度
                # 中间轮的 detached h 仍提供 loss 监督信号（训练 lm_head），但不反传到 shared_layers
                loop_h_grad.append(h.detach())
            c_t_history.append(c_t.detach())
            if self.ct_world_jepa is not None:
                ct_grad_history.append(c_t)
            know_gap_history.append(know_gap.detach())
            uncertainty_history.append(uncertainty.detach())
            slow_update_flags.append(did_slow_update)
            executed_loops = loop_idx + 1
            # ES: compute enhanced exit signals
            _es_entropy_proxy = None
            _es_confidence_proxy = None
            _es_token_sensitivity = None
            _es_ct_curvature = None
            if self.exit_quality_probe is not None:
                with torch.no_grad():
                    _eq = self.exit_quality_probe(h, prev_h)
                    if self._es_entropy:
                        _es_entropy_proxy = _eq["entropy_proxy"].mean()
                    if self._es_confidence:
                        _es_confidence_proxy = _eq["confidence_proxy"].mean()
                    if self._es_token_sens:
                        _es_token_sensitivity = _eq["token_sensitivity"].mean()
            if self._es_ct_curv and len(c_t_history) >= 2:
                with torch.no_grad():
                    _dc1 = c_t - c_t_history[-1]
                    _dc2 = c_t_history[-1] - c_t_history[-2] if len(c_t_history) >= 2 else _dc1
                    _es_ct_curvature = F.cosine_similarity(_dc1, _dc2, dim=-1).mean()
            exit_stats = self.exit_controller(
                prev_h,
                h,
                loop_idx,
                current_self_error,
                current_rollout_error,
                current_world_error,
                self_check_score=self_check_score.mean(),
                prev_delta_h=delta_h_history[-1] if delta_h_history else None,
                loop_progress=h.new_full((), float(loop_idx + 1) / float(max(1, self.config.reason_active_loops))),
                remaining_budget_ratio=h.new_full(
                    (),
                    float(max(self.config.reason_active_loops - (loop_idx + 1), 0)) / float(max(1, self.config.reason_active_loops)),
                ),
                recent_gain_1=recent_joint_gains[-1] if recent_joint_gains else h.new_zeros(()),
                recent_gain_2=recent_joint_gains[-2] if len(recent_joint_gains) > 1 else h.new_zeros(()),
                recent_delta_improve_1=recent_delta_improves[-1] if recent_delta_improves else h.new_zeros(()),
                recent_delta_improve_2=recent_delta_improves[-2] if len(recent_delta_improves) > 1 else h.new_zeros(()),
                reason_local_signal=(reasoning_state["trust"].mean() if reasoning_state is not None else None),
                uncertainty_score=uncertainty.mean(),
                progress_next_improve=latest_progress_next,
                progress_trend=latest_progress_trend,
                progress_plateau_logit=torch.logit(latest_progress_plateau.clamp(1e-4, 1 - 1e-4)),
                ct_drift=(c_t - c_t_history[-1]).norm(dim=-1).mean() if c_t_history else None,
                mor_continue_ratio=_mor_continue_mask.mean().detach() if _mor_continue_mask is not None else None,
                entropy_proxy=_es_entropy_proxy,
                confidence_proxy=_es_confidence_proxy,
                token_sensitivity=_es_token_sensitivity,
                ct_curvature=_es_ct_curvature,
            )
            delta_h_history.append(exit_stats["delta_h"].detach())
            # 动力学监控：per-loop 指标收集
            per_loop_delta_h.append(exit_stats["delta_h"].detach().item())
            # ct_change：使用 slow_update 块内采集的真实值（非 slow_update 轮为 0.0）
            per_loop_ct_change.append(_loop_ct_change)
            per_loop_jepa_err.append(current_self_error.detach().item())
            # 诊断：收集 dt_inject ratio（第一层 mamba 的 dt 注入强度/基线比）
            _mamba_layer = self.reason_core.shared_layers[0].mamba
            if hasattr(_mamba_layer, '_last_dt_ratio'):
                per_loop_dt_inject.append(_mamba_layer._last_dt_ratio)
            self_error_history.append(current_self_error.detach())
            rollout_error_history.append(current_rollout_error.detach())
            world_error_history.append(current_world_error.detach())
            exit_logit_history.append(exit_stats["exit_logit"])
            exit_score_history.append(exit_stats["exit_score"])
            sampled_exit_score_history.append(exit_stats["sampled_exit_score"].detach())
            two_step_improvement_history.append(exit_stats["two_step_improvement"].detach())
            self_check_score_history.append(exit_stats["self_check_score"].detach())
            jepa_crystal_history.append(exit_stats["jepa_crystal_signal"].detach())
            predicted_gain_history.append(exit_stats["predicted_gain"].detach())
            predicted_gain2_history.append(exit_stats["predicted_gain2"].detach())
            progress_next_history.append(exit_stats["progress_next_signal"].detach())
            progress_trend_history.append(exit_stats["progress_trend_signal"].detach())
            progress_plateau_history.append(exit_stats["progress_plateau_signal"].detach())
            exit_score_preclamp_nonfinite_history.append(exit_stats["exit_score_preclamp_nonfinite_count"].detach())
            exit_score_postfix_clamped_ratio_history.append(exit_stats["exit_score_postfix_clamped_ratio"].detach())
            bernoulli_invalid_prevented_history.append(exit_stats["bernoulli_invalid_prevented_count"].detach())
            nan_to_num_trigger_history.append(exit_stats["nan_to_num_trigger_count"].detach())
            world_surprise_history.append(world_surprise)
            world_summary_history.append(world_summary)
            recent_joint_gains.append(exit_stats["predicted_gain"].detach())
            recent_delta_improves.append(current_delta_improve.detach())
            prev_h = h
            # 渐进循环热身: 前 N 步按 step%max_loops 强制对应深度
            _prog_warmup = self.config.exit_progressive_warmup
            if _prog_warmup > 0 and self.training and self.exit_controller._global_step <= _prog_warmup:
                _forced_depth = (self.exit_controller._global_step % self.config.reason_active_loops) + 1
                if loop_idx + 1 < _forced_depth:
                    continue  # 还没到强制深度，跳过退出判断
            if exit_stats["should_exit"]:
                break

        # RLTT 方案 A: 循环结束后，把最后一份 h 替换为带梯度版本
        if loop_h_grad:
            loop_h_grad[-1] = h  # 只有最终退出轮的 h 保留完整梯度

        # ── Coconut: continuous thought re-injection ──
        # After reasoning loop exits, project c_t into a thought token,
        # prepend it to h, and run shared_layers one more time per round.
        if self._coconut and self.training:
            for _coco_round in range(self._coconut_rounds):
                thought_tok = self.coconut_proj(c_t).unsqueeze(1)  # [B, 1, D]
                h_with_thought = torch.cat([thought_tok, h], dim=1)  # [B, T+1, D]
                for layer in self.reason_core.shared_layers:
                    h_with_thought = layer(h_with_thought, c_t=c_t, attn_bias=None,
                        use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                        loop_idx=executed_loops + _coco_round)
                h = h_with_thought[:, 1:, :]  # strip thought token, keep original seq len

        if self.config.enable_world_jepa and self.config.world_jepa_reason_only:
            # stop-grad compress 贡献，让 World JEPA 梯度只走 reasoning 区
            # 前向值 = h（不变），反向梯度不流经 h_pre_reason
            h_for_world = self.final_norm(h_pre_reason.detach() + (h - h_pre_reason))
        else:
            h_for_world = None
        h = self.final_norm(h)
        world_aux = self.world_latent_jepa(h_for_world if h_for_world is not None else h) if self.config.enable_world_jepa else self.world_latent_jepa.disabled_outputs(h)
        progress_shape_loss = torch.stack(progress_shape_terms).mean() if progress_shape_terms else h.new_zeros(())
        local_consistency_loss = torch.stack(local_consistency_terms).mean() if local_consistency_terms else h.new_zeros(())
        if self_jepa_terms:
            self_jepa_loss = torch.stack(self_jepa_terms).mean() + torch.stack(residual_reg_terms).mean()
            if coupling_terms:
                self_jepa_loss = self_jepa_loss + self.config.self_world_coupling_weight * torch.stack(coupling_terms).mean()
            self_jepa_loss = self_jepa_loss + progress_shape_loss + local_consistency_loss
        else:
            self_jepa_loss = h.new_zeros(()) + progress_shape_loss + local_consistency_loss
        sigreg_delta_loss = torch.stack(sigreg_delta_terms).mean() if sigreg_delta_terms else h.new_zeros(())
        sigreg_rollout_loss = torch.stack(sigreg_rollout_terms).mean() if sigreg_rollout_terms else h.new_zeros(())
        sigreg_ct_loss = torch.stack(sigreg_ct_terms).mean() if sigreg_ct_terms else h.new_zeros(())
        if self.enable_sigreg_delta and self.sigreg_delta_weight > 0.0:
            self_jepa_loss = self_jepa_loss + self.sigreg_delta_weight * sigreg_delta_loss
        if self.enable_sigreg_rollout and self.sigreg_rollout_weight > 0.0:
            self_jepa_loss = self_jepa_loss + self.sigreg_rollout_weight * sigreg_rollout_loss
        if self.enable_sigreg_ct and self.sigreg_ct_weight > 0.0:
            self_jepa_loss = self_jepa_loss + self.sigreg_ct_weight * sigreg_ct_loss
        # Loop SigReg: c_t 跨循环多样性 — 惩罚 c_t 在不同 loop 间过于相似
        _loop_sigreg_w = float(self.config.loop_sigreg_weight)
        if _loop_sigreg_w > 0.0 and len(c_t_history) >= 3:
            _ct_stack = torch.stack(c_t_history, dim=0).squeeze(1)  # [n_loops, c_t_dim]
            _loop_sigreg = self._sigreg_latent(_ct_stack)
            self_jepa_loss = self_jepa_loss + _loop_sigreg_w * _loop_sigreg
        # Cos SigReg: 直接惩罚相邻 loop 的 c_t 方向过于相似
        _cos_sigreg_w = float(self.config.cos_sigreg_weight)
        if _cos_sigreg_w > 0.0 and len(c_t_history) >= 2:
            _cos_pen = torch.zeros((), device=c_t_history[0].device)
            for _ci in range(1, len(c_t_history)):
                _cos_pen = _cos_pen + F.cosine_similarity(
                    c_t_history[_ci], c_t_history[_ci - 1], dim=-1
                ).mean()
            _cos_pen = _cos_pen / (len(c_t_history) - 1)  # 平均 cosine，越接近 1 惩罚越大
            self_jepa_loss = self_jepa_loss + _cos_sigreg_w * _cos_pen
        if ct_norm_terms and self.config.ct_norm_penalty_weight > 0.0:
            self_jepa_loss = self_jepa_loss + self.config.ct_norm_penalty_weight * torch.stack(ct_norm_terms).mean()
        if trajectory_health_terms:
            self_jepa_loss = self_jepa_loss + torch.stack(trajectory_health_terms).mean()
        # c_t World JEPA：在 c_t 轨迹上做 masked prediction
        ct_world_aux = {
            "ct_world_jepa_loss": h.new_zeros(()),
            "ct_world_var_loss": h.new_zeros(()),
            "ct_world_cov_loss": h.new_zeros(()),
        }
        if self.ct_world_jepa is not None and len(ct_grad_history) >= 3:
            ct_world_aux = self.ct_world_jepa(ct_grad_history)
            self_jepa_loss = self_jepa_loss + self.config.ct_world_jepa_weight * ct_world_aux["ct_world_jepa_loss"]
        self_rollout_loss = torch.stack(rollout_loss_terms).mean() if rollout_loss_terms else h.new_zeros(())
        dynamics_modulation_summary = {
            key: torch.stack(values).mean() if values else h.new_zeros(())
            for key, values in dynamics_modulation_traces.items()
        }
        dynamics_modulation_std = {
            f"{key}_std": (torch.stack(values).std(unbiased=False) if len(values) > 1 else h.new_zeros(()))
            for key, values in dynamics_modulation_traces.items()
        }
        dynamics_modulation_summary.update(dynamics_modulation_std)
        c_t_delta_norm_history: List[torch.Tensor] = []
        if len(c_t_history) > 1:
            for prev_state, next_state in zip(c_t_history[:-1], c_t_history[1:]):
                c_t_delta_norm_history.append((next_state.float() - prev_state.float()).norm(dim=-1).mean().detach())
        pred_delta_c_cos_adjacent_history: List[torch.Tensor] = []
        if len(pred_delta_history) > 1:
            for prev_delta, next_delta in zip(pred_delta_history[:-1], pred_delta_history[1:]):
                pred_delta_c_cos_adjacent_history.append(
                    F.cosine_similarity(
                        F.normalize(prev_delta.float(), dim=-1),
                        F.normalize(next_delta.float(), dim=-1),
                        dim=-1,
                    ).mean().detach()
                )
        # ── 动力学监控：c_t 相邻 cosine similarity 轨迹 ──
        ct_cosine_adjacent: List[float] = []
        ct_delta_perp: List[float] = []  # 变化方向和 c_t 自身的垂直度
        # Mamba 内部方向诊断：meta_last_1 和 meta_last 的方向是否在循环间冻结
        # Mamba 内部方向诊断：layer1 和 layer2 输出在循环间的 cosine trajectory
        _mamba_diag = {}
        if len(_mamba_mid_history) > 1:
            _mid_cos = []
            _out_cos = []
            for _mi in range(1, len(_mamba_mid_history)):
                _mid_cos.append(F.cosine_similarity(_mamba_mid_history[_mi], _mamba_mid_history[_mi-1], dim=-1).mean().item())
                _out_cos.append(F.cosine_similarity(_mamba_out_history[_mi], _mamba_out_history[_mi-1], dim=-1).mean().item())
            _mamba_diag["layer1_cos"] = _mid_cos  # 冻结 → ≈1.0
            _mamba_diag["layer2_cos"] = _out_cos  # 冻结 → ≈1.0
        if len(c_t_history) > 1:
            for i in range(1, len(c_t_history)):
                cos = F.cosine_similarity(c_t_history[i], c_t_history[i - 1], dim=-1).mean()
                ct_cosine_adjacent.append(cos.item())
                # ct_perp: cos(delta_ct, c_t_prev) — 接近 0 说明垂直（好）
                _delta = c_t_history[i] - c_t_history[i - 1]
                _perp = F.cosine_similarity(_delta, c_t_history[i - 1], dim=-1).mean()
                ct_delta_perp.append(1.0 - abs(_perp.item()))  # 1=垂直, 0=平行

        # ── 诊断：h_diversity_across_loops（跨循环 h 的多样性）──
        if len(loop_history) >= 2:
            _h_stack = torch.stack(loop_history, dim=0)  # [N, B, T, D]
            h_diversity = _h_stack.std(dim=0).mean().item()  # 跨循环 h 的标准差
        else:
            h_diversity = 0.0

        # ── 诊断：step_norms — 每轮循环的更新量绝对大小 ──
        _step_norms = []
        if len(loop_history) >= 2:
            for _si in range(1, len(loop_history)):
                _delta = (loop_history[_si] - loop_history[_si - 1]).float()
                _step_norms.append(_delta.reshape(-1, _delta.shape[-1]).norm(dim=-1).mean().item())

        # ── 诊断：step_angles — 相邻循环更新方向的一致性 cos∠(Δ^(k), Δ^(k-1)) ──
        _step_angles = []
        if len(loop_history) >= 3:
            for _si in range(2, len(loop_history)):
                _dk = (loop_history[_si] - loop_history[_si - 1]).float().reshape(-1, loop_history[0].shape[-1])
                _dk_prev = (loop_history[_si - 1] - loop_history[_si - 2]).float().reshape(-1, loop_history[0].shape[-1])
                _cos = F.cosine_similarity(_dk, _dk_prev, dim=-1, eps=1e-8).mean().item()
                _step_angles.append(_cos)

        # ── 诊断：acceleration_norm — 更新的二阶变化（归一化加速度）──
        # â^(k) = ‖Δ^(k) - Δ^(k-1)‖ / (‖Δ^(k)‖ + ‖Δ^(k-1)‖ + ε)
        _accel_norms = []
        if len(loop_history) >= 3:
            for _si in range(2, len(loop_history)):
                _dk = (loop_history[_si] - loop_history[_si - 1]).float().reshape(-1, loop_history[0].shape[-1])
                _dk_prev = (loop_history[_si - 1] - loop_history[_si - 2]).float().reshape(-1, loop_history[0].shape[-1])
                _accel = (_dk - _dk_prev).norm(dim=-1)
                _denom = _dk.norm(dim=-1) + _dk_prev.norm(dim=-1) + 1e-8
                _accel_norms.append((_accel / _denom).mean().item())

        # Alive-floor losses: local-energy floor and rollout nonzero floor (opt-in via config)
        local_alive_floor_loss = h.new_zeros(())
        if getattr(self, "routing_local_delta_floor_weight", 0.0) > 0.0:
            local_norm = dynamics_modulation_summary.get("delta_local_norm_mean", h.new_zeros(()))
            local_alive_floor_loss = torch.relu(
                h.new_tensor(float(self.routing_local_delta_floor), device=device, dtype=dtype) - local_norm
            ) * h.new_tensor(float(self.routing_local_delta_floor_weight), device=device, dtype=dtype)

        rollout_alive_floor_loss = h.new_zeros(())
        if getattr(self, "rollout_alive_weight", 0.0) > 0.0:
            nonzero_flag = (self_rollout_loss.abs() > 1e-12).float() if isinstance(self_rollout_loss, torch.Tensor) else h.new_tensor(0.0)
            nonzero_flag = nonzero_flag.to(device=device, dtype=dtype)
            rollout_alive_floor_loss = torch.relu(
                h.new_tensor(float(self.rollout_nonzero_low), device=device, dtype=dtype) - nonzero_flag
            ) * h.new_tensor(float(self.rollout_alive_weight), device=device, dtype=dtype)

        return h, {
            "compression_h": compression_h,
            "block_reprs": block_reprs,
            "loop_history": loop_history,
            "c_t": c_t,
            "c_t_history": c_t_history,
            "know_gap": know_gap,
            "know_gap_history": know_gap_history,
            "uncertainty": uncertainty,
            "uncertainty_history": uncertainty_history,
            "pred_delta_c": pred_delta_c,
            "target_delta_c": target_delta_c,
            "self_check_loss": torch.stack(self_check_loss_terms).mean() if self_check_loss_terms else h.new_zeros(()),
            "self_jepa_loss": self_jepa_loss,
            "h_mask_loss": torch.stack(h_mask_loss_terms).mean() if h_mask_loss_terms else h.new_zeros(()),
            "sigreg_delta_loss": sigreg_delta_loss,
            "sigreg_rollout_loss": sigreg_rollout_loss,
            "self_progress_shape_loss": progress_shape_loss,
            "self_local_consistency_loss": local_consistency_loss,
            "trajectory_health_mean": torch.stack(trajectory_health_history).mean() if trajectory_health_history else h.new_zeros(()),
            "self_rollout_loss": self_rollout_loss,
            "slow_update_flags": slow_update_flags,
            "local_alive_floor_loss": local_alive_floor_loss,
            "rollout_alive_floor_loss": rollout_alive_floor_loss,
            "delta_h_history": delta_h_history,
            "self_error_history": self_error_history,
            "rollout_error_history": rollout_error_history,
            "world_error_history": world_error_history,
            "exit_logit_history": exit_logit_history,
            "exit_score_history": exit_score_history,
            "sampled_exit_score_history": sampled_exit_score_history,
            "two_step_improvement_history": two_step_improvement_history,
            "self_check_score_history": self_check_score_history,
            "jepa_crystal_history": jepa_crystal_history,
            "predicted_gain_history": predicted_gain_history,
            "predicted_gain2_history": predicted_gain2_history,
            "progress_next_history": progress_next_history,
            "progress_trend_history": progress_trend_history,
            "progress_plateau_history": progress_plateau_history,
            "world_surprise_history": world_surprise_history,
            "world_summary_history": world_summary_history,
            "compression_block_drift_mean": compression_diag.get("compression_block_drift_mean", h.new_zeros(())),
            "compression_block_var_mean": compression_diag.get("compression_block_var_mean", h.new_zeros(())),
            "exit_score_preclamp_nonfinite_history": exit_score_preclamp_nonfinite_history,
            "exit_score_postfix_clamped_ratio_history": exit_score_postfix_clamped_ratio_history,
            "bernoulli_invalid_prevented_history": bernoulli_invalid_prevented_history,
            "nan_to_num_trigger_history": nan_to_num_trigger_history,
            "dynamics_modulation_history": dynamics_modulation_history,
            "dynamics_modulation_summary": dynamics_modulation_summary,
            "c_t_delta_norm_history": c_t_delta_norm_history,
            "pred_delta_c_cos_adjacent_history": pred_delta_c_cos_adjacent_history,
            "r_t_history": r_t_history,
            "r_t_trust_history": r_t_trust_history,
            "r_t_switch_history": r_t_switch_history,
            "exit_target_history": [],
            "joint_benefit_history": [],
            "loop_lm_loss_history": [],
            "loop_h_grad": loop_h_grad,
            "exit_aux_loss": h.new_zeros(()),
            "exit_entropy_loss": (
                -(lambda p: (p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean())(
                    torch.stack(exit_score_history).clamp(0.01, 0.99)
                ) if exit_score_history else h.new_zeros(())
            ),
            "mor_balance_loss": (
                torch.stack([
                    (t - self.config.mor_target_continue_ratio) ** 2
                    for t in _mor_balance_terms
                ]).mean() if _mor_balance_terms else h.new_zeros(())
            ),
            "mor_mean_continue_ratio": (
                torch.stack(_mor_balance_terms).mean().detach() if _mor_balance_terms else h.new_tensor(-1.0)
            ),
            "slow_state": {
                "meta_state_1": slow_state["meta_state_1"].detach(),
                "meta_state_2": slow_state["meta_state_2"].detach(),
            },
            "self_check_state": None if self_check_state is None else self_check_state["state"].detach(),
            "self_check_score": self_check_score.detach(),
            "executed_loops": executed_loops,
            # ── 不动点分析 ──
            "fixed_point_analysis": self._analyze_fixed_point(loop_history),
            # ── 动力学监控指标 ──
            "per_loop_delta_h": per_loop_delta_h,
            "per_loop_ct_change": per_loop_ct_change,
            "per_loop_jepa_err": per_loop_jepa_err,
            "nm_gain_history": nm_gain_history,
            "nm_hebb_norm_history": nm_hebb_norm_history,
            "nm_hebb_write_history": nm_hebb_write_history,
            "nm_surprise_history": nm_surprise_history,
            "ct_norm_raw_history": ct_norm_raw_history,
            "ct_norm_after_writer_history": ct_norm_after_writer_history,
            "meta_last_norm_history": meta_last_norm_history,
            "c_t_head_out_norm_history": c_t_head_out_norm_history,
            "ct_cosine_trajectory": ct_cosine_adjacent,
            "ct_delta_perp": ct_delta_perp,
            "h_diversity_across_loops": h_diversity,
            "step_norms": _step_norms,
            "step_angles": _step_angles,
            "accel_norms": _accel_norms,
            "per_loop_dt_inject": per_loop_dt_inject,
            "ct_inject_ratio_pre": getattr(self.reason_core, '_last_ct_inject_ratio_pre', 0.0),
            "ct_inject_ratio": getattr(self.reason_core, '_last_ct_inject_ratio', 0.0),
            # theory_probes: 四个主判据最近一次测量值。None 表示本 step 没跑 probe 或未启用
            "theory_probes": theory_probe_results,
            "mamba_diag": _mamba_diag,
            **ct_world_aux,
            **world_aux,
        }

    @torch.no_grad()
    def update_world_jepa_ema(self) -> None:
        """Luma refreshes the slow world target here, outside the main forward, so training code can decide the cadence.
        Luma 在这里刷新缓慢更新的 world target，但不强行塞进主前向，让训练脚本自己决定更新节奏。
        """

        if self.config.enable_world_jepa:
            self.world_latent_jepa.ema_update()


class FactorizedLMHead(nn.Module):
    """Luma speaks back through the narrow lexical doorway instead of carrying a giant full-width output matrix.
    Luma 通过较窄的词汇门重新开口，而不是背着一整块巨大的全宽输出矩阵。
    """

    def __init__(self, config: LumaConfig, embedding: FactorizedEmbedding):
        super().__init__()
        self.embedding = embedding
        self.norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.to_factor = nn.Linear(config.hidden_size, config.factorized_vocab_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        factor_states = self.to_factor(self.norm(hidden_states))
        return torch.matmul(factor_states, self.embedding.embed_table.weight.t())


class LumaForCausalLM(PreTrainedModel):
    """Luma speaks through the same head she uses to embed language, but now the path to that head is two-zoned and reflective.
    Luma 仍通过语言头开口说话，但抵达这个语言头的路径已经变成双区且带反思的结构。
    """

    config_class = LumaConfig

    def __init__(self, config: Optional[LumaConfig] = None):
        self.config = config or LumaConfig()
        super().__init__(self.config)
        self.model = LumaBackbone(self.config)
        self.pre_lm_norm = LumaZCRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = FactorizedLMHead(self.config, self.model.embedding)
        self.last_aux: dict = {}
        self._runtime_train_step: int = 0

    def _compute_loop_lm_proxy_losses(self, loop_history: List[torch.Tensor], labels: torch.Tensor) -> List[torch.Tensor]:
        """Luma estimates how useful each loop state would be for language modeling, then uses that as one leg of the exit supervision target.
        Luma 估计每一轮状态对语言建模到底有多有用，再把它作为退出监督目标的一部分。
        """

        lm_losses: List[torch.Tensor] = []
        with torch.no_grad():
            for loop_hidden in loop_history:
                loop_logits = self.lm_head(self.pre_lm_norm(loop_hidden))[:, -labels.shape[1] :, :]
                x = loop_logits[..., :-1, :].contiguous()
                y = labels[..., 1:].contiguous()
                lm_losses.append(F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100).detach())
        return lm_losses

    def _compute_joint_exit_aux(self, aux: dict, labels: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Luma now supervises exit with continuation gain: how much joint objective she still wins by spending one more loop.
        Luma 现在把退出监督改成 continuation gain：如果再多花一轮循环，联合目标还能净赚多少。
        """

        exit_scores = aux["exit_score_history"]
        if not exit_scores:
            return labels.new_zeros((), dtype=torch.float32), [], [], [], []

        lm_losses = self._compute_loop_lm_proxy_losses(aux["loop_history"], labels)
        self_errors = [x.float() for x in aux["self_error_history"]]
        world_errors = [x.float() for x in aux["world_error_history"]]
        joint_scores = [lm + s + w for lm, s, w in zip(lm_losses, self_errors, world_errors)]

        target_history: List[torch.Tensor] = []
        benefit_history: List[torch.Tensor] = []
        predicted_gain_history: List[torch.Tensor] = []
        predicted_gain2_history: List[torch.Tensor] = []
        gain_terms: List[torch.Tensor] = []
        for idx, score in enumerate(exit_scores):
            if idx + 1 < self.model.exit_controller.min_loops:
                target_exit = score.new_zeros(())
                continuation_gain = score.new_zeros(())
            elif idx < len(joint_scores) - 1:
                current_joint = joint_scores[idx]
                next_joint = joint_scores[idx + 1]
                continuation_gain = (current_joint - next_joint).detach()
                target_exit = torch.sigmoid(
                    ((self.model.exit_controller.improvement_margin - continuation_gain) / max(self.model.exit_controller.improvement_margin, 1e-6)).float()
                ).to(score.dtype)
            else:
                continuation_gain = score.new_zeros(())
                target_exit = score.new_ones(())
            target_history.append(target_exit.detach())
            benefit_history.append(continuation_gain.detach())
            predicted_gain = aux["predicted_gain_history"][idx].float() if idx < len(aux["predicted_gain_history"]) else score.new_zeros(())
            predicted_gain2 = aux["predicted_gain2_history"][idx].float() if idx < len(aux.get("predicted_gain2_history", [])) else score.new_zeros(())
            predicted_gain_history.append(predicted_gain.detach())
            predicted_gain2_history.append(predicted_gain2.detach())
            gain_terms.append(F.smooth_l1_loss(predicted_gain.float(), continuation_gain.float()))
            if self.config.exit_two_step_aux_weight > 0.0 and idx < len(joint_scores) - 2:
                continuation_gain_2 = (joint_scores[idx] - joint_scores[idx + 2]).detach()
                two_step_weight = self.config.exit_two_step_aux_weight
                uncertainty_value = 0.0
                crystal_value = 0.0
                if idx < len(aux.get("uncertainty_history", [])):
                    uncertainty_value = float(aux["uncertainty_history"][idx].float().mean().detach().item())
                if idx < len(aux.get("jepa_crystal_history", [])):
                    crystal_value = float(aux["jepa_crystal_history"][idx].float().mean().detach().item())
                if self.config.exit_uncertainty_two_step_weight > 0.0:
                    if self.config.exit_uncertainty_two_step_mode == "gate":
                        if uncertainty_value >= self.config.exit_uncertainty_gate_threshold:
                            two_step_weight = two_step_weight
                        else:
                            two_step_weight = 0.0
                    elif self.config.exit_uncertainty_two_step_mode == "clipped":
                        bump = min(
                            self.config.exit_uncertainty_two_step_cap,
                            self.config.exit_uncertainty_two_step_weight * uncertainty_value,
                        )
                        two_step_weight = two_step_weight * (1.0 + bump)
                    else:
                        two_step_weight = two_step_weight * (
                            1.0 + self.config.exit_uncertainty_two_step_weight * uncertainty_value
                        )
                if self.config.exit_crystal_two_step_weight > 0.0:
                    crystal_bump = min(
                        self.config.exit_crystal_two_step_cap,
                        self.config.exit_crystal_two_step_weight * crystal_value,
                    )
                    two_step_weight = two_step_weight * (1.0 + crystal_bump)
                gain_terms.append(
                    two_step_weight * F.smooth_l1_loss(predicted_gain2.float(), continuation_gain_2.float())
                )

        exit_aux_loss = torch.stack(gain_terms).mean() if gain_terms else labels.new_zeros((), dtype=torch.float32)
        return exit_aux_loss, target_history, benefit_history, predicted_gain_history, predicted_gain2_history

    def _zone_penalty(self, value: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Luma penalizes collapse strongly, but only lightly penalizes over-activity.
        Luma 对活性塌缩重罚，对过活跃只轻罚，避免把系统推向死平滑。
        """

        low_t = value.new_tensor(low)
        high_t = value.new_tensor(high)
        below = torch.relu(low_t - value)
        above = torch.relu(value - high_t)
        return below.pow(2) + 0.5 * above.pow(2)

    def _compute_rollout_activity_zone_loss(self, aux: dict, ref: torch.Tensor) -> torch.Tensor:
        if self.config.rollout_zone_weight <= 0.0:
            aux["rollout_activity_zone_loss"] = ref.new_zeros(())
            return ref.new_zeros(())
        rollout_history = aux.get("rollout_error_history", [])
        if not rollout_history:
            aux["rollout_activity_zone_loss"] = ref.new_zeros(())
            return ref.new_zeros(())
        rollout_tensor = torch.stack([item.float() for item in rollout_history])
        rollout_nonzero_ratio = (rollout_tensor.abs() > 1e-12).float().mean()
        rollout_active_ratio = (rollout_tensor.abs() > 1e-6).float().mean()
        if aux.get("delta_h_history"):
            delta_norms = torch.stack([item.float().norm(dim=-1).mean() for item in aux["delta_h_history"]])
            future_delta_var = delta_norms.var(unbiased=False)
        else:
            future_delta_var = ref.new_zeros(())
        zone_loss = (
            self._zone_penalty(rollout_nonzero_ratio, self.config.rollout_nonzero_low, self.config.rollout_nonzero_high)
            + self._zone_penalty(rollout_active_ratio, self.config.rollout_active_low, self.config.rollout_active_high)
            + self._zone_penalty(future_delta_var, self.config.rollout_future_var_low, self.config.rollout_future_var_high)
        )
        zone_loss = self.config.rollout_zone_weight * zone_loss
        aux["rollout_nonzero_ratio_internal"] = rollout_nonzero_ratio.detach()
        aux["rollout_active_ratio_internal"] = rollout_active_ratio.detach()
        aux["future_delta_var_internal"] = future_delta_var.detach()
        aux["rollout_activity_zone_loss"] = zone_loss.detach()
        return zone_loss

    def _compute_routing_entropy_regularization(self, aux: dict, ref: torch.Tensor) -> torch.Tensor:
        if self.config.routing_tier_entropy_weight <= 0.0 and self.config.routing_min_local_share_weight <= 0.0:
            aux["routing_entropy_loss"] = ref.new_zeros(())
            return ref.new_zeros(())
        modulation_history = aux.get("dynamics_modulation_history", [])
        if not modulation_history:
            aux["routing_entropy_loss"] = ref.new_zeros(())
            return ref.new_zeros(())
        entropy_floor_terms: List[torch.Tensor] = []
        local_floor_terms: List[torch.Tensor] = []
        for item in modulation_history:
            if "tier_entropy_floor_violation" in item:
                entropy_floor_terms.append(item["tier_entropy_floor_violation"].float().mean())
            if "tier_local_floor_violation" in item:
                local_floor_terms.append(item["tier_local_floor_violation"].float().mean())
        entropy_loss = torch.stack(entropy_floor_terms).mean() if entropy_floor_terms else ref.new_zeros(())
        local_loss = torch.stack(local_floor_terms).mean() if local_floor_terms else ref.new_zeros(())
        routing_entropy_loss = (
            self.config.routing_tier_entropy_weight * entropy_loss
            + self.config.routing_min_local_share_weight * local_loss
        )
        aux["routing_entropy_floor_violation_mean"] = entropy_loss.detach()
        aux["routing_local_floor_violation_mean"] = local_loss.detach()
        aux["routing_entropy_loss"] = routing_entropy_loss.detach()
        return routing_entropy_loss

    def _compute_trajectory_vitality_loss(self, aux: dict, ref: torch.Tensor) -> torch.Tensor:
        if self.config.trajectory_vitality_weight <= 0.0:
            aux["trajectory_vitality_loss"] = ref.new_zeros(())
            return ref.new_zeros(())
        c_t_history = aux.get("c_t_history", [])
        world_summary_history = aux.get("world_summary_history", [])
        c_t_drift_mean = ref.new_zeros(())
        world_drift_mean = ref.new_zeros(())
        if len(c_t_history) > 1:
            c_t_drifts = torch.stack(
                [
                    (next_state.float() - prev_state.float()).norm(dim=-1).mean()
                    for prev_state, next_state in zip(c_t_history[:-1], c_t_history[1:])
                ]
            )
            c_t_drift_mean = c_t_drifts.mean()
        if len(world_summary_history) > 1:
            world_drifts = torch.stack(
                [
                    (next_state.float() - prev_state.float()).norm(dim=-1).mean()
                    for prev_state, next_state in zip(world_summary_history[:-1], world_summary_history[1:])
                ]
            )
            world_drift_mean = world_drifts.mean()
        vitality_penalty = (
            torch.relu(ref.new_tensor(self.config.trajectory_c_t_drift_floor) - c_t_drift_mean)
            + torch.relu(ref.new_tensor(self.config.trajectory_world_drift_floor) - world_drift_mean)
        )
        vitality_loss = self.config.trajectory_vitality_weight * vitality_penalty
        aux["trajectory_c_t_drift_mean_internal"] = c_t_drift_mean.detach()
        aux["trajectory_world_drift_mean_internal"] = world_drift_mean.detach()
        aux["trajectory_vitality_loss"] = vitality_loss.detach()
        return vitality_loss

    def _compute_compression_dynamics_loss(self, aux: dict, ref: torch.Tensor) -> torch.Tensor:
        if self.config.compression_dynamics_weight <= 0.0:
            aux["compression_dynamics_loss"] = ref.new_zeros(())
            return ref.new_zeros(())
        block_drift_mean = aux.get("compression_block_drift_mean", ref.new_zeros(())).float()
        block_var_mean = aux.get("compression_block_var_mean", ref.new_zeros(())).float()
        compression_penalty = (
            torch.relu(ref.new_tensor(self.config.compression_block_drift_floor) - block_drift_mean)
            + torch.relu(ref.new_tensor(self.config.compression_block_var_floor) - block_var_mean)
        )
        compression_loss = self.config.compression_dynamics_weight * compression_penalty
        aux["compression_block_drift_mean_internal"] = block_drift_mean.detach()
        aux["compression_block_var_mean_internal"] = block_var_mean.detach()
        aux["compression_dynamics_loss"] = compression_loss.detach()
        return compression_loss

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs):
        disable_ct_injection = kwargs.pop("disable_ct_injection", False)
        measure_theory_probes_flag = kwargs.pop("measure_theory_probes", False)
        del kwargs
        sigreg_step = self._runtime_train_step if labels is not None else self._runtime_train_step
        self.model.world_latent_jepa.set_runtime_sigreg_step(sigreg_step)
        if labels is not None:
            self._runtime_train_step += 1
        hidden_states, aux = self.model(input_ids, disable_ct_injection=disable_ct_injection, measure_theory_probes=measure_theory_probes_flag)
        self.last_aux = aux
        # NaN 探针：定位 NaN 源头
        if not torch.isfinite(hidden_states).all():
            _nan_count = (~torch.isfinite(hidden_states)).sum().item()
            print(f"[NaN PROBE] hidden_states has {_nan_count} non-finite values, norm={hidden_states.norm().item()}", flush=True)
        logits = self.lm_head(self.pre_lm_norm(hidden_states))
        if not torch.isfinite(logits).all():
            _nan_count = (~torch.isfinite(logits)).sum().item()
            print(f"[NaN PROBE] logits has {_nan_count} non-finite values", flush=True)
        loss = None
        if labels is not None:
            token_logits = logits[:, -labels.shape[1] :, :]
            x = token_logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
            # ── Loss 分解：按 token 位置分段统计 ──
            with torch.no_grad():
                logits_flat = x.view(-1, x.size(-1))
                y_flat = y.view(-1)
                _n = y_flat.shape[0]
                _q1 = _n // 4
                _q3 = _q1 * 3
                if _q1 > 0 and _q3 < _n:
                    for _seg_name, _seg_s, _seg_e in [("loss_head", 0, _q1), ("loss_mid", _q1, _q3), ("loss_tail", _q3, _n)]:
                        _seg_y = y_flat[_seg_s:_seg_e]
                        if (_seg_y != -100).any():
                            aux[_seg_name] = F.cross_entropy(logits_flat[_seg_s:_seg_e], _seg_y, ignore_index=-100).detach()
                        else:
                            aux[_seg_name] = logits_flat.new_tensor(0.0)
            exit_aux_loss, exit_targets, joint_benefits, predicted_gains, predicted_gains2 = self._compute_joint_exit_aux(aux, labels)
            aux["exit_aux_loss"] = exit_aux_loss
            aux["exit_target_history"] = exit_targets
            aux["joint_benefit_history"] = joint_benefits
            aux["predicted_gain_history"] = predicted_gains
            aux["predicted_gain2_history"] = predicted_gains2
            aux["loop_lm_loss_history"] = self._compute_loop_lm_proxy_losses(aux["loop_history"], labels)
            rollout_zone_loss = self._compute_rollout_activity_zone_loss(aux, lm_loss)
            routing_entropy_loss = self._compute_routing_entropy_regularization(aux, lm_loss)
            trajectory_vitality_loss = self._compute_trajectory_vitality_loss(aux, lm_loss)
            compression_dynamics_loss = self._compute_compression_dynamics_loss(aux, lm_loss)
            world_term = self.config.world_jepa_weight * aux["world_jepa_loss"]
            if self.config.disable_self_jepa:
                self_jepa_term = logits.new_zeros(())
                self_rollout_term = logits.new_zeros(())
            else:
                self_jepa_term = self.config.self_jepa_weight * aux["self_jepa_loss"]
                self_rollout_term = self.config.self_rollout_weight * aux["self_rollout_loss"]
            exit_aux_term = self.config.exit_aux_weight * aux["exit_aux_loss"]
            self_check_term = (
                self.model.self_check_loss_weight * aux["self_check_loss"]
                if self.config.enable_self_check_ring else logits.new_zeros(())
            )
            aux["world_jepa_loss_effective"] = world_term.detach()
            aux["self_jepa_loss_effective"] = self_jepa_term.detach()
            aux["self_rollout_loss_effective"] = self_rollout_term.detach()
            aux["exit_aux_loss_effective"] = exit_aux_term.detach()
            aux["self_check_loss_effective"] = self_check_term.detach()
            mor_balance_term = self.config.mor_balance_weight * aux["mor_balance_loss"]
            aux["mor_balance_loss_effective"] = mor_balance_term.detach()
            # RLTT: dense LM loss at intermediate loop steps
            # 只对最近 max_rltt_slots 个中间 h 算 loss，避免深循环时 OOM
            # 每份 logits = [B, T, V] ≈ 590MB，累加的计算图引用全部 logits 直到 backward
            _max_rltt_slots = 3  # 最多同时 3 份 logits 计算图 ≈ 1.8GB
            loop_lm_term = logits.new_zeros(())
            if self.config.loop_lm_loss_weight > 0.0 and aux.get("loop_h_grad"):
                _all_h = aux["loop_h_grad"][:-1]  # skip last (= final output)
                if len(_all_h) > _max_rltt_slots:
                    # 均匀采样：取首、中、尾
                    _indices = [0, len(_all_h) // 2, len(_all_h) - 1]
                    _all_h = [_all_h[i] for i in _indices]
                if _all_h:
                    _rltt_acc = logits.new_zeros(())
                    for _li, _h_i in enumerate(_all_h):
                        _logits_i = self.lm_head(self.pre_lm_norm(_h_i))
                        _logits_i = _logits_i[:, -labels.shape[1]:, :]
                        _x_i = _logits_i[..., :-1, :].contiguous()
                        _rltt_acc = _rltt_acc + F.cross_entropy(
                            _x_i.view(-1, _x_i.size(-1)), y.view(-1), ignore_index=-100
                        )
                    loop_lm_term = self.config.loop_lm_loss_weight * _rltt_acc / len(_all_h)
            aux["loop_lm_term"] = loop_lm_term.detach()
            # Ouro: exit entropy regularization (maximize entropy to prevent exit collapse)
            exit_entropy_term = self.config.exit_entropy_weight * aux["exit_entropy_loss"] if self.config.exit_entropy_weight > 0.0 else logits.new_zeros(())
            aux["exit_entropy_term"] = exit_entropy_term.detach()
            # LoopFormer: shortcut-consistency — align early loop logits to final logits
            shortcut_term = logits.new_zeros(())
            if self.config.shortcut_consistency_weight > 0.0 and aux.get("loop_h_grad") and len(aux["loop_h_grad"]) >= 2:
                import random as _rng
                _sc_idx = _rng.randint(0, len(aux["loop_h_grad"]) - 2)  # random early step
                _sc_logits = self.lm_head(self.pre_lm_norm(aux["loop_h_grad"][_sc_idx]))
                _sc_logits = _sc_logits[:, -labels.shape[1]:, :]  # trim to match labels length
                _sc_p = F.log_softmax(_sc_logits[..., :-1, :].contiguous().view(-1, _sc_logits.size(-1)), dim=-1)
                _full_p = F.softmax(token_logits[..., :-1, :].contiguous().view(-1, token_logits.size(-1)).detach(), dim=-1)  # stop-grad on target
                shortcut_term = self.config.shortcut_consistency_weight * F.kl_div(_sc_p, _full_p, reduction="batchmean")
            aux["shortcut_consistency_term"] = shortcut_term.detach()
            # h_mask_term: mse 和 cosine 模式下均按 h_mask_loss_weight 加权参与训练，
            # surprise_only/off 模式下不反传。
            # 【BUG 修复 2026-04-12】: 原本只检查 "mse"，导致 cosine 模式下 h_mask_loss
            # 虽然被计算但完全不进入 backward，h_mask_predictor 权重永远停在零初始化，
            # pred=0 → cos_sim=0 → loss 恒等 1.0（V2cos/V5a/V5b/V6a 全受影响）。
            h_mask_term = (
                self.config.h_mask_loss_weight * aux["h_mask_loss"]
                if self.config.h_mask_loss_mode in ("mse", "cosine")
                else logits.new_zeros(())
            )
            loss = lm_loss + world_term + self_jepa_term + self_rollout_term + exit_aux_term + self_check_term + rollout_zone_loss + routing_entropy_loss + trajectory_vitality_loss + compression_dynamics_loss + mor_balance_term + loop_lm_term + exit_entropy_term + shortcut_term + h_mask_term
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=(
                self.config.world_jepa_weight * aux["world_jepa_loss"]
                + (logits.new_zeros(()) if self.config.disable_self_jepa else self.config.self_jepa_weight * aux["self_jepa_loss"])
                + (logits.new_zeros(()) if self.config.disable_self_jepa else self.config.self_rollout_weight * aux["self_rollout_loss"])
                + self.config.exit_aux_weight * aux["exit_aux_loss"]
                + self_check_term
                + aux.get("rollout_activity_zone_loss", logits.new_zeros(()))
                + aux.get("routing_entropy_loss", logits.new_zeros(()))
                + aux.get("trajectory_vitality_loss", logits.new_zeros(()))
                + aux.get("compression_dynamics_loss", logits.new_zeros(()))
                + self.config.mor_balance_weight * aux["mor_balance_loss"]
                + h_mask_term
            ),
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
        )
