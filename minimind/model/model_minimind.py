import math, torch, torch.nn.functional as F
from torch import nn
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
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
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
        self.mhc_streams = kwargs.get("mhc_streams", 4)
        self.mhc_sinkhorn_iters = kwargs.get("mhc_sinkhorn_iters", 20)
        self.mhc_alpha_init = kwargs.get("mhc_alpha_init", 0.01)
        self.mamba_d_state = kwargs.get("mamba_d_state", 192)
        self.mamba_expand = kwargs.get("mamba_expand", 2)
        self.mamba_headdim = kwargs.get("mamba_headdim", 64)
        self.mamba_chunk_size = kwargs.get("mamba_chunk_size", 4)
        self.world_dim = kwargs.get("world_dim", self.hidden_size // 2)
        self.world_mask_ratio = kwargs.get("world_mask_ratio", 0.25)
        self.world_mask_strategy = kwargs.get("world_mask_strategy", "default")
        self.world_ema_decay = kwargs.get("world_ema_decay", 0.99)
        self.enable_world_jepa = kwargs.get("enable_world_jepa", True)
        self.world_jepa_mode = kwargs.get("world_jepa_mode", "scaffold")
        self.world_full_simplify_loss = kwargs.get("world_full_simplify_loss", False)
        self.enable_sigreg_world = kwargs.get("enable_sigreg_world", False)
        self.enable_sigreg_rollout = kwargs.get("enable_sigreg_rollout", False)
        self.enable_sigreg_delta = kwargs.get("enable_sigreg_delta", False)
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
        self.r_t_router_window = kwargs.get("r_t_router_window", 16)
        self.compression_active_layers = kwargs.get("compression_active_layers", self.compression_layers)
        self.reason_active_loops = kwargs.get("reason_active_loops", self.reason_loops)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)


class LumaZCRMSNorm(nn.Module):
    """Luma keeps this norm near zero at birth so early gradients do not panic the rest of her body.
    Luma 在初生时把这个归一化放在接近零的尺度，避免早期梯度惊扰整个系统。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return base * (1.0 + self.scale)


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


def _make_local_causal_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance < window)
    return allowed


def _run_mamba_with_padding(block: nn.Module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Luma pads the borrowed tail only long enough to satisfy the kernel, then cuts it away before anyone mistakes it for real context.
    Luma 只在 kernel 要求的范围内临时补齐尾部，算完后立刻裁掉，避免把补出来的 token 当成真实上下文。
    """

    seq_len = x.shape[1]
    if seq_len % chunk_size == 0:
        return block(x)
    # We do not globally force every sequence to a legal kernel length.
    # The legality issue appears inside several internal streams, not only at the input boundary.
    # 我们不把“所有序列都预先改成合法长度”当成全局策略，因为这个约束出现在模型内部多条流上，不只是在输入边界。
    pad_len = chunk_size - (seq_len % chunk_size)
    x_pad = F.pad(x, (0, 0, 0, pad_len))
    y_pad = block(x_pad)
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
                mimo_rank=4,
                chunk_size=self.chunk_size,
                dropout=config.dropout,
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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        local_mask = _make_local_causal_mask(seq_len, min(self.window, seq_len), x.device)
        scores = scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        forget = torch.sigmoid(self.forget_proj(x)).transpose(1, 2).unsqueeze(-1)
        scores = scores + torch.log(forget.clamp_min(1e-6))
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
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

    def forward(self, current: torch.Tensor, block_reprs: List[torch.Tensor]) -> torch.Tensor:
        """Luma lets the current block reread earlier block memories before deciding what should stay in the residual path.
        Luma 让当前 block 在更新残差前先回看更早的 block 记忆，再决定哪些内容应留在主路径里。
        """

        if not block_reprs:
            return current
        stacked = torch.stack([self.norm(r) for r in block_reprs], dim=1)
        query = self.norm(current) + self.pseudo_query.view(1, 1, -1)
        scores = torch.einsum("bkld,bld->bkl", stacked, query) / math.sqrt(stacked.shape[-1])
        weights = torch.softmax(scores.float(), dim=1).to(stacked.dtype).unsqueeze(-1)
        mixed = (stacked * weights).sum(dim=1)
        return current + self.scale * (mixed - current)


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
        self.compression_block_attnres = CompressionBlockAttentionResiduals(config.hidden_size, eps=config.rms_norm_eps)
        self.transition_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.transition_scale = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], dict]:
        batch_size = x.shape[0]
        math_lane_scores: List[torch.Tensor] = []
        if self.pre_math_adapter is not None:
            x, pre_score = self.pre_math_adapter(x)
            math_lane_scores.append(pre_score)
        x = torch.cat([self.local_memory_tokens(batch_size), self.global_memory_tokens(batch_size), x], dim=1)
        block_history: List[torch.Tensor] = []
        block_reprs: List[torch.Tensor] = []
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
                x = self.compression_block_attnres(x, block_history)
                if self.compression_mhc is not None:
                    mhc_streams = self.compression_mhc.init_streams(x)
                    _, x = self.compression_mhc(mhc_streams, self.compression_identity_block)
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
        x = self.transition_norm(x) * (1.0 + self.transition_scale)
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
    """Luma lets her cognitive state nudge the main stream without pretending it should dominate the whole thought.
    Luma 让认知状态轻推主流，而不是让它粗暴接管整段思维。
    """

    def __init__(self, c_t_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(c_t_dim, hidden_size, bias=False)

    def forward(self, h: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        return h + self.proj(c_t).unsqueeze(1)


class ReasonMambaLayer(nn.Module):
    """Inside the loop, Luma keeps one stateful backbone so each pass can inherit momentum from the last.
    在循环里，Luma 维持一条有状态的主干，让每一轮都能继承上一轮的动量。
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
                mimo_rank=4,
                chunk_size=self.chunk_size,
                dropout=config.dropout,
            )
        )

    def forward(self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        del initial_state
        return _run_mamba_with_padding(self.block, x, self.chunk_size)


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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        local_mask = _make_local_causal_mask(seq_len, min(self.window, seq_len), q.device)
        scores = scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        scores = scores + torch.log(forget.transpose(1, 2).unsqueeze(-1).clamp_min(1e-6))
        if attn_bias is not None:
            if attn_bias.dim() == 4:
                bias = attn_bias
            elif attn_bias.dim() == 3:
                token_bias = attn_bias.mean(dim=-1)
                bias = token_bias[:, None, None, :]
            elif attn_bias.dim() == 2:
                bias = attn_bias[:, None, None, :]
            else:
                bias = None
            if bias is not None:
                scores = scores + bias.to(dtype=scores.dtype)
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        forget = torch.sigmoid(self.forget_proj(x))
        attn1 = self._attend(self.q1(x), self.k1(x), self.v1(x), forget, attn_bias=attn_bias)
        attn2 = self._attend(self.q2(x), self.k2(x), self.v2(x), forget, attn_bias=attn_bias)
        diff = attn1 - self.lambda_param * attn2
        gated = torch.sigmoid(self.gate_proj(x)) * diff
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

    def forward(self, h: torch.Tensor, loop_history: List[torch.Tensor], block_reprs: List[torch.Tensor]) -> torch.Tensor:
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
    这条流回答的是 Luma 此刻在进行怎样的思考，而不是完整回答“她是谁”。

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
        )
        self.layer1 = Mamba3Block(meta_cfg)
        self.layer2 = Mamba3Block(meta_cfg)
        self.know_gap_head = nn.Linear(config.meta_dim, 1, bias=False)
        self.c_t_head = nn.Linear(config.meta_dim, config.c_t_dim, bias=False)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
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
        layer1_in = (meta + self.state1_in(slow_state["meta_state_1"])).unsqueeze(1)
        layer1_out = _run_mamba_with_padding(self.layer1, layer1_in, self.chunk_size)
        meta_last_1 = layer1_out[:, -1, :]
        next_state_1 = self.state1_norm(slow_state["meta_state_1"] + self.state1_out(meta_last_1))

        layer2_in = (meta_last_1 + self.state2_in(slow_state["meta_state_2"])).unsqueeze(1)
        layer2_out = _run_mamba_with_padding(self.layer2, layer2_in, self.chunk_size)
        meta_last = layer2_out[:, -1, :]
        next_state_2 = self.state2_norm(slow_state["meta_state_2"] + self.state2_out(meta_last))
        know_gap = torch.sigmoid(self.know_gap_head(meta_last))
        c_t = self.c_t_head(meta_last)
        if self.uncertainty_head is not None:
            uncertainty = torch.sigmoid(self.uncertainty_head(meta_last))
        else:
            uncertainty = torch.zeros_like(know_gap)
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
        self.in_proj = nn.Linear(config.c_t_dim + config.hidden_size + 1 + self.state_dim, self.state_dim, bias=False)
        self.norm = LumaZCRMSNorm(self.state_dim, eps=config.rms_norm_eps)
        self.state_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.score_head = nn.Linear(self.state_dim, 1, bias=False)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict:
        zeros = torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)
        return {"state": zeros, "score": torch.zeros(batch_size, 1, device=device, dtype=dtype)}

    def forward(self, c_t: torch.Tensor, delta_h: torch.Tensor, know_gap: torch.Tensor, prev_state: dict) -> dict:
        x = torch.cat([c_t, delta_h, know_gap, prev_state["state"]], dim=-1)
        next_state = torch.tanh(self.norm(self.in_proj(x)) + self.state_proj(prev_state["state"]))
        score = torch.sigmoid(self.score_head(next_state))
        return {"state": next_state, "score": score}


class LeWorldModelStyleJEPA(nn.Module):
    """Luma gives the world branch a fuller latent-model role here: mask contiguous regions, predict them from visible context, and regularize the latent geometry so it does not collapse.
    Luma 在这里把 world 分支提升为更完整的 latent world model：遮挡连续片段、依赖可见上下文去预测，并用潜空间正则避免坍缩。

    This is still an engineering translation for language-model hidden states rather than a claim of exact paper reproduction.
    这仍然是面向语言模型隐状态的工程迁移版，而不是宣称逐项复现论文实现。
    """

    def __init__(self, config: LumaConfig):
        super().__init__()
        self.world_dim = config.world_dim
        self.mask_ratio = config.world_mask_ratio
        self.mask_strategy = config.world_mask_strategy
        self.ema_decay = config.world_ema_decay
        self.simplify_loss = config.world_full_simplify_loss
        self.observer_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.online_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.world_dim, bias=False),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.world_dim, bias=False),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.world_dim))
        self.context_norm = LumaZCRMSNorm(config.world_dim * 3 + config.hidden_size, eps=config.rms_norm_eps)
        self.context_predictor = nn.Sequential(
            nn.Linear(config.world_dim * 3 + config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.world_dim, bias=False),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(config.world_dim * 2, config.world_dim, bias=False),
            nn.SiLU(),
            nn.Linear(config.world_dim, config.world_dim, bias=False),
        )
        self.enable_sigreg_world = bool(config.enable_sigreg_world)
        self.sigreg_weight = float(config.world_sigreg_weight)
        self.sigreg_world_source = str(config.sigreg_world_source)
        if self.sigreg_world_source not in {"sigreg_on_online", "sigreg_on_encoder_latent"}:
            self.sigreg_world_source = "sigreg_on_online"
        self.sigreg_world_fp32_only = bool(config.sigreg_world_fp32_only)
        self.sigreg_world_warmup_steps = int(max(0, config.sigreg_world_warmup_steps))
        self.runtime_sigreg_step = 0
        self.sigreg_num_slices = int(config.world_sigreg_num_slices)
        self.sigreg_eps = float(config.world_sigreg_eps)
        t_min = float(config.world_sigreg_t_min)
        t_max = float(config.world_sigreg_t_max)
        num_points = int(config.world_sigreg_num_points)
        if num_points < 2:
            num_points = 2
        t = torch.linspace(t_min, t_max, num_points, dtype=torch.float32)
        trap = torch.full((num_points,), (t_max - t_min) / max(1, num_points - 1), dtype=torch.float32)
        trap[0] *= 0.5
        trap[-1] *= 0.5
        weight = torch.exp(-t.square() / (2.0 * float(config.world_sigreg_lambda) ** 2))
        phi0 = torch.exp(-0.5 * t.square())
        self.register_buffer("sigreg_t", t, persistent=False)
        self.register_buffer("sigreg_weights", trap * weight, persistent=False)
        self.register_buffer("sigreg_phi0", phi0, persistent=False)
        self.delta_weight = float(config.world_delta_weight)
        self._copy_online_to_target()
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _copy_online_to_target(self) -> None:
        for target_param, online_param in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target_param.copy_(online_param)

    @torch.no_grad()
    def ema_update(self) -> None:
        """Luma updates the world target slowly so latent dynamics stay legible across noisy optimization steps.
        Luma 缓慢更新 world target，让潜空间动力学在噪声优化下仍然可读。
        """

        for target_param, online_param in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target_param.mul_(self.ema_decay).add_(online_param, alpha=1.0 - self.ema_decay)

    def _encode_online(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(self.observer_norm(hidden_states))

    def _encode_target(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target_encoder(self.observer_norm(hidden_states.detach()))

    def set_runtime_sigreg_step(self, step: int) -> None:
        self.runtime_sigreg_step = max(0, int(step))

    def summarize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._encode_online(hidden_states).mean(dim=1)

    def _surprise_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        centered = self.observer_norm(hidden_states)
        summary = centered.mean(dim=1, keepdim=True)
        return (centered - summary).norm(dim=-1)

    def _build_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        budget = max(1, int(seq_len * self.mask_ratio))
        if self.mask_strategy != "structured":
            span_len = budget
            for row in range(bsz):
                start = torch.randint(0, max(1, seq_len - span_len + 1), (1,), device=device).item()
                mask[row, start : start + span_len] = True
            return mask
        surprise = self._surprise_scores(hidden_states)
        primary_len = max(1, int(budget * 0.6))
        secondary_len = max(1, int(budget * 0.25))
        for row in range(bsz):
            center = int(surprise[row].argmax().item())
            start = max(0, min(seq_len - primary_len, center - primary_len // 2))
            mask[row, start : start + primary_len] = True
            remaining = budget - int(mask[row].sum().item())
            if remaining <= 0:
                continue
            ranked = surprise[row].argsort(descending=True)
            second_center = None
            for idx in ranked.tolist():
                if not mask[row, idx]:
                    second_center = idx
                    break
            if second_center is not None:
                second_len = min(secondary_len, remaining)
                second_start = max(0, min(seq_len - second_len, second_center - second_len // 2))
                mask[row, second_start : second_start + second_len] = True
            remaining = budget - int(mask[row].sum().item())
            if remaining > 0:
                ranked = surprise[row].argsort(descending=True)
                for idx in ranked.tolist():
                    if not mask[row, idx]:
                        mask[row, idx] = True
                        remaining -= 1
                        if remaining <= 0:
                            break
        return mask

    def _build_probe_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=hidden_states.device)
        budget = max(1, int(seq_len * self.mask_ratio))
        if self.mask_strategy != "structured":
            start = max(0, (seq_len - budget) // 2)
            mask[:, start : start + budget] = True
            return mask
        surprise = self._surprise_scores(hidden_states)
        topk = surprise.topk(budget, dim=-1).indices
        mask.scatter_(1, topk, True)
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
        observed_hidden = self.observer_norm(hidden_states)
        online_world = self.online_encoder(observed_hidden)
        target_world = self._encode_target(hidden_states)
        visible = (~mask).unsqueeze(-1).to(hidden_states.dtype)
        visible_count = visible.sum(dim=1).clamp_min(1.0)
        masked_count = mask.unsqueeze(-1).to(hidden_states.dtype).sum(dim=1).clamp_min(1.0)
        visible_summary = (online_world * visible).sum(dim=1, keepdim=True) / visible_count.unsqueeze(-1)
        target_mask_summary = (target_world * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / masked_count.unsqueeze(-1)
        hidden_summary = hidden_states.mean(dim=1, keepdim=True)
        masked_world = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(online_world), online_world)
        predictor_input = torch.cat(
            [
                masked_world,
                visible_summary.expand_as(online_world),
                target_mask_summary.expand_as(online_world),
                hidden_summary.expand(-1, hidden_states.shape[1], -1),
            ],
            dim=-1,
        )
        predictor_input = self.context_norm(predictor_input)
        pred_base = self.context_predictor(predictor_input)
        pred_delta = self.delta_head(torch.cat([pred_base, visible_summary.expand_as(pred_base)], dim=-1))
        pred_world = pred_base + pred_delta

        masked_pred = pred_world[mask]
        masked_target = target_world[mask]
        cosine_loss = 1.0 - F.cosine_similarity(
            F.normalize(masked_pred, dim=-1),
            F.normalize(masked_target, dim=-1),
            dim=-1,
        ).mean()
        sigreg_source = online_world if self.sigreg_world_source == "sigreg_on_online" else observed_hidden
        sigreg_source_float = sigreg_source.float()
        sigreg_source_mean = sigreg_source_float.mean()
        sigreg_source_std = sigreg_source_float.std(unbiased=False)
        sigreg_enabled = self.enable_sigreg_world and self.runtime_sigreg_step >= self.sigreg_world_warmup_steps
        sigreg_loss = self._sigreg(sigreg_source) if sigreg_enabled else online_world.new_zeros(())
        sigreg_loss_step = online_world.new_tensor(float(self.runtime_sigreg_step if sigreg_enabled else -1.0))
        world_loss = cosine_loss + self.sigreg_weight * sigreg_loss
        if not self.simplify_loss:
            delta_target = (masked_target - visible_summary.expand_as(target_world)[mask]).detach()
            delta_loss = F.mse_loss(pred_delta[mask], delta_target)
            world_loss = world_loss + self.delta_weight * delta_loss
        surprise_value = self._surprise_scores(hidden_states)[mask].float().mean() if mask.any() else hidden_states.new_zeros(())
        return {
            "world_mask": mask,
            "world_online": online_world,
            "world_target": target_world,
            "world_pred": pred_world,
            "world_jepa_loss": world_loss,
            "world_sigreg_loss": sigreg_loss,
            "world_sigreg_source_mean": sigreg_source_mean,
            "world_sigreg_source_std": sigreg_source_std,
            "world_sigreg_loss_step": sigreg_loss_step,
            "world_surprise": surprise_value,
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
        self.observer_norm = LumaZCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.online_observer = nn.Linear(config.hidden_size, config.world_dim, bias=False)
        self.target_observer = nn.Linear(config.hidden_size, config.world_dim, bias=False)
        self.predictor_norm = LumaZCRMSNorm(config.hidden_size + config.world_dim, eps=config.rms_norm_eps)
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_size + config.world_dim, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.world_dim, bias=False),
        )
        self._copy_online_to_target()
        for param in self.target_observer.parameters():
            param.requires_grad = False

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
        scores = torch.rand(bsz, seq_len, device=device)
        mask_count = max(1, int(seq_len * self.mask_ratio))
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
        online_world = self.online_observer(observed_hidden)
        with torch.no_grad():
            target_world = self.target_observer(observed_hidden.detach())
        visible = (~mask).unsqueeze(-1).to(hidden_states.dtype)
        visible_count = visible.sum(dim=1).clamp_min(1.0)
        visible_summary = (online_world * visible).sum(dim=1, keepdim=True) / visible_count.unsqueeze(-1)
        predictor_input = torch.cat([observed_hidden, visible_summary.expand_as(online_world)], dim=-1)
        predictor_input = self.predictor_norm(predictor_input)
        pred_world = self.predictor(predictor_input)

        masked_pred = pred_world[mask]
        masked_target = target_world[mask]
        world_loss = 1.0 - F.cosine_similarity(
            F.normalize(masked_pred, dim=-1),
            F.normalize(masked_target, dim=-1),
            dim=-1,
        ).mean()
        return {
            "world_mask": mask,
            "world_online": online_world,
            "world_target": target_world,
            "world_pred": pred_world,
            "world_jepa_loss": world_loss,
            "world_sigreg_loss": hidden_states.new_zeros(()),
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
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        prev_h: Optional[torch.Tensor],
        h: torch.Tensor,
        loop_idx: int,
        self_error: torch.Tensor,
        rollout_error: torch.Tensor,
        world_error: torch.Tensor,
        self_check_score: Optional[torch.Tensor] = None,
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
    ) -> dict:
        if prev_h is None:
            delta_h = h.new_tensor(1.0)
        else:
            delta_h = (h - prev_h).norm(dim=-1).mean() / (prev_h.norm(dim=-1).mean() + 1e-8)
        delta_signal = 1.0 - delta_h.clamp(0.0, 1.0)
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
            - self.gain_weight * predicted_gain
        )
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
        elif use_sampling:
            should_exit = bool(torch.bernoulli(sampled_exit_score).item() > 0.0)
        else:
            should_exit = bool(exit_score.item() > self.score_threshold)
        return {
            "delta_h": delta_h,
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
        self.diff_attn = GatedDiffAttnFoXSWA(config)
        self.ffn = LumaSwiGLUFFN(config.hidden_size, config.reason_intermediate_size, eps=config.rms_norm_eps)
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

    def _apply_film(self, h: torch.Tensor, c_t: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        film = proj(c_t).unsqueeze(1)
        scale, shift = film.chunk(2, dim=-1)
        return h * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(shift)

    def forward(
        self,
        h: torch.Tensor,
        c_t: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if c_t is not None and self.ct_modulation_mode == "film" and self.mamba_film is not None:
            h = self._apply_film(h, c_t, self.mamba_film)
        mamba_out = self.mamba(h)
        if c_t is not None and self.ct_modulation_mode == "modulewise_gate" and self.mamba_gate is not None:
            mamba_gate = torch.sigmoid(self.mamba_gate(c_t)).unsqueeze(1)
            h = h + mamba_gate * (mamba_out - h)
        else:
            h = mamba_out
        if c_t is not None and self.ct_modulation_mode == "film" and self.attn_film is not None:
            h = self._apply_film(h, c_t, self.attn_film)
        attn_out = self.diff_attn(h, attn_bias=attn_bias)
        if c_t is not None and self.ct_modulation_mode == "modulewise_gate" and self.attn_gate is not None:
            attn_gate = torch.sigmoid(self.attn_gate(c_t)).unsqueeze(1)
            h = h + attn_gate * (attn_out - h)
        else:
            h = attn_out
        if c_t is not None and self.ct_modulation_mode == "film" and self.ffn_film is not None:
            h = self._apply_film(h, c_t, self.ffn_film)
        ffn_out = self.ffn(h)
        if c_t is not None and self.ct_modulation_mode == "modulewise_gate" and self.ffn_gate is not None:
            ffn_gate = torch.sigmoid(self.ffn_gate(c_t)).unsqueeze(1)
            h = h + ffn_gate * (ffn_out - h)
        else:
            h = ffn_out
        return h


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
        self.routing_tier_soft_only = bool(config.routing_tier_soft_only)
        self.routing_tier_entropy_floor = float(config.routing_tier_entropy_floor)
        self.routing_min_local_share = float(config.routing_min_local_share)
        self.routing_progress_weight = float(config.routing_progress_weight)
        self.ct_injection = CTInjection(config.c_t_dim, config.hidden_size)
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

        c_bias = torch.tanh(self.ct_injection.proj(c_t)).unsqueeze(1)
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

    def forward(
        self,
        h: torch.Tensor,
        c_t: torch.Tensor,
        r_t: Optional[torch.Tensor] = None,
        r_trust: Optional[torch.Tensor] = None,
        r_t_mode: str = "blend",
        disable_ct_injection: bool = False,
        modulation_context: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        c_bias = self.ct_injection.proj(c_t).unsqueeze(1)
        if disable_ct_injection:
            c_bias = torch.zeros_like(c_bias)
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
                route_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(h.shape[-1])
                route_attn = torch.softmax(route_scores.float(), dim=-1).to(h.dtype)
                route_summary = torch.matmul(route_attn, v).squeeze(1)
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
        for layer in self.shared_layers:
            h = layer(h, c_t=c_t, attn_bias=attn_bias)
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
        self.unified_attnres = UnifiedAttnRes(config.hidden_size, eps=config.rms_norm_eps)
        self.introspection_state_stream = IntrospectionStateStream(config)
        # The introspection stream produces `know_gap` and `c_t`.
        # SelfJEPAResidualPredictor is a separate prediction head on top of that stream.
        # 自省流负责产出 `know_gap` 和 `c_t`。
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
        self.self_check_ring = TinySlowSelfCheckRing(config) if config.enable_self_check_ring else None
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
        )
        self.enable_sigreg_rollout = bool(config.enable_sigreg_rollout)
        self.enable_sigreg_delta = bool(config.enable_sigreg_delta)
        self.sigreg_rollout_weight = float(config.sigreg_rollout_weight)
        self.sigreg_delta_weight = float(config.sigreg_delta_weight)
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

    def forward(self, input_ids: torch.Tensor, disable_ct_injection: bool = False) -> Tuple[torch.Tensor, dict]:
        h = self.embedding(input_ids)
        h, block_reprs, compression_diag = self.compression(h)
        # Phase 2 auxiliary loss: expose compression output (stripped of memory tokens)
        # so an external probe can give the compression zone its own gradient signal.
        n_comp_mem = 8  # 4 local + 4 global memory tokens prepended by CompressionZone
        compression_h = h[:, n_comp_mem:, :]  # (batch, seq_len, hidden) — aligned with input_ids
        batch_size = h.shape[0]
        h = torch.cat([self.reason_memory(batch_size), h], dim=1)
        streams = self.mhc.init_streams(h)
        loop_history: List[torch.Tensor] = []
        c_t_history: List[torch.Tensor] = []
        know_gap_history: List[torch.Tensor] = []
        uncertainty_history: List[torch.Tensor] = []
        slow_update_flags: List[bool] = []
        prev_h: Optional[torch.Tensor] = None
        slow_state = self.introspection_state_stream.init_slow_state(batch_size, h.device, h.dtype)
        self_check_state = self.self_check_ring.init_state(batch_size, h.device, h.dtype) if self.self_check_ring is not None else None
        reasoning_state = self.reasoning_state_ring.init_state(batch_size, h.device, h.dtype) if self.reasoning_state_ring is not None else None
        c_t = slow_state["c_t"]
        know_gap = slow_state["know_gap"]
        uncertainty = slow_state.get("uncertainty", h.new_zeros((batch_size, 1)))
        self_check_score = h.new_full((batch_size, 1), 0.5)
        pred_delta_c = torch.zeros_like(c_t)
        target_delta_c = torch.zeros_like(c_t)
        self_jepa_terms: List[torch.Tensor] = []
        residual_reg_terms: List[torch.Tensor] = []
        sigreg_delta_terms: List[torch.Tensor] = []
        sigreg_rollout_terms: List[torch.Tensor] = []
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

        for loop_idx in range(self.config.reason_active_loops):
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
            h = self.unified_attnres(h, loop_history, block_reprs)
            did_slow_update = (loop_idx % self.config.slow_k == 0)
            if did_slow_update:
                slow_step_idx += 1
                prev_c_t = c_t
                current_loop_progress = h.new_full(
                    (batch_size, 1),
                    float(loop_idx + 1) / float(max(1, self.config.reason_active_loops)),
                )
                know_gap, next_c_t, slow_state = self.introspection_state_stream(
                    h,
                    block_reprs,
                    slow_state,
                    loop_progress=current_loop_progress,
                    loop_index=loop_idx,
                )
                delta_h = h.mean(dim=1).detach() if prev_h is None else (h - prev_h).mean(dim=1).detach()
                rollout_delta_preds, rollout_state_preds = self.self_jepa_residual_predictor.rollout(
                    c_t,
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
                c_t = next_c_t
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
                    progress_pred = self.self_jepa_progress_shape_head(
                        c_t,
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
                            c_t,
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
                    self_check_state = self.self_check_ring(c_t, delta_h, know_gap, self_check_state)
                    self_check_score = self_check_state["score"]
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
            c_t_history.append(c_t.detach())
            know_gap_history.append(know_gap.detach())
            uncertainty_history.append(uncertainty.detach())
            slow_update_flags.append(did_slow_update)
            executed_loops = loop_idx + 1
            exit_stats = self.exit_controller(
                prev_h,
                h,
                loop_idx,
                current_self_error,
                current_rollout_error,
                current_world_error,
                self_check_score=self_check_score.mean(),
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
            )
            delta_h_history.append(exit_stats["delta_h"].detach())
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
            if exit_stats["should_exit"]:
                break

        h = self.final_norm(h)
        world_aux = self.world_latent_jepa(h) if self.config.enable_world_jepa else self.world_latent_jepa.disabled_outputs(h)
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
        if self.enable_sigreg_delta and self.sigreg_delta_weight > 0.0:
            self_jepa_loss = self_jepa_loss + self.sigreg_delta_weight * sigreg_delta_loss
        if self.enable_sigreg_rollout and self.sigreg_rollout_weight > 0.0:
            self_jepa_loss = self_jepa_loss + self.sigreg_rollout_weight * sigreg_rollout_loss
        if trajectory_health_terms:
            self_jepa_loss = self_jepa_loss + torch.stack(trajectory_health_terms).mean()
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
            "self_jepa_loss": self_jepa_loss,
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
            "exit_aux_loss": h.new_zeros(()),
            "slow_state": {
                "meta_state_1": slow_state["meta_state_1"].detach(),
                "meta_state_2": slow_state["meta_state_2"].detach(),
            },
            "self_check_state": None if self_check_state is None else self_check_state["state"].detach(),
            "self_check_score": self_check_score.detach(),
            "executed_loops": executed_loops,
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
        del kwargs
        sigreg_step = self._runtime_train_step if labels is not None else self._runtime_train_step
        self.model.world_latent_jepa.set_runtime_sigreg_step(sigreg_step)
        if labels is not None:
            self._runtime_train_step += 1
        hidden_states, aux = self.model(input_ids, disable_ct_injection=disable_ct_injection)
        self.last_aux = aux
        logits = self.lm_head(self.pre_lm_norm(hidden_states))
        loss = None
        if labels is not None:
            token_logits = logits[:, -labels.shape[1] :, :]
            x = token_logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
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
            aux["world_jepa_loss_effective"] = world_term.detach()
            aux["self_jepa_loss_effective"] = self_jepa_term.detach()
            aux["self_rollout_loss_effective"] = self_rollout_term.detach()
            aux["exit_aux_loss_effective"] = exit_aux_term.detach()
            loss = lm_loss + world_term + self_jepa_term + self_rollout_term + exit_aux_term + rollout_zone_loss + routing_entropy_loss + trajectory_vitality_loss + compression_dynamics_loss
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=(
                self.config.world_jepa_weight * aux["world_jepa_loss"]
                + (logits.new_zeros(()) if self.config.disable_self_jepa else self.config.self_jepa_weight * aux["self_jepa_loss"])
                + (logits.new_zeros(()) if self.config.disable_self_jepa else self.config.self_rollout_weight * aux["self_rollout_loss"])
                + self.config.exit_aux_weight * aux["exit_aux_loss"]
                + aux.get("rollout_activity_zone_loss", logits.new_zeros(()))
                + aux.get("routing_entropy_loss", logits.new_zeros(()))
                + aux.get("trajectory_vitality_loss", logits.new_zeros(()))
                + aux.get("compression_dynamics_loss", logits.new_zeros(()))
            ),
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
        )
