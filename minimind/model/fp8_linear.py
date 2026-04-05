"""
FP8 Mixed-Precision Linear Layer for Luma.

策略:
  Forward:  weight + input 动态量化到 FP8-E4M3, 用 _scaled_mm tensor core GEMM
  Backward: 从 FP8 saved tensors 反量化回 bf16 做 matmul
  Master:   weights 保留 bf16, optimizer 用 bf16

VRAM 节省:
  - Saved activations 用 FP8 (1 byte vs bf16 2 bytes → 50% activation memory)
  - weight 不需要额外保存 (parameter 本身就在，backward 直接引用)
  - _scaled_mm FP8 tensor core 比 bf16 快 ~2x

Usage:
    from fp8_linear import convert_to_fp8
    convert_to_fp8(model)
"""

import torch
import torch.nn as nn
from torch import Tensor


E4M3 = torch.float8_e4m3fn


def _to_fp8_e4m3(x: Tensor):
    """动态量化�� FP8-E4M3，返回 (fp8_tensor, scale_inv: float32 scalar)。"""
    amax = x.detach().abs().amax()
    fp8_max = torch.finfo(E4M3).max  # 448.0
    scale = (fp8_max / amax.clamp(min=1e-12)).float()
    x_fp8 = (x.float() * scale).clamp(
        min=torch.finfo(E4M3).min, max=torch.finfo(E4M3).max
    ).to(E4M3)
    return x_fp8, scale.reciprocal()


class _FP8Matmul(torch.autograd.Function):
    """FP8 forward, backward 从 FP8 activation 反量���回 bf16。

    内存优化: save_for_backward 只保存 FP8 activation (1 byte/elem) + scale (scalar)，
    不保存 bf16 activation (2 bytes/elem)。weight 通过 forward 参数直接引用，不额外保存。
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None):
        # Forward: FP8 GEMM
        x_fp8, x_sinv = _to_fp8_e4m3(x)
        w_fp8, w_sinv = _to_fp8_e4m3(weight)

        out = torch._scaled_mm(
            x_fp8, w_fp8.t(),
            scale_a=x_sinv, scale_b=w_sinv,
            out_dtype=torch.bfloat16,
        )

        # 只保存 FP8 activation + scale (省内存), weight 通过闭包引用
        ctx.save_for_backward(x_fp8, x_sinv)
        ctx._weight_ref = weight  # 不进 saved_tensors，避免重复保存
        ctx.has_bias = bias is not None

        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_fp8, x_sinv = ctx.saved_tensors
        weight = ctx._weight_ref

        grad = grad_output.to(torch.bfloat16)

        # 反量化 x: fp8 → bf16
        x_bf16 = x_fp8.to(torch.bfloat16) * x_sinv

        # dX = grad @ W  (bf16)
        grad_x = grad.mm(weight)
        # dW = grad^T @ X  (bf16, 用反量化的 x)
        grad_w = grad.t().mm(x_bf16)

        grad_bias = grad.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, grad_bias


class FP8Linear(nn.Module):
    """nn.Linear drop-in: forward FP8 tensor core, backward bf16 from dequantized FP8."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16)
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16)
        ) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).to(torch.bfloat16) if x.ndim > 2 else x.to(torch.bfloat16)
        out = _FP8Matmul.apply(x2d, self.weight, self.bias)
        return out.reshape(*shape[:-1], self.out_features) if x.ndim > 2 else out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, fp8=True"

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8Linear":
        """从 nn.Linear 就地转换（共享权重，零拷���）。"""
        fp8 = cls(linear.in_features, linear.out_features,
                  bias=linear.bias is not None,
                  device=linear.weight.device, dtype=linear.weight.dtype)
        fp8.weight = linear.weight
        if linear.bias is not None:
            fp8.bias = linear.bias
        return fp8


def convert_to_fp8(model: nn.Module, min_size: int = 4096) -> int:
    """In-place 替换 nn.Linear → FP8Linear。

    跳过: 参数量 < min_size, 或维度不是 16 的倍数 (_scaled_mm 对齐要求)。
    """
    count = 0
    skipped = 0
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child.weight.numel() >= min_size:
                if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                    skipped += 1
                    continue
                setattr(module, attr_name, FP8Linear.from_linear(child))
                count += 1
    if skipped:
        import sys
        print(f"FP8: skipped {skipped} layers (dim not aligned to 16)", file=sys.stderr)
    return count
