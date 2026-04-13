"""FP8 saved_tensors_hooks for Mamba3 activation memory savings.

Phase E / Mamba 激活内存瓶颈解法（避开 triton kernel 重写）：
forward 结束时，把保存给 backward 的 bf16 tensor 量化成 fp8+scale 存储；
backward 前 dequantize 回 bf16 送入原 kernel。

原理：
- 量化只发生在 saved_tensors 存储阶段（即 save_for_backward 调用时刻）
- dequantize 发生在 unpack 阶段（backward 读取时刻）
- 前向计算路径零改动 → 数值和 bf16 一致
- 反向需要的激活从 2 bytes 降到 1 byte（大概 45-50% 节省）

per-block scale 粒度默认 128 elements 沿最后一维，outlier-robust。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

# Torch FP8 原生类型
_FP8_TYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_TYPE).max  # = 448.0


def _quantize_fp8_per_block(x: torch.Tensor, block_size: int = 128) -> tuple:
    """将 tensor 按最后一维每 block_size 个元素分组，各组独立 scale 量化到 fp8_e4m3fn。

    返回: (x_fp8, scale_fp32, orig_shape, orig_dtype, orig_device, block_size)
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    orig_device = x.device
    # 只对最后一维 pad / reshape
    D = orig_shape[-1]
    flat = x.reshape(-1, D)  # [N, D]
    N = flat.shape[0]
    pad = (block_size - D % block_size) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    Dp = flat.shape[-1]
    blocks = flat.view(N, Dp // block_size, block_size)  # [N, G, B]
    absmax = blocks.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)  # [N, G, 1]
    scale = absmax / _FP8_MAX  # per-block scale, bf16 范围内
    x_fp8 = (blocks.float() / scale).to(_FP8_TYPE)  # [N, G, B]
    return (x_fp8, scale.to(torch.float32).squeeze(-1), orig_shape, orig_dtype, pad, block_size)


def _dequantize_fp8_per_block(packed: tuple) -> torch.Tensor:
    """反量化：fp8_block + scale → 原 dtype。"""
    x_fp8, scale, orig_shape, orig_dtype, pad, block_size = packed
    # scale: [N, G]; x_fp8: [N, G, B]
    blocks_f = x_fp8.float() * scale.unsqueeze(-1)  # [N, G, B]
    N, G, B = blocks_f.shape
    flat = blocks_f.view(N, G * B)  # [N, Dp]
    if pad:
        flat = flat[:, : -pad]
    return flat.reshape(orig_shape).to(orig_dtype)


class Fp8ActivationContext:
    """torch.autograd.graph.saved_tensors_hooks context manager.

    用法：
        with Fp8ActivationContext(block_size=128):
            y = model(x)
        y.backward()  # hook 自动触发 dequantize
    """

    def __init__(self, block_size: int = 128, min_numel: int = 1024, debug: bool = False):
        self.block_size = block_size
        # 小 tensor 不量化（overhead 大于收益）
        self.min_numel = min_numel
        self.debug = debug
        self._ctx = None
        self._pack_count = 0
        self._unpack_count = 0
        self._skipped_count = 0
        self._bytes_before = 0
        self._bytes_after = 0

    def _pack(self, x: torch.Tensor):
        self._pack_count += 1
        # 跳过小 tensor 和非浮点
        if (
            x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or x.numel() < self.min_numel
            or x.shape[-1] < self.block_size
        ):
            self._skipped_count += 1
            return ("raw", x)
        try:
            self._bytes_before += x.nbytes
            packed = _quantize_fp8_per_block(x, self.block_size)
            self._bytes_after += packed[0].nbytes + packed[1].nbytes
            return ("fp8", packed)
        except Exception:
            self._skipped_count += 1
            return ("raw", x)

    def _unpack(self, packed):
        self._unpack_count += 1
        kind, payload = packed
        if kind == "fp8":
            return _dequantize_fp8_per_block(payload)
        return payload

    def stats(self) -> dict:
        return {
            "pack_count": self._pack_count,
            "unpack_count": self._unpack_count,
            "skipped": self._skipped_count,
            "bytes_before_MB": self._bytes_before / 1e6,
            "bytes_after_MB": self._bytes_after / 1e6,
            "saved_MB": (self._bytes_before - self._bytes_after) / 1e6,
            "saved_pct": (
                (1 - self._bytes_after / self._bytes_before) * 100
                if self._bytes_before > 0 else 0
            ),
        }

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)


def smoke_test():
    """简单测试：bf16 tensor 量化→反量化后的最大相对误差。"""
    torch.manual_seed(42)
    x = torch.randn(8, 768, device="cuda", dtype=torch.bfloat16) * 3.0
    packed = _quantize_fp8_per_block(x, block_size=128)
    y = _dequantize_fp8_per_block(packed)
    diff = (x.float() - y.float()).abs()
    rel = diff / x.float().abs().clamp_min(1e-6)
    print(f"x shape: {x.shape} dtype={x.dtype}")
    print(f"diff: max={diff.max().item():.4f} mean={diff.mean().item():.6f}")
    print(f"rel:  max={rel.max().item():.4f} mean={rel.mean().item():.4f}")
    print(f"fp8 nbytes: {packed[0].nbytes} scale nbytes: {packed[1].nbytes}")
    print(f"orig bf16 nbytes: {x.nbytes}")
    print(f"savings: {(1 - (packed[0].nbytes + packed[1].nbytes) / x.nbytes) * 100:.1f}%")


if __name__ == "__main__":
    smoke_test()
