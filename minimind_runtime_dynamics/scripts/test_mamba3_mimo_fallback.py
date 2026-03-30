import torch

from model.mamba3_module import Mamba3Config, Mamba3Block


def run_train_backward(device: str = "cuda"):
    cfg = Mamba3Config(
        d_model=768,
        d_state=128,
        is_mimo=True,
        mimo_rank=4,
        chunk_size=16,
        train_use_siso_fallback=True,
        dropout=0.0,
    )
    block = Mamba3Block(cfg).to(device)
    block.train()
    x = torch.randn(1, 64, 768, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = block(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    return float(loss.detach().cpu())


def run_eval_forward(device: str = "cuda"):
    cfg = Mamba3Config(
        d_model=768,
        d_state=128,
        is_mimo=True,
        mimo_rank=4,
        chunk_size=16,
        train_use_siso_fallback=True,
        dropout=0.0,
    )
    block = Mamba3Block(cfg).to(device)
    block.eval()
    with torch.no_grad():
        x = torch.randn(1, 64, 768, device=device, dtype=torch.bfloat16)
        y = block(x)
    return tuple(y.shape)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test.")

    torch.cuda.set_device(0)
    train_loss = run_train_backward("cuda")
    print(f"train_backward_ok=True loss={train_loss:.6f}")

    shape = run_eval_forward("cuda")
    print(f"eval_mimo_forward_ok=True y_shape={shape}")
