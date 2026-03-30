import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset
from luma_stage0.optimizers import LumaCosineScheduler, LumaMuonAdamWOptimizer, LumaOptimizerConfig
from model.model_minimind import LumaConfig, LumaForCausalLM
from trainer.trainer_utils import Logger, SkipBatchSampler, init_distributed_mode, is_main_process, lm_checkpoint, setup_seed

warnings.filterwarnings("ignore")


def init_luma_model(luma_config: LumaConfig, tokenizer_path: str, from_weight: str, save_dir: str, device: str):
    """Luma enters formal pretraining through her own config path, not by pretending to be the old MiniMind skeleton.

    Luma 进入正式预训练时走的是自己的配置路径，而不是伪装成旧的 MiniMind 骨架。
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    tokenizer_vocab_size = len(tokenizer)
    if luma_config.vocab_size < tokenizer_vocab_size:
        luma_config.vocab_size = tokenizer_vocab_size
    model = LumaForCausalLM(luma_config)
    if from_weight != "none":
        weight_path = f"{save_dir}/{from_weight}_{luma_config.hidden_size}.pth"
        weights = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(weights, strict=False)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f"Luma Params: {total_params:.3f}M")
    Logger(f"Luma Trainable Params: {trainable_params:.3f}M")
    return model.to(device), tokenizer


def build_luma_config(args: argparse.Namespace) -> LumaConfig:
    """Luma keeps the formal pretrain defaults aligned with the current preferred stack: full world JEPA plus self-check.

    Luma 会把正式预训练默认值对齐到当前偏好的栈：full world JEPA 加 self-check。
    """

    return LumaConfig(
        vocab_size=args.vocab_size,
        factorized_vocab_dim=args.factorized_vocab_dim,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        reason_intermediate_size=args.reason_intermediate_size or args.intermediate_size,
        reason_shared_depth=args.reason_shared_depth,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        compression_layers=args.compression_layers,
        compression_active_layers=args.compression_active_layers or args.compression_layers,
        reason_loops=args.reason_loops,
        reason_loops_max=args.reason_loops_max,
        reason_active_loops=args.reason_active_loops or args.reason_loops,
        slow_k=args.slow_k,
        c_t_dim=args.c_t_dim,
        meta_dim=args.meta_dim,
        meta_state=args.meta_state,
        mamba_d_state=args.mamba_d_state,
        mamba_expand=args.mamba_expand,
        mamba_headdim=args.mamba_headdim,
        mamba_chunk_size=args.mamba_chunk_size,
        world_mask_ratio=args.world_mask_ratio,
        world_ema_decay=args.world_ema_decay,
        enable_world_jepa=bool(args.enable_world_jepa),
        world_jepa_mode=args.world_jepa_mode,
        enable_self_check_ring=bool(args.enable_self_check_ring),
        self_check_dim=args.self_check_dim,
        self_check_k=args.self_check_k,
        self_rollout_steps=args.self_rollout_steps,
        self_rollout_weight=args.self_rollout_weight,
        self_jepa_residual_reg=args.self_jepa_residual_reg,
        exit_train_use_sampling=bool(args.exit_train_use_sampling),
        exit_eval_use_sampling=bool(args.exit_eval_use_sampling),
        exit_sampling_temperature=args.exit_sampling_temperature,
        max_position_embeddings=args.max_seq_len,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
    )


def train_epoch(epoch, loader, iters, optimizer, scheduler, scaler, model, args, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss
            aux_loss = res.aux_loss if res.aux_loss is not None else loss.new_zeros(())
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer.matrix_optimizer)
            scaler.unscale_(optimizer.scalar_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            raw_model.model.update_world_jepa_ema()

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            current_logits_loss = current_loss - current_aux_loss
            matrix_lr = next(group["lr"] for group in optimizer.param_groups if group.get("optim_family") == "muon")
            scalar_lr = next(group["lr"] for group in optimizer.param_groups if group.get("optim_family") == "adamw")
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, "
                f"matrix_lr: {matrix_lr:.8f}, scalar_lr: {scalar_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "matrix_lr": matrix_lr,
                        "scalar_lr": scalar_lr,
                        "epoch_time": eta_min,
                    }
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            ckp = f"{args.save_dir}/{args.save_weight}_{luma_config.hidden_size}.pth"
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                luma_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=args.checkpoint_dir,
                optimizer_config=asdict(optimizer.config),
                trainer_args=vars(args),
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer.matrix_optimizer)
        scaler.unscale_(optimizer.scalar_optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Luma formal pretraining trainer")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="resume/checkpoint 目录")
    parser.add_argument("--tokenizer_path", type=str, default="../model/qwen3_5_tokenizer", help="tokenizer 路径")
    parser.add_argument("--save_weight", default="luma_pretrain", type=str, help="保存权重前缀")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--matrix_lr", type=float, default=0.02, help="Muon 学习率")
    parser.add_argument("--scalar_lr", type=float, default=3e-4, help="AdamW 学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--intermediate_size", default=3072, type=int, help="主 FFN 维度")
    parser.add_argument("--reason_intermediate_size", default=0, type=int, help="推理共享块 FFN 维度，0 表示跟随主 FFN")
    parser.add_argument("--reason_shared_depth", default=2, type=int, help="共享推理 block 内部的真实层数")
    parser.add_argument("--num_attention_heads", default=12, type=int, help="注意力头数")
    parser.add_argument("--num_key_value_heads", default=3, type=int, help="KV 头数")
    parser.add_argument("--compression_layers", default=24, type=int, help="压缩区总层数")
    parser.add_argument("--compression_active_layers", default=0, type=int, help="实际启用的压缩层数，0 表示全部")
    parser.add_argument("--reason_loops", default=15, type=int, help="推理循环数")
    parser.add_argument("--reason_loops_max", default=20, type=int, help="推理循环最大预算")
    parser.add_argument("--reason_active_loops", default=0, type=int, help="实际启用循环数，0 表示跟随 reason_loops")
    parser.add_argument("--slow_k", default=1, type=int, help="自省流慢环间隔")
    parser.add_argument("--self_check_k", default=2, type=int, help="self_check ring 更新间隔")
    parser.add_argument("--self_rollout_steps", default=10, type=int, help="self rollout 深度")
    parser.add_argument("--self_rollout_weight", default=0.5, type=float, help="rollout loss 权重")
    parser.add_argument("--self_jepa_residual_reg", default=0.01, type=float, help="self delta 正则")
    parser.add_argument("--world_jepa_mode", default="full", type=str, choices=["scaffold", "full"], help="world JEPA 模式")
    parser.add_argument("--enable_world_jepa", default=1, type=int, choices=[0, 1], help="是否启用 world JEPA")
    parser.add_argument("--enable_self_check_ring", default=1, type=int, choices=[0, 1], help="是否启用 self_check ring")
    parser.add_argument("--self_check_dim", default=16, type=int, help="self_check hidden 维度")
    parser.add_argument("--world_mask_ratio", default=0.25, type=float, help="world mask ratio")
    parser.add_argument("--world_ema_decay", default=0.99, type=float, help="world EMA decay")
    parser.add_argument("--meta_dim", default=96, type=int, help="自省流维度")
    parser.add_argument("--meta_state", default=32, type=int, help="自省流状态维度")
    parser.add_argument("--c_t_dim", default=64, type=int, help="认知状态维度")
    parser.add_argument("--mamba_d_state", default=192, type=int, help="Mamba d_state")
    parser.add_argument("--mamba_expand", default=2, type=int, help="Mamba expand")
    parser.add_argument("--mamba_headdim", default=64, type=int, help="Mamba head dim")
    parser.add_argument("--mamba_chunk_size", default=4, type=int, help="Mamba kernel chunk size")
    parser.add_argument("--max_seq_len", default=512, type=int, help="最大长度")
    parser.add_argument("--vocab_size", default=151936, type=int, help="词表大小")
    parser.add_argument("--factorized_vocab_dim", default=192, type=int, help="因子化词表维度")
    parser.add_argument("--bos_token_id", default=1, type=int, help="bos token id")
    parser.add_argument("--eos_token_id", default=2, type=int, help="eos token id")
    parser.add_argument("--muon_momentum", default=0.95, type=float, help="Muon momentum")
    parser.add_argument("--betas", nargs=2, default=(0.9, 0.95), type=float, help="AdamW betas")
    parser.add_argument("--eps", default=1e-8, type=float, help="AdamW eps")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="weight decay")
    parser.add_argument("--muon_clip_factor", default=1.0, type=float, help="MuonClip 因子")
    parser.add_argument("--modular_norm_power", default=0.5, type=float, help="Modular-Norm 缩放幂次")
    parser.add_argument("--use_8bit_muon", default=1, type=int, choices=[0, 1], help="是否启用实验性 8-bit Muon 状态")
    parser.add_argument("--use_8bit_adamw", default=1, type=int, choices=[0, 1], help="是否启用 AdamW8bit")
    parser.add_argument("--exit_train_use_sampling", default=1, type=int, choices=[0, 1], help="训练是否软退出")
    parser.add_argument("--exit_eval_use_sampling", default=0, type=int, choices=[0, 1], help="评估是否软退出")
    parser.add_argument("--exit_sampling_temperature", default=1.0, type=float, help="软退出温度")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument("--from_weight", default="none", type=str, help="基于哪个权重继续训练")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否启用 swanlab")
    parser.add_argument("--wandb_project", type=str, default="Luma-Pretrain", help="swanlab 项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用 torch.compile")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    luma_config = build_luma_config(args)
    ckp_data = lm_checkpoint(luma_config, weight=args.save_weight, save_dir=args.checkpoint_dir) if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb.init(
            project=args.wandb_project,
            name=f"Luma-Pretrain-H{args.hidden_size}-D{args.reason_shared_depth}-R{args.reason_loops}",
            id=wandb_id,
            resume=resume,
        )

    model, tokenizer = init_luma_model(luma_config, args.tokenizer_path, args.from_weight, args.save_dir, args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    optimizer_config = LumaOptimizerConfig(
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        muon_momentum=args.muon_momentum,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
        muon_clip_factor=args.muon_clip_factor,
        modular_norm_power=args.modular_norm_power,
        use_8bit_muon=bool(args.use_8bit_muon),
        use_8bit_adamw=bool(args.use_8bit_adamw),
    )
    optimizer = LumaMuonAdamWOptimizer(model, optimizer_config)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = math.ceil(len(train_ds) / max(args.batch_size * world_size, 1))
    total_optimizer_steps = max(1, math.ceil((steps_per_epoch * args.epochs) / args.accumulation_steps))
    scheduler = LumaCosineScheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        matrix_base_lr=args.matrix_lr,
        scalar_base_lr=args.scalar_lr,
    )

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        if "scheduler" in ckp_data:
            scheduler.load_state_dict(ckp_data["scheduler"])
        if "scaler" in ckp_data:
            scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个 step，从 step {start_step + 1} 开始")
            train_epoch(epoch, loader, len(loader) + skip, optimizer, scheduler, scaler, model, args, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), optimizer, scheduler, scaler, model, args, 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
