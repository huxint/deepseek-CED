"""From-scratch next-token training with validation, mixed precision and resumable checkpoints."""

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self, TypedDict

import torch
from torch.amp.grad_scaler import GradScaler

from .config import ModelConfig
from .data import TokenCorpus
from .model import CEDLanguageModel
from .runtime import (
    Checkpoint,
    Precision,
    PrecisionRequest,
    autocast_context,
    load_checkpoint,
    resolve_device,
    resolve_precision,
    save_checkpoint,
)
from .tokenizer import ByteTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainOptions:
    config: Path = Path("configs/micro.json")
    data: Path = Path("data/demo")
    out: Path = Path("runs/micro")
    resume: Path | None = None
    steps: int = 500
    batch_size: int = 8
    seq_len: int = 128
    grad_accum: int = 1
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 20
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 5
    log_interval: int = 10
    device: str = "auto"
    precision: PrecisionRequest = "auto"
    threads: int = 4
    seed: int = 1337

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> Self:
        return cls(**vars(build_parser().parse_args(argv)))


class TrainingSummary(TypedDict):
    step: int
    initial_val_loss: float
    best_val_loss: float


def learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    min_ratio: float,
) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(
    model: CEDLanguageModel,
    corpus: TokenCorpus,
    args: TrainOptions,
    device: torch.device,
    precision: Precision,
) -> float:
    generator = torch.Generator().manual_seed(args.seed + 1)
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(args.eval_batches):
        x, y = corpus.batch("val", args.batch_size, args.seq_len, generator, device)
        with autocast_context(device, precision):
            losses.append(model(x, y).loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, help="total target steps, including resumed steps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min-lr-ratio", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"))
    parser.add_argument("--threads", type=int, help="PyTorch CPU worker threads")
    parser.add_argument("--seed", type=int)
    parser.set_defaults(**asdict(TrainOptions()))
    return parser


def run_training(args: TrainOptions) -> TrainingSummary:
    config = ModelConfig.from_json(args.config)
    if config.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError("the byte tokenizer requires vocab_size=258")
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    corpus = TokenCorpus(args.data)
    corpus.check_length(args.seq_len)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        if checkpoint["model_config"] != config.to_dict():
            raise ValueError("resume model config differs from the checkpoint")
        if checkpoint["data_metadata"] != corpus.metadata:
            raise ValueError("resume data differs from the checkpoint")
        for name in (
            "seq_len",
            "batch_size",
            "grad_accum",
            "lr",
            "min_lr_ratio",
            "warmup_steps",
            "weight_decay",
            "seed",
            "max_grad_norm",
        ):
            if checkpoint["train_args"][name] != getattr(args, name):
                raise ValueError(
                    f"resume requires --{name.replace('_', '-')}={checkpoint['train_args'][name]}"
                )
        if args.steps <= checkpoint["step"]:
            raise ValueError("--steps must exceed the checkpoint's completed step")
    if not args.resume and (args.out / "last.pt").exists():
        raise FileExistsError(f"{args.out}/last.pt exists; use --resume or a different --out")
    args.out.mkdir(parents=True, exist_ok=True)
    model = CEDLanguageModel(config).to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": args.weight_decay},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scaler = GradScaler("cuda", enabled=precision == "fp16")
    start_step, best_val = 0, math.inf
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if precision == "fp16" and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step, best_val = checkpoint["step"], checkpoint["best_val_loss"]
        generator.set_state(checkpoint["rng"]["sampler"])
        torch.set_rng_state(checkpoint["rng"]["cpu"])
        if device.type == "cuda" and len(checkpoint["rng"]["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        del checkpoint
    train_args = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(args).items()}
    logger.info(
        "device=%s precision=%s params=%s tokens/update=%s",
        device,
        precision,
        f"{model.parameter_count():,}",
        f"{args.batch_size * args.seq_len * args.grad_accum:,}",
    )
    initial_val = evaluate(model, corpus, args, device, precision)
    logger.info("step=%d val_loss=%.4f", start_step, initial_val)
    model.train()
    accumulated_loss, accumulated_seconds, accumulated_steps = 0.0, 0.0, 0
    with (args.out / "metrics.jsonl").open("a" if args.resume else "w", encoding="utf-8") as log:
        for index in range(start_step, args.steps):
            started = time.perf_counter()
            rate = learning_rate(index, args.steps, args.warmup_steps, args.lr, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for _ in range(args.grad_accum):
                x, y = corpus.batch("train", args.batch_size, args.seq_len, generator, device)
                with autocast_context(device, precision):
                    loss = model(x, y).loss
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("non-finite loss; reduce --lr or use --precision fp32")
                train_loss += loss.detach().item() / args.grad_accum
                scaler.scale(loss / args.grad_accum).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                args.max_grad_norm,
                error_if_nonfinite=precision != "fp16",
            )
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            accumulated_seconds += time.perf_counter() - started
            accumulated_loss += train_loss
            accumulated_steps += 1
            step = index + 1
            validate = step % args.eval_interval == 0 or step == args.steps
            if step % args.log_interval == 0 or validate or step == 1:
                metrics: dict[str, int | float | None] = {
                    "step": step,
                    "train_loss": accumulated_loss / accumulated_steps,
                    "lr": rate,
                    "grad_norm": norm.item() if torch.isfinite(norm).item() else None,
                    "tokens_per_second": (
                        accumulated_steps
                        * args.batch_size
                        * args.seq_len
                        * args.grad_accum
                        / max(accumulated_seconds, 1e-9)
                    ),
                }
                if validate:
                    val_loss = evaluate(model, corpus, args, device, precision)
                    metrics["val_loss"] = val_loss
                    improved = val_loss < best_val
                    best_val = min(best_val, val_loss)
                    payload: Checkpoint = {
                        "format_version": 1,
                        "model_config": config.to_dict(),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val,
                        "val_loss": val_loss,
                        "train_args": train_args,
                        "data_metadata": corpus.metadata,
                        "torch_version": str(torch.__version__),
                        "rng": {
                            "cpu": torch.get_rng_state(),
                            "sampler": generator.get_state(),
                            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        },
                    }
                    save_checkpoint(args.out / "last.pt", payload)
                    if improved:
                        save_checkpoint(args.out / "best.pt", payload)
                    del payload
                text = json.dumps(metrics, allow_nan=False)
                log.write(text + "\n")
                log.flush()
                logger.info("%s", text)
                accumulated_loss, accumulated_seconds, accumulated_steps = 0.0, 0.0, 0
    logger.info(
        "Saved %s (step %d); best val_loss=%.4f", args.out / "last.pt", args.steps, best_val
    )
    return {"step": args.steps, "initial_val_loss": initial_val, "best_val_loss": best_val}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_training(TrainOptions.from_cli())


if __name__ == "__main__":
    main()
