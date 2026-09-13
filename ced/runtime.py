"""Device selection, precision and checkpoint I/O at the application boundary."""

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
from torch import Tensor

from .metadata import CorpusMetadata

type Precision = Literal["fp32", "bf16", "fp16"]
type PrecisionRequest = Precision | Literal["auto"]


class RandomState(TypedDict):
    cpu: Tensor
    sampler: Tensor
    cuda: list[Tensor]


class Checkpoint(TypedDict):
    format_version: int
    model_config: dict[str, Any]
    model: dict[str, Tensor]
    optimizer: dict[str, Any]
    scaler: dict[str, Any]
    step: int
    best_val_loss: float
    val_loss: float
    train_args: dict[str, Any]
    data_metadata: CorpusMetadata
    torch_version: str
    rng: RandomState


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    return device


def resolve_precision(name: PrecisionRequest, device: torch.device) -> Precision:
    if name == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    return name


def autocast_context(
    device: torch.device, precision: Precision
) -> torch.autocast | nullcontext[None]:
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=dtype)


def save_checkpoint(path: Path, payload: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path) -> Checkpoint:
    checkpoint: Checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint
