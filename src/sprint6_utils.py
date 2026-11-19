"""Dataset and evaluation helpers shared by both Sprint 6 fine-tuning tasks."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

EOS_ID = 50256


def partitions(items: list, seed: int = 123, train_fraction: float = .70,
               validation_fraction: float = .15) -> tuple[list, list, list]:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Invalid partition fractions")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n < 7:
        raise ValueError("At least 7 samples are required for a 3-way split")
    a = max(1, int(n * train_fraction))
    b = max(a + 1, int(n * (train_fraction + validation_fraction)))
    if b >= n:
        b = n - 1
    return shuffled[:a], shuffled[a:b], shuffled[b:]


def save_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def trainable_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()
            if param.requires_grad}


def apply_trainable_state(model: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    current = dict(model.named_parameters())
    if set(weights) != {name for name, p in current.items() if p.requires_grad}:
        raise ValueError("Adapter keys do not match the configured trainable parameters")
    with torch.no_grad():
        for name, value in weights.items():
            if current[name].shape != value.shape:
                raise ValueError("Adapter tensor shape mismatch: " + name)
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU was found")
    if requested == "cpu":
        torch.set_num_threads(1)
    return torch.device(requested)
