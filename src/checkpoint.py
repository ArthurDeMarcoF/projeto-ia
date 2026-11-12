"""Persistência dos pesos, da arquitetura e do vocabulário da Sprint 5."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from src.gpt import GPTModel
from src.tokenization import SimpleTokenizerV2


def _validate_vocab(tokenizer: SimpleTokenizerV2, vocab_size: int) -> None:
    if sorted(tokenizer.str_to_int.values()) != list(range(vocab_size)):
        raise ValueError("Os IDs do vocabulário são incompatíveis com a arquitetura.")


def save_checkpoint(
    path: Path,
    model: GPTModel,
    tokenizer: SimpleTokenizerV2,
    architecture: dict,
    training: dict,
    epoch: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    _validate_vocab(tokenizer, architecture["vocab_size"])
    payload = {
        "model_state_dict": model.state_dict(),
        "architecture": architecture,
        "tokenizer_vocab": tokenizer.str_to_int,
        "training": training,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path, device: str | torch.device = "cpu",
) -> tuple[GPTModel, SimpleTokenizerV2, dict]:
    """Recupera o GPT em modo de avaliação e o mapeamento original token → ID."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    tokenizer = SimpleTokenizerV2(payload["tokenizer_vocab"])
    _validate_vocab(tokenizer, architecture["vocab_size"])
    model = GPTModel(**architecture)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), tokenizer, payload
