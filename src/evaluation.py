"""Cross-entropy e perplexidade para a previsão do próximo token."""

import argparse
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint
from src.data import DEFAULT_CORPUS, create_dataloader_v1, split_corpus
from src.gpt import GPTModel
from src.tokenization import SimpleTokenizerV2


def batch_loss(model: GPTModel, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Calcula a perda diferenciável de um lote; entradas e modelo no mesmo dispositivo."""
    logits = model(input_ids)
    # Cada linha de [B*T, V] prevê o próximo token de uma posição.
    # O Dataset já deslocou os alvos; cross_entropy aplica log-softmax internamente.
    return F.cross_entropy(logits.flatten(0, 1), target_ids.flatten())


@torch.no_grad()
def evaluate_loader(model: GPTModel, loader: DataLoader) -> float:
    """Calcula a perda média por token, incluindo o último lote mesmo se for menor."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for inputs, targets in loader:
            token_count = targets.numel()
            loss = batch_loss(model, inputs.to(device), targets.to(device))
            total_loss += loss.item() * token_count
            total_tokens += token_count
        if total_tokens == 0:
            raise ValueError("O DataLoader de avaliação não contém tokens.")
        return total_loss / total_tokens
    finally:
        model.train(was_training)


def make_evaluation_loaders(
    train_text: str,
    validation_text: str,
    tokenizer: SimpleTokenizerV2,
    context_length: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    options = dict(
        batch_size=batch_size, max_length=context_length, stride=context_length,
        shuffle=False, drop_last=False,
    )
    train_loader = create_dataloader_v1(train_text, tokenizer, **options)
    validation_loader = create_dataloader_v1(validation_text, tokenizer, **options)
    if not len(train_loader.dataset) or not len(validation_loader.dataset):
        raise ValueError("Treino e validação precisam conter mais tokens que o tamanho do contexto.")
    return train_loader, validation_loader


def perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf


def evaluate(
    model: GPTModel, train_loader: DataLoader, validation_loader: DataLoader,
) -> dict[str, float]:
    train_loss = evaluate_loader(model, train_loader)
    validation_loss = evaluate_loader(model, validation_loader)
    return {
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_perplexity": perplexity(train_loss),
        "validation_perplexity": perplexity(validation_loss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um checkpoint no corpus usado no treino.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, args.device)
    training = checkpoint["training"]
    text = args.data.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != training["corpus_sha256"]:
        parser.error("O corpus informado difere do usado para treinar o checkpoint.")
    train_text, validation_text = split_corpus(text, training["train_fraction"])
    train_loader, validation_loader = make_evaluation_loaders(
        train_text, validation_text, tokenizer, model.context_length, training["eval_batch_size"],
    )
    for name, value in evaluate(model, train_loader, validation_loader).items():
        print(f"{name} = {value:.6f}")


if __name__ == "__main__":
    main()
