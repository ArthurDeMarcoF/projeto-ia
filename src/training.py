"""Pré-treinamento do GPT no corpus local, com avaliação por época."""

import argparse
import csv
import hashlib
import math
from pathlib import Path
from time import perf_counter

import torch

from src.checkpoint import save_checkpoint
from src.data import DEFAULT_CORPUS, create_dataloader_v1, split_corpus
from src.evaluation import batch_loss, evaluate, make_evaluation_loaders
from src.gpt import GPTModel
from src.tokenization import create_simple_tokenizer


def run_training(
    text: str,
    checkpoint_path: Path,
    *,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 0.0004,
    context_length: int = 12,
    seed: int = 123,
    device: str | torch.device | None = None,
) -> tuple[list[dict], dict]:
    """Inicializa, treina e salva um GPT; retorna as métricas e os dados da execução."""
    if min(epochs, batch_size, context_length) < 1:
        raise ValueError("Épocas, batch size e contexto devem ser positivos.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("O learning rate deve ser positivo e finito.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_fraction = 0.8
    train_stride = max(1, context_length // 3)
    eval_batch_size = 8
    train_text, validation_text = split_corpus(text, train_fraction)
    tokenizer = create_simple_tokenizer(train_text)
    train_eval_loader, validation_loader = make_evaluation_loaders(
        train_text, validation_text, tokenizer, context_length, eval_batch_size,
    )
    train_loader = create_dataloader_v1(
        train_text, tokenizer, batch_size=batch_size, max_length=context_length,
        stride=train_stride, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    architecture = dict(
        vocab_size=tokenizer.vocab_size, context_length=context_length, embedding_dim=32,
        num_heads=4, num_layers=2, ff_hidden_dim=128, dropout=0.1, qkv_bias=False,
    )
    # Cada chamada inicia do mesmo estado quando arquitetura e seed coincidem.
    torch.manual_seed(seed)
    model = GPTModel(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    training = dict(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        optimizer="AdamW", weight_decay=0.01, seed=seed, device=str(device),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        train_fraction=train_fraction, train_stride=train_stride, eval_batch_size=eval_batch_size,
        train_windows=len(train_loader.dataset), train_eval_windows=len(train_eval_loader.dataset),
        validation_windows=len(validation_loader.dataset), steps_per_epoch=len(train_loader),
        corpus_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        torch_version=str(torch.__version__), cpu_threads=torch.get_num_threads(),
    )
    print(f"Arquitetura = {architecture}\nTreinamento = {training}", flush=True)

    start = perf_counter()
    history = [{"epoch": 0, "step": 0, **evaluate(model, train_eval_loader, validation_loader)}]
    print(f"Inicial | treino {history[0]['train_loss']:.4f} | validação {history[0]['validation_loss']:.4f}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(model, inputs.to(device), targets.to(device))
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, train_eval_loader, validation_loader)
        history.append({"epoch": epoch, "step": epoch * len(train_loader), **metrics})
        print(
            f"Época {epoch}/{epochs} | treino {metrics['train_loss']:.4f}"
            f" | validação {metrics['validation_loss']:.4f}",
            flush=True,
        )

    seconds = perf_counter() - start
    save_checkpoint(checkpoint_path, model, tokenizer, architecture, training, epochs, optimizer)
    return history, {**architecture, **training, "seconds": seconds}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o GPT e salva um checkpoint e o histórico CSV.")
    parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0004)
    parser.add_argument("--context-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("experimentos/treino_sprint5"))
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)
    checkpoint = args.output_dir / "modelo.pt"
    history, info = run_training(
        args.data.read_text(encoding="utf-8"), checkpoint, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        context_length=args.context_length, seed=args.seed, device=args.device,
    )
    history_path = args.output_dir / "historico.csv"
    write_csv(history_path, history)
    print(f"Treinamento e avaliações em {info['seconds']:.2f} s.\n{checkpoint}\n{history_path}")


if __name__ == "__main__":
    main()
