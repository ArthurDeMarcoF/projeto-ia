"""Compara as configurações A, B e C, com inicialização independente."""

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.checkpoint import load_checkpoint
from src.data import DEFAULT_CORPUS
from src.generation import compare_strategies
from src.training import run_training, write_csv


CONFIGS = {
    "A": dict(epochs=2, batch_size=4, learning_rate=0.0004),
    "B": dict(epochs=5, batch_size=4, learning_rate=0.0004),
    "C": dict(epochs=5, batch_size=8, learning_rate=0.0004),
}


def plot_losses(path: Path, histories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
    markers = {"A": "o", "B": "x", "C": "s"}
    for axis, metric, title in zip(
        axes, ("train_loss", "validation_loss"), ("Treinamento", "Validação"),
    ):
        for name, history in histories.items():
            axis.plot(
                [row["epoch"] for row in history], [row[metric] for row in history],
                label=name, color=colors[name], marker=markers[name],
            )
        axis.set(title=title, xlabel="Época (0 = antes do treino)", ylabel="Cross-entropy (nats/token)")
        axis.set_xticks(range(max(len(history) for history in histories.values())))
        axis.grid(alpha=0.25)
        axis.legend(title="Experimento")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", choices=tuple(CONFIGS), nargs="+", default=list(CONFIGS))
    parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=Path("experimentos/resultados_sprint5"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)
    text = args.data.read_text(encoding="utf-8")
    histories = {}
    summary = []
    samples = []
    for name in dict.fromkeys(args.experiments):
        print(f"\nExperimento {name}", flush=True)
        checkpoint = args.output_dir / "checkpoints" / f"{name}.pt"
        history, info = run_training(text, checkpoint, device=args.device, **CONFIGS[name])
        histories[name] = history
        result = {"experiment": name, **info, "optimizer_steps": history[-1]["step"]}
        for phase, row in (("initial", history[0]), ("final", history[-1])):
            for metric in ("train_loss", "validation_loss", "train_perplexity", "validation_perplexity"):
                result[f"{phase}_{metric}"] = row[metric]
        summary.append(result)

        model, tokenizer, _ = load_checkpoint(checkpoint, args.device)
        samples.append(f"Experimento {name}\n" + compare_strategies(model, tokenizer, "Um modelo de linguagem"))

    write_csv(args.output_dir / "experimentos.csv", summary)
    write_csv(args.output_dir / "historico.csv", [
        {"experiment": name, **row}
        for name, history in histories.items()
        for row in history
    ])
    plot_losses(args.output_dir / "perdas.png", histories)
    (args.output_dir / "amostras.txt").write_text("\n".join(samples), encoding="utf-8")
    print(f"\nResultados gravados em {args.output_dir}.")


if __name__ == "__main__":
    main()
