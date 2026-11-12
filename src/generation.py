
"""Inferência autoregressiva com greedy decoding ou amostragem."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.checkpoint import load_checkpoint
from src.gpt import GPTModel
from src.tokenization import SimpleTokenizerV2


@torch.inference_mode()
def generate_tokens(
    model: GPTModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Acrescenta novos tokens à entrada; resultado [B, T + max_new_tokens].

    A cada passo, usa apenas os últimos `model.context_length` tokens;
    seleciona logits da última posição [B, V], aplica temperatura/top-k e
    escolhe o próximo ID por argmax ou torch.multinomial.

    A ausência de treinamento implica texto normalmente incoerente.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids deve ser [B, T] com T >= 1.")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens não pode ser negativo.")

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature deve ser positiva e finita.")

    if top_k is not None and (top_k < 1 or top_k > model.vocab_size):
        raise ValueError("top_k deve estar entre 1 e vocab_size.")

    was_training = model.training
    model.eval()

    try:
        result = input_ids.clone()

        for _ in range(max_new_tokens):
            context = result[:, -model.context_length :]
            logits = model(context)[:, -1, :]

            if do_sample:
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k, dim=-1)
                    logits = torch.full_like(
                        logits, float("-inf")
                    ).scatter(-1, indices, values)

                probabilities = torch.softmax(
                    logits / temperature, dim=-1
                )

                next_id = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)

            result = torch.cat((result, next_id), dim=1)

        return result

    finally:
        model.train(was_training)


def generate_text(
    model: GPTModel,
    tokenizer: SimpleTokenizerV2,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = False,
    seed: int = 123,
) -> str:
    device = next(model.parameters()).device

    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    generator = (
        torch.Generator(device=device).manual_seed(seed)
        if do_sample
        else None
    )

    output_ids = generate_tokens(
        model,
        input_ids,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        generator=generator,
    )

    return tokenizer.decode(output_ids[0].tolist())


def compare_strategies(
    model: GPTModel,
    tokenizer: SimpleTokenizerV2,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.2,
    top_k: int | None = None,
    seed: int = 123,
) -> str:
    if top_k is None:
        top_k = min(5, tokenizer.vocab_size)

    options = (
        ("Greedy Decoding", {}),
        (
            f"Temperature Sampling (T={temperature:g})",
            {"temperature": temperature, "do_sample": True},
        ),
        (
            f"Top-k Sampling (k={top_k}, T=1)",
            {"top_k": top_k, "do_sample": True},
        ),
    )

    lines = [
        f"Prompt = {prompt}\n"
        f"Novos tokens = {max_new_tokens} | seed = {seed}"
    ]

    for name, settings in options:
        result = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            seed=seed,
            **settings,
        )
        lines.append(f"{name}\n{result}")

    return "\n\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara três estratégias de geração com um checkpoint."
    )

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Um modelo de linguagem")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument(
        "--top-k",
        type=int,
        help="Padrão de 5, limitado pelo tamanho do vocabulário.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)

    model, tokenizer, _ = load_checkpoint(
        args.checkpoint,
        args.device,
    )

    print(
        compare_strategies(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.seed,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
