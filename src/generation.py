"""Inferência autoregressiva com greedy decoding ou amostragem."""

from __future__ import annotations

import torch

from src.gpt import GPTModel


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
    if temperature <= 0:
        raise ValueError("temperature deve ser maior que zero.")
    if top_k is not None and (top_k < 1 or top_k > model.vocab_size):
        raise ValueError("top_k deve estar entre 1 e vocab_size.")

    was_training = model.training
    model.eval()  # Desativa dropout; não altera permanentemente o modo anterior.
    try:
        result = input_ids.clone()
        for _ in range(max_new_tokens):
            context = result[:, -model.context_length :]
            logits = model(context)[:, -1, :]  # [B, V]: previsão após o último token.
            logits = logits / temperature
            if top_k is not None:
                threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            if do_sample:
                next_id = torch.multinomial(probabilities, num_samples=1, generator=generator)
            else:
                next_id = torch.argmax(probabilities, dim=-1, keepdim=True)
            result = torch.cat((result, next_id), dim=1)
        return result
    finally:
        model.train(was_training)
