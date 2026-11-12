"""Preparação das sequências de treinamento e DataLoader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "data" / "texto_teste.txt"


def split_corpus(text: str, train_fraction: float) -> tuple[str, str]:
    """Divide o corpus em trechos contíguos antes da tokenização e das janelas."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction deve estar entre 0 e 1.")
    words = text.split()
    boundary = int(len(words) * train_fraction)
    return " ".join(words[:boundary]), " ".join(words[boundary:])


def encode_text(tokenizer: Any, text: str) -> List[int]:
    """Codifica texto aceitando tanto o tokenizador simples quanto tiktoken."""
    try:
        return tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    except TypeError:
        return tokenizer.encode(text)


class GPTDatasetV1(Dataset):
    """Dataset de pares entrada-alvo usando janela deslizante.

    Para uma sequência [t0, t1, t2, t3] e contexto 3:
    entrada = [t0, t1, t2]
    alvo    = [t1, t2, t3]

    Assim, cada posição da entrada aprende a prever o próximo token.
    """

    def __init__(self, text: str, tokenizer: Any, max_length: int, stride: int):
        if max_length <= 0:
            raise ValueError("max_length deve ser maior que zero.")
        if stride <= 0:
            raise ValueError("stride deve ser maior que zero.")

        self.input_ids = []
        self.target_ids = []

        token_ids = encode_text(tokenizer, text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    text: str,
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Organiza o Dataset em lotes prontos para o treinamento."""
    dataset = GPTDatasetV1(text, tokenizer, max_length=max_length, stride=stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=generator,
    )
