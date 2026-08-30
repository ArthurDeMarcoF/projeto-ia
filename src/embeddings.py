"""Token Embeddings e Positional Embeddings."""

from __future__ import annotations

import torch
from torch import nn


class TokenAndPositionEmbedding(nn.Module):
    """Soma embeddings de token e de posição.

    Entrada:
        input_ids: [batch_size, context_length]

    Saída:
        embeddings: [batch_size, context_length, embedding_dim]
    """

    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int):
        super().__init__()
        self.context_length = context_length
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids deve possuir formato [batch_size, context_length].")

        sequence_length = input_ids.shape[1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequência com {sequence_length} posições excede o contexto "
                f"máximo de {self.context_length}."
            )

        token_embeddings = self.token_embedding(input_ids)

        positions = torch.arange(sequence_length, device=input_ids.device)
        positional_embeddings = self.position_embedding(positions)

        # [B, T, D] + [T, D] -> [B, T, D] por broadcasting.
        return token_embeddings + positional_embeddings

    def components(self, input_ids: torch.Tensor):
        """Retorna componentes separados para fins de experimentação."""
        sequence_length = input_ids.shape[1]
        positions = torch.arange(sequence_length, device=input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        positional_embeddings = self.position_embedding(positions)
        combined = token_embeddings + positional_embeddings
        return token_embeddings, positional_embeddings, combined
