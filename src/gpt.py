"""GPT didático: embeddings da Sprint 2 + atenção da Sprint 3 + blocos da Sprint 4.

O modelo é inicializado com pesos aleatórios. O treinamento ficará para a
Sprint 5; esta implementação demonstra o forward e a produção de logits.
"""

from __future__ import annotations

import torch
from torch import nn

from src.embeddings import TokenAndPositionEmbedding  # Sprint 2.
from src.transformer import LayerNorm, TransformerBlock  # Sprint 4.


class GPTModel(nn.Module):
    """GPT decoder-only simplificado que produz logits [B, T, V]."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        ff_hidden_dim: int | None = None,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if min(vocab_size, context_length, embedding_dim, num_heads, num_layers) <= 0:
            raise ValueError("V, contexto, D, heads e blocos devem ser positivos.")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim precisa ser divisível por num_heads.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout deve estar no intervalo [0, 1).")
        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * embedding_dim
        if ff_hidden_dim <= 0:
            raise ValueError("ff_hidden_dim deve ser positivo.")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.embeddings = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_length=context_length,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *(
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    context_length=context_length,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            )
        )
        self.final_norm = LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids deve ter formato [batch_size, context_length].")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("input_ids deve conter Token IDs inteiros.")
        if input_ids.shape[1] == 0 or input_ids.shape[1] > self.context_length:
            raise ValueError("A sequência deve conter de 1 até context_length tokens.")

        # [B, T] -> [B, T, D] -> N x [B, T, D] -> [B, T, V]
        x = self.embedding_dropout(self.embeddings(input_ids))
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.lm_head(x)
