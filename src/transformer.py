"""Componentes de um Transformer Block (GPT decoder-only) da Sprint 4."""

from __future__ import annotations

import torch
from torch import nn

from src.attention import MultiHeadAttention  # Reutiliza a Sprint 3.


class LayerNorm(nn.Module):
    """Normaliza cada token ao longo da dimensão D de [B, T, D]."""

    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        if embedding_dim <= 0 or eps <= 0:
            raise ValueError("embedding_dim e eps devem ser positivos.")
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized + self.shift


class FeedForward(nn.Module):
    """Rede aplicada independentemente a cada token: D -> F -> D."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim e hidden_dim devem ser positivos.")
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Bloco GPT pre-LayerNorm com duas conexões residuais.

    Entrada e saída: [B, T, D]. A atenção mantém [B, T, D], e a
    FeedForward expande internamente para [B, T, F] antes de voltar a D.
    """

    def __init__(
        self,
        embedding_dim: int,
        context_length: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout deve estar no intervalo [0, 1).")
        self.norm1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            d_in=embedding_dim,
            d_out=embedding_dim,
            context_length=context_length,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)
        self.shortcut_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Primeira conexão residual: atenção causal.
        x = x + self.shortcut_dropout(self.attention(self.norm1(x)))
        # Segunda conexão residual: rede FeedForward.
        x = x + self.shortcut_dropout(self.feed_forward(self.norm2(x)))
        return x
