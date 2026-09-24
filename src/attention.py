"""Mecanismos de atenção implementados manualmente para a Sprint 3.

As classes deste módulo recebem a saída da Sprint 2 no formato [B, T, D]
e não utilizam ``nn.MultiheadAttention`` nem implementações prontas de atenção.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

AttentionOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _validate_qkv(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Valida somente as relações de dimensões necessárias ao cálculo."""
    if queries.ndim < 3 or keys.ndim < 3 or values.ndim < 3:
        raise ValueError("Q, K e V devem possuir ao menos 3 dimensões.")
    if queries.shape[:-2] != keys.shape[:-2] or keys.shape[:-2] != values.shape[:-2]:
        raise ValueError("Q, K e V devem possuir as mesmas dimensões de lote/heads.")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("Q e K devem possuir a mesma dimensão de características.")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError("K e V devem possuir a mesma quantidade de tokens.")


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula ``softmax(QKᵀ / sqrt(d_k))V`` com operações básicas.

    Q, K e V podem ter formato [B, T, D] ou [B, H, T, D]. Uma máscara
    booleana usa ``True`` nas posições permitidas e deve ser compatível por
    broadcasting com a matriz de scores [..., T_query, T_key].

    A divisão por ``sqrt(d_k)`` reduz a magnitude dos scores quando ``d_k``
    cresce. Sem a escala, o softmax pode ficar excessivamente saturado e
    produzir gradientes muito pequenos.

    Os pesos retornados são os pesos normalizados antes do dropout. Assim, eles
    continuam interpretáveis e cada linha soma 1; o contexto usa a versão com
    dropout durante o treinamento.
    """
    _validate_qkv(queries, keys, values)

    d_k = queries.shape[-1]
    if d_k <= 0:
        raise ValueError("A dimensão d_k deve ser maior que zero.")

    # [..., T_query, D] @ [..., D, T_key] -> [..., T_query, T_key]
    attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
    attention_scores = attention_scores / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError("A máscara de atenção deve possuir dtype torch.bool.")
        try:
            torch.broadcast_shapes(attention_scores.shape, mask.shape)
        except RuntimeError as exc:
            raise ValueError(
                "A máscara não é compatível com o formato dos attention scores."
            ) from exc
        attention_scores = attention_scores.masked_fill(
            ~mask.to(device=attention_scores.device), float("-inf")
        )

    attention_weights = torch.softmax(attention_scores, dim=-1)
    weights_used = dropout(attention_weights) if dropout is not None else attention_weights

    # [..., T_query, T_key] @ [..., T_key, D_value] -> [..., T_query, D_value]
    context_vectors = torch.matmul(weights_used, values)
    return context_vectors, attention_weights


class SelfAttention(nn.Module):
    """Autoatenção de uma head com projeções aprendíveis para Q, K e V.

    Entrada:  x       [B, T, d_in]
    Projeções Q/K/V:  [B, T, d_out]
    Pesos de atenção: [B, T, T]
    Saída:             [B, T, d_out]
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError("d_in e d_out devem ser maiores que zero.")

        self.d_in = d_in
        self.d_out = d_out
        self.query_projection = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key_projection = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value_projection = nn.Linear(d_in, d_out, bias=qkv_bias)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("A entrada deve possuir formato [B, T, d_in].")
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"A última dimensão da entrada deve ser d_in={self.d_in}, "
                f"mas foi recebida {x.shape[-1]}."
            )

    def project_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expõe Q, K e V para estudo sem complicar o uso normal da classe."""
        self._validate_input(x)
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        return queries, keys, values

    def forward(
        self, x: torch.Tensor, return_attention_weights: bool = False
    ) -> AttentionOutput:
        queries, keys, values = self.project_qkv(x)
        context, attention_weights = scaled_dot_product_attention(
            queries, keys, values
        )
        if return_attention_weights:
            return context, attention_weights
        return context


class CausalAttention(SelfAttention):
    """Autoatenção que impede cada token de observar posições futuras."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        if context_length <= 0:
            raise ValueError("context_length deve ser maior que zero.")
        super().__init__(d_in=d_in, d_out=d_out, qkv_bias=qkv_bias)
        self.context_length = context_length
        self.attention_dropout = nn.Dropout(dropout)

        # True representa uma conexão permitida. A máscara não é treinável,
        # por isso é registrada como buffer e acompanha o módulo entre devices.
        causal_mask = torch.tril(
            torch.ones(context_length, context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self, x: torch.Tensor, return_attention_weights: bool = False
    ) -> AttentionOutput:
        queries, keys, values = self.project_qkv(x)
        sequence_length = x.shape[1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequência com {sequence_length} posições excede o contexto "
                f"máximo de {self.context_length}."
            )

        mask = self.causal_mask[:sequence_length, :sequence_length]
        context, attention_weights = scaled_dot_product_attention(
            queries,
            keys,
            values,
            mask=mask,
            dropout=self.attention_dropout,
        )
        if return_attention_weights:
            return context, attention_weights
        return context


class MultiHeadAttention(nn.Module):
    """Multi-Head Causal Attention implementada manualmente.

    O vetor de dimensão ``d_out`` é dividido em ``num_heads`` partes. Cada
    head calcula atenção em um subespaço de dimensão ``head_dim``. Ao final,
    as heads são concatenadas e passam por ``out_projection``.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError("d_in e d_out devem ser maiores que zero.")
        if num_heads <= 0:
            raise ValueError("num_heads deve ser maior que zero.")
        if d_out % num_heads != 0:
            raise ValueError(
                f"d_out ({d_out}) deve ser divisível por num_heads ({num_heads})."
            )
        if context_length <= 0:
            raise ValueError("context_length deve ser maior que zero.")

        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.query_projection = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key_projection = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value_projection = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_projection = nn.Linear(d_out, d_out)
        self.attention_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(
            torch.ones(context_length, context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("A entrada deve possuir formato [B, T, d_in].")
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"A última dimensão da entrada deve ser d_in={self.d_in}, "
                f"mas foi recebida {x.shape[-1]}."
            )
        if x.shape[1] > self.context_length:
            raise ValueError(
                f"Sequência com {x.shape[1]} posições excede o contexto "
                f"máximo de {self.context_length}."
            )

    def project_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projeta e separa Q, K e V no formato [B, H, T, head_dim]."""
        self._validate_input(x)
        batch_size, sequence_length, _ = x.shape

        # [B, T, d_in] -> [B, T, d_out]
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        # [B, T, d_out] -> [B, T, H, head_dim] -> [B, H, T, head_dim]
        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            )
            return tensor.transpose(1, 2)

        return split_heads(queries), split_heads(keys), split_heads(values)

    def forward(
        self, x: torch.Tensor, return_attention_weights: bool = False
    ) -> AttentionOutput:
        queries, keys, values = self.project_qkv(x)
        batch_size, _, sequence_length, _ = queries.shape
        mask = self.causal_mask[:sequence_length, :sequence_length]

        # [B, H, T, head_dim] -> contexto [B, H, T, head_dim]
        # e pesos [B, H, T, T].
        context, attention_weights = scaled_dot_product_attention(
            queries,
            keys,
            values,
            mask=mask,
            dropout=self.attention_dropout,
        )

        # Reúne as heads: [B, H, T, head_dim] -> [B, T, H, head_dim]
        # -> [B, T, d_out]. contiguous() reorganiza a memória antes de view().
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, sequence_length, self.d_out)
        output = self.out_projection(context)

        if return_attention_weights:
            return output, attention_weights
        return output
