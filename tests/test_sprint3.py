from pathlib import Path

import pytest
import torch
from torch import nn

from src.attention import (
    CausalAttention,
    MultiHeadAttention,
    SelfAttention,
    scaled_dot_product_attention,
)
from src.data import create_dataloader_v1
from src.embeddings import TokenAndPositionEmbedding
from src.tokenization import create_simple_tokenizer

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "data" / "texto_teste.txt").read_text(encoding="utf-8")


def test_self_attention_qkv_output_and_weight_shapes():
    torch.manual_seed(123)
    x = torch.randn(2, 5, 8)
    attention = SelfAttention(d_in=8, d_out=12)

    queries, keys, values = attention.project_qkv(x)
    output, weights = attention(x, return_attention_weights=True)

    assert queries.shape == keys.shape == values.shape == torch.Size([2, 5, 12])
    assert output.shape == torch.Size([2, 5, 12])
    assert weights.shape == torch.Size([2, 5, 5])


def test_attention_weights_sum_to_one():
    torch.manual_seed(123)
    x = torch.randn(2, 6, 8)
    _, weights = SelfAttention(8, 8)(x, return_attention_weights=True)
    expected = torch.ones(2, 6)
    assert torch.allclose(weights.sum(dim=-1), expected, atol=1e-6)


def test_scaled_dot_product_attention_shapes_and_values():
    torch.manual_seed(123)
    queries = torch.randn(2, 4, 6)
    keys = torch.randn(2, 4, 6)
    values = torch.randn(2, 4, 10)

    context, weights = scaled_dot_product_attention(queries, keys, values)

    assert context.shape == torch.Size([2, 4, 10])
    assert weights.shape == torch.Size([2, 4, 4])
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_scaled_dot_product_attention_rejects_incompatible_qkv():
    queries = torch.randn(2, 4, 6)
    keys = torch.randn(2, 4, 5)
    values = torch.randn(2, 4, 7)
    with pytest.raises(ValueError, match="Q e K"):
        scaled_dot_product_attention(queries, keys, values)


def test_causal_attention_blocks_future_tokens():
    torch.manual_seed(123)
    x = torch.randn(2, 5, 8)
    attention = CausalAttention(8, 8, context_length=5, dropout=0.0)
    _, weights = attention(x, return_attention_weights=True)

    future_positions = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.allclose(weights[:, future_positions], torch.zeros(2, 10))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_causal_attention_rejects_sequence_larger_than_context():
    attention = CausalAttention(8, 8, context_length=4)
    with pytest.raises(ValueError, match="excede o contexto"):
        attention(torch.randn(2, 5, 8))


def test_multi_head_attention_shapes_and_number_of_heads():
    torch.manual_seed(123)
    x = torch.randn(3, 6, 16)
    attention = MultiHeadAttention(
        d_in=16,
        d_out=16,
        context_length=6,
        dropout=0.0,
        num_heads=4,
    )
    output, weights = attention(x, return_attention_weights=True)
    queries, keys, values = attention.project_qkv(x)

    assert attention.head_dim == 4
    assert queries.shape == keys.shape == values.shape == torch.Size([3, 4, 6, 4])
    assert output.shape == torch.Size([3, 6, 16])
    assert weights.shape == torch.Size([3, 4, 6, 6])
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 4, 6), atol=1e-6)


def test_multi_head_attention_rejects_invalid_head_configuration():
    with pytest.raises(ValueError, match="num_heads deve ser maior"):
        MultiHeadAttention(8, 8, context_length=4, dropout=0.0, num_heads=0)
    with pytest.raises(ValueError, match="deve ser divisível"):
        MultiHeadAttention(8, 10, context_length=4, dropout=0.0, num_heads=3)


def test_return_attention_weights_is_optional():
    x = torch.randn(2, 4, 8)
    attention = MultiHeadAttention(8, 8, 4, dropout=0.0, num_heads=2)
    output = attention(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([2, 4, 8])


def test_dropout_does_not_change_reported_normalized_weights_in_eval_mode():
    torch.manual_seed(123)
    x = torch.randn(2, 4, 8)
    attention = CausalAttention(8, 8, 4, dropout=0.5)
    attention.eval()
    _, weights = attention(x, return_attention_weights=True)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_embedding_to_attention_integration_with_real_project_data():
    torch.manual_seed(123)
    tokenizer = create_simple_tokenizer(TEXT)
    loader = create_dataloader_v1(
        TEXT,
        tokenizer,
        batch_size=2,
        max_length=8,
        stride=4,
        shuffle=False,
        drop_last=False,
    )
    input_batch, _ = next(iter(loader))
    embeddings = TokenAndPositionEmbedding(
        tokenizer.vocab_size, embedding_dim=16, context_length=8
    )(input_batch)
    attention = MultiHeadAttention(16, 16, 8, dropout=0.0, num_heads=4)

    output, weights = attention(embeddings, return_attention_weights=True)

    assert embeddings.shape == torch.Size([2, 8, 16])
    assert output.shape == embeddings.shape
    assert weights.shape == torch.Size([2, 4, 8, 8])


def test_scaled_attention_accepts_four_dimensional_multi_head_tensors():
    torch.manual_seed(123)
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 5, 4)
    v = torch.randn(2, 3, 5, 4)
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    context, weights = scaled_dot_product_attention(
        q, k, v, mask=mask, dropout=nn.Dropout(0.0)
    )
    assert context.shape == torch.Size([2, 3, 5, 4])
    assert weights.shape == torch.Size([2, 3, 5, 5])
