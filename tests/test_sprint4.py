"""Testes da Sprint 4: dimensões, causalidade, gradientes e geração."""

import pytest
import torch

from src.generation import generate_tokens
from src.gpt import GPTModel
from src.transformer import FeedForward, LayerNorm, TransformerBlock


def make_model(**kwargs):
    config = dict(vocab_size=30, context_length=8, embedding_dim=16,
                  num_heads=4, num_layers=2, ff_hidden_dim=32, dropout=0.0)
    config.update(kwargs)
    return GPTModel(**config)


def test_layer_norm_normaliza_o_ultimo_eixo():
    torch.manual_seed(1)
    layer = LayerNorm(16)
    result = layer(torch.randn(2, 5, 16))
    assert result.shape == (2, 5, 16)
    assert torch.allclose(result.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(result.var(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-3)


def test_feed_forward_e_transformer_preservam_formato():
    x = torch.randn(2, 5, 16, requires_grad=True)
    ff = FeedForward(16, 64)
    block = TransformerBlock(16, 8, 4, 64, dropout=0.0)
    assert ff(x).shape == x.shape
    assert block(x).shape == x.shape
    block(x).sum().backward()
    assert x.grad is not None
    assert block.attention.query_projection.weight.grad is not None
    assert block.feed_forward.layers[0].weight.grad is not None


def test_gpt_logit_shape_e_backprop():
    model = make_model()
    tokens = torch.randint(0, 30, (2, 5))
    logits = model(tokens)
    assert logits.shape == (2, 5, 30)
    logits.sum().backward()
    assert model.lm_head.weight.grad is not None
    assert model.embeddings.token_embedding.weight.grad is not None


def test_causalidade_na_saida_final_gpt():
    torch.manual_seed(42)
    model = make_model().eval()
    prefix = torch.tensor([[1, 2, 3]])
    extended = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.inference_mode():
        short_logits = model(prefix)
        long_logits = model(extended)
    # Os tokens futuros não podem alterar logits das posições passadas.
    assert torch.allclose(short_logits, long_logits[:, :3, :], atol=1e-5)


def test_entrada_longa_e_heads_invalidas():
    model = make_model()
    with pytest.raises(ValueError):
        model(torch.ones((2, 9), dtype=torch.long))
    with pytest.raises(ValueError):
        make_model(num_heads=3)


def test_geracao_greedy_reutiliza_contexto_e_preserva_modo():
    torch.manual_seed(123)
    model = make_model(context_length=4)
    model.train()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    generated = generate_tokens(model, prompt, max_new_tokens=7)
    assert generated.shape == (2, 10)
    assert torch.equal(generated[:, :3], prompt)
    assert model.training is True
    assert generated.min() >= 0 and generated.max() < model.vocab_size
    assert not generated.requires_grad


def test_amostragem_com_seed_e_topk_1_igual_greedy():
    model = make_model().eval()
    prompt = torch.tensor([[1, 2, 3]])
    greedy = generate_tokens(model, prompt, 4)
    top1 = generate_tokens(model, prompt, 4, top_k=1, do_sample=True)
    assert torch.equal(greedy, top1)
    sample1 = generate_tokens(model, prompt, 4, top_k=5, do_sample=True,
                              generator=torch.Generator().manual_seed(10))
    sample2 = generate_tokens(model, prompt, 4, top_k=5, do_sample=True,
                              generator=torch.Generator().manual_seed(10))
    assert torch.equal(sample1, sample2)


def test_parametros_de_geracao_invalidos():
    model = make_model()
    prompt = torch.tensor([[1, 2]])
    with pytest.raises(ValueError):
        generate_tokens(model, prompt, 2, temperature=0)
    with pytest.raises(ValueError):
        generate_tokens(model, prompt, 2, top_k=0)
    with pytest.raises(ValueError):
        generate_tokens(model, prompt, 2, top_k=31)
    with pytest.raises(ValueError):
        generate_tokens(model, prompt, -1)
