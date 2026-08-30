from pathlib import Path

import torch

from src.data import GPTDatasetV1, create_dataloader_v1
from src.embeddings import TokenAndPositionEmbedding
from src.tokenization import create_simple_tokenizer, tokenize_text

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "data" / "texto_teste.txt").read_text(encoding="utf-8")


def test_tokenization_produces_tokens():
    tokens = tokenize_text("Olá, mundo!")
    assert tokens == ["Olá", ",", "mundo", "!"]


def test_encode_decode_known_text():
    tokenizer = create_simple_tokenizer(TEXT)
    phrase = "O projeto é incremental."
    ids = tokenizer.encode(phrase)
    decoded = tokenizer.decode(ids)
    assert decoded == phrase


def test_dataset_shifts_target_by_one():
    tokenizer = create_simple_tokenizer(TEXT)
    dataset = GPTDatasetV1(TEXT, tokenizer, max_length=8, stride=1)
    x, y = dataset[0]
    assert torch.equal(x[1:], y[:-1])
    assert x.shape == y.shape == torch.Size([8])


def test_dataloader_and_embedding_shapes():
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
    x, y = next(iter(loader))
    assert x.shape == y.shape == torch.Size([2, 8])

    layer = TokenAndPositionEmbedding(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16,
        context_length=8,
    )
    out = layer(x)
    assert out.shape == torch.Size([2, 8, 16])
