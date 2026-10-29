"""Demonstração executável da Sprint 4, sem treinamento."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation import generate_tokens
from src.gpt import GPTModel
from src.tokenization import create_simple_tokenizer


def main() -> None:
    torch.manual_seed(123)
    corpus = (ROOT / "data" / "texto_teste.txt").read_text(encoding="utf-8")
    tokenizer = create_simple_tokenizer(corpus)
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        context_length=12,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        ff_hidden_dim=128,
        dropout=0.1,
    )
    prompt = "Um modelo de linguagem"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    logits = model.eval()(ids)
    generated = generate_tokens(
        model, ids, max_new_tokens=8, temperature=1.0, top_k=5,
        do_sample=True, generator=torch.Generator().manual_seed(123),
    )

    print("Prompt:", prompt)
    print("Token IDs:", ids.tolist())
    print("Embedding shape: [B, T, D] =", (ids.shape[0], ids.shape[1], model.embedding_dim))
    print("Logits shape: [B, T, V] =", tuple(logits.shape))
    print("Próximos Token IDs:", generated[0, ids.shape[1]:].tolist())
    print("Texto gerado (sem treinamento):", tokenizer.decode(generated[0].tolist()))
    print("Aviso: os pesos ainda são aleatórios; a Sprint 5 fará o treinamento.")


if __name__ == "__main__":
    main()
