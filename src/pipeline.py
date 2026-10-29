"""Demonstração do pipeline acumulado das Sprints 2, 3 e 4."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attention import MultiHeadAttention
from src.data import create_dataloader_v1
from src.embeddings import TokenAndPositionEmbedding
from src.gpt import GPTModel
from src.generation import generate_tokens
from src.tokenization import (
    EOT_TOKEN,
    build_vocabulary,
    create_simple_tokenizer,
    get_gpt2_tokenizer,
    tokenize_text,
)

DATA_FILE = ROOT / "data" / "texto_teste.txt"


def main() -> None:
    torch.manual_seed(123)
    text = DATA_FILE.read_text(encoding="utf-8")

    print("\n=== 1. TOKENIZAÇÃO ===")
    example = "Um modelo de linguagem aprende padrões."
    tokens = tokenize_text(example)
    print("Texto:", example)
    print("Tokens:", tokens)

    print("\n=== 2. VOCABULÁRIO E TOKEN IDs ===")
    corpus_tokens = tokenize_text(text)
    str_to_int, int_to_str = build_vocabulary(corpus_tokens)
    tokenizer = create_simple_tokenizer(text)
    ids = tokenizer.encode(example)
    print("Tamanho do vocabulário:", len(str_to_int))
    print("Token IDs:", ids)
    print("Decode:", tokenizer.decode(ids))
    print(f"Token especial {EOT_TOKEN!r} -> ID {str_to_int[EOT_TOKEN]}")
    print("Exemplo ID -> Token:", ids[0], "->", int_to_str[ids[0]])

    print("\n=== 3. TOKENIZAÇÃO BPE DO GPT-2 (quando tiktoken estiver instalado) ===")
    try:
        gpt2_tokenizer = get_gpt2_tokenizer()
        bpe_ids = gpt2_tokenizer.encode(example)
        print("GPT-2 Token IDs:", bpe_ids)
        print("GPT-2 Decode:", gpt2_tokenizer.decode(bpe_ids))
    except RuntimeError as exc:
        print("BPE não executado:", exc)

    print("\n=== 4. SEQUÊNCIAS + DATALOADER ===")
    context_length = 8
    dataloader = create_dataloader_v1(
        text,
        tokenizer,
        batch_size=2,
        max_length=context_length,
        stride=4,
        shuffle=False,
        drop_last=False,
    )
    input_batch, target_batch = next(iter(dataloader))
    print("Input batch shape:", tuple(input_batch.shape), "= [batch_size, context_length]")
    print("Target batch shape:", tuple(target_batch.shape))
    print("Primeira entrada:", input_batch[0].tolist())
    print("Primeiro alvo:  ", target_batch[0].tolist())

    print("\n=== 5. EMBEDDINGS + POSITIONAL EMBEDDINGS ===")
    embedding_dim = 16
    embedding_layer = TokenAndPositionEmbedding(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
    )
    embeddings = embedding_layer(input_batch)
    print(
        "Embedding shape:",
        tuple(embeddings.shape),
        "= [batch_size, context_length, embedding_dim]",
    )

    print("\n=== 6. MULTI-HEAD CAUSAL ATTENTION ===")
    attention = MultiHeadAttention(
        d_in=embedding_dim,
        d_out=embedding_dim,
        context_length=context_length,
        dropout=0.0,
        num_heads=4,
    )
    attention_output, attention_weights = attention(
        embeddings, return_attention_weights=True
    )
    print("Attention output shape:", tuple(attention_output.shape), "= [B, T, D]")
    print("Attention weights shape:", tuple(attention_weights.shape), "= [B, H, T, T]")

    print("\nPipeline final:")
    print(
        "Texto -> Tokens -> Token IDs -> Sequências -> Embeddings -> "
        "Positional Embeddings -> Multi-Head Causal Attention -> Transformer Blocks -> GPT -> Logits -> Geração"
    )
    print("\n=== 7. MODELO GPT ===")
    gpt = GPTModel(
        vocab_size=tokenizer.vocab_size, context_length=context_length,
        embedding_dim=embedding_dim, num_heads=4, num_layers=2,
        ff_hidden_dim=64, dropout=0.0,
    )
    logits = gpt.eval()(input_batch)
    print("GPT logits shape:", tuple(logits.shape), "= [B, T, V]")

    print("\n=== 8. GERAÇÃO AUTOREGRESSIVA ===")
    generated_ids = generate_tokens(gpt, input_batch[:1, :3], max_new_tokens=5)
    print("Token IDs gerados:", generated_ids.tolist())
    print("Texto gerado (não treinado):", tokenizer.decode(generated_ids[0].tolist()))
    print("Os parâmetros da GPT ainda serão treinados na Sprint 5.")


if __name__ == "__main__":
    main()
