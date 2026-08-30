"""Experimentos da Sprint 2.

Executar a partir da raiz do projeto:
    python experimentos/sprint2_experimentos.py

O script salva os resultados em experimentos/resultados_sprint2.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import GPTDatasetV1, create_dataloader_v1
from src.embeddings import TokenAndPositionEmbedding
from src.tokenization import create_simple_tokenizer, tokenize_text

DATA_FILE = ROOT / "data" / "texto_teste.txt"
OUTPUT_FILE = ROOT / "experimentos" / "resultados_sprint2.md"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    torch.manual_seed(123)
    text = DATA_FILE.read_text(encoding="utf-8")
    tokenizer = create_simple_tokenizer(text)

    sections = [
        "# Resultados dos Experimentos — Sprint 2",
        "",
        "Resultados gerados automaticamente por `experimentos/sprint2_experimentos.py`.",
        "",
        f"**Tokens no corpus (tokenizador didático):** {len(tokenize_text(text))}",
        f"**Tamanho do vocabulário:** {tokenizer.vocab_size}",
        "",
    ]

    # 1. Diferentes textos e quantidade de tokens
    phrases = [
        "Um modelo de linguagem aprende padrões.",
        "O texto precisa ser dividido em unidades menores.",
        "A ordem dos tokens também precisa ser representada.",
        "O Dataset organiza as amostras e o DataLoader combina várias amostras em lotes.",
    ]
    rows = []
    for phrase in phrases:
        tokens = tokenize_text(phrase)
        ids = tokenizer.encode(phrase)
        rows.append((phrase, len(tokens), len(ids)))

    sections += [
        "## Experimento 1 — Diferentes sequências de entrada",
        "",
        markdown_table(["Texto", "Tokens", "Token IDs"], rows),
        "",
        "**Observação:** textos maiores ou com mais sinais de pontuação tendem a produzir mais unidades de processamento. A quantidade exata depende das regras do tokenizador.",
        "",
    ]

    # 2. Context length x quantidade de amostras
    context_rows = []
    for context_length in [4, 8, 16, 32]:
        dataset = GPTDatasetV1(text, tokenizer, max_length=context_length, stride=1)
        context_rows.append((context_length, 1, len(dataset)))

    sections += [
        "## Experimento 2 — Tamanho do contexto × quantidade de amostras",
        "",
        markdown_table(["Context length", "Stride", "Amostras"], context_rows),
        "",
        "**Observação:** mantendo o stride fixo, aumentar o contexto reduz a quantidade de janelas válidas que cabem no mesmo texto.",
        "",
    ]

    # 3. Batch size
    batch_rows = []
    for batch_size in [1, 2, 4, 8]:
        loader = create_dataloader_v1(
            text,
            tokenizer,
            batch_size=batch_size,
            max_length=8,
            stride=8,
            shuffle=False,
            drop_last=False,
        )
        x, y = next(iter(loader))
        batch_rows.append(
            (
                batch_size,
                len(loader.dataset),
                len(loader),
                str(tuple(x.shape)),
                str(tuple(y.shape)),
            )
        )

    sections += [
        "## Experimento 3 — Diferentes tamanhos de lote",
        "",
        markdown_table(
            ["Batch size", "Amostras", "Lotes", "Input shape", "Target shape"],
            batch_rows,
        ),
        "",
        "**Observação:** o batch size não altera as amostras do Dataset; ele altera quantas amostras são agrupadas em cada iteração e, consequentemente, o número de lotes.",
        "",
    ]

    # 4. Embedding dimension e custo
    loader = create_dataloader_v1(
        text,
        tokenizer,
        batch_size=4,
        max_length=8,
        stride=8,
        shuffle=False,
        drop_last=False,
    )
    input_batch, _ = next(iter(loader))

    embedding_rows = []
    for embedding_dim in [8, 16, 32, 64]:
        layer = TokenAndPositionEmbedding(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            context_length=8,
        )
        output = layer(input_batch)

        params = sum(p.numel() for p in layer.parameters())
        memory_kb = params * 4 / 1024  # float32

        # Medição simples e apenas indicativa; varia de máquina para máquina.
        repetitions = 200
        start = perf_counter()
        with torch.no_grad():
            for _ in range(repetitions):
                layer(input_batch)
        elapsed_ms = (perf_counter() - start) * 1000 / repetitions

        embedding_rows.append(
            (
                embedding_dim,
                str(tuple(output.shape)),
                params,
                f"{memory_kb:.2f}",
                f"{elapsed_ms:.4f}",
            )
        )

    sections += [
        "## Experimento 4 — Dimensão do embedding e custo",
        "",
        markdown_table(
            [
                "Embedding dim",
                "Output shape",
                "Parâmetros",
                "Memória aprox. (KB, float32)",
                "Tempo médio forward (ms)",
            ],
            embedding_rows,
        ),
        "",
        "**Observação:** aumentar `embedding_dim` aumenta a última dimensão do tensor e também a quantidade de parâmetros das matrizes de embeddings. O tempo é apenas indicativo e deve ser interpretado com cautela, pois depende do hardware e do ambiente.",
        "",
    ]

    # 5. Positional embeddings
    repeated_id = tokenizer.encode("modelo")[0]
    repeated_input = torch.tensor([[repeated_id, repeated_id]], dtype=torch.long)
    layer = TokenAndPositionEmbedding(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=8,
        context_length=2,
    )
    token_emb, pos_emb, combined = layer.components(repeated_input)

    same_token_embedding = torch.allclose(token_emb[0, 0], token_emb[0, 1])
    same_position_embedding = torch.allclose(pos_emb[0], pos_emb[1])
    same_combined_embedding = torch.allclose(combined[0, 0], combined[0, 1])

    positional_rows = [
        ("Token ID nas duas posições", repeated_id, repeated_id),
        ("Token embedding é igual?", same_token_embedding, "esperado: True"),
        ("Positional embedding é igual?", same_position_embedding, "esperado: False"),
        ("Representação final é igual?", same_combined_embedding, "esperado: False"),
    ]

    sections += [
        "## Experimento 5 — Efeito da posição",
        "",
        markdown_table(["Verificação", "Resultado", "Referência"], positional_rows),
        "",
        "**Observação:** o mesmo Token ID possui o mesmo token embedding, mas recebe vetores posicionais diferentes. Após a soma, as representações finais tornam-se diferentes, permitindo ao Transformer distinguir posições distintas da sequência.",
        "",
    ]

    OUTPUT_FILE.write_text("\n".join(sections), encoding="utf-8")
    print(f"Resultados salvos em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
