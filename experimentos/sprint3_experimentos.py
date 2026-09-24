"""Experimentos e visualizações da Sprint 3.

Executar a partir da raiz do projeto:
    python experimentos/sprint3_experimentos.py

O script usa o corpus, o DataLoader e os embeddings reais da Sprint 2. Os
resultados numéricos e a análise são gerados a partir desta execução.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attention import CausalAttention, MultiHeadAttention, SelfAttention
from src.data import create_dataloader_v1
from src.embeddings import TokenAndPositionEmbedding
from src.tokenization import create_simple_tokenizer

DATA_FILE = ROOT / "data" / "texto_teste.txt"
RESULTS_FILE = ROOT / "experimentos" / "resultados_sprint3.md"
ANALYSIS_FILE = ROOT / "experimentos" / "analise_sprint3.md"
FIGURES_DIR = ROOT / "experimentos" / "figuras"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def mean_entropy(weights):
    """Entropia média das linhas: -soma(p * ln(p))."""
    safe_weights = weights.clamp_min(1e-12)
    return -(weights * safe_weights.log()).sum(dim=-1).mean().item()


def format_matrix(matrix, decimals=3):
    rows = []
    for row in matrix.detach().cpu().tolist():
        rows.append("[" + ", ".join(f"{value:.{decimals}f}" for value in row) + "]")
    return "\n".join(rows)


def save_heatmap(matrix, token_labels, path, title, annotate=True):
    data = matrix.detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(data, cmap="viridis", vmin=0)
    axis.set_title(title)
    axis.set_xlabel("Tokens observados / keys")
    axis.set_ylabel("Tokens consultando / queries")
    axis.set_xticks(range(len(token_labels)), token_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(token_labels)), token_labels)
    if annotate and len(token_labels) <= 10:
        threshold = float(data.max()) / 2 if data.size else 0
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                color = "white" if data[row, column] < threshold else "black"
                axis.text(
                    column,
                    row,
                    f"{data[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
    figure.colorbar(image, ax=axis, label="Peso de atenção")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_sequence_comparison(sequence_results, path):
    figure, axes = plt.subplots(1, len(sequence_results), figsize=(18, 5))
    for axis, result in zip(axes, sequence_results):
        data = result["weights"][0, 0].detach().cpu().numpy()
        labels = result["tokens"]
        image = axis.imshow(data, cmap="viridis", vmin=0)
        axis.set_title(result["short_title"])
        axis.set_xlabel("Keys")
        axis.set_ylabel("Queries")
        axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
        axis.set_yticks(range(len(labels)), labels, fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Head 1 para diferentes sequências (parâmetros não treinados)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    torch.manual_seed(123)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    text = DATA_FILE.read_text(encoding="utf-8")
    tokenizer = create_simple_tokenizer(text)
    context_length = 8
    loader = create_dataloader_v1(
        text,
        tokenizer,
        batch_size=2,
        max_length=context_length,
        stride=4,
        shuffle=False,
        drop_last=False,
    )
    input_batch, _ = next(iter(loader))
    first_tokens = [tokenizer.int_to_str[int(token_id)] for token_id in input_batch[0]]

    # Experimento 1 — diferentes dimensões de embedding.
    embedding_rows = []
    for embedding_dim in [8, 16, 32, 64]:
        embedding_layer = TokenAndPositionEmbedding(
            tokenizer.vocab_size, embedding_dim, context_length
        )
        embeddings = embedding_layer(input_batch)
        attention = MultiHeadAttention(
            embedding_dim,
            embedding_dim,
            context_length,
            dropout=0.0,
            num_heads=4,
        )
        output = attention(embeddings)
        embedding_rows.append(
            (
                embedding_dim,
                str(tuple(embeddings.shape)),
                str(tuple(output.shape)),
                4,
                attention.head_dim,
                parameter_count(attention),
            )
        )

    # Uma representação base comum aos demais experimentos.
    embedding_dim = 32
    embedding_layer = TokenAndPositionEmbedding(
        tokenizer.vocab_size, embedding_dim, context_length
    )
    embeddings = embedding_layer(input_batch)

    # Experimento 2 — diferentes números de heads.
    heads_rows = []
    attention_by_heads = {}
    for num_heads in [1, 2, 4, 8]:
        attention = MultiHeadAttention(
            embedding_dim,
            embedding_dim,
            context_length,
            dropout=0.0,
            num_heads=num_heads,
        )
        attention.eval()
        output, weights = attention(embeddings, return_attention_weights=True)
        attention_by_heads[num_heads] = (attention, output, weights)
        heads_rows.append(
            (
                embedding_dim,
                num_heads,
                attention.head_dim,
                str(tuple(output.shape)),
                str(tuple(weights.shape)),
                parameter_count(attention),
            )
        )

    # Experimento 3 — a relação D / H determina head_dim.
    head_dimension_rows = [
        (
            embedding_dim,
            num_heads,
            attention_by_heads[num_heads][0].head_dim,
            f"{embedding_dim} / {num_heads}",
        )
        for num_heads in [2, 4, 8]
    ]

    # Experimento 4 — Self-Attention simples versus Multi-Head causal.
    self_attention = SelfAttention(embedding_dim, embedding_dim)
    self_attention.eval()
    self_output, self_weights = self_attention(
        embeddings, return_attention_weights=True
    )
    multi_attention, multi_output, multi_weights = attention_by_heads[4]
    comparison_rows = [
        (
            "Self-Attention (não causal)",
            1,
            str(tuple(self_weights.shape)),
            str(tuple(self_output.shape)),
            parameter_count(self_attention),
        ),
        (
            "Multi-Head Causal Attention",
            multi_attention.num_heads,
            str(tuple(multi_weights.shape)),
            str(tuple(multi_output.shape)),
            parameter_count(multi_attention),
        ),
    ]
    save_heatmap(
        self_weights[0],
        first_tokens,
        FIGURES_DIR / "sprint3_self_attention.png",
        "Self-Attention — amostra 1",
    )
    for head_index in range(multi_attention.num_heads):
        save_heatmap(
            multi_weights[0, head_index],
            first_tokens,
            FIGURES_DIR / f"sprint3_multi_head_{head_index + 1}.png",
            f"Multi-Head Causal Attention — head {head_index + 1}",
        )

    # Experimento 5 — efeito numérico da escala 1/sqrt(d_k).
    queries, keys, _ = self_attention.project_qkv(embeddings)
    scores_without_scale = torch.matmul(queries, keys.transpose(-2, -1))
    scores_with_scale = scores_without_scale / math.sqrt(queries.shape[-1])
    weights_without_scale = torch.softmax(scores_without_scale, dim=-1)
    weights_with_scale = torch.softmax(scores_with_scale, dim=-1)
    scale_rows = [
        (
            "Sem escala",
            f"{scores_without_scale.abs().mean().item():.6f}",
            f"{scores_without_scale.abs().max().item():.6f}",
            f"{weights_without_scale.max().item():.6f}",
            f"{mean_entropy(weights_without_scale):.6f}",
        ),
        (
            "Com escala 1/sqrt(d_k)",
            f"{scores_with_scale.abs().mean().item():.6f}",
            f"{scores_with_scale.abs().max().item():.6f}",
            f"{weights_with_scale.max().item():.6f}",
            f"{mean_entropy(weights_with_scale):.6f}",
        ),
    ]

    # Experimento 6 — máscara causal, exibida também numericamente.
    causal_attention = CausalAttention(
        embedding_dim,
        embedding_dim,
        context_length,
        dropout=0.0,
    )
    causal_attention.eval()
    causal_queries, causal_keys, _ = causal_attention.project_qkv(embeddings)
    scores_before_mask = torch.matmul(
        causal_queries, causal_keys.transpose(-2, -1)
    ) / math.sqrt(causal_queries.shape[-1])
    _, causal_weights = causal_attention(embeddings, return_attention_weights=True)
    future_mask = torch.triu(
        torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1
    )
    maximum_future_weight = causal_weights[0][future_mask].abs().max().item()
    save_heatmap(
        causal_weights[0],
        first_tokens,
        FIGURES_DIR / "sprint3_causal_attention.png",
        "Causal Attention — região futura zerada",
    )

    # Experimento 7 — sequências reais diferentes, ainda sem treinamento.
    phrases = [
        ("Sequência 1", "Um modelo de linguagem aprende padrões."),
        ("Sequência 2", "A ordem dos tokens também precisa ser representada."),
        ("Sequência 3", "O projeto é incremental."),
    ]
    sequence_embedding = TokenAndPositionEmbedding(
        tokenizer.vocab_size, embedding_dim, context_length=12
    )
    sequence_attention = MultiHeadAttention(
        embedding_dim,
        embedding_dim,
        context_length=12,
        dropout=0.0,
        num_heads=4,
    )
    sequence_attention.eval()
    sequence_results = []
    sequence_rows = []
    for short_title, phrase in phrases:
        ids = tokenizer.encode(phrase)
        phrase_input = torch.tensor([ids], dtype=torch.long)
        phrase_embeddings = sequence_embedding(phrase_input)
        phrase_output, phrase_weights = sequence_attention(
            phrase_embeddings, return_attention_weights=True
        )
        tokens = [tokenizer.int_to_str[token_id] for token_id in ids]
        result = {
            "short_title": short_title,
            "phrase": phrase,
            "tokens": tokens,
            "weights": phrase_weights,
        }
        sequence_results.append(result)
        sequence_rows.append(
            (
                phrase,
                len(ids),
                str(tuple(phrase_weights.shape)),
                str(tuple(phrase_output.shape)),
                f"{phrase_weights[:, :, 1:, :].max().item():.6f}",
                f"{mean_entropy(phrase_weights):.6f}",
            )
        )
    save_sequence_comparison(
        sequence_results, FIGURES_DIR / "sprint3_diferentes_sequencias.png"
    )

    results_sections = [
        "# Resultados dos Experimentos — Sprint 3",
        "",
        "Resultados gerados automaticamente por `experimentos/sprint3_experimentos.py` com `torch.manual_seed(123)`.",
        "",
        f"**Input IDs real:** `{tuple(input_batch.shape)}`",
        f"**Embeddings base:** `{tuple(embeddings.shape)}`",
        f"**Tokens da primeira amostra:** `{' | '.join(first_tokens)}`",
        "",
        "Os pesos são de parâmetros inicializados aleatoriamente; os resultados demonstram o mecanismo matemático, não relações linguísticas aprendidas.",
        "",
        "## Experimento 1 — Diferentes dimensões de embedding",
        "",
        markdown_table(
            [
                "Embedding dim",
                "Input shape",
                "Output shape",
                "Heads",
                "Head dim",
                "Parâmetros da atenção",
            ],
            embedding_rows,
        ),
        "",
        "## Experimento 2 — Diferentes números de heads",
        "",
        markdown_table(
            [
                "Embedding dim",
                "Heads",
                "Head dim",
                "Output shape",
                "Attention weights shape",
                "Parâmetros",
            ],
            heads_rows,
        ),
        "",
        "## Experimento 3 — Diferentes dimensões por head",
        "",
        markdown_table(
            ["Embedding dim", "Heads", "Head dim", "Cálculo"],
            head_dimension_rows,
        ),
        "",
        "## Experimento 4 — Self-Attention × Multi-Head Attention",
        "",
        markdown_table(
            ["Mecanismo", "Heads", "Pesos", "Saída", "Parâmetros"],
            comparison_rows,
        ),
        "",
        "## Experimento 5 — Attention com escala × sem escala",
        "",
        f"Neste experimento, `d_k = {queries.shape[-1]}` e o divisor é `sqrt(d_k) = {math.sqrt(queries.shape[-1]):.6f}`.",
        "",
        markdown_table(
            [
                "Configuração",
                "Média absoluta dos scores",
                "Máximo absoluto dos scores",
                "Maior peso",
                "Entropia média",
            ],
            scale_rows,
        ),
        "",
        "A entropia de cada linha é calculada por `-soma(p * ln(p))`; valores menores indicam uma distribuição mais concentrada.",
        "",
        "## Experimento 6 — Máscara causal",
        "",
        "### Scores escalados antes da máscara (amostra 1)",
        "",
        "```text",
        format_matrix(scores_before_mask[0]),
        "```",
        "",
        "### Máscara causal (1 = permitido; 0 = futuro bloqueado)",
        "",
        "```text",
        format_matrix(causal_attention.causal_mask.to(torch.float32), decimals=0),
        "```",
        "",
        "### Attention weights depois da máscara e do softmax (amostra 1)",
        "",
        "```text",
        format_matrix(causal_weights[0]),
        "```",
        "",
        f"**Maior peso em posição futura (`j > i`):** `{maximum_future_weight:.6f}`",
        "",
        "## Experimento 7 — Diferentes sequências de entrada",
        "",
        markdown_table(
            [
                "Texto",
                "Tokens",
                "Pesos",
                "Saída",
                "Maior peso (queries i > 0)",
                "Entropia média",
            ],
            sequence_rows,
        ),
        "",
        "As matrizes diferem porque as entradas e posições projetadas são diferentes. Como não houve treinamento, elas não sustentam interpretações semânticas, sintáticas ou de compreensão.",
        "",
        "## Figuras geradas",
        "",
        "- `figuras/sprint3_self_attention.png`",
        "- `figuras/sprint3_causal_attention.png`",
        "- `figuras/sprint3_multi_head_1.png` a `sprint3_multi_head_4.png`",
        "- `figuras/sprint3_diferentes_sequencias.png`",
        "",
    ]
    RESULTS_FILE.write_text("\n".join(results_sections), encoding="utf-8")

    unscaled_entropy = mean_entropy(weights_without_scale)
    scaled_entropy = mean_entropy(weights_with_scale)
    analysis_sections = [
        "# Análise dos Resultados — Sprint 3",
        "",
        "Esta análise é gerada junto aos resultados e utiliza somente valores medidos na execução com seed 123. As projeções Q/K/V ainda não foram treinadas.",
        "",
        "## 1. Diferentes dimensões de embedding",
        "",
        f"**Resultado observado:** as quatro configurações preservaram `[B, T] = [{input_batch.shape[0]}, {input_batch.shape[1]}]` e produziram a última dimensão 8, 16, 32 ou 64. Os parâmetros da atenção passaram de {embedding_rows[0][-1]} para {embedding_rows[-1][-1]}.",
        "",
        "**Causa técnica:** as projeções Q, K, V e a projeção final são matrizes lineares cuja quantidade de elementos cresce com as dimensões de entrada e saída.",
        "",
        "**Relação com a teoria e consequência:** `D` define a largura da representação contextual. Aumentá-la oferece mais componentes vetoriais por token, mas aumenta parâmetros e operações; nenhuma conclusão de velocidade foi feita sem medição.",
        "",
        "## 2. Diferentes números de heads",
        "",
        f"**Resultado observado:** com `D=32`, os pesos mudaram de `{heads_rows[0][4]}` para `{heads_rows[-1][4]}`, enquanto todas as saídas permaneceram `{heads_rows[0][3]}` e todas as configurações tiveram {heads_rows[0][-1]} parâmetros.",
        "",
        "**Causa técnica:** variar `H` apenas reorganiza a mesma dimensão total em subespaços; as matrizes de projeção continuam mapeando 32 para 32.",
        "",
        "**Relação com a teoria e consequência:** cada head calcula uma matriz `[T,T]` independente. Durante o treinamento, heads distintas podem especializar-se em relações diferentes, sem que a quantidade de heads, isoladamente, prove melhor qualidade.",
        "",
        "## 3. Diferentes dimensões por head",
        "",
        "**Resultado observado:** para `D=32`, usar 2, 4 e 8 heads produziu `head_dim` 16, 8 e 4, respectivamente.",
        "",
        "**Causa técnica:** a implementação exige `head_dim = d_out / num_heads` e divide o último eixo sem perder elementos.",
        "",
        "**Relação com a teoria e consequência:** mais heads, mantendo `D` fixo, significa mais subespaços paralelos, porém cada um possui menos componentes.",
        "",
        "## 4. Self-Attention × Multi-Head Attention",
        "",
        f"**Resultado observado:** a Self-Attention gerou pesos `{tuple(self_weights.shape)}` e {parameter_count(self_attention)} parâmetros; a Multi-Head gerou `{tuple(multi_weights.shape)}` e {parameter_count(multi_attention)} parâmetros. Ambas retornaram `{tuple(self_output.shape)}`.",
        "",
        "**Causa técnica:** a dimensão `H` aparece nos pesos da Multi-Head, as heads são concatenadas de volta em `D`, e a implementação Multi-Head inclui uma projeção linear final. Ela também aplica a máscara causal usada pelo GPT.",
        "",
        "**Relação com a teoria e consequência:** múltiplas heads permitem representar simultaneamente diferentes projeções e padrões de relação. A concatenação preserva a interface `[B,T,D]` necessária ao Transformer Block.",
        "",
        "## 5. Attention com escala × sem escala",
        "",
        f"**Resultado observado:** a média absoluta dos scores caiu de {scores_without_scale.abs().mean().item():.6f} para {scores_with_scale.abs().mean().item():.6f}; o maior peso caiu de {weights_without_scale.max().item():.6f} para {weights_with_scale.max().item():.6f}; a entropia média mudou de {unscaled_entropy:.6f} para {scaled_entropy:.6f}.",
        "",
        f"**Causa técnica:** os scores foram divididos por `sqrt(32) = {math.sqrt(32):.6f}` antes do softmax, reduzindo suas diferenças de magnitude.",
        "",
        "**Relação com a teoria e consequência:** scores grandes levam o softmax a distribuições mais saturadas. A escala mantém os pesos menos concentrados neste ensaio e favorece gradientes úteis em dimensões maiores.",
        "",
        "## 6. Máscara causal",
        "",
        f"**Resultado observado:** o maior peso acima da diagonal principal foi exatamente {maximum_future_weight:.6f}.",
        "",
        "**Causa técnica:** antes do softmax, as posições futuras receberam `-inf`; `exp(-inf)` é zero na normalização.",
        "",
        "**Relação com a teoria e consequência:** o GPT é autoregressivo. Bloquear `j > i` impede vazamento do token futuro que o modelo deve aprender a prever.",
        "",
        "## 7. Diferentes sequências de entrada",
        "",
        f"**Resultado observado:** as três sequências produziram matrizes com comprimentos {', '.join(str(row[1]) for row in sequence_rows)} e maiores pesos {', '.join(str(row[4]) for row in sequence_rows)}.",
        "",
        "**Causa técnica:** embeddings de token e posição distintos geram Q e K distintos, alterando os produtos escalares e o softmax.",
        "",
        "**Relação com a teoria e consequência:** o mecanismo reage numericamente à entrada e suporta comprimentos variáveis até `context_length`. Como os parâmetros são aleatórios, não se pode afirmar que uma head aprendeu semântica, sintaxe ou que o modelo compreendeu o texto.",
        "",
        "## Conclusão",
        "",
        "A execução confirma o fluxo `[B,T,D] → Q/K/V → [B,H,T,T] → [B,T,D]`. A escala controla a magnitude antes do softmax, a máscara elimina conexões futuras e a concatenação das heads devolve uma representação compatível com os Transformer Blocks da Sprint 4.",
        "",
    ]
    ANALYSIS_FILE.write_text("\n".join(analysis_sections), encoding="utf-8")

    print("Input IDs:", tuple(input_batch.shape))
    print("Embeddings:", tuple(embeddings.shape))
    print("Multi-Head output:", tuple(multi_output.shape))
    print("Multi-Head weights:", tuple(multi_weights.shape))
    print(f"Resultados salvos em: {RESULTS_FILE}")
    print(f"Análise salva em: {ANALYSIS_FILE}")
    print(f"Figuras salvas em: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
