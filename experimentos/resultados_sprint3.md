# Resultados dos Experimentos — Sprint 3

Resultados gerados automaticamente por `experimentos/sprint3_experimentos.py` com `torch.manual_seed(123)`.

**Input IDs real:** `(2, 8)`
**Embeddings base:** `(2, 8, 32)`
**Tokens da primeira amostra:** `Um | modelo | de | linguagem | aprende | padrões | a | partir`

Os pesos são de parâmetros inicializados aleatoriamente; os resultados demonstram o mecanismo matemático, não relações linguísticas aprendidas.

## Experimento 1 — Diferentes dimensões de embedding

| Embedding dim | Input shape | Output shape | Heads | Head dim | Parâmetros da atenção |
| --- | --- | --- | --- | --- | --- |
| 8 | (2, 8, 8) | (2, 8, 8) | 4 | 2 | 264 |
| 16 | (2, 8, 16) | (2, 8, 16) | 4 | 4 | 1040 |
| 32 | (2, 8, 32) | (2, 8, 32) | 4 | 8 | 4128 |
| 64 | (2, 8, 64) | (2, 8, 64) | 4 | 16 | 16448 |

## Experimento 2 — Diferentes números de heads

| Embedding dim | Heads | Head dim | Output shape | Attention weights shape | Parâmetros |
| --- | --- | --- | --- | --- | --- |
| 32 | 1 | 32 | (2, 8, 32) | (2, 1, 8, 8) | 4128 |
| 32 | 2 | 16 | (2, 8, 32) | (2, 2, 8, 8) | 4128 |
| 32 | 4 | 8 | (2, 8, 32) | (2, 4, 8, 8) | 4128 |
| 32 | 8 | 4 | (2, 8, 32) | (2, 8, 8, 8) | 4128 |

## Experimento 3 — Diferentes dimensões por head

| Embedding dim | Heads | Head dim | Cálculo |
| --- | --- | --- | --- |
| 32 | 2 | 16 | 32 / 2 |
| 32 | 4 | 8 | 32 / 4 |
| 32 | 8 | 4 | 32 / 8 |

## Experimento 4 — Self-Attention × Multi-Head Attention

| Mecanismo | Heads | Pesos | Saída | Parâmetros |
| --- | --- | --- | --- | --- |
| Self-Attention (não causal) | 1 | (2, 8, 8) | (2, 8, 32) | 3072 |
| Multi-Head Causal Attention | 4 | (2, 4, 8, 8) | (2, 8, 32) | 4128 |

## Experimento 5 — Attention com escala × sem escala

Neste experimento, `d_k = 32` e o divisor é `sqrt(d_k) = 5.656854`.

| Configuração | Média absoluta dos scores | Máximo absoluto dos scores | Maior peso | Entropia média |
| --- | --- | --- | --- | --- |
| Sem escala | 3.911841 | 14.925028 | 0.999910 | 0.528817 |
| Com escala 1/sqrt(d_k) | 0.691522 | 2.638397 | 0.565670 | 1.824351 |

A entropia de cada linha é calculada por `-soma(p * ln(p))`; valores menores indicam uma distribuição mais concentrada.

## Experimento 6 — Máscara causal

### Scores escalados antes da máscara (amostra 1)

```text
[0.235, 0.629, 0.647, 0.029, -1.513, 0.414, 1.009, 0.160]
[-0.862, 0.104, 0.269, -1.555, 0.650, -0.323, 0.775, 0.725]
[1.233, 0.233, -0.841, 0.233, -0.065, -0.290, -0.078, -0.392]
[-0.707, -0.005, 0.474, -0.486, 0.226, -0.300, -0.439, 0.789]
[0.047, -0.059, -0.320, -0.008, -0.076, -0.640, -0.002, -0.255]
[-0.526, 0.184, -0.381, -0.003, 1.183, -0.429, -0.292, 0.740]
[-0.713, -0.697, -0.948, -0.218, 1.613, -0.441, -0.787, 0.896]
[-0.039, 0.121, -0.026, -0.001, 0.071, -0.882, 0.416, 0.724]
```

### Máscara causal (1 = permitido; 0 = futuro bloqueado)

```text
[1, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 0, 0]
[1, 1, 1, 1, 1, 1, 1, 0]
[1, 1, 1, 1, 1, 1, 1, 1]
```

### Attention weights depois da máscara e do softmax (amostra 1)

```text
[1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
[0.276, 0.724, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
[0.670, 0.246, 0.084, 0.000, 0.000, 0.000, 0.000, 0.000]
[0.133, 0.268, 0.433, 0.166, 0.000, 0.000, 0.000, 0.000]
[0.226, 0.203, 0.157, 0.214, 0.200, 0.000, 0.000, 0.000]
[0.080, 0.163, 0.092, 0.135, 0.442, 0.088, 0.000, 0.000]
[0.059, 0.060, 0.047, 0.097, 0.605, 0.078, 0.055, 0.000]
[0.105, 0.124, 0.107, 0.109, 0.118, 0.045, 0.166, 0.226]
```

**Maior peso em posição futura (`j > i`):** `0.000000`

## Experimento 7 — Diferentes sequências de entrada

| Texto | Tokens | Pesos | Saída | Maior peso (queries i > 0) | Entropia média |
| --- | --- | --- | --- | --- | --- |
| Um modelo de linguagem aprende padrões. | 7 | (1, 4, 7, 7) | (1, 7, 32) | 0.912294 | 1.094098 |
| A ordem dos tokens também precisa ser representada. | 9 | (1, 4, 9, 9) | (1, 9, 32) | 0.752449 | 1.275647 |
| O projeto é incremental. | 5 | (1, 4, 5, 5) | (1, 5, 32) | 0.671201 | 0.872782 |

As matrizes diferem porque as entradas e posições projetadas são diferentes. Como não houve treinamento, elas não sustentam interpretações semânticas, sintáticas ou de compreensão.

## Figuras geradas

- `figuras/sprint3_self_attention.png`
- `figuras/sprint3_causal_attention.png`
- `figuras/sprint3_multi_head_1.png` a `sprint3_multi_head_4.png`
- `figuras/sprint3_diferentes_sequencias.png`
