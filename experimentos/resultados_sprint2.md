# Resultados dos Experimentos — Sprint 2

Resultados gerados automaticamente por `experimentos/sprint2_experimentos.py`.

**Tokens no corpus (tokenizador didático):** 416
**Tamanho do vocabulário:** 216

## Experimento 1 — Diferentes sequências de entrada

| Texto | Tokens | Token IDs |
| --- | --- | --- |
| Um modelo de linguagem aprende padrões. | 7 | 7 |
| O texto precisa ser dividido em unidades menores. | 9 | 9 |
| A ordem dos tokens também precisa ser representada. | 9 | 9 |
| O Dataset organiza as amostras e o DataLoader combina várias amostras em lotes. | 14 | 14 |

**Observação:** textos maiores ou com mais sinais de pontuação tendem a produzir mais unidades de processamento. A quantidade exata depende das regras do tokenizador.

## Experimento 2 — Tamanho do contexto × quantidade de amostras

| Context length | Stride | Amostras |
| --- | --- | --- |
| 4 | 1 | 412 |
| 8 | 1 | 408 |
| 16 | 1 | 400 |
| 32 | 1 | 384 |

**Observação:** mantendo o stride fixo, aumentar o contexto reduz a quantidade de janelas válidas que cabem no mesmo texto.

## Experimento 3 — Diferentes tamanhos de lote

| Batch size | Amostras | Lotes | Input shape | Target shape |
| --- | --- | --- | --- | --- |
| 1 | 51 | 51 | (1, 8) | (1, 8) |
| 2 | 51 | 26 | (2, 8) | (2, 8) |
| 4 | 51 | 13 | (4, 8) | (4, 8) |
| 8 | 51 | 7 | (8, 8) | (8, 8) |

**Observação:** o batch size não altera as amostras do Dataset; ele altera quantas amostras são agrupadas em cada iteração e, consequentemente, o número de lotes.

## Experimento 4 — Dimensão do embedding e custo

| Embedding dim | Output shape | Parâmetros | Memória aprox. (KB, float32) | Tempo médio forward (ms) |
| --- | --- | --- | --- | --- |
| 8 | (4, 8, 8) | 1792 | 7.00 | 0.0160 |
| 16 | (4, 8, 16) | 3584 | 14.00 | 0.0158 |
| 32 | (4, 8, 32) | 7168 | 28.00 | 0.0196 |
| 64 | (4, 8, 64) | 14336 | 56.00 | 0.0159 |

**Observação:** aumentar `embedding_dim` aumenta a última dimensão do tensor e também a quantidade de parâmetros das matrizes de embeddings. O tempo é apenas indicativo e deve ser interpretado com cautela, pois depende do hardware e do ambiente.

## Experimento 5 — Efeito da posição

| Verificação | Resultado | Referência |
| --- | --- | --- |
| Token ID nas duas posições | 120 | 120 |
| Token embedding é igual? | True | esperado: True |
| Positional embedding é igual? | False | esperado: False |
| Representação final é igual? | False | esperado: False |

**Observação:** o mesmo Token ID possui o mesmo token embedding, mas recebe vetores posicionais diferentes. Após a soma, as representações finais tornam-se diferentes, permitindo ao Transformer distinguir posições distintas da sequência.
