# Resultados — Sprint 4

Medições locais em CPU com `torch.manual_seed(123)`, `torch.set_num_threads(1)`,
2 aquecimentos e 5 execuções por configuração. Latências variam conforme hardware e carga.
**Todas as redes possuem parâmetros aleatórios, não treinados.**

## 1. Comparação arquitetural

| Grupo | Valor | D | H | N | F | T | Logits | Parâmetros | Pesos FP32 (KiB) | Forward CPU (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Blocos | 1 | 32 | 4 | 1 | 128 | 8 | (2, 8, 216) | 26752 | 104.5 | 0.952 |
| Blocos | 2 | 32 | 4 | 2 | 128 | 8 | (2, 8, 216) | 39360 | 153.8 | 1.495 |
| Blocos | 4 | 32 | 4 | 4 | 128 | 8 | (2, 8, 216) | 64576 | 252.2 | 2.623 |
| Heads | 1 | 32 | 1 | 2 | 128 | 8 | (2, 8, 216) | 39360 | 153.8 | 1.338 |
| Heads | 2 | 32 | 2 | 2 | 128 | 8 | (2, 8, 216) | 39360 | 153.8 | 1.859 |
| Heads | 4 | 32 | 4 | 2 | 128 | 8 | (2, 8, 216) | 39360 | 153.8 | 1.267 |
| Heads | 8 | 32 | 8 | 2 | 128 | 8 | (2, 8, 216) | 39360 | 153.8 | 1.277 |
| Embedding D | 16 | 16 | 4 | 2 | 64 | 8 | (2, 8, 216) | 13536 | 52.9 | 1.453 |
| Embedding D | 32 | 32 | 4 | 2 | 128 | 8 | (2, 8, 216) | 39360 | 153.8 | 2.023 |
| Embedding D | 64 | 64 | 4 | 2 | 256 | 8 | (2, 8, 216) | 127872 | 499.5 | 1.705 |
| FeedForward F | 64 | 32 | 4 | 2 | 64 | 8 | (2, 8, 216) | 31040 | 121.2 | 1.341 |
| FeedForward F | 128 | 32 | 4 | 2 | 128 | 8 | (2, 8, 216) | 39360 | 153.8 | 1.481 |
| FeedForward F | 256 | 32 | 4 | 2 | 256 | 8 | (2, 8, 216) | 56000 | 218.8 | 1.650 |
| Contexto T | 4 | 32 | 4 | 2 | 128 | 4 | (2, 4, 216) | 39232 | 153.2 | 1.320 |
| Contexto T | 8 | 32 | 4 | 2 | 128 | 8 | (2, 8, 216) | 39360 | 153.8 | 1.405 |
| Contexto T | 16 | 32 | 4 | 2 | 128 | 16 | (2, 16, 216) | 39616 | 154.8 | 1.732 |

**B=2** em todas as entradas. D: embedding; H: heads; N: blocos; F: FF intermediária; T: contexto.
Memória FP32 considera apenas parâmetros (4 bytes por elemento), não ativações, gradientes ou otimizador.

## 2. Dropout

- Diferença média absoluta entre duas passagens em `train()` com dropout 0.3: `0.371949`.
- Diferença média absoluta entre duas passagens em `eval()`: `0.000000`.

## 3. Temperatura e top-k nos mesmos logits

| Temperatura | Top-k | Entropia (nats) | Maior probabilidade | Tokens possíveis |
| --- | --- | --- | --- | --- |
| 0.5 | Sem filtro | 4.59192 | 0.08873 | 216 |
| 1.0 | Sem filtro | 5.19743 | 0.02444 | 216 |
| 1.5 | Sem filtro | 5.29843 | 0.01459 | 216 |
| 1.0 | 5 | 1.58576 | 0.26561 | 5 |
| 1.0 | 10 | 2.25042 | 0.16865 | 10 |

A entropia é `-soma(p * ln(p))`. Top-k elimina logits fora das k maiores posições antes do softmax.

## 4. Geração autoregressiva

- Prompt Token IDs: `[22, 120, 64, 110]`.
- Greedy — 6 IDs previstos: `[11, 148, 162, 8, 83, 9]`.
- Amostragem — top-k=5, temperature=1.0, seed=123: `[11, 164, 106, 8, 83, 37]`.
- Greedy texto: `Um modelo de linguagem Esses posição quantos Depois embedding Em`.
- Amostragem texto: `Um modelo de linguagem Esses reais informação Depois embedding arquitetura`.

Essas sequências demonstram o fluxo de inferência; não representam linguagem aprendida.
