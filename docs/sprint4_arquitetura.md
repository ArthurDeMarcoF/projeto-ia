# Sprint 4 — Arquitetura GPT e geração autoregressiva

## Objetivo e continuidade

O código reutiliza `TokenAndPositionEmbedding` (`src/embeddings.py`, Sprint 2) e `MultiHeadAttention` (`src/attention.py`, Sprint 3). Não substitui o mecanismo de atenção pela versão pronta do PyTorch. A Sprint 4 implementa os componentes adicionais, a integração no GPT e a inferência; o ajuste dos pesos por treinamento será realizado na Sprint 5.

## Etapas e dimensões

Legenda: `B` = batch size; `T` = comprimento da sequência; `D` = dimensão do embedding; `H` = heads; `d_h = D/H`; `F` = largura da FeedForward; `N` = número de blocos; `V` = tamanho do vocabulário.

| Etapa | Entrada | Saída |
| --- | --- | --- |
| Token IDs | texto tokenizado | `[B,T]` |
| Embeddings de token + posição (Sprint 2) | `[B,T]` | `[B,T,D]` |
| Atenção multi-head (Sprint 3, dentro do bloco) | `[B,T,D]` | `[B,T,D]`; pesos `[B,H,T,T]` |
| LayerNorm | `[B,T,D]` | `[B,T,D]` |
| FeedForward | `[B,T,D]` | `[B,T,F]` intermediário e `[B,T,D]` final |
| Bloco Transformer (repetido `N` vezes) | `[B,T,D]` | `[B,T,D]` |
| LayerNorm final | `[B,T,D]` | `[B,T,D]` |
| Language Model Head | `[B,T,D]` | **logits** `[B,T,V]` |
| Softmax dos logits na última posição | `[B,V]` | probabilidades `[B,V]` |

### Como o Transformer Block funciona

Este projeto segue **pre-LN**, isto é, normaliza antes de cada subcamada:

```text
Entrada x [B,T,D]
  |-- Atalho residual -------------------------------|
  +-- LayerNorm -> Multi-Head Attention -> Dropout -> soma = x1

Entrada x1 [B,T,D]
  |-- Atalho residual -------------------------------------------|
  +-- LayerNorm -> Linear(D,F) -> GELU -> Linear(F,D) -> Dropout -> soma = y

Saída y [B,T,D]
```

Matematicamente: `x1 = x + Dropout(Attention(LayerNorm(x)))`; `y = x1 + Dropout(FeedForward(LayerNorm(x1)))`.

A LayerNorm normaliza cada token sobre `D` características, com escala (`gamma`) e deslocamento (`beta`) treináveis. As conexões residuais conservam a dimensão e abrem caminhos diretos para o gradiente. A FeedForward não mistura posições: o compartilhamento de informações entre tokens acontece na atenção.

### Geração autoregressiva

O modelo prevê as pontuações de todos os tokens do vocabulário em cada posição (`[B,T,V]`). Para continuar a sequência, utiliza-se a **última posição**: `logits[:, -1, :]` (`[B,V]`). Divide-se por temperatura, aplica-se top-k quando solicitado, e o softmax transforma pontuações em probabilidades. A escolha pode ser `argmax` (greedy) ou amostragem `torch.multinomial`; o ID selecionado é concatenado aos IDs anteriores e o processo recomeça.

**Exemplo:** `[A, B, C] -> prevê D -> [A, B, C, D] -> prevê E -> ...`. Durante a geração, sequências maiores que `context_length` têm a parte mais antiga descartada na entrada do modelo; os IDs gerados continuam presentes no tensor retornado.

### Limites desta Sprint

- O GPT usa **pesos iniciais aleatórios**: o texto gerado pode não ter coerência, e isso não é falha do laço autoregressivo.
- Logits **não são probabilidades**; o softmax as produz. Não há treinamento na Sprint 4.
- O modelo **não usa pesos pré-treinados**. Sem a Sprint 5 não é possível afirmar aprendizado.
- Os resultados de latência CPU variam conforme hardware; o script mede, e não estima, o tempo local.
- O tokenizador da Sprint 2 é didático e possui `<|unk|>` para tokens não vistos.

## Como executar

Na raiz do repositório:

```bash
pip install -r requirements.txt
python -m pytest -q
python src/demo_sprint4.py
python experimentos/sprint4_experimentos.py
```

A última execução grava `experimentos/resultados_sprint4.md` e `experimentos/analise_sprint4.md` com medições e interpretação. `notebooks/sprint4_gpt.ipynb` mostra o fluxo de forma interativa.

## Preparação para apresentação

**Por que a saída do bloco ainda tem D características?** Porque as conexões residuais exigem dimensões compatíveis e a segunda projeção da FeedForward retorna de F para D.

**Qual diferença entre número de heads e dimensão D?** Com D fixo, `head_dim = D/H`. Aumentar H não aumenta, por si só, o número de parâmetros das projeções Q/K/V na implementação atual.

**Por que o modelo não escreve texto coerente?** Porque os parâmetros ainda são aleatórios; a geração prova o mecanismo matemático, não aprendizagem linguística.

**Para que serve a máscara causal?** Para que um token não utilize posições futuras ao calcular suas representações. Isso viabiliza a previsão do token seguinte sem vazamento.

**O que a Sprint 5 acrescentará?** Perda de predição do próximo token, gradientes, otimizador, atualização de pesos e avaliação após treino.
