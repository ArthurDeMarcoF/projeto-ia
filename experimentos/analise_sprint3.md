# Análise dos Resultados — Sprint 3

Esta análise é gerada junto aos resultados e utiliza somente valores medidos na execução com seed 123. As projeções Q/K/V ainda não foram treinadas.

## 1. Diferentes dimensões de embedding

**Resultado observado:** as quatro configurações preservaram `[B, T] = [2, 8]` e produziram a última dimensão 8, 16, 32 ou 64. Os parâmetros da atenção passaram de 264 para 16448.

**Causa técnica:** as projeções Q, K, V e a projeção final são matrizes lineares cuja quantidade de elementos cresce com as dimensões de entrada e saída.

**Relação com a teoria e consequência:** `D` define a largura da representação contextual. Aumentá-la oferece mais componentes vetoriais por token, mas aumenta parâmetros e operações; nenhuma conclusão de velocidade foi feita sem medição.

## 2. Diferentes números de heads

**Resultado observado:** com `D=32`, os pesos mudaram de `(2, 1, 8, 8)` para `(2, 8, 8, 8)`, enquanto todas as saídas permaneceram `(2, 8, 32)` e todas as configurações tiveram 4128 parâmetros.

**Causa técnica:** variar `H` apenas reorganiza a mesma dimensão total em subespaços; as matrizes de projeção continuam mapeando 32 para 32.

**Relação com a teoria e consequência:** cada head calcula uma matriz `[T,T]` independente. Durante o treinamento, heads distintas podem especializar-se em relações diferentes, sem que a quantidade de heads, isoladamente, prove melhor qualidade.

## 3. Diferentes dimensões por head

**Resultado observado:** para `D=32`, usar 2, 4 e 8 heads produziu `head_dim` 16, 8 e 4, respectivamente.

**Causa técnica:** a implementação exige `head_dim = d_out / num_heads` e divide o último eixo sem perder elementos.

**Relação com a teoria e consequência:** mais heads, mantendo `D` fixo, significa mais subespaços paralelos, porém cada um possui menos componentes.

## 4. Self-Attention × Multi-Head Attention

**Resultado observado:** a Self-Attention gerou pesos `(2, 8, 8)` e 3072 parâmetros; a Multi-Head gerou `(2, 4, 8, 8)` e 4128 parâmetros. Ambas retornaram `(2, 8, 32)`.

**Causa técnica:** a dimensão `H` aparece nos pesos da Multi-Head, as heads são concatenadas de volta em `D`, e a implementação Multi-Head inclui uma projeção linear final. Ela também aplica a máscara causal usada pelo GPT.

**Relação com a teoria e consequência:** múltiplas heads permitem representar simultaneamente diferentes projeções e padrões de relação. A concatenação preserva a interface `[B,T,D]` necessária ao Transformer Block.

## 5. Attention com escala × sem escala

**Resultado observado:** a média absoluta dos scores caiu de 3.911841 para 0.691522; o maior peso caiu de 0.999910 para 0.565670; a entropia média mudou de 0.528817 para 1.824351.

**Causa técnica:** os scores foram divididos por `sqrt(32) = 5.656854` antes do softmax, reduzindo suas diferenças de magnitude.

**Relação com a teoria e consequência:** scores grandes levam o softmax a distribuições mais saturadas. A escala mantém os pesos menos concentrados neste ensaio e favorece gradientes úteis em dimensões maiores.

## 6. Máscara causal

**Resultado observado:** o maior peso acima da diagonal principal foi exatamente 0.000000.

**Causa técnica:** antes do softmax, as posições futuras receberam `-inf`; `exp(-inf)` é zero na normalização.

**Relação com a teoria e consequência:** o GPT é autoregressivo. Bloquear `j > i` impede vazamento do token futuro que o modelo deve aprender a prever.

## 7. Diferentes sequências de entrada

**Resultado observado:** as três sequências produziram matrizes com comprimentos 7, 9, 5 e maiores pesos 0.912294, 0.752449, 0.671201.

**Causa técnica:** embeddings de token e posição distintos geram Q e K distintos, alterando os produtos escalares e o softmax.

**Relação com a teoria e consequência:** o mecanismo reage numericamente à entrada e suporta comprimentos variáveis até `context_length`. Como os parâmetros são aleatórios, não se pode afirmar que uma head aprendeu semântica, sintaxe ou que o modelo compreendeu o texto.

## Conclusão

A execução confirma o fluxo `[B,T,D] → Q/K/V → [B,H,T,T] → [B,T,D]`. A escala controla a magnitude antes do softmax, a máscara elimina conexões futuras e a concatenação das heads devolve uma representação compatível com os Transformer Blocks da Sprint 4.
