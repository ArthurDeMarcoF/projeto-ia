# Análise dos Resultados — Sprint 2

Esta análise relaciona os resultados produzidos em `resultados_sprint2.md` com os conceitos estudados no Capítulo 2.

## 1. Por que um LLM não pode trabalhar diretamente com texto bruto?

Redes neurais executam operações matemáticas sobre tensores numéricos. Uma string como `"modelo de linguagem"` não pode ser multiplicada por matrizes de pesos ou utilizada diretamente em operações de atenção. Por isso, o texto passa por uma sequência de transformações: primeiro é dividido em tokens, depois os tokens são convertidos em Token IDs e, por fim, os IDs são usados para consultar vetores de embeddings.

O resultado do pipeline deixa de ser texto e passa a possuir uma estrutura numérica bem definida. Antes dos embeddings, um lote possui formato `[batch_size, context_length]`. Depois dos embeddings, passa a possuir formato `[batch_size, context_length, embedding_dim]`.

## 2. Qual é a função do vocabulário?

O vocabulário define o mapeamento entre cada token reconhecido e um identificador inteiro único. Ele funciona como uma tabela de referência usada tanto no processo de codificação (`token → Token ID`) quanto no processo inverso (`Token ID → token`).

No tokenizador didático desta Sprint, o vocabulário também contém tokens especiais, como `<|unk|>` para unidades desconhecidas e `<|endoftext|>` para marcar separações ou fim de texto.

## 3. Qual é a diferença entre um token e um Token ID?

Um token ainda é uma unidade textual, como uma palavra, subpalavra ou sinal de pontuação. Um Token ID é apenas o número inteiro associado a esse token no vocabulário.

Por exemplo, um token pode ser `"modelo"`, enquanto seu ID pode ser `42`. O valor `42` não significa que esse token é semanticamente maior, menor ou mais importante que o token de ID `10`; o número funciona apenas como índice.

## 4. Por que os Token IDs não são utilizados diretamente como representação semântica?

Os IDs são valores discretos e arbitrários. A distância numérica entre dois IDs não representa semelhança de significado. Se `"modelo"` possui ID 42 e `"linguagem"` ID 43, isso não significa que eles sejam mais semelhantes do que outro token com ID 100.

Usar os IDs diretamente faria a rede interpretar uma numeração administrativa como se fosse uma relação matemática. Por isso, os IDs são usados como índices de uma matriz de embeddings.

## 5. Qual é a função dos embeddings?

A camada de embeddings transforma cada Token ID em um vetor de números reais com dimensão `embedding_dim`. Esses vetores são parâmetros treináveis. No início do treinamento eles não carregam uma organização semântica aprendida; essa estrutura é ajustada gradualmente durante o treinamento do modelo.

Nos experimentos, aumentar a dimensão do embedding aumentou simultaneamente a última dimensão da saída e a quantidade de parâmetros das matrizes de embeddings. Portanto, uma representação maior aumenta a capacidade de representação, mas também aumenta memória e custo computacional.

## 6. Por que é necessário representar a posição dos tokens?

O Transformer não recebe a ordem dos tokens por meio de recorrência. Assim, apenas consultar o token embedding não seria suficiente para distinguir o mesmo token usado em posições diferentes.

O experimento de posição utiliza o mesmo Token ID duas vezes. Os dois token embeddings são iguais, porém os positional embeddings são diferentes. Depois da soma entre os dois componentes, as representações finais se tornam diferentes. Isso demonstra que a posição acrescenta informação de ordem sem alterar a dimensão final do tensor.

## 7. Qual é a relação entre tamanho do contexto e quantidade de amostras de treinamento?

O contexto define quantos tokens compõem cada sequência de entrada. Em um corpus fixo e com o mesmo `stride`, um contexto maior exige mais tokens para formar cada amostra. Consequentemente, sobram menos posições válidas para iniciar novas janelas.

O experimento com `stride = 1` mostra essa redução diretamente: conforme `context_length` aumenta, a quantidade de amostras do Dataset diminui. Em contrapartida, cada amostra fornece ao modelo uma quantidade maior de contexto anterior.

## 8. Qual é o impacto da dimensão do embedding?

Se a entrada possui formato `[B, T]`, a camada de embeddings gera uma saída `[B, T, D]`, onde:

- `B` é o `batch_size`;
- `T` é o `context_length`;
- `D` é o `embedding_dim`.

Aumentar `D` aumenta a quantidade de valores armazenados para cada token e também o número de parâmetros treináveis. Na implementação desta Sprint, os parâmetros de embeddings crescem aproximadamente de forma linear com `embedding_dim`, pois existem uma matriz para os tokens e outra para as posições.

## 9. Qual é a função do DataLoader no pipeline?

O `Dataset` define quais são as amostras individuais de entrada e alvo. Já o `DataLoader` organiza essas amostras em lotes e controla aspectos como `batch_size`, embaralhamento e descarte do último lote incompleto.

Os experimentos demonstram que mudar o `batch_size` não altera o conteúdo nem a quantidade total de amostras criadas pelo Dataset. O que muda é a forma como essas amostras são agrupadas e a quantidade de iterações necessárias para percorrer os dados.

## 10. O que desta Sprint será usado pelo mecanismo de atenção?

A Sprint seguinte receberá a saída produzida após a soma entre Token Embeddings e Positional Embeddings. O formato esperado é:

`[batch_size, context_length, embedding_dim]`

O mecanismo de atenção utilizará esses vetores para construir as representações de **queries**, **keys** e **values**. Portanto, tokenização, organização das sequências, tamanho do contexto e dimensão dos embeddings definidos nesta Sprint influenciam diretamente a entrada da atenção.

## Conclusão

Os experimentos confirmam o encadeamento das etapas do processamento inicial. O texto bruto precisa ser transformado em unidades discretas, os tokens precisam ser mapeados para IDs, e esses IDs precisam ser convertidos em vetores treináveis. A informação posicional complementa os vetores com a ordem da sequência, enquanto `Dataset` e `DataLoader` organizam os dados em tensores compatíveis com o treinamento.

O principal resultado desta Sprint é a produção de um tensor tridimensional contendo uma representação vetorial para cada token de cada sequência do lote. Esse tensor constitui a interface entre a preparação dos dados realizada agora e o mecanismo de atenção que será implementado na Sprint seguinte.
