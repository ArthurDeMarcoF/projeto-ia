# Análise — Sprint 4

Os valores a seguir são obtidos da execução do script, em CPU, com pesos inicializados aleatoriamente.
Não houve treinamento e não é possível concluir qualidade linguística pelos textos produzidos.

## Número de blocos

**Resultado:** N=1 teve 26752 parâmetros, e N=4 teve 64576.
**Causa:** cada Transformer Block adiciona atenção, LayerNorm e FeedForward treináveis.
**Teoria/consequência:** maior profundidade amplia a capacidade representacional e o custo de computação, mas não garante melhor texto sem treinamento.

## Número de heads

**Resultado:** H=1 e H=8 tiveram ambos 39360 parâmetros, com D=32.
**Causa:** cada projeção continua D->D; alterar H só redistribui D em `head_dim=D/H`.
**Teoria/consequência:** o número de matrizes de atenção cresce com H; tempo e memória de ativações podem mudar, apesar dos parâmetros iguais.

## Dimensão de embedding

**Resultado:** D=16 usou 13536 parâmetros e D=64 usou 127872.
**Causa:** projeções e FeedForward têm matrizes cujo tamanho depende de D e F; neste ensaio F=4D.
**Teoria/consequência:** ampliar D aumenta capacidade e custo computacional e de memória.

## Dimensão FeedForward

**Resultado:** F=64 teve 31040 parâmetros e F=256 teve 56000.
**Causa:** as camadas D->F->D possuem pesos proporcionais a F.
**Teoria/consequência:** F amplia transformações por token; não altera dimensões de entrada/saída do bloco.

## Tamanho de contexto

**Resultado:** T=4 gerou logits (2, 4, 216) e T=16 gerou (2, 16, 216).
**Causa:** atenção calcula scores [B,H,T,T] e a embedding posicional possui T vetores.
**Teoria/consequência:** maior T permite mais posições no contexto, mas atenção usa memória quadrática em T; o custo de forward medido é dependente da CPU.

## Dropout e inferência

**Resultado:** diferença média em train=0.371949 versus eval=0.000000.
**Causa:** dropout sorteia posições apenas em train(), enquanto eval() desativa a regularização.
**Teoria/consequência:** a inferência deve operar com dropout desligado para evitar aleatoriedade não desejada.

## Temperatura, top-k e seleção

**Resultado:** a temperatura 0.5 teve entropia 4.59192, e 1.5 teve 5.29843; top-k=5 deixou 5 candidatos.
**Causa:** dividir logits por temperatura maior achata softmax; top-k filtra candidatos.
**Teoria/consequência:** a amostragem muda a seleção do próximo token, enquanto greedy escolhe o de maior probabilidade.
A geração é autoregressiva: o ID gerado entra no contexto da previsão seguinte.

## Limitações

- Pesos não treinados: o conteúdo textual não tem valor semântico demonstrável.
- Medições CPU locais: não são benchmarks comparáveis a outras máquinas.
- A Sprint 5 introduzirá função de perda, backpropagation e treinamento dos pesos.
