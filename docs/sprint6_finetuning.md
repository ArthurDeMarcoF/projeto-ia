# Sprint 6 - Carregamento de pesos e Fine-Tuning

**Projeto:** Construcao de um Large Language Model (From Scratch)

**Autores:** Arthur de Marco Faggion e Lucas Zamoner Locatelli

**Prazo:** 03/12/2026

## 1. Objetivos

A Sprint 6 encerra o desenvolvimento incremental do GPT iniciado nas etapas anteriores.
O foco esta em transferir pesos pre-treinados de um GPT-2 para a arquitetura
propria, medir como uma base pre-treinada se comporta em geracao e adaptar a
rede a (a) classificacao de SMS e (b) resposta a instrucoes.

A implementacao preserva o codigo produzido nas Sprints 2-5. Os novos scripts
fazem uso de `src.gpt.GPTModel`, do mecanismo de atencao manual e dos blocos
Transformer ja implementados, sem substituir a inferencia pelo wrapper externo.

## 2. Entregavel 1 - Glossario

O conjunto adicional de 18 conceitos da Sprint 6 e incorporado ao documento
`docs/glossario.md`, mantendo as definicoes anteriores. Os 14 conceitos
obrigatorios foram contemplados, com acronimos e exemplos de aplicacao.

## 3. Entregavel 2 - Carregamento de pesos

Os pesos obtidos de um checkpoint GPT-2 pre-treinado atraves de `transformers`
sao transferidos manualmente para `src.gpt.GPTModel`, elemento por elemento,
conforme a associacao abaixo:

| Fonte no GPT-2 | Destino na arquitetura do projeto | Operacao |
| --- | --- | --- |
| `transformer.wte.weight` | `embeddings.token_embedding.weight` | Copia |
| `transformer.wpe.weight` | `embeddings.position_embedding.weight` | Copia |
| `h[i].ln_1 / ln_2` | `blocks[i].norm1 / norm2` | Peso -> escala; bias -> shift |
| `h[i].attn.c_attn` | `blocks[i].attention.query/key/value_projection` | Separacao Q/K/V e transposicao de matriz |
| `h[i].attn.c_proj` | `blocks[i].attention.out_projection` | Transposicao da matriz de peso |
| `h[i].mlp.c_fc / c_proj` | `blocks[i].feed_forward.layers[0] / [2]` | Transposicao da matriz de peso |
| `transformer.ln_f` | `final_norm` | Peso -> escala; bias -> shift |
| `lm_head.weight` | `lm_head.weight` | Copia |

A diferenca na orientacao das matrizes ocorre porque a camada `Conv1D`
usada pelo GPT-2 disponibiliza pesos na orientacao `[entrada, saida]`, enquanto
a camada `nn.Linear` utilizada no projeto emprega `[saida, entrada]`.

O checkpoint da Sprint 5 nao pode receber esses pesos diretamente: possui
vocabulário local de 182 tokens, embeddings de 32 dimensoes e 2 blocos. O
GPT-2 pequeno emprega vocabulário de 50.257 tokens, dimensao 768, 12 blocos e
12 cabecas de atencao, com contexto de 1.024 posicoes. O tokenizador GPT-2
BPE e carregado para manter a equivalencia dos Token IDs. Nao se pode
substitui-lo pelo tokenizador didatico de palavras da Sprint 2.

A geracao e comparada com duas instancias da MESMA arquitetura e mesmo prompt:
uma inicializada aleatoriamente e outra com os pesos pre-treinados. O comando
`--compare-random` realiza essa comparacao com `src.generation.generate_text`.
Nao confundir o treino local diminuto da Sprint 5 com um pre-treinamento
externo abrangente; ambos sao pesos aprendidos, mas com escalas e corpus
muito diferentes.

**Evidencia a registrar:** prompt, pesos carregados, configuracao da rede,
saida do modelo aleatorio e saida do modelo pre-treinado. Executar o comando
explicado no `README.md` e copiar as saidas reais para o relatorio.

## 4. Entregavel 3 - Fine-tuning para classificacao

O modulo `src/finetune_classification.py` le a base UCI de SMS em formato TSV
ou CSV, converte os rotulos `ham` e `spam` para 0 e 1, normaliza espacos e
remove mensagens duplicadas e aplica tokenizacao BPE do GPT-2. A divisao estratificada reserva 70% dos
exemplos para treino, 15% para validacao e 15% para teste (proporcoes
aproximadas por classe).

O `MessageDataset` retorna tokens e rotulos, e `collate_messages` faz padding
com o Token ID `50256`. A cabeca de classificacao (`nn.Linear(768,2)` para o
GPT-2 pequeno) transforma a representacao da ULTIMA posicao real do texto em
dois logits. No processo de batelada, a posicao do ultimo token real e
recuperada do comprimento de cada exemplo, impedindo que o padding seja
considerado como texto do usuario.

Todas as camadas sao congeladas inicialmente. Sao liberados por padrao apenas
o ultimo bloco Transformer, a normalizacao final e a cabeca de classificacao.
O objetivo e a entropia cruzada supervisionada, com pesos de classe estimados
exclusivamente a partir do conjunto de treino para diminuir o efeito do
desbalanceamento. O treinamento usa AdamW.

## 5. Entregavel 4 - Avaliacao da classificacao

O algoritmo mede as metricas de validacao ANTES do fine-tuning e em cada epoca
subsequente. Seleciona o estado de menor perda de validacao e, somente depois,
avalia o conjunto de teste reservado. Isso separa selecao de modelo e
avaliacao final.

As metricas gravadas em `metrics_classification.json` sao acuracia, precisao
para spam, revocacao para spam, F1 para spam e matriz de confusao com valores
TN, FP, FN e TP. A comparacao antes/depois comeca com uma nova cabeca de
classificacao de pesos aleatorios conectada a um GPT ja pre-treinado, de modo
que a linha-base nao representa um classificador treinado.

**Resultados numericos:** preencher com a saida real da execucao. Nao existem
numeros de teste aferidos para esta Sprint enquanto o script nao for executado
com a base real. A acuracia deve ser interpretada junto do desbalanceamento.

## 6. Entregavel 5 - Fine-tuning para instrucoes

O modulo `src/finetune_instructions.py` recebe uma lista JSON com os campos
`instruction`, `input` opcional e `output` (ou `response`). O prompt contem
marcadores de instrucao, entrada e resposta. A rede precisa prever os tokens
seguintes da resposta; os tokens do prompt e do padding recebem alvo `-100`
para nao participarem da perda supervisionada.

O GPT preserva a sua camada de saida para vocabulario (`lm_head`) em vez de
receber uma cabeca de classificacao. Sao treinaveis, por padrao, o ultimo
bloco Transformer, a normalizacao final e `lm_head`. Treino, validacao e
teste permanecem separados (70/15/15 aproximadamente). A selecao do checkpoint
e feita pela menor perda de tokens de resposta na validacao.

A avaliacao registra perda media por token de resposta e acuracia por token
nos dados reservados. Alem disso, produz respostas completas para alguns
exemplos de teste ANTES e DEPOIS do ajuste, juntamente com a referencia
esperada. A comparacao qualitativa deve verificar adequacao, completude e
possiveis repeticoes ou alucinacoes. Acuracia por token nao equivale a uma
metrica geral de qualidade linguistica.

## 7. Reprodutibilidade e limitacoes

Dependencias adicionais: `transformers` e `safetensors`. A carga dos pesos
externos e do tokenizador necessita de internet na primeira execucao. O
fine-tuning de GPT-2 pode ser demorado em CPU; uma GPU compativel com CUDA
permite experimentar com mais exemplos ou epocas. Fixar semente e divisao de
dados, registrar hiperparametros, verificar os exemplos reservados e salvar
os resultados JSON sao requisitos para uma comparacao reprodutivel.

O arquivo `data/instructions_example.json` contem apenas dez exemplos criados
para validar o formato do pipeline. NAO representa o dataset da disciplina
nem deve ser usado para tirar conclusoes de desempenho. O dataset
padronizado da disciplina deve ser colocado em `data/instructions.json` e a
base de spam deve ser fornecida em `data/SMSSpamCollection` ou pelo caminho
informado em `--data`.

## 8. Referencias

- Sebastian Raschka. *Build a Large Language Model (From Scratch)*, capitulos 6 e 7.
- Hugging Face. GPT-2 Model Documentation: https://huggingface.co/docs/transformers/model_doc/gpt2
- UCI Machine Learning Repository. SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Repositorio de codigo da obra: https://github.com/rasbt/LLMs-from-scratch
