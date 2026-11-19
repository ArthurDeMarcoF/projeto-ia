# Sprint 6 - Resultados dos experimentos

> Preencher esta analise com resultados REAIS dos comandos da Sprint 6.
> Nao substituir as metricas por numeros simulados.

## A. Pesos aleatorios x pesos pre-treinados

- Modelo externo utilizado:
- Quantidade de tensores copiados:
- Prompt usado:
- Geracao com pesos aleatorios:
- Geracao com pesos pre-treinados:
- Interpretacao: coerencia local, repeticoes e limitacoes.

## B. Classificacao de mensagens

- Arquivo de dados, numero de exemplos, quantidade de spam/ham:
- Quantidade de treino, validacao e teste:
- Camadas descongeladas, parametros treinaveis:
- Epocas, batch size, learning rate, dispositivo:
- Acuracia de validacao ANTES do fine-tuning:
- Acuracia de validacao DEPOIS do fine-tuning:
- Acuracia de TESTE final:
- Precisao, revocacao e F1 de spam no teste:
- Matriz de confusao (TN/FP/FN/TP):
- Discussao: erros de classificacao, desbalanceamento e generalizacao.

Copiar os resultados de `experimentos/resultados_sprint6/classificacao/metrics_classification.json`.

## C. Instruction fine-tuning

- Arquivo e quantidade de instrucoes:
- Divisao treino/validacao/teste:
- Camadas descongeladas, parametros treinaveis:
- Epocas, batch size, learning rate, dispositivo:
- Perda de validacao nos tokens da resposta ANTES do fine-tuning:
- Perda de validacao DEPOIS do fine-tuning:
- Perda por token e acuracia no conjunto de TESTE:
- Tres exemplos de instrucao, entrada e resposta esperada:
- Resposta gerada ANTES x DEPOIS:
- Discussao: fidelidade a instrucao, formato, repeticoes e limitacoes.

Copiar os resultados de `experimentos/resultados_sprint6/instrucoes/metrics_instructions.json`.

## D. Conclusao

A arquitetura construida nas etapas anteriores foi reutilizada com pesos
pre-treinados, classificador supervisionado e ajuste por instrucoes. Descrever
as mudancas observadas e limitar as conclusoes aos experimentos executados.
