
# Construção de um Large Language Model (From Scratch) 🧠

Este repositório armazena todo o desenvolvimento do **Projeto Final** do componente curricular de Inteligência Artificial e Sistemas Inteligentes do curso de Engenharia de Computação (UNOESC).

## 🎯 Objetivo do Projeto

O objetivo principal é compreender os fundamentos computacionais dos modelos de linguagem por meio da implementação prática e incremental dos principais componentes de uma arquitetura baseada em Transformer, construindo um LLM do zero.

A referência principal para este desenvolvimento é a obra *"Build a Large Language Model (From Scratch)"*, de Sebastian Raschka. Diferentemente do uso de APIs prontas, este projeto foca na implementação progressiva das estruturas internas do modelo utilizando **Python** e **PyTorch**.

## 👥 Autores (Dupla)

* Arthur de Marco Faggion
* Lucas Zamoner Locatelli

## 📂 Estrutura do Repositório

Para manter a organização e acompanhar a evolução do projeto, os artefatos estão divididos nos seguintes diretórios:

* `/src`: Código-fonte das implementações (tokenização, embeddings, atenção, GPT, etc.).
* `/notebooks`: Jupyter Notebooks utilizados para testes, desenvolvimento passo a passo e visualização.
* `/docs`: Documentação do projeto, incluindo o **Glossário Técnico cumulativo**.
* `/experimentos`: Relatórios, análises de resultados e testes de diferentes configurações e parâmetros.

Na Sprint 3, `src/attention.py` implementa manualmente Self-Attention, Scaled Dot-Product Attention, Causal Attention e Multi-Head Attention. Os experimentos e heatmaps correspondentes estão em `/experimentos`.

Na Sprint 4, `src/transformer.py` implementa LayerNorm, FeedForward e Transformer Block. `src/gpt.py` monta o GPT e `src/generation.py` executa a geração autoregressiva. As demonstrações dessa sprint usam pesos iniciais e podem ser executadas com `python src/demo_sprint4.py` e `python experimentos/sprint4_experimentos.py`. [Documentação da Sprint 4](docs/sprint4_arquitetura.md)

Na Sprint 5, o mesmo GPT é treinado no corpus local para prever o próximo token. A implementação registra entropia cruzada e perplexidade, compara três configurações de treino e gera texto a partir dos checkpoints. [Documentação e resultados da Sprint 5](docs/sprint5_treinamento.md)


### 📚 Glossário Técnico

O glossário será atualizado de forma cumulativa ao longo das Sprints.

➡️ [Acessar Glossário Técnico](docs/glossario.md)

➡️ [Revisar Quiz da Sprint 3](docs/quiz_sprint3.md)

## 🚀 Sprints e Evolução

- [X] **Sprint 0:** Preparação do ambiente (Python, PyTorch e Git), criação e organização do repositório.
- [X] **Sprint 1:** Introdução aos LLMs, Glossário do Capítulo 1 e arquitetura geral dos modelos GPT.
- [X] **Sprint 2:** Tokenização, vocabulário, Token IDs, Embeddings e preparação dos dados (Capítulo 2).
- [X] **Sprint 3:** Mecanismos de Attention (Self-Attention, Causal Attention e Multi-Head Attention) (Capítulo 3).
- [X] **Sprint 4:** Arquitetura GPT e Transformer Blocks (Capítulo 4).
- [X] **Sprint 5:** Treinamento, avaliação, geração de texto e checkpoints (Capítulo 5).
- [ ] **Sprint 6:** Fine-Tuning e adaptação para tarefas específicas (Capítulos 6 e 7).

## 🛠️ Tecnologias e Ambiente

* **Linguagem:** Python 3.x
* **Framework:** PyTorch
* **Controle de Versão:** Git / GitHub

## Executar a Sprint 5

Na raiz do repositório, em um ambiente Python 3.12, instale as dependências e execute os experimentos em CPU. O comando dos experimentos grava as métricas, o gráfico, as amostras de texto e os checkpoints de A, B e C.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m experimentos.sprint5_experimentos --device cpu
python -m src.evaluation --checkpoint experimentos/resultados_sprint5/checkpoints/C.pt --device cpu
python -m src.generation --checkpoint experimentos/resultados_sprint5/checkpoints/C.pt --device cpu
```

Para treinar um modelo com configuração própria, use `python -m src.training --help`. Consulte o guia técnico para o protocolo, as opções e a interpretação dos resultados. [Guia da Sprint 5](docs/sprint5_treinamento.md)
