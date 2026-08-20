<<<<<<< HEAD

# Glossário Técnico — Projeto LLM

**Componente curricular:** Inteligência Artificial e Sistemas Inteligentes
**Curso:** Engenharia da Computação — UNOESC
**Professor:** Kleyton Hoffmann
**Acadêmicos:** Arthur de Marco Faggion e Lucas Zamoner Locatelli
**Ano:** 2026

Este glossário reúne, de forma cumulativa, os principais conceitos estudados durante o desenvolvimento do Projeto LLM.

---

## Capítulo 1 — Understanding Large Language Models

=======
# Glossário Técnico — Projeto LLM

## Capítulo 1 — Understanding Large Language Models

**Componente curricular:** Inteligência Artificial e Sistemas Inteligentes  
**Curso:** Engenharia da Computação — UNOESC  
**Professor:** Kleyton Hoffmann  
**Acadêmicos:** Arthur de Marco e Lucas Zamoner  
**Ano:** 2026

Conceitos fundamentais sobre Grandes Modelos de Linguagem.

---

>>>>>>> 815cb78e9f5288dbd05b1149dcb7434949d60c8d
### Ajuste Fino de Instruções (Instruct Tuning)

Processo de treinamento adicional em que um modelo pré-treinado é refinado em conjuntos de dados formatados especificamente para aprender a seguir as instruções (prompts) fornecidas pelo usuário de forma mais eficaz.

### Análise de Ativação de Neurônios (Neuron Activation Analysis)

Abordagem de inteligência artificial explicável que busca desvendar o padrão de ativação e o propósito de neurônios artificiais individuais em um LLM para compreender quais conceitos disparam certas áreas da rede.

### Análise de Circuitos (Circuit Analysis)

Técnica ambiciosa de interpretabilidade mecanicista que tenta decifrar como grupos de neurônios e parâmetros (features) trabalham em conjunto como circuitos para implementar algoritmos, buscando explicar capacidades cognitivas como o raciocínio simbólico.

### Antropomorfismo de IA (AI Anthropomorphism)

A prática de atribuir características, intenções e capacidades cognitivas tipicamente humanas, como “entender”, “raciocinar” ou “acreditar”, a sistemas de inteligência artificial.

### Aprendizado por Reforço com Feedback Humano (RLHF — Reinforcement Learning from Human Feedback)

Técnica de ajuste onde avaliadores humanos classificam as respostas do modelo segundo a qualidade ou adequação. O LLM utiliza essas classificações para otimizar sua política e gerar respostas preferenciais e alinhadas aos valores humanos.

### Arquitetura Encoder-Decoder

Arquitetura base de modelos como BERT e T5 que se sobressai em tarefas que exigem um forte alinhamento entre a sequência de entrada (codificada) e a sequência de saída (decodificada), como a tradução de máquina.

### Arquitetura Transformer

A infraestrutura de rede neural que é a base dos LLMs modernos. Ao contrário de modelos sequenciais, o Transformer processa toda a entrada simultaneamente através de blocos de autoatenção, resolvendo o problema de dependências longas em textos e permitindo grande paralelização de treinamento.

### Embeddings de Tokens (Token Embeddings)

Representações de vetores contínuos mapeadas a partir de tokens. Estes vetores numéricos capturam as possíveis características semânticas das partes do texto dentro do espaço vetorial do modelo e são aprimorados ao longo do treinamento.

### Grandes Modelos de Linguagem (Large Language Models — LLMs)

Sistemas avançados de inteligência artificial com bilhões de parâmetros (pesos), fundamentados na arquitetura Transformer, que deixaram de ser especialistas de uma única tarefa para operarem como modelos generalistas solucionadores de problemas textuais complexos.

### Inteligência Artificial Explicável (Explainable AI / Mechanistic Interpretability)

Área de pesquisa que tem como objetivo entender os caminhos lógicos ou matemáticos pelos quais uma rede neural, considerada uma “caixa-preta”, toma decisões, buscando mapear níveis computacionais e algorítmicos dos LLMs.

### Mecanismo de Atenção (Attention Mechanism)

Um componente central dos Transformers que permite ao modelo integrar e pesar a importância da informação contextual de todo o texto anterior ao definir a representação atual de cada token.

### Modelagem de Linguagem Não Supervisionada (Unsupervised Language Modeling)

O objetivo clássico e simples de treinamento que instrui o modelo a prever iterativamente qual será o próximo token em uma sequência com base no contexto prévio fornecido, alavancando vastas quantidades de dados não rotulados da internet.

### Modelos de Linguagem para Raciocínio (Reasoning LLMs)

Modelos especializados treinados para gerar internamente uma sequência invisível ou visível de tokens formando um raciocínio lógico em passos (chain-of-thought) antes de entregar a resposta final ao usuário.

### Modelos Multimodais (Multimodal Models)

LLMs avançados capazes de processar, entender e produzir não apenas dados baseados em texto, mas diferentes meios de entrada e saída, como imagens, sinais de áudio ou vídeo, através de processos de tokenização específicos.

### Neurônios Polisemânticos (Polysemantic Neurons)

Neurônios na arquitetura profunda do modelo que são ativados simultaneamente por conceitos múltiplos e muitas vezes não relacionados, dificultando bastante a identificação de sua utilidade exata.

### Processamento de Linguagem Natural (NLP — Natural Language Processing)

Subcampo de inteligência artificial com foco no ensino e na construção de sistemas computacionais capazes de derivar sentido ou interpretar a linguagem humana.

### Raciocínio Simbólico (Symbolic Reasoning)

A capacidade cognitiva fundamental de representar ideias, entidades ou eventos como símbolos e manipulá-los em múltiplas etapas. Estudos apontam que os LLMs manifestam comportamentos que simulam este raciocínio lógico-simbólico nas camadas mais profundas de processamento.

### Redes Neurais Recorrentes (RNNs — Recurrent Neural Networks)

Modelos tradicionais de processamento sequencial de texto. Elas rastreiam o contexto em um “vetor de estado oculto” atualizado palavra por palavra, mas sofrem de perdas de informação ao lidar com blocos longos de texto.

### Sondas Lineares (Linear Probes)

Método analítico onde um modelo de classificação linear simples é acoplado sobre a ativação em camadas intermediárias do modelo de linguagem. Isso ajuda a desvendar se, e quando, o modelo deduziu corretamente se uma informação inserida era verdadeira ou falsa.

### Teoria da Mente (Theory of Mind — ToM)

A capacidade presente em humanos de inferir as crenças, emoções e pontos de vista de outras pessoas e antecipar comportamentos baseados nessas inferências. Existe um forte debate científico investigando se os LLMs demonstram essa mesma capacidade frente a cenários de ambiguidade.

### Tokens

Frações de sequências de caracteres obtidas após a quebra de um documento pelo Tokenizer. Podem simbolizar uma palavra inteira, raízes, sufixos e até pontuações, desprovidas de um significado por si sós até o processamento.

---

<<<<<<< HEAD
> Este glossário é cumulativo. Os conceitos dos próximos capítulos serão adicionados a este mesmo documento ao longo das Sprints do Projeto LLM.
=======
> Este glossário é cumulativo e será atualizado ao longo das próximas Sprints do Projeto LLM.
>>>>>>> 815cb78e9f5288dbd05b1149dcb7434949d60c8d
