
# Glossário Técnico — Projeto LLM

**Componente curricular:** Inteligência Artificial e Sistemas Inteligentes
**Curso:** Engenharia da Computação — UNOESC
**Professor:** Kleyton Hoffmann
**Acadêmicos:** Arthur de Marco Faggion e Lucas Zamoner Locatelli
**Ano:** 2026

Este glossário reúne, de forma cumulativa, os principais conceitos estudados durante o desenvolvimento do Projeto LLM.

---

## Capítulo 1 — Understanding Large Language Models

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

## Capítulo 2 — Working with Text Data

### Amostragem com Janela Deslizante (Sliding Window Approach)

Técnica de amostragem e divisão de dados que utiliza um bloco de tamanho fixo (o contexto) para ler um texto iterativamente, movendo-se passo a passo através de um incremento (stride). Sua função principal no modelo de linguagem é criar os milhares de pares de entrada e alvo (input-target) necessários durante o treinamento para a tarefa de adivinhar a próxima palavra. Este método é aplicado logo após a etapa de tokenização, definindo os blocos exatos de Token IDs que a rede processará simultaneamente. Como exemplo, na frase "Eu gosto muito de estudar" usando uma janela de tamanho 3 e incremento de 1: a primeira iteração tem como entrada "Eu gosto muito" e predição "de"; na segunda iteração, a janela desliza, tendo como entrada "gosto muito de" e predição "estudar".

### Codificação em Pares de Bytes (Byte Pair Encoding — BPE)

Método avançado de tokenização que agrupa iterativamente os pares de caracteres ou subpalavras mais frequentes no texto para formar novos tokens. Sua função no modelo de linguagem é permitir o processamento de textos com palavras fora do vocabulário inicial, dividindo palavras desconhecidas em unidades menores já mapeadas. O BPE representa uma evolução da etapa de tokenização e é fundamental na construção do vocabulário do modelo, mitigando a dependência de tokens especiais. Por exemplo, se a palavra "inexplorado" for desconhecida, o BPE pode quebrá-la nos sub-tokens conhecidos "in", "explora" e "do".

### Embeddings Posicionais (Positional Embeddings)

Vetores numéricos associados a posições sequenciais que são adicionados aos embeddings dos tokens originais para codificar a posição de cada um na entrada. A função desse mecanismo no modelo de linguagem é fornecer a noção de ordem e sequência temporal das palavras, superando a limitação da arquitetura Transformer, que processa as entradas de forma paralela e não possui um sentido inerente de ordem. Esse conceito trabalha de forma aditiva com as representações contínuas convencionais, sendo que no GPT utilizam-se representações absolutas somadas ponto a ponto. Como exemplo, na frase "A maçã caiu", a palavra "caiu" recebe o embedding posicional de número 3, diferenciando-a estruturalmente de uma ocorrência da mesma palavra na posição 1.

### Identificadores de Token (Token IDs)

A representação numérica em formato de número inteiro de um token específico. A sua função no modelo de linguagem é atuar como um formato numérico intermediário entre os fragmentos de texto (strings) e os complexos vetores multidimensionais. Os Token IDs são a entrega final do processo de tokenização a partir do mapeamento exigido pelo vocabulário, servindo como entrada imediata para a busca nas matrizes de embeddings. Um exemplo prático ocorre quando a sequência processada ["Olá", ",", "tudo", "bem"] se transforma em uma matriz matemática pura, como [15496, 11, 2309, 142].

### Representações Vetoriais Contínuas (Embeddings / Token Embeddings)

Conversão de dados discretos, como palavras ou tokens, em um formato vetorial numérico composto por valores contínuos em um espaço multidimensional. No modelo de linguagem, sua função é traduzir o texto para um formato matemático que as redes neurais conseguem processar, posicionando palavras com significados semelhantes de forma geometricamente próxima. Os embeddings recebem os Token IDs gerados e são somados aos embeddings posicionais antes de serem processados pelas camadas profundas da rede neural. Um exemplo computacional prático é a conversão do ID da palavra "gato" em um vetor denso com 768 dimensões, como [0.12, -0.45, 0.89, ...].

### Tokenização (Tokenization)

A conversão prévia do texto bruto em unidades ou fragmentos analíticos isolados, denominados tokens. A função desse processo no modelo de linguagem é normalizar e fragmentar a entrada textual extensa para que o sistema consiga "ler" e agrupar partes da linguagem em blocos rastreáveis e limitados. A tokenização é o primeiro passo do fluxo de dados para a LLM, gerando as subpalavras para que o algoritmo de BPE crie o vocabulário final. Um exemplo conceitual simples é a frase de entrada "A IA evoluiu." sendo dividida na lista de fragmentos ['A', 'IA', 'evoluiu', '.'].

### Tokens Especiais (Special Tokens)

Elementos adicionados artificialmente ao vocabulário para sinalizar formatações, limites e contextos, em vez de representarem palavras normais do idioma. No contexto do modelo de linguagem, eles auxiliam o LLM na compreensão de limites lógicos de documentos, indicando eventos como fim de texto, instruções em branco ou tokens de preenchimento. Eles ficam embutidos permanentemente no vocabulário; contudo, graças ao BPE, tokens como o de palavra desconhecida raramente precisam ser acionados. Um exemplo computacional clássico é o token <|endoftext|> inserido automaticamente para separar dois artigos distintos da Wikipedia durante o treinamento do GPT.

### Vocabulário (Vocabulary)

Dicionário completo e de tamanho fixo gerado pelo tokenizador para indexar e mapear todos os tokens únicos para números inteiros exclusivos. A função do vocabulário no modelo de linguagem é estabelecer o escopo geral de tudo o que a rede consegue compreender, reconhecer e prever na sua camada final de saída. Ele é construído diretamente pelo modelo de BPE e pelo processo de tokenização, definindo o limite de tamanho que pautará as matrizes de embeddings. Computacionalmente, o vocabulário do GPT-2, por exemplo, comporta 50.257 elementos, o que significa que o seu dicionário possui 50.257 chaves associando números inteiros a representações de texto.

## Capítulo 3 — Coding Attention Mechanisms

### Mecanismo de Atenção (Attention Mechanism)

Operação que combina informações de uma sequência atribuindo pesos diferentes aos vetores disponíveis. Para cada token consultando, os pesos determinam quanto cada token observado contribui para seu novo vetor de contexto. A atenção conecta diretamente posições distantes sem exigir o processamento recorrente token a token.

### Autoatenção (Self-Attention)

Caso de atenção em que queries, keys e values são derivados da mesma entrada `X`. Sua função é contextualizar cada token usando os demais tokens da própria sequência. Para uma entrada `[B,T,D]`, a autoatenção produz uma representação contextual para cada uma das `T` posições.

### Consulta (Query — Q)

Projeção aprendível que representa aquilo que uma posição procura nas outras posições. É calculada por `Q = XW_q`. Cada query é comparada com todas as keys para formar uma linha da matriz de scores.

### Chave (Key — K)

Projeção aprendível que representa como cada posição pode ser encontrada por uma query. É calculada por `K = XW_k`. A transposição de K permite comparar cada query com todas as keys por meio de `QKᵀ`.

### Valor (Value — V)

Projeção aprendível que contém a informação efetivamente combinada após o cálculo dos pesos. É calculada por `V = XW_v`. Enquanto Q e K determinam os pesos, V determina o conteúdo do vetor de contexto.

### Score de Atenção (Attention Score)

Valor de compatibilidade obtido pelo produto escalar entre uma query e uma key. Antes do softmax, os scores podem assumir qualquer valor real. A matriz `QKᵀ` possui formato `[B,T,T]`, pois cada uma das `T` queries é comparada com cada uma das `T` keys.

### Peso de Atenção (Attention Weight)

Score transformado pelo softmax em um valor não negativo e normalizado. Cada linha da matriz de pesos soma 1 e descreve a distribuição das contribuições das keys para uma query. Em atenção causal, pesos de posições futuras são zero.

### Vetor de Contexto (Context Vector)

Soma ponderada dos values calculada por `attention_weights @ V`. O vetor resultante reúne as informações acessíveis a uma query segundo seus pesos. Para uma head, o resultado possui formato `[B,T,D_out]`.

### Produto Escalar (Dot Product)

Soma dos produtos entre componentes correspondentes de dois vetores. Na atenção, mede a compatibilidade entre Q e K: valores maiores levam a maior peso relativo após o softmax. Computacionalmente, todos os pares são calculados em paralelo por multiplicação matricial.

### Atenção por Produto Escalar Escalonado (Scaled Dot-Product Attention)

Forma de atenção definida por `softmax(QKᵀ / sqrt(d_k))V`. A divisão por `sqrt(d_k)` controla a magnitude dos scores quando a dimensão das keys cresce, reduzindo a saturação excessiva do softmax e favorecendo gradientes úteis.

### Softmax

Função que converte uma linha de scores em uma distribuição normalizada: aplica a exponencial e divide cada valor pela soma da linha. Na atenção, sua função é produzir pesos não negativos que somam 1. Diferenças muito grandes entre scores geram distribuições mais concentradas.

### Atenção Causal (Causal Attention)

Autoatenção adequada à modelagem autoregressiva, na qual a posição `i` pode consultar somente posições `j <= i`. Sua função no GPT é impedir acesso à informação futura usada como alvo de previsão.

### Máscara Causal (Causal Mask)

Matriz triangular que identifica conexões permitidas e proibidas. Antes do softmax, scores com `j > i` recebem `-inf`; depois da normalização, esses pesos tornam-se zero. Como a máscara não é treinável, pode ser armazenada como buffer do módulo.

### Autoatenção Mascarada (Masked Self-Attention)

Autoatenção na qual uma máscara restringe quais pares de posições podem interagir. No GPT, a máscara é causal, mas outras arquiteturas podem empregar máscaras com finalidades distintas, como ignorar posições de preenchimento.

### Atenção Multi-Head (Multi-Head Attention)

Mecanismo que divide as projeções Q, K e V em várias heads e calcula atenção em paralelo. Após o cálculo independente, os contextos são concatenados e projetados para a dimensão de saída. Durante o treinamento, heads diferentes podem aprender relações em subespaços distintos.

### Head de Atenção (Attention Head)

Uma unidade de atenção com suas próprias fatias de Q, K e V. Seus pesos possuem formato `[B,T,T]`. Na forma vetorizada, as heads aparecem como o eixo H em `[B,H,T,T]`.

### Dimensão da Head (Head Dimension)

Quantidade de componentes processados por head, calculada por `head_dim = d_out / num_heads`. Por isso, `d_out` precisa ser divisível por `num_heads`. Exemplo: `d_out=32` e 4 heads resultam em `head_dim=8`.

### Dropout

Regularização que zera aleatoriamente parte dos pesos durante o treinamento e reescala os valores restantes. Na atenção, reduz a dependência excessiva de conexões específicas. Durante avaliação, o dropout é desativado.

### Projeção de Query (Query Projection)

Camada linear aprendível `W_q` que transforma a entrada em queries. Sua saída mantém os eixos de lote e sequência e substitui a última dimensão: `[B,T,D_in] → [B,T,D_out]`.

### Projeção de Key (Key Projection)

Camada linear aprendível `W_k` que transforma a entrada em keys compatíveis com as queries. Q e K precisam ter a mesma última dimensão para que o produto `QKᵀ` seja definido.

### Projeção de Value (Value Projection)

Camada linear aprendível `W_v` que transforma a entrada nos conteúdos combinados pelos pesos. Depois de `attention_weights @ V`, sua última dimensão determina a dimensão dos vetores de contexto antes da projeção final.
