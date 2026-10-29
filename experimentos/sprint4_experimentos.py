"""Experimentos reproduzíveis da Sprint 4, sem treinamento.

Execute na raiz do projeto: python experimentos/sprint4_experimentos.py
Grava resultados e análise somente com valores efetivamente calculados.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation import generate_tokens
from src.gpt import GPTModel
from src.tokenization import create_simple_tokenizer


def count_parameters(model: GPTModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def average_forward_ms(model: GPTModel, ids: torch.Tensor, repeats: int = 5) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(2):
            model(ids)
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(ids)
            elapsed.append((time.perf_counter() - start) * 1000)
    return sum(elapsed) / len(elapsed)


def main():
    torch.manual_seed(123)
    # Evita que muitos threads distorçam medições de modelos pequenos na CPU.
    torch.set_num_threads(1)
    corpus = (ROOT / "data" / "texto_teste.txt").read_text(encoding="utf-8")
    tokenizer = create_simple_tokenizer(corpus)
    vocab_size = tokenizer.vocab_size
    all_ids = tokenizer.encode(corpus)
    BASE = dict(vocab_size=vocab_size, context_length=8, embedding_dim=32,
                num_heads=4, num_layers=2, ff_hidden_dim=128, dropout=0.0)
    baseline_ids = torch.tensor([all_ids[:8], all_ids[8:16]], dtype=torch.long)

    configurations = [
        ("Blocos", "1", {"num_layers": 1}),
        ("Blocos", "2", {"num_layers": 2}),
        ("Blocos", "4", {"num_layers": 4}),
        ("Heads", "1", {"num_heads": 1}),
        ("Heads", "2", {"num_heads": 2}),
        ("Heads", "4", {"num_heads": 4}),
        ("Heads", "8", {"num_heads": 8}),
        ("Embedding D", "16", {"embedding_dim": 16, "ff_hidden_dim": 64}),
        ("Embedding D", "32", {"embedding_dim": 32, "ff_hidden_dim": 128}),
        ("Embedding D", "64", {"embedding_dim": 64, "ff_hidden_dim": 256}),
        ("FeedForward F", "64", {"ff_hidden_dim": 64}),
        ("FeedForward F", "128", {"ff_hidden_dim": 128}),
        ("FeedForward F", "256", {"ff_hidden_dim": 256}),
        ("Contexto T", "4", {"context_length": 4}),
        ("Contexto T", "8", {"context_length": 8}),
        ("Contexto T", "16", {"context_length": 16}),
    ]

    rows = []
    for category, value, change in configurations:
        torch.manual_seed(123)
        config = {**BASE, **change}
        model = GPTModel(**config)
        T = config["context_length"]
        input_ids = torch.tensor([all_ids[:T], all_ids[T:2*T]], dtype=torch.long)
        logits = model(input_ids)
        params = count_parameters(model)
        ms = average_forward_ms(model, input_ids)
        rows.append((category, value, config["embedding_dim"], config["num_heads"],
                     config["num_layers"], config["ff_hidden_dim"], T,
                     str(tuple(logits.shape)), params, f"{4*params / 1024:.1f}", f"{ms:.3f}"))

    # Comparação do dropout durante treino e em avaliação, sem atualizar pesos.
    torch.manual_seed(123)
    dropout_model = GPTModel(**{**BASE, "dropout": 0.3})
    dropout_model.train()
    with torch.no_grad():
        train_a = dropout_model(baseline_ids)
        train_b = dropout_model(baseline_ids)
    train_diff = (train_a - train_b).abs().mean().item()
    dropout_model.eval()
    with torch.inference_mode():
        eval_a = dropout_model(baseline_ids)
        eval_b = dropout_model(baseline_ids)
    eval_diff = (eval_a - eval_b).abs().mean().item()

    # Temperatura e top-k sobre OS MESMOS logits (modelo fixo; não treinado).
    torch.manual_seed(123)
    model = GPTModel(**BASE).eval()
    prompt = torch.tensor([all_ids[:4]], dtype=torch.long)
    with torch.inference_mode():
        next_logits = model(prompt)[0, -1, :]
    sampling_rows = []
    for temperature, top_k in [(0.5, None), (1.0, None), (1.5, None), (1.0, 5), (1.0, 10)]:
        values = next_logits / temperature
        if top_k is not None:
            threshold = torch.topk(values, top_k).values[-1]
            values = values.masked_fill(values < threshold, float("-inf"))
        prob = torch.softmax(values, dim=-1)
        entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum().item()
        sampling_rows.append((temperature, top_k if top_k is not None else "Sem filtro",
                              f"{entropy:.5f}", f"{prob.max().item():.5f}",
                              int((prob > 0).sum().item())))

    greedy = generate_tokens(model, prompt, max_new_tokens=6)
    sampled = generate_tokens(model, prompt, max_new_tokens=6, do_sample=True,
                              top_k=5, generator=torch.Generator().manual_seed(123))
    greedy_tail = greedy[0, prompt.shape[1]:].tolist()
    sampled_tail = sampled[0, prompt.shape[1]:].tolist()

    result_text = "\n".join([
        "# Resultados — Sprint 4", "",
        "Medições locais em CPU com `torch.manual_seed(123)`, `torch.set_num_threads(1)`,",
        "2 aquecimentos e 5 execuções por configuração. Latências variam conforme hardware e carga.",
        "**Todas as redes possuem parâmetros aleatórios, não treinados.**", "",
        "## 1. Comparação arquitetural", "",
        markdown_table(["Grupo", "Valor", "D", "H", "N", "F", "T", "Logits", "Parâmetros", "Pesos FP32 (KiB)", "Forward CPU (ms)"], rows), "",
        "**B=2** em todas as entradas. D: embedding; H: heads; N: blocos; F: FF intermediária; T: contexto.",
        "Memória FP32 considera apenas parâmetros (4 bytes por elemento), não ativações, gradientes ou otimizador.", "",
        "## 2. Dropout", "",
        f"- Diferença média absoluta entre duas passagens em `train()` com dropout 0.3: `{train_diff:.6f}`.",
        f"- Diferença média absoluta entre duas passagens em `eval()`: `{eval_diff:.6f}`.", "",
        "## 3. Temperatura e top-k nos mesmos logits", "",
        markdown_table(["Temperatura", "Top-k", "Entropia (nats)", "Maior probabilidade", "Tokens possíveis"], sampling_rows), "",
        "A entropia é `-soma(p * ln(p))`. Top-k elimina logits fora das k maiores posições antes do softmax.", "",
        "## 4. Geração autoregressiva", "",
        f"- Prompt Token IDs: `{prompt[0].tolist()}`.",
        f"- Greedy — 6 IDs previstos: `{greedy_tail}`.",
        f"- Amostragem — top-k=5, temperature=1.0, seed=123: `{sampled_tail}`.",
        "- Greedy texto: `" + tokenizer.decode(greedy[0].tolist()).replace('`', '') + "`.",
        "- Amostragem texto: `" + tokenizer.decode(sampled[0].tolist()).replace('`', '') + "`.", "",
        "Essas sequências demonstram o fluxo de inferência; não representam linguagem aprendida.", "",
    ])
    (ROOT / "experimentos" / "resultados_sprint4.md").write_text(result_text, encoding="utf-8")

    def values(group):
        return [row for row in rows if row[0] == group]

    heads = values("Heads")
    blocks = values("Blocos")
    dims = values("Embedding D")
    ff = values("FeedForward F")
    context = values("Contexto T")
    analysis = "\n".join([
        "# Análise — Sprint 4", "",
        "Os valores a seguir são obtidos da execução do script, em CPU, com pesos inicializados aleatoriamente.",
        "Não houve treinamento e não é possível concluir qualidade linguística pelos textos produzidos.", "",
        "## Número de blocos", "",
        f"**Resultado:** N={blocks[0][4]} teve {blocks[0][8]} parâmetros, e N={blocks[-1][4]} teve {blocks[-1][8]}.",
        "**Causa:** cada Transformer Block adiciona atenção, LayerNorm e FeedForward treináveis.",
        "**Teoria/consequência:** maior profundidade amplia a capacidade representacional e o custo de computação, mas não garante melhor texto sem treinamento.", "",
        "## Número de heads", "",
        f"**Resultado:** H={heads[0][3]} e H={heads[-1][3]} tiveram ambos {heads[0][8]} parâmetros, com D=32.",
        "**Causa:** cada projeção continua D->D; alterar H só redistribui D em `head_dim=D/H`.",
        "**Teoria/consequência:** o número de matrizes de atenção cresce com H; tempo e memória de ativações podem mudar, apesar dos parâmetros iguais.", "",
        "## Dimensão de embedding", "",
        f"**Resultado:** D={dims[0][2]} usou {dims[0][8]} parâmetros e D={dims[-1][2]} usou {dims[-1][8]}.",
        "**Causa:** projeções e FeedForward têm matrizes cujo tamanho depende de D e F; neste ensaio F=4D.",
        "**Teoria/consequência:** ampliar D aumenta capacidade e custo computacional e de memória.", "",
        "## Dimensão FeedForward", "",
        f"**Resultado:** F={ff[0][5]} teve {ff[0][8]} parâmetros e F={ff[-1][5]} teve {ff[-1][8]}.",
        "**Causa:** as camadas D->F->D possuem pesos proporcionais a F.",
        "**Teoria/consequência:** F amplia transformações por token; não altera dimensões de entrada/saída do bloco.", "",
        "## Tamanho de contexto", "",
        f"**Resultado:** T={context[0][6]} gerou logits {context[0][7]} e T={context[-1][6]} gerou {context[-1][7]}.",
        "**Causa:** atenção calcula scores [B,H,T,T] e a embedding posicional possui T vetores.",
        "**Teoria/consequência:** maior T permite mais posições no contexto, mas atenção usa memória quadrática em T; o custo de forward medido é dependente da CPU.", "",
        "## Dropout e inferência", "",
        f"**Resultado:** diferença média em train={train_diff:.6f} versus eval={eval_diff:.6f}.",
        "**Causa:** dropout sorteia posições apenas em train(), enquanto eval() desativa a regularização.",
        "**Teoria/consequência:** a inferência deve operar com dropout desligado para evitar aleatoriedade não desejada.", "",
        "## Temperatura, top-k e seleção", "",
        f"**Resultado:** a temperatura 0.5 teve entropia {sampling_rows[0][2]}, e 1.5 teve {sampling_rows[2][2]}; top-k=5 deixou {sampling_rows[3][4]} candidatos.",
        "**Causa:** dividir logits por temperatura maior achata softmax; top-k filtra candidatos.",
        "**Teoria/consequência:** a amostragem muda a seleção do próximo token, enquanto greedy escolhe o de maior probabilidade.",
        "A geração é autoregressiva: o ID gerado entra no contexto da previsão seguinte.", "",
        "## Limitações", "",
        "- Pesos não treinados: o conteúdo textual não tem valor semântico demonstrável.",
        "- Medições CPU locais: não são benchmarks comparáveis a outras máquinas.",
        "- A Sprint 5 introduzirá função de perda, backpropagation e treinamento dos pesos.", "",
    ])
    (ROOT / "experimentos" / "analise_sprint4.md").write_text(analysis, encoding="utf-8")
    print("Arquivos gerados:")
    print(" - experimentos/resultados_sprint4.md")
    print(" - experimentos/analise_sprint4.md")
    print("Comparações arquiteturais:", len(rows))
    print("Parâmetros do modelo base:", count_parameters(model))


if __name__ == "__main__":
    main()
