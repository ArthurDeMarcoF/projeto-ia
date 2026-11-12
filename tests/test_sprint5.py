"""Testes rápidos das métricas, do treinamento e da reutilização de checkpoints."""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.checkpoint import load_checkpoint, save_checkpoint
from src.data import DEFAULT_CORPUS, split_corpus
from src.evaluation import batch_loss, evaluate_loader, perplexity
from src.generation import compare_strategies, generate_tokens
from src.gpt import GPTModel
from src.tokenization import UNK_TOKEN, create_simple_tokenizer
from src.training import run_training


def test_loss_do_proximo_token_e_perplexidade():
    model = GPTModel(7, 4, 8, 2, 1)
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 4], [5, 6, 0]])
    logits = model(inputs)
    assert logits.shape == (2, 3, 7)
    expected = -logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).mean()
    torch.testing.assert_close(batch_loss(model, inputs, targets), expected)
    assert perplexity(math.log(7)) == pytest.approx(7)


def test_avaliacao_pondera_lotes_e_restaura_modo():
    model = GPTModel(7, 4, 8, 2, 1, dropout=0.5).eval()
    inputs = torch.tensor([[1, 2], [2, 3], [5, 6]])
    targets = torch.tensor([[2, 3], [3, 4], [6, 0]])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    with torch.no_grad():
        expected = batch_loss(model, inputs, targets).item()

    modes = []
    hook = model.register_forward_pre_hook(
        lambda module, _: modes.append((module.training, torch.is_grad_enabled()))
    )
    model.train()
    assert evaluate_loader(model, loader) == pytest.approx(expected)
    assert modes == [(False, False), (False, False)]
    assert model.training
    empty_loader = DataLoader(TensorDataset(inputs[:0], targets[:0]))
    with pytest.raises(ValueError, match="não contém tokens"):
        evaluate_loader(model, empty_loader)
    assert model.training
    model.eval()
    evaluate_loader(model, loader)
    assert not model.training
    hook.remove()


def test_split_isola_vocabulario_da_validacao():
    train_text, validation_text = split_corpus("a b c d exclusivo", 0.8)
    tokenizer = create_simple_tokenizer(train_text)
    assert train_text == "a b c d"
    assert validation_text == "exclusivo"
    assert tokenizer.encode(validation_text) == [tokenizer.str_to_int[UNK_TOKEN]]


def test_treinamento_atualiza_pesos_e_reinicia_experimentos(tmp_path):
    text = DEFAULT_CORPUS.read_text(encoding="utf-8")
    path = tmp_path / "first.pt"
    first, _ = run_training(text, path, epochs=1, batch_size=4, seed=42, device="cpu")
    second, _ = run_training(text, tmp_path / "second.pt", epochs=1, batch_size=8, seed=42, device="cpu")
    assert first[0] == second[0]
    assert [row["epoch"] for row in first] == [0, 1]
    assert first[-1]["step"] > second[-1]["step"] > 0
    for row in first + second:
        assert math.isfinite(row["train_loss"])
        assert math.isfinite(row["validation_loss"])
        assert row["validation_perplexity"] == pytest.approx(math.exp(row["validation_loss"]))

    trained, _, checkpoint = load_checkpoint(path)
    torch.manual_seed(42)
    initial = GPTModel(**checkpoint["architecture"])
    assert not torch.equal(initial.lm_head.weight, trained.lm_head.weight)


def test_checkpoint_reproduz_logits_e_gera_texto(tmp_path):
    tokenizer = create_simple_tokenizer("Um modelo de linguagem aprende padrões .")
    config = dict(vocab_size=tokenizer.vocab_size, context_length=4, embedding_dim=8,
                  num_heads=2, num_layers=1, dropout=0.3)
    model = GPTModel(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ids = torch.tensor([tokenizer.encode("Um modelo de linguagem aprende")])
    inputs = ids[:, :-1]
    batch_loss(model, inputs, ids[:, 1:]).backward()
    optimizer.step()
    path = tmp_path / "model.pt"
    training = {"batch_size": 1, "learning_rate": 0.001}
    save_checkpoint(path, model, tokenizer, config, training, 1, optimizer)

    restored, restored_tokenizer, checkpoint = load_checkpoint(path)
    assert not restored.training
    assert checkpoint["epoch"] == 1
    assert checkpoint["training"] == training
    assert checkpoint["optimizer_state_dict"]["state"]
    assert restored_tokenizer.str_to_int == tokenizer.str_to_int
    with torch.no_grad():
        torch.testing.assert_close(model.eval()(inputs), restored(inputs), atol=0, rtol=0)
    prompt = "Um modelo de linguagem"
    original_text = compare_strategies(model, tokenizer, prompt, max_new_tokens=6)
    restored_text = compare_strategies(restored, restored_tokenizer, prompt, max_new_tokens=6)
    assert original_text == restored_text
    assert restored_text.count(prompt) >= 4


def test_amostragem_aplica_temperatura_e_limita_top_k_com_empate(monkeypatch):
    model = GPTModel(7, 4, 8, 2, 1)
    scores = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    model.register_forward_hook(lambda _, inputs, output: scores.expand_as(output))
    seen = []
    multinomial = torch.multinomial

    def record_probabilities(probabilities, **kwargs):
        seen.append(probabilities)
        return multinomial(probabilities, **kwargs)

    monkeypatch.setattr(torch, "multinomial", record_probabilities)
    prompt = torch.tensor([[0]])
    generate_tokens(model, prompt, 1, temperature=2, do_sample=True)
    torch.testing.assert_close(seen[0], (scores / 2).softmax(-1).unsqueeze(0))
    generate_tokens(model, prompt, 1, top_k=2, do_sample=True)
    assert seen[1].count_nonzero().item() == 2
