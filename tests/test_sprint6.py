"""Unit checks for Sprint 6 without downloading pretrained weights."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.gpt import GPTModel
from src.pretrained import architecture_from_hf, copy_hf_gpt2_weights
from src.finetune_classification import (GPTClassifier, MessageDataset, collate_messages,
                                         configure_trainable as configure_classifier,
                                         evaluate_classifier, read_messages,
                                         stratified_partitions)
from src.finetune_instructions import (InstructionDataset, collate_instructions,
                                       instruction_loss, configure_trainable as configure_instruction,
                                       format_prompt)
from src.sprint6_utils import EOS_ID, apply_trainable_state, trainable_state


class DemoTokenizer:
    def encode(self, text):
        return [max(0, ord(ch) % 300) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def toy_gpt():
    return GPTModel(vocab_size=50257, context_length=32,
                    embedding_dim=8, num_heads=2, num_layers=2,
                    ff_hidden_dim=32, qkv_bias=True)


def test_architecture_checks_gpt2_config():
    config = SimpleNamespace(model_type="gpt2", activation_function="gelu_new",
                             scale_attn_weights=True, scale_attn_by_inverse_layer_idx=False,
                             reorder_and_upcast_attn=False, layer_norm_epsilon=1e-5,
                             vocab_size=50257, n_positions=32, n_embd=8, n_head=2,
                             n_layer=2, n_inner=None)
    arch = architecture_from_hf(config)
    assert arch["qkv_bias"] and arch["ff_hidden_dim"] == 32
    config.model_type = "bert"
    with pytest.raises(ValueError):
        architecture_from_hf(config)


def test_pretrained_copy_maps_qkv_and_conv1d_weights():
    torch.manual_seed(7)
    model = toy_gpt()
    d = 8

    def conv(input_dim, output_dim):
        return SimpleNamespace(weight=torch.randn(input_dim, output_dim),
                               bias=torch.randn(output_dim))

    def norm():
        return SimpleNamespace(weight=torch.randn(d), bias=torch.randn(d))

    def layer():
        return SimpleNamespace(ln_1=norm(), ln_2=norm(),
                               attn=SimpleNamespace(c_attn=conv(d, 3*d), c_proj=conv(d, d)),
                               mlp=SimpleNamespace(c_fc=conv(d, 4*d), c_proj=conv(4*d, d)))

    original = SimpleNamespace(transformer=SimpleNamespace(
        wte=SimpleNamespace(weight=torch.randn(50257, d)),
        wpe=SimpleNamespace(weight=torch.randn(32, d)),
        h=[layer(), layer()], ln_f=norm()),
        lm_head=SimpleNamespace(weight=torch.randn(50257, d)))
    copies = copy_hf_gpt2_weights(model, original)
    assert copies == 37  # 16 tensors/block, 5 outside the blocks
    q = original.transformer.h[0].attn.c_attn.weight[:, :d]
    torch.testing.assert_close(model.blocks[0].attention.query_projection.weight, q.T)
    torch.testing.assert_close(model.blocks[1].feed_forward.layers[2].weight,
                               original.transformer.h[1].mlp.c_proj.weight.T)
    torch.testing.assert_close(model.lm_head.weight, original.lm_head.weight)


def test_csv_tsv_split_without_leakage(tmp_path):
    rows = ([f"ham\tregular message {i}" for i in range(20)] +
            [f"spam\twinning prize {i}" for i in range(20)])
    path = tmp_path / "SMSSpamCollection"
    path.write_text("\n".join(rows), encoding="utf-8")
    training, validation, testing = stratified_partitions(read_messages(path))
    assert len(training) + len(validation) + len(testing) == 40
    assert set(training).isdisjoint(validation)
    assert set(training).isdisjoint(testing)
    assert set(validation).isdisjoint(testing)
    assert {label for _, label in validation} == {0, 1}


def test_classifier_padding_and_frozen_layers():
    model = GPTClassifier(toy_gpt())
    configure_classifier(model, last_blocks=1)
    assert not any(p.requires_grad for p in model.backbone.blocks[0].parameters())
    assert all(p.requires_grad for p in model.backbone.blocks[-1].parameters())
    dataset = MessageDataset([("short", 0), ("a very long message", 1)], DemoTokenizer(), 30)
    inputs, lengths, labels = collate_messages([dataset[0], dataset[1]])
    assert inputs.shape[1] == max(lengths)
    assert inputs[0, lengths[0]:].eq(EOS_ID).all()
    result = evaluate_classifier(model, [(inputs, lengths, labels)], "cpu")
    assert result["samples"] == 2
    assert result["accuracy"] in (0, .5, 1)


def test_instruction_masks_prompt_but_trains_response():
    row = {"instruction": "Say hello", "input": "", "output": "HELLO"}
    assert "### Response:" in format_prompt(row)
    dataset = InstructionDataset([row], DemoTokenizer(), max_tokens=30)
    x, y = collate_instructions([dataset[0]])
    assert (y == -100).any() and (y != -100).any()
    assert int(y[0, -1]) != -100
    model = toy_gpt()
    configure_instruction(model, last_blocks=1)
    assert not any(p.requires_grad for p in model.blocks[0].parameters())
    before = trainable_state(model)
    loss = instruction_loss(model, x, y)
    loss.backward()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
    optimizer.step()
    assert any(not torch.equal(before[k], after) for k, after in trainable_state(model).items())
    apply_trainable_state(model, before)
    for k, value in trainable_state(model).items():
        torch.testing.assert_close(value, before[k])


def test_against_reference_hf_gpt2_logits_if_installed():
    """Strong equivalence check: custom GPT forward versus Hugging Face GPT-2."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(
        vocab_size=101, n_positions=16, n_ctx=16, n_embd=16, n_layer=2,
        n_head=4, attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0,
        activation_function="gelu_new",
    )
    ref = transformers.GPT2LMHeadModel(cfg).eval()
    own = GPTModel(**architecture_from_hf(cfg)).eval()
    copy_hf_gpt2_weights(own, ref)
    ids = torch.tensor([[1, 2, 8, 4], [10, 55, 3, 4]])
    with torch.no_grad():
        expected = ref(ids).logits
        actual = own(ids)
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)


def test_instruction_truncated_response_keeps_eos():
    row = {"instruction": "Example", "input": "", "output": "A very long response " * 100}
    dataset = InstructionDataset([row], DemoTokenizer(), max_tokens=30)
    _, labels = dataset[0]
    assert labels[-1].item() == EOS_ID
    assert (labels != -100).any()
