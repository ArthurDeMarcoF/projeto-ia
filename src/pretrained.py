"""Map GPT-2 pretrained weights to the GPTModel built in earlier sprints.

Requires transformers only for downloading the external reference checkpoint.
The model used for inference is our own src.gpt.GPTModel, not an HF wrapper.
"""
from __future__ import annotations

import argparse
from typing import Any

import torch

from src.gpt import GPTModel
from src.tokenization import get_gpt2_tokenizer
from src.generation import generate_text


def architecture_from_hf(config: Any) -> dict:
    """Require a GPT-2 configuration compatible with our implementation."""
    if config.model_type != "gpt2":
        raise ValueError("Only GPT-2 architecture checkpoints are supported.")
    if config.activation_function != "gelu_new":
        raise ValueError("This GPTModel implements GELU tanh (gelu_new) only.")
    if not getattr(config, "scale_attn_weights", True):
        raise ValueError("Unscaled attention is not implemented.")
    if getattr(config, "scale_attn_by_inverse_layer_idx", False):
        raise ValueError("Layer-dependent attention scaling is not implemented.")
    if getattr(config, "reorder_and_upcast_attn", False):
        raise ValueError("Upcast attention is not implemented.")
    if float(config.layer_norm_epsilon) != 1e-5:
        raise ValueError("Unsupported layer normalization epsilon.")
    return dict(
        vocab_size=int(config.vocab_size),
        context_length=int(config.n_positions),
        embedding_dim=int(config.n_embd),
        num_heads=int(config.n_head),
        num_layers=int(config.n_layer),
        ff_hidden_dim=int(config.n_inner or 4 * config.n_embd),
        dropout=0.0,  # Inference mode; train scripts can configure dropout if needed.
        qkv_bias=True,
    )


def _copy(target: torch.Tensor, value: torch.Tensor, label: str) -> None:
    if target.shape != value.shape:
        raise ValueError(f"Incompatible shape for {label}: {tuple(target.shape)} vs {tuple(value.shape)}")
    target.copy_(value.detach().to(device=target.device, dtype=target.dtype))


def copy_hf_gpt2_weights(model: GPTModel, hf_model: Any) -> int:
    """Copy every GPT-2 tensor; transpose HF Conv1D weights to nn.Linear.

    HF stores each Q/K/V matrix concatenated along its second dimension.
    HF Conv1D uses [input, output], whereas torch.nn.Linear uses [output, input].
    """
    ref = hf_model.transformer
    if len(model.blocks) != len(ref.h):
        raise ValueError("The number of Transformer blocks differs.")

    copies = 0

    def put(target, source, name):
        nonlocal copies
        _copy(target, source, name)
        copies += 1

    with torch.no_grad():
        put(model.embeddings.token_embedding.weight, ref.wte.weight, "token embedding")
        put(model.embeddings.position_embedding.weight, ref.wpe.weight, "position embedding")
        for i, (block, original) in enumerate(zip(model.blocks, ref.h)):
            prefix = f"block {i}"
            put(block.norm1.scale, original.ln_1.weight, prefix + " norm1 scale")
            put(block.norm1.shift, original.ln_1.bias, prefix + " norm1 shift")
            put(block.norm2.scale, original.ln_2.weight, prefix + " norm2 scale")
            put(block.norm2.shift, original.ln_2.bias, prefix + " norm2 shift")

            attention = block.attention
            weights = original.attn.c_attn.weight.chunk(3, dim=1)
            biases = original.attn.c_attn.bias.chunk(3, dim=0)
            for letter, name, w, b in zip(
                "qkv", ("query_projection", "key_projection", "value_projection"), weights, biases
            ):
                projection = getattr(attention, name)
                put(projection.weight, w.T, prefix + f" {letter} weight")
                if projection.bias is None:
                    raise ValueError("GPT-2 requires qkv_bias=True")
                put(projection.bias, b, prefix + f" {letter} bias")
            put(attention.out_projection.weight, original.attn.c_proj.weight.T, prefix + " attn out weight")
            put(attention.out_projection.bias, original.attn.c_proj.bias, prefix + " attn out bias")

            linear1, linear2 = block.feed_forward.layers[0], block.feed_forward.layers[2]
            put(linear1.weight, original.mlp.c_fc.weight.T, prefix + " mlp fc weight")
            put(linear1.bias, original.mlp.c_fc.bias, prefix + " mlp fc bias")
            put(linear2.weight, original.mlp.c_proj.weight.T, prefix + " mlp proj weight")
            put(linear2.bias, original.mlp.c_proj.bias, prefix + " mlp proj bias")
        put(model.final_norm.scale, ref.ln_f.weight, "final norm scale")
        put(model.final_norm.shift, ref.ln_f.bias, "final norm shift")
        # GPT-2 ties output projection to token embeddings; our model stores a copy.
        put(model.lm_head.weight, hf_model.lm_head.weight, "language modeling head")
    return copies


def load_pretrained_gpt2(
    model_name: str = "gpt2", device: str | torch.device = "cpu"
) -> tuple[GPTModel, Any, dict]:
    """Download GPT-2 and load its parameters into the educational GPTModel."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Install the transformers package to load GPT-2 weights.") from exc

    external = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False).eval()
    architecture = architecture_from_hf(external.config)
    # GPT-2 encoding is required. A local token vocabulary cannot be substituted.
    if architecture["vocab_size"] != 50257:
        raise ValueError("The pretrained GPT-2 checkpoint must use its 50257-token vocabulary.")
    model = GPTModel(**architecture)
    copied = copy_hf_gpt2_weights(model, external)
    del external
    tokenizer = get_gpt2_tokenizer()
    return model.to(device).eval(), tokenizer, {"model_name": model_name, "architecture": architecture, "tensors_copied": copied}


def main() -> None:
    parser = argparse.ArgumentParser(description="Load official pretrained GPT-2 weights in our GPTModel.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="Artificial intelligence is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--compare-random", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    model, tokenizer, metadata = load_pretrained_gpt2(args.model, args.device)
    print("Imported:", metadata)
    print("PRETRAINED:", generate_text(model, tokenizer, args.prompt, args.max_new_tokens))
    if args.compare_random:
        # Reuse exactly the same architecture and input encoding for a fair comparison.
        torch.manual_seed(123)
        random_model = GPTModel(**metadata["architecture"]).to(args.device).eval()
        print("RANDOM:", generate_text(random_model, tokenizer, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
