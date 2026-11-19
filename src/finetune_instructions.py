"""Sprint 6: instruction supervised fine-tuning on {instruction,input,output} JSON.

Only response tokens contribute to cross-entropy. Prompt tokens and padding
have target -100, and never contribute to the training objective.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.generation import generate_tokens
from src.pretrained import load_pretrained_gpt2
from src.sprint6_utils import (EOS_ID, partitions, resolve_device, save_json,
                               trainable_count, trainable_state, apply_trainable_state)


def load_instruction_data(path: Path) -> list[dict[str, str]]:
    """Accept JSON array or JSONL; input is optional, output/response is required."""
    if path.suffix.lower() == ".jsonl":
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        entries = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list):
        raise ValueError("Instruction dataset must contain a list")
    parsed = []
    for row in entries:
        if not isinstance(row, dict) or not isinstance(row.get("instruction"), str):
            raise ValueError("Each example requires a string 'instruction'")
        response = row.get("output", row.get("response"))
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Each example requires a nonempty 'output' or 'response'")
        user_input = row.get("input", "")
        if not isinstance(user_input, str):
            raise ValueError("The optional 'input' must be a string")
        parsed.append({"instruction": row["instruction"], "input": user_input, "output": response})
    if len(parsed) < 7:
        raise ValueError("At least 7 examples required for train/validation/test")
    return parsed


def format_prompt(item: dict) -> str:
    prompt = "Below is an instruction. Provide a helpful response.\n\n### Instruction:\n" + item["instruction"].strip()
    if item.get("input", "").strip():
        prompt += "\n\n### Input:\n" + item["input"].strip()
    return prompt + "\n\n### Response:\n"


class InstructionDataset(Dataset):
    def __init__(self, examples: list[dict[str, str]], tokenizer, max_tokens: int = 256):
        if max_tokens < 4:
            raise ValueError("max_tokens must be >= 4")
        self.samples = []
        for example in examples:
            prompt_ids = tokenizer.encode(format_prompt(example))
            response_ids = tokenizer.encode(example["output"]) + [EOS_ID]
            # Reserve at least a response token and EOS, preferring the prompt tail.
            if len(prompt_ids) > max_tokens - 2:
                prompt_ids = prompt_ids[-(max_tokens - 2):]
            available_response_tokens = max_tokens - len(prompt_ids)
            if len(response_ids) > available_response_tokens:
                # Truncated responses must still end in EOS for stop-token learning.
                response_ids = response_ids[:available_response_tokens - 1] + [EOS_ID]
            all_ids = prompt_ids + response_ids
            inputs = torch.tensor(all_ids[:-1], dtype=torch.long)
            labels = torch.tensor(all_ids[1:], dtype=torch.long)
            # Label[j] predicts token all_ids[j+1]. Mask prompt-to-prompt targets.
            labels[:max(0, len(prompt_ids) - 1)] = -100
            if not (labels != -100).any():
                raise ValueError("Example has no response targets")
            self.samples.append((inputs, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_instructions(batch):
    max_length = max(len(inputs) for inputs, _ in batch)
    x = torch.full((len(batch), max_length), EOS_ID, dtype=torch.long)
    y = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for i, (inputs, targets) in enumerate(batch):
        x[i, :len(inputs)] = inputs
        y[i, :len(targets)] = targets
    return x, y


def configure_trainable(model, last_blocks: int = 1) -> int:
    """Freeze GPT embeddings/early blocks; train upper layers and LM head."""
    if not 0 < last_blocks <= len(model.blocks):
        raise ValueError("last_blocks must be in the GPT block range")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in list(model.blocks)[-last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.final_norm.parameters():
        parameter.requires_grad = True
    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True
    return trainable_count(model)


def instruction_loss(model, inputs, targets):
    logits = model(inputs)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)


@torch.no_grad()
def evaluate_instructions(model, loader, device):
    previous_mode = model.training
    model.eval()
    total = 0
    total_loss = 0.0
    correct = 0
    try:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            valid = targets != -100
            amount = int(valid.sum().item())
            if amount:
                total_loss += float(F.cross_entropy(logits.flatten(0, 1),
                                                    targets.flatten(), ignore_index=-100,
                                                    reduction="sum"))
                correct += int(((logits.argmax(-1) == targets) & valid).sum())
                total += amount
    finally:
        model.train(previous_mode)
    if total == 0:
        raise ValueError("Evaluation has no supervised response tokens")
    return {"response_token_loss": total_loss / total,
            "response_token_accuracy": correct / total,
            "evaluated_response_tokens": total}


@torch.no_grad()
def respond(model, tokenizer, row, device, max_new_tokens=48):
    ids = tokenizer.encode(format_prompt(row))[-model.context_length:]
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    generated = generate_tokens(model, prompt, max_new_tokens=max_new_tokens)
    result = generated[0, len(ids):].tolist()
    if EOS_ID in result:
        result = result[:result.index(EOS_ID)]
    return tokenizer.decode(result).strip()


def run(args):
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model, tokenizer, _ = load_pretrained_gpt2(args.model, device)
    if args.max_tokens > model.context_length:
        raise ValueError("max_tokens exceeds the GPT context length")
    trainable = configure_trainable(model, args.last_blocks)
    if args.predict is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["model_name"] != args.model or checkpoint["last_blocks"] != args.last_blocks:
            raise ValueError("Adapter model or last_blocks mismatch")
        apply_trainable_state(model, checkpoint["trainable_state_dict"])
        print(respond(model, tokenizer, {"instruction": args.predict, "input": ""}, device,
                      args.max_new_tokens))
        return

    samples = load_instruction_data(args.data)
    training, validation, testing = partitions(samples, seed=args.seed)
    datasets = [InstructionDataset(s, tokenizer, args.max_tokens) for s in
                (training, validation, testing)]
    loaders = [DataLoader(d, batch_size=args.batch_size, shuffle=(i == 0),
                          collate_fn=collate_instructions)
               for i, d in enumerate(datasets)]
    baseline = evaluate_instructions(model, loaders[1], device)
    demo_rows = testing[:min(3, len(testing))]
    before = [respond(model, tokenizer, row, device, args.max_new_tokens) for row in demo_rows]
    print("Samples/splits:", len(samples), [len(d) for d in datasets],
          "trainable:", trainable, "baseline:", baseline)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=0.01)
    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, labels in loaders[0]:
            optimizer.zero_grad(set_to_none=True)
            loss = instruction_loss(model, inputs.to(device), labels.to(device))
            loss.backward()
            optimizer.step()
        metrics = evaluate_instructions(model, loaders[1], device)
        print(f"Epoch {epoch} validation: {metrics}")
        history.append({"epoch": epoch, **metrics})
        if metrics["response_token_loss"] < best_loss:
            best_loss = metrics["response_token_loss"]
            best_state = trainable_state(model)
    apply_trainable_state(model, best_state)
    test_metrics = evaluate_instructions(model, loaders[2], device)
    after = [respond(model, tokenizer, row, device, args.max_new_tokens) for row in demo_rows]
    examples = [{"instruction": row["instruction"], "input": row["input"],
                 "expected": row["output"], "before": a, "after": b}
                for row, a, b in zip(demo_rows, before, after)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "instructions_adapter.pt"
    torch.save({"model_name": args.model, "last_blocks": args.last_blocks,
                "trainable_state_dict": best_state}, checkpoint_path)
    save_json(args.output_dir / "metrics_instructions.json", {
        "model": args.model, "seed": args.seed, "trainable_parameters": trainable,
        "data_split": {"train": len(training), "validation": len(validation), "test": len(testing)},
        "baseline_validation": baseline, "history": history,
        "held_out_test": test_metrics, "response_evaluation": examples,
    })
    print("Held-out response metrics:", test_metrics)
    print("Saved:", checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/instructions.json"))
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--last-blocks", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("experimentos/resultados_sprint6/instrucoes"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experimentos/resultados_sprint6/instrucoes/instructions_adapter.pt"))
    parser.add_argument("--predict", help="Run one instruction using a saved adapter")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
