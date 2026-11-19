"""Sprint 6: supervised GPT-2 spam/ham fine-tuning and held-out evaluation.

Accepted dataset formats:
  - UCI SMSSpamCollection: TAB-separated, no header, ham/spam + text.
  - CSV or TSV with label,text (or Label,Text / v1,v2) columns.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.pretrained import load_pretrained_gpt2
from src.sprint6_utils import (EOS_ID, partitions, resolve_device, save_json,
                               trainable_count, trainable_state, apply_trainable_state)

LABELS = {"ham": 0, "spam": 1, "0": 0, "1": 1, "nao spam": 0,
          "not spam": 0, "nao_spam": 0}


def read_messages(path: Path) -> list[tuple[str, int]]:
    """Read rows and normalize labels without importing heavyweight pandas."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    if not rows:
        raise ValueError("Empty classification dataset")
    header = [item.strip().lower() for item in rows[0]]
    names = ({"label", "class", "v1"}, {"text", "message", "sms", "v2"})
    first = next((i for i, name in enumerate(header) if name in names[0]), None)
    second = next((i for i, name in enumerate(header) if name in names[1]), None)
    has_header = first is not None and second is not None
    first = first if has_header else 0
    second = second if has_header else 1
    result = []
    for row in rows[1:] if has_header else rows:
        if len(row) <= max(first, second):
            continue
        label = row[first].strip().lower()
        message = " ".join(row[second].split())
        if not message:
            continue
        if label not in LABELS:
            raise ValueError(f"Unexpected label: {label!r}; expected ham/spam or 0/1")
        result.append((message, LABELS[label]))
    # Remove duplicated message texts before splitting to avoid leakage.
    deduplicated = {}
    for message, label in result:
        key = message.casefold()
        if key in deduplicated and deduplicated[key][1] != label:
            raise ValueError("Conflicting labels assigned to the same message")
        deduplicated[key] = (message, label)
    result = list(deduplicated.values())
    counts = Counter(label for _, label in result)
    if not all(counts.get(i, 0) >= 7 for i in (0, 1)):
        raise ValueError("At least 7 messages from each class are needed")
    return result


def stratified_partitions(items: list[tuple[str, int]], seed: int = 123):
    """Split EACH class before fine-tuning; never evaluate on trained messages."""
    groups = {label: [item for item in items if item[1] == label] for label in (0, 1)}
    parts = [partitions(groups[label], seed=seed + label) for label in (0, 1)]
    result = []
    for split_index in range(3):
        split = parts[0][split_index] + parts[1][split_index]
        random.Random(seed + 10 + split_index).shuffle(split)
        result.append(split)
    return tuple(result)


class MessageDataset(Dataset):
    def __init__(self, examples: list[tuple[str, int]], tokenizer, max_tokens: int = 128):
        if max_tokens < 2:
            raise ValueError("max_tokens must be >= 2")
        self.entries = []
        for message, label in examples:
            token_ids = tokenizer.encode(message)[:max_tokens]
            self.entries.append((token_ids or [EOS_ID], label))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]


def collate_messages(batch: list[tuple[list[int], int]]):
    lengths = torch.tensor([len(ids) for ids, _ in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    inputs = torch.full((len(batch), max_len), EOS_ID, dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        inputs[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    targets = torch.tensor([label for _, label in batch], dtype=torch.long)
    return inputs, lengths, targets


class GPTClassifier(nn.Module):
    """Use the last REAL token embedding as features for a 2-class head."""
    def __init__(self, backbone, classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embedding_dim, classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        x = self.backbone.embedding_dropout(self.backbone.embeddings(input_ids))
        x = self.backbone.blocks(x)
        x = self.backbone.final_norm(x)
        last = x[torch.arange(x.shape[0], device=x.device), lengths - 1]
        return self.classifier(last)


def configure_trainable(model: GPTClassifier, last_blocks: int = 1) -> int:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if not 0 < last_blocks <= len(model.backbone.blocks):
        raise ValueError("last_blocks must be between 1 and total GPT blocks")
    for block in list(model.backbone.blocks)[-last_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.backbone.final_norm.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    return trainable_count(model)


@torch.no_grad()
def evaluate_classifier(model, loader, device) -> dict:
    previous_mode = model.training
    model.eval()
    counts = [0, 0, 0, 0]  # TN, FP, FN, TP
    total_loss = 0.0
    total = 0
    try:
        for inputs, lengths, labels in loader:
            labels = labels.to(device)
            logits = model(inputs.to(device), lengths.to(device))
            predictions = logits.argmax(-1)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total += len(labels)
            counts[0] += int(((predictions == 0) & (labels == 0)).sum())
            counts[1] += int(((predictions == 1) & (labels == 0)).sum())
            counts[2] += int(((predictions == 0) & (labels == 1)).sum())
            counts[3] += int(((predictions == 1) & (labels == 1)).sum())
    finally:
        model.train(previous_mode)
    if not total:
        raise ValueError("Cannot evaluate an empty dataset")
    tn, fp, fn, tp = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"samples": total, "loss": total_loss / total,
            "accuracy": (tp + tn) / total, "precision_spam": precision,
            "recall_spam": recall, "f1_spam": f1,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


@torch.no_grad()
def classify_text(model: GPTClassifier, tokenizer, text: str, device,
                  max_tokens: int = 128) -> dict:
    ids = tokenizer.encode(text)[:max_tokens] or [EOS_ID]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    previous_mode = model.training
    model.eval()
    try:
        scores = F.softmax(model(x, lengths), dim=-1)[0]
    finally:
        model.train(previous_mode)
    prediction = int(scores.argmax())
    return {"prediction": "spam" if prediction else "ham",
            "probability_spam": float(scores[1])}


def run(args):
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    base, tokenizer, info = load_pretrained_gpt2(args.model, device)
    model = GPTClassifier(base).to(device)
    if args.max_tokens > base.context_length:
        raise ValueError("max_tokens exceeds the GPT context length")
    trainable = configure_trainable(model, args.last_blocks)

    if args.predict is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["model_name"] != args.model or checkpoint["last_blocks"] != args.last_blocks:
            raise ValueError("Adapter model or last_blocks mismatch")
        apply_trainable_state(model, checkpoint["trainable_state_dict"])
        print(classify_text(model, tokenizer, args.predict, device, args.max_tokens))
        return

    samples = read_messages(args.data)
    training, validation, testing = stratified_partitions(samples, seed=args.seed)
    datasets = [MessageDataset(rows, tokenizer, args.max_tokens)
                for rows in (training, validation, testing)]
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=(i == 0),
                          collate_fn=collate_messages) for i, dataset in enumerate(datasets)]
    print("Total / splits:", len(samples), [len(d) for d in datasets],
          "trainable parameters:", trainable)
    baseline = evaluate_classifier(model, loaders[1], device)
    print("Before fine-tuning (validation):", baseline)
    class_counts = Counter(label for _, label in training)
    weights = torch.tensor([len(training) / (2 * class_counts[i]) for i in (0, 1)],
                           dtype=torch.float, device=device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=0.01)
    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for input_ids, lengths, labels in loaders[0]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids.to(device), lengths.to(device))
            loss = F.cross_entropy(logits, labels.to(device), weight=weights)
            loss.backward()
            optimizer.step()
        metrics = evaluate_classifier(model, loaders[1], device)
        history.append({"epoch": epoch, **metrics})
        print(f"Epoch {epoch}: validation={metrics}")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_state = trainable_state(model)
    apply_trainable_state(model, best_state)
    final_validation = evaluate_classifier(model, loaders[1], device)
    final_test = evaluate_classifier(model, loaders[2], device)
    demo = {message: classify_text(model, tokenizer, message, device, args.max_tokens)
            for message in ("Congratulations! Claim your free prize now",
                            "Can we meet tomorrow at school?")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "spam_adapter.pt"
    torch.save({"model_name": args.model, "last_blocks": args.last_blocks,
                "trainable_state_dict": best_state}, checkpoint_path)
    save_json(args.output_dir / "metrics_classification.json", {
        "model": args.model, "seed": args.seed, "trainable_parameters": trainable,
        "data_split": {"train": len(training), "validation": len(validation),
                       "test": len(testing)}, "baseline_validation": baseline,
        "history": history, "final_validation": final_validation,
        "final_test": final_test, "examples": demo,
    })
    print("Final held-out TEST metrics:", final_test)
    print("Saved:", checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/SMSSpamCollection"))
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--last-blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("experimentos/resultados_sprint6/classificacao"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experimentos/resultados_sprint6/classificacao/spam_adapter.pt"))
    parser.add_argument("--predict", help="Classify a new text using an existing adapter")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
