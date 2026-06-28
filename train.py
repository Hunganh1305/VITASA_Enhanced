"""
train.py — Fine-tune PhoBERT trên ViTASA cho task TASA (entry point chính).

Task: Token classification với BIO tagging.
Mỗi subword token được gán nhãn O, B-ASPECT#SENTIMENT, hoặc I-ASPECT#SENTIMENT.

Ví dụ dùng (4 config ablation):
    # C1 — baseline gốc
    python train.py --domain mobile --loss ce

    # C2 — + Text Normalization
    python train.py --domain mobile --loss ce --normalize

    # C3 — + Focal Loss
    python train.py --domain mobile --loss focal

    # C4 — + cả hai
    python train.py --domain mobile --loss focal --normalize

Chạy trên Google Colab:
    !python train.py --domain mobile --loss focal --normalize --epochs 10
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from text_normalization.normalizer import TextNormalizer
from imbalanced_learning.losses import (
    FocalLoss,
    WeightedCrossEntropyLoss,
    compute_class_weights,
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "vinai/phobert-base-v2"
MAX_LEN = 256
BATCH_SIZE = 16
LR = 2e-5
SEED = 42


# ── Offset mapping ────────────────────────────────────────────────────────────

def build_offset_map(original: str, normalized: str) -> dict[int, int]:
    """Map mỗi char position trong original → char position trong normalized.

    Dùng difflib.SequenceMatcher để align 2 chuỗi. Cần thiết khi normalization
    thay đổi độ dài text (vd "nhanhhhhh" → "nhanh") và labels vẫn dùng
    char offset của original text.
    """
    offset_map: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, original, normalized, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                offset_map[i1 + k] = j1 + k
        elif op in ("replace", "delete"):
            for k in range(i2 - i1):
                offset_map[i1 + k] = j1
    return offset_map


# ── Label utils ───────────────────────────────────────────────────────────────

def build_label_map(samples: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    cats: set[str] = set()
    for s in samples:
        for _, _, cat in s["label"]:
            cats.add(cat)
    tags = ["O"] + [f"B-{c}" for c in sorted(cats)] + [f"I-{c}" for c in sorted(cats)]
    label2id = {t: i for i, t in enumerate(tags)}
    id2label = {i: t for t, i in label2id.items()}
    return label2id, id2label


# ── Dataset ───────────────────────────────────────────────────────────────────

class TASADataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        label2id: dict[str, int],
        normalizer: TextNormalizer | None = None,
        max_len: int = MAX_LEN,
    ):
        self.items = self._preprocess(samples, tokenizer, label2id, normalizer, max_len)

    @staticmethod
    def _preprocess(samples, tokenizer, label2id, normalizer, max_len):
        o_id = label2id["O"]
        records = []
        for s in samples:
            original = s["data"]

            if normalizer is not None:
                normalized = normalizer(original)
                offset_map = build_offset_map(original, normalized)
                text = normalized
            else:
                offset_map = None
                text = original

            enc = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
            )
            offsets = enc["offset_mapping"]  # [(char_start, char_end), ...]

            token_labels = [o_id] * len(offsets)

            for char_start, char_end, cat in s["label"]:
                b_tag = f"B-{cat}"
                i_tag = f"I-{cat}"
                if b_tag not in label2id:
                    continue

                # Remap char offsets nếu đã normalize
                if offset_map is not None:
                    norm_start = offset_map.get(char_start, char_start)
                    # char_end - 1 vì offset_map map từng char, end là exclusive
                    norm_end = offset_map.get(char_end - 1, char_end - 1) + 1
                else:
                    norm_start, norm_end = char_start, char_end

                first = True
                for i, (cs, ce) in enumerate(offsets):
                    if cs == 0 and ce == 0:  # special token ([CLS], [SEP], padding)
                        continue
                    if cs < norm_end and ce > norm_start:  # token overlap với span
                        if first:
                            token_labels[i] = label2id[b_tag]
                            first = False
                        else:
                            token_labels[i] = label2id.get(i_tag, o_id)

            records.append({
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(token_labels, dtype=torch.long),
            })
        return records

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class TASAModel(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state))  # [B, L, C]


# ── Loss factory ──────────────────────────────────────────────────────────────

def build_loss_fn(loss_type: str, train_labels_flat: list[int], num_labels: int, device):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    weights = compute_class_weights(
        train_labels_flat, num_classes=num_labels, strategy="effective_number"
    ).to(device)

    if loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(class_weights=weights)

    if loss_type == "focal":
        return FocalLoss(gamma=2.0, alpha=weights)

    raise ValueError(f"Unknown loss: {loss_type!r}. Chọn ce | weighted_ce | focal")


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)  # [B, L, C]
        B, L, C = logits.shape
        loss = loss_fn(logits.view(B * L, C), labels.view(B * L))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, label2id, id2label, device) -> float:
    """Trả về macro F1 trên tất cả non-O labels (span-level token F1)."""
    model.eval()
    o_id = label2id["O"]
    non_o_ids = [i for i in id2label if i != o_id]
    all_preds, all_true = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(-1)  # [B, L]

        for pred_row, label_row, mask_row in zip(preds, labels, attention_mask):
            active = mask_row.bool()
            all_preds.extend(pred_row[active].cpu().tolist())
            all_true.extend(label_row[active].cpu().tolist())

    return f1_score(
        all_true, all_preds,
        labels=non_o_ids,
        average="macro",
        zero_division=0,
    )


# ── Data split ────────────────────────────────────────────────────────────────

def split_data(samples, train_ratio=0.7, dev_ratio=0.1, seed=SEED):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    return shuffled[:n_train], shuffled[n_train:n_train + n_dev], shuffled[n_train + n_dev:]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["mobile", "restaurant", "hotel"], required=True)
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="ce")
    parser.add_argument("--normalize", action="store_true", help="Áp dụng TextNormalizer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", type=str, default=None,
                        help="Thư mục lưu checkpoint (default: experiments/results/<config>/)")
    args = parser.parse_args()

    config_name = f"{args.domain}_loss-{args.loss}{'_norm' if args.normalize else ''}"
    output_dir = Path(args.output) if args.output else ROOT / "experiments" / "results" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {config_name}")

    # Load data
    data_path = ROOT / "baseline" / "data" / args.domain / f"{args.domain}.jsonl"
    with open(data_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    train_samples, dev_samples, test_samples = split_data(samples)
    print(f"Split: train={len(train_samples)} dev={len(dev_samples)} test={len(test_samples)}")

    # Label map (built from full dataset để đủ coverage)
    label2id, id2label = build_label_map(samples)
    num_labels = len(label2id)
    print(f"Labels: {num_labels} (including O)")

    # Tokenizer & normalizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    normalizer = TextNormalizer() if args.normalize else None

    # Datasets
    print("Preprocessing datasets...")
    train_ds = TASADataset(train_samples, tokenizer, label2id, normalizer)
    dev_ds = TASADataset(dev_samples, tokenizer, label2id, normalizer)
    test_ds = TASADataset(test_samples, tokenizer, label2id, normalizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Loss function — collect training labels để tính class weights
    train_labels_flat = [lbl.item() for item in train_ds for lbl in item["labels"]]
    loss_fn = build_loss_fn(args.loss, train_labels_flat, num_labels, device)

    # Model
    model = TASAModel(num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Training loop
    best_dev_f1 = -1.0
    results = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        dev_f1 = evaluate(model, dev_loader, label2id, id2label, device)
        print(f"Epoch {epoch}/{args.epochs} — loss: {train_loss:.4f} | dev F1: {dev_f1:.4f}")
        results.append({"epoch": epoch, "train_loss": train_loss, "dev_f1": dev_f1})

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Evaluate best model on test set
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_f1 = evaluate(model, test_loader, label2id, id2label, device)
    print(f"\n=== Test F1: {test_f1:.4f} (config: {config_name}) ===")

    # Save results
    summary = {
        "config": config_name,
        "domain": args.domain,
        "loss": args.loss,
        "normalize": args.normalize,
        "epochs": args.epochs,
        "best_dev_f1": best_dev_f1,
        "test_f1": test_f1,
        "epoch_logs": results,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
