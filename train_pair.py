"""
train_pair.py — ✨ NEW (2026-07-20) Fine-tune ViSoBERT / PhoBERT trên ViTASA
theo ĐÚNG formulation của paper gốc: TARGET-ASPECT PAIR CLASSIFICATION
(không phải BIO span tagging như train.py cũ).

Thay thế train.py vì phát hiện ra train.py mô hình hoá sai task.
Metric giờ là macro F1 trên 3 lớp sentiment, có thể so sánh trực tiếp với
baseline ViTASD (61.77 / 41.12 / 52.64) — không còn mất trong số lớp khổng lồ.

╔══════════════════════════════════════════════════════════════════════════════╗
║ TẠI SAO CÓ FILE NÀY (đọc kỹ trước khi sửa) — 2026-07-20                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

`train.py` (bản cũ) mô hình hoá task thành BIO sequence labeling: mỗi subword
token được gán 1 nhãn trong tập {O, B-ASPECT#SENT, I-ASPECT#SENT}. Cách này tạo
ra số lớp khổng lồ và macro F1 bị tính trên toàn bộ số lớp đó:

    Domain      | số lớp BIO | macro F1 đo được (PhoBERT+CE, 30 epoch)
    ------------|------------|----------------------------------------
    mobile      |    59      | 28.48%
    restaurant  |   127      |  5.95%
    hotel       |   233      |  9.78%

trong khi baseline ViTASD của paper báo cáo 61.77 / 41.12 / 52.64%. Khoảng cách
30-43 điểm này KHÔNG phải do bug training, backbone, hay thiếu CRF — mà do đang
đo trên một thang đo hoàn toàn khác (macro F1 trên 233 lớp span-level exact-match
so với macro F1 trên 3 lớp sentiment).

Ba bằng chứng dẫn tới kết luận formulation đúng là pair classification:

1. File restaurant.jsonl / hotel.jsonl có sẵn key `labels` (số nhiều) mà
   train.py cũ KHÔNG hề đọc, nội dung đúng dạng cặp (aspect, sentiment) và
   KHÔNG có char offset:
       "{FACILITIES#DESIGN&FEATURES, negative}, {SERVICE#GENERAL, positive}, ..."
   Nếu task gốc là span tagging thì field này vô nghĩa.

2. Abstract paper nói "over 500,000 target-aspect pairs", nhưng toàn bộ dataset
   chỉ có ~18,800 span annotation. Con số 500k chỉ giải thích được khi enumerate
   mọi cặp (comment × aspect) — tức mỗi cặp là 1 sample cần phân loại.

3. Paper so sánh ViTASD với CG-BERT, QACG-BERT, BERT-pair-QA, BERT-pair-NLI —
   toàn bộ đều là họ mô hình pair-classification kiểu SentiHood (Saeidi et al.
   2016), nhận input (câu, aspect) và xuất ra 1 nhãn sentiment. Không model nào
   trong danh sách đó là sequence labeling.

╔══════════════════════════════════════════════════════════════════════════════╗
║ FORMULATION MỚI                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Với mỗi comment và mỗi aspect a thuộc tập aspect của domain, sinh ra 1 sample:

    input : [CLS] <câu đã normalize/segment> [SEP] <mô tả aspect a> [SEP]
    output: 1 trong 4 lớp {none, positive, negative, neutral}

"none" nghĩa là comment không nói gì về aspect a. Đây là cách BERT-pair-NLI
(Sun et al., 2019) xây dựng auxiliary sentence, và là lý do số sample nở ra
đúng bằng con số "target-aspect pairs" mà paper báo cáo:

    mobile     : 2000 comments × 10 aspects =  20,000 samples
    restaurant : 2000 comments × 27 aspects =  54,000 samples
    hotel      : 2000 comments × 54 aspects = 108,000 samples

METRIC: macro F1 trên 3 lớp sentiment (positive/negative/neutral), LOẠI lớp
"none" — đây mới là con số so sánh trực tiếp được với baseline 61.77/41.12/52.64.
Nếu tính cả "none" thì điểm bị thổi phồng vì "none" chiếm 76-94% dữ liệu.

LƯU Ý VỀ IMBALANCED LEARNING (Module 2): formulation này làm cho đóng góp của
Module 2 trở nên hợp lý hơn hẳn so với bản BIO cũ — phân bố lớp thực tế là:

    mobile     : none 76.5%  positive 14.4%  negative 8.1%  neutral 1.1%
    restaurant : none 90.7%  positive  7.5%  negative 1.6%  neutral 0.2%
    hotel      : none 94.5%  positive  3.9%  negative 1.5%  neutral 0.04%

Neutral hiếm tới mức 47 mẫu/108,000 ở hotel — đúng bài toán mà Focal Loss và
class weighting sinh ra để giải.

╔══════════════════════════════════════════════════════════════════════════════╗
║ VÍ DỤ CHẠY                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

    # C1 — baseline (CE thường, không normalize) ← đây là baseline để so với paper
    python train_pair.py --domain mobile --loss ce

    # C2 — + Text Normalization (Module 1)
    python train_pair.py --domain mobile --loss ce --normalize

    # C3 — + Imbalanced Learning (Module 2)
    python train_pair.py --domain mobile --loss focal

    # C4 — Full Model (cả 2 module)
    python train_pair.py --domain mobile --loss focal --normalize

    # Chạy nhanh để smoke-test pipeline (lấy 10% dữ liệu, 1 epoch)
    python train_pair.py --domain mobile --loss ce --epochs 1 --subsample 0.1

Các flag --model / --keep-expressive / --weight-strategy / --no-segment giữ
nguyên ý nghĩa như train.py cũ.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

try:
    from underthesea import word_tokenize as _underthesea_word_tokenize
    _HAS_UNDERTHESEA = True
except ImportError:
    _HAS_UNDERTHESEA = False

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from text_normalization.normalizer import TextNormalizer
from imbalanced_learning.losses import (
    FocalLoss,
    WeightedCrossEntropyLoss,
    compute_class_weights,
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "phobert": "vinai/phobert-base-v2",
    "visobert": "uitnlp/visobert",
}

# Ngắn hơn train.py cũ (256): input giờ là (câu + aspect) nhưng số sample nở ra
# 10-54 lần, nên cần cân bằng lại chi phí tính toán. 160 token đủ phủ ~99% comment.
MAX_LEN = 160
BATCH_SIZE = 32
LR = 2e-5
SEED = 42

# Thứ tự cố định: index 0 = "none" (lớp không có ý kiến về aspect này).
# 3 lớp sentiment thật nằm ở index 1..3 — macro F1 chỉ tính trên nhóm này.
SENTIMENTS = ["none", "positive", "negative", "neutral"]
SENT2ID = {s: i for i, s in enumerate(SENTIMENTS)}
NONE_ID = SENT2ID["none"]
SENTIMENT_IDS = [i for s, i in SENT2ID.items() if s != "none"]


# ── Word segmentation ─────────────────────────────────────────────────────────

def segment_text(text: str) -> str:
    """Ghép âm tiết thành từ tiếng Việt bằng underscore ("học sinh" -> "học_sinh").

    PhoBERT được pretrain trên text đã word-segment kiểu này (RDRSegmenter).
    Khác với train.py cũ, ở đây KHÔNG cần remap char offset vì formulation mới
    không dùng span offset nữa — chỉ cần chuỗi text để đưa vào encoder.
    """
    if not text.strip():
        return text
    return _underthesea_word_tokenize(text, format="text")


# ── Aspect utils ──────────────────────────────────────────────────────────────

def extract_aspect_set(samples: list[dict]) -> list[str]:
    """Lấy toàn bộ aspect (đã bỏ phần #SENTIMENT) xuất hiện trong domain.

    Nhãn trong dataset có dạng "FOOD#QUALITY#POSITIVE" (restaurant/hotel — aspect
    gồm 2 tầng ENTITY#ATTRIBUTE) hoặc "BATTERY#POSITIVE" (mobile — 1 tầng).
    Sentiment luôn là phần sau dấu # cuối cùng, nên rsplit("#", 1) xử lý được cả
    hai dạng.
    """
    aspects: set[str] = set()
    for s in samples:
        for _, _, cat in s["label"]:
            aspects.add(cat.rsplit("#", 1)[0])
    return sorted(aspects)


def aspect_to_text(aspect: str) -> str:
    """Chuyển mã aspect thành chuỗi tự nhiên hơn để làm auxiliary sentence.

    "ROOM_AMENITIES#DESIGN&FEATURES" -> "room amenities design features"

    Lý do không feed thẳng mã gốc: tokenizer sẽ băm "ROOM_AMENITIES#DESIGN&FEATURES"
    thành chuỗi subword vô nghĩa. Tách ra thành từ rời giúp encoder tận dụng được
    ngữ nghĩa sẵn có của từng từ (kể cả khi là tiếng Anh — cả PhoBERT lẫn ViSoBERT
    đều có gặp từ tiếng Anh trong lúc pretrain).
    """
    return aspect.replace("#", " ").replace("&", " ").replace("_", " ").lower()


def build_pair_labels(sample: dict) -> tuple[dict[str, str], int]:
    """Trả về ({aspect: sentiment}, số_xung_đột) cho 1 comment, từ field `label`.

    Không dùng field `labels` (chuỗi text) vì: (1) mobile.jsonl không có field
    này, (2) nó là dạng string cần parse thêm, (3) `label` chứa cùng thông tin
    aspect+sentiment và có mặt ở cả 3 domain -> nhất quán hơn.

    Khi 1 comment gán nhiều sentiment khác nhau cho CÙNG 1 aspect (vd vừa khen
    vừa chê chất lượng đồ ăn), giữ lần xuất hiện ĐẦU TIÊN. Đây là điểm mất thông
    tin đã biết của formulation pair-classification (mỗi cặp chỉ giữ được 1 nhãn)
    — số lần xảy ra được đếm và in ra lúc load để đánh giá mức ảnh hưởng.
    """
    out: dict[str, str] = {}
    conflicts = 0
    for _, _, cat in sample["label"]:
        aspect, sent = cat.rsplit("#", 1)
        sent = sent.lower()
        if sent not in SENT2ID or sent == "none":
            continue
        if aspect in out:
            if out[aspect] != sent:
                conflicts += 1
            continue
        out[aspect] = sent
    return out, conflicts


# ── Dataset ───────────────────────────────────────────────────────────────────

class TASAPairDataset(Dataset):
    """Mỗi item = 1 cặp (comment, aspect) -> 1 nhãn trong {none, pos, neg, neu}."""

    def __init__(
        self,
        samples: list[dict],
        aspects: list[str],
        tokenizer,
        normalizer: TextNormalizer | None = None,
        max_len: int = MAX_LEN,
        segment: bool = True,
        subsample_none: float = 1.0,
        rng_seed: int = SEED,
    ):
        self.items = self._preprocess(
            samples, aspects, tokenizer, normalizer, max_len, segment,
            subsample_none, rng_seed,
        )

    @staticmethod
    def _preprocess(
        samples, aspects, tokenizer, normalizer, max_len, segment,
        subsample_none, rng_seed,
    ):
        rng = random.Random(rng_seed)
        aspect_texts = {a: aspect_to_text(a) for a in aspects}
        records = []
        total_conflicts = 0

        for s in samples:
            # NFC-normalize để text nhất quán (ViTASA có lẫn Unicode dựng sẵn và
            # tổ hợp). Ở đây không còn ràng buộc offset như train.py cũ, nhưng vẫn
            # giữ để chuỗi đưa vào tokenizer là ổn định giữa các lần chạy.
            text = unicodedata.normalize("NFC", s["data"])

            if normalizer is not None:
                text = normalizer(text)
            if segment:
                text = segment_text(text)

            pair_labels, conflicts = build_pair_labels(s)
            total_conflicts += conflicts

            for aspect in aspects:
                sent = pair_labels.get(aspect, "none")
                label_id = SENT2ID[sent]

                # Cân bằng thô: bỏ bớt ngẫu nhiên cặp "none" nếu được yêu cầu.
                # Mặc định 1.0 = giữ hết (đúng phân bố gốc của benchmark).
                if label_id == NONE_ID and subsample_none < 1.0:
                    if rng.random() > subsample_none:
                        continue

                enc = tokenizer(
                    text,
                    aspect_texts[aspect],
                    max_length=max_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None,
                )
                records.append({
                    "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                    "label": torch.tensor(label_id, dtype=torch.long),
                })

        if total_conflicts:
            print(f"  [info] {total_conflicts} cặp bị xung đột sentiment (giữ nhãn đầu tiên)")
        return records

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class TASAPairModel(nn.Module):
    """Encoder + classifier trên [CLS] — kiến trúc chuẩn của họ BERT-pair.

    Khác train.py cũ (classifier chạy trên TỪNG token để gán BIO), ở đây chỉ
    phân loại 1 lần cho cả cặp (câu, aspect), lấy biểu diễn [CLS] làm đại diện.
    """

    def __init__(self, num_labels: int, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [B, H] — vector tại vị trí [CLS]
        return self.classifier(self.dropout(cls))  # [B, C]


# ── Loss factory ──────────────────────────────────────────────────────────────

def build_loss_fn(loss_type, train_labels, num_labels, device, weight_strategy="effective_number"):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    weights = compute_class_weights(
        train_labels, num_classes=num_labels, strategy=weight_strategy
    ).to(device)
    print(f"  Class weights ({weight_strategy}): " + ", ".join(
        f"{SENTIMENTS[i]}={weights[i]:.3f}" for i in range(num_labels)
    ))

    if loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(class_weights=weights)
    if loss_type == "focal":
        return FocalLoss(gamma=2.0, alpha=weights)
    raise ValueError(f"Unknown loss: {loss_type!r}")


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        loss = loss_fn(logits, batch["label"].to(device))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, report: bool = False) -> dict:
    """Macro F1 trên 3 lớp sentiment, LOẠI lớp "none".

    Đây là con số so sánh trực tiếp được với baseline ViTASD (61.77 / 41.12 /
    52.64%). Trả về thêm f1_with_none và accuracy để tiện chẩn đoán — nhưng
    KHÔNG dùng 2 số đó để so với paper: "none" chiếm 76-94% dữ liệu nên bất kỳ
    metric nào tính cả "none" đều bị thổi phồng.
    """
    model.eval()
    preds, trues = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
        trues.extend(batch["label"].tolist())

    macro_f1 = f1_score(trues, preds, labels=SENTIMENT_IDS, average="macro", zero_division=0)
    out = {
        "macro_f1": macro_f1,
        "macro_f1_with_none": f1_score(trues, preds, average="macro", zero_division=0),
        "accuracy": float(np.mean(np.array(preds) == np.array(trues))),
    }
    if report:
        print(classification_report(
            trues, preds,
            labels=list(range(len(SENTIMENTS))),
            target_names=SENTIMENTS,
            zero_division=0, digits=4,
        ))
    return out


# ── Data split ────────────────────────────────────────────────────────────────

def split_data(samples, train_ratio=0.7, dev_ratio=0.1, seed=SEED):
    """Split ở mức COMMENT (không phải mức cặp) — tránh leak: nếu split ở mức cặp
    thì các cặp của cùng 1 comment sẽ nằm cả ở train lẫn test, khiến điểm test
    cao ảo. Tỷ lệ 7:1:2 theo đúng README của ViTASA."""
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train, n_dev = int(n * train_ratio), int(n * dev_ratio)
    return shuffled[:n_train], shuffled[n_train:n_train + n_dev], shuffled[n_train + n_dev:]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["mobile", "restaurant", "hotel"], required=True)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="ce")
    p.add_argument("--normalize", action="store_true", help="Bật Module 1 (Text Normalization)")
    p.add_argument("--keep-expressive", action="store_true",
                   help="Chỉ dùng khi --normalize. Giữ marker biểu cảm (kk/haha) thay vì xóa.")
    p.add_argument("--weight-strategy",
                   choices=["effective_number", "inverse_frequency", "pos_neg_ratio"],
                   default="effective_number")
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="visobert")
    p.add_argument("--epochs", type=int, default=3,
                   help="Mặc định thấp hơn train.py cũ vì số sample/epoch lớn hơn 10-54 lần.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--no-segment", action="store_true")
    p.add_argument("--subsample", type=float, default=1.0,
                   help="Lấy ngẫu nhiên tỷ lệ COMMENT này để chạy nhanh (smoke test). 1.0 = full.")
    p.add_argument("--subsample-none", type=float, default=1.0,
                   help="Giữ lại tỷ lệ này của các cặp nhãn 'none' (1.0 = giữ hết, đúng "
                        "phân bố benchmark). Giảm xuống giúp train nhanh hơn nhiều nhưng "
                        "làm lệch phân bố so với paper — chỉ dùng để thử nghiệm.")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    if not args.no_segment and not _HAS_UNDERTHESEA:
        sys.exit("❌ Cần underthesea: pip3 install -r requirements.txt (hoặc thêm --no-segment)")
    if args.keep_expressive and not args.normalize:
        sys.exit("❌ --keep-expressive chỉ có tác dụng khi bật --normalize.")

    model_name = MODEL_REGISTRY[args.model]
    config_name = (
        f"pair_{args.domain}_loss-{args.loss}"
        f"{'_norm' if args.normalize else ''}"
        f"{'_keepexpr' if args.keep_expressive else ''}"
        f"_{args.model}"
        f"{'_noseg' if args.no_segment else ''}"
        f"{'' if args.weight_strategy == 'effective_number' else '_w-' + args.weight_strategy}"
    )
    output_dir = Path(args.output) if args.output else ROOT / "experiments" / "results_pair" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}\nConfig: {config_name}")

    # Load
    with open(ROOT / "baseline" / "data" / args.domain / f"{args.domain}.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    if args.subsample < 1.0:
        keep = int(len(samples) * args.subsample)
        samples = random.Random(SEED).sample(samples, keep)
        print(f"[subsample] dùng {keep} comment (smoke test)")

    aspects = extract_aspect_set(samples)
    print(f"Aspects: {len(aspects)} → mỗi comment sinh ra {len(aspects)} cặp")

    train_s, dev_s, test_s = split_data(samples)
    print(f"Split (mức comment): train={len(train_s)} dev={len(dev_s)} test={len(test_s)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    normalizer = TextNormalizer(keep_expressive_markers=args.keep_expressive) if args.normalize else None
    segment = not args.no_segment

    print(f"Preprocessing... (normalize={'ON' if normalizer else 'OFF'}, segment={'ON' if segment else 'OFF'})")
    mk = lambda S, sub: TASAPairDataset(S, aspects, tokenizer, normalizer, MAX_LEN, segment, sub)
    train_ds = mk(train_s, args.subsample_none)
    dev_ds = mk(dev_s, 1.0)    # dev/test luôn giữ nguyên phân bố gốc để đánh giá công bằng
    test_ds = mk(test_s, 1.0)
    print(f"Pairs: train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    train_labels = [int(it["label"]) for it in train_ds.items]
    dist = Counter(train_labels)
    print("Train label distribution: " + ", ".join(
        f"{SENTIMENTS[i]}={dist.get(i,0)} ({dist.get(i,0)/len(train_labels)*100:.1f}%)"
        for i in range(len(SENTIMENTS))
    ))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    loss_fn = build_loss_fn(args.loss, train_labels, len(SENTIMENTS), device, args.weight_strategy)
    model = TASAPairModel(len(SENTIMENTS), model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    best_dev_f1, results = -1.0, []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        m = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch}/{args.epochs} — loss: {train_loss:.4f} | "
              f"dev macro F1 (3 sentiment): {m['macro_f1']*100:.2f}% | acc: {m['accuracy']*100:.2f}%")
        results.append({"epoch": epoch, "train_loss": train_loss, **m})

        if m["macro_f1"] > best_dev_f1:
            best_dev_f1 = m["macro_f1"]
            sd = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
                  for k, v in model.state_dict().items()}
            torch.save(sd, output_dir / "best_model.pt")

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    print("\n--- Test set ---")
    test_m = evaluate(model, test_loader, device, report=True)

    baseline = {"mobile": 61.77, "restaurant": 41.12, "hotel": 52.64}[args.domain]
    print(f"=== Test macro F1 (3 sentiment): {test_m['macro_f1']*100:.2f}% "
          f"| baseline ViTASD {args.domain}: {baseline}% "
          f"| chênh lệch: {test_m['macro_f1']*100 - baseline:+.2f} ===")

    summary = {
        "config": config_name, "formulation": "target-aspect pair classification",
        "domain": args.domain, "model": args.model, "loss": args.loss,
        "normalize": args.normalize, "keep_expressive": args.keep_expressive,
        "weight_strategy": args.weight_strategy, "epochs": args.epochs,
        "n_aspects": len(aspects),
        "n_pairs": {"train": len(train_ds), "dev": len(dev_ds), "test": len(test_ds)},
        "best_dev_f1": best_dev_f1, "test": test_m,
        "baseline_vitasd": baseline, "epoch_logs": results,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
