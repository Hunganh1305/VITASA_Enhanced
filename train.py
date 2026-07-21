"""
train.py — Fine-tune ViSoBERT / PhoBERT trên ViTASA (BIO span tagging formulation).

⚠️  CẢNH BÁO — 2026-07-20: File này dùng formulation SAI (BIO tagging thay vì
pair classification). Điều này khiến macro F1 bị tính trên số lớp sai (59/127/233
lớp span) so với thang đo đúng của paper (4 lớp, 3 lớp sentiment).

>>> DÙNG train_pair.py THAY VÀO ĐỂ SO SÁNH VỚI BASELINE VITASD <<<

File này chỉ giữ lại để tham khảo hoặc nếu cần debug công thức BIO cũ. Mọi kết quả
mới (ablation, final scores) phải dùng train_pair.py.

Chi tiết → FORMULATION_FIX_2026-07-20.md

Task cũ: Token classification với BIO tagging.
Mỗi subword token được gán nhãn O, B-ASPECT#SENTIMENT, hoặc I-ASPECT#SENTIMENT.

Backbone mặc định ĐÃ ĐỔI sang ViSoBERT (2026-07-12): quick-test + full ablation 20
epoch cho thấy ViSoBERT vượt PhoBERT rõ rệt ở cả 3 domain (đặc biệt Restaurant +61%,
Hotel gần gấp đôi) — hợp lý vì ViSoBERT pretrain trên văn bản mạng xã hội tiếng Việt
thật, khớp domain của ViTASA hơn PhoBERT (pretrain trên Wikipedia/báo chí). Kết quả
PhoBERT cũ được giữ tham khảo ở experiments/results_phobert_final/. Dùng --model
phobert nếu cần chạy lại backbone cũ để so sánh.

Word segmentation (underthesea) được BẬT MẶC ĐỊNH cho mọi config khi dùng PhoBERT —
PhoBERT cần input đã ghép âm tiết thành từ bằng "_". Với ViSoBERT thì CHƯA rõ có cần
segmentation hay không (quick-test mới chỉ thử segment=True) — dùng --no-segment để
so sánh nếu muốn kiểm tra thêm.

Ví dụ dùng (4 config ablation, backbone mặc định visobert):
    # C1 — baseline (CE thường)
    python train.py --domain mobile --loss ce

    # C2 — + Text Normalization
    python train.py --domain mobile --loss ce --normalize

    # C3 — + Focal Loss
    python train.py --domain mobile --loss focal

    # C4 — + cả hai (Full Model)
    python train.py --domain mobile --loss focal --normalize

    # Chạy lại với PhoBERT (backbone cũ) để so sánh
    python train.py --domain mobile --loss focal --normalize --model phobert

Chạy trên Google Colab:
    !python train.py --domain mobile --loss focal --normalize --epochs 20

Option ablation bổ sung (2026-07-16, dựa trên finding của paper ViGoEmotions
— EACL 2026 — về emoji/marker biểu cảm và class weighting, xem
imbalanced_learning/losses.py và text_normalization/normalizer.py để biết
chi tiết công thức/lý do):
    # Giữ nguyên marker biểu cảm (kk/haha...) thay vì xóa khi normalize
    python train.py --domain restaurant --loss focal --normalize --keep-expressive

    # Đổi class-weighting strategy (mặc định vẫn effective_number, không đổi
    # behavior cũ) sang pos_neg_ratio kiểu ViGoEmotions
    python train.py --domain hotel --loss weighted_ce --weight-strategy pos_neg_ratio

Tối ưu dung lượng checkpoint (2026-07-16, mặc định BẬT — không cần flag gì
thêm): mọi checkpoint từ giờ tự động lưu ở fp16 thay vì fp32, giảm ~50% dung
lượng (~373MB -> ~187MB cho ViSoBERT) mà không ảnh hưởng khả năng load lại để
eval/inference (đã verify: load fp16 checkpoint vào model fp32 hoạt động bình
thường qua cơ chế cast tự động của PyTorch, sai số ~1e-5, không đáng kể). Dùng
--no-fp16-checkpoint nếu vì lý do nào đó cần giữ nguyên fp32.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

try:
    from seqeval.metrics import f1_score as seqeval_f1_score
    _HAS_SEQEVAL = True
except ImportError:
    _HAS_SEQEVAL = False

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

MAX_LEN = 256
BATCH_SIZE = 16
LR = 2e-5
SEED = 42
IGNORE_INDEX = -100  # nhãn cho vị trí không train (CLS/SEP/PAD) — loss & eval sẽ loại các vị trí này


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


def remap_span(char_start: int, char_end: int, *offset_maps: dict[int, int] | None) -> tuple[int, int]:
    """Áp lần lượt nhiều offset_map (original → normalized → segmented, ...) lên
    1 char span. offset_map nào là None thì bỏ qua (giữ nguyên vị trí)."""
    ns, ne = char_start, char_end
    for offset_map in offset_maps:
        if offset_map is None:
            continue
        ns = offset_map.get(ns, ns)
        ne = offset_map.get(ne - 1, ne - 1) + 1
    return ns, ne


# ── Word segmentation (Vietnamese) ───────────────────────────────────────────

def segment_text(text: str) -> str:
    """Ghép âm tiết thành từ tiếng Việt bằng underscore (vd "học sinh" ->
    "học_sinh"), dùng underthesea.word_tokenize. PhoBERT được pretrain trên
    text đã word-segment kiểu này (RDRSegmenter) — nếu feed thẳng text tách
    theo khoảng trắng (tách theo ÂM TIẾT chứ không phải TỪ), input sẽ lệch
    hẳn so với phân phối lúc pretrain, làm giảm chất lượng embedding.
    """
    if not text.strip():
        return text
    return _underthesea_word_tokenize(text, format="text")


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
        segment: bool = True,
    ):
        self.items = self._preprocess(samples, tokenizer, label2id, normalizer, max_len, segment)

    @staticmethod
    def _preprocess(samples, tokenizer, label2id, normalizer, max_len, segment: bool = True):
        """Word-level alignment — không cần fast tokenizer hay offset_mapping.

        Approach:
        1. (tuỳ chọn) Word-segment text bằng underthesea — ghép âm tiết thành từ
           bằng "_", đúng format PhoBERT được pretrain (RDRSegmenter).
        2. Split text (đã segment) thành "words" với char offsets (dùng regex,
           mỗi "word" giờ có thể là 1 từ ghép nhiều âm tiết vd "học_sinh").
        3. Gán B/I/O label cho từng word dựa trên char spans của dataset (đã
           remap qua offset_map từ text gốc -> text đã normalize -> text đã segment).
        4. Tokenize từng word riêng để đếm số subword tokens.
        5. Propagate word label → tất cả subword tokens của word đó.
        """
        o_id = label2id["O"]
        records = []

        for s in samples:
            # NFC-normalize trước khi cắt offset: một số sample trong ViTASA lưu
            # text với Unicode dựng sẵn lẫn tổ hợp không đồng nhất (vd "tắt" có thể
            # là 1 codepoint dựng sẵn hoặc "t"+"ă"+dấu sắc tổ hợp). char_start/char_end
            # trong "label" được đánh số trên bản NFC, nên nếu không chuẩn hoá ở đây,
            # những sample bị lệch chuẩn hoá sẽ bị trỏ offset sai lệch, làm hỏng nhãn.
            original = unicodedata.normalize("NFC", s["data"])

            if normalizer is not None:
                base_text = normalizer(original)
                offset_map_norm = build_offset_map(original, base_text)
            else:
                base_text = original
                offset_map_norm = None

            if segment:
                text = segment_text(base_text)
                offset_map_seg = build_offset_map(base_text, text)
            else:
                text = base_text
                offset_map_seg = None

            # Bước 1: tách words (đã segment nếu bật) và lưu char offsets
            word_matches = list(re.finditer(r"\S+", text))
            words = [m.group() for m in word_matches]
            word_spans = [(m.start(), m.end()) for m in word_matches]

            # Bước 2: gán label ở word level
            word_labels = [o_id] * len(words)
            for char_start, char_end, cat in s["label"]:
                b_tag = f"B-{cat}"
                i_tag = f"I-{cat}"
                if b_tag not in label2id:
                    continue

                ns, ne = remap_span(char_start, char_end, offset_map_norm, offset_map_seg)

                first = True
                for wi, (ws, we) in enumerate(word_spans):
                    if ws < ne and we > ns:
                        word_labels[wi] = label2id[b_tag] if first else label2id.get(i_tag, o_id)
                        first = False

            # Bước 3: tokenize toàn bộ sequence, padding đến max_len
            enc = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )

            # Bước 4: propagate word label → subword tokens
            # Mặc định IGNORE_INDEX cho mọi vị trí (CLS, SEP, PAD) — chỉ những vị trí
            # thuộc 1 word thật (dưới đây) mới được set nhãn thật (kể cả "O").
            # Lý do: avg comment chỉ ~36 từ nhưng MAX_LEN=256 → ~85% mỗi sequence là
            # PAD. Nếu gán PAD = "O" như trước, loss bị áp đảo bởi PAD-là-O, model chỉ
            # cần học "luôn đoán O" để minimize loss → không học được nhãn thật (đây là
            # nguyên nhân chính khiến dev F1 ~ 0 ở các lần train trước).
            input_ids = enc["input_ids"]
            token_labels = [IGNORE_INDEX] * len(input_ids)

            token_idx = 1  # bỏ qua CLS
            for wi, word in enumerate(words):
                if token_idx >= len(input_ids):
                    break
                n_sub = len(tokenizer.tokenize(word)) or 1
                for j in range(n_sub):
                    if token_idx + j < len(input_ids):
                        token_labels[token_idx + j] = word_labels[wi]
                token_idx += n_sub

            records.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
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
    def __init__(self, num_labels: int, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state))  # [B, L, C]


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: Path, fp16: bool = True) -> None:
    """Lưu state_dict, mặc định cast về fp16 để giảm ~50% dung lượng file
    (2026-07-16, theo yêu cầu tối ưu dung lượng sau mỗi lần train — mỗi
    checkpoint ViSoBERT full-precision ~373MB, fp16 còn ~187MB).

    An toàn để load lại vào model fp32 bình thường: PyTorch tự cast dtype khi
    copy tensor trong load_state_dict (Module._load_from_state_dict dùng
    `param.data.copy_(input_param)`, hỗ trợ cast dtype). Chỉ mất chút độ
    chính xác số học (không đáng kể cho việc load lại để eval/inference), và
    chỉ ảnh hưởng checkpoint LƯU RA — không ảnh hưởng gì đến quá trình train
    (vẫn train fp32 bình thường, chỉ cast lúc save).

    Dùng --no-fp16-checkpoint nếu cần giữ nguyên fp32 (vd định fine-tune tiếp
    từ checkpoint này và muốn giữ tối đa độ chính xác optimizer state — dù
    hiện train.py không lưu optimizer state nên trường hợp này hiếm gặp).
    """
    state_dict = model.state_dict()
    if fp16:
        state_dict = {
            k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in state_dict.items()
        }
    torch.save(state_dict, path)


# ── Loss factory ──────────────────────────────────────────────────────────────

def build_loss_fn(
    loss_type: str,
    train_labels_flat: list[int],
    num_labels: int,
    device,
    weight_strategy: str = "effective_number",
):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    weights = compute_class_weights(
        train_labels_flat, num_classes=num_labels, strategy=weight_strategy
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
        logits_flat = logits.view(B * L, C)
        labels_flat = labels.view(B * L)

        # Chỉ tính loss trên vị trí thật (không phải CLS/SEP/PAD) — loại bỏ
        # trước khi đưa vào loss_fn nên không cần sửa gì trong losses.py
        # (CrossEntropyLoss / WeightedCrossEntropyLoss / FocalLoss đều dùng được).
        active = labels_flat != IGNORE_INDEX
        loss = loss_fn(logits_flat[active], labels_flat[active])

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, label2id, id2label, device) -> float:
    """Trả về span-level (entity-level exact-match) macro F1 qua seqeval — cùng
    kiểu metric mà paper ViTASD gốc dùng để báo cáo, nên so sánh trực tiếp được
    với baseline (61.77% / 41.12% / 52.64%). Yêu cầu span dự đoán khớp cả biên
    (start/end) lẫn category thì mới tính là true positive, khác với token-level
    F1 (chỉ so từng subword riêng lẻ) vốn dễ cho điểm cao ảo hơn.

    Nếu seqeval chưa được cài (pip install seqeval / trong requirements.txt),
    fallback về token-level macro F1 loại "O" như bản cũ và in cảnh báo — số đó
    KHÔNG so sánh trực tiếp được với baseline ViTASD.
    """
    model.eval()
    o_id = label2id["O"]
    non_o_ids = [i for i in id2label if i != o_id]

    true_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []
    all_preds_flat, all_true_flat = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(-1)  # [B, L]

        for pred_row, label_row, mask_row in zip(preds, labels, attention_mask):
            active = mask_row.bool() & (label_row != IGNORE_INDEX)
            true_ids = label_row[active].cpu().tolist()
            pred_ids = pred_row[active].cpu().tolist()
            if not true_ids:
                continue
            true_seqs.append([id2label[i] for i in true_ids])
            pred_seqs.append([id2label[i] for i in pred_ids])
            all_true_flat.extend(true_ids)
            all_preds_flat.extend(pred_ids)

    if _HAS_SEQEVAL:
        return seqeval_f1_score(true_seqs, pred_seqs, average="macro", zero_division=0)

    print(
        "[WARN] seqeval chưa được cài (pip install seqeval) — dùng token-level F1 "
        "thay thế, số này KHÔNG so sánh trực tiếp được với baseline ViTASD (span-level)."
    )
    return f1_score(
        all_true_flat, all_preds_flat,
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
    parser.add_argument(
        "--keep-expressive", action="store_true",
        help="Chỉ có tác dụng khi bật --normalize. Giữ nguyên marker biểu cảm/"
             "tiếng cười (kk, haha, hihi...) thay vì xóa — ablation dựa trên "
             "finding của paper ViGoEmotions (EACL 2026) rằng giữ nguyên "
             "emoji/marker cảm xúc cho Macro-F1 cao hơn xóa/convert. Mặc định "
             "TẮT (giữ hành vi cũ: xóa các marker này).",
    )
    parser.add_argument(
        "--weight-strategy",
        choices=["effective_number", "inverse_frequency", "pos_neg_ratio"],
        default="effective_number",
        help="Chiến lược tính class weight cho --loss weighted_ce|focal. Mặc "
             "định effective_number (không đổi behavior cũ). 'pos_neg_ratio' "
             "là công thức pos_weight kiểu ViGoEmotions (EACL 2026), tăng "
             "weight mạnh hơn cho class hiếm (vd Neutral) — xem "
             "imbalanced_learning/losses.py.",
    )
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="visobert",
                        help="Backbone model (default: visobert — vượt PhoBERT rõ rệt trong "
                             "thử nghiệm 2026-07-12, xem experiments/results_phobert_final/ "
                             "để so sánh lại nếu cần)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--no-segment", action="store_true",
                        help="TẮT word segmentation (underthesea). Mặc định BẬT vì PhoBERT "
                             "cần input đã ghép âm tiết thành từ (RDRSegmenter-style) — chỉ "
                             "tắt để debug/so sánh, không dùng cho kết quả chính thức.")
    parser.add_argument("--output", type=str, default=None,
                        help="Thư mục lưu checkpoint (default: experiments/results/<config>/)")
    parser.add_argument(
        "--no-fp16-checkpoint", action="store_true",
        help="TẮT lưu checkpoint dạng fp16 (mặc định BẬT từ 2026-07-16 để giảm "
             "~50% dung lượng file — ~373MB -> ~187MB cho ViSoBERT). Chỉ nên bật "
             "cờ này nếu có lý do cụ thể cần giữ checkpoint fp32 nguyên bản.",
    )
    args = parser.parse_args()

    if not args.no_segment and not _HAS_UNDERTHESEA:
        sys.exit(
            "❌ underthesea chưa được cài (pip install underthesea) — cần cho word segmentation.\n"
            "   Chạy: pip3 install -r requirements.txt\n"
            "   Hoặc thêm --no-segment để tắt tạm thời (KHÔNG khuyến nghị, PhoBERT sẽ nhận "
            "input sai format)."
        )

    if args.keep_expressive and not args.normalize:
        sys.exit("❌ --keep-expressive chỉ có tác dụng khi bật --normalize.")

    model_name = MODEL_REGISTRY[args.model]
    model_suffix = f"_{args.model}" if args.model != "phobert" else ""
    seg_suffix = "_noseg" if args.no_segment else ""
    expr_suffix = "_keepexpr" if args.keep_expressive else ""
    weight_suffix = (
        f"_w-{args.weight_strategy}" if args.weight_strategy != "effective_number" else ""
    )
    config_name = (
        f"{args.domain}_loss-{args.loss}{'_norm' if args.normalize else ''}"
        f"{expr_suffix}{model_suffix}{seg_suffix}{weight_suffix}"
    )
    output_dir = Path(args.output) if args.output else ROOT / "experiments" / "results" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) GPU qua Metal — cho phép train local, tránh giới
        # hạn thời gian phiên chạy của Colab free tier khi cần train nhiều epoch.
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    normalizer = (
        TextNormalizer(keep_expressive_markers=args.keep_expressive)
        if args.normalize else None
    )

    # Datasets
    segment = not args.no_segment
    print(f"Preprocessing datasets... (word segmentation: {'ON' if segment else 'OFF'})")
    train_ds = TASADataset(train_samples, tokenizer, label2id, normalizer, segment=segment)
    dev_ds = TASADataset(dev_samples, tokenizer, label2id, normalizer, segment=segment)
    test_ds = TASADataset(test_samples, tokenizer, label2id, normalizer, segment=segment)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Loss function — collect training labels để tính class weights
    # (loại IGNORE_INDEX vì đó không phải nhãn thật, tránh pha loãng tần suất class)
    train_labels_flat = [
        lbl.item() for item in train_ds for lbl in item["labels"] if lbl.item() != IGNORE_INDEX
    ]
    loss_fn = build_loss_fn(
        args.loss, train_labels_flat, num_labels, device,
        weight_strategy=args.weight_strategy,
    )

    # Model
    model = TASAModel(num_labels, model_name).to(device)
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
            save_checkpoint(model, output_dir / "best_model.pt", fp16=not args.no_fp16_checkpoint)

    # Evaluate best model on test set
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_f1 = evaluate(model, test_loader, label2id, id2label, device)
    print(f"\n=== Test F1: {test_f1:.4f} (config: {config_name}) ===")

    # Save results
    summary = {
        "config": config_name,
        "domain": args.domain,
        "model": args.model,
        "loss": args.loss,
        "normalize": args.normalize,
        "keep_expressive": args.keep_expressive,
        "weight_strategy": args.weight_strategy,
        "checkpoint_fp16": not args.no_fp16_checkpoint,
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
