"""
train_vitasd_baseline.py — ✨ NEW (2026-08-10) Best-effort reproduction của
ViTASD (baseline gốc trong paper) để KIỂM TRA con số 61.77 / 41.12 / 52.64%
mà tác giả công bố, TRƯỚC KHI dùng nó làm điểm so sánh cho 2 module đóng góp.

╔══════════════════════════════════════════════════════════════════════════════╗
║ ĐỌC KỸ TRƯỚC KHI CHẠY — đây là bản "best-effort", KHÔNG PHẢI bản chính xác   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Paper ViTASD (Tran, Huynh, Le, Nguyen, Nguyen — Computer Speech & Language,
2025) chỉ đọc được phần preview (abstract, intro, section snippets) do
ScienceDirect/OpenReview khoá full text và repo GitHub không có code/checkpoint
dù abstract nói là có. Dựa trên preview đọc được ngày 2026-08-10:

  ĐÃ XÁC NHẬN (lấy trực tiếp từ text công khai):
    - Backbone: PhoBERT (không phải BERT gốc, không phải ViSoBERT)
    - Split: random 7:1:2 (train:dev:test) — ở mức comment
    - Kiến trúc "inspired by Zhang et al. (2020)", thay embedding BERT bằng
      PhoBERT embedding
    - Keyword bài báo có "Multi-head attention" → kiến trúc CÓ thêm 1 lớp
      attention, không phải chỉ [CLS] + linear classifier đơn thuần như
      BERT-pair-QA/NLI

  CHƯA XÁC NHẬN (paper không cho đọc, đang dùng giá trị phỏng đoán hợp lý):
    - Learning rate, batch size, số epoch, optimizer, scheduler cụ thể
    - Cách xây auxiliary sentence (câu mô tả aspect) — đang dùng dạng phẳng
      giống train_pair.py, GIỐNG CG-BERT/BERT-pair chứ chưa chắc giống ViTASD
    - Macro F1 của paper có tính lớp "none" hay không
    - Chi tiết chính xác của lớp multi-head attention (số head, đặt trước/sau
      pooling, có residual/layernorm không, Fig. 6 trong paper mới có hình vẽ
      nhưng bị khoá) — file này dùng thiết kế phổ biến nhất: self-attention
      trên toàn bộ sequence output của PhoBERT, cộng residual + LayerNorm,
      rồi lấy vector [CLS] để phân loại

MỤC ĐÍCH của file: chạy baseline "gần nhất có thể" với mô tả công khai của
paper, xem macro F1 ra bao nhiêu. Nếu ra số gần 61.77/41.12/52.64% → giả định
hợp lý, dùng luôn làm baseline. Nếu vẫn cách xa → vấn đề không nằm ở
hyperparameter mà nằm ở chi tiết kiến trúc/auxiliary sentence/metric chưa biết
→ BẮT BUỘC phải xin paper đầy đủ hoặc source code từ thầy Kiệt, không đoán tiếp
được nữa.

Dùng lại toàn bộ hạ tầng data/eval của train_pair.py (đã verify logic split,
gán nhãn, không leak) — chỉ thay model + bỏ 2 module đóng góp (Normalization,
Imbalanced Learning) để đây là baseline THUẦN, không bị nhiễu bởi cải tiến của
mình.

Cách chạy:
    # Smoke test
    python3 train_vitasd_baseline.py --domain mobile --epochs 1 --subsample 0.1

    # Baseline thật, từng domain (khớp đúng domain trong bảng baseline)
    python3 train_vitasd_baseline.py --domain mobile     --epochs 10 --fp16
    python3 train_vitasd_baseline.py --domain restaurant --epochs 10 --fp16
    python3 train_vitasd_baseline.py --domain hotel      --epochs 10 --fp16

Kết quả lưu ở experiments/results_vitasd_repro/<domain>/results.json, có cả
macro_f1 (loại "none") và macro_f1_with_none để đối chiếu cả 2 khả năng.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Dùng lại hạ tầng đã verify của train_pair.py — KHÔNG viết lại logic
# split/dataset/eval để tránh lệch/duplicate bug.
from train_pair import (
    MODEL_REGISTRY, MAX_LEN, SEED,
    extract_aspect_set, split_data, TASAPairDataset,
    SENTIMENTS, train_epoch, evaluate,
    _HAS_UNDERTHESEA,
)

BASELINE_VITASD = {"mobile": 61.77, "restaurant": 41.12, "hotel": 52.64}


# ── Model: PhoBERT + multi-head self-attention (best-effort) ───────────────────

class TASAPairModelMHA(nn.Module):
    """Best-effort reproduction của kiến trúc ViTASD: encoder (PhoBERT) + 1 lớp
    multi-head self-attention trên toàn bộ sequence output + residual/LayerNorm,
    rồi phân loại trên vector [CLS].

    ⚠️ Đây KHÔNG chắc là kiến trúc chính xác trong Fig. 6 của paper — chỉ là
    thiết kế phổ biến khớp với mô tả công khai ("PhoBERT embeddings" +
    "Multi-head attention"). Nếu xin được source code gốc, thay class này.
    """

    def __init__(self, num_labels: int, model_name: str, num_heads: int = 8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.layernorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state  # [B, T, H]

        # key_padding_mask: True = vị trí bị bỏ qua (padding)
        key_padding_mask = attention_mask == 0
        attn_out, _ = self.mha(seq, seq, seq, key_padding_mask=key_padding_mask)
        seq = self.layernorm(seq + attn_out)  # residual, chuẩn transformer block

        cls = seq[:, 0]  # [B, H] — vector [CLS] sau khi đã "nhìn" toàn câu qua MHA
        return self.classifier(self.dropout(cls))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["mobile", "restaurant", "hotel"], required=True)
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="phobert",
                   help="Mặc định PhoBERT — đúng backbone tác giả dùng (xác nhận từ paper).")
    p.add_argument("--epochs", type=int, default=10,
                   help="Số epoch của tác giả KHÔNG công bố công khai — 10 là giá trị "
                        "thử nghiệm hợp lý, tăng nếu dev F1 vẫn đang tăng ở epoch cuối.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num-heads", type=int, default=8,
                   help="Số head của lớp multi-head attention (giả định — Fig. 6 bị khoá).")
    p.add_argument("--no-segment", action="store_true",
                   help="KHÔNG khuyến khích tắt — PhoBERT pre-train trên text đã tách từ.")
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    if not args.no_segment and not _HAS_UNDERTHESEA:
        sys.exit("❌ Cần underthesea cho PhoBERT: pip3 install -r requirements.txt "
                  "(hoặc thêm --no-segment, KHÔNG khuyến khích).")

    model_name = MODEL_REGISTRY[args.model]
    config_name = f"vitasd_repro_{args.domain}_{args.model}"
    output_dir = Path(args.output) if args.output else \
        ROOT / "experiments" / "results_vitasd_repro" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}\nConfig: {config_name}")
    print("⚠️  Best-effort reproduction — xem docstring đầu file để biết phần nào "
          "đã xác nhận từ paper, phần nào là giả định.")

    with open(ROOT / "baseline" / "data" / args.domain / f"{args.domain}.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    if args.subsample < 1.0:
        import random
        keep = int(len(samples) * args.subsample)
        samples = random.Random(SEED).sample(samples, keep)
        print(f"[subsample] dùng {keep} comment (smoke test)")

    aspects = extract_aspect_set(samples)
    print(f"Aspects: {len(aspects)}")

    train_s, dev_s, test_s = split_data(samples)  # 7:1:2, mức comment, seed cố định
    print(f"Split (mức comment): train={len(train_s)} dev={len(dev_s)} test={len(test_s)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    segment = not args.no_segment

    # normalizer=None: baseline THUẦN, không bật Module 1 (Text Normalization)
    mk = lambda S: TASAPairDataset(S, aspects, tokenizer, None, MAX_LEN, segment, 1.0)
    train_ds, dev_ds, test_ds = mk(train_s), mk(dev_s), mk(test_s)
    print(f"Pairs: train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Loss = CE thuần, KHÔNG dùng Focal/weighted — đây là baseline, không phải
    # bản có Module 2 (Imbalanced Learning).
    loss_fn = nn.CrossEntropyLoss()
    model = TASAPairModelMHA(len(SENTIMENTS), model_name, num_heads=args.num_heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Mixed precision (fp16): {'ON' if use_amp else 'OFF'}")

    best_dev_f1, results = -1.0, []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, scaler)
        m = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch}/{args.epochs} — loss: {train_loss:.4f} | "
              f"dev macro F1 (loại none): {m['macro_f1']*100:.2f}% | "
              f"dev macro F1 (có none): {m['macro_f1_with_none']*100:.2f}% | "
              f"acc: {m['accuracy']*100:.2f}%")
        results.append({"epoch": epoch, "train_loss": train_loss, **m})

        if m["macro_f1"] > best_dev_f1:
            best_dev_f1 = m["macro_f1"]
            sd = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
                  for k, v in model.state_dict().items()}
            torch.save(sd, output_dir / "best_model.pt")

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    print("\n--- Test set ---")
    test_m = evaluate(model, test_loader, device, report=True)

    baseline = BASELINE_VITASD[args.domain]
    print(f"\n=== Test macro F1 (loại none): {test_m['macro_f1']*100:.2f}% "
          f"| macro F1 (có none): {test_m['macro_f1_with_none']*100:.2f}% "
          f"| baseline paper: {baseline}% "
          f"| chênh lệch (loại none): {test_m['macro_f1']*100 - baseline:+.2f} "
          f"| chênh lệch (có none): {test_m['macro_f1_with_none']*100 - baseline:+.2f} ===")

    summary = {
        "config": config_name,
        "purpose": "best-effort reproduction của ViTASD baseline — xem docstring đầu file",
        "confirmed_from_paper": ["backbone=PhoBERT", "split=7:1:2 random", "has multi-head attention layer"],
        "assumed_not_confirmed": ["lr", "batch_size", "epochs", "auxiliary_sentence_format",
                                   "macro_f1_includes_none", "exact MHA architecture"],
        "domain": args.domain, "model": args.model,
        "lr": args.lr, "batch_size": args.batch_size, "num_heads": args.num_heads,
        "epochs": args.epochs,
        "n_aspects": len(aspects),
        "n_pairs": {"train": len(train_ds), "dev": len(dev_ds), "test": len(test_ds)},
        "best_dev_f1": best_dev_f1, "test": test_m,
        "baseline_vitasd_paper": baseline,
        "diff_vs_paper_excl_none": test_m["macro_f1"] * 100 - baseline,
        "diff_vs_paper_incl_none": test_m["macro_f1_with_none"] * 100 - baseline,
        "epoch_logs": results,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
