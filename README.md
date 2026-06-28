# Enhancing Vietnamese Targeted Aspect Sentiment Analysis
### with Social Media Text Normalization and Imbalanced Learning

Master's thesis project — University of Information Technology (UIT)

---

## Overview

This project proposes two contributions to improve Vietnamese TASA on social media text:

1. **Text Normalization** — rule-based normalization of teencode, abbreviations, and elongated characters before feeding into PhoBERT
2. **Imbalanced Learning** — Focal Loss and Weighted Cross-Entropy to handle polarity imbalance (POSITIVE 73% / NEGATIVE 24.5% / NEUTRAL 2.3%)

Baseline model: PhoBERT fine-tuned on [ViTASA dataset](https://github.com/kh4nh12/ViTASA) (mobile / restaurant / hotel domains)

---

## Project Structure

```
VITASA_Enhanced/
├── train.py                 # Fine-tune PhoBERT (4 ablation configs)
├── demo_normalizer.py       # Interactive demo — Text Normalization
├── demo_losses.py           # Interactive demo — Imbalanced Learning
├── requirements.txt
│
├── text_normalization/      # Contribution #1
│   ├── normalizer.py        # TextNormalizer class
│   ├── data/
│   │   └── teencode_dict.json
│   └── tests/
│
├── imbalanced_learning/     # Contribution #2
│   ├── losses.py            # FocalLoss, WeightedCrossEntropyLoss, compute_class_weights
│   └── tests/
│
├── baseline/
│   └── data/                # ViTASA dataset (mobile / restaurant / hotel)
│
└── experiments/
    └── results/             # Checkpoints + results.json per config
```

---

## Ablation Study (4 configs)

| Config | Text Norm | Loss | Description |
|--------|-----------|------|-------------|
| C1 | ❌ | CE | Baseline reproduction |
| C2 | ✅ | CE | + Text Normalization only |
| C3 | ❌ | Focal | + Imbalanced Learning only |
| C4 | ✅ | Focal | + Both contributions |

---

## Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run training (single config):**
```bash
python train.py --domain mobile --loss focal --normalize --epochs 10
```

**Run all 4 configs × 3 domains on Google Colab:**
```bash
for domain in mobile restaurant hotel; do
    python train.py --domain $domain --loss ce --epochs 10
    python train.py --domain $domain --loss ce --normalize --epochs 10
    python train.py --domain $domain --loss focal --epochs 10
    python train.py --domain $domain --loss focal --normalize --epochs 10
done
```

**Demo modules:**
```bash
python demo_normalizer.py
python demo_losses.py
```

**Run tests:**
```bash
pytest text_normalization/tests/ imbalanced_learning/tests/ -v
```

---

## Dataset

ViTASA dataset — 2,000 annotated social media comments per domain (mobile / restaurant / hotel).
Source: [github.com/kh4nh12/ViTASA](https://github.com/kh4nh12/ViTASA)
