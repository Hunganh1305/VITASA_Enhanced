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
├── train_pair.py            # ✨ NEW — Fine-tune PhoBERT (pair classification) ← USE THIS
├── run_pair_ablation.sh     # ✨ NEW — Run 4 configs × 3 domains
├── summarize_pair_results.py # ✨ NEW — Print ablation results table
├── FORMULATION_FIX_2026-07-20.md  # ✨ NEW — Why BIO tagging was wrong
│
├── train.py                 # ⚠️  OLD — BIO span tagging (deprecated, legacy only)
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

## ⚠️ IMPORTANT UPDATE — 2026-07-20

**Formulation fix detected:** Previous results (train.py) used **BIO span tagging** (wrong),
but paper ViTASA uses **target-aspect pair classification** (correct). This caused massive
gap in macro F1 thresholds.

**→ Use `train_pair.py` instead of `train.py` for all new experiments & baseline comparison.**

Details: [FORMULATION_FIX_2026-07-20.md](FORMULATION_FIX_2026-07-20.md)

---

## Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run training — PAIR CLASSIFICATION (correct formulation):**
```bash
python train_pair.py --domain mobile --loss ce --epochs 3
```

**Run full ablation (4 configs × 3 domains):**
```bash
./run_pair_ablation.sh
python3 summarize_pair_results.py  # Print results table
```

**Legacy — BIO span tagging (not recommended for baseline comparison):**
```bash
python train.py --domain mobile --loss focal --normalize --epochs 10
```
[See warning inside train.py]

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
