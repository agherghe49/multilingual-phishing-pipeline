"""
experiments/cross_locale_transfer.py

Cross-locale transferability: train the classifier on a single language
and test on all others.

Scenarios:
  1. Per-locale: train on locale X (phishing+ham), test on all 5 locales
  2. En-only: train on en-US only, test multilingually
  3. Multilingual: train on all 5 locales (reference baseline)

Key question: is multilingual data necessary? Does a model trained on
English generalize to Romanian, German, etc. phishing?

Usage:
    python experiments/cross_locale_transfer.py
    python experiments/cross_locale_transfer.py --quick  # skip full per-locale matrix
"""

import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

TRAIN_JSONL   = OUTPUT_DIR / "train.jsonl"
TEST_JSONL    = OUTPUT_DIR / "test.jsonl"
CLASSIFIER_HF = "xlm-roberta-base"
OUT_DIR       = OUTPUT_DIR / "cross_locale_transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCALES = ["ro-RO", "en-US", "de-DE", "fr-FR", "it-IT"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def filter_by_locale(data: list[dict], locale: str) -> list[dict]:
    return [d for d in data if d.get("locale") == locale]


def filter_by_locales(data: list[dict], locales: list[str]) -> list[dict]:
    return [d for d in data if d.get("locale") in locales]


# ── Classifier ────────────────────────────────────────────────────────────────

def train_classifier(train_texts, train_labels, tag: str, seed: int = 42,
                     max_per_label: int = 1500):
    """Fine-tunes XLM-RoBERTa. Returns (trainer, tokenizer)."""
    import gc
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from torch.utils.data import Dataset

    random.seed(seed)
    gc.collect(); torch.cuda.empty_cache()

    # Stratified sampling by label
    by_lbl = defaultdict(list)
    for i, l in enumerate(train_labels):
        by_lbl[l].append(i)
    n_each = min(max_per_label, min(len(v) for v in by_lbl.values()))
    chosen = []
    for idxs in by_lbl.values():
        random.shuffle(idxs)
        chosen.extend(idxs[:n_each])
    random.shuffle(chosen)
    t_texts  = [train_texts[i]  for i in chosen]
    t_labels = [train_labels[i] for i in chosen]

    print(f"  [clf-{tag}] training on {len(t_texts)} examples "
          f"({sum(t_labels)} phishing + {len(t_labels)-sum(t_labels)} ham)")

    tok = AutoTokenizer.from_pretrained(CLASSIFIER_HF)

    class EmailDS(torch.utils.data.Dataset):
        def __init__(self, texts, labels=None):
            enc = tok(texts, truncation=True, padding=True,
                      max_length=256, return_tensors="pt")
            self.ids   = enc["input_ids"]
            self.mask  = enc["attention_mask"]
            self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            item = {"input_ids": self.ids[i], "attention_mask": self.mask[i]}
            if self.labels is not None: item["labels"] = self.labels[i]
            return item

    train_ds = EmailDS(t_texts, t_labels)
    model    = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_HF, num_labels=2)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / f"clf_{tag}"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()
    return trainer, tok, EmailDS


def eval_on_locale(trainer, tok, EmailDS, test_data, locale: str) -> dict:
    """Evaluates the classifier on test data from a single locale."""
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    loc_data = [d for d in test_data if d.get("locale") == locale]
    if not loc_data:
        return {}

    texts  = [d["email_text"] for d in loc_data]
    labels = [d["label"]      for d in loc_data]

    ds      = EmailDS(texts, labels)
    preds   = trainer.predict(ds)
    probs   = torch.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()
    y_pred  = (probs >= 0.5).astype(int)
    y_true  = np.array(labels)

    n_ph = int(y_true.sum())
    if n_ph == 0 or n_ph == len(y_true):
        return {}

    return {
        "n":           len(y_true),
        "n_phishing":  n_ph,
        "f1_phishing": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision":   round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall":      round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "fnr":         round(float(1 - recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "auc_roc":     round(float(roc_auc_score(y_true, probs)), 4),
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_heatmap(matrix: dict, metric: str, title: str, out_path: Path):
    """matrix[train_locale][test_locale] = metric value"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    locales = LOCALES
    data = np.zeros((len(locales), len(locales)))
    for i, tr in enumerate(locales):
        for j, te in enumerate(locales):
            data[i, j] = matrix.get(tr, {}).get(te, {}).get(metric, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap="RdYlGn" if metric == "f1_phishing" else "RdYlGn_r",
                   vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(locales))); ax.set_xticklabels(locales, fontsize=11)
    ax.set_yticks(range(len(locales))); ax.set_yticklabels(locales, fontsize=11)
    ax.set_xlabel("Test locale", fontsize=12, fontweight="bold")
    ax.set_ylabel("Train locale", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    for i in range(len(locales)):
        for j in range(len(locales)):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if v < 0.4 or v > 0.85 else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[xlocale] Heatmap saved: {out_path}")


def plot_summary(results: dict, out_path: Path):
    """Bar chart: average F1 per training scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(results.keys())
    avg_f1  = []
    avg_fnr = []
    for sc in scenarios:
        vals_f1  = [v["f1_phishing"] for v in results[sc].values() if "f1_phishing" in v]
        vals_fnr = [v["fnr"]         for v in results[sc].values() if "fnr" in v]
        avg_f1.append(round(np.mean(vals_f1) if vals_f1 else 0, 4))
        avg_fnr.append(round(np.mean(vals_fnr) if vals_fnr else 0, 4))

    x = np.arange(len(scenarios))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w/2, avg_f1,  w, label="Avg F1-phishing",  color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + w/2, avg_fnr, w, label="Avg FNR",          color="#F44336", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in scenarios], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Cross-locale transferability: average F1 and FNR per scenario",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(list(bars1) + list(bars2), avg_f1 + avg_fnr):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width()/2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[xlocale] Summary plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Skip full per-locale evaluation (en-only vs. multilingual only)")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    import gc
    import torch

    print("[xlocale] Loading train/test data...")
    train_data = load_jsonl(TRAIN_JSONL)
    test_data  = load_jsonl(TEST_JSONL)
    print(f"[xlocale] Train: {len(train_data)} | Test: {len(test_data)}")

    for loc in LOCALES:
        n = sum(1 for d in test_data if d.get("locale") == loc)
        print(f"  Test {loc}: {n}")

    # ──────────────────────────────────────────────────────────────────────
    # SCENARIO 1: En-US only
    # ──────────────────────────────────────────────────────────────────────
    print("\n[xlocale] === SCENARIO 1: Training on en-US only ===")
    en_train = filter_by_locale(train_data, "en-US")
    en_texts  = [d["email_text"] for d in en_train]
    en_labels = [d["label"]      for d in en_train]
    print(f"[xlocale] en-US train: {len(en_texts)} "
          f"({sum(en_labels)} phishing + {len(en_labels)-sum(en_labels)} ham)")

    trainer_en, tok_en, DS_en = train_classifier(en_texts, en_labels, "en_only", args.seed)

    en_only_results = {}
    for loc in LOCALES:
        r = eval_on_locale(trainer_en, tok_en, DS_en, test_data, loc)
        if r:
            en_only_results[loc] = r
            print(f"  en-US → {loc}: F1={r['f1_phishing']:.4f} FNR={r['fnr']:.4f}")

    trainer_en.model.cpu()
    del trainer_en
    gc.collect(); torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────
    # SCENARIO 2: Multilingual (all 5 locales) — reference baseline
    # ──────────────────────────────────────────────────────────────────────
    print("\n[xlocale] === SCENARIO 2: Training multilingual (all locales) ===")
    all_texts  = [d["email_text"] for d in train_data]
    all_labels = [d["label"]      for d in train_data]

    trainer_multi, tok_multi, DS_multi = train_classifier(
        all_texts, all_labels, "multilingual", args.seed)

    multi_results = {}
    for loc in LOCALES:
        r = eval_on_locale(trainer_multi, tok_multi, DS_multi, test_data, loc)
        if r:
            multi_results[loc] = r
            print(f"  multi → {loc}: F1={r['f1_phishing']:.4f} FNR={r['fnr']:.4f}")

    trainer_multi.model.cpu()
    del trainer_multi
    gc.collect(); torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────
    # SCENARIO 3: Per-locale (skip if --quick)
    # ──────────────────────────────────────────────────────────────────────
    per_locale_matrix = {}  # matrix[train_locale][test_locale] = metrics

    if not args.quick:
        print("\n[xlocale] === SCENARIO 3: Per-locale cross-transfer matrix ===")
        for train_loc in LOCALES:
            print(f"\n[xlocale] Training on {train_loc}...")
            loc_train = filter_by_locale(train_data, train_loc)
            loc_texts  = [d["email_text"] for d in loc_train]
            loc_labels = [d["label"]      for d in loc_train]

            trainer_loc, tok_loc, DS_loc = train_classifier(
                loc_texts, loc_labels, f"train_{train_loc.replace('-','_')}", args.seed)

            per_locale_matrix[train_loc] = {}
            for test_loc in LOCALES:
                r = eval_on_locale(trainer_loc, tok_loc, DS_loc, test_data, test_loc)
                if r:
                    per_locale_matrix[train_loc][test_loc] = r
                    print(f"  {train_loc} → {test_loc}: "
                          f"F1={r['f1_phishing']:.4f} FNR={r['fnr']:.4f}")

            trainer_loc.model.cpu()
            del trainer_loc
            gc.collect(); torch.cuda.empty_cache()

    # ── Print comparison table ────────────────────────────────────────────
    print("\n" + "="*70)
    print("CROSS-LOCALE TRANSFERABILITY — Summary")
    print("="*70)
    print(f"{'Test locale':<10} {'en-US F1':>10} {'en-US FNR':>11} "
          f"{'Multi F1':>10} {'Multi FNR':>11}")
    print("-"*55)
    for loc in LOCALES:
        en_f1  = en_only_results.get(loc, {}).get("f1_phishing", 0)
        en_fnr = en_only_results.get(loc, {}).get("fnr", 0)
        ml_f1  = multi_results.get(loc, {}).get("f1_phishing", 0)
        ml_fnr = multi_results.get(loc, {}).get("fnr", 0)
        print(f"{loc:<10} {en_f1:>10.4f} {en_fnr:>11.4f} "
              f"{ml_f1:>10.4f} {ml_fnr:>11.4f}")
    print("="*70)

    en_avg_f1  = np.mean([v.get("f1_phishing",0) for v in en_only_results.values()])
    ml_avg_f1  = np.mean([v.get("f1_phishing",0) for v in multi_results.values()])
    print(f"\nAvg F1 en-only:        {en_avg_f1:.4f}")
    print(f"Avg F1 multilingual:   {ml_avg_f1:.4f}")
    print(f"Multilingual gap:      {ml_avg_f1-en_avg_f1:+.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    summary_results = {
        "En-US only":    en_only_results,
        "Multilingual":  multi_results,
    }
    plot_summary(summary_results, OUT_DIR / "cross_locale_summary.png")

    if per_locale_matrix:
        plot_heatmap(per_locale_matrix, "f1_phishing",
                     "F1-phishing: train locale (Y) vs. test locale (X)",
                     OUT_DIR / "cross_locale_f1_heatmap.png")
        plot_heatmap(per_locale_matrix, "fnr",
                     "FNR: train locale (Y) vs. test locale (X)",
                     OUT_DIR / "cross_locale_fnr_heatmap.png")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "config":           {"classifier": CLASSIFIER_HF, "seed": args.seed},
        "en_only":          en_only_results,
        "multilingual":     multi_results,
        "per_locale_matrix": per_locale_matrix,
        "summary": {
            "en_avg_f1":  round(float(en_avg_f1), 4),
            "multi_avg_f1": round(float(ml_avg_f1), 4),
            "gap":        round(float(ml_avg_f1 - en_avg_f1), 4),
        },
    }
    out_json = OUT_DIR / "cross_locale_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[xlocale] Results saved: {out_json}")


if __name__ == "__main__":
    main()
