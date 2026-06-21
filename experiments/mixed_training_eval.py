"""
experiments/mixed_training_eval.py

Demonstrează că mixed training (sintetic + real) rezolvă gap-ul cross-domain.

Setup:
  - Sintetic-only:  train pe 1000/clasă sintetic  → test sintetic + test real
  - Mixed:          train pe 1000/clasă sintetic + 400/clasă real → test sintetic + test real

Finding așteptat:
  - Sintetic-only:  F1_sintetic≈1.0,  F1_real≈0.09
  - Mixed:          F1_sintetic≈0.99, F1_real≈0.70-0.85

Rulare:
    python experiments/mixed_training_eval.py
    python experiments/mixed_training_eval.py --n-real-train 200 --n-real-test 200
"""

import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

CLASSIFIER_HF = "xlm-roberta-base"
REAL_DS_HF    = "cybersectony/PhishingEmailDetectionv2.0"
OUT_DIR       = OUTPUT_DIR / "mixed_training"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON      = OUT_DIR / "mixed_training_results.json"
OUT_PNG       = OUT_DIR / "mixed_training_comparison.png"


# ── Date ─────────────────────────────────────────────────────────────────────

def load_synthetic(n_train: int, n_test: int, seed: int):
    def load_jsonl(p):
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]

    rng        = random.Random(seed)
    train_all  = load_jsonl(OUTPUT_DIR / "train.jsonl")
    test_all   = load_jsonl(OUTPUT_DIR / "test.jsonl")

    ph_tr = rng.sample([r for r in train_all if r["label"] == 1],
                        min(n_train, sum(1 for r in train_all if r["label"] == 1)))
    hm_tr = rng.sample([r for r in train_all if r["label"] == 0],
                        min(n_train, sum(1 for r in train_all if r["label"] == 0)))
    ph_te = rng.sample([r for r in test_all  if r["label"] == 1],
                        min(n_test,  sum(1 for r in test_all  if r["label"] == 1)))
    hm_te = rng.sample([r for r in test_all  if r["label"] == 0],
                        min(n_test,  sum(1 for r in test_all  if r["label"] == 0)))

    train = [(r["email_text"], r["label"]) for r in ph_tr + hm_tr]
    test  = [(r["email_text"], r["label"]) for r in ph_te + hm_te]
    rng.shuffle(train)
    return train, test


def load_real(n_train: int, n_test: int, seed: int):
    from datasets import load_dataset
    rng = random.Random(seed + 100)

    print(f"[real] Încarc {REAL_DS_HF} ...")
    ds = load_dataset(REAL_DS_HF, split="train")

    phishing = [(r["content"], 1) for r in ds
                if r["label"] == 1 and len(r.get("content", "")) > 80]
    ham      = [(r["content"], 0) for r in ds
                if r["label"] == 0 and len(r.get("content", "")) > 80]

    print(f"[real] Disponibil: {len(phishing)} phishing, {len(ham)} ham")

    n_ph_tr = min(n_train, len(phishing) - n_test)
    n_hm_tr = min(n_train, len(ham)      - n_test)

    ph_shuffled = rng.sample(phishing, len(phishing))
    hm_shuffled = rng.sample(ham,      len(ham))

    ph_train, ph_test = ph_shuffled[:n_ph_tr], ph_shuffled[n_ph_tr:n_ph_tr + n_test]
    hm_train, hm_test = hm_shuffled[:n_hm_tr], hm_shuffled[n_hm_tr:n_hm_tr + n_test]

    train = ph_train + hm_train
    test  = ph_test  + hm_test
    rng.shuffle(train)
    return train, test


# ── Clasificator ─────────────────────────────────────────────────────────────

def train_and_eval(train_pairs, test_synt_pairs, test_real_pairs,
                   tag: str = "clf", seed: int = 42) -> dict:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from torch.utils.data import Dataset as TorchDataset
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

    class DS(TorchDataset):
        def __init__(self, enc, labels):
            self.enc = enc; self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    tok = AutoTokenizer.from_pretrained(CLASSIFIER_HF)

    tr_texts, tr_labels = zip(*train_pairs)
    enc_tr = tok(list(tr_texts), truncation=True, padding=True, max_length=256)
    ds_tr  = DS(enc_tr, list(tr_labels))

    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_HF, num_labels=2)
    ckpt  = OUT_DIR / f"clf_{tag}"

    args = TrainingArguments(
        output_dir=str(ckpt),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr)
    print(f"  [{tag}] Antrenare pe {len(tr_labels)} exemple ...")
    trainer.train()

    def evaluate(pairs, name):
        texts, labels = zip(*pairs)
        enc  = tok(list(texts), truncation=True, padding=True, max_length=256)
        ds   = DS(enc, list(labels))
        out  = trainer.predict(ds)
        pred = np.argmax(out.predictions, axis=1)
        prob = torch.softmax(torch.tensor(out.predictions), dim=-1)[:, 1].numpy()

        f1   = f1_score(list(labels), pred, pos_label=1, zero_division=0)
        fnr  = 1.0 - recall_score(list(labels), pred, pos_label=1, zero_division=0)
        prec = precision_score(list(labels), pred, pos_label=1, zero_division=0)
        try:    auc = roc_auc_score(list(labels), prob)
        except: auc = 0.5
        print(f"  [{tag}] {name}: F1={f1:.4f}  FNR={fnr:.4f}  AUC={auc:.4f}")
        return {"f1": round(float(f1), 4), "fnr": round(float(fnr), 4),
                "precision": round(float(prec), 4), "auc_roc": round(float(auc), 4),
                "n": len(labels)}

    m_synt = evaluate(test_synt_pairs, "sintetic test")
    m_real = evaluate(test_real_pairs, "real test")

    import shutil
    if ckpt.exists(): shutil.rmtree(ckpt)

    return {"synthetic": m_synt, "real": m_real,
            "n_train": len(tr_labels), "tag": tag}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(synt_only: dict, mixed: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics  = ["F1 (sintetic)", "F1 (real)", "FNR (real)"]
    so_vals  = [synt_only["synthetic"]["f1"], synt_only["real"]["f1"],
                synt_only["real"]["fnr"]]
    mix_vals = [mixed["synthetic"]["f1"], mixed["real"]["f1"],
                mixed["real"]["fnr"]]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, so_vals,  width, label="Sintetic-only",
                color="#2196F3", alpha=0.85, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, mix_vals, width, label="Mixed (sint + real)",
                color="#4CAF50", alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Valoare metrică")
    ax.set_title("Mixed Training vs Sintetic-Only\n(XLM-RoBERTa, n_train=1000 sint + 400 real)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, color="#1565C0")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, color="#2E7D32")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-synt-train", type=int, default=1000)
    parser.add_argument("--n-real-train", type=int, default=400)
    parser.add_argument("--n-synt-test",  type=int, default=200)
    parser.add_argument("--n-real-test",  type=int, default=200)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print("MIXED TRAINING EVALUATION")
    print(f"n_synt_train={args.n_synt_train}  n_real_train={args.n_real_train}")
    print(f"n_synt_test={args.n_synt_test}    n_real_test={args.n_real_test}")
    print("="*60)

    synt_train, synt_test = load_synthetic(args.n_synt_train, args.n_synt_test, args.seed)
    real_train, real_test = load_real(args.n_real_train, args.n_real_test, args.seed)

    print(f"\n[data] Sintetic train: {len(synt_train)}  |  Sintetic test: {len(synt_test)}")
    print(f"[data] Real train: {len(real_train)}      |  Real test: {len(real_test)}")

    # ── Experiment 1: Sintetic-only ───────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Experiment 1: Sintetic-only")
    synt_only = train_and_eval(
        train_pairs       = synt_train,
        test_synt_pairs   = synt_test,
        test_real_pairs   = real_test,
        tag  = "synt_only",
        seed = args.seed,
    )

    # ── Experiment 2: Mixed ────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Experiment 2: Mixed (sintetic + real)")
    mixed_train = synt_train + real_train
    random.shuffle(mixed_train)
    mixed = train_and_eval(
        train_pairs       = mixed_train,
        test_synt_pairs   = synt_test,
        test_real_pairs   = real_test,
        tag  = "mixed",
        seed = args.seed,
    )

    # ── Rezultate ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("REZULTATE FINALE")
    print(f"{'─'*60}")
    print(f"{'Experiment':25} | {'F1 sintetic':>12} | {'F1 real':>9} | {'FNR real':>9}")
    print(f"{'─'*60}")
    print(f"{'Sintetic-only':25} | {synt_only['synthetic']['f1']:>12.4f} | "
          f"{synt_only['real']['f1']:>9.4f} | {synt_only['real']['fnr']:>9.4f}")
    print(f"{'Mixed (+400 real)':25} | {mixed['synthetic']['f1']:>12.4f} | "
          f"{mixed['real']['f1']:>9.4f} | {mixed['real']['fnr']:>9.4f}")
    print(f"{'─'*60}")
    delta_f1_real = mixed["real"]["f1"] - synt_only["real"]["f1"]
    delta_fnr     = mixed["real"]["fnr"] - synt_only["real"]["fnr"]
    print(f"{'Δ Mixed vs Synt-only':25} | {'':>12} | {delta_f1_real:>+9.4f} | {delta_fnr:>+9.4f}")
    print("="*60)

    output = {
        "config": vars(args),
        "synt_only": synt_only,
        "mixed":     mixed,
        "delta": {
            "f1_synt":  round(mixed["synthetic"]["f1"] - synt_only["synthetic"]["f1"], 4),
            "f1_real":  round(delta_f1_real, 4),
            "fnr_real": round(delta_fnr,     4),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] → {OUT_JSON}")

    plot(synt_only, mixed, OUT_PNG)

    import shutil
    pics = Path(__file__).parent.parent / "raport4" / "pics"
    if pics.exists():
        shutil.copy(OUT_PNG, pics / "mixed_training_comparison.png")
        print(f"[copy] → raport4/pics/mixed_training_comparison.png")


if __name__ == "__main__":
    main()
