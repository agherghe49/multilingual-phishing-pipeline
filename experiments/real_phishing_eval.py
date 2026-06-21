"""
experiments/real_phishing_eval.py

Evaluare cross-domain: clasificator antrenat pe phishing sintetic,
testat pe emailuri de phishing REALE (cybersectony/PhishingEmailDetectionv2.0).

Experimente:
  A) Train sintetic → Test real: generalizare cross-domain
  B) Train real → Test GRPO sintetic: rezistența GRPO la detectori reali

Rulare:
    python experiments/real_phishing_eval.py
    python experiments/real_phishing_eval.py --n-real 500 --n-train 1000
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
OUT_DIR       = OUTPUT_DIR / "real_phishing_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON      = OUT_DIR / "real_phishing_results.json"
OUT_PNG       = OUT_DIR / "real_phishing_comparison.png"


# ── Încărcare date reale ──────────────────────────────────────────────────────

def load_real_dataset(n_phishing: int = 500, n_ham: int = 500,
                      seed: int = 42) -> tuple[list[str], list[int]]:
    """
    Încarcă phishing real din cybersectony dataset.
    label 0 = ham (legitimate email text)
    label 1 = phishing email text
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    print(f"[real] Încarc {REAL_DS_HF} ...")
    ds = load_dataset(REAL_DS_HF, split="train")

    real_phishing = [r["content"] for r in ds
                     if r["label"] == 1 and len(r.get("content", "")) > 100]
    real_ham      = [r["content"] for r in ds
                     if r["label"] == 0 and len(r.get("content", "")) > 100]

    print(f"[real] Phishing disponibil: {len(real_phishing)}, Ham: {len(real_ham)}")

    phish_sample = rng.sample(real_phishing, min(n_phishing, len(real_phishing)))
    ham_sample   = rng.sample(real_ham,      min(n_ham,      len(real_ham)))

    texts  = phish_sample + ham_sample
    labels = [1] * len(phish_sample) + [0] * len(ham_sample)

    return texts, labels


# ── Clasificator ─────────────────────────────────────────────────────────────

def train_and_eval(train_texts, train_labels,
                   test_texts,  test_labels,
                   tag: str = "eval", seed: int = 42) -> dict:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from torch.utils.data import Dataset as TorchDataset
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                  roc_auc_score, average_precision_score)

    class EmailDS(TorchDataset):
        def __init__(self, enc, labels):
            self.enc = enc; self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    tok   = AutoTokenizer.from_pretrained(CLASSIFIER_HF)
    tr_e  = tok(train_texts, truncation=True, padding=True, max_length=256)
    te_e  = tok(test_texts,  truncation=True, padding=True, max_length=256)
    ds_tr = EmailDS(tr_e, train_labels)
    ds_te = EmailDS(te_e, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_HF, num_labels=2
    )
    ckpt = OUT_DIR / f"clf_{tag}"
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
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr,
                      eval_dataset=ds_te)
    trainer.train()

    out   = trainer.predict(ds_te)
    pred  = np.argmax(out.predictions, axis=1)
    proba = torch.softmax(torch.tensor(out.predictions), dim=-1)[:, 1].numpy()

    f1   = f1_score(test_labels, pred, pos_label=1, zero_division=0)
    fnr  = 1.0 - recall_score(test_labels, pred, pos_label=1, zero_division=0)
    prec = precision_score(test_labels, pred, pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(test_labels, proba)
    except Exception:
        auc = 0.0
    try:
        pr_auc = average_precision_score(test_labels, proba)
    except Exception:
        pr_auc = 0.0

    import shutil
    if ckpt.exists():
        shutil.rmtree(ckpt)

    return {
        "f1":        round(float(f1),     4),
        "fnr":       round(float(fnr),    4),
        "precision": round(float(prec),   4),
        "auc_roc":   round(float(auc),    4),
        "pr_auc":    round(float(pr_auc), 4),
        "n_test":    len(test_labels),
        "n_train":   len(train_labels),
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(exp_a: dict, exp_b: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Experiment A: cross-domain generalizare
    categories = ["Sintetic\n(in-domain)", "Real\n(cross-domain)"]
    f1_vals = [exp_a["synthetic_f1"], exp_a["real_f1"]]
    fnr_vals = [exp_a["synthetic_fnr"], exp_a["real_fnr"]]

    x = range(len(categories))
    axes[0].bar(x, f1_vals, color=["#4CAF50", "#FF9800"], alpha=0.85,
                edgecolor="black", linewidth=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("F1 (phishing class)")
    axes[0].set_title("Exp A: Generalizare cross-domain\n(train sintetic → test real)")
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(f1_vals):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")

    ax2 = axes[0].twinx()
    ax2.plot(x, fnr_vals, "rs--", linewidth=2, markersize=8, label="FNR")
    ax2.set_ylabel("FNR", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1.1)
    for i, v in enumerate(fnr_vals):
        ax2.text(i + 0.15, v + 0.02, f"FNR={v:.2f}", color="red", fontsize=9)

    # Experiment B: rezistența GRPO la detector antrenat pe real
    grpo_fnr  = exp_b.get("grpo_fnr", 0)
    std_fnr   = exp_b.get("synthetic_test_fnr", 0)
    cats_b    = ["Phishing sintetic\n(standard)", "Phishing GRPO\n(sintetic)"]
    fnr_b     = [std_fnr, grpo_fnr]

    axes[1].bar(range(2), fnr_b, color=["#2196F3", "#F44336"], alpha=0.85,
                edgecolor="black", linewidth=0.8)
    axes[1].set_xticks(range(2)); axes[1].set_xticklabels(cats_b)
    axes[1].set_ylabel("FNR (False Negative Rate)")
    axes[1].set_title("Exp B: Rezistența GRPO\n(train real → test sintetic/GRPO)")
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate(fnr_b):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")

    plt.suptitle("Evaluare cross-domain: sintetic vs real phishing", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Salvat → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-real",  type=int, default=500)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--skip-b",  action="store_true",
                        help="Skip experiment B (necesită emailuri GRPO generate)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print("REAL PHISHING CROSS-DOMAIN EVAL")
    print(f"n_real={args.n_real}  n_train={args.n_train}  seed={args.seed}")
    print("="*60)

    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    train_all = load_jsonl(OUTPUT_DIR / "train.jsonl")
    test_all  = load_jsonl(OUTPUT_DIR / "test.jsonl")
    rng = random.Random(args.seed)

    # ── Experiment A: train sintetic → test real ──────────────────────────────
    print("\n--- Experiment A: Generalizare cross-domain ---")

    # Train pe sintetic
    phish_tr = rng.sample([r for r in train_all if r["label"] == 1],
                           min(args.n_train, len([r for r in train_all if r["label"]==1])))
    ham_tr   = rng.sample([r for r in train_all if r["label"] == 0],
                           min(args.n_train, len([r for r in train_all if r["label"]==0])))
    train_sub  = phish_tr + ham_tr
    rng.shuffle(train_sub)
    tr_texts  = [r["email_text"] for r in train_sub]
    tr_labels = [r["label"] for r in train_sub]

    # Test sintetic in-domain (baseline)
    phish_te_s = rng.sample([r for r in test_all if r["label"] == 1],
                              min(args.n_real, len([r for r in test_all if r["label"]==1])))
    ham_te_s   = rng.sample([r for r in test_all if r["label"] == 0],
                              min(args.n_real, len([r for r in test_all if r["label"]==0])))
    te_synt = phish_te_s + ham_te_s
    te_synt_texts  = [r["email_text"] for r in te_synt]
    te_synt_labels = [r["label"] for r in te_synt]

    # Test real cross-domain
    real_texts, real_labels = load_real_dataset(args.n_real, args.n_real, args.seed)

    print(f"[A] Train: {len(tr_texts)} sintetic | Test: {len(te_synt_texts)} sintetic "
          f"+ {len(real_texts)} real")

    metrics_synt = train_and_eval(tr_texts, tr_labels, te_synt_texts,
                                   te_synt_labels, tag="synt", seed=args.seed)
    print(f"  In-domain (sintetic):  F1={metrics_synt['f1']:.4f}  FNR={metrics_synt['fnr']:.4f}")

    metrics_real = train_and_eval(tr_texts, tr_labels, real_texts, real_labels,
                                   tag="real", seed=args.seed)
    print(f"  Cross-domain (real):   F1={metrics_real['f1']:.4f}  FNR={metrics_real['fnr']:.4f}")

    exp_a = {
        "synthetic_f1":  metrics_synt["f1"],
        "synthetic_fnr": metrics_synt["fnr"],
        "real_f1":       metrics_real["f1"],
        "real_fnr":      metrics_real["fnr"],
        "details_synthetic": metrics_synt,
        "details_real":      metrics_real,
    }

    # ── Experiment B: train real → test GRPO sintetic ─────────────────────────
    exp_b = {}
    if not args.skip_b:
        print("\n--- Experiment B: Rezistența GRPO la detector antrenat pe real ---")

        # Train pe date reale
        real_tr_texts, real_tr_labels = load_real_dataset(
            n_phishing=args.n_train, n_ham=args.n_train, seed=args.seed + 1
        )

        # Test GRPO emails (din generator_comparison dacă există, altfel skip)
        grpo_cache = OUTPUT_DIR / "generator_comparison" / "grpo_qwen_emails.json"
        if grpo_cache.exists():
            with open(grpo_cache) as f:
                grpo_emails = json.load(f)

            ham_test_real = rng.sample([r for r in test_all if r["label"] == 0],
                                        min(len(grpo_emails), 100))
            ham_test_texts = [r["email_text"] for r in ham_test_real]

            te_std_texts  = te_synt_texts[:100] + [r["email_text"] for r in ham_te_s[:100]]
            te_std_labels = [1]*100 + [0]*100

            te_grpo_texts  = grpo_emails[:100] + ham_test_texts
            te_grpo_labels = [1]*len(grpo_emails[:100]) + [0]*len(ham_test_texts)

            print(f"[B] Train: {len(real_tr_texts)} real | "
                  f"Test: {len(te_std_texts)} std + {len(te_grpo_texts)} GRPO")

            m_std_b  = train_and_eval(real_tr_texts, real_tr_labels,
                                       te_std_texts, te_std_labels,
                                       tag="real_vs_std", seed=args.seed)
            m_grpo_b = train_and_eval(real_tr_texts, real_tr_labels,
                                       te_grpo_texts, te_grpo_labels,
                                       tag="real_vs_grpo", seed=args.seed)

            print(f"  Sintetic standard: F1={m_std_b['f1']:.4f}  FNR={m_std_b['fnr']:.4f}")
            print(f"  GRPO sintetic:     F1={m_grpo_b['f1']:.4f}  FNR={m_grpo_b['fnr']:.4f}")

            exp_b = {
                "synthetic_test_fnr": m_std_b["fnr"],
                "grpo_fnr":           m_grpo_b["fnr"],
                "details_standard":   m_std_b,
                "details_grpo":       m_grpo_b,
            }
        else:
            print(f"[skip B] {grpo_cache} nu există, rulează mai întâi generator_fnr_comparison.py")

    # ── Salvare ───────────────────────────────────────────────────────────────
    output = {
        "config": {"n_real": args.n_real, "n_train": args.n_train,
                   "dataset": REAL_DS_HF, "seed": args.seed},
        "experiment_A": exp_a,
        "experiment_B": exp_b,
        "summary": {
            "cross_domain_f1_drop": round(exp_a["synthetic_f1"] - exp_a["real_f1"], 4),
            "cross_domain_fnr_increase": round(exp_a["real_fnr"] - exp_a["synthetic_fnr"], 4),
        }
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Rezultate → {OUT_JSON}")

    if exp_b:
        plot_results(exp_a, exp_b, OUT_PNG)
        import shutil
        pics = Path(__file__).parent.parent / "raport4" / "pics"
        if pics.exists():
            shutil.copy(OUT_PNG, pics / "real_phishing_comparison.png")

    print(f"\n{'='*60}")
    print("SUMAR:")
    print(f"  In-domain F1:      {exp_a['synthetic_f1']:.4f}")
    print(f"  Cross-domain F1:   {exp_a['real_f1']:.4f}  (drop: {output['summary']['cross_domain_f1_drop']:+.4f})")
    print(f"  Cross-domain FNR:  {exp_a['real_fnr']:.4f}  (vs in-domain: {exp_a['synthetic_fnr']:.4f})")
    print("="*60)


if __name__ == "__main__":
    main()
