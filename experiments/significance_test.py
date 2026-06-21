"""
experiments/significance_test.py

Runs main experiments over N seeds and reports mean ± std.
Covers: scaling laws (XLM-RoBERTa, mDeBERTa-v3, TF-IDF+LR) + adversarial eval FNR.

Usage:
    python experiments/significance_test.py
    python experiments/significance_test.py --seeds 42,123,456,789 --n-values 500,1000,2000
"""

import sys
import json
import argparse
import random
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "significance"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = RESULTS_DIR / "significance_results.json"
OUT_CSV  = RESULTS_DIR / "significance_results.csv"


# ── Import helpers from scaling_laws ─────────────────────────────────────────

def load_data():
    from experiments.scaling_laws import load_jsonl
    from config import OUTPUT_DIR
    train = load_jsonl(OUTPUT_DIR / "train.jsonl")
    test  = load_jsonl(OUTPUT_DIR / "test.jsonl")
    return train, test


def run_one_seed(train, test, model_key: str, n: int, seed: int) -> dict:
    """Runs a single experiment (model, n, seed) and returns metrics."""
    from experiments.scaling_laws import run_lr, run_transformer, balanced_sample

    MODEL_CFGS = {
        "lr": {
            "type": "lr",
            "display": "TF-IDF+LR",
        },
        "xlm-roberta": {
            "type": "transformer",
            "hf_model": "xlm-roberta-base",
            "display": "XLM-RoBERTa",
            "multilingual": True,
            "use_fast": True,
            "bf16_only": False,
            "no_amp": False,
        },
        "mdeberta": {
            "type": "transformer",
            "hf_model": "microsoft/mdeberta-v3-base",
            "display": "mDeBERTa-v3",
            "multilingual": True,
            "use_fast": False,
            "bf16_only": True,
            "no_amp": True,
        },
    }

    cfg = MODEL_CFGS[model_key]
    if cfg["type"] == "lr":
        result = run_lr(train, test, n, seed=seed, tag=cfg["display"])
    else:
        result = run_transformer(
            train, test, n,
            hf_model=cfg["hf_model"],
            display=cfg["display"],
            multilingual=cfg["multilingual"],
            seed=seed,
            use_fast=cfg.get("use_fast", True),
            bf16_only=cfg.get("bf16_only", False),
            no_amp=cfg.get("no_amp", False),
        )
    return result


# ── Adversarial eval with variable seed ──────────────────────────────────────

def run_adversarial_one_seed(seed: int, n_train: int = 1000, n_grpo: int = 100) -> dict:
    """
    Re-runs adversarial evaluation with a different seed (different split).
    Uses pre-generated GRPO emails (adv_grpo_emails.json).
    """
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from torch.utils.data import Dataset as TorchDataset
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from config import OUTPUT_DIR

    random.seed(seed)
    np.random.seed(seed)

    # Load pre-generated GRPO emails
    grpo_path = OUTPUT_DIR / "adversarial_eval" / "adv_grpo_emails.json"
    with open(grpo_path) as f:
        grpo_raw = json.load(f)

    # grpo_raw poate fi list de str sau list de dict
    if isinstance(grpo_raw[0], dict):
        grpo_texts = [r.get("text", r.get("email_text", "")) for r in grpo_raw]
    else:
        grpo_texts = [str(r) for r in grpo_raw]

    random.shuffle(grpo_texts)
    grpo_texts = grpo_texts[:n_grpo]

    # Load standard data
    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    train_all = load_jsonl(OUTPUT_DIR / "train.jsonl")
    test_all  = load_jsonl(OUTPUT_DIR / "test.jsonl")

    # Stratified sampling for training
    rng = random.Random(seed)
    phishing_train = [r for r in train_all if r["label"] == 1]
    ham_train      = [r for r in train_all if r["label"] == 0]
    per_cls = n_train // 2
    train_sub = (rng.sample(phishing_train, min(per_cls, len(phishing_train))) +
                 rng.sample(ham_train, min(per_cls, len(ham_train))))
    rng.shuffle(train_sub)

    # Test: phishing standard
    phishing_test = [r for r in test_all if r["label"] == 1]
    ham_test      = [r for r in test_all if r["label"] == 0]
    n_ham_test = min(len(ham_test), len(phishing_test))
    test_std = rng.sample(phishing_test, min(100, len(phishing_test))) + \
               rng.sample(ham_test, min(100, n_ham_test))

    train_texts  = [r["email_text"] for r in train_sub]
    train_labels = [r["label"] for r in train_sub]
    std_texts    = [r["email_text"] for r in test_std]
    std_labels   = [r["label"] for r in test_std]

    # Test adversarial: GRPO phishing + ham din test
    ham_for_adv  = rng.sample(ham_test, min(n_grpo, len(ham_test)))
    adv_texts    = grpo_texts + [r["email_text"] for r in ham_for_adv]
    adv_labels   = [1] * len(grpo_texts) + [0] * len(ham_for_adv)

    class EmailDataset(TorchDataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels    = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tok_train = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    tok_std   = tokenizer(std_texts,   truncation=True, padding=True, max_length=256)
    tok_adv   = tokenizer(adv_texts,   truncation=True, padding=True, max_length=256)

    ds_train = EmailDataset(tok_train, train_labels)
    ds_std   = EmailDataset(tok_std,   std_labels)
    ds_adv   = EmailDataset(tok_adv,   adv_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=2
    )

    ckpt_dir = RESULTS_DIR / f"clf_seed{seed}"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_train,
                      eval_dataset=ds_std)
    trainer.train()

    def evaluate(ds, labels):
        preds_out = trainer.predict(ds)
        preds = np.argmax(preds_out.predictions, axis=1)
        f1   = f1_score(labels, preds, pos_label=1, zero_division=0)
        fnr  = 1.0 - recall_score(labels, preds, pos_label=1, zero_division=0)
        prec = precision_score(labels, preds, pos_label=1, zero_division=0)
        return {"f1": round(float(f1), 4), "fnr": round(float(fnr), 4),
                "precision": round(float(prec), 4)}

    std_metrics = evaluate(ds_std, std_labels)
    adv_metrics = evaluate(ds_adv, adv_labels)

    import shutil
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    return {
        "seed": seed,
        "standard": std_metrics,
        "adversarial": adv_metrics,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    default="42,123,456")
    parser.add_argument("--n-values", default="100,500,1000,2000")
    parser.add_argument("--models",   default="lr,xlm-roberta,mdeberta")
    parser.add_argument("--skip-scaling",    action="store_true")
    parser.add_argument("--skip-adversarial", action="store_true")
    args = parser.parse_args()

    seeds    = [int(s) for s in args.seeds.split(",")]
    n_values = [int(n) for n in args.n_values.split(",")]
    models   = [m.strip() for m in args.models.split(",")]

    results = {"scaling": {}, "adversarial": []}

    # ── 1. Scaling laws × seeds ──────────────────────────────────────────────
    if not args.skip_scaling:
        print("\n" + "="*60)
        print("SCALING LAWS — statistical significance")
        print("="*60)
        train, test = load_data()

        for model_key in models:
            results["scaling"][model_key] = {}
            for n in n_values:
                seed_results = []
                print(f"\n  [{model_key}] n={n}")
                for seed in seeds:
                    print(f"    seed={seed} ...", end=" ", flush=True)
                    r = run_one_seed(train, test, model_key, n, seed)
                    if r:
                        seed_results.append(r)
                        print(f"F1={r.get('f1',0):.4f} FNR={r.get('fnr',0):.4f}")
                    else:
                        print("skip (error)")

                if seed_results:
                    f1_vals  = [r.get("f1",   0) for r in seed_results]
                    fnr_vals = [r.get("fnr",  0) for r in seed_results]
                    pr_vals  = [r.get("pr_auc", 0) for r in seed_results]
                    results["scaling"][model_key][str(n)] = {
                        "f1_mean":    round(float(np.mean(f1_vals)),  4),
                        "f1_std":     round(float(np.std(f1_vals)),   4),
                        "fnr_mean":   round(float(np.mean(fnr_vals)), 4),
                        "fnr_std":    round(float(np.std(fnr_vals)),  4),
                        "pr_auc_mean":round(float(np.mean(pr_vals)),  4),
                        "pr_auc_std": round(float(np.std(pr_vals)),   4),
                        "n_seeds":    len(seed_results),
                        "per_seed":   [{"seed": s["seed"] if "seed" in s else seeds[i],
                                        "f1": s.get("f1", 0), "fnr": s.get("fnr", 0)}
                                       for i, s in enumerate(seed_results)],
                    }
                    print(f"  → F1={results['scaling'][model_key][str(n)]['f1_mean']:.4f}"
                          f" ±{results['scaling'][model_key][str(n)]['f1_std']:.4f}")

    # ── 2. Adversarial eval × seeds ──────────────────────────────────────────
    if not args.skip_adversarial:
        print("\n" + "="*60)
        print("ADVERSARIAL EVAL — statistical significance")
        print("="*60)
        adv_runs = []
        for seed in seeds:
            print(f"\n  seed={seed} ...")
            r = run_adversarial_one_seed(seed)
            adv_runs.append(r)
            print(f"    standard  F1={r['standard']['f1']:.4f}  FNR={r['standard']['fnr']:.4f}")
            print(f"    adversarial F1={r['adversarial']['f1']:.4f}  FNR={r['adversarial']['fnr']:.4f}")

        adv_fnr = [r["adversarial"]["fnr"] for r in adv_runs]
        std_f1  = [r["standard"]["f1"]     for r in adv_runs]
        results["adversarial"] = {
            "runs": adv_runs,
            "standard_f1_mean":    round(float(np.mean(std_f1)),   4),
            "standard_f1_std":     round(float(np.std(std_f1)),    4),
            "adversarial_fnr_mean": round(float(np.mean(adv_fnr)), 4),
            "adversarial_fnr_std":  round(float(np.std(adv_fnr)),  4),
        }
        print(f"\n  ADVERSARIAL FNR: {results['adversarial']['adversarial_fnr_mean']:.4f}"
              f" ± {results['adversarial']['adversarial_fnr_std']:.4f}")

    # ── Salvare ───────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved → {OUT_JSON}")

    # CSV pentru scaling
    if results["scaling"]:
        import csv
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "n", "f1_mean", "f1_std", "fnr_mean", "fnr_std",
                        "pr_auc_mean", "pr_auc_std", "n_seeds"])
            for model_key, ns in results["scaling"].items():
                for n_str, v in ns.items():
                    w.writerow([model_key, n_str, v["f1_mean"], v["f1_std"],
                                v["fnr_mean"], v["fnr_std"],
                                v["pr_auc_mean"], v["pr_auc_std"], v["n_seeds"]])
        print(f"[OK] CSV → {OUT_CSV}")


if __name__ == "__main__":
    main()
