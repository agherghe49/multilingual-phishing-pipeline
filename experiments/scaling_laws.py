"""
experiments/scaling_laws.py

Experiment scaling laws: măsoară cum evoluează performanța detectorului
de phishing în funcție de numărul de date de antrenament.

Modele suportate:
  lr            — TF-IDF + Logistic Regression (multilingv, baseline)
  distilbert-ml — distilbert-base-multilingual-cased (multilingv)
  xlm-roberta   — xlm-roberta-base (multilingv, standard 2019)
  mdeberta      — microsoft/mdeberta-v3-base (multilingv, SOTA 2023)
  modernbert    — answerdotai/ModernBERT-base (English only, SOTA 2024)
  distilbert-en — distilbert-base-uncased (English only, legacy)

Modele multilingve rulează pe întreg test set-ul (en/ro/de/fr/it).
Modele English-only rulează automat pe subsetul en-US.

Rulare:
  python experiments/scaling_laws.py                        # lr only
  python experiments/scaling_laws.py --model multilingual   # ml modele
  python experiments/scaling_laws.py --model english        # en modele
  python experiments/scaling_laws.py --model all            # toate
  python experiments/scaling_laws.py --model xlm-roberta,mdeberta

Preset-uri:
  lr            → [lr]
  bert          → [distilbert-ml]                           (backward compat)
  both          → [lr, distilbert-ml]                       (backward compat)
  multilingual  → [lr, distilbert-ml, xlm-roberta, mdeberta]
  english       → [lr, distilbert-en, modernbert]
  all           → toate modelele
"""

import sys
import csv
import json
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

TRAIN_PATH  = OUTPUT_DIR / "train.jsonl"
TEST_PATH   = OUTPUT_DIR / "test.jsonl"
RESULTS_DIR = OUTPUT_DIR / "scaling_laws"

# ── Registry modele ───────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "lr": {
        "type":         "lr",
        "multilingual": True,
        "display":      "TF-IDF+LR",
        "hf_model":     None,
    },
    "distilbert-ml": {
        "type":         "bert",
        "multilingual": True,
        "display":      "mDistilBERT",
        "hf_model":     "distilbert-base-multilingual-cased",
    },
    "xlm-roberta": {
        "type":         "bert",
        "multilingual": True,
        "display":      "XLM-RoBERTa",
        "hf_model":     "xlm-roberta-base",
    },
    "modernbert": {
        "type":         "bert",
        "multilingual": False,
        "display":      "ModernBERT",
        "hf_model":     "answerdotai/ModernBERT-base",
    },
    "mdeberta": {
        "type":         "bert",
        "multilingual": True,
        "display":      "mDeBERTa-v3",
        "hf_model":     "microsoft/mdeberta-v3-base",
        "use_fast":     False,   # spm.model incompatibil cu fast tokenizer (tiktoken bug)
        "bf16_only":    False,   # FP32 — head de clasificare nou init poate exploda în BF16/FP16
        "no_amp":       True,    # dezactivează AMP complet pentru stabilitate
    },
    "distilbert-en": {
        "type":         "bert",
        "multilingual": False,
        "display":      "DistilBERT-EN",
        "hf_model":     "distilbert-base-uncased",
    },
}

PRESETS: dict[str, list[str]] = {
    "lr":           ["lr"],
    "bert":         ["distilbert-ml"],                              # backward compat
    "both":         ["lr", "distilbert-ml"],                        # backward compat
    "multilingual": ["lr", "distilbert-ml", "xlm-roberta", "mdeberta"],
    "english":      ["lr", "distilbert-en", "modernbert"],
    "all":          list(MODEL_REGISTRY.keys()),
}


# ── Metrici ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true:       list[int],
    y_pred:       list[int],
    y_proba:      "np.ndarray",
    test_records: list[dict],
    n:            int,
    model_tag:    str,
    scope:        str,
) -> dict:
    """
    Calculează set complet de metrici pentru clasificare binară phishing.

    Metrici returnate:
      f1          — F1 binary pentru clasa phishing (label=1)
      f1_macro    — media F1 pe ambele clase (phishing + ham)
      precision   — precizie pentru clasa phishing
      recall      — recall pentru clasa phishing (= rata de detecție)
      fnr         — False Negative Rate = 1 - recall (rata atacurilor ratate)
      auc_roc     — AUC-ROC, threshold-independent
      pr_auc      — AUC Precision-Recall, mai informativ pt securitate
      accuracy    — acuratețe globală
      fnr_t30/40/50/60/70 — FNR la diferite praguri de decizie
      thresh_p95  — pragul optim unde precision >= 0.95
      fnr_at_p95  — FNR minim obținut la thresh_p95
      f1_<locale> — F1 per locale (dacă există ambele clase în locale)
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, accuracy_score, average_precision_score,
    )
    from collections import defaultdict

    y_proba_arr = np.asarray(y_proba)
    y_true_arr  = np.asarray(y_true)

    # NaN/Inf poate apărea la modele instabile (e.g. mDeBERTa BF16 n mic) — fallback la 0.5
    if not np.isfinite(y_proba_arr).all():
        print(f"  [warn] {model_tag} n={n}: probabilități NaN/Inf → fallback 0.5")
        y_proba_arr = np.where(np.isfinite(y_proba_arr), y_proba_arr, 0.5)
        y_pred      = (y_proba_arr >= 0.5).astype(int).tolist()

    recall = recall_score(y_true, y_pred, zero_division=0)
    result = {
        "n":         n,
        "model":     model_tag,
        "scope":     scope,
        "f1":        round(f1_score(y_true, y_pred, zero_division=0),                    4),
        "f1_macro":  round(f1_score(y_true, y_pred, average="macro", zero_division=0),   4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0),             4),
        "recall":    round(recall,                                                        4),
        "fnr":       round(1.0 - recall,                                                 4),
        "auc_roc":   round(roc_auc_score(y_true, y_proba_arr),                           4),
        "pr_auc":    round(average_precision_score(y_true, y_proba_arr),                 4),
        "accuracy":  round(accuracy_score(y_true, y_pred),                               4),
    }

    # Threshold sweep — FNR la praguri fixe
    for t_int in [30, 40, 50, 60, 70]:
        t = t_int / 100.0
        y_pred_t = (y_proba_arr >= t).astype(int)
        fnr_t = 1.0 - recall_score(y_true_arr, y_pred_t, zero_division=0)
        result[f"fnr_t{t_int}"] = round(fnr_t, 4)

    # Operating point: prag optim unde precision >= 0.95
    best_thresh   = None
    best_fnr_p95  = None
    for t_raw in np.arange(0.01, 1.0, 0.01):
        y_pred_t = (y_proba_arr >= t_raw).astype(int)
        if y_pred_t.sum() == 0:
            continue
        prec_t = precision_score(y_true_arr, y_pred_t, zero_division=0)
        if prec_t >= 0.95:
            fnr_t = 1.0 - recall_score(y_true_arr, y_pred_t, zero_division=0)
            if best_fnr_p95 is None or fnr_t < best_fnr_p95:
                best_fnr_p95 = fnr_t
                best_thresh  = round(float(t_raw), 2)

    result["thresh_p95"] = best_thresh
    result["fnr_at_p95"] = round(best_fnr_p95, 4) if best_fnr_p95 is not None else None

    # Per-locale F1
    by_locale: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for rec, pred in zip(test_records, y_pred):
        loc = rec.get("locale", "unknown")
        by_locale[loc][0].append(rec["label"])
        by_locale[loc][1].append(pred)

    for loc, (yt, yp) in sorted(by_locale.items()):
        if len(set(yt)) < 2:
            result[f"f1_{loc}"] = None
        else:
            result[f"f1_{loc}"] = round(f1_score(yt, yp, zero_division=0), 4)

    return result


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fișier negăsit: {path}\n"
            "Rulează mai întâi: python experiments/split_dataset.py"
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_locale(records: list[dict], locale: str = "en-US") -> list[dict]:
    return [r for r in records if r.get("locale") == locale]


def extract_xy(records: list[dict]) -> tuple[list[str], list[int]]:
    return [r["email_text"] for r in records], [r["label"] for r in records]


def balanced_sample(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng      = random.Random(seed)
    phishing = [r for r in records if r.get("label") == 1]
    ham      = [r for r in records if r.get("label") == 0]
    per_class = n // 2
    combined = (
        rng.sample(phishing, min(per_class, len(phishing))) +
        rng.sample(ham,      min(per_class, len(ham)))
    )
    rng.shuffle(combined)
    return combined


# ── Classifier TF-IDF+LR ─────────────────────────────────────────────────────

def run_lr(
    train_records: list[dict],
    test_records:  list[dict],
    n:             int,
    seed:          int = 42,
    tag:           str = "TF-IDF+LR",
) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    subset      = balanced_sample(train_records, n, seed)
    X_tr, y_tr  = extract_xy(subset)
    X_te, y_te  = extract_xy(test_records)

    vec    = TfidfVectorizer(max_features=50_000, sublinear_tf=True, ngram_range=(1, 2))
    X_tr_v = vec.fit_transform(X_tr)
    X_te_v = vec.transform(X_te)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
    clf.fit(X_tr_v, y_tr)

    y_pred  = clf.predict(X_te_v)
    y_proba = clf.predict_proba(X_te_v)[:, 1]

    return compute_metrics(y_te, y_pred, y_proba, test_records, n, tag, "multilingual")


# ── Classifier Transformer ────────────────────────────────────────────────────

def run_transformer(
    train_records: list[dict],
    test_records:  list[dict],
    n:             int,
    hf_model:      str,
    display:       str,
    multilingual:  bool,
    seed:          int  = 42,
    use_fast:      bool = True,
    bf16_only:     bool = False,
    no_amp:        bool = False,
) -> dict:
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer,
        )
        from datasets import Dataset as HFDataset
    except ImportError:
        print(f"  [warn] transformers/torch neinstalate, skip {display}")
        return {}

    # English-only models → filtrăm pe en-US
    scope = "multilingual"
    if not multilingual:
        train_records = filter_locale(train_records, "en-US")
        test_records  = filter_locale(test_records,  "en-US")
        scope = "en-US"
        print(f"  [{display}] English-only → subset en-US: "
              f"train={len(train_records)}, test={len(test_records)}")

    if len([r for r in train_records if r['label']==0]) == 0 or \
       len([r for r in test_records  if r['label']==0]) == 0:
        print(f"  [skip] {display}: test set fără ham pentru scope={scope}")
        return {}

    subset      = balanced_sample(train_records, n, seed)
    X_tr, y_tr  = extract_xy(subset)
    X_te, y_te  = extract_xy(test_records)

    max_len = 512
    tok = AutoTokenizer.from_pretrained(hf_model, use_fast=use_fast)

    def tokenize(texts: list[str]) -> dict:
        return tok(texts, truncation=True, padding=True, max_length=max_len)

    tr_ds = HFDataset.from_dict({**tokenize(X_tr), "labels": y_tr})
    te_ds = HFDataset.from_dict({**tokenize(X_te), "labels": y_te})

    load_dtype = "float32" if no_amp else "auto"
    model_hf = AutoModelForSequenceClassification.from_pretrained(
        hf_model, num_labels=2, torch_dtype=load_dtype, ignore_mismatched_sizes=True,
    )

    safe_name = hf_model.replace("/", "_")
    training_args = TrainingArguments(
        output_dir                  = str(RESULTS_DIR / f"{safe_name}_n{n}"),
        num_train_epochs            = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        learning_rate               = 2e-5,
        weight_decay                = 0.01,
        eval_strategy               = "epoch",
        save_strategy               = "no",
        load_best_model_at_end      = False,
        fp16                        = torch.cuda.is_available() and not bf16_only and not no_amp,
        bf16                        = torch.cuda.is_available() and bf16_only and not no_amp,
        seed                        = seed,
        report_to                   = "none",
        logging_steps               = 50,
    )

    trainer = Trainer(
        model         = model_hf,
        args          = training_args,
        train_dataset = tr_ds,
        eval_dataset  = te_ds,
    )
    trainer.train()

    preds_out   = trainer.predict(te_ds)
    logits      = preds_out.predictions
    y_pred      = logits.argmax(axis=-1).tolist()
    y_proba     = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    return compute_metrics(y_te, y_pred, y_proba, test_records, n, display, scope)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(results: list[dict], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib neinstalat, skip plot")
        return

    ml_results = [r for r in results if r.get("scope") != "en-US"]
    en_results = [r for r in results if r.get("scope") == "en-US"]
    has_en     = bool(en_results)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

    def _add_curves(ax, group, metric_key, invert=False):
        """Plotează curbele unui metric pentru toate modelele dintr-un grup."""
        by_model = defaultdict(list)
        for r in group:
            by_model[r["model"]].append(r)
        for i, (model_tag, pts) in enumerate(sorted(by_model.items())):
            pts.sort(key=lambda x: x["n"])
            ns  = [p["n"]             for p in pts]
            vals= [p.get(metric_key, None) for p in pts]
            if any(v is None for v in vals):
                continue
            ax.plot(ns, vals, marker="o", label=model_tag, color=colors[i % len(colors)])
        ax.set_xscale("log")
        ax.set_xlabel("Nr. date antrenament")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if invert:
            ax.set_ylim(-0.05, 1.05)
            ax.invert_yaxis()
        else:
            ax.set_ylim(-0.05, 1.05)

    # Layout: 2 rânduri × 2 coloane pentru multilingv + coloană extra pentru EN
    ncols = 3 if has_en else 2
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 9))

    # Rândul 1: F1 macro + FNR
    axes[0][0].set_title("F1 Macro (ambele clase) — Multilingv")
    _add_curves(axes[0][0], ml_results, "f1_macro")
    axes[0][0].set_ylabel("F1 Macro")

    axes[0][1].set_title("FNR — Rata atacurilor ratate — Multilingv")
    _add_curves(axes[0][1], ml_results, "fnr")
    axes[0][1].set_ylabel("FNR (↓ mai bun)")
    axes[0][1].yaxis.set_inverted(False)
    # Adaugă linie de referință la 0 (ideal)
    axes[0][1].axhline(0, color="gray", linestyle="--", alpha=0.4)

    # Rândul 2: F1 binary + PR-AUC
    axes[1][0].set_title("F1 Phishing (binary) — Multilingv")
    _add_curves(axes[1][0], ml_results, "f1")
    axes[1][0].set_ylabel("F1 Phishing")

    axes[1][1].set_title("PR-AUC — Multilingv")
    _add_curves(axes[1][1], ml_results, "pr_auc")
    axes[1][1].set_ylabel("PR-AUC")

    # Coloana English-only
    if has_en:
        axes[0][2].set_title("F1 Macro — English only")
        _add_curves(axes[0][2], en_results, "f1_macro")
        axes[0][2].set_ylabel("F1 Macro")

        axes[1][2].set_title("FNR — English only")
        _add_curves(axes[1][2], en_results, "fnr")
        axes[1][2].set_ylabel("FNR (↓ mai bun)")
        axes[1][2].axhline(0, color="gray", linestyle="--", alpha=0.4)

    plt.suptitle("Scaling Laws — Detector Phishing Email Sintetic", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot] {output_path}")


# ── Salvare CSV (append-safe) ─────────────────────────────────────────────────

def save_results(results: list[dict], csv_path: Path, overwrite: bool = False) -> None:
    if not results:
        return
    mode      = "w" if overwrite else "a"
    write_hdr = overwrite or not csv_path.exists()
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_hdr:
            writer.writeheader()
        writer.writerows(results)
    print(f"[CSV] {csv_path} ({len(results)} rânduri {'scrise' if overwrite else 'adăugate'})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_models(model_arg: str) -> list[str]:
    """Parsează --model: preset sau listă separată prin virgulă."""
    if model_arg in PRESETS:
        return PRESETS[model_arg]
    keys = [m.strip() for m in model_arg.split(",")]
    unknown = [k for k in keys if k not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Modele necunoscute: {unknown}. Disponibile: {list(MODEL_REGISTRY)}")
    return keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaling laws pentru detectorul de phishing"
    )
    parser.add_argument(
        "--steps", type=str, default="100,250,500,1000,2000,5000",
        help="Subset-uri de antrenament, separate prin virgulă",
    )
    parser.add_argument(
        "--model", type=str, default="lr",
        help=(
            "Preset sau listă de modele: "
            "lr | bert | both | multilingual | english | all | "
            "<model1>,<model2>,..."
        ),
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Suprascrie CSV existent (implicit: append)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    steps       = [int(s.strip()) for s in args.steps.split(",")]
    model_keys  = parse_models(args.model)

    train_all = load_jsonl(TRAIN_PATH)
    test_all  = load_jsonl(TEST_PATH)

    print(f"[scaling] Train: {len(train_all)}, Test: {len(test_all)}")
    print(f"[scaling] Steps: {steps}")
    print(f"[scaling] Modele: {model_keys}\n")

    csv_path = RESULTS_DIR / "scaling_results.csv"
    all_results: list[dict] = []

    for model_key in model_keys:
        cfg = MODEL_REGISTRY[model_key]
        print(f"\n{'='*60}")
        print(f"Model: {cfg['display']}  |  "
              f"scope: {'multilingv' if cfg['multilingual'] else 'en-US only'}")
        print(f"{'='*60}")

        model_results = []

        for n in steps:
            # Verifică dacă avem suficiente date (cu filtrare pentru en-only)
            check_records = train_all
            if not cfg["multilingual"]:
                check_records = filter_locale(train_all, "en-US")
            ph_n  = sum(1 for r in check_records if r["label"] == 1)
            ham_n = sum(1 for r in check_records if r["label"] == 0)
            if n > min(ph_n, ham_n) * 2:
                print(f"  [skip] n={n} > date disponibile (ph={ph_n}, ham={ham_n})")
                continue

            print(f"\n--- n={n} ---")

            if cfg["type"] == "lr":
                r = run_lr(train_all, test_all, n, args.seed, tag=cfg["display"])
                model_results.append(r)
                print(f"  {cfg['display']:20s}  F1={r['f1']:.4f}  FNR={r['fnr']:.4f}  PR-AUC={r['pr_auc']:.4f}")

            elif cfg["type"] == "bert":
                r = run_transformer(
                    train_records = train_all,
                    test_records  = test_all,
                    n             = n,
                    hf_model      = cfg["hf_model"],
                    display       = cfg["display"],
                    multilingual  = cfg["multilingual"],
                    seed          = args.seed,
                    use_fast      = cfg.get("use_fast", True),
                    bf16_only     = cfg.get("bf16_only", False),
                    no_amp        = cfg.get("no_amp", False),
                )
                if r:
                    model_results.append(r)
                    print(f"  {cfg['display']:20s}  F1={r['f1']:.4f}  FNR={r['fnr']:.4f}  PR-AUC={r['pr_auc']:.4f}  [{r['scope']}]")

        all_results.extend(model_results)
        # Append după fiecare model (fault-tolerant)
        save_results(model_results, csv_path, overwrite=(args.overwrite and model_key == model_keys[0]))

    # Plot final
    plot_results(all_results, RESULTS_DIR / "scaling_laws.png")

    print("\n" + "="*85)
    print(f"{'n':>8}  {'model':22s}  {'scope':14s}  {'F1':>7}  {'F1mac':>7}  {'FNR':>7}  {'PR-AUC':>7}")
    print("-"*85)
    for r in sorted(all_results, key=lambda x: (x["scope"], x["model"], x["n"])):
        print(
            f"{r['n']:>8}  {r['model']:22s}  {r.get('scope','?'):14s}  "
            f"{r.get('f1',0):>7.4f}  {r.get('f1_macro',0):>7.4f}  "
            f"{r.get('fnr',0):>7.4f}  {r.get('pr_auc',0):>7.4f}"
        )
