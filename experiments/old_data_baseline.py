"""
experiments/old_data_baseline.py

Demonstrează necesitatea datelor sintetice moderne prin compararea
a două scenarii de antrenament pe ACELAȘI test set:

  SCENARIUL A — "Detectorul clasic":
    Antrenat pe SpamAssassin spam/ham (date din 2002-2003, ~20 de ani vechi)
    → evaluat pe test set-ul sintetic modern (test.jsonl)

  SCENARIUL B — "Detectorul modern":
    Antrenat pe datele sintetice generate (train.jsonl)
    → evaluat pe același test set

Concluzie așteptată:
  Detectorul clasic performează semnificativ mai slab pe atacuri moderne
  (mai ales pe locales non-EN: ro, de, fr, it), justificând necesitatea
  generării de date sintetice relevante temporal și lingvistic.

Referință paper: https://arxiv.org/html/2502.12904v2

Rulare:
  python experiments/old_data_baseline.py
  python experiments/old_data_baseline.py --model bert
  python experiments/old_data_baseline.py --n-old 2000 --n-new 2000
  python experiments/old_data_baseline.py --no-download   # dacă arhivele sunt cached

Output:
  outputs/baseline/comparison_results.csv
  outputs/baseline/comparison_bar.png
  outputs/baseline/per_locale_heatmap.png
  outputs/baseline/confusion_matrices.png
"""

import sys
import json
import tarfile
import hashlib
import argparse
import urllib.request
from pathlib import Path
from email import policy
from email.parser import BytesParser
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

TRAIN_PATH  = OUTPUT_DIR / "train.jsonl"
TEST_PATH   = OUTPUT_DIR / "test.jsonl"
RESULTS_DIR = OUTPUT_DIR / "baseline"

CACHE_DIR = Path(__file__).parent / ".cache"

# SpamAssassin corpus public (2002-2003)
SPAM_URLS = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2",
]
HAM_URLS = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
]


# ── Descărcare și parsare ─────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    fname = dest / url.split("/")[-1]
    if fname.exists():
        print(f"  [cache] {fname.name}")
        return fname
    print(f"  Descărcare {fname.name} ...")
    urllib.request.urlretrieve(url, fname)
    return fname


def _extract_body(raw: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True) or b""
                    return body.decode(errors="replace").strip()
        body = msg.get_payload(decode=True) or b""
        return body.decode(errors="replace").strip()
    except Exception:
        return ""


def _load_from_archives(urls: list[str], n: int, label: int) -> list[dict]:
    records = []
    seen: set[str] = set()

    for url in urls:
        if len(records) >= n:
            break
        archive = _download(url, CACHE_DIR)
        try:
            with tarfile.open(archive, "r:bz2") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    if Path(member.name).name.startswith("cmds"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    body = _extract_body(f.read())
                    if len(body) < 40:
                        continue
                    h = hashlib.sha1(body[:150].encode()).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    records.append({
                        "email_text": body,
                        "label":      label,
                        "locale":     "en-US",  # SpamAssassin e exclusiv în engleză
                        "source":     "spamassassin",
                        "id":         f"sa_{label}_{h[:12]}",
                    })
                    if len(records) >= n:
                        break
        except Exception as e:
            print(f"  [warn] {archive.name}: {e}")

    return records


def load_old_phishing(n: int) -> list[dict]:
    print(f"\n[old] Încarc spam SpamAssassin (phishing vechi, ~2003) — max {n}")
    records = _load_from_archives(SPAM_URLS, n, label=1)
    print(f"[old] {len(records)} emailuri spam/phishing vechi încărcate")
    return records


def load_old_ham(n: int) -> list[dict]:
    print(f"[old] Încarc ham SpamAssassin (legitim vechi) — max {n}")
    records = _load_from_archives(HAM_URLS, n, label=0)
    print(f"[old] {len(records)} emailuri ham vechi încărcate")
    return records


# ── Încărcare date noi ────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fișier negăsit: {path}\n"
            "Rulează mai întâi:\n"
            "  python data/assemble.py --n 5000\n"
            "  python experiments/split_dataset.py"
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Classifier TF-IDF + LR ────────────────────────────────────────────────────

def run_tfidf_lr(
    train: list[dict],
    test:  list[dict],
    seed:  int = 42,
) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, accuracy_score, confusion_matrix,
    )

    X_tr = [r["email_text"] for r in train]
    y_tr = [r["label"]      for r in train]
    X_te = [r["email_text"] for r in test]
    y_te = [r["label"]      for r in test]

    vec     = TfidfVectorizer(max_features=50_000, sublinear_tf=True, ngram_range=(1, 2))
    X_tr_v  = vec.fit_transform(X_tr)
    X_te_v  = vec.transform(X_te)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
    clf.fit(X_tr_v, y_tr)

    y_pred  = clf.predict(X_te_v)
    y_proba = clf.predict_proba(X_te_v)[:, 1]
    cm      = confusion_matrix(y_te, y_pred)

    return {
        "f1":        round(f1_score(y_te, y_pred),        4),
        "precision": round(precision_score(y_te, y_pred), 4),
        "recall":    round(recall_score(y_te, y_pred),    4),
        "auc":       round(roc_auc_score(y_te, y_proba),  4),
        "accuracy":  round(accuracy_score(y_te, y_pred),  4),
        "cm":        cm.tolist(),
        "y_pred":    y_pred.tolist(),
        "y_true":    y_te,
    }


# ── Classifier DistilBERT ─────────────────────────────────────────────────────

def run_bert(
    train:      list[dict],
    test:       list[dict],
    seed:       int = 42,
    model_name: str = "distilbert-base-multilingual-cased",
) -> dict:
    """
    Fine-tunează DistilBERT multilingv.
    Multilingv (nu base-uncased) deoarece test set-ul conține ro/de/fr/it/en.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer,
        )
        from datasets import Dataset as HFDataset
        from sklearn.metrics import (
            f1_score, precision_score, recall_score,
            roc_auc_score, accuracy_score, confusion_matrix,
        )
    except ImportError:
        print("[warn] transformers/datasets/torch neinstalate, skip BERT")
        return {}

    X_tr = [r["email_text"] for r in train]
    y_tr = [r["label"]      for r in train]
    X_te = [r["email_text"] for r in test]
    y_te = [r["label"]      for r in test]

    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize(texts: list[str]) -> dict:
        return tok(texts, truncation=True, padding=True, max_length=256, return_tensors=None)

    tr_ds = HFDataset.from_dict({**tokenize(X_tr), "labels": y_tr})
    te_ds = HFDataset.from_dict({**tokenize(X_te), "labels": y_te})

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir                  = str(RESULTS_DIR / "bert_tmp"),
        num_train_epochs            = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        learning_rate               = 2e-5,
        weight_decay                = 0.01,
        eval_strategy               = "epoch",
        save_strategy               = "no",
        load_best_model_at_end      = False,
        fp16                        = torch.cuda.is_available(),
        seed                        = seed,
        report_to                   = "none",
        logging_steps               = 50,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = tr_ds,
        eval_dataset  = te_ds,
    )
    trainer.train()

    preds_out = trainer.predict(te_ds)
    logits    = preds_out.predictions
    y_pred    = logits.argmax(axis=-1).tolist()
    y_proba   = (
        torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    )
    cm = confusion_matrix(y_te, y_pred)

    return {
        "f1":        round(f1_score(y_te, y_pred),        4),
        "precision": round(precision_score(y_te, y_pred), 4),
        "recall":    round(recall_score(y_te, y_pred),    4),
        "auc":       round(roc_auc_score(y_te, y_proba),  4),
        "accuracy":  round(accuracy_score(y_te, y_pred),  4),
        "cm":        cm.tolist(),
        "y_pred":    y_pred,
        "y_true":    y_te,
    }


# ── Analiză per-locale ────────────────────────────────────────────────────────

def per_locale_f1(test: list[dict], y_pred: list[int]) -> dict[str, float]:
    """Calculează F1 per-locale pe test set."""
    from sklearn.metrics import f1_score

    by_locale: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for r, pred in zip(test, y_pred):
        locale = r.get("locale", "unknown")
        by_locale[locale][0].append(r["label"])
        by_locale[locale][1].append(pred)

    results = {}
    for locale, (y_true, y_pr) in sorted(by_locale.items()):
        if len(set(y_true)) < 2:
            results[locale] = float("nan")
        else:
            results[locale] = round(f1_score(y_true, y_pr), 4)
    return results


# ── Vizualizări ───────────────────────────────────────────────────────────────

def plot_comparison_bar(results: list[dict], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib neinstalat, skip plot")
        return

    metrics   = ["f1", "precision", "recall", "auc", "accuracy"]
    labels    = [r["scenario"] for r in results]
    colors    = ["#d9534f", "#5cb85c", "#5bc0de", "#f0ad4e"]
    x         = np.arange(len(metrics))
    bar_width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res[m] for m in metrics]
        offset = (i - len(results) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=res["scenario"],
                      color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Scor")
    ax.set_title(
        "Date vechi vs. date sintetice moderne — comparație detector phishing\n"
        "(test set: emailuri sintetice moderne, multilingv)"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Random baseline")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot] {output_path}")


def plot_per_locale_heatmap(
    locale_results: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    scenarios = list(locale_results.keys())
    locales   = sorted({
        loc
        for res in locale_results.values()
        for loc in res.keys()
        if not (isinstance(res[loc], float) and res[loc] != res[loc])  # exclude nan
    })

    matrix = np.array([
        [locale_results[sc].get(loc, float("nan")) for loc in locales]
        for sc in scenarios
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(locales) * 1.5), len(scenarios) * 1.2 + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(locales)))
    ax.set_xticklabels(locales, fontsize=11)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=10)

    for i in range(len(scenarios)):
        for j in range(len(locales)):
            val = matrix[i, j]
            if val != val:  # nan
                txt = "N/A"
            else:
                txt = f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="black" if 0.35 < val < 0.85 else "white")

    plt.colorbar(im, ax=ax, label="F1 Score")
    ax.set_title("F1 per locale — detector antrenat pe date vechi vs. sintetice moderne")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot] {output_path}")


def plot_confusion_matrices(results: list[dict], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = np.array(res["cm"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(res["scenario"], fontsize=10)
        ax.set_xlabel("Prezis")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Ham (0)", "Phishing (1)"])
        ax.set_yticklabels(["Ham (0)", "Phishing (1)"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.suptitle("Matrici de confuzie — test set sintetic modern", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot] {output_path}")


# ── Raport tabel ──────────────────────────────────────────────────────────────

def print_report(results: list[dict], locale_results: dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print("REZULTATE COMPARATIVE — test set sintetic modern")
    print("=" * 70)
    print(f"{'Scenariu':<35} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUC':>7} {'Acc':>7}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['scenario']:<35} "
            f"{r['f1']:>7.4f} "
            f"{r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} "
            f"{r['auc']:>7.4f} "
            f"{r['accuracy']:>7.4f}"
        )

    print("\nF1 per locale:")
    print(f"{'Locale':<12}", end="")
    for r in results:
        short = r["scenario"][:20]
        print(f"  {short:>20}", end="")
    print()
    print("-" * (12 + 22 * len(results)))

    all_locales = sorted({
        loc
        for res in locale_results.values()
        for loc in res.keys()
    })
    for locale in all_locales:
        print(f"{locale:<12}", end="")
        for r in results:
            val = locale_results[r["scenario"]].get(locale, float("nan"))
            if val != val:
                print(f"  {'N/A':>20}", end="")
            else:
                marker = " ⚠" if val < 0.5 else ""
                print(f"  {val:>18.4f}{marker}", end="")
        print()

    # Concluzie automată
    if len(results) >= 2:
        old_f1 = results[0]["f1"]
        new_f1 = results[1]["f1"]
        delta  = new_f1 - old_f1
        print(f"\nDelta F1 (modern − clasic): {delta:+.4f}")
        if delta > 0.1:
            print("→ Datele sintetice moderne îmbunătățesc semnificativ detecția.")
        elif delta > 0:
            print("→ Datele sintetice moderne aduc o îmbunătățire moderată.")
        else:
            print("→ Atentie: performanța nu s-a îmbunătățit — verifică calitatea datelor.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline comparativ: date vechi vs. date sintetice moderne"
    )
    p.add_argument(
        "--model",
        choices=["lr", "bert", "both"],
        default="lr",
        help="Classifier (lr=TF-IDF+LR, bert=DistilBERT multilingv, both=ambele)",
    )
    p.add_argument(
        "--n-old",
        type=int,
        default=2000,
        help="Nr. emailuri vechi (spam+ham) din SpamAssassin",
    )
    p.add_argument(
        "--n-new",
        type=int,
        default=None,
        help="Nr. emailuri noi din train.jsonl (implicit: toate)",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Folosește doar arhivele deja descărcate în .cache/",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Test set sintetic modern ─────────────────────────────────────────────
    print("[load] Încarc test.jsonl (test set sintetic modern) ...")
    test_records = load_jsonl(TEST_PATH)
    test_ph  = sum(1 for r in test_records if r["label"] == 1)
    test_ham = sum(1 for r in test_records if r["label"] == 0)
    print(f"[load] Test: {len(test_records)} total ({test_ph} phishing, {test_ham} ham)")

    locale_dist = Counter(r.get("locale", "?") for r in test_records)
    print(f"[load] Locales în test: {dict(locale_dist)}")

    if test_ham == 0:
        print(
            "\n[EROARE] Test set-ul nu conține emailuri ham (label=0)!\n"
            "Rulează mai întâi:\n"
            "  python data/assemble.py --n 5000\n"
            "  python experiments/split_dataset.py\n"
        )
        sys.exit(1)

    # ── Scenariul A: date vechi SpamAssassin ─────────────────────────────────
    print("\n[A] Construiesc setul de antrenament VECHI (SpamAssassin 2003) ...")
    half = args.n_old // 2
    old_phishing = load_old_phishing(half)
    old_ham      = load_old_ham(half)
    old_train    = old_phishing + old_ham

    if not old_train:
        print("[EROARE] Nu am putut descărca datele SpamAssassin.")
        sys.exit(1)

    old_ph_n  = sum(1 for r in old_train if r["label"] == 1)
    old_ham_n = sum(1 for r in old_train if r["label"] == 0)
    print(f"[A] Train vechi: {len(old_train)} ({old_ph_n} spam, {old_ham_n} ham) — exclusiv en-US, ~2003")

    # ── Scenariul B: date sintetice noi ──────────────────────────────────────
    print("\n[B] Încarc setul de antrenament SINTETIC NOU (train.jsonl) ...")
    new_train_all = load_jsonl(TRAIN_PATH)
    new_ph_n  = sum(1 for r in new_train_all if r["label"] == 1)
    new_ham_n = sum(1 for r in new_train_all if r["label"] == 0)

    if args.n_new:
        import random
        random.seed(args.seed)
        random.shuffle(new_train_all)
        new_train = new_train_all[: args.n_new]
    else:
        new_train = new_train_all

    print(f"[B] Train nou: {len(new_train)} ({new_ph_n} phishing, {new_ham_n} ham) — multilingv, modern")

    # ── Rulare experimente ────────────────────────────────────────────────────
    all_results: list[dict] = []
    locale_results: dict[str, dict] = {}

    def run_and_collect(train: list[dict], scenario_name: str) -> None:
        print(f"\n{'─'*60}")
        print(f"Antrenament: {scenario_name}")
        print(f"  Train: {len(train)} | Test: {len(test_records)}")

        if args.model in ("lr", "both"):
            print("  → TF-IDF + Logistic Regression ...")
            res = run_tfidf_lr(train, test_records, seed=args.seed)
            tag = f"{scenario_name} [TF-IDF+LR]"
            res["scenario"] = tag
            all_results.append(res)
            locale_results[tag] = per_locale_f1(test_records, res["y_pred"])
            print(f"  F1={res['f1']:.4f}  AUC={res['auc']:.4f}  Acc={res['accuracy']:.4f}")

        if args.model in ("bert", "both"):
            print("  → DistilBERT multilingv ...")
            res = run_bert(train, test_records, seed=args.seed)
            if res:
                tag = f"{scenario_name} [mBERT]"
                res["scenario"] = tag
                all_results.append(res)
                locale_results[tag] = per_locale_f1(test_records, res["y_pred"])
                print(f"  F1={res['f1']:.4f}  AUC={res['auc']:.4f}  Acc={res['accuracy']:.4f}")

    run_and_collect(old_train, "A) Date vechi (SpamAssassin 2003)")
    run_and_collect(new_train, "B) Date sintetice moderne")

    # ── Raport și vizualizări ─────────────────────────────────────────────────
    print_report(all_results, locale_results)

    # Curăță cheile interne înainte de CSV
    csv_results = [
        {k: v for k, v in r.items() if k not in ("cm", "y_pred", "y_true")}
        for r in all_results
    ]
    # Per-locale ca coloane aplatizate
    for res, csv_r in zip(all_results, csv_results):
        scenario = res["scenario"]
        for loc, f1 in locale_results.get(scenario, {}).items():
            csv_r[f"f1_{loc}"] = f1

    try:
        import csv
        csv_path = RESULTS_DIR / "comparison_results.csv"
        if csv_results:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
                writer.writeheader()
                writer.writerows(csv_results)
            print(f"\n[CSV] {csv_path}")
    except Exception as e:
        print(f"[warn] CSV: {e}")

    plot_comparison_bar(all_results, RESULTS_DIR / "comparison_bar.png")
    plot_per_locale_heatmap(locale_results, RESULTS_DIR / "per_locale_heatmap.png")
    plot_confusion_matrices(all_results, RESULTS_DIR / "confusion_matrices.png")

    print(f"\n[done] Toate rezultatele salvate în {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
