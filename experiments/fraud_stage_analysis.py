"""
experiments/fraud_stage_analysis.py

Analiză per fraud_stage:
  1. Reward heuristic (base vs GRPO) per stage — din grpo_eval.json + dataset.jsonl
  2. FNR per stage — reantrenare XLM-RoBERTa pe train.jsonl, test pe fiecare stage separat
  3. FNR adversarial per stage — test cu emailuri GRPO din adv_grpo_emails.json

Rulare:
    python experiments/fraud_stage_analysis.py
    python experiments/fraud_stage_analysis.py --skip-clf
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
from training.grpo_train import _heuristic_quality, _diversity_scores, _format_score

OUT_DIR       = OUTPUT_DIR / "fraud_stage_analysis"
DATASET_PATH  = OUTPUT_DIR / "dataset.jsonl"
TRAIN_JSONL   = OUTPUT_DIR / "train.jsonl"
TEST_JSONL    = OUTPUT_DIR / "test.jsonl"
GRPO_EVAL     = OUTPUT_DIR / "grpo_eval.json"
ADV_EMAILS    = OUTPUT_DIR / "adversarial_eval" / "adv_grpo_emails.json"
CLASSIFIER_HF = "xlm-roberta-base"

OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGES = ["authority", "urgency"]


# ── 1. Reward per stage (heuristic + grpo_eval) ───────────────────────────────

def analyze_reward_per_stage() -> dict:
    """Reward heuristic din dataset.jsonl + base/GRPO din grpo_eval.json."""
    data = [json.loads(l) for l in open(DATASET_PATH, encoding="utf-8") if l.strip()]
    phishing = [d for d in data if d["label"] == 1]

    by_stage = defaultdict(list)
    for d in phishing:
        by_stage[d.get("fraud_stage", "unknown")].append(d)

    heuristic = {}
    for stage in STAGES:
        emails = by_stage.get(stage, [])
        if not emails:
            continue
        texts  = [e["email_text"] for e in emails[:200]]
        stages = [e.get("fraud_stage", stage) for e in emails[:200]]
        divs   = _diversity_scores(texts)
        qs = [_heuristic_quality(texts[i], emails[i].get("locale","en-US"), stages[i])
              for i in range(len(texts))]
        fs = [_format_score(texts[i]) for i in range(len(texts))]
        rs = [0.5*q + 0.3*d + 0.2*f for q, d, f in zip(qs, divs, fs)]
        heuristic[stage] = {
            "n":           len(emails),
            "avg_reward":  round(float(np.mean(rs)), 4),
            "avg_quality": round(float(np.mean(qs)), 4),
            "avg_format":  round(float(np.mean(fs)), 4),
            "avg_words":   round(float(np.mean([len(t.split()) for t in texts])), 1),
        }

    # GRPO eval per stage
    grpo_by_stage = defaultdict(lambda: {"base": [], "grpo": []})
    if GRPO_EVAL.exists():
        with open(GRPO_EVAL, encoding="utf-8") as f:
            gdata = json.load(f)
        for s in gdata["samples"]:
            grpo_by_stage[s["fraud_stage"]]["base"].append(s["base_score"]["reward"])
            grpo_by_stage[s["fraud_stage"]]["grpo"].append(s["grpo_score"]["reward"])

    grpo_impact = {}
    for stage in STAGES:
        d = grpo_by_stage.get(stage, {"base": [], "grpo": []})
        if not d["base"]:
            continue
        avg_base = round(float(np.mean(d["base"])), 4)
        avg_grpo = round(float(np.mean(d["grpo"])), 4)
        grpo_impact[stage] = {
            "n":     len(d["base"]),
            "base":  avg_base,
            "grpo":  avg_grpo,
            "delta": round(avg_grpo - avg_base, 4),
        }

    return heuristic, grpo_impact


# ── 2. FNR per stage — clasificator ───────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def train_classifier(train_texts, train_labels, seed=42):
    """Fine-tunează XLM-RoBERTa pe train set."""
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from torch.utils.data import Dataset

    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(CLASSIFIER_HF)

    class EmailDS(Dataset):
        def __init__(self, texts, labels):
            enc = tok(texts, truncation=True, padding=True,
                      max_length=256, return_tensors="pt")
            self.ids   = enc["input_ids"]
            self.mask  = enc["attention_mask"]
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "attention_mask": self.mask[i],
                    "labels": self.labels[i]}

    # Stratified sampling 50/50 up to 3000
    by_lbl = defaultdict(list)
    for i, l in enumerate(train_labels):
        by_lbl[l].append(i)
    n_each = min(1500, min(len(v) for v in by_lbl.values()))
    chosen = []
    for idxs in by_lbl.values():
        random.shuffle(idxs)
        chosen.extend(idxs[:n_each])
    random.shuffle(chosen)
    train_texts  = [train_texts[i]  for i in chosen]
    train_labels = [train_labels[i] for i in chosen]

    print(f"[stage] Antrenez clasificator pe {len(train_texts)} exemple...")
    train_ds = EmailDS(train_texts, train_labels)

    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_HF, num_labels=2)
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "clf_stage"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()
    print("[stage] Antrenare completă.")
    return trainer, tok


def eval_fnr_per_stage(trainer, tok, phishing_by_stage: dict, label: str) -> dict:
    """Evaluează FNR pentru fiecare fraud_stage."""
    import torch
    from torch.utils.data import Dataset
    from sklearn.metrics import recall_score, f1_score

    class SimpleDS(Dataset):
        def __init__(self, texts):
            enc = tok(texts, truncation=True, padding=True,
                      max_length=256, return_tensors="pt")
            self.ids  = enc["input_ids"]
            self.mask = enc["attention_mask"]
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "attention_mask": self.mask[i]}

    results = {}
    for stage, texts in phishing_by_stage.items():
        if not texts:
            continue
        ds = SimpleDS(texts)
        preds_out = trainer.predict(ds)
        logits = preds_out.predictions
        probs  = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        y_pred = (probs >= 0.5).astype(int)
        y_true = [1] * len(texts)
        fnr    = round(float(1 - recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4)
        f1     = round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4)
        results[stage] = {
            "n":    len(texts),
            "fnr":  fnr,
            "f1":   f1,
            "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        }
        print(f"  [{label}] {stage}: n={len(texts)}, FNR={fnr:.4f}, F1={f1:.4f}")
    return results


# ── 3. Plot ────────────────────────────────────────────────────────────────────

def plot_results(heuristic: dict, grpo_impact: dict,
                 fnr_baseline: dict, fnr_adversarial: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = STAGES
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: Reward per stage (heuristic din dataset)
    ax = axes[0]
    rewards = [heuristic.get(s, {}).get("avg_reward", 0) for s in stages]
    bars = ax.bar(stages, rewards, color=["#1976D2", "#E53935"], alpha=0.85)
    ax.set_title("Reward heuristic per fraud stage\n(dataset phishing)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, rewards):
        ax.annotate(f"{v:.4f}", (bar.get_x() + bar.get_width()/2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11)

    # Panel 2: Base vs GRPO reward per stage
    ax = axes[1]
    x = np.arange(len(stages))
    base_vals = [grpo_impact.get(s, {}).get("base", 0) for s in stages]
    grpo_vals = [grpo_impact.get(s, {}).get("grpo", 0) for s in stages]
    deltas    = [grpo_impact.get(s, {}).get("delta", 0) for s in stages]
    ax.bar(x - w/2, base_vals, w, label="Base model",      color="#607D8B", alpha=0.85)
    ax.bar(x + w/2, grpo_vals, w, label="GRPO fine-tuned", color="#F44336", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    for i, d in enumerate(deltas):
        ax.annotate(f"Δ{d:+.4f}", (x[i], max(base_vals[i], grpo_vals[i])),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=10, color="green" if d > 0 else "red")
    ax.set_title("Impact GRPO per fraud stage\n(reward API, 20 prompturi)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: FNR per stage (baseline vs adversarial)
    ax = axes[2]
    fnr_base = [fnr_baseline.get(s, {}).get("fnr", 0) for s in stages]
    fnr_adv  = [fnr_adversarial.get(s, {}).get("fnr", 0) for s in stages]
    ax.bar(x - w/2, fnr_base, w, label="Test baseline (phishing baza)",   color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, fnr_adv,  w, label="Test adversarial (phishing GRPO)", color="#FF5722", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_title("FNR per fraud stage\n(clasificator XLM-RoBERTa)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, (b, a) in enumerate(zip(fnr_base, fnr_adv)):
        ax.annotate(f"{b:.3f}", (x[i]-w/2, b), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10)
        ax.annotate(f"{a:.3f}", (x[i]+w/2, a), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10)

    fig.suptitle("Analiză per Fraud Stage: reward, impact GRPO și dificultate detecție",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[stage] Plot salvat: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-clf", action="store_true",
                        help="Sare reantrenarea clasificatorului")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Reward per stage ───────────────────────────────────────────────
    print("[stage] Calculez reward per fraud stage...")
    heuristic, grpo_impact = analyze_reward_per_stage()

    print("\n[stage] Reward heuristic (dataset phishing):")
    for stage, r in heuristic.items():
        print(f"  {stage:<12}: n={r['n']:>5}, reward={r['avg_reward']:.4f}, "
              f"quality={r['avg_quality']:.4f}, words={r['avg_words']:.1f}")

    print("\n[stage] GRPO impact per stage:")
    for stage, r in grpo_impact.items():
        print(f"  {stage:<12}: n={r['n']:>3}, base={r['base']:.4f}, "
              f"grpo={r['grpo']:.4f}, Δ={r['delta']:+.4f}")

    # ── 2. FNR per stage ─────────────────────────────────────────────────
    fnr_baseline    = {}
    fnr_adversarial = {}

    if not args.skip_clf:
        print("\n[stage] Încarc train/test data...")
        train_data = load_jsonl(TRAIN_JSONL)
        test_data  = load_jsonl(TEST_JSONL)

        train_texts  = [d["email_text"] for d in train_data]
        train_labels = [d["label"]      for d in train_data]

        # Phishing din test.jsonl grupat pe stage
        test_phishing_by_stage = defaultdict(list)
        for d in test_data:
            if d["label"] == 1:
                test_phishing_by_stage[d.get("fraud_stage", "unknown")].append(d["email_text"])
        test_phishing_by_stage = {s: test_phishing_by_stage[s] for s in STAGES
                                   if s in test_phishing_by_stage}

        print(f"[stage] Test phishing per stage: "
              + ", ".join(f"{s}={len(v)}" for s, v in test_phishing_by_stage.items()))

        # GRPO emails din adv_grpo_emails.json — reconstruim stage labels
        # generare adversarial: stages cycle ["initial_contact","trust_building",
        # "urgency_pressure","credential_harvest","payment_extraction"] → i%5
        # Mapăm la dataset stages: urgency_pressure/payment_extraction → urgency
        # restul → authority
        adv_by_stage = defaultdict(list)
        if ADV_EMAILS.exists():
            with open(ADV_EMAILS, encoding="utf-8") as f:
                adv_emails = json.load(f)
            gen_stages = ["initial_contact", "trust_building", "urgency_pressure",
                          "credential_harvest", "payment_extraction"]
            stage_map = {
                "initial_contact":   "authority",
                "trust_building":    "authority",
                "urgency_pressure":  "urgency",
                "credential_harvest":"authority",
                "payment_extraction":"urgency",
            }
            for i, email_text in enumerate(adv_emails):
                gen_stage = gen_stages[i % len(gen_stages)]
                ds_stage  = stage_map[gen_stage]
                adv_by_stage[ds_stage].append(email_text)
            print(f"[stage] GRPO emails per stage (reconstruit): "
                  + ", ".join(f"{s}={len(v)}" for s, v in adv_by_stage.items()))

        trainer, tok = train_classifier(train_texts, train_labels, seed=args.seed)

        print("\n[stage] FNR baseline (phishing standard din test.jsonl):")
        fnr_baseline = eval_fnr_per_stage(trainer, tok, test_phishing_by_stage, "baseline")

        if adv_by_stage:
            print("\n[stage] FNR adversarial (phishing GRPO):")
            fnr_adversarial = eval_fnr_per_stage(trainer, tok, dict(adv_by_stage), "adversarial")

        import torch
        del trainer
        torch.cuda.empty_cache()

    # ── 3. Print tabel ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ANALIZĂ PER FRAUD STAGE")
    print("="*70)
    print(f"\n{'Stage':<14} {'N':>5} {'H.Reward':>9} {'Base R':>8} {'GRPO R':>8} {'Δ':>7} "
          f"{'FNR base':>10} {'FNR adv':>9}")
    print("-"*70)
    for stage in STAGES:
        h  = heuristic.get(stage, {})
        g  = grpo_impact.get(stage, {})
        fb = fnr_baseline.get(stage, {})
        fa = fnr_adversarial.get(stage, {})
        print(f"{stage:<14} {h.get('n',0):>5} {h.get('avg_reward',0):>9.4f} "
              f"{g.get('base',0):>8.4f} {g.get('grpo',0):>8.4f} {g.get('delta',0):>+7.4f} "
              f"{fb.get('fnr',0):>10.4f} {fa.get('fnr',0):>9.4f}")
    print("="*70)

    # ── 4. Plot ───────────────────────────────────────────────────────────
    plot_results(heuristic, grpo_impact, fnr_baseline, fnr_adversarial,
                 OUT_DIR / "fraud_stage_analysis.png")

    # ── 5. Salvare JSON ───────────────────────────────────────────────────
    output = {
        "heuristic_reward": heuristic,
        "grpo_impact":      grpo_impact,
        "fnr_baseline":     fnr_baseline,
        "fnr_adversarial":  fnr_adversarial,
    }
    out_json = OUT_DIR / "fraud_stage_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[stage] Rezultate salvate: {out_json}")


if __name__ == "__main__":
    main()
