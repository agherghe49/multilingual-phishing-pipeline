"""
experiments/adversarial_loop.py

Iterative adversarial game: 3 rounds of classifier adaptation vs. GRPO phishing.

Protocol:
  Round 0: C₀ trained on STANDARD phishing (train.jsonl)
            → tested on 200 GRPO emails (batch A, seed=42)  → FNR₀

  Round 1: C₁ trained on STANDARD phishing + batch A (GRPO emails from round 0)
            → tested on 200 FRESH GRPO emails (batch B, seed=123)  → FNR₁

  Round 2: C₂ trained on STANDARD phishing + batch A + batch B
            → tested on 200 FRESH GRPO emails (batch C, seed=456)  → FNR₂

Key question: does FNR decrease as the classifier accumulates GRPO examples?
Or does it remain high (GRPO is persistently hard to detect)?

Usage:
    python experiments/adversarial_loop.py
    python experiments/adversarial_loop.py --skip-gen   # reuse cached emails
"""

import sys
import json
import argparse
import random
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, MAX_ROUNDS, LOCALES
from prompts import build_prompt

GRPO_DIR      = OUTPUT_DIR / "grpo_model"
TRAIN_JSONL   = OUTPUT_DIR / "train.jsonl"
TEST_JSONL    = OUTPUT_DIR / "test.jsonl"
BASE_MODEL_HF = "Qwen/Qwen2.5-7B-Instruct"
CLASSIFIER_HF = "xlm-roberta-base"
OUT_DIR       = OUTPUT_DIR / "adversarial_loop"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_GRPO = 200  # GRPO emails per round


# ── GRPO email generation ─────────────────────────────────────────────────────

def generate_grpo_emails(n: int, seed: int) -> list[str]:
    """Generates n phishing emails with the GRPO model (different seed per round)."""
    import gc
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    random.seed(seed)
    stages = ["initial_contact", "trust_building", "urgency_pressure",
              "credential_harvest", "payment_extraction"]
    topics = ["account verification", "prize notification", "banking alert",
              "package delivery", "tech support", "invoice fraud"]

    prompts_msgs = []
    for i in range(n):
        locale    = LOCALES[i % len(LOCALES)]
        round_num = (i % MAX_ROUNDS) + 1
        stage     = stages[i % len(stages)]
        topic     = random.choice(topics)
        pd = build_prompt(round_num=round_num, topic=topic,
                          fraud_stage=stage, context_docs=[], locale=locale)
        prompts_msgs.append({
            "messages": [
                {"role": "system", "content": pd["system"]},
                {"role": "user",   "content": pd["user"]},
            ]
        })

    gc.collect()
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print(f"[loop] Loading GRPO model (seed={seed}, n={n})...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_HF, trust_remote_code=True, token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_HF, quantization_config=bnb_config,
        trust_remote_code=True, low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base, str(GRPO_DIR))
    model.eval()

    emails = []
    for i, p in enumerate(prompts_msgs):
        text = tokenizer.apply_chat_template(
            p["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=500, do_sample=True,
                temperature=0.85, pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        emails.append(tokenizer.decode(generated, skip_special_tokens=True))
        if (i + 1) % 20 == 0:
            print(f"[loop]   {i+1}/{n} emails generated")

    model.cpu()
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return emails


# ── Classifier training and evaluation ───────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def train_and_eval_fnr(
    train_texts: list[str],
    train_labels: list[int],
    test_phishing: list[str],
    test_ham: list[str],
    tag: str,
    seed: int = 42,
) -> dict:
    """Trains XLM-RoBERTa and returns FNR on the adversarial test set."""
    import gc
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from torch.utils.data import Dataset
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                  roc_auc_score, average_precision_score)

    random.seed(seed)

    # Stratified 50/50 sampling (max 3 000)
    by_lbl = defaultdict(list)
    for i, l in enumerate(train_labels):
        by_lbl[l].append(i)
    n_each = min(1500, min(len(v) for v in by_lbl.values()))
    chosen = []
    for idxs in by_lbl.values():
        random.shuffle(idxs)
        chosen.extend(idxs[:n_each])
    random.shuffle(chosen)
    t_texts  = [train_texts[i]  for i in chosen]
    t_labels = [train_labels[i] for i in chosen]

    print(f"[loop] Training {tag}: {len(t_texts)} examples "
          f"({sum(t_labels)} phishing + {len(t_labels)-sum(t_labels)} ham)")

    tok = AutoTokenizer.from_pretrained(CLASSIFIER_HF)

    class EmailDS(Dataset):
        def __init__(self, texts, labels=None):
            enc = tok(texts, truncation=True, padding=True,
                      max_length=256, return_tensors="pt")
            self.ids   = enc["input_ids"]
            self.mask  = enc["attention_mask"]
            self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            item = {"input_ids": self.ids[i], "attention_mask": self.mask[i]}
            if self.labels is not None:
                item["labels"] = self.labels[i]
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
        logging_steps=100,
        report_to="none",
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    # Evaluate on adversarial test set: GRPO phishing + ham
    test_texts_all  = test_phishing + test_ham
    test_labels_all = [1] * len(test_phishing) + [0] * len(test_ham)

    test_ds  = EmailDS(test_texts_all, test_labels_all)
    preds    = trainer.predict(test_ds)
    probs    = torch.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()
    y_pred   = (probs >= 0.5).astype(int)
    y_true   = np.array(test_labels_all)

    results = {
        "f1_phishing": round(float(f1_score(y_true, y_pred, pos_label=1)), 4),
        "precision":   round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall":      round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "fnr":         round(float(1 - recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "auc_roc":     round(float(roc_auc_score(y_true, probs)), 4),
        "n_phishing":  len(test_phishing),
        "n_ham":       len(test_ham),
        "n_train":     len(t_texts),
        "n_grpo_train": int(sum(1 for l in t_labels if l == 1 and l > 0)),
    }

    model.cpu()
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_loop(rounds: list[dict], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    round_labels = [f"Round {r['round']}" for r in rounds]
    fnr_vals     = [r["fnr"]         for r in rounds]
    f1_vals      = [r["f1_phishing"] for r in rounds]
    recall_vals  = [r["recall"]      for r in rounds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, title, color, ylabel in [
        (axes[0], fnr_vals,    "FNR per round",     "#F44336", "FNR"),
        (axes[1], recall_vals, "Recall per round",  "#2196F3", "Recall"),
        (axes[2], f1_vals,     "F1-phishing per round", "#4CAF50", "F1"),
    ]:
        ax.plot(round_labels, vals, marker="o", linewidth=2.5,
                color=color, markersize=10)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Iterative adversarial game: classifier adaptation to GRPO phishing\n"
        "(classifier retrained with accumulated GRPO emails per round)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[loop] Plot saved: {out_path}")


def plot_escalation_detail(rounds: list[dict], out_path: Path):
    """Detailed plot showing N GRPO emails in the training set per round."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    x = [r["round"] for r in rounds]
    fnr  = [r["fnr"]  for r in rounds]
    x_labels = [f"Round {i}\n(C{i})" for i in x]

    ax2 = ax.twinx()
    bars = ax2.bar(x, [r.get("n_grpo_in_train", 0) for r in rounds],
                   alpha=0.2, color="#9C27B0", label="GRPO emails in training")
    ax.plot(x, fnr, marker="o", linewidth=2.5, color="#F44336",
            markersize=12, label="FNR", zorder=5)

    for i, (xi, fi) in enumerate(zip(x, fnr)):
        ax.annotate(f"FNR={fi:.1%}", (xi, fi), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=12,
                    fontweight="bold", color="#F44336")

    ax.set_xlabel("Adversarial round", fontsize=12)
    ax.set_ylabel("FNR (missed phishing rate)", fontsize=12, color="#F44336")
    ax2.set_ylabel("N GRPO emails in training", fontsize=12, color="#9C27B0")
    ax.set_ylim(0, 1.15)
    ax.set_title("Escalation curve: FNR vs. experiența clasificatorului cu GRPO",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    bars_h, bars_l = ax2.get_legend_handles_labels()
    ax.legend(lines + bars_h, labels + bars_l, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[loop] Escalation plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-grpo",   type=int, default=N_GRPO,
                        help="GRPO emails per round")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Reuse cached GRPO emails")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    seeds = [args.seed, 123, 456]  # different seeds per round for diversity

    # ── Load base data ────────────────────────────────────────────────────
    print("[loop] Loading train/test data...")
    train_data = load_jsonl(TRAIN_JSONL)
    test_data  = load_jsonl(TEST_JSONL)

    base_train_texts  = [d["email_text"] for d in train_data]
    base_train_labels = [d["label"]      for d in train_data]
    ham_test          = [d["email_text"] for d in test_data if d["label"] == 0]
    print(f"[loop] Train: {len(base_train_texts)} | Ham test: {len(ham_test)}")

    # ── Generate / load GRPO emails per round ────────────────────────────
    grpo_batches = []
    for i, seed in enumerate(seeds):
        cache = OUT_DIR / f"grpo_batch_round{i}_seed{seed}.json"
        if args.skip_gen and cache.exists():
            print(f"[loop] Loading round {i} batch from cache ({cache.name})...")
            with open(cache) as f:
                emails = json.load(f)
        else:
            emails = generate_grpo_emails(args.n_grpo, seed)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(emails, f, ensure_ascii=False, indent=2)
            print(f"[loop] Round {i} batch: {len(emails)} emails saved")
        grpo_batches.append(emails)

    # ── 3 evaluation rounds ───────────────────────────────────────────────
    rounds_results = []
    accumulated_grpo = []

    print("\n" + "="*65)
    print("ITERATIVE ADVERSARIAL GAME — 3 ROUNDS")
    print("="*65)

    for round_idx in range(3):
        print(f"\n[loop] ── ROUND {round_idx} ──")

        # Training set: base + GRPO accumulated from previous rounds
        train_texts  = base_train_texts  + [e for batch in accumulated_grpo for e in batch]
        train_labels = base_train_labels + [1] * sum(len(b) for b in accumulated_grpo)

        # Test on GRPO emails from the CURRENT ROUND (fresh, unseen during training)
        test_grpo = grpo_batches[round_idx]
        n_ham_test = min(len(test_grpo), len(ham_test))
        test_ham_sample = random.sample(ham_test, n_ham_test)

        n_grpo_in_train = sum(len(b) for b in accumulated_grpo)
        print(f"[loop] Training on: {len(train_texts)} total "
              f"({len(base_train_texts)} standard + {n_grpo_in_train} GRPO)")
        print(f"[loop] Testing on: {len(test_grpo)} GRPO + {n_ham_test} ham")

        result = train_and_eval_fnr(
            train_texts=train_texts,
            train_labels=train_labels,
            test_phishing=test_grpo,
            test_ham=test_ham_sample,
            tag=f"round{round_idx}",
            seed=args.seed + round_idx,
        )
        result["round"]           = round_idx
        result["n_grpo_in_train"] = n_grpo_in_train
        result["seed_test"]       = seeds[round_idx]
        rounds_results.append(result)

        print(f"[loop] Round {round_idx}: FNR={result['fnr']:.4f} "
              f"F1={result['f1_phishing']:.4f} Recall={result['recall']:.4f}")

        # Add this round's emails to the accumulated pool
        accumulated_grpo.append(test_grpo)

    # ── Print final table ─────────────────────────────────────────────────
    print("\n" + "="*75)
    print("ITERATIVE ADVERSARIAL GAME RESULTS")
    print("="*75)
    print(f"{'Round':<8} {'GRPO in train':>14} {'FNR':>8} {'Recall':>8} "
          f"{'F1-ph':>8} {'AUC-ROC':>9}")
    print("-"*65)
    for r in rounds_results:
        trend = ""
        if r["round"] > 0:
            delta = r["fnr"] - rounds_results[r["round"]-1]["fnr"]
            trend = f" ({'↓' if delta < 0 else '↑'}{abs(delta):.4f})"
        print(f"Round {r['round']:<3} {r['n_grpo_in_train']:>14} {r['fnr']:>8.4f} "
              f"{r['recall']:>8.4f} {r['f1_phishing']:>8.4f} {r['auc_roc']:>9.4f}{trend}")
    print("="*75)

    fnr_delta_01 = rounds_results[1]["fnr"] - rounds_results[0]["fnr"]
    fnr_delta_12 = rounds_results[2]["fnr"] - rounds_results[1]["fnr"]
    print(f"\nConclusions:")
    if fnr_delta_01 < -0.05:
        print(f"  → FNR drops by {fnr_delta_01:.4f} (R0→R1): classifier adapts to GRPO!")
    elif fnr_delta_01 > 0.05:
        print(f"  → FNR rises by {fnr_delta_01:+.4f} (R0→R1): GRPO becomes harder to detect!")
    else:
        print(f"  → FNR stable (Δ={fnr_delta_01:+.4f}): no significant adaptation R0→R1")

    if fnr_delta_12 < -0.05:
        print(f"  → FNR drops by {fnr_delta_12:.4f} (R1→R2): continued classifier adaptation")
    else:
        print(f"  → FNR Δ={fnr_delta_12:+.4f} (R1→R2): adaptation plateau")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_loop(rounds_results, OUT_DIR / "adversarial_loop.png")
    plot_escalation_detail(rounds_results, OUT_DIR / "escalation_curve.png")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "config": {"n_grpo_per_round": args.n_grpo, "seeds": seeds,
                   "classifier": CLASSIFIER_HF},
        "rounds": rounds_results,
    }
    out_json = OUT_DIR / "adversarial_loop_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[loop] Results saved: {out_json}")


if __name__ == "__main__":
    main()
