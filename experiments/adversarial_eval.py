"""
experiments/adversarial_eval.py

Evaluare adversarială: antrenăm un clasificator XLM-RoBERTa pe phishing generat
de modelul de bază, apoi testăm pe phishing generat de modelul GRPO fine-tunat.

Întrebarea cheie: emailurile GRPO sunt mai greu de detectat de un clasificator
antrenat pe phishing clasic (baza)?

Pași:
  1. Antrenare: XLM-RoBERTa pe train.jsonl (phishing baza + ham)
  2. Test baseline: outputs/test.jsonl (phishing baza + ham)
  3. Generare: N emailuri phishing cu modelul GRPO
  4. Test adversarial: phishing GRPO + ham din test.jsonl
  5. Comparație F1, FNR, Precision, Recall

Rulare:
    python experiments/adversarial_eval.py
    python experiments/adversarial_eval.py --n-grpo 200 --train-size 2000
"""

import sys
import json
import argparse
import os
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, MAX_ROUNDS, LOCALES
from prompts import build_prompt
from orchestrator import _infer_fraud_stage

GRPO_DIR   = OUTPUT_DIR / "grpo_model"
TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
TEST_JSONL  = OUTPUT_DIR / "test.jsonl"
BASE_MODEL_HF = "Qwen/Qwen2.5-7B-Instruct"
CLASSIFIER_HF = "xlm-roberta-base"
OUT_DIR    = OUTPUT_DIR / "adversarial_eval"
OUT_JSON   = OUT_DIR / "adversarial_results.json"
OUT_PNG    = OUT_DIR / "adversarial_comparison.png"


# ── Generare emailuri GRPO ────────────────────────────────────────────────────

def generate_grpo_emails(n: int, seed: int = 42) -> list[str]:
    """Generează n emailuri phishing cu modelul GRPO fine-tunat."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    random.seed(seed)
    locales = ["ro-RO", "en-US", "de-DE", "fr-FR", "it-IT"]
    stages  = ["initial_contact", "trust_building", "urgency_pressure",
               "credential_harvest", "payment_extraction"]

    # Construim prompturi diverse
    prompts_msgs = []
    for i in range(n):
        locale     = locales[i % len(locales)]
        round_num  = (i % MAX_ROUNDS) + 1
        stage      = stages[i % len(stages)]
        topic      = random.choice(["account verification", "prize notification",
                                    "banking alert", "package delivery",
                                    "tech support", "invoice fraud"])
        pd = build_prompt(round_num=round_num, topic=topic,
                          fraud_stage=stage, context_docs=[], locale=locale)
        prompts_msgs.append({
            "messages": [
                {"role": "system", "content": pd["system"]},
                {"role": "user",   "content": pd["user"]},
            ]
        })

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print(f"[adv] Încarc GRPO model pentru generare {n} emailuri...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_HF, trust_remote_code=True, token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_HF, quantization_config=bnb_config,
        trust_remote_code=True, dtype=torch.bfloat16,
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
        if (i + 1) % 10 == 0:
            print(f"[adv] Generat {i+1}/{n} emailuri GRPO")

    del model, base
    torch.cuda.empty_cache()
    return emails


# ── Antrenare și evaluare clasificator ────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def train_and_eval(train_texts, train_labels, test_texts, test_labels,
                   model_name: str = CLASSIFIER_HF, tag: str = "test",
                   train_size: int = 2000):
    """Fine-tunează XLM-RoBERTa și returnează metrici pe setul de test."""
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from torch.utils.data import Dataset
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                  roc_auc_score, average_precision_score)

    # Eșantionare stratificată pe label (50% phishing, 50% ham)
    # Motivație: ham-ul include SpamAssassin (en-only, 2003) + sintetic multilingual;
    # sampling random ar putea supra-reprezenta ham-ul en-only față de phishing multilingual.
    if len(train_texts) > train_size:
        from collections import defaultdict
        by_label = defaultdict(list)
        for i, lbl in enumerate(train_labels):
            by_label[lbl].append(i)
        n_per_label = train_size // len(by_label)
        chosen = []
        for lbl, idxs in by_label.items():
            random.shuffle(idxs)
            chosen.extend(idxs[:n_per_label])
        random.shuffle(chosen)
        train_texts  = [train_texts[i]  for i in chosen]
        train_labels = [train_labels[i] for i in chosen]

    print(f"[adv] Antrenare {model_name} pe {len(train_texts)} exemple "
          f"(stratificat pe label, tag={tag})")

    tok = AutoTokenizer.from_pretrained(model_name)

    class EmailDataset(Dataset):
        def __init__(self, texts, labels):
            enc = tok(texts, truncation=True, padding=True,
                      max_length=256, return_tensors="pt")
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {"input_ids": self.input_ids[i],
                    "attention_mask": self.attention_mask[i],
                    "labels": self.labels[i]}

    train_ds = EmailDataset(train_texts, train_labels)
    test_ds  = EmailDataset(test_texts,  test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / f"clf_{tag}"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
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

    # Predicții
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs  = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    y_pred = (probs >= 0.5).astype(int)
    y_true = np.array(test_labels)

    results = {
        "f1_macro":   round(float(f1_score(y_true, y_pred, average="macro")),   4),
        "f1_phishing":round(float(f1_score(y_true, y_pred, pos_label=1)),        4),
        "precision":  round(float(precision_score(y_true, y_pred, pos_label=1)), 4),
        "recall":     round(float(recall_score(y_true, y_pred, pos_label=1)),    4),
        "fnr":        round(float(1 - recall_score(y_true, y_pred, pos_label=1)),4),
        "auc_roc":    round(float(roc_auc_score(y_true, probs)),                 4),
        "pr_auc":     round(float(average_precision_score(y_true, probs)),       4),
        "n_test":     len(y_true),
        "n_phishing": int(y_true.sum()),
    }

    del model
    torch.cuda.empty_cache()
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(baseline: dict, adversarial: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["f1_macro", "f1_phishing", "precision", "recall", "fnr", "auc_roc"]
    labels  = ["F1-macro", "F1-phishing", "Precision", "Recall", "FNR", "AUC-ROC"]

    x     = np.arange(len(metrics))
    width = 0.35
    base_vals = [baseline[m] for m in metrics]
    adv_vals  = [adversarial[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, base_vals,  width, label="Test baseline (phishing baza)",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, adv_vals,   width, label="Test adversarial (phishing GRPO)",
                   color="#F44336", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Scor", fontsize=12)
    ax.set_title("Evaluare adversarială: clasificator antrenat pe phishing baza\nvs. testat pe phishing GRPO fine-tunat",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[adv] Plot salvat: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-grpo",     type=int, default=200,
                        help="Emailuri GRPO generate pentru test adversarial")
    parser.add_argument("--train-size", type=int, default=3000,
                        help="Exemple folosite la antrenare clasificator")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--skip-gen",   action="store_true",
                        help="Sare generarea GRPO (folosește cache din adv_grpo_emails.json)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ── 1. Încarcă train / test ───────────────────────────────────────────
    print("[adv] Încarc train.jsonl și test.jsonl...")
    train_data = load_jsonl(TRAIN_JSONL)
    test_data  = load_jsonl(TEST_JSONL)

    train_texts  = [d["email_text"] for d in train_data]
    train_labels = [d["label"]      for d in train_data]
    test_texts   = [d["email_text"] for d in test_data]
    test_labels  = [d["label"]      for d in test_data]

    from collections import Counter
    src_counts = Counter(d.get("source", "?") for d in train_data)
    print(f"[adv] Train: {len(train_texts)} | Test baseline: {len(test_texts)}")
    print(f"[adv] Surse train: {dict(src_counts)}")
    print(f"[adv] NOTĂ: ham real (SpamAssassin) este exclusiv en-US — "
          f"sampling stratificat pe label asigură echilibru phishing/ham.")

    # ── 2. Generează emailuri GRPO ────────────────────────────────────────
    grpo_cache = OUT_DIR / "adv_grpo_emails.json"
    if args.skip_gen and grpo_cache.exists():
        print(f"[adv] Încarc emailuri GRPO din cache ({grpo_cache})...")
        with open(grpo_cache, encoding="utf-8") as f:
            grpo_emails = json.load(f)
        print(f"[adv] {len(grpo_emails)} emailuri GRPO din cache")
    else:
        grpo_emails = generate_grpo_emails(args.n_grpo, args.seed)
        with open(grpo_cache, "w", encoding="utf-8") as f:
            json.dump(grpo_emails, f, ensure_ascii=False, indent=2)
        print(f"[adv] {len(grpo_emails)} emailuri GRPO salvate în {grpo_cache}")

    # ── 3. Construiește test adversarial: GRPO phishing + ham din test ────
    ham_test   = [(d["email_text"], 0) for d in test_data if d["label"] == 0]
    n_ham      = min(len(grpo_emails), len(ham_test))
    adv_texts  = grpo_emails[:n_ham] + [h[0] for h in ham_test[:n_ham]]
    adv_labels = [1] * n_ham + [0] * n_ham
    # Shuffle
    combined   = list(zip(adv_texts, adv_labels))
    random.shuffle(combined)
    adv_texts, adv_labels = zip(*combined)
    adv_texts, adv_labels = list(adv_texts), list(adv_labels)
    print(f"[adv] Test adversarial: {len(adv_texts)} emailuri "
          f"({n_ham} GRPO phishing + {n_ham} ham)")

    # ── 4. Antrenare clasificator (o singură dată) ────────────────────────
    baseline_results    = train_and_eval(
        train_texts, train_labels, test_texts, test_labels,
        tag="baseline", train_size=args.train_size,
    )
    adversarial_results = train_and_eval(
        train_texts, train_labels, adv_texts, adv_labels,
        tag="adversarial", train_size=args.train_size,
    )

    # ── 5. Afișare rezultate ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("EVALUARE ADVERSARIALĂ — XLM-RoBERTa")
    print("="*65)
    print(f"{'Metric':<15} {'Baseline':>10} {'Adversarial':>12} {'Δ':>8}")
    print("-"*50)
    for m in ["f1_macro", "f1_phishing", "precision", "recall", "fnr", "auc_roc", "pr_auc"]:
        b = baseline_results[m]
        a = adversarial_results[m]
        d = a - b
        sign = "▲" if d > 0.001 else ("▼" if d < -0.001 else "≈")
        print(f"{m:<15} {b:>10.4f} {a:>12.4f} {sign}{abs(d):>7.4f}")
    print("="*65)
    print(f"\nConcluzii:")
    fnr_delta = adversarial_results["fnr"] - baseline_results["fnr"]
    if fnr_delta > 0.05:
        print(f"  → FNR crește cu {fnr_delta:+.4f}: GRPO generează phishing MAI GREU de detectat!")
    elif fnr_delta > 0.01:
        print(f"  → FNR crește ușor ({fnr_delta:+.4f}): impact marginal al GRPO asupra detecției")
    else:
        print(f"  → FNR Δ={fnr_delta:+.4f}: clasificatorul detectează la fel de bine phishing-ul GRPO")

    # ── 6. Plot și salvare ────────────────────────────────────────────────
    plot_comparison(baseline_results, adversarial_results, OUT_PNG)

    output = {
        "config": {
            "n_grpo": len(grpo_emails), "train_size": args.train_size,
            "seed": args.seed, "classifier": CLASSIFIER_HF,
            "sampling": "stratified_by_label",
        },
        "dataset_notes": {
            "ham_real_en_only": "SpamAssassin 2003 (~4018 emailuri) este exclusiv en-US; "
                                "ham non-englez provine doar din surse sintetice (LLM). "
                                "Clasificatorul poate învăța parțial artefacte de generare LLM "
                                "în loc de pattern-uri lingvistice reale de phishing.",
            "synthetic_bias": "Atât ham sintetic cât și phishing-ul GRPO sunt generate de LLM "
                               "(DeepSeek/Qwen) — clasificatorul poate detecta stilul LLM, "
                               "nu conținutul phishing în sine.",
        },
        "baseline":    baseline_results,
        "adversarial": adversarial_results,
        "delta": {m: round(adversarial_results[m] - baseline_results[m], 4)
                  for m in baseline_results if isinstance(baseline_results[m], float)},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[adv] Rezultate salvate: {OUT_JSON}")


if __name__ == "__main__":
    main()
