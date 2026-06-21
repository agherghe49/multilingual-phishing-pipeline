"""
experiments/generator_fnr_comparison.py

Compares detection FNR for 3 phishing sources:
  1. Standard pipeline (DeepSeek + RAG + SC) — from test.jsonl
  2. Base Qwen2.5-7B-Instruct (no GRPO)
  3. GRPO fine-tuned Qwen2.5-7B-Instruct

Key question: does GRPO produce phishing that is harder to detect than the base model?

Usage:
    python experiments/generator_fnr_comparison.py
    python experiments/generator_fnr_comparison.py --n-gen 50 --n-train 1000
"""

import sys
import json
import argparse
import random
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUT_DIR, MAX_ROUNDS, LOCALES
from prompts import build_prompt

GRPO_DIR      = OUTPUT_DIR / "grpo_model"
BASE_MODEL_HF = "Qwen/Qwen2.5-7B-Instruct"
CLASSIFIER_HF = "xlm-roberta-base"
OUT_DIR       = OUTPUT_DIR / "generator_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON      = OUT_DIR / "generator_fnr_results.json"
OUT_PNG       = OUT_DIR / "generator_fnr_comparison.png"


# ── Email generation ──────────────────────────────────────────────────────────

def build_prompts(n: int, seed: int) -> list[dict]:
    """Builds n diverse prompts for phishing generation."""
    from orchestrator import _infer_fraud_stage
    random.seed(seed)
    stages = ["initial_contact", "trust_building", "urgency_pressure",
              "credential_harvest", "payment_extraction"]
    topics = ["account verification", "banking alert", "prize notification",
              "package delivery", "tech support", "invoice fraud",
              "password reset", "suspicious activity"]
    msgs = []
    for i in range(n):
        locale = LOCALES[i % len(LOCALES)]
        stage  = stages[i % len(stages)]
        topic  = random.choice(topics)
        pd = build_prompt(round_num=(i % MAX_ROUNDS) + 1, topic=topic,
                          fraud_stage=stage, context_docs=[], locale=locale)
        msgs.append({
            "messages": [
                {"role": "system", "content": pd["system"]},
                {"role": "user",   "content": pd["user"]},
            ],
            "locale": locale,
        })
    return msgs


def generate_emails(n: int, seed: int, use_grpo: bool) -> list[str]:
    """Generates n phishing emails with the base or GRPO model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tag = "GRPO" if use_grpo else "Base"
    print(f"\n[gen] Loading {tag} model ({BASE_MODEL_HF}) ...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_HF, trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_HF, quantization_config=bnb,
        trust_remote_code=True, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )

    if use_grpo:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(GRPO_DIR))
        print(f"[gen] PEFT adapter loaded from {GRPO_DIR}")
    else:
        model = base

    model.eval()

    prompts = build_prompts(n, seed)
    emails  = []

    for i, p in enumerate(prompts):
        text = tokenizer.apply_chat_template(
            p["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=400, do_sample=True,
                temperature=0.85, pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        email_txt = tokenizer.decode(generated, skip_special_tokens=True).strip()
        emails.append(email_txt)
        if (i + 1) % 10 == 0:
            print(f"[gen] {tag}: {i+1}/{n} emails generated")

    del model, base
    import torch; torch.cuda.empty_cache()
    return emails


# ── Classifier ───────────────────────────────────────────────────────────────

def train_classifier(train_texts, train_labels, seed: int = 42):
    """Trains XLM-RoBERTa and returns trainer + tokenizer."""
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from torch.utils.data import Dataset as TorchDataset

    class EmailDS(TorchDataset):
        def __init__(self, enc, labels):
            self.enc = enc; self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    tok = AutoTokenizer.from_pretrained(CLASSIFIER_HF)
    enc = tok(train_texts, truncation=True, padding=True, max_length=256)
    ds  = EmailDS(enc, train_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_HF, num_labels=2
    )
    ckpt = OUT_DIR / "clf_tmp"
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
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    return trainer, tok


def compute_fnr(trainer, tok, texts: list[str], labels: list[int]) -> dict:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from sklearn.metrics import f1_score, precision_score, recall_score

    class SimpleDS(TorchDataset):
        def __init__(self, enc, labels):
            self.enc = enc; self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    enc  = tok(texts, truncation=True, padding=True, max_length=256)
    ds   = SimpleDS(enc, labels)
    out  = trainer.predict(ds)
    pred = np.argmax(out.predictions, axis=1)

    f1   = f1_score(labels, pred, pos_label=1, zero_division=0)
    fnr  = 1.0 - recall_score(labels, pred, pos_label=1, zero_division=0)
    prec = precision_score(labels, pred, pos_label=1, zero_division=0)

    return {
        "f1":        round(float(f1),   4),
        "fnr":       round(float(fnr),  4),
        "precision": round(float(prec), 4),
        "n":         len(texts),
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    fnr_vals = [results[l]["fnr"] for l in labels]
    f1_vals  = [results[l]["f1"]  for l in labels]

    colors = ["#2196F3", "#FF9800", "#F44336"]
    x = range(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    bars1 = ax1.bar(x, fnr_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("FNR (False Negative Rate)")
    ax1.set_title("FNR per Generator\n(↑ = harder to detect)")
    ax1.set_ylim(0, 1.0)
    for bar, val in zip(bars1, fnr_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    bars2 = ax2.bar(x, f1_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("F1 (phishing class)")
    ax2.set_title("Detection F1 per Generator\n(↓ = harder to detect)")
    ax2.set_ylim(0, 1.05)
    for bar, val in zip(bars2, f1_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Generator comparison: Standard pipeline vs Base Qwen vs GRPO",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gen",   type=int, default=100,
                        help="Emails generated per model (base + GRPO)")
    parser.add_argument("--n-train", type=int, default=1000,
                        help="Classifier training examples (per class)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip generation and reuse cached emails if available")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print("GENERATOR FNR COMPARISON")
    print(f"n_gen={args.n_gen}  n_train={args.n_train}  seed={args.seed}")
    print("="*60)

    # ── Load standard data ────────────────────────────────────────────────────
    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    train_all = load_jsonl(OUTPUT_DIR / "train.jsonl")
    test_all  = load_jsonl(OUTPUT_DIR / "test.jsonl")

    rng = random.Random(args.seed)
    phishing_train = [r for r in train_all if r["label"] == 1]
    ham_train      = [r for r in train_all if r["label"] == 0]
    per_cls        = args.n_train
    train_sub = (rng.sample(phishing_train, min(per_cls, len(phishing_train))) +
                 rng.sample(ham_train,      min(per_cls, len(ham_train))))
    rng.shuffle(train_sub)

    train_texts  = [r["email_text"] for r in train_sub]
    train_labels = [r["label"]      for r in train_sub]

    # Standard phishing test (from test set — DeepSeek pipeline)
    std_phishing = rng.sample([r for r in test_all if r["label"] == 1],
                              min(args.n_gen, len([r for r in test_all if r["label"]==1])))
    std_texts    = [r["email_text"] for r in std_phishing]

    # Ham for evaluation
    ham_test = rng.sample([r for r in test_all if r["label"] == 0],
                          min(args.n_gen, len([r for r in test_all if r["label"]==0])))
    ham_texts = [r["email_text"] for r in ham_test]

    # ── Generate Base Qwen emails ─────────────────────────────────────────────
    base_cache = OUT_DIR / "base_qwen_emails.json"
    if args.skip_gen and base_cache.exists():
        with open(base_cache) as f:
            base_emails = json.load(f)
        print(f"[cache] Base Qwen: {len(base_emails)} emails loaded")
    else:
        base_emails = generate_emails(args.n_gen, args.seed, use_grpo=False)
        with open(base_cache, "w") as f:
            json.dump(base_emails, f, ensure_ascii=False, indent=2)
        print(f"[saved] {len(base_emails)} emailuri Base Qwen → {base_cache}")

    # ── Generate GRPO Qwen emails ─────────────────────────────────────────────
    grpo_cache = OUT_DIR / "grpo_qwen_emails.json"
    if args.skip_gen and grpo_cache.exists():
        with open(grpo_cache) as f:
            grpo_emails = json.load(f)
        print(f"[cache] GRPO Qwen: {len(grpo_emails)} emails loaded")
    else:
        grpo_emails = generate_emails(args.n_gen, args.seed + 1, use_grpo=True)
        with open(grpo_cache, "w") as f:
            json.dump(grpo_emails, f, ensure_ascii=False, indent=2)
        print(f"[saved] {len(grpo_emails)} emailuri GRPO → {grpo_cache}")

    # ── Train classifier ──────────────────────────────────────────────────────
    print(f"\n[clf] Training classifier on {len(train_texts)} examples ...")
    trainer, tok = train_classifier(train_texts, train_labels, seed=args.seed)

    # ── Evaluate FNR per source ───────────────────────────────────────────────
    results = {}

    print("\n[eval] Standard pipeline (DeepSeek+RAG+SC) ...")
    std_labels_eval = [1] * len(std_texts) + [0] * len(ham_texts)
    results["Standard\n(DeepSeek)"] = compute_fnr(trainer, tok,
                                                    std_texts + ham_texts, std_labels_eval)

    print("[eval] Base Qwen (no GRPO) ...")
    base_labels_eval = [1] * len(base_emails) + [0] * len(ham_texts[:len(base_emails)])
    results["Base Qwen\n(no GRPO)"] = compute_fnr(trainer, tok,
                                                    base_emails + ham_texts[:len(base_emails)],
                                                    base_labels_eval)

    print("[eval] GRPO Qwen (fine-tuned) ...")
    grpo_labels_eval = [1] * len(grpo_emails) + [0] * len(ham_texts[:len(grpo_emails)])
    results["GRPO Qwen\n(fine-tuned)"] = compute_fnr(trainer, tok,
                                                       grpo_emails + ham_texts[:len(grpo_emails)],
                                                       grpo_labels_eval)

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  {'Generator':30} | {'FNR':>6} | {'F1':>6} | {'Prec':>6}")
    print("-"*55)
    for name, m in results.items():
        print(f"  {name.replace(chr(10),' '):30} | {m['fnr']:>6.4f} | {m['f1']:>6.4f} | {m['precision']:>6.4f}")
    print("="*55)

    # ── Save results ──────────────────────────────────────────────────────────
    key_grpo = "GRPO Qwen\n(fine-tuned)"
    key_base = "Base Qwen\n(no GRPO)"
    grpo_fnr = results[key_grpo]["fnr"]
    base_fnr = results[key_base]["fnr"]
    output = {
        "config": {"n_gen": args.n_gen, "n_train": args.n_train, "seed": args.seed},
        "results": results,
        "interpretation": (
            f"GRPO FNR={grpo_fnr:.4f} vs Base FNR={base_fnr:.4f} — "
            f"delta={grpo_fnr - base_fnr:+.4f}"
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results → {OUT_JSON}")

    plot_results({k.replace("\n", " "): v for k, v in results.items()}, OUT_PNG)

    # Copy plot to raport pics
    import shutil
    pics_dir = Path(__file__).parent.parent / "raport4" / "pics"
    if pics_dir.exists():
        shutil.copy(OUT_PNG, pics_dir / "generator_fnr_comparison.png")
        print(f"[copy] Plot → {pics_dir}/generator_fnr_comparison.png")

    # Cleanup
    clf_tmp = OUT_DIR / "clf_tmp"
    if clf_tmp.exists():
        shutil.rmtree(clf_tmp)


if __name__ == "__main__":
    main()
