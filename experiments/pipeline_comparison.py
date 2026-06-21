"""
experiments/pipeline_comparison.py

Compară calitatea emailurilor phishing la fiecare etapă a pipeline-ului:

  Etapa 1 — LLM de bază, fără RAG, fără self-correction
  Etapa 2 — LLM cu RAG, fără self-correction  (din audit_log: generation_result)
  Etapa 3 — LLM cu RAG + self-correction       (din audit_log: email_text final)
  Etapa 4 — GRPO fine-tuned                    (din grpo_eval.json)

Arată contribuția fiecărei componente la calitatea finală.
Scorarea se face cu reward heuristic pentru consistență între etape.

Rulare:
    python experiments/pipeline_comparison.py
    python experiments/pipeline_comparison.py --n 100
"""

import sys
import json
import random
import os
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, KB_DIR, LOCALES, MAX_ROUNDS
from prompts import build_prompt
from orchestrator import _infer_fraud_stage
from training.grpo_train import _heuristic_quality, _diversity_scores, _format_score

OUT_DIR  = OUTPUT_DIR / "pipeline_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG  = OUTPUT_DIR / "audit_log.jsonl"
GRPO_EVAL  = OUTPUT_DIR / "grpo_eval.json"


# ── Scorare consistentă cu reward heuristic ───────────────────────────────────

def score_email(text: str, locale: str, stage: str) -> dict:
    q = _heuristic_quality(text, locale, stage)
    f = _format_score(text)
    return {
        "quality":   round(q, 4),
        "format":    round(f, 4),
        "words":     len(text.split()),
        "reward":    round(0.5 * q + 0.3 * 0.5 + 0.2 * f, 4),  # diversity=0.5 placeholder
    }


def score_batch(emails: list[dict]) -> dict:
    """Scorează o listă de emailuri și returnează medii."""
    texts   = [e["text"]   for e in emails]
    locales = [e["locale"] for e in emails]
    stages  = [e["stage"]  for e in emails]

    div_scores = _diversity_scores(texts)

    rewards, qualities, formats, divs, words = [], [], [], [], []
    for text, locale, stage, div in zip(texts, locales, stages, div_scores):
        q = _heuristic_quality(text, locale, stage)
        f = _format_score(text)
        r = round(0.5 * q + 0.3 * div + 0.2 * f, 4)
        rewards.append(r)
        qualities.append(q)
        formats.append(f)
        divs.append(div)
        words.append(len(text.split()))

    return {
        "n":           len(emails),
        "reward":      round(float(np.mean(rewards)), 4),
        "quality":     round(float(np.mean(qualities)), 4),
        "diversity":   round(float(np.mean(divs)), 4),
        "format":      round(float(np.mean(formats)), 4),
        "avg_words":   round(float(np.mean(words)), 1),
        "pct_degenerate": round(sum(1 for w in words if w < 80) / len(words) * 100, 1),
    }


# ── Etapa 1: LLM fără RAG, fără self-correction ───────────────────────────────

def generate_no_rag(n: int, seed: int = 42) -> list[dict]:
    """Generează emailuri cu LLM de bază fără context RAG."""
    from generator import generate_email
    from config import DEFAULT_GENERATOR_MODEL

    random.seed(seed)
    scenarios = []
    for path in sorted(KB_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        scenarios.extend(items)

    results = []
    locales_cycle = LOCALES * (n // len(LOCALES) + 1)
    random.shuffle(locales_cycle)

    print(f"[pipeline] Etapa 1: generez {n} emailuri fără RAG...")
    for i in range(n):
        scenario  = random.choice(scenarios)
        locale    = locales_cycle[i]
        round_num = (i % MAX_ROUNDS) + 1
        stage     = _infer_fraud_stage(scenario, round_num)
        topic     = scenario.get("subcategory", "account verification")

        # Prompt fără context_docs
        pd = build_prompt(round_num=round_num, topic=topic,
                          fraud_stage=stage, context_docs=[], locale=locale)

        result = generate_email(
            system_prompt = pd["system"],
            user_prompt   = pd["user"],
            locale        = locale,
            round_num     = round_num,
            scenario_id   = scenario.get("id", 0),
            fraud_stage   = stage,
            model_name    = DEFAULT_GENERATOR_MODEL,
        )

        if result.success and result.email_text.strip():
            results.append({"text": result.email_text,
                             "locale": locale, "stage": stage})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n} ...")

    print(f"  → {len(results)} emailuri generate")
    return results


# ── Etapa 2 & 3: din audit_log ─────────────────────────────────────────────────

def load_audit_stages(n: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Etapa 2 = generation_result['email_text'] (cu RAG, fără self-correction)
    Etapa 3 = email_text final (cu RAG + self-correction)
    """
    random.seed(seed)
    with open(AUDIT_LOG, encoding="utf-8") as f:
        logs = [json.loads(l) for l in f if l.strip()]

    random.shuffle(logs)
    sample = logs[:n]

    stage2, stage3 = [], []
    for entry in sample:
        locale = entry.get("locale", "en-US")
        stage  = entry.get("fraud_stage", "authority")
        # Etapa 2: emailul generat înainte de self-correction
        raw = entry.get("generation_result", {})
        raw_text = raw.get("email_text", "") if isinstance(raw, dict) else ""
        if raw_text.strip():
            stage2.append({"text": raw_text, "locale": locale, "stage": stage})
        # Etapa 3: emailul final după self-correction
        final_text = entry.get("email_text", "")
        if final_text.strip():
            stage3.append({"text": final_text, "locale": locale, "stage": stage})

    return stage2, stage3


# ── Etapa 4: din grpo_eval.json ───────────────────────────────────────────────

def load_grpo_stage() -> tuple[list[dict], list[dict]]:
    """Base model și GRPO fine-tuned din grpo_eval.json."""
    with open(GRPO_EVAL, encoding="utf-8") as f:
        data = json.load(f)

    base_emails, grpo_emails = [], []
    for s in data["samples"]:
        locale = s["locale"]
        stage  = s["fraud_stage"]
        if s["base_email"].strip():
            base_emails.append({"text": s["base_email"], "locale": locale, "stage": stage})
        if s["grpo_email"].strip():
            grpo_emails.append({"text": s["grpo_email"], "locale": locale, "stage": stage})

    return base_emails, grpo_emails


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_pipeline(results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages  = list(results.keys())
    metrics = ["reward", "quality", "diversity", "format"]
    labels  = ["Reward total", "Calitate", "Diversitate", "Format"]
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = [results[s].get(metric, 0) for s in stages]
        bars = ax.bar(stages, vals, color=color, alpha=0.85)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(stages, rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width()/2, v),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Comparație pipeline de generare — calitate per etapă",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[pipeline] Plot salvat: {out_path}")


def plot_degenerate(results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = list(results.keys())
    words  = [results[s].get("avg_words", 0)       for s in stages]
    degen  = [results[s].get("pct_degenerate", 0)  for s in stages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(stages, words, color="#607D8B", alpha=0.85)
    ax1.set_title("Lungime medie email (cuvinte)", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(stages, rotation=15, ha="right", fontsize=9)
    ax1.axhline(80, color="red", linestyle="--", alpha=0.6, label="Prag minim (80)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(words):
        ax1.annotate(f"{v:.0f}", (i, v), textcoords="offset points",
                     xytext=(0,4), ha="center", fontsize=10)

    ax2.bar(stages, degen, color="#F44336", alpha=0.85)
    ax2.set_title("Emailuri degenerate % (sub 80 cuvinte)", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(stages, rotation=15, ha="right", fontsize=9)
    ax2.set_ylim(0, max(degen) * 1.3 + 1)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(degen):
        ax2.annotate(f"{v:.1f}%", (i, v), textcoords="offset points",
                     xytext=(0,4), ha="center", fontsize=10)

    fig.suptitle("Calitatea formatului per etapă pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[pipeline] Plot format salvat: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=100,
                        help="Emailuri per etapă (etapa 1 generează, 2-4 din fișiere)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-gen", action="store_true",
                        help="Sare generarea etapei 1 (folosește cache)")
    args = parser.parse_args()

    cache_path = OUT_DIR / "stage1_no_rag.json"

    # ── Etapa 1: No RAG ───────────────────────────────────────────────────
    if args.skip_gen and cache_path.exists():
        print("[pipeline] Încarc Etapa 1 din cache...")
        with open(cache_path) as f:
            stage1_emails = json.load(f)
    else:
        stage1_emails = generate_no_rag(args.n, args.seed)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(stage1_emails, f, ensure_ascii=False, indent=2)

    # ── Etapa 2 & 3: audit_log ────────────────────────────────────────────
    print(f"[pipeline] Încarc Etapele 2 & 3 din audit_log ({args.n} samples)...")
    stage2_emails, stage3_emails = load_audit_stages(args.n, args.seed)

    # ── Etapa 4: GRPO ─────────────────────────────────────────────────────
    print("[pipeline] Încarc Etapa 4 din grpo_eval.json...")
    base_emails, grpo_emails = load_grpo_stage()

    # ── Scorare ───────────────────────────────────────────────────────────
    print("\n[pipeline] Scorez toate etapele...")
    results = {
        "1. No RAG":            score_batch(stage1_emails),
        "2. RAG (no SC)":       score_batch(stage2_emails),
        "3. RAG + Self-Corr.":  score_batch(stage3_emails),
        "4. Base model":        score_batch(base_emails),
        "5. GRPO fine-tuned":   score_batch(grpo_emails),
    }

    # ── Print tabel ───────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("COMPARAȚIE PIPELINE — calitate per etapă")
    print("="*80)
    print(f"{'Etapă':<22} {'N':>5} {'Reward':>8} {'Quality':>8} {'Diversity':>10} "
          f"{'Format':>8} {'Words':>7} {'Degen%':>8}")
    print("-"*80)
    for stage_name, r in results.items():
        print(f"{stage_name:<22} {r['n']:>5} {r['reward']:>8.4f} {r['quality']:>8.4f} "
              f"{r['diversity']:>10.4f} {r['format']:>8.4f} {r['avg_words']:>7.1f} "
              f"{r['pct_degenerate']:>7.1f}%")
    print("="*80)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_pipeline(results, OUT_DIR / "pipeline_comparison.png")
    plot_degenerate(results, OUT_DIR / "pipeline_format.png")

    # ── Salvare JSON ──────────────────────────────────────────────────────
    out_json = OUT_DIR / "pipeline_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"config": {"n": args.n, "seed": args.seed},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[pipeline] Rezultate salvate: {out_json}")


if __name__ == "__main__":
    main()
