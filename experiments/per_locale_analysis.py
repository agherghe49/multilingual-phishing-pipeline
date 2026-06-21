"""
experiments/per_locale_analysis.py

Per-language analysis: generation quality, detection difficulty, GRPO impact.

Combines data from:
  - audit_log.jsonl        → generation quality + self-correction per locale
  - scaling_results.csv    → F1/FNR detection per locale (if available)
  - grpo_eval.json         → base vs. GRPO reward per locale

Usage:
    python experiments/per_locale_analysis.py
"""

import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from training.grpo_train import _heuristic_quality, _diversity_scores, _format_score

OUT_DIR    = OUTPUT_DIR / "per_locale_analysis"
AUDIT_LOG  = OUTPUT_DIR / "audit_log.jsonl"
GRPO_EVAL  = OUTPUT_DIR / "grpo_eval.json"
SCALING_CSV = OUTPUT_DIR / "scaling_laws" / "scaling_results.csv"
DATASET    = OUTPUT_DIR / "dataset.jsonl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCALES = ["ro-RO", "en-US", "de-DE", "fr-FR", "it-IT"]


# ── 1. Generation quality per locale (from audit_log) ────────────────────────

def analyze_generation_per_locale() -> dict:
    with open(AUDIT_LOG, encoding="utf-8") as f:
        logs = [json.loads(l) for l in f if l.strip()]

    locale_data = defaultdict(lambda: {
        "scores": [], "iters": [], "accepted": 0, "total": 0
    })

    for entry in logs:
        locale = entry.get("locale", "unknown")
        locale_data[locale]["scores"].append(entry.get("final_score", 0))
        locale_data[locale]["iters"].append(entry.get("total_iters", 1))
        locale_data[locale]["total"] += 1
        if entry.get("accepted", False):
            locale_data[locale]["accepted"] += 1

    results = {}
    for locale in LOCALES:
        d = locale_data.get(locale, {"scores": [], "iters": [], "accepted": 0, "total": 0})
        if not d["scores"]:
            continue
        results[locale] = {
            "n":              d["total"],
            "acceptance_rate": round(d["accepted"] / d["total"] * 100, 1) if d["total"] else 0,
            "avg_score":      round(float(np.mean(d["scores"])), 3),
            "std_score":      round(float(np.std(d["scores"])), 3),
            "avg_iters":      round(float(np.mean(d["iters"])), 3),
            "pct_multi_iter": round(sum(1 for i in d["iters"] if i > 1) / len(d["iters"]) * 100, 1),
        }
    return results


# ── 2. Heuristic reward per locale (from dataset.jsonl) ──────────────────────

def analyze_reward_per_locale() -> dict:
    with open(DATASET, encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]

    phishing = [d for d in data if d["label"] == 1]

    locale_emails = defaultdict(list)
    for d in phishing:
        locale = d.get("locale", "unknown")
        locale_emails[locale].append(d)

    results = {}
    for locale in LOCALES:
        emails = locale_emails.get(locale, [])
        if not emails:
            continue
        texts  = [e["email_text"] for e in emails]
        stages = [e.get("fraud_stage", "authority") for e in emails]
        divs   = _diversity_scores(texts[:200])  # sample for speed

        sample_size = min(200, len(texts))
        qs = [_heuristic_quality(texts[i], locale, stages[i]) for i in range(sample_size)]
        fs = [_format_score(texts[i]) for i in range(sample_size)]
        rs = [0.5*q + 0.3*d + 0.2*f for q, d, f in zip(qs, divs[:sample_size], fs)]

        results[locale] = {
            "n_phishing":   len(emails),
            "avg_reward":   round(float(np.mean(rs)), 4),
            "avg_quality":  round(float(np.mean(qs)), 4),
            "avg_format":   round(float(np.mean(fs)), 4),
            "avg_words":    round(float(np.mean([len(t.split()) for t in texts[:sample_size]])), 1),
        }
    return results


# ── 3. GRPO impact per locale ────────────────────────────────────────────────


def analyze_grpo_per_locale() -> dict:
    with open(GRPO_EVAL, encoding="utf-8") as f:
        data = json.load(f)

    locale_data = defaultdict(lambda: {"base": [], "grpo": []})
    for s in data["samples"]:
        locale = s["locale"]
        locale_data[locale]["base"].append(s["base_score"]["reward"])
        locale_data[locale]["grpo"].append(s["grpo_score"]["reward"])

    results = {}
    for locale in LOCALES:
        d = locale_data.get(locale, {"base": [], "grpo": []})
        if not d["base"]:
            continue
        avg_base = round(float(np.mean(d["base"])), 4)
        avg_grpo = round(float(np.mean(d["grpo"])), 4)
        results[locale] = {
            "n":        len(d["base"]),
            "base":     avg_base,
            "grpo":     avg_grpo,
            "delta":    round(avg_grpo - avg_base, 4),
        }
    return results


# ── 4. Plot ──────────────────────────────────────────────────────────────────

def plot_per_locale(gen_results: dict, reward_results: dict,
                    grpo_results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    locales = LOCALES
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Self-correction score per locale
    ax = axes[0][0]
    scores = [gen_results.get(l, {}).get("avg_score", 0) for l in locales]
    stds   = [gen_results.get(l, {}).get("std_score", 0) for l in locales]
    bars = ax.bar(locales, scores, color="#4CAF50", alpha=0.85, yerr=stds, capsize=5)
    ax.set_title("Average self-correction score per locale", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.axhline(6.0, color="red", linestyle="--", alpha=0.5, label="Acceptance threshold (6.0)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, scores):
        ax.annotate(f"{v:.2f}", (bar.get_x()+bar.get_width()/2, v),
                    textcoords="offset points", xytext=(0,4), ha="center", fontsize=10)

    # Panel 2: Self-correction iterations per locale
    ax = axes[0][1]
    iters   = [gen_results.get(l, {}).get("avg_iters", 1) for l in locales]
    pct_multi = [gen_results.get(l, {}).get("pct_multi_iter", 0) for l in locales]
    x = np.arange(len(locales))
    w = 0.35
    ax.bar(x - w/2, iters,     w, label="Avg iterations",    color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, [p/100 * 3 for p in pct_multi], w,
           label="% multi-iter (scaled)", color="#FF9800", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(locales)
    ax.set_title("Self-correction complexity per locale", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Heuristic reward per locale (from dataset)
    ax = axes[1][0]
    rewards   = [reward_results.get(l, {}).get("avg_reward", 0)  for l in locales]
    qualities = [reward_results.get(l, {}).get("avg_quality", 0) for l in locales]
    x = np.arange(len(locales))
    ax.bar(x - w/2, rewards,   w, label="Reward total",  color="#9C27B0", alpha=0.85)
    ax.bar(x + w/2, qualities, w, label="Quality score", color="#E91E63", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(locales)
    ax.set_title("Generation quality (heuristic reward) per locale", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: GRPO reward delta per locale
    ax = axes[1][1]
    if grpo_results:
        base_vals  = [grpo_results.get(l, {}).get("base",  0) for l in locales]
        grpo_vals  = [grpo_results.get(l, {}).get("grpo",  0) for l in locales]
        deltas     = [grpo_results.get(l, {}).get("delta", 0) for l in locales]
        x = np.arange(len(locales))
        ax.bar(x - w/2, base_vals, w, label="Base model",   color="#607D8B", alpha=0.85)
        ax.bar(x + w/2, grpo_vals, w, label="GRPO fine-tuned", color="#F44336", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(locales)
        for i, d in enumerate(deltas):
            ax.annotate(f"Δ{d:+.3f}", (x[i], max(base_vals[i], grpo_vals[i])),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color="green" if d > 0 else "red")
        ax.set_title("GRPO impact per locale (API reward)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 0.8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-language analysis: generation, self-correction and GRPO",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[locale] Plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[locale] Analyzing generation per locale from audit_log...")
    gen_results = analyze_generation_per_locale()

    print("[locale] Computing heuristic reward per locale from dataset...")
    reward_results = analyze_reward_per_locale()

    print("[locale] Analyzing GRPO impact per locale...")
    grpo_results = analyze_grpo_per_locale() if GRPO_EVAL.exists() else {}

    # ── Print table ───────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("PER-LANGUAGE ANALYSIS")
    print("="*90)
    print(f"\n{'Locale':<8} {'N gen':>6} {'SC score':>9} {'Acc%':>6} {'Avg iter':>9} "
          f"{'HReward':>8} {'Base R':>8} {'GRPO R':>8} {'Δ':>7}")
    print("-"*80)
    for locale in LOCALES:
        g  = gen_results.get(locale, {})
        r  = reward_results.get(locale, {})
        gr = grpo_results.get(locale, {})
        print(f"{locale:<8} {g.get('n',0):>6} {g.get('avg_score',0):>9.3f} "
              f"{g.get('acceptance_rate',0):>6.1f}% {g.get('avg_iters',1):>9.3f} "
              f"{r.get('avg_reward',0):>8.4f} "
              f"{gr.get('base',0):>8.4f} {gr.get('grpo',0):>8.4f} "
              f"{gr.get('delta',0):>+7.4f}")
    print("="*90)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_per_locale(gen_results, reward_results, grpo_results,
                    OUT_DIR / "per_locale_analysis.png")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "generation":  gen_results,
        "reward":      reward_results,
        "grpo_impact": grpo_results,
    }
    out_json = OUT_DIR / "per_locale_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[locale] Results saved: {out_json}")


if __name__ == "__main__":
    main()
