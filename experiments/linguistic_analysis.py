"""
experiments/linguistic_analysis.py

Linguistic analysis of generated emails: vocabulary, syntactic complexity,
keyword density per locale and fraud_stage.

No GPU or API calls required — runs on the existing dataset.jsonl.

Usage:
    python experiments/linguistic_analysis.py
"""

import sys
import json
import re
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR / "linguistic_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

URGENCY_KW = [
    "urgent", "immediately", "expires", "deadline", "warning", "act now",
    "limited time", "within 24", "within 48", "imediat", "urgent", "termen",
    "expiră", "dringend", "sofort", "frist", "immédiatement", "d'urgence",
    "délai", "urgente", "subito", "entro", "scadenza",
]

AUTHORITY_KW = [
    "official", "department", "compliance", "regulation", "authority",
    "ministry", "government", "bank", "verified", "legal", "security team",
    "oficial", "departament", "conformitate", "autoritate", "bancă",
    "behörde", "ministerium", "sicherheit", "officiel", "ministère",
    "sécurité", "ufficiale", "ministero", "sicurezza", "banca",
]

THREAT_KW = [
    "suspended", "blocked", "terminated", "legal action", "penalty",
    "suspendat", "blocat", "acțiune legală", "gesperrt", "gesperrt",
    "suspendu", "bloqué", "poursuite", "sospeso", "bloccato", "azione legale",
]


def load_dataset(path: Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def type_token_ratio(text: str) -> float:
    """Vocabulary diversity: unique types / total tokens."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def avg_sentence_length(text: str) -> float:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return round(sum(lengths) / len(lengths), 2)


def keyword_density(text: str, keywords: list[str]) -> float:
    """Keywords per 100 words."""
    text_lower = text.lower()
    words = len(text_lower.split())
    if words == 0:
        return 0.0
    hits = sum(1 for kw in keywords if kw in text_lower)
    return round(hits / words * 100, 4)


def analyze_corpus(emails: list[dict]) -> dict:
    """Computes linguistic metrics for a list of emails."""
    if not emails:
        return {}
    texts = [e["email_text"] for e in emails]
    return {
        "n":              len(texts),
        "avg_words":      round(np.mean([len(t.split()) for t in texts]), 1),
        "std_words":      round(np.std([len(t.split()) for t in texts]), 1),
        "avg_ttr":        round(np.mean([type_token_ratio(t) for t in texts]), 4),
        "avg_sent_len":   round(np.mean([avg_sentence_length(t) for t in texts]), 2),
        "urgency_density":   round(np.mean([keyword_density(t, URGENCY_KW) for t in texts]), 4),
        "authority_density": round(np.mean([keyword_density(t, AUTHORITY_KW) for t in texts]), 4),
        "threat_density":    round(np.mean([keyword_density(t, THREAT_KW) for t in texts]), 4),
    }


def plot_per_locale(results_phishing: dict, results_ham: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    locales = sorted(results_phishing.keys())
    metrics = ["avg_words", "avg_ttr", "urgency_density", "authority_density"]
    labels  = ["Average length (words)", "TTR (vocab diversity)",
               "Urgency density (/100 words)", "Authority density (/100 words)"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, metric, label in zip(axes, metrics, labels):
        ph_vals  = [results_phishing[l].get(metric, 0) for l in locales]
        ham_vals = [results_ham[l].get(metric, 0)      for l in locales]
        x = np.arange(len(locales))
        w = 0.35
        ax.bar(x - w/2, ph_vals,  w, label="Phishing",  color="#F44336", alpha=0.85)
        ax.bar(x + w/2, ham_vals, w, label="Legitimate ham", color="#2196F3", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(locales, fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for i, (pv, hv) in enumerate(zip(ph_vals, ham_vals)):
            ax.annotate(f"{pv:.2f}", (x[i]-w/2, pv), textcoords="offset points",
                        xytext=(0,3), ha="center", fontsize=8)
            ax.annotate(f"{hv:.2f}", (x[i]+w/2, hv), textcoords="offset points",
                        xytext=(0,3), ha="center", fontsize=8)

    fig.suptitle("Linguistic analysis: Phishing vs. Ham per Locale",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[ling] Locale plot saved: {out_path}")


def plot_per_stage(results_by_stage: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages  = sorted(results_by_stage.keys())
    metrics = ["avg_words", "urgency_density", "authority_density", "threat_density"]
    labels  = ["Average length (words)", "Urgency density",
               "Authority density", "Threat density"]
    colors  = ["#FF5722", "#E91E63", "#9C27B0", "#3F51B5"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = [results_by_stage[s].get(metric, 0) for s in stages]
        bars = ax.bar(stages, vals, color=color, alpha=0.85)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticklabels(stages, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width()/2, v),
                        textcoords="offset points", xytext=(0,3),
                        ha="center", fontsize=9)

    fig.suptitle("Linguistic features per Fraud Stage (phishing)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[ling] Fraud stage plot saved: {out_path}")


def main():
    print("[ling] Loading dataset...")
    data = load_dataset(OUTPUT_DIR / "dataset.jsonl")
    print(f"[ling] {len(data)} emailuri")

    phishing = [d for d in data if d["label"] == 1]
    ham      = [d for d in data if d["label"] == 0]
    print(f"[ling] {len(phishing)} phishing | {len(ham)} ham")

    # ── Per-locale analysis ───────────────────────────────────────────────
    locales = ["ro-RO", "en-US", "de-DE", "fr-FR", "it-IT"]
    results_phishing = {}
    results_ham      = {}

    print("\n[ling] Per-locale analysis:")
    print(f"{'Locale':<8} {'Label':<10} {'N':>5} {'Words':>7} {'TTR':>7} {'Urgency':>9} {'Authority':>11}")
    print("-" * 65)

    for locale in locales:
        ph  = [d for d in phishing if d.get("locale") == locale]
        hm  = [d for d in ham      if d.get("locale") == locale]
        results_phishing[locale] = analyze_corpus(ph)
        results_ham[locale]      = analyze_corpus(hm)
        rp = results_phishing[locale]
        rh = results_ham[locale]
        print(f"{locale:<8} {'phishing':<10} {rp['n']:>5} {rp['avg_words']:>7.1f} "
              f"{rp['avg_ttr']:>7.4f} {rp['urgency_density']:>9.4f} {rp['authority_density']:>11.4f}")
        print(f"{locale:<8} {'ham':<10} {rh['n']:>5} {rh['avg_words']:>7.1f} "
              f"{rh['avg_ttr']:>7.4f} {rh['urgency_density']:>9.4f} {rh['authority_density']:>11.4f}")
        print()

    # ── Per fraud_stage analysis (phishing only) ──────────────────────────
    stages = ["initial_contact", "trust_building", "urgency_pressure",
              "credential_harvest", "payment_extraction", "authority", "urgency"]
    results_by_stage = {}

    print("\n[ling] Per fraud_stage analysis (phishing):")
    print(f"{'Stage':<22} {'N':>5} {'Words':>7} {'Urgency':>9} {'Authority':>11} {'Threat':>8}")
    print("-" * 65)

    for stage in stages:
        ph_stage = [d for d in phishing if d.get("fraud_stage") == stage]
        if not ph_stage:
            continue
        r = analyze_corpus(ph_stage)
        results_by_stage[stage] = r
        print(f"{stage:<22} {r['n']:>5} {r['avg_words']:>7.1f} "
              f"{r['urgency_density']:>9.4f} {r['authority_density']:>11.4f} {r['threat_density']:>8.4f}")

    # ── Global phishing vs ham ────────────────────────────────────────────
    print("\n[ling] Global phishing vs ham:")
    rph  = analyze_corpus(phishing)
    rham = analyze_corpus(ham)
    for metric in ["avg_words", "avg_ttr", "avg_sent_len",
                   "urgency_density", "authority_density", "threat_density"]:
        print(f"  {metric:<22}: phishing={rph[metric]:>8}  ham={rham[metric]:>8}  "
              f"Δ={round(rph[metric]-rham[metric], 4):>+8}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_per_locale(results_phishing, results_ham,
                    OUT_DIR / "linguistic_per_locale.png")
    if results_by_stage:
        plot_per_stage(results_by_stage,
                       OUT_DIR / "linguistic_per_stage.png")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "per_locale_phishing": results_phishing,
        "per_locale_ham":      results_ham,
        "per_stage":           results_by_stage,
        "global":              {"phishing": rph, "ham": rham},
    }
    out_json = OUT_DIR / "linguistic_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[ling] Results saved: {out_json}")


if __name__ == "__main__":
    main()
