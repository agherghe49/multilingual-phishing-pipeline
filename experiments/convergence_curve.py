"""
experiments/convergence_curve.py

Plots the GRPO convergence curve: reward (and its components) vs. number of steps.
Evaluated points: step 0 (base), step 200, step 400, step 600 (final model).

Usage:
    python experiments/convergence_curve.py
    python experiments/convergence_curve.py --reward heuristic --n 20
"""

import sys
import json
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from training.eval_grpo import load_prompts, generate_batch, score_emails

GRPO_DIR    = OUTPUT_DIR / "grpo_model"
EVAL_JSON   = OUTPUT_DIR / "grpo_eval.json"          # step-600 (model final)
CKPT_200    = GRPO_DIR / "checkpoint-200"
CKPT_400    = GRPO_DIR / "checkpoint-400"
BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
OUT_PNG     = OUTPUT_DIR / "grpo_convergence.png"
OUT_JSON    = OUTPUT_DIR / "grpo_convergence.json"


def load_step400_results():
    """Loads already-computed results for step 400."""
    if not EVAL_JSON.exists():
        raise FileNotFoundError(f"grpo_eval.json not found at {EVAL_JSON}. Run eval_grpo.py first.")
    with open(EVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)
    base_scores = [s["base_score"] for s in data["samples"]]
    grpo_scores = [s["grpo_score"] for s in data["samples"]]
    return base_scores, grpo_scores, data["config"]


def avg(scores, key):
    return round(sum(s[key] for s in scores) / len(scores), 4)


def plot_convergence(results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    steps    = sorted(results.keys())
    metrics  = ["reward", "quality", "diversity", "format"]
    labels   = {"reward": "Total reward", "quality": "Quality",
                 "diversity": "Diversity", "format": "Format"}
    colors   = {"reward": "#1f77b4", "quality": "#ff7f0e",
                 "diversity": "#2ca02c", "format": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        vals = [results[s][metric] for s in steps]
        ax.plot(steps, vals, "o-", color=colors[metric], linewidth=2,
                markersize=8, label=labels[metric])
        ax.set_title(labels[metric], fontsize=13, fontweight="bold")
        ax.set_xlabel("GRPO steps")
        ax.set_ylabel("Average score")
        ax.set_xticks(steps)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: "Base" if x == 0 else f"{int(x)}"))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(vals[0], color="gray", linestyle="--", alpha=0.4, label="Baseline")
        for x, y in zip(steps, vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10)

    fig.suptitle("GRPO Convergence — Qwen2.5-7B-Instruct (QLoRA, RTX 4090)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[convergence] Plot salvat: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=20)
    parser.add_argument("--reward", choices=["heuristic", "api"], default="heuristic")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as e:
        print(f"Import error: {e}"); return

    # ── Load step 400 results from existing file ──────────────────────────
    print("[convergence] Loading step 400 results from grpo_eval.json...")
    base_scores_400, grpo_scores_400, cfg = load_step400_results()
    n_prompts = min(args.n, len(base_scores_400))
    base_scores_400 = base_scores_400[:n_prompts]
    grpo_scores_400 = grpo_scores_400[:n_prompts]
    print(f"[convergence] {n_prompts} prompts from grpo_eval.json (seed={cfg['seed']})")

    # Reuse the same seed as grpo_eval.json for consistency
    seed = cfg.get("seed", args.seed)
    prompts = load_prompts(n_prompts, seed)[:n_prompts]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    print("\n[convergence] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def eval_checkpoint(ckpt_path, label):
        import gc
        print(f"\n[convergence] Evaluating {label} ({ckpt_path.name})...")
        gc.collect()
        torch.cuda.empty_cache()
        base_m = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config,
            trust_remote_code=True, dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
            low_cpu_mem_usage=True,
        )
        grpo_m = PeftModel.from_pretrained(base_m, str(ckpt_path))
        emails = generate_batch(grpo_m, tokenizer, prompts)
        scores = score_emails(emails, prompts, args.reward)
        grpo_m.cpu()
        del grpo_m, base_m
        gc.collect()
        torch.cuda.empty_cache()
        return scores

    # ── Step 200 ──────────────────────────────────────────────────────────
    scores_200 = eval_checkpoint(CKPT_200, "checkpoint-200")

    # ── Step 400 ──────────────────────────────────────────────────────────
    scores_400 = eval_checkpoint(CKPT_400, "checkpoint-400")

    # ── Assemble results ───────────────────────────────────────────────────
    # grpo_eval.json = final model (step 600)
    results = {
        0:   {m: avg(base_scores_400, m) for m in ["reward", "quality", "diversity", "format"]},
        200: {m: avg(scores_200,      m) for m in ["reward", "quality", "diversity", "format"]},
        400: {m: avg(scores_400,      m) for m in ["reward", "quality", "diversity", "format"]},
        600: {m: avg(grpo_scores_400, m) for m in ["reward", "quality", "diversity", "format"]},
    }

    print("\n[convergence] Results:")
    print(f"{'Metric':<12} {'Base':>8} {'200p':>8} {'400p':>8} {'600p':>8}")
    print("-" * 52)
    for m in ["reward", "quality", "diversity", "format"]:
        print(f"{m:<12} {results[0][m]:>8.4f} {results[200][m]:>8.4f} "
              f"{results[400][m]:>8.4f} {results[600][m]:>8.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_convergence(results, OUT_PNG)

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {"config": {"n": n_prompts, "reward": args.reward, "seed": seed},
              "results": {str(k): v for k, v in results.items()}}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[convergence] JSON saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
