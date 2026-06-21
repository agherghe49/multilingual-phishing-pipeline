"""
training/eval_grpo.py

Compară emailurile generate de modelul de bază vs. modelul GRPO fine-tunat.
Folosește aceleași prompturi pentru ambele modele, scorează cu reward-ul compus
și afișează o comparație tabelară + exemple concrete.

Rulare:
    python training/eval_grpo.py
    python training/eval_grpo.py --n 10 --reward api
"""

import sys
import json
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import KB_DIR, OUTPUT_DIR, LOCALES, MAX_ROUNDS
from prompts import build_prompt
from orchestrator import _infer_fraud_stage
from training.grpo_train import (
    _extract_text, _heuristic_quality, _api_quality,
    _diversity_scores, _format_score,
)

GRPO_DIR    = OUTPUT_DIR / "grpo_model"
BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
EVAL_OUTPUT = OUTPUT_DIR / "grpo_eval.json"


def load_prompts(n: int = 10, seed: int = 42) -> list[dict]:
    """Selectează n prompturi reprezentative din KB."""
    import random
    random.seed(seed)

    scenarios = []
    for path in sorted(KB_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        scenarios.extend(items)

    # Selectăm o combinație variată de locale și fraud_stage
    sample_locales = ["ro-RO", "en-US", "de-DE", "fr-FR", "it-IT"]
    records = []
    for i, scenario in enumerate(random.sample(scenarios, min(n * 2, len(scenarios)))):
        locale      = sample_locales[i % len(sample_locales)]
        round_num   = (i % MAX_ROUNDS) + 1
        topic       = scenario.get("subcategory", "account verification")
        fraud_stage = _infer_fraud_stage(scenario, round_num)
        prompt_d    = build_prompt(
            round_num=round_num, topic=topic,
            fraud_stage=fraud_stage, context_docs=[], locale=locale,
        )
        records.append({
            "messages": [
                {"role": "system", "content": prompt_d["system"]},
                {"role": "user",   "content": prompt_d["user"]},
            ],
            "locale":      locale,
            "fraud_stage": fraud_stage,
            "topic":       topic,
        })
        if len(records) >= n:
            break
    return records


def generate_batch(model, tokenizer, prompts: list[dict], max_new_tokens: int = 500) -> list[str]:
    """Generează câte un email per prompt."""
    import torch
    results = []
    model.eval()
    for p in prompts:
        text = tokenizer.apply_chat_template(
            p["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.85,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True))
    return results


def score_emails(emails: list[str], prompts: list[dict], reward_type: str) -> list[dict]:
    """Scorează lista de emailuri cu reward-ul compus."""
    div_scores = _diversity_scores(emails)
    results = []
    for email, prompt, div in zip(emails, prompts, div_scores):
        locale = prompt["locale"]
        stage  = prompt["fraud_stage"]
        if reward_type == "api":
            quality = _api_quality(email, locale, stage)
        else:
            quality = _heuristic_quality(email, locale, stage)
        fmt     = _format_score(email)
        reward  = round(0.4 * quality + 0.4 * div + 0.2 * fmt, 4)
        results.append({
            "quality":   round(quality, 4),
            "diversity": round(div, 4),
            "format":    round(fmt, 4),
            "reward":    reward,
            "words":     len(email.split()),
        })
    return results


def print_comparison(prompts, base_emails, grpo_emails, base_scores, grpo_scores):
    """Afișează tabel comparativ și exemple."""
    print("\n" + "="*80)
    print("COMPARAȚIE: Model de bază vs. GRPO fine-tuned")
    print("="*80)
    print(f"\n{'#':<4} {'Locale':<8} {'Stage':<12} {'Base R':<8} {'GRPO R':<8} {'Δ':<8} {'Base W':<8} {'GRPO W'}")
    print("-"*70)

    deltas = []
    for i, (p, bs, gs) in enumerate(zip(prompts, base_scores, grpo_scores)):
        delta = gs["reward"] - bs["reward"]
        deltas.append(delta)
        sign = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "≈")
        print(f"{i+1:<4} {p['locale']:<8} {p['fraud_stage']:<12} "
              f"{bs['reward']:<8.4f} {gs['reward']:<8.4f} "
              f"{sign}{abs(delta):<7.4f} {bs['words']:<8} {gs['words']}")

    avg_base = sum(s["reward"] for s in base_scores) / len(base_scores)
    avg_grpo = sum(s["reward"] for s in grpo_scores) / len(grpo_scores)
    avg_delta = avg_grpo - avg_base
    wins_grpo = sum(1 for d in deltas if d > 0.005)
    wins_base = sum(1 for d in deltas if d < -0.005)

    print("-"*70)
    print(f"{'MEDIE':<4} {'':<8} {'':<12} {avg_base:<8.4f} {avg_grpo:<8.4f} "
          f"{'▲' if avg_delta > 0 else '▼'}{abs(avg_delta):<7.4f}")
    print(f"\nGRPO câștigă:  {wins_grpo}/{len(prompts)} emailuri")
    print(f"Base câștigă:  {wins_base}/{len(prompts)} emailuri")
    print(f"Reward mediu:  Base={avg_base:.4f}  GRPO={avg_grpo:.4f}  Δ={avg_delta:+.4f}")

    # Detaliu per componentă
    avg_q_base = sum(s["quality"] for s in base_scores) / len(base_scores)
    avg_q_grpo = sum(s["quality"] for s in grpo_scores) / len(grpo_scores)
    avg_f_base = sum(s["format"] for s in base_scores) / len(base_scores)
    avg_f_grpo = sum(s["format"] for s in grpo_scores) / len(grpo_scores)
    avg_d_base = sum(s["diversity"] for s in base_scores) / len(base_scores)
    avg_d_grpo = sum(s["diversity"] for s in grpo_scores) / len(grpo_scores)
    print(f"\nDetaliu componente:")
    print(f"  Quality:   Base={avg_q_base:.4f}  GRPO={avg_q_grpo:.4f}  Δ={avg_q_grpo-avg_q_base:+.4f}")
    print(f"  Format:    Base={avg_f_base:.4f}  GRPO={avg_f_grpo:.4f}  Δ={avg_f_grpo-avg_f_base:+.4f}")
    print(f"  Diversity: Base={avg_d_base:.4f}  GRPO={avg_d_grpo:.4f}  Δ={avg_d_grpo-avg_d_base:+.4f}")

    # Exemplu cel mai mare câștig GRPO
    best_idx = max(range(len(deltas)), key=lambda i: deltas[i])
    print(f"\n{'='*80}")
    print(f"EXEMPLU — cel mai mare câștig GRPO (#{best_idx+1}, Δ={deltas[best_idx]:+.4f})")
    print(f"Locale: {prompts[best_idx]['locale']}  |  Stage: {prompts[best_idx]['fraud_stage']}")
    print(f"\n--- MODEL DE BAZĂ (reward={base_scores[best_idx]['reward']:.4f}) ---")
    print(base_emails[best_idx][:600] + ("..." if len(base_emails[best_idx]) > 600 else ""))
    print(f"\n--- GRPO FINE-TUNED (reward={grpo_scores[best_idx]['reward']:.4f}) ---")
    print(grpo_emails[best_idx][:600] + ("..." if len(grpo_emails[best_idx]) > 600 else ""))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluare comparativă base vs. GRPO")
    parser.add_argument("--n",       type=int, default=10,          help="Număr prompturi evaluate")
    parser.add_argument("--reward",  choices=["heuristic", "api"],  default="heuristic")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--grpo-dir", type=str, default=str(GRPO_DIR))
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as e:
        print(f"Eroare import: {e}")
        return

    grpo_path = Path(args.grpo_dir)
    if not grpo_path.exists():
        print(f"Model GRPO negăsit la {grpo_path}. Rulează mai întâi grpo_train.py.")
        return

    print(f"[eval] Prompturi: {args.n} | Reward: {args.reward} | GRPO: {grpo_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    print("\n[eval] Încarc tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = load_prompts(args.n, args.seed)
    print(f"[eval] {len(prompts)} prompturi pregătite\n")

    # ── Generare cu modelul de bază ──────────────────────────────────────
    print("[eval] Încarc modelul de bază...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        trust_remote_code=True, dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    print("[eval] Generez cu modelul de bază...")
    base_emails = generate_batch(base_model, tokenizer, prompts)
    base_scores = score_emails(base_emails, prompts, args.reward)
    del base_model
    torch.cuda.empty_cache()

    # ── Generare cu modelul GRPO ─────────────────────────────────────────
    print("\n[eval] Încarc modelul GRPO (base + LoRA adapter)...")
    grpo_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        trust_remote_code=True, dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    grpo_model = PeftModel.from_pretrained(grpo_base, str(grpo_path))
    print("[eval] Generez cu modelul GRPO...")
    grpo_emails = generate_batch(grpo_model, tokenizer, prompts)
    grpo_scores = score_emails(grpo_emails, prompts, args.reward)
    del grpo_model, grpo_base
    torch.cuda.empty_cache()

    # ── Afișare rezultate ────────────────────────────────────────────────
    print_comparison(prompts, base_emails, grpo_emails, base_scores, grpo_scores)

    # ── Salvare JSON ─────────────────────────────────────────────────────
    output = {
        "config": {"n": args.n, "reward": args.reward, "seed": args.seed},
        "summary": {
            "avg_reward_base": round(sum(s["reward"] for s in base_scores) / len(base_scores), 4),
            "avg_reward_grpo": round(sum(s["reward"] for s in grpo_scores) / len(grpo_scores), 4),
        },
        "samples": [
            {
                "locale": p["locale"], "fraud_stage": p["fraud_stage"],
                "base_email": be, "base_score": bs,
                "grpo_email": ge, "grpo_score": gs,
            }
            for p, be, bs, ge, gs in zip(prompts, base_emails, base_scores, grpo_emails, grpo_scores)
        ],
    }
    output["summary"]["delta"] = round(
        output["summary"]["avg_reward_grpo"] - output["summary"]["avg_reward_base"], 4
    )

    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] Rezultate salvate în {EVAL_OUTPUT}")


if __name__ == "__main__":
    main()
