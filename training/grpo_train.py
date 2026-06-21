"""
training/grpo_train.py

Fine-tunează un model mic (Qwen2.5-7B-Instruct) cu GRPO (Group Relative
Policy Optimization) pentru a genera emailuri phishing de calitate mai
înaltă, fără a mai depinde de correction loop la inferență.

Arhitectura:
  - Policy model:     Qwen2.5-7B-Instruct cu QLoRA (4-bit NF4)
  - Reference model:  aceeași bază înghețată (partajat prin PEFT)
  - Reward compus:    0.5 * quality + 0.3 * diversity + 0.2 * format
    - quality:   evaluator via DeepSeek API (--reward api)
                 sau heuristică lightweight (--reward heuristic)
    - diversity: penalizează repetarea față de celelalte completări din grup
    - format:    verifică structura email (salut, corp ≥80 cuvinte, sign-off)
    - GATE:      emailuri sub min_length cuvinte (implicit 80) primesc reward=0
  - Dataset:          scenariile KB transformate în prompturi de generare

Hardware (auto-detectat la pornire):
  - RTX 5090 / ≥28GB → profil "high":   rank=32, G=8, fără gradient checkpointing
  - RTX 4090 / 20-28GB → profil "medium": rank=16, G=4, gradient checkpointing + paged_adamw_8bit
  - <20GB → profil "low":               rank=8,  G=2, gradient checkpointing + paged_adamw_8bit

Rulare:
  python training/grpo_train.py
  python training/grpo_train.py --reward api        # DeepSeek evaluator (recomandat)
  python training/grpo_train.py --reward heuristic  # fără apel API extra
  python training/grpo_train.py --steps 500 --group-size 8
"""

import sys
import json
import re
import os
import argparse
import httpx
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    KB_DIR, OUTPUT_DIR,
    EVALUATOR_API_URL,
    LOCALES, MAX_ROUNDS,
)
from prompts import build_prompt
from orchestrator import _infer_fraud_stage

GRPO_OUTPUT_DIR = OUTPUT_DIR / "grpo_model"

DEEPSEEK_MODEL_ID = "deepseek-v4-flash"


# ── Detectare hardware și profil GPU ─────────────────────────────────────────

def detect_gpu_profile() -> dict:
    """
    Detectează GPU-ul disponibil și returnează un profil de configurare GRPO.

    Profile:
      high   — ≥28GB VRAM (RTX 5090, A100 40GB, etc.)
      medium — 20-28GB   (RTX 4090 24GB, RTX 3090 24GB)
      low    — <20GB     (RTX 3080 10GB, RTX 4080 16GB, etc.)

    Returnează un dict cu toate setările dependente de hardware.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no cuda")
        props     = torch.cuda.get_device_properties(0)
        vram_gb   = props.total_memory / (1024 ** 3)
        gpu_name  = props.name
    except Exception:
        vram_gb  = 0.0
        gpu_name = "CPU/unknown"

    if vram_gb >= 28:
        profile = "high"
        cfg = {
            "lora_rank":              32,
            "lora_alpha":             64,
            "num_generations":        8,
            "gradient_checkpointing": False,
            "optim":                  "adamw_torch",
        }
    elif vram_gb >= 20:
        profile = "medium"
        cfg = {
            "lora_rank":              16,
            "lora_alpha":             32,
            "num_generations":        4,
            "gradient_checkpointing": True,
            "optim":                  "paged_adamw_8bit",
        }
    else:
        profile = "low"
        cfg = {
            "lora_rank":              8,
            "lora_alpha":             16,
            "num_generations":        2,
            "gradient_checkpointing": True,
            "optim":                  "paged_adamw_8bit",
        }

    cfg["vram_gb"]  = round(vram_gb, 1)
    cfg["gpu_name"] = gpu_name
    cfg["profile"]  = profile
    return cfg

# ── Dataset de antrenament GRPO ───────────────────────────────────────────

def build_grpo_dataset(max_samples: int = 2000) -> list[dict]:
    """
    Construiește dataset-ul de prompturi pentru GRPO din scenariile KB.
    Fiecare înregistrare are câmpul 'prompt' (format chat messages pentru TRL).
    """
    scenarios = []
    for path in sorted(KB_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        scenarios.extend(items)

    records = []
    for scenario in scenarios:
        for locale in LOCALES:
            for round_num in range(1, MAX_ROUNDS + 1):
                topic       = scenario.get("subcategory", "account verification")
                fraud_stage = _infer_fraud_stage(scenario, round_num)
                prompt_d    = build_prompt(
                    round_num    = round_num,
                    topic        = topic,
                    fraud_stage  = fraud_stage,
                    context_docs = [],       # fără RAG în GRPO (cost GPU)
                    locale       = locale,
                )
                records.append({
                    "prompt": [
                        {"role": "system", "content": prompt_d["system"]},
                        {"role": "user",   "content": prompt_d["user"]},
                    ],
                    "_locale":      locale,
                    "_fraud_stage": fraud_stage,
                    "_scenario_id": scenario.get("id", 0),
                })
                if len(records) >= max_samples:
                    return records
    return records


# ── Componente reward ─────────────────────────────────────────────────────

def _heuristic_quality(email_text: str, locale: str, fraud_stage: str) -> float:
    """Scor de calitate rapid bazat pe euristici lingvistice. Returnează [0, 1]."""
    text_lower = email_text.lower()
    score = 0.0

    urgency_kw = ["urgent", "immediately", "expires", "within", "hours",
                  "deadline", "warning", "⚠", "🚨", "limited time", "act now",
                  "imediat", "termen", "expiră", "dringend", "sofort",
                  "immédiatement", "urgente", "subito"]
    score += min(sum(1 for kw in urgency_kw if kw in text_lower) * 0.8, 3.0)

    authority_kw = ["department", "official", "compliance", "regulation",
                    "verified", "authority", "ministry", "bank", "government",
                    "departament", "oficial", "conformitate", "Behörde",
                    "officiel", "réglementation", "ufficiale", "normativa"]
    score += min(sum(1 for kw in authority_kw if kw in text_lower) * 0.8, 3.0)

    words = len(email_text.split())
    if 80 <= words <= 300:
        score += 2.0
    elif words > 50:
        score += 1.0

    obvious_tells = ["[url]", "[phone]", "<<<", "xxx", "!!!"]
    score -= sum(1 for t in obvious_tells if t in text_lower) * 0.5

    return max(0.0, min(score / 10.0, 1.0))


def _api_quality(email_text: str, locale: str, fraud_stage: str) -> float:
    """
    Scor de calitate via DeepSeek Flash API.
    Returnează medie normalizată (urgency + authority + realism) / 30 → [0, 1].
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return _heuristic_quality(email_text, locale, fraud_stage)

    system = (
        "You are a phishing email quality evaluator for academic research. "
        "Score the email on three axes (0-10 each): "
        "urgency (creates time pressure), "
        "authority (impersonates trusted entity), "
        "realism (looks like a real email, not obviously fake). "
        'Respond ONLY with JSON: {"urgency": <float>, "authority": <float>, "realism": <float>}'
    )
    user = (
        f"Locale: {locale}\nFraud stage: {fraud_stage}\n\n"
        f"EMAIL:\n---\n{email_text[:800]}\n---\n\nScore JSON only."
    )

    try:
        resp = httpx.post(
            EVALUATOR_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       DEEPSEEK_MODEL_ID,
                "messages":    [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens":  80,
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        u = float(parsed.get("urgency",   0))
        a = float(parsed.get("authority", 0))
        r = float(parsed.get("realism",   0))
        return round((u + a + r) / 30.0, 4)
    except Exception:
        return _heuristic_quality(email_text, locale, fraud_stage)


def _diversity_scores(texts: list[str]) -> list[float]:
    """
    Penalizează completările similare în cadrul aceluiași grup GRPO.
    Scor = 1 - max_cosine_similarity(text_i, celelalte texte din grup).
    """
    if len(texts) <= 1:
        return [1.0] * len(texts)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        import numpy as np

        vec  = TfidfVectorizer(max_features=500).fit_transform(texts)
        sims = cos_sim(vec)
        scores = []
        for i in range(len(texts)):
            others  = [sims[i][j] for j in range(len(texts)) if j != i]
            max_sim = float(max(others)) if others else 0.0
            scores.append(round(1.0 - max_sim, 4))
        return scores
    except Exception:
        return [1.0] * len(texts)


def _format_score(text: str) -> float:
    """Verifică structura minimă a unui email: salut, corp ≥80 cuvinte, sign-off."""
    score = 0.0

    greeting = r"(dear|hi\b|hello|stimate|bună ziua|sehr geehrte|cher|gentile|buongiorno)"
    if re.search(greeting, text, re.IGNORECASE):
        score += 0.4

    if len(text.split()) >= 80:
        score += 0.4
    elif len(text.split()) >= 40:
        score += 0.2

    signoff = r"(regards|sincerely|best wishes|cu stimă|cu respect|mit freundlichen|cordialement|cordiali saluti)"
    if re.search(signoff, text, re.IGNORECASE):
        score += 0.2

    return min(score, 1.0)


# ── Reward function compusă ───────────────────────────────────────────────

def _extract_text(completion) -> str:
    """Normalizează completarea la string — trl ≥1.4 poate pasa liste de mesaje chat."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return " ".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in completion)
    return str(completion)


def make_reward_fn(reward_type: str, min_length: int = 80):
    """Returnează funcția de reward compatibilă cu TRL GRPOTrainer.

    Gate de lungime: emailuri sub min_length cuvinte primesc reward=0,
    eliminând outputurile degenerate fără gradient signal fals.
    Weights: 0.5×quality + 0.3×diversity + 0.2×format (calitate prioritizată).
    """

    def reward_fn(completions, prompts=None, **kwargs) -> list[float]:
        texts        = [_extract_text(c) for c in completions]
        locales      = kwargs.get("_locale",      ["en-US"] * len(texts))
        fraud_stages = kwargs.get("_fraud_stage",  ["authority"] * len(texts))

        # Gate de lungime minimă — elimină outputurile degenerate
        word_counts = [len(t.split()) for t in texts]

        quality_scores = []
        for text, locale, stage in zip(texts, locales, fraud_stages):
            if reward_type == "api":
                q = _api_quality(text, locale, stage)
            else:
                q = _heuristic_quality(text, locale, stage)
            quality_scores.append(q)

        div_scores = _diversity_scores(texts)
        fmt_scores = [_format_score(t) for t in texts]

        rewards = []
        for wc, q, d, f in zip(word_counts, quality_scores, div_scores, fmt_scores):
            if wc < min_length:
                rewards.append(0.0)   # penalizare hard pentru outputuri prea scurte
            else:
                rewards.append(round(0.5 * q + 0.3 * d + 0.2 * f, 4))
        return rewards

    return reward_fn


# ── Training ───────────────────────────────────────────────────────────────

def train(
    base_model:  str            = "Qwen/Qwen2.5-7B-Instruct",
    reward_type: str            = "heuristic",
    max_steps:   int            = 500,
    group_size:  int            = 0,   # 0 = auto din detect_gpu_profile()
    seed:        int            = 42,
    resume_from: Optional[str]  = None,
    min_length:  int            = 80,  # gate lungime minimă email (cuvinte)
) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from trl import GRPOTrainer, GRPOConfig
        from peft import LoraConfig
        from datasets import Dataset as HFDataset
    except ImportError as e:
        print(f"Eroare import: {e}")
        print(
            "Instalează dependințele:\n"
            "  pip install trl peft bitsandbytes transformers datasets scikit-learn"
        )
        return

    # ── Auto-detectare GPU și profil hardware ─────────────────────────────
    gpu = detect_gpu_profile()
    print(f"[GRPO] GPU detectat:  {gpu['gpu_name']} ({gpu['vram_gb']} GB) → profil '{gpu['profile']}'")

    # group_size 0 înseamnă "auto" — se preia din profilul GPU
    effective_group_size = group_size if group_size > 0 else gpu["num_generations"]

    print(f"[GRPO] Base model:   {base_model}")
    print(f"[GRPO] Reward:       {reward_type}")
    print(f"[GRPO] Max steps:    {max_steps}")
    print(f"[GRPO] Group size:   {effective_group_size}  (LoRA rank={gpu['lora_rank']}, optim={gpu['optim']})")
    print(f"[GRPO] Min length:   {min_length} cuvinte (gate reward=0 sub prag)")
    print(f"[GRPO] Output dir:   {GRPO_OUTPUT_DIR}\n")

    # ── QLoRA config (4-bit NF4 + double quant) ──────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_compute_dtype     = torch.bfloat16,
        bnb_4bit_use_double_quant  = True,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # necesar pentru generare batch GRPO

    # ── Model (încărcat explicit — trl ≥1.4 nu mai acceptă model_init_kwargs) ──
    print("[GRPO] Încarc modelul cu QLoRA 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_config,
        trust_remote_code   = True,
        dtype               = torch.bfloat16,
        token               = os.environ.get("HF_TOKEN"),
    )

    # ── LoRA config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r              = gpu["lora_rank"],
        lora_alpha     = gpu["lora_alpha"],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )

    # ── GRPO config ────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir                  = str(GRPO_OUTPUT_DIR),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        num_generations             = effective_group_size,
        max_completion_length       = 600,   # trl ≥1.4: înlocuiește max_new_tokens
        learning_rate               = 1e-5,
        num_train_epochs            = 1,
        max_steps                   = max_steps,
        bf16                        = True,
        gradient_checkpointing      = gpu["gradient_checkpointing"],
        optim                       = gpu["optim"],
        warmup_steps                = max(1, int(max_steps * 0.1)),
        logging_steps               = 10,
        save_steps                  = 100,
        seed                        = seed,
        report_to                   = "none",
        remove_unused_columns       = False,
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    raw_records = build_grpo_dataset(max_samples=max_steps * effective_group_size * 2)
    dataset     = HFDataset.from_list(raw_records)
    print(f"[GRPO] Dataset: {len(dataset)} prompturi\n")

    # ── Reward function ───────────────────────────────────────────────────
    reward_fn = make_reward_fn(reward_type, min_length=min_length)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model             = model,
        args              = grpo_config,
        train_dataset     = dataset,
        reward_funcs      = [reward_fn],
        processing_class  = tokenizer,   # trl ≥1.4: înlocuiește 'tokenizer'
        peft_config       = lora_config,
    )

    if resume_from:
        print(f"[GRPO] Reluând din checkpoint: {resume_from}")
    print("[GRPO] Pornesc antrenamentul...")
    trainer.train(resume_from_checkpoint=resume_from)

    GRPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(GRPO_OUTPUT_DIR))
    tokenizer.save_pretrained(str(GRPO_OUTPUT_DIR))
    print(f"\n[GRPO] Model salvat în {GRPO_OUTPUT_DIR}")
    print(
        "Pentru inferență:\n"
        "  from peft import PeftModel\n"
        f"  model = PeftModel.from_pretrained(base_model, '{GRPO_OUTPUT_DIR}')"
    )


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning pentru generator de emailuri phishing"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model HuggingFace de bază (implicit: Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--reward",
        choices=["heuristic", "api"],
        default="heuristic",
        help=(
            "Tipul funcției de reward:\n"
            "  heuristic — rapid, fără apel API extra\n"
            "  api       — calitate mai bună, folosește DeepSeek Flash (necesită DEEPSEEK_API_KEY)"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Numărul maxim de pași de antrenament",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help=(
            "Dimensiunea grupului GRPO (G). "
            "0 (implicit) = auto din profilul GPU detectat (4090→4, 5090→8)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint din care să se reia (ex: outputs/grpo_model/checkpoint-200)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=80,
        help="Lungime minimă email (cuvinte) — emailuri mai scurte primesc reward=0",
    )
    args = parser.parse_args()

    train(
        base_model  = args.base_model,
        reward_type = args.reward,
        max_steps   = args.steps,
        group_size  = args.group_size,
        seed        = args.seed,
        resume_from = args.resume,
        min_length  = args.min_length,
    )
