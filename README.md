# Multilingual Phishing Email Generation and Detection

An end-to-end pipeline for **synthetic multilingual phishing email generation** and **automatic detection**, combining Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), self-correction, and GRPO fine-tuning.

Part of the Master's thesis *"Generarea și detecția atacurilor phishing via e-mail"* — Politehnica University of Bucharest, 2025–2026.

---

## Overview

This project addresses two complementary problems:

1. **Generation** — producing a realistic multilingual dataset of synthetic phishing emails (5 languages) using an LLM + RAG + self-correction pipeline
2. **Detection** — training and evaluating classifiers (TF-IDF+LR, mDistilBERT, XLM-RoBERTa, mDeBERTa-v3, ModernBERT) with a systematic scaling laws analysis
3. **Adversarial fine-tuning** — improving the generator with GRPO (Group Relative Policy Optimization) to produce phishing that evades standard detectors

**Final dataset**: 16,818 emails across five European languages (Romanian, English, German, French, Italian).

**Key results:**
- All classifiers converge at F1 ≈ 1.0 from n = 500 training examples
- Mixed training with only 17% real data resolves a 98% FNR cross-domain gap
- GRPO fine-tuning produces phishing evading detection in 51% of cases
- Defenders adapt completely after a single retraining round with 200 examples

---

## Project Structure

```
.
├── config.py                  # All settings: models, thresholds, paths
├── orchestrator.py            # Main entry point for phishing generation
├── generator.py               # OpenAI-compatible API client
├── prompts.py                 # Locale-aware prompt construction + RAG context
├── evaluator.py               # Self-correction loop (urgency/authority/realism scoring)
├── retriever.py               # FAISS index with Qwen3-Embedding-0.6B
├── requirements.txt
├── setup.sh                   # Dependency installation (auto-detects GPU SM)
├── test_retriever.py
│
├── knowledge_base/
│   ├── fp_base.json           # Fraud scenarios: authority stage (rounds 1–2)
│   └── fp_levelup.json        # Fraud scenarios: urgency/payment stage (rounds 3–4)
│
├── data/
│   ├── generate_ham.py        # Synthetic legitimate email generation
│   └── assemble.py            # Dataset assembly from multiple sources
│
├── experiments/
│   ├── split_dataset.py       # Train/test split (80/20, stratified)
│   ├── old_data_baseline.py   # SpamAssassin 2003 vs. synthetic data comparison
│   ├── scaling_laws.py        # F1/FNR/PR-AUC vs. training volume (100→5000)
│   ├── adversarial_eval.py    # Classifier evaluation on GRPO-generated phishing
│   ├── adversarial_loop.py    # Iterative attacker-defender game (3 rounds)
│   ├── mixed_training_eval.py # Cross-domain gap + mixed training experiment
│   ├── cross_locale_transfer.py # Cross-lingual transferability evaluation
│   ├── explainability_analysis.py # LIME + Integrated Gradients
│   ├── linguistic_analysis.py # Urgency/authority/threat density per locale
│   └── ...
│
├── training/
│   ├── grpo_train.py          # GRPO fine-tuning (Qwen2.5-7B, QLoRA 4-bit)
│   └── eval_grpo.py           # Base vs. GRPO model comparison
│
└── streamlit_app/
    └── app.py                 # Interactive demo
```

---

## Setup

### 1. Install dependencies

```bash
bash setup.sh
# Auto-detects GPU and installs the appropriate PyTorch build
conda activate rag
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env with your credentials
source .env
```

Required keys depend on your chosen backend (see `.env.example`):
- `DEEPSEEK_API_KEY` — for DeepSeek generator/evaluator
- `HF_TOKEN` — for downloading HuggingFace models
- `TOGETHER_API_KEY` / `KIMI_API_KEY` — for alternative generators
- `VLLM_API_KEY` — for local vLLM serving (can be a placeholder)

---

## Usage

### Generate phishing emails

```bash
# Full run (all locales, all scenarios)
python orchestrator.py

# Quick test — 10 emails, Romanian only
python orchestrator.py --limit 10 --locale ro-RO

# Resume after interruption (automatic via checkpoint)
python orchestrator.py
```

### Generate ham emails

```bash
python data/generate_ham.py --n 3000
python data/generate_ham.py --n 600 --locale en-US
```

### Prepare dataset splits

```bash
python experiments/split_dataset.py
# → outputs/train.jsonl  (80%, stratified by locale + label)
# → outputs/test.jsonl   (20%, balanced)
```

### Run detection experiments

```bash
# Baseline: legacy data vs. synthetic
python experiments/old_data_baseline.py

# Scaling laws (all models)
python experiments/scaling_laws.py --model all --steps 100,250,500,1000,2500,5000

# Single model
python experiments/scaling_laws.py --model xlm-roberta --steps 100,500,1000

# Cross-domain + mixed training
python experiments/mixed_training_eval.py

# Cross-locale transferability
python experiments/cross_locale_transfer.py

# Explainability (LIME + Integrated Gradients)
python experiments/explainability_analysis.py
```

**Available model keys:**

| Key | Model | Type |
|-----|-------|------|
| `lr` | TF-IDF + Logistic Regression | baseline |
| `distilbert-ml` | distilbert-base-multilingual-cased | multilingual |
| `xlm-roberta` | xlm-roberta-base | multilingual |
| `mdeberta` | microsoft/mdeberta-v3-base | multilingual |
| `modernbert` | answerdotai/ModernBERT-base | en-US only |

### GRPO fine-tuning

```bash
# Quick test (no API calls, heuristic reward)
python training/grpo_train.py --reward heuristic --steps 50

# Full run with API evaluator
python training/grpo_train.py --reward api --steps 600

# Resume from checkpoint
python training/grpo_train.py --reward api --steps 600 \
    --resume outputs/grpo_model/checkpoint-400
```

Hardware tested: RTX 4090 (24 GB VRAM) — QLoRA NF4, LoRA rank=16, batch=1, G=4 → ~18 GB VRAM.

---

## Dataset Format

Each entry in `dataset.jsonl`:

```json
{
  "id":              "deepseek-v4-flash_42_ro-RO_3_1714000000",
  "email_text":      "Stimate utilizator,\n\nContul dumneavoastră...",
  "label":           1,
  "locale":          "ro-RO",
  "fraud_stage":     "urgency_pressure",
  "round_num":       3,
  "scenario_id":     42,
  "source":          "synthetic_phishing",
  "final_score":     7.8,
  "accepted":        true,
  "total_iters":     2,
  "generator_model": "deepseek-v4-flash",
  "generated_at":    "2026-05-17T10:00:00+00:00"
}
```

| Field | Values | Description |
|-------|--------|-------------|
| `label` | `0` / `1` | 0 = legitimate ham, 1 = phishing |
| `locale` | `en-US` / `ro-RO` / `de-DE` / `fr-FR` / `it-IT` | Email language |
| `fraud_stage` | `initial_contact` / `trust_building` / `urgency_pressure` / `credential_harvest` / `payment_extraction` | Attack escalation stage |
| `source` | `synthetic_phishing` / `synthetic_ham` / `spamassassin_ham` | Data origin |
| `accepted` | `true` / `false` | Passed quality gate (`final_score ≥ 6.0`) |

---

## Configuration

All settings are centralized in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_GENERATOR_MODEL` | `deepseek-v4-flash` | LLM used for generation |
| `EVALUATOR_BACKEND` | `api` | `api` (DeepSeek) or `vllm` (local OSS) |
| `DEFAULT_EMBEDDER` | `qwen3` | Embedding model for RAG |
| `LOCALES` | 5 languages | `en-US`, `ro-RO`, `de-DE`, `fr-FR`, `it-IT` |
| `SCORE_THRESHOLD` | `6.0` | Minimum quality gate score (0–10) |
| `MAX_CORRECTION_ITERS` | `3` | Max self-correction iterations per email |
| `TEMPERATURE` | `0.85` | Generator temperature |
| `TOP_K_DOCS` | `2` | RAG: number of retrieved scenarios |

---

## Ethical Statement

This repository is released for **academic and defensive security research only**. The generated phishing emails and pipeline code are intended to:
- Train and benchmark phishing detection classifiers
- Study attacker-defender dynamics in a controlled research setting
- Enable organizations to build multilingual phishing detectors with minimal labeled data

Any use of this code or dataset to conduct actual phishing attacks is strictly prohibited and likely illegal under applicable law.

---

## Citation

If you use this code or dataset in your research, please cite:

```
Gherghe, G.-A., Dascălu, M. (2026). Multilingual Phishing Email Generation and
Detection: A Synthetic Data Pipeline with GRPO Fine-Tuning.
In Proceedings of SYNASC 2026.
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
