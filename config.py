"""
config.py — toate setările pipeline-ului într-un singur loc.
Modifică DOAR acest fișier când vrei să schimbi modele, praguri sau volume.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
KB_DIR     = BASE_DIR / "knowledge_base"
NEG_DIR    = DATA_DIR / "negatives"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH    = OUTPUT_DIR / "dataset.jsonl"
AUDIT_LOG_PATH  = OUTPUT_DIR / "audit_log.jsonl"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"

# ── Modele generator ───────────────────────────────────────────────────────
# Orice endpoint compatibil OpenAI poate fi adăugat.
# Cheie = numele din --model CLI arg.
GENERATOR_MODELS: dict[str, dict] = {
    "deepseek-v4-flash": {
        "api_url":     "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "kimi-k2.6": {
        "api_url":     "https://api.moonshot.cn/v1/chat/completions",
        "api_key_env": "KIMI_API_KEY",
    },
    "qwen2.5-72b-instruct": {
        "api_url":     "https://api.together.xyz/v1/chat/completions",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "llama-3.3-70b-instruct": {
        "api_url":     "https://api.together.xyz/v1/chat/completions",
        "api_key_env": "TOGETHER_API_KEY",
    },
    # model servit local prin vLLM (ex. Qwen2.5-7B pe 4090)
    # model_id = numele real trimis în API payload (diferit de cheia CLI)
    "vllm-local": {
        "api_url":     "http://localhost:8000/v1/chat/completions",
        "api_key_env": "VLLM_API_KEY",
        "model_id":    "Qwen/Qwen2.5-7B-Instruct",
    },
}

DEFAULT_GENERATOR_MODEL = "deepseek-v4-flash"

# ── Evaluator ─────────────────────────────────────────────────────────────
# "api"  = DeepSeek V4-Pro thinking mode (necesită DEEPSEEK_API_KEY)
# "vllm" = model OSS servit local prin vLLM (complet reproductibil, recomandat)
EVALUATOR_BACKEND = "api"

# Backend API (DeepSeek)
EVALUATOR_MODEL   = "deepseek-v4-flash"   # v4-pro era prea lent (thinking mode); flash e 5-10x mai rapid
EVALUATOR_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Backend vLLM (OSS, recomandat pentru reproductibilitate)
# Pe RTX 4090 (24GB): Qwen2.5-7B-Instruct (FP16 ~14GB) sau
#   Qwen2.5-14B-Instruct-AWQ (INT4 ~8GB) — lasă memorie pentru GRPO
# Pornire: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --dtype bfloat16
VLLM_EVALUATOR_URL   = "http://localhost:8001/v1/chat/completions"
VLLM_EVALUATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── Embedder RAG ───────────────────────────────────────────────────────────
# (model_hf_name, embedding_dim)
# Comparate pe MTEB Retrieval — qwen3 și nemotron sunt top performers
# Atenție: nemotron (NV-Embed-v2, ~14GB FP16) concurează cu vLLM pe VRAM 4090
EMBEDDER_OPTIONS: dict[str, tuple[str, int]] = {
    "qwen3":    ("Qwen/Qwen3-Embedding-0.6B",               1024),
    "nemotron": ("nvidia/NV-Embed-v2",                      4096),
    "minilm":   ("sentence-transformers/all-MiniLM-L6-v2",  384),
}
DEFAULT_EMBEDDER = "qwen3"

EMBEDDER_MODEL = EMBEDDER_OPTIONS[DEFAULT_EMBEDDER][0]
EMBEDDING_DIM  = EMBEDDER_OPTIONS[DEFAULT_EMBEDDER][1]

# ── Parametri generare ─────────────────────────────────────────────────────
LOCALES        = ["en-US", "ro-RO", "de-DE", "fr-FR", "it-IT"]
MAX_ROUNDS     = 4
TEMPERATURE    = 0.85
MAX_TOKENS_GEN = 1100

# ── Parametri self-correction ──────────────────────────────────────────────
MAX_CORRECTION_ITERS = 3
SCORE_THRESHOLD      = 6.0

# ── Parametri volum ────────────────────────────────────────────────────────
TARGET_PHISHING = 2000
TARGET_HAM      = 2000

# ── Request settings ───────────────────────────────────────────────────────
REQUEST_TIMEOUT = 60
REQUEST_DELAY   = 0.5
MAX_RETRIES     = 3

# ── FAISS index ────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = OUTPUT_DIR / "faiss_index"
TOP_K_DOCS       = 2

# ── Deduplication ──────────────────────────────────────────────────────────
# Hash SHA1 pe primele N caractere pentru near-dedup în checkpoint
DEDUP_HASH_CHARS = 150
