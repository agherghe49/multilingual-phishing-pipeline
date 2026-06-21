"""
config.py — all pipeline settings in one place.
Only modify THIS file to change models, thresholds, or volumes.
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

# ── Generator models ───────────────────────────────────────────────────────
# Any OpenAI-compatible endpoint can be added.
# Key = name used in the --model CLI arg.
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
    # model served locally via vLLM (e.g. Qwen2.5-7B on RTX 4090)
    # model_id = actual name sent in API payload (may differ from the CLI key)
    "vllm-local": {
        "api_url":     "http://localhost:8000/v1/chat/completions",
        "api_key_env": "VLLM_API_KEY",
        "model_id":    "Qwen/Qwen2.5-7B-Instruct",
    },
}

DEFAULT_GENERATOR_MODEL = "deepseek-v4-flash"

# ── Evaluator ─────────────────────────────────────────────────────────────
# "api"  = DeepSeek V4-Pro thinking mode (requires DEEPSEEK_API_KEY)
# "vllm" = OSS model served locally via vLLM (fully reproducible, recommended)
EVALUATOR_BACKEND = "api"

# API backend (DeepSeek)
EVALUATOR_MODEL   = "deepseek-v4-flash"   # v4-pro was too slow (thinking mode); flash is 5–10× faster
EVALUATOR_API_URL = "https://api.deepseek.com/v1/chat/completions"

# vLLM backend (OSS, recommended for reproducibility)
# On RTX 4090 (24 GB): Qwen2.5-7B-Instruct (FP16 ~14 GB) or
#   Qwen2.5-14B-Instruct-AWQ (INT4 ~8 GB) — leaves room for GRPO
# Start: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --dtype bfloat16
VLLM_EVALUATOR_URL   = "http://localhost:8001/v1/chat/completions"
VLLM_EVALUATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── RAG embedder ───────────────────────────────────────────────────────────
# (model_hf_name, embedding_dim)
# Compared on MTEB Retrieval — qwen3 and nemotron are top performers
# Note: nemotron (NV-Embed-v2, ~14 GB FP16) competes with vLLM for VRAM on the 4090
EMBEDDER_OPTIONS: dict[str, tuple[str, int]] = {
    "qwen3":    ("Qwen/Qwen3-Embedding-0.6B",               1024),
    "nemotron": ("nvidia/NV-Embed-v2",                      4096),
    "minilm":   ("sentence-transformers/all-MiniLM-L6-v2",  384),
}
DEFAULT_EMBEDDER = "qwen3"

EMBEDDER_MODEL = EMBEDDER_OPTIONS[DEFAULT_EMBEDDER][0]
EMBEDDING_DIM  = EMBEDDER_OPTIONS[DEFAULT_EMBEDDER][1]

# ── Generation parameters ──────────────────────────────────────────────────
LOCALES        = ["en-US", "ro-RO", "de-DE", "fr-FR", "it-IT"]
MAX_ROUNDS     = 4
TEMPERATURE    = 0.85
MAX_TOKENS_GEN = 1100

# ── Self-correction parameters ─────────────────────────────────────────────
MAX_CORRECTION_ITERS = 3
SCORE_THRESHOLD      = 6.0

# ── Volume parameters ──────────────────────────────────────────────────────
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
# SHA1 hash of the first N characters for near-dedup in checkpoint
DEDUP_HASH_CHARS = 150
