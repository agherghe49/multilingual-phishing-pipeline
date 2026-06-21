"""
generator.py

Generates phishing emails via any OpenAI-compatible endpoint.
Supports any model registered in GENERATOR_MODELS in config.py.
"""

import os
import time
import httpx
from typing import Optional
from dataclasses import dataclass, asdict

from config import (
    GENERATOR_MODELS, DEFAULT_GENERATOR_MODEL,
    TEMPERATURE, MAX_TOKENS_GEN,
    REQUEST_TIMEOUT, REQUEST_DELAY, MAX_RETRIES,
)


@dataclass
class GenerationResult:
    email_text:        str
    model_used:        str
    locale:            str
    round_num:         int
    scenario_id:       int
    fraud_stage:       str
    prompt_tokens:     int
    completion_tokens: int
    latency_ms:        float
    success:           bool
    error: Optional[str] = None


def _call_api(url: str, headers: dict, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt
                print(f"  [rate limit] waiting {wait}s...")
                time.sleep(wait)
            elif attempt == MAX_RETRIES:
                raise
            else:
                time.sleep(REQUEST_DELAY * attempt)
        except (httpx.TimeoutException, httpx.ConnectError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(REQUEST_DELAY * attempt)
    raise RuntimeError("All retries exhausted")


def generate_email(
    system_prompt: str,
    user_prompt:   str,
    locale:        str,
    round_num:     int,
    scenario_id:   int,
    fraud_stage:   str,
    model_name:    str = DEFAULT_GENERATOR_MODEL,
    api_key:       Optional[str] = None,
) -> GenerationResult:
    """
    Generates a single phishing email via any model from GENERATOR_MODELS.

    Args:
        model_name: key from GENERATOR_MODELS (e.g. "deepseek-v4-flash", "vllm-local")
        api_key:    manual key override; otherwise read from the configured env variable
    """
    if model_name not in GENERATOR_MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(GENERATOR_MODELS)}"
        )

    cfg      = GENERATOR_MODELS[model_name]
    url      = cfg["api_url"]
    key      = api_key or os.environ.get(cfg["api_key_env"], "")
    model_id = cfg.get("model_id", model_name)  # model_id for local vLLM (may differ from CLI key)

    if not key and "localhost" not in url:
        raise EnvironmentError(
            f"Missing {cfg['api_key_env']} from environment variables"
        )

    headers = {
        "Authorization": f"Bearer {key or 'local'}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model_id,
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS_GEN,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }

    start = time.monotonic()
    try:
        data    = _call_api(url, headers, payload)
        latency = (time.monotonic() - start) * 1000

        text  = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        time.sleep(REQUEST_DELAY)

        return GenerationResult(
            email_text        = text,
            model_used        = model_name,
            locale            = locale,
            round_num         = round_num,
            scenario_id       = scenario_id,
            fraud_stage       = fraud_stage,
            prompt_tokens     = usage.get("prompt_tokens", 0),
            completion_tokens = usage.get("completion_tokens", 0),
            latency_ms        = round(latency, 1),
            success           = True,
        )

    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return GenerationResult(
            email_text        = "",
            model_used        = model_name,
            locale            = locale,
            round_num         = round_num,
            scenario_id       = scenario_id,
            fraud_stage       = fraud_stage,
            prompt_tokens     = 0,
            completion_tokens = 0,
            latency_ms        = round(latency, 1),
            success           = False,
            error             = str(e),
        )


def result_to_dict(r: GenerationResult) -> dict:
    return asdict(r)
