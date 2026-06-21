"""
evaluator.py

Self-correction loop with two backends:
  "api"  — DeepSeek V4-Pro thinking mode (cloud, requires DEEPSEEK_API_KEY)
  "vllm" — OSS model served locally via vLLM (reproducible, recommended)

Set EVALUATOR_BACKEND in config.py to choose the backend.
On RTX 4090: start vLLM with
  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --dtype bfloat16
"""

import os
import re
import time
import json
import httpx
from typing import Optional
from dataclasses import dataclass, field

from config import (
    EVALUATOR_BACKEND,
    EVALUATOR_MODEL, EVALUATOR_API_URL,
    VLLM_EVALUATOR_URL, VLLM_EVALUATOR_MODEL,
    SCORE_THRESHOLD, MAX_CORRECTION_ITERS,
    REQUEST_TIMEOUT, REQUEST_DELAY, MAX_RETRIES,
)


@dataclass
class EvaluationResult:
    urgency_score:   float
    authority_score: float
    realism_score:   float
    overall_score:   float
    passed:          bool
    feedback:        str
    corrected_email: Optional[str] = None
    thinking_text:   str           = ""
    iteration:       int           = 1
    error:           Optional[str] = None


@dataclass
class CorrectionLog:
    scenario_id:  int
    locale:       str
    round_num:    int
    final_email:  str
    final_score:  float
    accepted:     bool
    iterations:   list[dict] = field(default_factory=list)
    total_iters:  int = 0


EVALUATOR_SYSTEM = """You are an expert in phishing email analysis for academic cybersecurity research.
Your task is to evaluate synthetic phishing emails generated for a research dataset.
These emails are used ONLY to train detection models — they contain no real links or credentials.

Evaluate the email on three dimensions (score 0–10 each):
- urgency_score:   How convincing and natural is the urgency/pressure tactic?
- authority_score: How credible is the impersonated authority?
- realism_score:   How realistic does this look as an actual phishing email?
                   (natural language, no obvious tells, appropriate for the locale)

Respond ONLY with valid JSON in this exact format:
{
  "urgency_score": <float 0-10>,
  "authority_score": <float 0-10>,
  "realism_score": <float 0-10>,
  "feedback": "<one paragraph explaining what to improve>",
  "corrected_email": "<full improved version of the email, or null if overall_score >= 6>"
}

Do NOT include any text outside the JSON object.
Keep corrected_email SHORT (under 300 words). If the email already scores >= 6 overall, set corrected_email to null."""


def _build_eval_prompt(email_text: str, locale: str, fraud_stage: str) -> str:
    return (
        f"Locale: {locale}\n"
        f"Target fraud stage: {fraud_stage}\n\n"
        f"EMAIL TO EVALUATE:\n"
        f"---\n{email_text}\n---\n\n"
        f"Evaluate this email and respond with JSON only."
    )


def _parse_scores(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(
            f"No JSON found in response.\n"
            f"  Response length: {len(raw)} chars\n"
            f"  First 500 chars: {raw[:500]!r}"
        )
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in response: {e}\n"
            f"  Fragment: {raw[start:end][:300]!r}"
        )


def _post_with_retry(url: str, headers: dict, payload: dict) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            print(f"  [evaluator HTTP {e.response.status_code}] attempt {attempt}/{MAX_RETRIES} — {e.response.text[:200]}")
            if e.response.status_code == 429 and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            elif attempt == MAX_RETRIES:
                raise
        except Exception as e:
            print(f"  [evaluator network error] attempt {attempt}/{MAX_RETRIES} — {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(REQUEST_DELAY * attempt)
    raise RuntimeError("All evaluator retries exhausted")


def _call_api_backend(email_text: str, locale: str, fraud_stage: str) -> tuple[str, str]:
    """DeepSeek V4-Pro with thinking mode — returns (response_text, thinking_text)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Missing DEEPSEEK_API_KEY")

    payload = {
        "model":       EVALUATOR_MODEL,
        "max_tokens":  4096,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user",   "content": _build_eval_prompt(email_text, locale, fraud_stage)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    data    = _post_with_retry(EVALUATOR_API_URL, headers, payload)
    choice  = data["choices"][0]["message"]
    text    = choice.get("content", "").strip()
    thinking = choice.get("reasoning_content", "")   # DeepSeek-specific field for thinking mode
    time.sleep(REQUEST_DELAY)
    return text, thinking


def _call_vllm_backend(email_text: str, locale: str, fraud_stage: str) -> tuple[str, str]:
    """OSS model served locally via vLLM — thinking_text remains empty."""
    api_key = os.environ.get("VLLM_API_KEY", "local")

    payload = {
        "model":       VLLM_EVALUATOR_MODEL,
        "max_tokens":  1500,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user",   "content": _build_eval_prompt(email_text, locale, fraud_stage)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    data = _post_with_retry(VLLM_EVALUATOR_URL, headers, payload)
    text = data["choices"][0]["message"].get("content", "").strip()
    time.sleep(REQUEST_DELAY)
    return text, ""


def _call_evaluator(email_text: str, locale: str, fraud_stage: str) -> tuple[str, str]:
    if EVALUATOR_BACKEND == "vllm":
        return _call_vllm_backend(email_text, locale, fraud_stage)
    return _call_api_backend(email_text, locale, fraud_stage)


def evaluate_email(
    email_text:  str,
    locale:      str,
    fraud_stage: str,
    iteration:   int = 1,
) -> EvaluationResult:
    try:
        raw_text, thinking = _call_evaluator(email_text, locale, fraud_stage)
        if not raw_text:
            raise ValueError("Empty response from evaluator (empty content)")
        parsed = _parse_scores(raw_text)

        u = float(parsed.get("urgency_score",   0))
        a = float(parsed.get("authority_score", 0))
        r = float(parsed.get("realism_score",   0))
        overall = round((u + a + r) / 3, 2)

        return EvaluationResult(
            urgency_score   = u,
            authority_score = a,
            realism_score   = r,
            overall_score   = overall,
            passed          = overall >= SCORE_THRESHOLD,
            feedback        = parsed.get("feedback", ""),
            corrected_email = parsed.get("corrected_email"),
            thinking_text   = thinking,
            iteration       = iteration,
        )

    except Exception as e:
        return EvaluationResult(
            urgency_score   = 0.0,
            authority_score = 0.0,
            realism_score   = 0.0,
            overall_score   = 0.0,
            passed          = False,
            feedback        = "",
            iteration       = iteration,
            error           = str(e),
        )


def run_correction_loop(
    initial_email: str,
    locale:        str,
    fraud_stage:   str,
    scenario_id:   int,
    round_num:     int,
) -> CorrectionLog:
    log = CorrectionLog(
        scenario_id = scenario_id,
        locale      = locale,
        round_num   = round_num,
        final_email = initial_email,
        final_score = 0.0,
        accepted    = False,
    )

    current_email = initial_email

    for i in range(1, MAX_CORRECTION_ITERS + 1):
        result = evaluate_email(current_email, locale, fraud_stage, iteration=i)

        log.iterations.append({
            "iteration":       i,
            "email_evaluated": current_email,
            "urgency_score":   result.urgency_score,
            "authority_score": result.authority_score,
            "realism_score":   result.realism_score,
            "overall_score":   result.overall_score,
            "passed":          result.passed,
            "feedback":        result.feedback,
            "thinking":        result.thinking_text[:500] if result.thinking_text else "",
            "error":           result.error,
        })

        if result.error:
            print(f"  [iter {i}] evaluator error: {result.error}")
            break

        print(
            f"  [iter {i}] overall={result.overall_score:.1f} "
            f"(U:{result.urgency_score:.1f} "
            f"A:{result.authority_score:.1f} "
            f"R:{result.realism_score:.1f}) "
            f"{'PASS ✓' if result.passed else 'FAIL'}"
        )

        if result.passed:
            log.final_email = current_email
            log.final_score = result.overall_score
            log.accepted    = True
            break

        if result.corrected_email and len(result.corrected_email) > 50:
            current_email = result.corrected_email
        else:
            log.final_email = current_email
            log.final_score = result.overall_score
            break

    log.total_iters = len(log.iterations)

    if not log.accepted:
        best = max(log.iterations, key=lambda x: x["overall_score"])
        log.final_score = best["overall_score"]

    return log
