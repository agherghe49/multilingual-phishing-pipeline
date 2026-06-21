"""
orchestrator.py

Coordinates the full generation pipeline:
  KB → RAG retrieval → prompt → generation → self-correction → checkpoint

Usage examples:
  python orchestrator.py                             # DeepSeek V4-Flash, all locales
  python orchestrator.py --model kimi-k2.6           # ablation: Kimi K2.6
  python orchestrator.py --model vllm-local          # local OSS model via vLLM
  python orchestrator.py --model qwen2.5-72b-instruct --embedder nemotron
  python orchestrator.py --locale ro-RO --limit 100  # quick test
  python orchestrator.py --no-resume                 # clean start
"""

import os
import json
import time
import hashlib
import argparse
import random
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

from config import (
    KB_DIR, OUTPUT_DIR, DATASET_PATH, AUDIT_LOG_PATH, CHECKPOINT_PATH,
    LOCALES, MAX_ROUNDS, GENERATOR_MODELS, DEFAULT_GENERATOR_MODEL,
    EMBEDDER_OPTIONS, DEFAULT_EMBEDDER,
    EVALUATOR_BACKEND, VLLM_EVALUATOR_MODEL,
    DEDUP_HASH_CHARS,
)
from generator import generate_email, result_to_dict
from prompts import build_prompt
from evaluator import run_correction_loop
from retriever import RAGRetriever


# ── Helpers ───────────────────────────────────────────────────────────────

def load_knowledge_base() -> list[dict]:
    scenarios = []
    for path in sorted(KB_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            scenarios.extend(data)
        else:
            scenarios.append(data)
    print(f"[KB] {len(scenarios)} scenarios loaded from {KB_DIR}")
    return scenarios


def load_checkpoint() -> tuple[set[str], set[str]]:
    """Returns (job_ids_done, content_hashes_done) for resume and dedup."""
    if not CHECKPOINT_PATH.exists():
        return set(), set()
    with open(CHECKPOINT_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return set(data.get("done", [])), set(data.get("hashes", []))
    # legacy format (plain list)
    return set(data), set()


def save_checkpoint(done: set[str], hashes: set[str]) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"done": list(done), "hashes": list(hashes)}, f)


def job_id(scenario_id: int, locale: str, round_num: int, model: str) -> str:
    return f"{model}_{scenario_id}_{locale}_{round_num}"


def content_hash(text: str) -> str:
    return hashlib.sha1(text[:DEDUP_HASH_CHARS].encode()).hexdigest()


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_done_per_locale(done: set[str], locales: list[str]) -> dict[str, int]:
    """Counts completed jobs per locale from checkpoint."""
    counts: dict[str, int] = {loc: 0 for loc in locales}
    for jid in done:
        for loc in locales:
            if f"_{loc}_" in jid:
                counts[loc] += 1
                break
    return counts


def build_task_list(
    scenarios:        list[dict],
    locales:          list[str],
    done:             set[str],
    model_name:       str,
    limit:            int | None = None,
    limit_per_locale: int | None = None,
) -> list[tuple]:
    done_per_locale = count_done_per_locale(done, locales) if limit_per_locale else {}

    per_locale: dict[str, list[tuple]] = {loc: [] for loc in locales}
    for scenario in scenarios:
        sid = scenario["id"]
        for locale in locales:
            for round_num in range(1, MAX_ROUNDS + 1):
                jid = job_id(sid, locale, round_num, model_name)
                if jid not in done:
                    per_locale[locale].append((scenario, locale, round_num))

    tasks = []
    for loc in locales:
        bucket = per_locale[loc]
        random.shuffle(bucket)
        if limit_per_locale:
            remaining = max(0, limit_per_locale - done_per_locale.get(loc, 0))
            bucket = bucket[:remaining]
        tasks.extend(bucket)

    random.shuffle(tasks)
    if limit:
        tasks = tasks[:limit]
    return tasks


def _infer_fraud_stage(scenario: dict, round_num: int) -> str:
    if "multi-rounds fraud" in scenario:
        for r in scenario["multi-rounds fraud"]:
            if r.get("round") == round_num:
                return r.get("fraud_stage", "authority")
    return "authority" if round_num <= 2 else "urgency"


# ── Main pipeline ─────────────────────────────────────────────────────────

def run_pipeline(
    model_name:       str = DEFAULT_GENERATOR_MODEL,
    embedder_key:     str = DEFAULT_EMBEDDER,
    locales:          list[str] | None = None,
    limit:            int | None = None,
    limit_per_locale: int | None = None,
    resume:           bool = True,
) -> None:
    locales = locales or LOCALES

    if model_name not in GENERATOR_MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(GENERATOR_MODELS)}"
        )
    if embedder_key not in EMBEDDER_OPTIONS:
        raise ValueError(
            f"Unknown embedder: '{embedder_key}'. "
            f"Available: {list(EMBEDDER_OPTIONS)}"
        )

    embedder_model, _ = EMBEDDER_OPTIONS[embedder_key]

    scenarios              = load_knowledge_base()
    done, content_hashes   = load_checkpoint() if resume else (set(), set())
    retriever              = RAGRetriever(embedder_model=embedder_model)

    tasks = build_task_list(scenarios, locales, done, model_name, limit, limit_per_locale)

    evaluator_tag = (
        f"vllm:{VLLM_EVALUATOR_MODEL}"
        if EVALUATOR_BACKEND == "vllm"
        else "deepseek-v4-pro"
    )

    done_per_locale  = count_done_per_locale(done, locales)
    tasks_per_locale = {loc: sum(1 for _, l, _ in tasks if l == loc) for loc in locales}
    print(f"\n[Pipeline] Generator:  {model_name}")
    print(f"[Pipeline] Evaluator:  {evaluator_tag}")
    print(f"[Pipeline] Embedder:   {embedder_model}")
    print(f"[Pipeline] Target:     {limit_per_locale or 'unlimited'} per locale")
    print(f"[Pipeline] Completed:  {done_per_locale}")
    print(f"[Pipeline] Pending:    {tasks_per_locale}")
    print(f"[Pipeline] Total:      {len(tasks)} tasks")
    print(f"[Pipeline] Output:     {DATASET_PATH}\n")

    stats = {"generated": 0, "accepted": 0, "rejected": 0, "dedup": 0, "errors": 0}

    for scenario, locale, round_num in tqdm(tasks, desc="Generating emails"):
        sid         = scenario["id"]
        fraud_stage = _infer_fraud_stage(scenario, round_num)
        topic       = scenario.get("subcategory", "account verification")
        jid         = job_id(sid, locale, round_num, model_name)

        context_docs = retriever.retrieve(
            query=f"{topic} {fraud_stage} {locale}",
        )

        prompt = build_prompt(
            round_num    = round_num,
            topic        = topic,
            fraud_stage  = fraud_stage,
            context_docs = context_docs,
            locale       = locale,
        )

        gen_result = generate_email(
            system_prompt = prompt["system"],
            user_prompt   = prompt["user"],
            locale        = locale,
            round_num     = round_num,
            scenario_id   = sid,
            fraud_stage   = fraud_stage,
            model_name    = model_name,
        )

        if not gen_result.success:
            stats["errors"] += 1
            print(f"\n[ERROR generation] {jid}: {gen_result.error}")
            done.add(jid)
            continue

        stats["generated"] += 1

        # Near-dedup: skip if an almost identical email was already generated
        chash = content_hash(gen_result.email_text)
        if chash in content_hashes:
            stats["dedup"] += 1
            done.add(jid)
            continue
        content_hashes.add(chash)

        corr_log = run_correction_loop(
            initial_email = gen_result.email_text,
            locale        = locale,
            fraud_stage   = fraud_stage,
            scenario_id   = sid,
            round_num     = round_num,
        )

        # If all iterations failed due to evaluator errors, skip entirely
        all_errors = all(it.get("error") for it in corr_log.iterations)
        if all_errors:
            stats["errors"] += 1
            print(f"\n[SKIP] {jid}: evaluator failed on all iterations")
            done.add(jid)
            continue

        if corr_log.accepted:
            stats["accepted"] += 1
        else:
            stats["rejected"] += 1

        record = {
            "id":          f"{model_name}_{sid}_{locale}_{round_num}_{int(time.time())}",
            "scenario_id": sid,
            "locale":      locale,
            "round_num":   round_num,
            "fraud_stage": fraud_stage,

            "email_text":  corr_log.final_email,
            "label":       1,

            "final_score":  corr_log.final_score,
            "accepted":     corr_log.accepted,
            "total_iters":  corr_log.total_iters,

            "generator_model":  model_name,
            "evaluator_model":  evaluator_tag,
            "embedder_model":   embedder_model,
            "generated_at":     datetime.now(timezone.utc).isoformat(),

            "prompt_tokens":     gen_result.prompt_tokens,
            "completion_tokens": gen_result.completion_tokens,
        }

        append_jsonl(DATASET_PATH, record)
        append_jsonl(AUDIT_LOG_PATH, {
            **record,
            "generation_result": result_to_dict(gen_result),
            "correction_log":    {"iterations": corr_log.iterations},
        })

        done.add(jid)
        if len(done) % 50 == 0:
            save_checkpoint(done, content_hashes)

    save_checkpoint(done, content_hashes)

    print(f"\n{'='*55}")
    print(f"RUN SUMMARY")
    print(f"  Generated: {stats['generated']}")
    print(f"  Accepted:  {stats['accepted']} "
          f"({stats['accepted']/max(stats['generated'],1)*100:.1f}%)")
    print(f"  Rejected:  {stats['rejected']}")
    print(f"  Dedup:     {stats['dedup']}")
    print(f"  Errors:    {stats['errors']}")
    print(f"  Dataset:   {DATASET_PATH}")
    print(f"{'='*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic phishing email generation pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GENERATOR_MODEL,
        choices=list(GENERATOR_MODELS),
        help=f"Generator model (default: {DEFAULT_GENERATOR_MODEL})",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=DEFAULT_EMBEDDER,
        choices=list(EMBEDDER_OPTIONS),
        help=f"RAG embedder (default: {DEFAULT_EMBEDDER}). "
             "Note: changing the embedder forces a FAISS index rebuild.",
    )
    parser.add_argument(
        "--locale",
        type=str,
        default=None,
        help="Run for a single locale only (e.g. ro-RO)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max total tasks (useful for quick tests)",
    )
    parser.add_argument(
        "--limit-per-locale",
        type=int,
        default=None,
        help="Max tasks per locale (e.g. 2000 → 10000 total across 5 locales)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and start from scratch",
    )
    args = parser.parse_args()

    locales = [args.locale] if args.locale else None

    run_pipeline(
        model_name       = args.model,
        embedder_key     = args.embedder,
        locales          = locales,
        limit            = args.limit,
        limit_per_locale = args.limit_per_locale,
        resume           = not args.no_resume,
    )
