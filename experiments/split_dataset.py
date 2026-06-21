"""
experiments/split_dataset.py

Împarte dataset.jsonl într-un set de antrenament și un test set curat.

Strategie:
  - Test set: fix, 200 phishing + 200 ham (stratificat pe locale)
  - Train set: restul

Motivație: test set-ul trebuie să rămână neatins pe toată durata
experimentelor — nu se antrenează niciodată pe el, se folosește
DOAR pentru evaluarea finală din disertație.

Rulare:
  python experiments/split_dataset.py
  python experiments/split_dataset.py --test-phishing 300 --test-ham 300

Output:
  outputs/train.jsonl
  outputs/test.jsonl
  outputs/split_stats.json   (statistici despre split)
"""

import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, DATASET_PATH

TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
TEST_PATH  = OUTPUT_DIR / "test.jsonl"
STATS_PATH = OUTPUT_DIR / "split_stats.json"


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset negăsit: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_sample(
    records:    list[dict],
    n:          int,
    key:        str = "locale",
) -> tuple[list[dict], list[dict]]:
    """
    Extrage n înregistrări stratificat după `key`.
    Returnează (sampled, remainder).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get(key, "unknown")].append(r)

    per_group = max(1, n // len(groups))
    sampled   = []

    for group_records in groups.values():
        random.shuffle(group_records)
        sampled.extend(group_records[:per_group])

    # completează dacă nu am ajuns la n
    sampled_ids = {r["id"] for r in sampled}
    remainder_pool = [r for r in records if r["id"] not in sampled_ids]
    random.shuffle(remainder_pool)
    deficit = n - len(sampled)
    if deficit > 0:
        sampled.extend(remainder_pool[:deficit])
        remainder_pool = remainder_pool[deficit:]

    sampled_ids  = {r["id"] for r in sampled}
    remainder    = [r for r in records if r["id"] not in sampled_ids]

    return sampled, remainder


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split(
    test_phishing: int = 200,
    test_ham:      int = 200,
    seed:          int = 42,
) -> None:
    random.seed(seed)

    records = load_dataset(DATASET_PATH)
    print(f"[split] {len(records)} înregistrări totale")

    phishing = [r for r in records if r.get("label") == 1]
    ham      = [r for r in records if r.get("label") == 0]
    print(f"[split] {len(phishing)} phishing, {len(ham)} ham")

    if len(phishing) < test_phishing:
        print(f"[warn] Doar {len(phishing)} phishing disponibil; test set redus.")
        test_phishing = len(phishing) // 5

    if len(ham) < test_ham:
        print(f"[warn] Doar {len(ham)} ham disponibil; test set redus.")
        test_ham = len(ham) // 5

    test_ph,  train_ph  = stratified_sample(phishing, test_phishing, key="locale")
    test_ham_, train_ham = stratified_sample(ham,      test_ham,      key="locale")

    test_set  = test_ph  + test_ham_
    train_set = train_ph + train_ham
    random.shuffle(test_set)
    random.shuffle(train_set)

    write_jsonl(TEST_PATH,  test_set)
    write_jsonl(TRAIN_PATH, train_set)

    # Statistici
    def label_counts(lst: list[dict]) -> dict:
        from collections import Counter
        return dict(Counter(r.get("label") for r in lst))

    def locale_counts(lst: list[dict]) -> dict:
        from collections import Counter
        return dict(Counter(r.get("locale", "?") for r in lst))

    stats = {
        "total":  len(records),
        "train":  {
            "n":       len(train_set),
            "labels":  label_counts(train_set),
            "locales": locale_counts(train_set),
        },
        "test": {
            "n":       len(test_set),
            "labels":  label_counts(test_set),
            "locales": locale_counts(test_set),
        },
        "seed": seed,
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[split] Train: {len(train_set)} înregistrări → {TRAIN_PATH}")
    print(f"[split] Test:  {len(test_set)} înregistrări → {TEST_PATH}")
    print(f"[split] Stats: {STATS_PATH}")
    print(f"\nDistribuție test:")
    for locale, cnt in sorted(locale_counts(test_set).items()):
        print(f"  {locale:10s} {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split stratificat dataset.jsonl → train.jsonl + test.jsonl"
    )
    parser.add_argument("--test-phishing", type=int, default=200)
    parser.add_argument("--test-ham",      type=int, default=200)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    split(
        test_phishing = args.test_phishing,
        test_ham      = args.test_ham,
        seed          = args.seed,
    )
