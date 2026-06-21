"""
data/assemble.py

Asamblează negative samples (ham, label=0) din corpusuri publice:
  - SpamAssassin ham corpus  (descărcat automat)
  - Enron email corpus       (necesită fișier CSV descărcat manual)

Rulare:
  python data/assemble.py --source spamassassin --n 2000
  python data/assemble.py --source enron --enron-csv path/to/emails.csv --n 2000
  python data/assemble.py --source both --n 2000   # combină ambele

Output:
  outputs/dataset.jsonl  — adaugă n înregistrări cu label=0 (append)
  outputs/negatives_added.txt — log cu fișierele procesate

SpamAssassin corpus:
  Descărcăm ham_2 (2500 mesaje) și ham (2551 mesaje) din arhivele publice.
  URL: https://spamassassin.apache.org/old/publiccorpus/

Enron corpus:
  Descărcați CSV-ul de pe Kaggle: "Enron Email Dataset" (emails.csv ~500MB)
  sau extrageți din arhiva tar: https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
"""

import os
import re
import sys
import json
import email
import hashlib
import tarfile
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, DATASET_PATH, DEDUP_HASH_CHARS

# URL-uri publice SpamAssassin ham corpus
SPAMASSASSIN_URLS = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
]

CACHE_DIR = Path(__file__).parent / ".cache"


# ── Descărcare ─────────────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> Path:
    fname = dest / url.split("/")[-1]
    if fname.exists():
        print(f"  [cache] {fname.name}")
        return fname
    print(f"  Descărcare {url.split('/')[-1]}...")
    urllib.request.urlretrieve(url, fname)
    return fname


# ── Parsing email ──────────────────────────────────────────────────────────

def _extract_body(raw: bytes) -> str:
    """Extrage textul plain din email raw (bytes)."""
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True) or b""
                    body = body.decode(errors="replace")
                    break
        else:
            body = msg.get_payload(decode=True) or b""
            body = body.decode(errors="replace")
        return _clean_text(body)
    except Exception:
        return ""


def _clean_text(text: str) -> str:
    """Elimină artefacte comune din emailuri brute."""
    # Remove forwarded headers, excess whitespace
    text = re.sub(r"[-]{3,}.*?[-]{3,}", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def _content_hash(text: str) -> str:
    return hashlib.sha1(text[:DEDUP_HASH_CHARS].encode()).hexdigest()


# ── SpamAssassin ───────────────────────────────────────────────────────────

def load_spamassassin_ham(n: int) -> list[str]:
    """Descarcă și extrage emailuri ham din corpusul SpamAssassin."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emails = []
    seen   = set()

    for url in SPAMASSASSIN_URLS:
        if len(emails) >= n:
            break
        archive = _download(url, CACHE_DIR)
        try:
            with tarfile.open(archive, "r:bz2") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = Path(member.name).name
                    # SpamAssassin ham files are named like "0001.xxx"
                    if name.startswith("cmds"):
                        continue
                    f    = tar.extractfile(member)
                    if f is None:
                        continue
                    raw  = f.read()
                    body = _extract_body(raw)
                    if len(body) < 50:
                        continue
                    chash = _content_hash(body)
                    if chash in seen:
                        continue
                    seen.add(chash)
                    emails.append(body)
                    if len(emails) >= n:
                        break
        except Exception as e:
            print(f"  [warn] Eroare la {archive.name}: {e}")

    print(f"[SpamAssassin] {len(emails)} emailuri ham extrase")
    return emails[:n]


# ── Enron corpus ───────────────────────────────────────────────────────────

def load_enron_ham(csv_path: str, n: int) -> list[str]:
    """
    Încarcă emailuri din CSV-ul Enron (coloana 'message').

    CSV format așteptat (Kaggle Enron dataset):
      file,message
      ...,"Message-ID: ...\n\nBody text..."
    """
    try:
        import csv
    except ImportError:
        raise ImportError("Modulul csv este necesar (inclus în stdlib)")

    emails = []
    seen   = set()
    path   = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fișierul Enron CSV nu a fost găsit: {csv_path}\n"
            "Descarcă de la: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset"
        )

    print(f"[Enron] Citesc {path.name}...")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(emails) >= n:
                break
            raw_msg = row.get("message", "")
            if not raw_msg:
                continue
            body = _extract_body(raw_msg.encode(errors="replace"))
            if len(body) < 50:
                continue
            chash = _content_hash(body)
            if chash in seen:
                continue
            seen.add(chash)
            emails.append(body)

    print(f"[Enron] {len(emails)} emailuri ham extrase")
    return emails[:n]


# ── Scriere dataset ────────────────────────────────────────────────────────

def write_ham_to_dataset(ham_emails: list[str], source: str) -> int:
    """Adaugă emailurile ham în dataset.jsonl cu label=0."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    ts      = datetime.now(timezone.utc).isoformat()

    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        for i, text in enumerate(ham_emails):
            record = {
                "id":          f"ham_{source}_{i}_{int(datetime.now().timestamp())}",
                "email_text":  text,
                "label":       0,           # 0 = ham (legitimate)
                "source":      source,
                "locale":      "en-US",     # corpusul Enron/SpamAssassin e în engleză
                "generated_at": ts,
                "final_score":  None,
                "accepted":     True,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return written


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asamblează negative samples (ham) în dataset.jsonl"
    )
    parser.add_argument(
        "--source",
        choices=["spamassassin", "enron", "both"],
        default="spamassassin",
        help="Sursa emailurilor ham",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Numărul de emailuri ham de adăugat",
    )
    parser.add_argument(
        "--enron-csv",
        type=str,
        default=None,
        help="Calea spre emails.csv din datasetul Enron (necesar dacă --source enron/both)",
    )
    args = parser.parse_args()

    total = 0

    if args.source in ("spamassassin", "both"):
        n_sa   = args.n if args.source == "spamassassin" else args.n // 2
        ham_sa = load_spamassassin_ham(n_sa)
        total += write_ham_to_dataset(ham_sa, "spamassassin")
        print(f"[OK] {total} emailuri SpamAssassin scrise în {DATASET_PATH}")

    if args.source in ("enron", "both"):
        if not args.enron_csv:
            print("Eroare: --enron-csv este necesar pentru sursa 'enron' sau 'both'")
            sys.exit(1)
        n_en   = args.n if args.source == "enron" else args.n - (args.n // 2)
        ham_en = load_enron_ham(args.enron_csv, n_en)
        written = write_ham_to_dataset(ham_en, "enron")
        total += written
        print(f"[OK] {written} emailuri Enron scrise în {DATASET_PATH}")

    print(f"\nTotal negative samples adăugate: {total}")
    print(f"Dataset: {DATASET_PATH}")
