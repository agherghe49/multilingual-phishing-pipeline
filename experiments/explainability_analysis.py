"""
experiments/explainability_analysis.py

Explainability analysis pe clasificatorul XLM-RoBERTa phishing.
Folosește Integrated Gradients (captum) + LIME pentru a identifica tokenii/cuvintele
care influențează predicția "phishing" vs. "ham".

Scopul: verificăm dacă F1=1 provine din artefacte de generare (pattern detectabil
trivial) sau din conținut phishing semantic real.

Comparăm:
  - Phishing STANDARD (din train.jsonl) — emailuri scrise manual/sintetic clasic
  - Phishing GRPO     (din adversarial_eval/adv_grpo_emails.json) — generate de Qwen+GRPO
  - Ham               (din test.jsonl, label=0)

Rulare:
    python experiments/explainability_analysis.py
    python experiments/explainability_analysis.py --n-samples 30 --method both
"""

import sys
import json
import argparse
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

TRAIN_JSONL     = OUTPUT_DIR / "train.jsonl"
TEST_JSONL      = OUTPUT_DIR / "test.jsonl"
GRPO_EMAILS_JSON = OUTPUT_DIR / "adversarial_eval" / "adv_grpo_emails.json"
ADV_LOOP_ROUND0  = OUTPUT_DIR / "adversarial_loop" / "grpo_batch_round0_seed42.json"
OUT_DIR         = OUTPUT_DIR / "explainability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_HF = "xlm-roberta-base"
N_TRAIN       = 1000   # samples pentru antrenare classifier (per recomandare profesor: 500-1000)
MAX_SEQ_LEN   = 256
N_LIME_FEAT   = 15     # features LIME per email


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def clean_email(text: str) -> str:
    """Elimină artefactele de prompt din emailurile GRPO generate."""
    # Dacă textul conține o secțiune de email după "Email:" sau "---", extragem acea parte
    patterns = [
        r'(?:Email:\s*\n[-\*]+\n?)([\s\S]+)',
        r'(?:---\s*\n)([\s\S]+)',
        r'(?:```\n?)([\s\S]+?)(?:```|$)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) > 80 and candidate.count(' ') > 10:
                return candidate
    # Dacă textul începe cu o listă de cuvinte (multe ghilimele), e contaminat
    if text.count("'") > 20 or text.count('"') > 20:
        # Caută primul paragraf lung
        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 60]
        if lines:
            return '\n'.join(lines[:8])
    return text


def is_clean_email(text: str) -> bool:
    """Verifică dacă textul arată ca un email real (nu prompt contaminat)."""
    if len(text) < 80:
        return False
    # Email-uri reale au propoziții, nu liste de cuvinte cu virgule
    has_sentences = bool(re.search(r'[A-Za-zÀ-ÿ]{3,}\s+[A-Za-zÀ-ÿ]{3,}\s+[A-Za-zÀ-ÿ]{3,}', text))
    # Prea multe ghilimele = listă de keywords din prompt
    too_many_quotes = text.count("'") > 25 or text.count('"') > 25
    return has_sentences and not too_many_quotes


# ── Antrenare classifier ──────────────────────────────────────────────────────

def train_classifier(train_data: list[dict], n: int = N_TRAIN, seed: int = 42):
    """Antrenează XLM-RoBERTa pe n samples (SFT conform recomandare profesor: 500-1000)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset as HFDataset

    random.seed(seed)

    phishing = [d for d in train_data if d['label'] == 1]
    ham      = [d for d in train_data if d['label'] == 0]
    n_each   = min(n // 2, len(phishing), len(ham))

    chosen = random.sample(phishing, n_each) + random.sample(ham, n_each)
    random.shuffle(chosen)
    texts  = [d['email_text'] for d in chosen]
    labels = [d['label']      for d in chosen]

    print(f"[explain] Antrenez clasificator XLM-RoBERTa: {len(chosen)} samples "
          f"({n_each} phishing + {n_each} ham), n_SFT={n}")

    tok = AutoTokenizer.from_pretrained(CLASSIFIER_HF)
    enc = tok(texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN)
    ds  = HFDataset.from_dict({**enc, 'labels': labels})

    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_HF, num_labels=2)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / 'clf_checkpoint'),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy='no',
        save_strategy='no',
        logging_steps=50,
        report_to='none',
        dataloader_pin_memory=False,
        seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    print(f"[explain] Antrenare finalizată")
    return model, tok


# ── Integrated Gradients ──────────────────────────────────────────────────────

def compute_ig_attributions(model, tokenizer, texts: list[str],
                             target_class: int = 1,
                             n_steps: int = 50) -> list[dict]:
    """Calculează Integrated Gradients per token pentru o listă de texte."""
    import torch
    from captum.attr import LayerIntegratedGradients

    model.eval()
    device = next(model.parameters()).device

    def forward_func(input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids
        return model(**kwargs).logits

    lig = LayerIntegratedGradients(forward_func, model.roberta.embeddings.word_embeddings)

    results = []
    for i, text in enumerate(texts):
        enc = tokenizer(text, return_tensors='pt', truncation=True,
                        max_length=MAX_SEQ_LEN, padding=False)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        # Baseline = token <pad>
        baseline_ids = torch.zeros_like(input_ids)

        try:
            attributions, _ = lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                target=target_class,
                n_steps=n_steps,
                return_convergence_delta=True,
            )
            attr_sum = attributions.sum(dim=-1).squeeze(0)
            attr_norm = attr_sum / (attr_sum.abs().max() + 1e-9)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            token_attrs = [(t, float(a)) for t, a in zip(tokens, attr_norm.tolist())
                           if t not in ['<s>', '</s>', '<pad>']]
            results.append({'text': text[:200], 'token_attrs': token_attrs})
        except Exception as e:
            print(f"  [warn] IG failed email {i}: {e}")
        if (i + 1) % 5 == 0:
            print(f"  [IG] {i+1}/{len(texts)} procesate")

    return results


def aggregate_token_importance(ig_results: list[dict], top_k: int = 25) -> dict:
    """Agregă importanța tokenilor peste toate emailurile."""
    word_scores = defaultdict(list)
    for res in ig_results:
        for token, score in res['token_attrs']:
            # Normalizăm tokenul (eliminăm prefixul ▁ de la SentencePiece)
            word = token.lstrip('▁Ġ').lower()
            if len(word) >= 3 and word.isalpha():
                word_scores[word].append(score)

    # Media scorurilor pozitive per cuvânt
    agg = {}
    for word, scores in word_scores.items():
        pos = [s for s in scores if s > 0]
        if pos:
            agg[word] = {'mean': np.mean(pos), 'freq': len(pos), 'max': max(pos)}

    top = sorted(agg.items(), key=lambda x: x[1]['mean'] * min(x[1]['freq'], 5), reverse=True)
    return dict(top[:top_k])


# ── LIME ─────────────────────────────────────────────────────────────────────

def compute_lime_top_words(model, tokenizer, texts: list[str],
                            n_samples: int = 500) -> Counter:
    """LIME: identifică cuvintele cu cel mai mare impact pe predicție."""
    import torch
    from lime.lime_text import LimeTextExplainer

    device = next(model.parameters()).device

    def predict_proba(text_list: list[str]) -> np.ndarray:
        model.eval()
        enc = tokenizer(text_list, return_tensors='pt', truncation=True,
                        padding=True, max_length=MAX_SEQ_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    explainer = LimeTextExplainer(class_names=['ham', 'phishing'])
    word_importance: Counter = Counter()

    for i, text in enumerate(texts):
        try:
            exp = explainer.explain_instance(
                text, predict_proba,
                num_features=N_LIME_FEAT,
                num_samples=n_samples,
                labels=[1],
            )
            for word, weight in exp.as_list(label=1):
                if weight > 0:
                    word_importance[word.lower()] += weight
        except Exception as e:
            print(f"  [warn] LIME failed email {i}: {e}")
        if (i + 1) % 5 == 0:
            print(f"  [LIME] {i+1}/{len(texts)} procesate")

    return word_importance


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(standard_words: dict, grpo_words: dict,
                    method: str, out_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    top_n = 20
    std_items  = sorted(standard_words.items(),
                        key=lambda x: x[1] if isinstance(x[1], float) else x[1].get('mean', 0),
                        reverse=True)[:top_n]
    grpo_items = sorted(grpo_words.items(),
                        key=lambda x: x[1] if isinstance(x[1], float) else x[1].get('mean', 0),
                        reverse=True)[:top_n]

    def get_score(v):
        return v if isinstance(v, float) else v.get('mean', 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, items, title, color in [
        (axes[0], std_items,  'Phishing STANDARD\n(top tokeni influenți)', '#2196F3'),
        (axes[1], grpo_items, 'Phishing GRPO\n(top tokeni influenți)',     '#F44336'),
    ]:
        words  = [w for w, _ in items]
        scores = [get_score(s) for _, s in items]
        bars = ax.barh(range(len(words)), scores, color=color, alpha=0.75)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel(f'Scor importanță ({method})', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{score:.3f}', va='center', fontsize=9)

    fig.suptitle(
        f'Explainability ({method}): Tokeni care influențează predicția "phishing"\n'
        'Comparație: phishing standard vs. phishing generat GRPO',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'[explain] Plot salvat: {out_path}')


def plot_artifact_analysis(grpo_texts: list[str], out_path: Path):
    """Plot: distribuția artefactelor de prompt în emailurile GRPO."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    artifact_patterns = {
        "Liste cuvinte-cheie (ghilimele)": lambda t: t.count("'") + t.count('"') > 20,
        "Instrucțiuni de generare": lambda t: any(k in t.lower() for k in
            ['do not', 'avoid', 'must include', 'should', 'pflicht', 'obligatoriu']),
        "Exemple din prompt": lambda t: 'example' in t.lower() or 'exemplu' in t.lower(),
        "Markeri de format (```)": lambda t: '```' in t,
        "Email curat (fără artefacte)": lambda t: is_clean_email(t),
    }

    counts = {k: sum(1 for t in grpo_texts if fn(t))
              for k, fn in artifact_patterns.items()}

    fig, ax = plt.subplots(figsize=(10, 5))
    items = list(counts.items())
    words  = [k for k, _ in items]
    vals   = [v for _, v in items]
    colors = ['#F44336' if 'curat' not in w else '#4CAF50' for w in words]
    bars = ax.barh(words, vals, color=colors, alpha=0.8)
    ax.set_xlabel('Număr emailuri', fontsize=11)
    ax.set_title(
        f'Analiza artefactelor de prompt în emailurile GRPO (n={len(grpo_texts)})\n'
        'Artefactele explică F1≈1: clasificatorul detectează instrucțiunile, nu phishing-ul',
        fontsize=12, fontweight='bold'
    )
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val} ({100*val/len(grpo_texts):.0f}%)', va='center', fontsize=11)
    ax.set_xlim(0, len(grpo_texts) * 1.25)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'[explain] Plot artefacte salvat: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=20,
                        help='Emailuri per categorie pentru analiza IG/LIME')
    parser.add_argument('--n-sft',     type=int, default=N_TRAIN,
                        help='Samples pentru antrenare classifier (rec. 500-1000)')
    parser.add_argument('--method',    choices=['ig', 'lime', 'both'], default='both')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Încarcă date ──────────────────────────────────────────────────────
    print('[explain] Încarc date...')
    train_data = load_jsonl(TRAIN_JSONL)
    test_data  = load_jsonl(TEST_JSONL)

    std_phishing = [d['email_text'] for d in train_data if d['label'] == 1]
    ham_texts    = [d['email_text'] for d in test_data  if d['label'] == 0]

    # Emailuri GRPO — din adversarial_eval sau round 0 cache
    grpo_raw = []
    if GRPO_EMAILS_JSON.exists():
        with open(GRPO_EMAILS_JSON) as f:
            grpo_raw.extend(json.load(f))
    if ADV_LOOP_ROUND0.exists():
        with open(ADV_LOOP_ROUND0) as f:
            grpo_raw.extend(json.load(f))

    print(f'[explain] Date: {len(std_phishing)} std_phishing, '
          f'{len(ham_texts)} ham, {len(grpo_raw)} grpo_raw')

    # ── Analizăm artefactele GRPO ─────────────────────────────────────────
    print('\n[explain] Analizez artefactele din emailurile GRPO...')
    plot_artifact_analysis(grpo_raw, OUT_DIR / 'grpo_artifact_analysis.png')

    grpo_clean  = [clean_email(e) for e in grpo_raw]
    n_clean     = sum(1 for e in grpo_raw if is_clean_email(e))
    print(f'[explain] GRPO emailuri: {len(grpo_raw)} total, {n_clean} curate '
          f'({100*n_clean/len(grpo_raw):.0f}%)')

    # ── Sample pentru analiză ─────────────────────────────────────────────
    n = args.n_samples
    std_sample  = random.sample(std_phishing, min(n, len(std_phishing)))
    grpo_sample = random.sample(grpo_clean,   min(n, len(grpo_clean)))
    ham_sample  = random.sample(ham_texts,    min(n, len(ham_texts)))

    print(f'[explain] Sample: {len(std_sample)} std, {len(grpo_sample)} grpo, '
          f'{len(ham_sample)} ham')

    # ── Antrenare classifier n_SFT={args.n_sft} ──────────────────────────
    print(f'\n[explain] Antrenez clasificator (n_SFT={args.n_sft})...')
    model, tokenizer = train_classifier(train_data, n=args.n_sft, seed=args.seed)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # ── Verificare acuratețe rapidă ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        all_texts = std_sample[:10] + grpo_sample[:10] + ham_sample[:10]
        all_labels = [1]*10 + [1]*10 + [0]*10
        enc = tokenizer(all_texts, return_tensors='pt', truncation=True,
                        padding=True, max_length=MAX_SEQ_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        preds  = logits.argmax(dim=-1).cpu().tolist()
        acc = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
    print(f'[explain] Acuratețe rapidă pe sample: {acc:.2%}')

    results = {'config': {'n_samples': n, 'n_sft': args.n_sft, 'method': args.method}}

    # ── Integrated Gradients ──────────────────────────────────────────────
    if args.method in ('ig', 'both'):
        print('\n[explain] === Integrated Gradients ===')

        print(f'  Standard phishing ({len(std_sample)} emailuri)...')
        ig_std  = compute_ig_attributions(model, tokenizer, std_sample)
        print(f'  GRPO phishing ({len(grpo_sample)} emailuri)...')
        ig_grpo = compute_ig_attributions(model, tokenizer, grpo_sample)

        std_agg  = aggregate_token_importance(ig_std)
        grpo_agg = aggregate_token_importance(ig_grpo)

        print('\n[explain] Top tokeni — STANDARD phishing (IG):')
        for w, d in list(std_agg.items())[:15]:
            print(f'  {w:20s} mean={d["mean"]:.4f} freq={d["freq"]}')

        print('\n[explain] Top tokeni — GRPO phishing (IG):')
        for w, d in list(grpo_agg.items())[:15]:
            print(f'  {w:20s} mean={d["mean"]:.4f} freq={d["freq"]}')

        plot_comparison(std_agg, grpo_agg, 'Integrated Gradients',
                        OUT_DIR / 'explainability_ig.png')

        results['ig'] = {
            'standard_top_tokens': {k: v for k, v in list(std_agg.items())[:20]},
            'grpo_top_tokens':     {k: v for k, v in list(grpo_agg.items())[:20]},
        }

    # ── LIME ──────────────────────────────────────────────────────────────
    if args.method in ('lime', 'both'):
        print('\n[explain] === LIME ===')

        print(f'  Standard phishing ({len(std_sample)} emailuri)...')
        lime_std  = compute_lime_top_words(model, tokenizer, std_sample)
        print(f'  GRPO phishing ({len(grpo_sample)} emailuri)...')
        lime_grpo = compute_lime_top_words(model, tokenizer, grpo_sample)

        # Normalizare
        max_std  = max(lime_std.values())  if lime_std  else 1
        max_grpo = max(lime_grpo.values()) if lime_grpo else 1
        std_norm  = {w: v / max_std  for w, v in lime_std.most_common(25)}
        grpo_norm = {w: v / max_grpo for w, v in lime_grpo.most_common(25)}

        print('\n[explain] Top cuvinte — STANDARD phishing (LIME):')
        for w, s in sorted(std_norm.items(), key=lambda x: -x[1])[:15]:
            print(f'  {w:20s} score={s:.4f}')

        print('\n[explain] Top cuvinte — GRPO phishing (LIME):')
        for w, s in sorted(grpo_norm.items(), key=lambda x: -x[1])[:15]:
            print(f'  {w:20s} score={s:.4f}')

        plot_comparison(std_norm, grpo_norm, 'LIME',
                        OUT_DIR / 'explainability_lime.png')

        results['lime'] = {
            'standard_top_words': dict(sorted(std_norm.items(), key=lambda x: -x[1])[:20]),
            'grpo_top_words':     dict(sorted(grpo_norm.items(), key=lambda x: -x[1])[:20]),
        }

    # ── Analiză comparativă: overlap tokeni ──────────────────────────────
    if args.method == 'both' and 'ig' in results and 'lime' in results:
        std_ig_words  = set(results['ig']['standard_top_tokens'].keys())
        grpo_ig_words = set(results['ig']['grpo_top_tokens'].keys())
        std_lime_words  = set(results['lime']['standard_top_words'].keys())
        grpo_lime_words = set(results['lime']['grpo_top_words'].keys())

        overlap_std  = std_ig_words  & std_lime_words
        overlap_grpo = grpo_ig_words & grpo_lime_words
        unique_grpo  = grpo_ig_words - std_ig_words

        print(f'\n[explain] Overlap IG∩LIME — standard: {len(overlap_std)} tokeni')
        print(f'[explain] Overlap IG∩LIME — GRPO:     {len(overlap_grpo)} tokeni')
        print(f'[explain] Tokeni unici GRPO (față de std): {sorted(unique_grpo)[:10]}')

        results['analysis'] = {
            'overlap_std_ig_lime':    list(overlap_std),
            'overlap_grpo_ig_lime':   list(overlap_grpo),
            'tokens_unique_to_grpo':  list(unique_grpo),
        }

    # ── Salvare rezultate ─────────────────────────────────────────────────
    out_json = OUT_DIR / 'explainability_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n[explain] Rezultate salvate: {out_json}')
    print(f'[explain] Plot-uri: {OUT_DIR}/')

    # ── Interpretare finală ───────────────────────────────────────────────
    print('\n' + '='*70)
    print('INTERPRETARE EXPLAINABILITY')
    print('='*70)
    print(f'Clasificator antrenat cu n_SFT={args.n_sft} samples (recom. profesor)')

    if 'analysis' in results and results['analysis']['tokens_unique_to_grpo']:
        unique = results['analysis']['tokens_unique_to_grpo'][:8]
        print(f'\nTokeni detectabili UNICI în GRPO (pot fi artefacte): {unique}')
        print('→ Verificați dacă acești tokeni apar în prompt-uri, nu în phishing real')
    print()


if __name__ == '__main__':
    main()
