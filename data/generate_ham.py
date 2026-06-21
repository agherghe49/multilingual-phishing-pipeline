"""
data/generate_ham.py

Generează emailuri legitime sintetice (ham, label=0) care seamănă ca format
și vocabular cu phishing-ul, dar sunt autentice.

Scopul este să creeze negative samples dificile pentru clasificator:
notificări de cont, confirmări de comandă, livrări, tranzacții bancare etc.
Acest tip de ham face scaling laws mai realist față de SpamAssassin (2003).

Rulare:
  python data/generate_ham.py --n 3000
  python data/generate_ham.py --n 600 --locale en-US
  python data/generate_ham.py --n 3000 --model kimi-k2.6

Output:
  outputs/dataset.jsonl  — append label=0
  outputs/ham_checkpoint.json — pentru resume
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OUTPUT_DIR, DATASET_PATH, LOCALES,
    REQUEST_DELAY, DEFAULT_GENERATOR_MODEL,
)
from generator import generate_email

# ── Topicuri pentru emailuri legitime ────────────────────────────────────────
# Fiecare topic are o descriere și o variantă — ciclăm prin ele pentru diversitate

TOPICS = [
    {
        "id":      "account_security",
        "en":      "account security notification — new login detected from a new device, no action required",
        "ro":      "notificare securitate cont — autentificare nouă detectată de pe un dispozitiv nou",
        "de":      "Sicherheitsbenachrichtigung — neue Anmeldung von neuem Gerät erkannt",
        "fr":      "notification de sécurité — nouvelle connexion détectée depuis un nouvel appareil",
        "it":      "notifica sicurezza account — nuovo accesso rilevato da un nuovo dispositivo",
    },
    {
        "id":      "order_confirmation",
        "en":      "order confirmation email — purchase completed successfully, order summary and tracking link",
        "ro":      "confirmare comandă — cumpărătură finalizată cu succes, sumar comandă și link urmărire",
        "de":      "Bestellbestätigung — Kauf erfolgreich abgeschlossen, Bestellübersicht und Tracking",
        "fr":      "confirmation de commande — achat finalisé avec succès, récapitulatif et lien de suivi",
        "it":      "conferma ordine — acquisto completato con successo, riepilogo ordine e link tracking",
    },
    {
        "id":      "delivery_update",
        "en":      "parcel delivery update — package out for delivery today, estimated arrival window",
        "ro":      "actualizare livrare colet — pachetul este în curs de livrare astăzi, interval estimat",
        "de":      "Paketlieferung Update — Paket ist heute unterwegs, geschätzte Ankunftszeit",
        "fr":      "mise à jour livraison — colis en cours de livraison aujourd'hui, créneau estimé",
        "it":      "aggiornamento consegna — pacco in consegna oggi, finestra di arrivo stimata",
    },
    {
        "id":      "bank_transaction",
        "en":      "bank transaction confirmation — payment processed successfully, transaction receipt",
        "ro":      "confirmare tranzacție bancară — plată procesată cu succes, chitanță tranzacție",
        "de":      "Banküberweisung Bestätigung — Zahlung erfolgreich verarbeitet, Transaktionsbeleg",
        "fr":      "confirmation de virement — paiement traité avec succès, reçu de transaction",
        "it":      "conferma transazione bancaria — pagamento elaborato con successo, ricevuta",
    },
    {
        "id":      "password_changed",
        "en":      "password successfully changed — confirmation that password was updated at user's request",
        "ro":      "parolă schimbată cu succes — confirmare că parola a fost actualizată la cererea utilizatorului",
        "de":      "Passwort erfolgreich geändert — Bestätigung der Passwortaktualisierung auf Benutzerwunsch",
        "fr":      "mot de passe modifié avec succès — confirmation que le mot de passe a été mis à jour",
        "it":      "password modificata con successo — conferma che la password è stata aggiornata",
    },
    {
        "id":      "subscription_renewal",
        "en":      "subscription renewal confirmation — annual plan renewed, next billing date and invoice",
        "ro":      "confirmare reînnoire abonament — plan anual reînnoit, data următoarei facturări",
        "de":      "Abonnement-Verlängerung — Jahresplan verlängert, nächstes Abrechnungsdatum",
        "fr":      "renouvellement d'abonnement — plan annuel renouvelé, prochaine date de facturation",
        "it":      "rinnovo abbonamento — piano annuale rinnovato, prossima data di fatturazione",
    },
    {
        "id":      "newsletter",
        "en":      "monthly newsletter — product updates, tips, and community news from the team",
        "ro":      "newsletter lunar — actualizări produse, sfaturi și noutăți comunitate de la echipă",
        "de":      "Monatlicher Newsletter — Produktupdates, Tipps und Community-News vom Team",
        "fr":      "newsletter mensuelle — mises à jour produits, conseils et actualités de l'équipe",
        "it":      "newsletter mensile — aggiornamenti prodotto, consigli e notizie dalla community",
    },
    {
        "id":      "service_update",
        "en":      "service update notification — new features available, brief changelog and what's new",
        "ro":      "notificare actualizare serviciu — funcții noi disponibile, jurnal modificări",
        "de":      "Dienst-Update-Benachrichtigung — neue Funktionen verfügbar, Änderungsprotokoll",
        "fr":      "notification de mise à jour — nouvelles fonctionnalités disponibles, journal des modifications",
        "it":      "notifica aggiornamento servizio — nuove funzionalità disponibili, registro modifiche",
    },
    {
        "id":      "account_statement",
        "en":      "monthly account statement ready — summary of activity, balance, and downloadable PDF",
        "ro":      "extras de cont lunar disponibil — sumar activitate, sold și PDF descărcabil",
        "de":      "Monatlicher Kontoauszug bereit — Aktivitätsübersicht, Kontostand und PDF",
        "fr":      "relevé de compte mensuel disponible — résumé activité, solde et PDF téléchargeable",
        "it":      "estratto conto mensile disponibile — riepilogo attività, saldo e PDF scaricabile",
    },
    {
        "id":      "appointment_reminder",
        "en":      "appointment reminder — scheduled meeting or service appointment confirmation and details",
        "ro":      "reminder programare — confirmare întâlnire sau programare serviciu și detalii",
        "de":      "Terminerinnerung — Bestätigung eines geplanten Termins oder Servicetermins",
        "fr":      "rappel de rendez-vous — confirmation d'une réunion ou d'un rendez-vous de service",
        "it":      "promemoria appuntamento — conferma riunione o appuntamento di servizio e dettagli",
    },
    # ── Topicuri CONFOUNDER: vocabular similar cu phishing, intenție legitimă ──
    {
        "id":      "gdpr_compliance",
        "en":      "GDPR compliance confirmation — your data processing consent was recorded, no action required, full privacy report attached",
        "ro":      "confirmare conformitate GDPR — consimțământul dvs. pentru prelucrarea datelor a fost înregistrat, nicio acțiune necesară",
        "de":      "DSGVO-Konformitätsbestätigung — Ihre Datenschutzzustimmung wurde aufgezeichnet, keine Maßnahmen erforderlich",
        "fr":      "confirmation de conformité RGPD — votre consentement au traitement des données a été enregistré, aucune action requise",
        "it":      "conferma conformità GDPR — il consenso al trattamento dei dati è stato registrato, nessuna azione richiesta",
    },
    {
        "id":      "security_all_clear",
        "en":      "security scan completed — no suspicious activity detected on your account, all systems secure, annual security report",
        "ro":      "scanare de securitate finalizată — nicio activitate suspectă detectată în contul dvs., toate sistemele sunt securizate",
        "de":      "Sicherheitsscan abgeschlossen — keine verdächtige Aktivität auf Ihrem Konto erkannt, alle Systeme sicher",
        "fr":      "analyse de sécurité terminée — aucune activité suspecte détectée sur votre compte, tous les systèmes sécurisés",
        "it":      "scansione di sicurezza completata — nessuna attività sospetta rilevata sul tuo account, tutti i sistemi sicuri",
    },
    {
        "id":      "identity_verified",
        "en":      "identity verification successful — your account has been fully verified, access restored, no further action needed",
        "ro":      "verificare identitate reușită — contul dvs. a fost verificat complet, accesul a fost restaurat",
        "de":      "Identitätsprüfung erfolgreich — Ihr Konto wurde vollständig verifiziert, Zugang wiederhergestellt",
        "fr":      "vérification d'identité réussie — votre compte a été entièrement vérifié, accès rétabli",
        "it":      "verifica identità riuscita — il tuo account è stato completamente verificato, accesso ripristinato",
    },
    {
        "id":      "account_review_ok",
        "en":      "account review completed — routine compliance check passed, no issues found, your account remains in good standing",
        "ro":      "revizuire cont finalizată — verificare de conformitate de rutină trecută, nicio problemă găsită, contul dvs. este în regulă",
        "de":      "Kontoüberprüfung abgeschlossen — routinemäßige Compliance-Prüfung bestanden, keine Probleme festgestellt",
        "fr":      "révision du compte terminée — vérification de conformité de routine réussie, aucun problème trouvé",
        "it":      "revisione account completata — controllo di conformità di routine superato, nessun problema riscontrato",
    },
    {
        "id":      "data_protection_report",
        "en":      "annual data protection report — summary of how your personal data was processed this year, GDPR rights reminder",
        "ro":      "raport anual protecția datelor — rezumat privind prelucrarea datelor dvs. personale în acest an, reamintire drepturi GDPR",
        "de":      "Jährlicher Datenschutzbericht — Zusammenfassung der Verarbeitung Ihrer persönlichen Daten, DSGVO-Rechte",
        "fr":      "rapport annuel de protection des données — résumé du traitement de vos données personnelles, rappel droits RGPD",
        "it":      "rapporto annuale sulla protezione dei dati — riepilogo del trattamento dei dati personali, promemoria diritti GDPR",
    },
    {
        "id":      "credentials_updated",
        "en":      "login credentials updated successfully — your username and password were changed at your request, session refreshed",
        "ro":      "datele de autentificare actualizate cu succes — numele de utilizator și parola au fost modificate la cererea dvs.",
        "de":      "Anmeldedaten erfolgreich aktualisiert — Benutzername und Passwort auf Ihren Wunsch geändert, Sitzung erneuert",
        "fr":      "identifiants de connexion mis à jour — votre nom d'utilisateur et mot de passe ont été modifiés à votre demande",
        "it":      "credenziali di accesso aggiornate con successo — nome utente e password modificati su richiesta, sessione aggiornata",
    },
    {
        "id":      "urgent_resolved",
        "en":      "urgent issue resolved — the security alert from earlier has been investigated and cleared, no action required from you",
        "ro":      "problemă urgentă rezolvată — alerta de securitate anterioară a fost investigată și rezolvată, nu este nevoie de nicio acțiune",
        "de":      "Dringendes Problem gelöst — die frühere Sicherheitswarnung wurde untersucht und behoben, keine Maßnahmen erforderlich",
        "fr":      "problème urgent résolu — l'alerte de sécurité précédente a été enquêtée et résolue, aucune action requise",
        "it":      "problema urgente risolto — l'avviso di sicurezza precedente è stato investigato e risolto, nessuna azione richiesta",
    },
]

LOCALE_CODE_TO_LANG = {
    "en-US": "en",
    "ro-RO": "ro",
    "de-DE": "de",
    "fr-FR": "fr",
    "it-IT": "it",
}

LOCALE_INSTRUCTIONS = {
    "en-US": "Write in professional American English. Use a real-sounding company name.",
    "ro-RO": "Scrie în română standard. Folosește un nume de companie credibil.",
    "de-DE": "Schreibe auf professionellem Deutsch (Sie-Form). Verwende einen realistischen Firmennamen.",
    "fr-FR": "Écris en français professionnel (vouvoiement). Utilise un nom d'entreprise réaliste.",
    "it-IT": "Scrivi in italiano professionale (Lei formale). Usa un nome aziendale credibile.",
}


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an email copywriter for a legitimate company.
Write ONLY the email body — no subject line, no metadata.
The email must be genuine, helpful, and non-threatening.
Maximum 160 words. Professional tone. No malicious intent whatsoever."""


def build_user_prompt(topic_desc: str, locale: str) -> str:
    lang_instruction = LOCALE_INSTRUCTIONS.get(locale, "Write in English.")
    return (
        f"Write a legitimate transactional email about: {topic_desc}\n\n"
        f"{lang_instruction}\n"
        f"Requirements:\n"
        f"- Real company name (invent a plausible one)\n"
        f"- Friendly but professional tone\n"
        f"- Include relevant details (date, reference number, etc.)\n"
        f"- NO threats, NO urgency pressure, NO requests for passwords or payment info\n"
        f"- End with a polite sign-off and contact info\n"
        f"- Maximum 160 words"
    )


# ── Checkpoint ────────────────────────────────────────────────────────────────

HAM_CHECKPOINT = OUTPUT_DIR / "ham_checkpoint.json"


def load_checkpoint() -> set:
    if HAM_CHECKPOINT.exists():
        data = json.loads(HAM_CHECKPOINT.read_text())
        return set(data.get("hashes", []))
    return set()


def save_checkpoint(hashes: set) -> None:
    HAM_CHECKPOINT.write_text(json.dumps({"hashes": list(hashes)}, indent=2))


def content_hash(text: str) -> str:
    return hashlib.sha1(text[:150].encode()).hexdigest()


# ── Generare și scriere ───────────────────────────────────────────────────────

def generate_ham_batch(
    n_total:    int,
    locales:    list[str],
    model_name: str,
    seed:       int = 42,
) -> None:
    random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seen_hashes = load_checkpoint()
    print(f"[resume] {len(seen_hashes)} emailuri deja generate (din checkpoint)")

    n_per_locale = n_total // len(locales)
    print(f"[plan] {n_per_locale} emailuri/locale × {len(locales)} locales = ~{n_total} total")
    print(f"[model] {model_name}\n")

    total_written = 0

    for locale in locales:
        lang_key  = LOCALE_CODE_TO_LANG[locale]
        generated = 0
        topics_cycle = TOPICS.copy()
        random.shuffle(topics_cycle)
        topic_idx = 0

        print(f"[{locale}] Generez {n_per_locale} emailuri ham ...")

        while generated < n_per_locale:
            topic = topics_cycle[topic_idx % len(topics_cycle)]
            topic_idx += 1

            topic_desc  = topic.get(lang_key, topic["en"])
            user_prompt = build_user_prompt(topic_desc, locale)

            result = generate_email(
                system_prompt = SYSTEM_PROMPT,
                user_prompt   = user_prompt,
                locale        = locale,
                round_num     = 0,
                scenario_id   = 0,
                fraud_stage   = "legitimate",
                model_name    = model_name,
            )

            if not result.success or not result.email_text.strip():
                print(f"  [warn] Generare eșuată: {result.error}")
                time.sleep(1)
                continue

            h = content_hash(result.email_text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            record = {
                "id":               f"ham_synth_{locale}_{topic['id']}_{h[:10]}",
                "email_text":       result.email_text,
                "label":            0,
                "locale":           locale,
                "topic":            topic["id"],
                "source":           "synthetic_ham",
                "generator_model":  model_name,
                "prompt_tokens":    result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "generated_at":     datetime.now(timezone.utc).isoformat(),
                "final_score":      None,
                "accepted":         True,
            }

            with open(DATASET_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            generated     += 1
            total_written += 1

            if generated % 50 == 0:
                save_checkpoint(seen_hashes)
                print(f"  [{locale}] {generated}/{n_per_locale} ...")

        save_checkpoint(seen_hashes)
        print(f"  [{locale}] ✓ {generated} emailuri scrise\n")

    print(f"[done] Total ham sintetic adăugat: {total_written}")
    print(f"[done] Dataset: {DATASET_PATH}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generează emailuri legitime sintetice (ham) pentru dataset"
    )
    parser.add_argument("--n",      type=int,   default=3000,
                        help="Număr total de emailuri ham de generat (implicit 3000)")
    parser.add_argument("--locale", type=str,   default=None,
                        help="Locale specific (ex. en-US); implicit toate 5")
    parser.add_argument("--model",  type=str,   default=DEFAULT_GENERATOR_MODEL,
                        help=f"Model generator (implicit {DEFAULT_GENERATOR_MODEL})")
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    target_locales = [args.locale] if args.locale else LOCALES

    generate_ham_batch(
        n_total    = args.n,
        locales    = target_locales,
        model_name = args.model,
        seed       = args.seed,
    )
