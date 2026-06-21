"""
generation/prompts.py

Wrapper peste prompt_templates.py din arhiva ta originală.
Adaugă context RAG și instrucțiunile de reproductibilitate.

Dacă ai prompt_templates.py în același director (sau în data/),
acest fișier îl importă direct. Altfel folosește implementarea built-in.
"""

from typing import Optional
import sys
from pathlib import Path

# Încearcă să importe prompt_templates.py original dacă există
_templates_loaded = False
try:
    for search_path in [
        Path(__file__).parent / "data",
        Path(__file__).parent,
    ]:
        candidate = search_path / "prompt_templates.py"
        if candidate.exists():
            sys.path.insert(0, str(search_path.parent))
            from prompt_templates import build_prompt as _original_build_prompt  # type: ignore
            _templates_loaded = True
            print(f"[prompts] Am încărcat prompt_templates.py din {candidate}")
            break
except ImportError:
    pass


# ── Fallback: implementare built-in ──────────────────────────────────────

_ROUND_SYSTEM = {
    1: ("You are a researcher generating realistic phishing emails for academic purposes. "
        "Round 1: Initial contact. Focus on establishing legitimacy and authority. "
        "Use professional tone. Do NOT include real URLs or phone numbers — use [URL], [PHONE]."),
    2: ("You are a researcher generating realistic phishing emails for academic purposes. "
        "Round 2: Establish authority. Emphasize compliance, official requirements. "
        "Authority score target: 4+. Do NOT include real contact details."),
    3: ("You are a researcher generating realistic phishing emails for academic purposes. "
        "Round 3: Peak urgency. Maximum pressure — deadlines, countdowns, final warnings. "
        "Urgency score target: 7+. Use ⚠️ 🚨 ⏳ for emphasis. No real contact details."),
    4: ("You are a researcher generating realistic phishing emails for academic purposes. "
        "Round 4: Sustained pressure. Balance urgency (5-6) with authority (4+). "
        "Final consequences. Do NOT include real contact details."),
}

_ROUND_INSTRUCTIONS = {
    1: "- Professional, official tone\n- Establish credibility\n- No excessive urgency\n- Verification or compliance request",
    2: "- Compliance and official requirements\n- Use: verified, official, regulation\n- Build credibility\n- Subtle urgency only",
    3: "- MAXIMUM urgency: deadlines, countdowns\n- Time pressure: 'within X hours', 'expires'\n- Combine with authority",
    4: "- Maintain urgency but reinforce authority\n- Final warnings and consequences\n- Combine urgency + authority tactics",
}

_LOCALE_MAX_WORDS: dict[str, int] = {
    "ro-RO": 180,
    "de-DE": 180,
    "fr-FR": 180,
    "it-IT": 180,
    "en-US": 180,
}


def get_available_locales() -> list[str]:
    return list(_LOCALE_MAX_WORDS.keys())


def _get_locale_note(locale: str) -> str:
    notes = {
        "ro-RO": (
            "LANGUAGE: Romanian — scrie ÎNTREGUL email în română, fără niciun cuvânt în engleză.\n"
            "- OBLIGATORIU: începe ÎNTOTDEAUNA cu o formulă de salut: "
            "'Stimate client,' / 'Stimate utilizator,' / 'Stimată doamnă,' / 'Stimate domn,'\n"
            "- Closing: 'Cu stimă,' / 'Cu respect,' urmată de numele departamentului și instituției fictive\n"
            "- Date format: '17 octombrie 2023'\n"
            "- OBLIGATORIU: folosește EXCLUSIV forma de politețe 'dumneavoastră' — NICIODATĂ 'tu/tău/te'. "
            "Folosiți diacritice corecte (ă, â, î, ș, ț) — niciodată 's' în loc de 'ș' sau 't' în loc de 'ț'.\n"
            "- INTERZIS: nu începe cu 'Vă informăm că contul dumneavoastră' — prea repetitiv. Folosește în schimb:\n"
            "  * 'Vă contactăm în legătură cu...'\n"
            "  * 'Ca urmare a verificării periodice...'\n"
            "  * 'Vă aducem la cunoștință că...'\n"
            "  * 'Echipa noastră a identificat...'\n"
            "  * 'Conform politicii interne de securitate...'\n"
            "  * 'În urma unui control automat al sistemelor noastre...'\n"
            "- Legal references: 'conform art. 6 din Regulamentul UE 2016/679', "
            "'în temeiul Legii nr. 190/2018 privind aplicarea GDPR', "
            "'potrivit art. 32 GDPR', 'conform art. 1349 din Codul Civil'\n"
            "- Vocabulary: 'vă rugăm să confirmați datele', 'contul dumneavoastră a fost temporar restricționat', "
            "'în cel mai scurt timp posibil', 'pentru a evita suspendarea definitivă', "
            "'acțiunea este obligatorie'\n"
            "- FICTIONAL institutions only — VARIAȚI numele, nu folosiți mereu TransilvaniaBank:\n"
            "  Bănci: 'TransilvaniaBank', 'BancaVerde', 'CreditRomân SA', 'NordBank România'\n"
            "  Telecom: 'RoNet', 'ConnectRo', 'TeleRom SA'\n"
            "  E-commerce: 'ShopRo', 'MegaShop', 'ComandaRapidă'\n"
            "  Autorități (fictive): 'Agenția de Conformitate Fiscală', 'Biroul de Verificare Digitală', "
            "'Centrul Național pentru Securitate Cibernetică', 'Direcția de Control Financiar'\n"
            "- Tone: formal-administrativ, ușor birocratic — max 180 cuvinte"
        ),
        "de-DE": (
            "LANGUAGE: German — schreibe die GESAMTE E-Mail auf Deutsch, kein Englisch.\n"
            "- PFLICHT: beginne IMMER mit einer Anrede: "
            "'Sehr geehrte Damen und Herren,' / 'Sehr geehrter Herr Müller,' / 'Sehr geehrte Frau Schmidt,'\n"
            "- Closing: 'Mit freundlichen Grüßen,' / 'Hochachtungsvoll,' gefolgt von Abteilung und Firmenname\n"
            "- Date format: '17. Oktober 2023'\n"
            "- PFLICHT: verwende ausschließlich die 'Sie'-Form. NIEMALS 'du'. "
            "Substantive werden großgeschrieben (Konto, Daten, Verifizierung, Sperrung).\n"
            "- VERBOTEN: beginne NICHT mit 'Gemäß den geltenden Datenschutzbestimmungen' — zu repetitiv. Alternativen:\n"
            "  * 'Im Rahmen unserer regelmäßigen Sicherheitsüberprüfung...'\n"
            "  * 'Wir möchten Sie darauf hinweisen, dass...'\n"
            "  * 'Unser System hat eine Unregelmäßigkeit festgestellt...'\n"
            "  * 'Im Zuge der aktuellen Compliance-Prüfung...'\n"
            "  * 'Wir wenden uns heute an Sie, da...'\n"
            "  * 'Aufgrund einer automatisierten Überprüfung Ihrer Kontodaten...'\n"
            "- Legal references (KONKRETE Paragraphennummern — NIEMALS §XX als Platzhalter): "
            "'gemäß §32 BDSG', 'gemäß Art. 6 Abs. 1 DSGVO', 'gemäß §15 TMG', '§45d TKG'\n"
            "- Vocabulary: 'Wir bitten Sie dringend', 'Ihre sofortige Aufmerksamkeit ist erforderlich', "
            "'vorübergehend eingeschränkt', 'dauerhafte Sperrung vermeiden', 'unverzüglich bestätigen'\n"
            "- FICTIONAL institutions only — VARIA die Namen, verwende nicht immer NordBank AG — "
            "KEINE echten Behörden (NICHT: BSI, BaFin, Bundesnetzagentur):\n"
            "  Banken: 'NordBank AG', 'RheinFinanz GmbH', 'BayernKredit AG', 'AlphaBank Deutschland'\n"
            "  Versicherungen: 'SecureVita GmbH', 'NordSchutz AG', 'AllianzPlus GmbH'\n"
            "  Behörden (fiktiv): 'Bundesdatenschutzamt', 'Amt für digitale Compliance', 'Datensicherheitsbehörde'\n"
            "  Online: 'ShopDirect GmbH', 'DeliveryNow GmbH', 'NordCareer AG'\n"
            "- Tone: sehr formal, offiziell, leicht bürokratisch — max 180 Wörter"
        ),
        "fr-FR": (
            "LANGUAGE: French — écris l'INTÉGRALITÉ de l'e-mail en français, sans anglais.\n"
            "- OBLIGATOIRE: commencer TOUJOURS par une formule de salutation: "
            "'Madame, Monsieur,' / 'Cher(e) client(e),' / 'Madame,' / 'Monsieur,'\n"
            "- Closing: 'Veuillez agréer nos salutations distinguées,' / 'Cordialement,' "
            "suivi du nom du service et de l'institution fictive\n"
            "- Date format: '17 octobre 2023'\n"
            "- OBLIGATOIRE: utiliser exclusivement le 'vous' de politesse. JAMAIS 'tu'. "
            "Accorder correctement les participes passés et les adjectifs.\n"
            "- INTERDIT: ne JAMAIS commencer par 'Suite à un contrôle de routine effectué par notre service' — trop répétitif. Choisir parmi:\n"
            "  * 'Nous vous informons qu'une anomalie a été détectée sur votre compte...'\n"
            "  * 'Dans le cadre de notre politique de sécurité...'\n"
            "  * 'Votre attention est requise concernant...'\n"
            "  * 'À la suite d'une vérification automatisée de nos systèmes...'\n"
            "  * 'Notre équipe de conformité a identifié une irrégularité...'\n"
            "  * 'Nous avons le regret de vous informer qu'une anomalie...'\n"
            "- Legal references: 'conformément au Règlement UE 2016/679 (RGPD)', "
            "'en application de la loi n°2018-493 relative à la protection des données', "
            "'selon l'art. 82 de la loi Informatique et Libertés', "
            "'dans le cadre de la directive 2013/11/UE'\n"
            "- Vocabulary: 'Nous vous invitons à', 'votre compte a été temporairement suspendu', "
            "'dans les meilleurs délais', 'pour éviter la suspension définitive', "
            "'cette procédure est obligatoire'\n"
            "- FICTIONAL institutions only — VARIEZ les noms, n'utilisez pas toujours RéseauAmis:\n"
            "  Banques: 'CréditSécurisé', 'BanqueDirecte', 'FinanceProtect', 'MonCompteNet', 'PatrimoineNet'\n"
            "  Livraison: 'MonColis', 'ExpressLivraison', 'ColisRapide', 'ChronoEnvoi'\n"
            "  Fisc (fictif): 'Service de Vérification Fiscale', 'Direction du Contrôle Numérique', 'Bureau de Conformité Fiscale'\n"
            "  Télécom: 'TélécomPlus', 'ConnectFrance', 'FranceMobile'\n"
            "  Social: 'RéseauAmis', 'MonEspace', 'VoxSocial', 'LienSocial'\n"
            "- Tone: formel, administratif, courtois mais ferme — max 180 mots"
        ),
        "it-IT": (
            "LANGUAGE: Italian — scrivi l'INTERA e-mail in italiano, senza inglese.\n"
            "- OBBLIGATORIO: inizia SEMPRE con una riga di saluto: 'Gentile Cliente,' / 'Gentile Utente,' / 'Egregio Signore,' / 'Gentile Signora,'\n"
            "- Closing: 'Distinti saluti,' / 'Cordiali saluti,' seguito dal nome del team e dell'ente fittizio\n"
            "- Date format: '17 ottobre 2023'\n"
            "- OBBLIGATORIO: usa SEMPRE la forma di cortesia 'Lei' con MAIUSCOLA per pronomi/possessivi: "
            "'Suo/Sua/Sue/Suoi/Le/La/Li' — MAI minuscolo ('suo/sua'). MAI 'tu'.\n"
            "- VIETATO iniziare con 'La informiamo che il Suo account' o 'Ai sensi del Regolamento UE' — troppo ripetitivo. Usa invece:\n"
            "  * 'A seguito di una verifica automatica dei Suoi dati...'\n"
            "  * 'Il nostro team di sicurezza ha rilevato un'anomalia...'\n"
            "  * 'Con la presente Le comunichiamo che una verifica è necessaria...'\n"
            "  * 'Nell'ambito delle nostre procedure di controllo interno...'\n"
            "  * 'Il nostro sistema ha rilevato una criticità relativa al Suo profilo...'\n"
            "  * 'In ottemperanza alle normative vigenti, Le comunichiamo che...'\n"
            "- Legal references: 'ai sensi del D.Lgs. 196/2003', 'art. 17 del Regolamento UE 2016/679', "
            "'in ottemperanza al Codice del Consumo (D.Lgs. 206/2005)', 'ai sensi dell'art. 32 GDPR'\n"
            "- Vocabulary: 'La invitiamo a procedere', 'è necessario procedere con la verifica entro', "
            "'il Suo account è stato temporaneamente limitato', 'entro e non oltre', "
            "'per evitare la sospensione definitiva'\n"
            "- FICTIONAL institutions only — VARIA i nomi, non usare sempre BancaSecura SpA:\n"
            "  Banche: 'CreditoNord Srl', 'ItalFinanza SpA', 'VerdeBank', 'BancoSicuro Italia', 'PatrimonioNet SpA'\n"
            "  Corrieri: 'SpeditoItalia', 'PaccoVeloce', 'ItaliaExpress', 'ConsegnaRapida'\n"
            "  E-commerce/Lavoro: 'AcquistoSicuro', 'ShopItalia', 'JobConnect SpA', 'LavoraOra'\n"
            "  Enti (fittizi): 'Agenzia per la Conformità Digitale', 'Ufficio di Controllo Finanziario', "
            "'Direzione Sicurezza Digitale', 'Sportello Digitale dell'Utente'\n"
            "- Tone: formale, cortese ma urgente — max 180 parole"
        ),
        "en-US": (
            "LANGUAGE: English (US) — professional corporate tone.\n"
            "- MANDATORY: always start with a salutation: "
            "'Dear Customer,' / 'Dear Valued Member,' / 'Dear [First Name],'\n"
            "- Closing: 'Sincerely,' / 'Best regards,' followed by team name and institution name\n"
            "- Date format: 'October 17, 2023'\n"
            "- MANDATORY: maintain formal register throughout — no contractions "
            "('do not' not 'don't', 'you have' not 'you've', 'we are' not 'we're').\n"
            "- FORBIDDEN opener: do NOT start with 'We have detected unusual activity on your account' "
            "or 'We are writing to inform you that your account' — too generic. Use instead:\n"
            "  * 'As part of our scheduled security review...'\n"
            "  * 'Our automated systems have flagged an inconsistency...'\n"
            "  * 'Pursuant to federal compliance requirements (Section 14-B)...'\n"
            "  * 'Your account requires immediate verification under...'\n"
            "  * 'In accordance with our updated Terms of Service (Rev. 2023-09)...'\n"
            "  * 'A routine audit of your account has revealed...'\n"
            "- Legal references: 'pursuant to Section 14-B of the Financial Security Act', "
            "'under the Digital Identity Verification Act (DIVA 2022)', "
            "'per 12 CFR Part 1005 (Regulation E)', 'under the Electronic Funds Transfer Act §205'\n"
            "- Vocabulary: 'immediate action is required', 'your account has been temporarily restricted', "
            "'failure to respond may result in permanent suspension', "
            "'please verify your identity within 48 hours', 'as required by federal regulation'\n"
            "- FICTIONAL institutions only — VARY names, do not always use PrimeVest Financial:\n"
            "  Banking: 'PrimeVest Financial', 'NorthShore Credit Union', 'CapitalEdge Bank', 'TrustPoint Financial'\n"
            "  Logistics: 'ShipFast Logistics', 'DeliverNow Inc.', 'PackageTrack Services', 'SwiftParcel'\n"
            "  HR/Tech: 'TalentVerify', 'CloudMail Services', 'SecureID Network', 'WorkBridge Solutions'\n"
            "- Tone: clear, professional, slightly urgent — max 180 words"
        ),
    }
    return notes.get(locale, notes["en-US"])


def _build_prompt_builtin(
    round_num: int,
    topic: str,
    fraud_stage: str,
    context_docs: list[dict],
    locale: str = "en-US",
    previous_stage: Optional[str] = None,
    max_context_docs: int = 2,
) -> dict[str, str]:
    """Construiește promptul fără a depinde de fișierul extern."""

    system = _ROUND_SYSTEM.get(round_num, _ROUND_SYSTEM[1])
    instructions = _ROUND_INSTRUCTIONS.get(round_num, _ROUND_INSTRUCTIONS[1])

    # Context RAG
    context_parts = []
    for i, doc in enumerate(context_docs[:max_context_docs], 1):
        meta    = doc.get("metadata", {})
        content = doc.get("content", "")[:300]
        context_parts.append(
            f"Example {i} (stage: {meta.get('fraud_stage', '?')}, "
            f"round: {meta.get('round', '?')}):\n{content}..."
        )
    context_text = "\n\n---\n\n".join(context_parts) or "No context available."

    locale_note = _get_locale_note(locale)
    max_words   = _LOCALE_MAX_WORDS.get(locale, 180)

    user = f"""Topic: {topic}
Fraud stage: {fraud_stage}
Round: {round_num}
Locale: {locale}
{f'Previous stage: {previous_stage}' if previous_stage else ''}

{locale_note}

INSTRUCTIONS:
{instructions}

CRITICAL REALISM RULES:
- Be subtle — real phishing emails are not obvious
- Use generic authority references: "our department", "compliance team"
- Do NOT use real brand names (Google, Wells Fargo, PayPal, etc.) — use fictional variants instead (e.g. "SecureBank", "CloudMail Services")
- Use [URL] as the ONLY placeholder for links — never write "[link]", "[link to page]", actual domains, or any other link format
- Use [PHONE] as the ONLY placeholder for phone numbers — never write real phone numbers
- Replace ALL template placeholders with realistic invented values — NEVER leave brackets in the output: [Name]→"John Miller", [First Name]→"Emily", [Applicant Name]→"Michael Chen", [Recipient Name]→"Dear Customer", [Company Name]→a fictional company name, [Job Title]→"Senior Account Manager", [Date]→use a plausible date in the locale's format shown above
- Natural, slightly imperfect phrasing
- Stay under {max_words} words
- ABSOLUTELY NO disclaimers, notes, or researcher commentary inside the email body

Context examples (style reference only — DO NOT copy verbatim):
{context_text}

Generate a unique phishing email following the instructions above. Output ONLY the email — no preamble, no notes."""

    return {"system": system, "user": user}


# ── API publică ───────────────────────────────────────────────────────────

def build_prompt(
    round_num: int,
    topic: str,
    fraud_stage: str,
    context_docs: list[dict],
    locale: str = "en-US",
    previous_stage: Optional[str] = None,
    max_context_docs: int = 2,
) -> dict[str, str]:
    """
    Construiește promptul complet pentru generare.

    Folosește prompt_templates.py original dacă e disponibil,
    altfel fallback la implementarea built-in.

    Returns:
        dict cu cheile 'system' și 'user'
    """
    if _templates_loaded:
        return _original_build_prompt(
            round_num      = round_num,
            topic          = topic,
            fraud_stage    = fraud_stage,
            context_docs   = context_docs,
            previous_stage = previous_stage,
            locale         = locale,
        )
    return _build_prompt_builtin(
        round_num        = round_num,
        topic            = topic,
        fraud_stage      = fraud_stage,
        context_docs     = context_docs,
        locale           = locale,
        previous_stage   = previous_stage,
        max_context_docs = max_context_docs,
    )