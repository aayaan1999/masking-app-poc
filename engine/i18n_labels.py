"""
engine/i18n_labels.py
GCC identity documents are frequently bilingual — English printed
alongside Urdu (and often Arabic, since Urdu shares its script and a
lot of expat-labor documentation in the Gulf carries both). This module
is the single place that maps a *concept* (date of birth, passport
number, ...) to every keyword we recognize for it, in every supported
language, so every detector stays in sync.

Matching is done as a plain substring test against OCR'd line text, not
word-by-word — this is what lets it work regardless of reading
direction. Tesseract lays LTR and RTL words out left-to-right by pixel
position; we never rely on "the value is after the label", only on
"the value is somewhere else on this same line" (see
detectors.line_value_excluding_label).

Arabic/Urdu text is normalized before comparison (see normalize()) —
Arabic has several "alef" letterforms (ا / أ / إ / آ) that fonts and
OCR both use interchangeably for the same word (e.g. a printed card
using "الإسم" won't literally contain the substring "الاسم"), plus
optional diacritics that add no meaning for matching purposes. Without
normalizing both the stored keyword and the OCR'd text the same way, a
label spelled with a different alef variant than whatever's hardcoded
here silently fails to match — which is exactly what happened with
"Name" on a real Emirates ID card.
"""

import re

_ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640]')
_ARABIC_ALEF_VARIANTS = re.compile(r'[إأآٱ]')


def normalize(text: str) -> str:
    """Lowercase + fold Arabic letterform/diacritic variants so keyword
    matching is robust to OCR/font spelling differences. No-op (beyond
    lowercasing) for non-Arabic-script text."""
    text = text.lower()
    text = _ARABIC_DIACRITICS.sub('', text)
    text = _ARABIC_ALEF_VARIANTS.sub('ا', text)
    text = text.replace('ة', 'ه').replace('ى', 'ي')
    return text

LABELS = {
    "name": {
        "en": ["name"],
        "ur": ["نام"],
        "ar": ["الاسم", "اسم"],
    },
    "father_name": {
        "en": ["father's name", "father name", "s/o"],
        "ur": ["والد کا نام", "ولد"],
        "ar": ["اسم الأب"],
    },
    "dob": {
        "en": ["date of birth", "birth date", "dob", "d.o.b"],
        "ur": ["تاریخ پیدائش"],
        "ar": ["تاريخ الميلاد"],
    },
    "date_of_issue": {
        "en": ["date of issue", "issue date", "issued on", "issuing date", "issue"],
        "ur": ["تاریخ اجراء", "اجراء کی تاریخ"],
        "ar": ["تاريخ الإصدار", "الإصدار"],
    },
    "date_of_expiry": {
        "en": ["date of expiry", "expiry date", "expiration date",
               "date of validation", "valid until", "validity", "valid till", "expiry"],
        "ur": ["تاریخ اختتام", "میعاد ختم", "تاریخ ختم"],
        "ar": ["تاريخ الانتهاء", "تاريخ الانتهاء الصلاحية", "الانتهاء"],
    },
    "nationality": {
        "en": ["nationality"],
        "ur": ["قومیت"],
        "ar": ["الجنسية"],
    },
    "gender": {
        "en": ["gender", "sex"],
        "ur": ["جنس"],
        "ar": ["الجنس"],
    },
    "address": {
        "en": ["address"],
        "ur": ["پتہ", "پتا"],
        "ar": ["العنوان"],
    },
    "passport_no": {
        "en": ["passport no", "passport number", "passport num"],
        "ur": ["پاسپورٹ نمبر"],
        "ar": ["رقم الجواز", "رقم جواز السفر"],
    },
    "national_id": {
        "en": ["national id", "id number", "id no", "identity number",
               "identity no", "civil id", "emirates id", "resident id",
               "iqama no", "iqama number", "qid"],
        "ur": ["شناختی کارڈ", "قومی شناختی کارڈ", "شناختی نمبر"],
        "ar": ["رقم الهوية", "الهوية الوطنية", "رقم الإقامة"],
    },
}


def all_keywords(concept: str):
    """Flat list of every keyword (all languages) for a concept."""
    langs = LABELS.get(concept, {})
    out = []
    for kws in langs.values():
        out.extend(kws)
    return out


def line_matches_concept(line_text: str, concept: str) -> bool:
    return contains_any_keyword(line_text, all_keywords(concept))


def matched_concepts(line_text: str):
    """Every concept whose keyword appears in this line."""
    return [c for c in LABELS if contains_any_keyword(line_text, all_keywords(c))]


def find_keyword_in_text(line_text: str, concept: str):
    """Returns the first matching keyword (original spelling, for
    locating its words) for this concept in this line, or None."""
    norm_line = normalize(line_text)
    for kw in all_keywords(concept):
        if normalize(kw) in norm_line:
            return kw
    return None


def _tokenize(text: str):
    return re.findall(r'\w+', normalize(text), re.UNICODE)


def contains_keyword(line_text: str, keyword: str) -> bool:
    """
    Word-boundary-aware match: True only if `keyword` appears as a
    whole token (or contiguous run of tokens, for multi-word keywords)
    in line_text — not merely as a substring. Plain substring matching
    on common short words like "state" or "pin" produces false
    positives inside unrelated words ("United States", "pinch"); this
    tokenizes both sides first so "state" no longer matches "states".
    """
    kw_tokens = _tokenize(keyword)
    if not kw_tokens:
        return False
    line_tokens = _tokenize(line_text)
    n = len(kw_tokens)
    return any(line_tokens[i:i + n] == kw_tokens for i in range(len(line_tokens) - n + 1))


def contains_any_keyword(line_text: str, keywords) -> bool:
    return any(contains_keyword(line_text, kw) for kw in keywords)
