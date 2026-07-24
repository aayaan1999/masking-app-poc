"""
engine/gcc_ids.py
Passport numbers and national ID numbers for the GCC states (UAE,
Saudi Arabia, Qatar, Kuwait, Bahrain, Oman).

Some of these have a near-unique shape (UAE's Emirates ID always starts
"784-") and can be matched on pattern alone. Most GCC national ID
numbers are just N plain digits, though, which is indistinguishable
from a dozen other things a document might contain — so for every
pattern that isn't distinctive on its own, a match only counts if a
recognized label (English, Urdu, or Arabic — see i18n_labels) for
"passport" or "national id" appears on the same line. That keeps this
from flooding the results with false positives on any random long
number.
"""

import re
from .ocr import words_bbox
from . import i18n_labels

# Distinctive enough to match on pattern alone.
EMIRATES_ID = re.compile(r'\b784-?\d{4}-?\d{7}-?\d{1}\b')

# Ambiguous shapes — digits only look like a GCC national ID when a
# passport/ID label is present on the same line (see below).
SAUDI_ID = re.compile(r'\b[12]\d{9}\b')          # Saudi national ID / Iqama
QATAR_QID = re.compile(r'\b\d{11}\b')             # Qatar QID
KUWAIT_CIVIL_ID = re.compile(r'\b\d{12}\b')       # Kuwait Civil ID
BAHRAIN_CPR = re.compile(r'\b\d{9}\b')            # Bahrain CPR
OMAN_ID = re.compile(r'\b\d{8}\b')                # Oman civil/resident ID

# Passport numbers vary by issuing country but are almost always
# 1-2 letters followed by 6-9 digits, or purely numeric.
PASSPORT_ALNUM = re.compile(r'\b[A-Z]{1,2}\d{6,9}\b')
PASSPORT_NUMERIC = re.compile(r'\b\d{7,9}\b')

_AMBIGUOUS_ID_PATTERNS = [SAUDI_ID, QATAR_QID, KUWAIT_CIVIL_ID, BAHRAIN_CPR, OMAN_ID]


def _line_has_concept(line_text, concept):
    return i18n_labels.line_matches_concept(line_text, concept)


def detect_emirates_id(words, lines, page, img_w, img_h, counter):
    out, seen = [], set()
    for i, w in enumerate(words):
        if EMIRATES_ID.search(w["text"]) and w["conf"] > 15:
            out.append({
                "id": counter.next(), "field_type": "national_id_uae",
                "display_label": "Emirates ID Number", "category": "identity",
                "value": w["text"], "page": page,
                "bbox": words_bbox(words, [i], img_w, img_h),
            })
            seen.add(i)
    return out, seen


def detect_gcc_national_id(words, lines, page, img_w, img_h, counter, already_claimed):
    """
    Digits-only GCC ID numbers — requires a "national id" style label
    (in any supported language) on the same line to fire at all.
    """
    out, seen = [], set()
    for line in lines:
        if not _line_has_concept(line["text"], "national_id"):
            continue
        for i in line["word_idxs"]:
            if i in already_claimed or i in seen:
                continue
            t = words[i]["text"]
            if any(p.match(t) for p in _AMBIGUOUS_ID_PATTERNS) and words[i]["conf"] > 20:
                out.append({
                    "id": counter.next(), "field_type": "national_id",
                    "display_label": "National / Civil ID Number", "category": "identity",
                    "value": t, "page": page,
                    "bbox": words_bbox(words, [i], img_w, img_h),
                })
                seen.add(i)
    return out, seen


def detect_passport_number(words, lines, page, img_w, img_h, counter, already_claimed):
    """
    Passport numbers requires a "passport" label (English/Urdu/Arabic)
    on the same line — the numeric-only shape is too ambiguous with
    phone numbers, IDs, etc. otherwise.
    """
    out, seen = [], set()
    for line in lines:
        if not _line_has_concept(line["text"], "passport_no"):
            continue
        for i in line["word_idxs"]:
            if i in already_claimed or i in seen:
                continue
            t = words[i]["text"]
            if (PASSPORT_ALNUM.match(t) or PASSPORT_NUMERIC.match(t)) and words[i]["conf"] > 20:
                out.append({
                    "id": counter.next(), "field_type": "passport_number",
                    "display_label": "Passport Number", "category": "identity",
                    "value": t, "page": page,
                    "bbox": words_bbox(words, [i], img_w, img_h),
                })
                seen.add(i)
    return out, seen


def run_gcc_detectors(words, lines, page, img_w, img_h, counter, already_claimed):
    instances = []
    claimed = set()

    found, seen = detect_emirates_id(words, lines, page, img_w, img_h, counter)
    instances += found
    claimed |= seen

    found, seen = detect_gcc_national_id(words, lines, page, img_w, img_h, counter,
                                          already_claimed | claimed)
    instances += found
    claimed |= seen

    found, seen = detect_passport_number(words, lines, page, img_w, img_h, counter,
                                          already_claimed | claimed)
    instances += found
    claimed |= seen

    return instances, claimed
