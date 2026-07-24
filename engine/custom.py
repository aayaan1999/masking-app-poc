"""
engine/custom.py
Free-text instruction handling: "mask all Unnati's transactions", "hide
Rohan's record", "redact 'Acme Corp' entries". Lets the user reach
arbitrary names/terms that no fixed field or column covers.
"""

import re
from .ocr import words_bbox

_ALL_STOPWORDS = {
    "fields", "pii", "records", "data", "information", "details",
    "aadhaar", "aadhar", "pan", "kyc", "documents", "entries",
}
_ROW_SCOPE_WORDS = re.compile(r'record|transaction|entr|statement|row|line|detail', re.I)


def extract_custom_targets(text: str):
    """Returns a list of (term, mode) where mode is 'row' or 'token'."""
    targets = []
    scope = "row" if _ROW_SCOPE_WORDS.search(text) else "token"

    for pat in (re.compile(r'"([^"]+)"'), re.compile(r"'([^']+)'")):
        for m in pat.finditer(text):
            term = m.group(1).strip()
            if term and not term.lower().endswith("s"):
                targets.append((term, scope))
    if targets:
        return targets

    m = re.search(
        r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\'s\s+"
        r"(record|transaction|entr|detail|data|statement)", text)
    if m:
        return [(m.group(1), "row")]

    m = re.search(
        r'(?:record[s]?|transaction[s]?|entries)\s+(?:of|for)\s+'
        r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)', text)
    if m:
        return [(m.group(1), "row")]

    m = re.search(r'\ball\s+([A-Z][a-zA-Z]+)\b', text)
    if m and m.group(1).lower() not in _ALL_STOPWORDS:
        return [(m.group(1), scope)]

    return targets


def _line_bbox(words, lines, word_idx, img_w, img_h):
    key = words[word_idx]["line_key"]
    for line in lines:
        if line["key"] == key:
            return words_bbox(words, line["word_idxs"], img_w, img_h)
    return None


def find_custom_target_instances(words, lines, page, img_w, img_h, term, mode, counter):
    term_words = [w.strip(",.:;()").lower() for w in term.split() if w.strip()]
    if not term_words:
        return []
    n = len(term_words)
    tokens = [w["text"].strip(",.:;()").lower() for w in words]
    instances = []
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == term_words:
            if mode == "row":
                bbox = _line_bbox(words, lines, i, img_w, img_h)
                idxs = list(range(i, i + n))
            else:
                idxs = list(range(i, i + n))
                bbox = words_bbox(words, idxs, img_w, img_h)
            if bbox:
                instances.append({
                    "id": counter.next(), "field_type": f"custom:{term.lower()}",
                    "display_label": f'"{term}"', "category": "custom",
                    "value": term, "page": page, "bbox": bbox,
                })
    return instances
