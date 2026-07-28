"""
engine/custom.py
Free-text instruction handling: "mask all Unnati's transactions", "hide
Rohan's record", "redact 'Acme Corp' entries". Lets the user reach
arbitrary names/terms that no fixed field or column covers.
"""

import re

_ALL_STOPWORDS = {
    "fields", "pii", "records", "data", "information", "details",
    "aadhaar", "aadhar", "pan", "kyc", "documents", "entries",
}
_ROW_SCOPE_WORDS = re.compile(r'record|transaction|entr|statement|row|line|detail', re.I)
_ACTION_WORDS = re.compile(r'\b(?:mask|redact|hide|remove|blackout|delete|censor|scrub)\b', re.I)
_NAME_HINT_WORDS = ("name", "person", "customer", "employee")


def extract_custom_targets(text: str):
    """Returns a list of (term, mode) where mode is 'row' or 'token'."""
    if not text:
        return []

    targets = []
    scope = "row" if _ROW_SCOPE_WORDS.search(text) else "token"

    # 1. Look for quoted text: "word" or 'word'
    for pat in (re.compile(r'"([^"]+)"'), re.compile(r"'([^']+)'")):
        for m in pat.finditer(text):
            term = m.group(1).strip()
            if term:
                targets.append((term, scope))

    # 2. Look for possessive + concept: "John's record"
    for m in re.finditer(
        r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\'s\s+(record|transaction|entr|detail|data|statement)",
        text):
        targets.append((m.group(1), "row"))

    # 3. Look for "record[s]? of/for [Name]"
    for m in re.finditer(
        r'(?:record[s]?|transaction[s]?|entries)\s+(?:of|for)\s+'
        r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)', text):
        targets.append((m.group(1), "row"))

    # 4. Look for mask/redact commands with names or values.
    command_patterns = (
        re.compile(
            r"\b(?:mask|redact|hide|remove|blackout|delete|censor|scrub)\b\s+"
            r"(?:the\s+)?(?:name\s+|person\s+|customer\s+|employee\s+|record\s+|row\s+|entry\s+|address\s+|passport\s+|id\s+|date\s+|dob\s+|issue\s+|expiry\s+|number\s+|code\s+|value\s+|company\s+|organization\s+|vendor\s+|business\s+|field\s+|label\s+|document\s+|item\s+|line\s+|statement\s+|transactions?\s+)?"
            r"(?:of\s+|for\s+|called\s+|is\s+|named\s+)?"
            r"(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9][A-Za-z0-9\s\-\/&]+?))"
            r"(?=$|\s+(?:everywhere|throughout|in the document|from the document|on the page|now|please|thanks)|[.,])",
            re.I),
        re.compile(
            r"\b(?:mask|redact|hide|remove|blackout|delete|censor|scrub)\b\s+"
            r"(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9][A-Za-z0-9\s\-\/&]+?))"
            r"(?=$|\s+(?:everywhere|throughout|in the document|from the document|on the page|now|please|thanks)|[.,])",
            re.I),
    )
    for pattern in command_patterns:
        for m in pattern.finditer(text):
            term = next((group for group in m.groups() if group), None)
            if term:
                term = term.strip().strip('.,;:"\\\' ')
                if term and term.lower() not in _ALL_STOPWORDS:
                    targets.append((term, scope))

    # 5. Look for "all [Word]"
    for m in re.finditer(r'\ball\s+([A-Za-z0-9][A-Za-z0-9\s\-\/&]{0,120}?)\b', text, re.I):
        term = m.group(1).strip()
        if term.lower() not in _ALL_STOPWORDS:
            targets.append((term, scope))

    if targets:
        return targets

    # 6. Fallback: strip action words and use remaining phrase
    simple_term = text.strip()
    if simple_term and len(simple_term) > 2:
        simple_term = re.sub(r'\b(?:mask|redact|hide|remove|blackout|delete|censor|scrub|all)\b\s*', '', simple_term, flags=re.I).strip()
        if simple_term and not any(stop in simple_term.lower() for stop in _ALL_STOPWORDS):
            simple_term = simple_term.strip('.,;:"\\\' ')
            if simple_term and len(simple_term) > 1:
                return [(simple_term, "token")]

    return targets


def _line_bbox(words, lines, word_idx, img_w, img_h):
    from .ocr import words_bbox

    key = words[word_idx]["line_key"]
    for line in lines:
        if line["key"] == key:
            return words_bbox(words, line["word_idxs"], img_w, img_h)
    return None


def find_custom_target_instances(words, lines, page, img_w, img_h, term, mode, counter):
    from .ocr import words_bbox

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


def find_custom_label_value_instances(words, lines, page, img_w, img_h, label, value, counter):
    """
    Finds instances where a label appears on the same or adjacent line
    as its corresponding value, and masks the value (or the line).
    Useful for patterns like "Product Code: ABC123" or "Product: XYZ"
    """
    from .ocr import words_bbox

    if not label or not value:
        return []
    
    label_words = [w.strip(",.:;()").lower() for w in label.split() if w.strip()]
    value_words = [w.strip(",.:;()").lower() for w in value.split() if w.strip()]
    
    if not label_words or not value_words:
        return []
    
    instances = []
    tokens = [w["text"].strip(",.:;()").lower() for w in words]
    label_n = len(label_words)
    value_n = len(value_words)
    
    # Search for label and value on the same line or nearby
    for label_i in range(len(tokens) - label_n + 1):
        if tokens[label_i:label_i + label_n] == label_words:
            # Found the label, now search for the value
            # Try same line first
            for value_i in range(len(tokens) - value_n + 1):
                if tokens[value_i:value_i + value_n] == value_words:
                    # Check if they're on the same line
                    if words[label_i]["line_key"] == words[value_i]["line_key"]:
                        idxs = list(range(value_i, value_i + value_n))
                        bbox = words_bbox(words, idxs, img_w, img_h)
                        if bbox:
                            instances.append({
                                "id": counter.next(),
                                "field_type": f"custom:{label.lower()}",
                                "display_label": f'{label}: [masked]',
                                "category": "custom",
                                "value": value,
                                "page": page,
                                "bbox": bbox,
                            })
                        break
    
    return instances
