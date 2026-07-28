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
_ROW_SCOPE_WORDS = re.compile(r'\b(?:full\s+row|full\s+rows|row|rows|line|lines|statement|statements|detail|details|record|transaction|entry)\b', re.I)
_ACTION_WORDS = re.compile(r'\b(?:mask|redact|hide|remove|blackout|delete|censor|scrub)\b', re.I)
_NAME_HINT_WORDS = ("name", "person", "customer", "employee")


def extract_custom_targets(text: str):
    """Returns a list of (term, mode) where mode is 'row' or 'token'."""
    if not text:
        return []

    scope = "row" if _ROW_SCOPE_WORDS.search(text) else "token"
    targets = []
    seen_terms = set()

    def add_target(term, mode):
        term = term.strip().strip('.,;:"\\\' ')
        if not term:
            return
        term = re.sub(r"\s+", " ", term).strip()
        if term.lower() in _ALL_STOPWORDS:
            return
        if re.fullmatch(r"(?:the\s+)?(?:name|person|customer|employee|record|records|row|rows|entry|entries|address|passport|id|date|dob|issue|expiry|number|code|value|company|organization|vendor|business|field|label|document|item|line|statement|transaction|transactions|description|full|all)(?:\s+.*)?", term, re.I):
            return
        if re.fullmatch(r"(?:mask|redact|hide|remove|blackout|delete|censor|scrub|all)\b.*", term, re.I):
            return
        if term.lower() not in seen_terms:
            seen_terms.add(term.lower())
            targets.append((term, mode))

    for pat in (re.compile(r'"([^"]+)"'), re.compile(r"'([^']+)'")):
        for m in pat.finditer(text):
            add_target(m.group(1), scope)

    action_pattern = re.compile(r"\b(?:mask|redact|hide|remove|blackout|delete|censor|scrub)\b", re.I)
    for match in action_pattern.finditer(text):
        start = match.end()
        next_action = action_pattern.search(text[start:])
        end = len(text) if next_action is None else start + next_action.start()
        fragment = text[start:end]
        fragment = re.split(r"\band\b|\bor\b|\bthen\b", fragment, maxsplit=1)[0]
        fragment = fragment.strip()

        quoted_match = re.search(r'"([^"]+)"|\'([^\']+)\'', fragment)
        if quoted_match:
            add_target(quoted_match.group(1) or quoted_match.group(2), scope)
            continue

        fragment = re.sub(r"^\s*(?:the\s+)?(?:name|person|customer|employee)\s+", "", fragment, flags=re.I)
        fragment = re.sub(r"^\s*(?:all|full|the)\s+", "", fragment, flags=re.I)
        fragment = re.sub(r"^\s*(?:record|records|row|rows|entry|entries|transaction|transactions|detail|details|statement|line|item|document|description)\s+", "", fragment, flags=re.I)
        fragment = re.sub(r"^\s*(?:for|of|called|named|is)\s+", "", fragment, flags=re.I)
        fragment = re.sub(r"\s+(?:everywhere|throughout|in the document|from the document|on the page|now|please|thanks).*$", "", fragment, flags=re.I)
        fragment = re.split(r"[.,;:]", fragment, maxsplit=1)[0].strip().strip('.,;:"\\\' ')
        if fragment and not re.fullmatch(r"(?:the|all|full|row|rows|record|records|entry|entries|transaction|transactions|detail|details|statement|line|item|document|description)", fragment, re.I):
            add_target(fragment, scope)

    if not targets:
        simple_term = text.strip()
        simple_term = re.sub(r"\b(?:mask|redact|hide|remove|blackout|delete|censor|scrub|all)\b\s*", "", simple_term, flags=re.I).strip()
        simple_term = re.sub(r"\s+(?:everywhere|throughout|in the document|from the document|on the page|now|please|thanks).*$", "", simple_term, flags=re.I)
        simple_term = re.split(r"[.,;:]", simple_term, maxsplit=1)[0].strip().strip('.,;:"\\\' ')
        if simple_term and len(simple_term) > 1:
            add_target(simple_term, "token")

    return sorted(targets, key=lambda item: text.lower().find(item[0].lower()))


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
