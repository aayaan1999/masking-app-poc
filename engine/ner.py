"""
engine/ner.py
Optional generic entity detector built on spaCy (open-source NER model).
Catches PERSON / ORG / GPE / MONEY mentions in free-form text that the
regex/label detectors miss — e.g. a name mentioned mid-sentence on a
cover letter, or a company name on an invoice.

This is best-effort and OPTIONAL: if spaCy or its English model isn't
installed, every function here quietly returns [] and the rest of the
pipeline (regex detectors + table detector) still works unmodified.
Install with:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

from .ocr import words_bbox

_NLP = None
_LOAD_ATTEMPTED = False

_LABEL_MAP = {
    "PERSON": ("entity:person", "Person Name (AI-detected)", "identity"),
    "ORG": ("entity:org", "Organization (AI-detected)", "generic"),
    "GPE": ("entity:gpe", "Location (AI-detected)", "generic"),
    "LOC": ("entity:gpe", "Location (AI-detected)", "generic"),
    "MONEY": ("entity:money", "Money Amount (AI-detected)", "financial"),
}


def _get_nlp():
    global _NLP, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _NLP
    _LOAD_ATTEMPTED = True
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except Exception:
        _NLP = None
    return _NLP


def ner_available() -> bool:
    return _get_nlp() is not None


def _find_span_word_idxs(words, line_word_idxs, span_text):
    span_tokens = [t.lower() for t in span_text.split()]
    n = len(span_tokens)
    line_tokens = [words[i]["text"].strip(",.:;()").lower() for i in line_word_idxs]
    for start in range(len(line_tokens) - n + 1):
        if line_tokens[start:start + n] == span_tokens:
            return line_word_idxs[start:start + n]
    return None


def detect_entities(words, lines, page, img_w, img_h, counter, already_claimed):
    nlp = _get_nlp()
    if nlp is None:
        return []

    instances = []
    for line in lines:
        remaining_idxs = [i for i in line["word_idxs"] if i not in already_claimed]
        if len(remaining_idxs) < len(line["word_idxs"]) * 0.5:
            continue  # line mostly already masked by another detector — skip

        # Low average OCR confidence on this line usually means garbled
        # text (very common for a script Tesseract's active language
        # models can't read, e.g. Arabic without an Arabic pack
        # installed) — feeding that into an English NER model tends to
        # produce nonsense "entities" rather than real ones, so skip it.
        confs = [words[i]["conf"] for i in line["word_idxs"] if words[i]["conf"] >= 0]
        if confs and (sum(confs) / len(confs)) < 50:
            continue

        text = line["text"].strip()
        if len(text) < 3:
            continue
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ not in _LABEL_MAP:
                continue
            clean = ent.text.strip()
            # Single very short "word" entities (1-2 letters) are almost
            # always OCR noise fragments, not real names/orgs/places.
            if len(clean) < 3 or not any(c.isalpha() for c in clean):
                continue
            field_type, display_label, category = _LABEL_MAP[ent.label_]
            idxs = _find_span_word_idxs(words, line["word_idxs"], ent.text) or line["word_idxs"]
            bbox = words_bbox(words, idxs, img_w, img_h)
            instances.append({
                "id": counter.next(), "field_type": field_type,
                "display_label": display_label, "category": category,
                "value": ent.text, "page": page, "bbox": bbox,
            })
    return instances
