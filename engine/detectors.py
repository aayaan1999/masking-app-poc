"""
engine/detectors.py
Detects specific, well-known PII fields (Aadhaar, PAN, DOB, phone, email,
card numbers, address, names) plus a generic "any Label: Value" detector
that catches fields the specific detectors don't know about (Father's
Name, Account Number, IFSC Code, Policy No, ...). Every detector returns
a list of `instance` dicts with a common shape so the pipeline can treat
them uniformly.
"""

import re
from .ocr import words_bbox
from . import i18n_labels

AADHAAR_12 = re.compile(r'^\d{12}$')
AADHAAR_4DIGIT = re.compile(r'^\d{4}$')
AADHAAR_8DIGIT = re.compile(r'^\d{8}$')
PAN_PATTERN = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
PHONE_PATTERN = re.compile(r'\b[6-9]\d{9}\b|\b\+91[-\s]?\d{10}\b')
EMAIL_PATTERN = re.compile(r'\b[\w._%+-]+@[\w.-]+\.\w{2,}\b')
CARD_FULL = re.compile(r'^\d{13,19}$')
CARD_GROUP_4 = re.compile(r'^\d{4}$')
PIN_PATTERN = re.compile(r'\b\d{6}\b')
PIN_FULLTOKEN = re.compile(r'^\d{6}$')  # whole-token match, so "105000.00" doesn't qualify
IFSC_PATTERN = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')
ACCOUNT_NO_PATTERN = re.compile(r'\b\d{9,18}\b')

NAME_KEYWORDS = ["name", "नाम"] + i18n_labels.all_keywords("name")
ADDR_KEYWORDS = ["address", "पता", "addr", "s/o", "w/o", "d/o", "house",
                 "village", "dist", "pin", "state", "road", "nagar", "colony"] + i18n_labels.all_keywords("address")

# Labels the specific detectors already own — the generic detector skips
# these so a field isn't reported twice under two different names.
_OWNED_LABEL_PATTERNS = re.compile(
    r'aadhaar|aadhar|uid|pan\b|permanent account|date of birth|\bdob\b|'
    r'd\.o\.b|phone|mobile|contact no|e-?mail|address|card number|'
    r'card no|debit card|credit card|date of issue|issue date|'
    r'date of expiry|expiry date|expiration date|validity|valid until|'
    r'valid till|date of validation|passport no|passport number|'
    r'national id|civil id|emirates id|iqama|\bqid\b|identity number',
    re.I,
)

_LABEL_LINE = re.compile(r'^\s*([A-Za-z][A-Za-z .\'/]{1,40}?)\s*[:\-]\s*(.+)$')

_KNOWN_LABEL_TOKENS = {
    "name", "date", "birth", "dob", "account", "number", "phone", "mobile",
    "email", "address", "passport", "policy", "customer", "employee",
    "license", "registration", "father", "mother", "national", "civil",
    "emirates", "iqama", "pan", "aadhaar", "card", "id",    "gender", "sex",}


def _mk(field_type, display_label, category, value, page, bbox, iid):
    return {
        "id": iid, "field_type": field_type, "display_label": display_label,
        "category": category, "value": value, "page": page, "bbox": bbox,
    }


class InstanceCounter:
    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return f"i{self.n}"


_ACCOUNT_CONTEXT = re.compile(r'account|a/c\b|acct', re.I)


def detect_aadhaar_number(words, lines, page, img_w, img_h, counter):
    out, seen = [], set()
    line_text_by_key = {l["key"]: l["text"] for l in lines}
    for i, w in enumerate(words):
        if i in seen or (w["conf"] is None or w["conf"] < 40):
            continue
        t = w["text"]
        # A bare 12-digit number is ambiguous with a bank account number —
        # if this word's own row talks about an account, treat it as one
        # rather than an Aadhaar number.
        if _ACCOUNT_CONTEXT.search(line_text_by_key.get(w.get("line_key"), "")):
            continue
        if AADHAAR_12.match(t):
            out.append(_mk("aadhaar_number", "Aadhaar Number", "identity", t,
                            page, words_bbox(words, [i], img_w, img_h), counter.next()))
            seen.add(i)
        elif (i < len(words) - 1 and AADHAAR_4DIGIT.match(t)
              and AADHAAR_8DIGIT.match(words[i + 1]["text"])
              and words[i]["line_key"] == words[i + 1]["line_key"]):
            idxs = [i, i + 1]
            val = " ".join(words[k]["text"] for k in idxs)
            out.append(_mk("aadhaar_number", "Aadhaar Number", "identity", val,
                            page, words_bbox(words, idxs, img_w, img_h), counter.next()))
            seen.update(idxs)
        elif (i < len(words) - 2 and AADHAAR_4DIGIT.match(t)
              and AADHAAR_4DIGIT.match(words[i + 1]["text"])
              and AADHAAR_4DIGIT.match(words[i + 2]["text"])
              and words[i]["line_key"] == words[i + 1]["line_key"] == words[i + 2]["line_key"]):
            idxs = [i, i + 1, i + 2]
            val = " ".join(words[k]["text"] for k in idxs)
            out.append(_mk("aadhaar_number", "Aadhaar Number", "identity", val,
                            page, words_bbox(words, idxs, img_w, img_h), counter.next()))
            seen.update(idxs)
    return out, seen


def detect_pan_number(words, lines, page, img_w, img_h, counter):
    out, seen = [], set()
    for i, w in enumerate(words):
        if PAN_PATTERN.search(w["text"]) and (w["conf"] is not None and w["conf"] > 10):
            out.append(_mk("pan_number", "PAN Number", "identity", w["text"],
                            page, words_bbox(words, [i], img_w, img_h), counter.next()))
            seen.add(i)
    return out, seen


DATE_PATTERN = re.compile(
    r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b|'
    r'\b\d{1,2}[\-\s][A-Za-z]{3,9}[\-\s]\d{2,4}\b'
)

DATE_CONCEPT_LABELS = {
    "dob": "Date of Birth",
    "date_of_issue": "Date of Issue",
    "date_of_expiry": "Date of Expiry",
}


def _normalize_label_text(text):
    return re.sub(r'\s+', ' ', text or "").strip(" ,.:;()[]{}")


def _looks_like_value(text):
    text = _normalize_label_text(text)
    if not text:
        return False
    if re.search(r'\d', text):
        return True
    if EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text):
        return True
    if DATE_PATTERN.search(text):
        return True
    parts = [p for p in text.split() if p]
    if len(parts) >= 2:
        return True
    return len(parts) == 1 and len(parts[0]) >= 2 and parts[0].lower() not in {"the", "and", "for", "of", "to"}


def _label_is_probable(label_text):
    label_text = _normalize_label_text(label_text).lower()
    if not label_text:
        return False
    if label_text in {"note", "important", "instructions"}:
        return False
    tokens = [t for t in re.split(r"\s+", label_text) if t]
    if len(tokens) > 1:
        last_token = re.sub(r"[^a-z0-9]+", "", tokens[-1])
        if not (last_token in _KNOWN_LABEL_TOKENS or last_token.endswith(("name", "number", "address", "id"))):
            return False
    if any(token in label_text for token in _KNOWN_LABEL_TOKENS):
        return True
    if label_text.endswith("name") or label_text.endswith("number") or label_text.endswith("address") or label_text.endswith("id"):
        return True
    return False


# Short capitalized words that are almost always *values* (a gender,
# title, marital status, ...) rather than the start of a field label —
# without this, a capitalized value sitting right before an unrelated
# later label ("... Male Treating Physician:") looks indistinguishable
# from a two-word label ("Male Treating") under pure capitalization.
_INLINE_VALUE_WORDS = {
    "male", "female", "mr", "mrs", "ms", "miss", "dr", "married", "single",
    "unmarried", "widowed", "divorced", "yes", "no", "na",
}


def _looks_like_inline_label(phrase):
    """
    True for a short, non-numeric, mostly-capitalized phrase — the shape
    of a field label ("Treating Physician", "Reg No", "Patient ID") as
    opposed to a value. Used only to find where a *second* label starts
    partway through an OCR line, not to fully validate a label the way
    _label_is_probable does.
    """
    phrase = phrase.strip(" .")
    if not phrase:
        return False
    tokens = phrase.split()
    if not (1 <= len(tokens) <= 4):
        return False
    if re.search(r'\d', phrase):
        return False
    if not all(re.match(r"^[A-Za-z][A-Za-z.'/]*$", t) for t in tokens):
        return False
    if tokens[0].strip(".").lower() in _INLINE_VALUE_WORDS:
        return False
    cap_count = sum(1 for t in tokens if t[:1].isupper())
    return cap_count >= max(1, len(tokens) - 1)


def _trim_value_at_next_label(words, idxs):
    """
    idxs: ordered (left-to-right) word indices making up a candidate
    field value. Bank/ID/patient forms routinely pack several "Label:
    value" pairs onto one printed row that OCR clusters into a single
    line (e.g. "Age / Gender: 34 / Male  Treating Physician: Dr. S. K.
    Verma (Reg No: 44582)"). Without this, a value span that isn't
    explicitly bounded elsewhere runs to the end of that whole OCR line,
    so redacting one field also redacts its unrelated neighbors on the
    same row. This trims the span the moment a second label-shaped
    phrase followed by a colon appears.
    """
    n = len(idxs)
    for t in range(n):
        tok = words[idxs[t]]["text"]
        if not (tok.endswith(":") or tok == ":"):
            continue
        last_tok = tok.rstrip(":")
        # Try the longest candidate label first (e.g. prefer "Treating
        # Physician" over just "Physician") so the value gets cut before
        # the label's own first word, not partway through it. Capped at 2
        # words back (not the 4 _looks_like_inline_label otherwise
        # allows): a real multi-word *value* — a first+last name being
        # the most common case — is just as capitalized as a label, so
        # searching further back risks eating the value's own trailing
        # words (e.g. "...Smith Patient ID:" misreading "Smith Patient
        # ID" as one 3-word label and truncating the surname).
        for label_len in range(min(2, t), 0, -1):
            start = t - label_len + 1
            if start <= 0:
                continue
            raw = [words[idxs[i]]["text"] for i in range(start, t)]
            phrase = " ".join(raw + ([last_tok] if last_tok else [])).strip()
            if _looks_like_inline_label(phrase):
                return idxs[:start]
    return idxs


def _extract_label_value_pair(words, line, next_line=None):
    line_text = _normalize_label_text(line["text"])
    if not line_text:
        return None

    m = _LABEL_LINE.match(line_text)
    if m:
        label, value = m.group(1).strip(), m.group(2).strip()
        if _label_is_probable(label) and _looks_like_value(value):
            idxs = line["word_idxs"]
            colon_pos = next((i for i, wi in enumerate(idxs) if ":" in words[wi]["text"]), None)
            value_idxs = list(idxs[colon_pos + 1:]) if colon_pos is not None else list(idxs[len(label.split()):])
            value_idxs = _trim_value_at_next_label(words, value_idxs)
            if value_idxs:
                value = " ".join(words[i]["text"] for i in value_idxs)
                return label, value, value_idxs

    def _next_line_value():
        next_parts = [words[i]["text"].strip(" ,.:;()[]{}") for i in next_line["word_idxs"]]
        next_parts = [p for p in next_parts if p]
        if not next_parts:
            return None
        value_idxs = _trim_value_at_next_label(words, list(next_line["word_idxs"]))
        if not value_idxs:
            return None
        value_text = " ".join(words[i]["text"] for i in value_idxs)
        if _looks_like_value(value_text):
            return value_text, value_idxs
        return None

    parts = [words[i]["text"].strip(" ,.:;()[]{}") for i in line["word_idxs"]]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        if next_line is not None and _label_is_probable(line_text):
            nv = _next_line_value()
            if nv:
                return line_text, nv[0], nv[1]
        return None

    if next_line is not None and _label_is_probable(line_text) and not re.search(r'\d', line_text) and not EMAIL_PATTERN.search(line_text) and not PHONE_PATTERN.search(line_text) and not DATE_PATTERN.search(line_text):
        nv = _next_line_value()
        if nv:
            return line_text, nv[0], nv[1]

    for label_len in range(min(4, len(parts)), 0, -1):
        label = " ".join(parts[:label_len])
        value_parts = parts[label_len:]
        if not value_parts:
            continue
        value_idxs = _trim_value_at_next_label(words, list(line["word_idxs"][label_len:]))
        if not value_idxs:
            continue
        value_text = " ".join(words[i]["text"] for i in value_idxs)
        if _label_is_probable(label) and _looks_like_value(value_text):
            return label, value_text, value_idxs

    if next_line is not None and _label_is_probable(line_text):
        nv = _next_line_value()
        if nv:
            return line_text, nv[0], nv[1]
    return None


def detect_labelled_dates(words, lines, page, img_w, img_h, counter):
    """
    Finds date-shaped tokens and classifies each one by whichever
    date-related label (DOB / issue / expiry) appears on the *same
    line* or on a nearby line. This handles scanned cards where the
    label and value are separated vertically.
    """
    out, seen = [], set()

    def line_date_idxs(line):
        return [i for i in line["word_idxs"] if DATE_PATTERN.search(words[i]["text"])
                and (words[i]["conf"] is not None and words[i]["conf"] > 20)]

    for li, line in enumerate(lines):
        concepts_here = [c for c in ("dob", "date_of_issue", "date_of_expiry")
                          if i18n_labels.line_matches_concept(line["text"], c)]
        date_idxs = line_date_idxs(line)

        if concepts_here and not date_idxs:
            # look up to two nearby lines for a matching date value
            for neighbor in lines[li + 1:li + 3]:
                date_idxs = line_date_idxs(neighbor)
                if date_idxs:
                    break

        if not concepts_here and date_idxs:
            # if a date appears without a concept, search nearby lines for one
            for neighbor in lines[max(0, li - 2):li]:
                neighbor_concepts = [c for c in ("dob", "date_of_issue", "date_of_expiry")
                                     if i18n_labels.line_matches_concept(neighbor["text"], c)]
                if neighbor_concepts:
                    concepts_here = neighbor_concepts
                    break

        if not date_idxs:
            continue

        if concepts_here:
            # If multiple concepts and multiple dates appear together,
            # assign each concept to the nearest date token when possible.
            if len(concepts_here) > 1 and len(date_idxs) > 1:
                date_positions = [((words[i]["left"] + words[i]["right"]) / 2, i)
                                  for i in date_idxs]
                concept_positions = []
                for concept in concepts_here:
                    kw = i18n_labels.find_keyword_in_text(line["text"], concept)
                    x = _find_keyword_x_center(words, line["word_idxs"], kw) if kw else (line["left"] + line["right"]) / 2
                    concept_positions.append((concept, x))
                assigned = set()
                for concept, x in sorted(concept_positions, key=lambda t: t[1]):
                    best = min(
                        ((abs(pos_x - x), idx) for pos_x, idx in date_positions if idx not in assigned),
                        default=None,
                    )
                    if best is None:
                        best = min(((abs(pos_x - x), idx) for pos_x, idx in date_positions), default=None)
                    if best:
                        _, idx = best
                        assigned.add(idx)
                        val = words[idx]["text"]
                        out.append(_mk(concept, DATE_CONCEPT_LABELS[concept], "identity", val,
                                        page, words_bbox(words, [idx], img_w, img_h), counter.next()))
            else:
                for concept in concepts_here:
                    val = " ".join(words[i]["text"] for i in date_idxs)
                    out.append(_mk(concept, DATE_CONCEPT_LABELS[concept], "identity", val,
                                    page, words_bbox(words, date_idxs, img_w, img_h), counter.next()))
        else:
            for i in date_idxs:
                out.append(_mk("date_unlabelled", "Date (unlabelled)", "generic", words[i]["text"],
                                page, words_bbox(words, [i], img_w, img_h), counter.next()))
        seen.update(date_idxs)
    return out, seen


def _find_keyword_x_center(words, line_word_idxs, keyword):
    """x-center of a (possibly multi-word) keyword's position within a line, or None."""
    kw_tokens = i18n_labels.normalize(keyword).split()
    n = len(kw_tokens)
    line_tokens = [i18n_labels.normalize(words[i]["text"].strip(",.:;()")) for i in line_word_idxs]
    for start in range(len(line_tokens) - n + 1):
        if line_tokens[start:start + n] == kw_tokens:
            idxs = line_word_idxs[start:start + n]
            xs = [(words[i]["left"] + words[i]["right"]) / 2 for i in idxs]
            return sum(xs) / len(xs)
    return None


def reassociate_unlabelled_dates(instances, lines, words):
    """
    Handles a very common ID-card footer layout: a row of date LABELS
    ("Date of Birth | Date of Issue | Date of Expiry") sitting directly
    above a row of date VALUES, rather than each label sharing a line
    with its own value. detect_labelled_dates only matches same-line
    pairs, so those dates come back as "date_unlabelled" — this looks
    for the nearest label row within a few line-heights and reassigns
    each date to whichever label sits closest to it horizontally.
    """
    label_positions = []  # (line, concept, x_center)
    for line in lines:
        for concept in ("dob", "date_of_issue", "date_of_expiry"):
            for kw in i18n_labels.all_keywords(concept):
                x = _find_keyword_x_center(words, line["word_idxs"], kw)
                if x is not None:
                    label_positions.append((line, concept, x))
                    break
    if not label_positions:
        return instances

    out = []
    for inst in instances:
        if inst["field_type"] != "date_unlabelled":
            out.append(inst)
            continue
        l, t, r, b = inst["bbox"]
        cx, cy = (l + r) / 2, (t + b) / 2
        row_h = max(1, b - t)
        best_concept, best_dist = None, None
        for line, concept, x in label_positions:
            line_cy = (line["top"] + line["bottom"]) / 2
            vert_dist = abs(line_cy - cy)
            if vert_dist > row_h * 6:  # only consider a nearby row, not the whole page
                continue
            dist = abs(x - cx) + vert_dist * 0.3
            if best_dist is None or dist < best_dist:
                best_dist, best_concept = dist, concept
        if best_concept:
            inst = dict(inst)
            inst["field_type"] = best_concept
            inst["display_label"] = DATE_CONCEPT_LABELS[best_concept]
            inst["category"] = "identity"
        out.append(inst)
    return out


def detect_phone(words, lines, page, img_w, img_h, counter):
    out, seen = [], set()
    for i, w in enumerate(words):
        if PHONE_PATTERN.search(w["text"]) and (w["conf"] is not None and w["conf"] > 35):
            out.append(_mk("phone_number", "Phone Number", "contact", w["text"],
                            page, words_bbox(words, [i], img_w, img_h), counter.next()))
            seen.add(i)
    return out, seen


def detect_email(words, lines, page, img_w, img_h, counter):
    out, seen = [], set()
    for i, w in enumerate(words):
        if EMAIL_PATTERN.search(w["text"]) and (w["conf"] is not None and w["conf"] > 35):
            out.append(_mk("email", "Email Address", "contact", w["text"],
                            page, words_bbox(words, [i], img_w, img_h), counter.next()))
            seen.add(i)
    return out, seen


def detect_card_number(words, lines, page, img_w, img_h, counter):
    out, seen = [], set()
    n = len(words)
    for i in range(n):
        if i in seen:
            continue
        w = words[i]
        if CARD_FULL.match(w["text"]) and (w["conf"] is not None and w["conf"] > 15) and not (
                len(w["text"]) == 15 and w["text"].startswith("784")):
            out.append(_mk("credit_card_number", "Card Number", "financial", w["text"],
                            page, words_bbox(words, [i], img_w, img_h), counter.next()))
            seen.add(i)
            continue
        if CARD_GROUP_4.match(w["text"]) and (w["conf"] is not None and w["conf"] > 25):
            group = [i]
            j = i + 1
            while (j < n and len(group) < 4 and CARD_GROUP_4.match(words[j]["text"])
                   and (words[j]["conf"] is not None and words[j]["conf"] > 25) and words[j]["line_key"] == w["line_key"]):
                group.append(j)
                j += 1
            if len(group) >= 3:
                val = " ".join(words[k]["text"] for k in group)
                out.append(_mk("credit_card_number", "Card Number", "financial", val,
                                page, words_bbox(words, group, img_w, img_h), counter.next()))
                seen.update(group)
    return out, seen


def detect_address(words, lines, page, img_w, img_h, counter):
    """
    Line-based: an address label pulls in the rest of its own printed
    row, plus at most one following row if that row doesn't look like
    the start of a *different* labelled field (so a 2-line address
    wrapped without a repeated label is still fully covered, without
    also sweeping up the next unrelated field).
    """
    out, seen = [], set()
    claimed_lines = set()
    for li, line in enumerate(lines):
        if not i18n_labels.contains_any_keyword(line["text"], ADDR_KEYWORDS):
            continue
        if li in claimed_lines:
            continue
        idxs = _trim_value_at_next_label(words, list(line["word_idxs"]))
        claimed_lines.add(li)

        if li + 1 < len(lines):
            nxt = lines[li + 1]
            gap = nxt["top"] - line["bottom"]
            avg_h = max(1, line["bottom"] - line["top"])
            if gap < avg_h * 1.5 and not _LABEL_LINE.match(nxt["text"]):
                idxs += _trim_value_at_next_label(words, list(nxt["word_idxs"]))
                claimed_lines.add(li + 1)

        val = " ".join(words[j]["text"] for j in idxs)[:80]
        out.append(_mk("address", "Address", "contact", val,
                        page, words_bbox(words, idxs, img_w, img_h), counter.next()))
        seen.update(idxs)

    for i, w in enumerate(words):
        if i in seen:
            continue
        tl = w["text"].lower()
        if (any(kw in tl for kw in ["s/o", "w/o", "d/o", "village", "dist", "taluk"])
                or PIN_FULLTOKEN.match(w["text"])) and (w["conf"] is not None and w["conf"] > 25):
            out.append(_mk("address", "Address", "contact", w["text"],
                            page, words_bbox(words, [i], img_w, img_h), counter.next()))
            seen.add(i)
    return out, seen


def detect_name(words, lines, page, img_w, img_h, counter):
    """
    Line-based and order-independent: captures every word on a
    name-labelled line except the label token(s) themselves, rather
    than assuming "the value follows the label". A pure LTR assumption
    breaks on RTL Urdu/Arabic lines, where Tesseract still lays words
    out left-to-right by pixel position but the value can sit on
    either side of the label depending on the printed layout.
    """
    out, seen = [], set()
    claimed_lines = set()
    for li, line in enumerate(lines):
        if li in claimed_lines:
            continue
        if not i18n_labels.contains_any_keyword(line["text"], NAME_KEYWORDS):
            pair = _extract_label_value_pair(words, line, lines[li + 1] if li + 1 < len(lines) else None)
            if not pair:
                continue
            label, value, value_idxs = pair
            if not _label_is_probable(label) or "name" not in label.lower():
                continue
            value_idxs = list(value_idxs)
        else:
            # Bound the value to the contiguous run of words next to the
            # name's own label — not every non-label word anywhere on the
            # line. A row that packs multiple fields together (e.g.
            # "Patient Name: John Smith  Patient ID: 12345") was
            # previously treated as one giant name value covering the
            # unrelated field too, since only the exact label token(s)
            # were excluded and everything else on the line was kept.
            idxs_line = line["word_idxs"]
            label_positions = [pos for pos, i in enumerate(idxs_line)
                                if i18n_labels.contains_any_keyword(words[i]["text"], NAME_KEYWORDS)]
            claimed_lines.add(li)
            if not label_positions:
                continue
            first_label_pos, last_label_pos = label_positions[0], label_positions[-1]

            after = [idxs_line[p] for p in range(last_label_pos + 1, len(idxs_line))
                     if words[idxs_line[p]]["text"].strip(" /-:|") != ""]
            after = _trim_value_at_next_label(words, after)
            if after:
                value_idxs = after
            else:
                # RTL layout — the value sits before the label instead.
                value_idxs = [idxs_line[p] for p in range(0, first_label_pos)
                               if words[idxs_line[p]]["text"].strip(" /-:|") != ""]

            if li + 1 < len(lines):
                # Value continues on the next row — either because this
                # was a label-only line (value wraps entirely), or because
                # a long name only partly fit before wrapping. Previously
                # only the label-only case pulled in the next line, so a
                # name that started next to its label but wrapped a second
                # or third token onto the following row got silently
                # truncated to just the first token(s), leaving the rest
                # of the name outside the mask box.
                nxt = lines[li + 1]
                gap = nxt["top"] - line["bottom"]
                avg_h = max(1, line["bottom"] - line["top"])
                if gap < avg_h * 1.5 and not _LABEL_LINE.match(nxt["text"]) and not i18n_labels.contains_any_keyword(nxt["text"], NAME_KEYWORDS):
                    value_idxs = value_idxs + list(nxt["word_idxs"])
                    claimed_lines.add(li + 1)

            if not value_idxs:
                continue
            value = " ".join(words[i]["text"] for i in value_idxs)

        if not value_idxs:
            continue
        out.append(_mk("person_name", "Name", "identity", value,
                        page, words_bbox(words, value_idxs, img_w, img_h), counter.next()))
        seen.update(value_idxs)
    return out, seen


def detect_generic_labels(words, lines, page, img_w, img_h, counter, already_claimed):
    """
    Catches any "Label: Value" line the specific detectors above don't
    already own — e.g. "Father's Name: ...", "Account No: ...",
    "IFSC Code: ...", "Policy No: ...", "Employee ID: ...". This is what
    makes the tool cover documents beyond the fixed Aadhaar/PAN field set.

    Requires reasonable average OCR confidence across the line. Without
    this, a bilingual document whose non-Latin script Tesseract can't
    read (e.g. Arabic text with no Arabic language pack installed)
    tends to produce low-confidence junk tokens that still happen to
    look like "Label: value" — surfacing those as fake fields is worse
    than missing a genuinely low-quality scan's real field, so this
    errs toward dropping anything OCR itself wasn't confident about.
    """
    out = []
    for li, line in enumerate(lines):
        if any(idx in already_claimed for idx in line["word_idxs"]):
            continue
        confs = [words[i]["conf"] for i in line["word_idxs"] if words[i]["conf"] is not None and words[i]["conf"] >= 0]
        if confs and (sum(confs) / len(confs)) < 45:
            continue
        pair = _extract_label_value_pair(words, line, lines[li + 1] if li + 1 < len(lines) else None)
        if not pair:
            continue
        label, value, value_idxs = pair
        if _OWNED_LABEL_PATTERNS.search(label) or len(value) < 2:
            continue
        if len(label) < 2 or label.lower() in {"note", "important", "instructions"}:
            continue
        bbox = words_bbox(words, value_idxs, img_w, img_h) if value_idxs else (
            line["left"], line["top"], line["right"], line["bottom"]
        )
        display = " ".join(w.capitalize() for w in label.split())
        out.append(_mk(
            f"label:{label.lower()}", display, "generic", value, page,
            bbox, counter.next(),
        ))
    return out


ALL_KNOWN_FIELD_TYPES = [
    "aadhaar_number", "person_name", "pan_number", "dob", "date_of_issue",
    "date_of_expiry", "address", "credit_card_number", "phone_number", "email",
]

FIELD_TYPE_LABELS = {
    "aadhaar_number": "Aadhaar Number",
    "person_name": "Name",
    "pan_number": "PAN Number",
    "dob": "Date of Birth",
    "date_of_issue": "Date of Issue",
    "date_of_expiry": "Date of Expiry",
    "address": "Address",
    "credit_card_number": "Card Number",
    "phone_number": "Phone Number",
    "email": "Email Address",
}


def run_known_detectors(words, lines, page, img_w, img_h, counter):
    """Runs every specific detector and returns (instances, claimed_word_idxs)."""
    instances = []
    claimed = set()
    for fn in (detect_aadhaar_number, detect_pan_number, detect_labelled_dates, detect_phone,
               detect_email, detect_card_number, detect_address, detect_name):
        found, seen = fn(words, lines, page, img_w, img_h, counter)
        instances += found
        claimed |= seen
    return instances, claimed
