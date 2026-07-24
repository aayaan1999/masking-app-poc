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
        if i in seen or w["conf"] < 40:
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
        if PAN_PATTERN.search(w["text"]) and w["conf"] > 10:
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


def detect_labelled_dates(words, lines, page, img_w, img_h, counter):
    """
    Finds date-shaped tokens and classifies each one by whichever
    date-related label (DOB / issue / expiry) appears on the *same
    line* — checked as "does this concept's keyword appear anywhere in
    the line", not "does it come right before the date". That
    direction-agnostic check is what makes this correct for RTL Urdu/
    Arabic lines, where the value can sit to either side of its label
    depending on layout, not just after it in pixel order.

    This replaces a blanket "any date-shaped text = Date of Birth"
    detector, which used to also catch expiry/issue/validity dates and
    mask them whenever DOB was selected.
    """
    out, seen = [], set()
    for line in lines:
        concepts_here = [c for c in ("dob", "date_of_issue", "date_of_expiry")
                          if i18n_labels.line_matches_concept(line["text"], c)]
        date_idxs = [i for i in line["word_idxs"] if DATE_PATTERN.search(words[i]["text"])
                     and words[i]["conf"] > 20]
        if not date_idxs:
            continue

        if concepts_here:
            # A line can legitimately name more than one date concept
            # (rare, but "Issued/Expiry: dd/mm - dd/mm" happens) — in
            # that case every date on the line gets tagged under every
            # concept found, since we can't reliably tell which date is
            # which without deeper layout parsing, and over-offering a
            # checkbox is far safer than silently under-masking one.
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
        if PHONE_PATTERN.search(w["text"]) and w["conf"] > 35:
            out.append(_mk("phone_number", "Phone Number", "contact", w["text"],
                            page, words_bbox(words, [i], img_w, img_h), counter.next()))
            seen.add(i)
    return out, seen


def detect_email(words, lines, page, img_w, img_h, counter):
    out, seen = [], set()
    for i, w in enumerate(words):
        if EMAIL_PATTERN.search(w["text"]) and w["conf"] > 35:
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
        if CARD_FULL.match(w["text"]) and w["conf"] > 15 and not (
                len(w["text"]) == 15 and w["text"].startswith("784")):
            out.append(_mk("credit_card_number", "Card Number", "financial", w["text"],
                            page, words_bbox(words, [i], img_w, img_h), counter.next()))
            seen.add(i)
            continue
        if CARD_GROUP_4.match(w["text"]) and w["conf"] > 25:
            group = [i]
            j = i + 1
            while (j < n and len(group) < 4 and CARD_GROUP_4.match(words[j]["text"])
                   and words[j]["conf"] > 25 and words[j]["line_key"] == w["line_key"]):
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
        idxs = list(line["word_idxs"])
        claimed_lines.add(li)

        if li + 1 < len(lines):
            nxt = lines[li + 1]
            gap = nxt["top"] - line["bottom"]
            avg_h = max(1, line["bottom"] - line["top"])
            if gap < avg_h * 1.5 and not _LABEL_LINE.match(nxt["text"]):
                idxs += list(nxt["word_idxs"])
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
                or PIN_FULLTOKEN.match(w["text"])) and w["conf"] > 25:
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
        if not i18n_labels.contains_any_keyword(line["text"], NAME_KEYWORDS) or li in claimed_lines:
            continue
        label_idxs = {i for i in line["word_idxs"]
                      if i18n_labels.contains_any_keyword(words[i]["text"], NAME_KEYWORDS)}
        value_idxs = [i for i in line["word_idxs"]
                      if i not in label_idxs and words[i]["text"].strip(" /-:|") != ""]
        claimed_lines.add(li)

        if not value_idxs and li + 1 < len(lines):
            # Label-only line (value wraps to the next row) — same
            # continuation guard used by detect_address.
            nxt = lines[li + 1]
            gap = nxt["top"] - line["bottom"]
            avg_h = max(1, line["bottom"] - line["top"])
            if gap < avg_h * 1.5 and not _LABEL_LINE.match(nxt["text"]):
                value_idxs = list(nxt["word_idxs"])
                claimed_lines.add(li + 1)

        if not value_idxs:
            continue
        val = " ".join(words[i]["text"] for i in value_idxs)
        out.append(_mk("person_name", "Name", "identity", val,
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
    for line in lines:
        if any(idx in already_claimed for idx in line["word_idxs"]):
            continue
        confs = [words[i]["conf"] for i in line["word_idxs"] if words[i]["conf"] >= 0]
        if confs and (sum(confs) / len(confs)) < 45:
            continue
        m = _LABEL_LINE.match(line["text"])
        if not m:
            continue
        label, value = m.group(1).strip(), m.group(2).strip()
        if _OWNED_LABEL_PATTERNS.search(label) or len(value) < 2:
            continue
        if len(label) < 2 or label.lower() in {"note", "important", "instructions"}:
            continue
        display = " ".join(w.capitalize() for w in label.split())
        out.append(_mk(
            f"label:{label.lower()}", display, "generic", value, page,
            (line["left"], line["top"], line["right"], line["bottom"]),
            counter.next(),
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
