"""
engine/tables.py
Bank statements (and any similar tabular document — invoices, ledgers)
don't have fixed "Name:"/"DOB:" style labels; they have a header row
("Date | Narration | Debit | Credit | Balance") followed by many data
rows. This module finds that header, works out each column's horizontal
band, and emits one instance per (row, column) cell so the UI can offer
"mask this whole column" (e.g. every Balance, every Narration) instead
of forcing the user to write a custom rule per row.
"""

import re
from .ocr import pad_bbox

COLUMN_KEYWORDS = {
    "date": ["date", "value date", "txn date", "transaction date"],
    "narration": ["narration", "description", "particulars", "details", "remarks"],
    "reference": ["cheque", "chq", "ref no", "reference", "utr", "instrument"],
    "debit": ["debit", "withdrawal", " dr "],
    "credit": ["credit", "deposit", " cr "],
    "balance": ["balance", "closing balance", "running balance"],
    "amount": ["amount"],
}
COLUMN_LABELS = {
    "date": "Statement Date", "narration": "Statement Narration",
    "reference": "Statement Reference No.", "debit": "Statement Debit Amount",
    "credit": "Statement Credit Amount", "balance": "Statement Balance",
    "amount": "Statement Amount",
}

ROW_DATE_START = re.compile(r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$|^\d{1,2}[\-\s][A-Za-z]{3}[\-\s]\d{2,4}$')


def _match_columns(line_text):
    """Returns the set of column keys whose keywords appear in this line."""
    tl = f" {line_text.lower()} "
    found = set()
    for col, kws in COLUMN_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            found.add(col)
    return found


def find_header_line(lines):
    best, best_score = None, 0
    for line in lines:
        cols = _match_columns(line["text"])
        if len(cols) >= 2 and len(cols) > best_score:
            best, best_score = line, len(cols)
    return best


def _column_anchor_x(words, line, col_key):
    """Left-x of the word/phrase in the header line matching this column."""
    kws = COLUMN_KEYWORDS[col_key]
    for idx in line["word_idxs"]:
        if any(kw.strip() in words[idx]["text"].lower() for kw in kws if kw.strip()):
            return words[idx]["left"]
    return line["left"]


def detect_table_columns(words, lines, page, img_w, img_h, counter):
    """
    Finds a bank-statement-style table on the page and returns
    (instances, claimed): one instance per detected cell, grouped by
    column, and the set of word indices — header row included — that
    belong to the table. Returns ([], set()) if no table header is found
    on this page (most ID-card documents won't have one, so this is a
    no-op for them).

    The header row's own words (e.g. "Date Narration Debit Credit
    Balance") were previously never marked claimed by anything, since
    only data-row cells become instances — leaving the generic "any
    Label: Value" detector free to independently (and wrongly) parse
    that whole header row as if "Date" were a label and the rest of the
    row its value. Returning it as claimed lets the pipeline exclude it
    from downstream detectors the same way any other already-detected
    text is excluded.
    """
    header = find_header_line(lines)
    if header is None:
        return [], set()

    cols_present = sorted(_match_columns(header["text"]),
                           key=lambda c: _column_anchor_x(words, header, c))
    if len(cols_present) < 2:
        return [], set()

    anchors = [(_column_anchor_x(words, header, c), c) for c in cols_present]
    anchors.sort()

    body_lines = [l for l in lines if l["top"] > header["bottom"]]
    body_lines.sort(key=lambda l: l["top"])

    claimed = set(header["word_idxs"])
    instances = []
    for line in body_lines:
        first_word_text = words[line["word_idxs"][0]]["text"]
        # Treat as a data row only if it starts with something date-shaped
        # or a currency-looking number — avoids masking footnotes/titles.
        looks_like_row = (ROW_DATE_START.match(first_word_text)
                           or re.match(r'^\d{1,3}(,\d{3})*(\.\d+)?$', first_word_text))
        if not looks_like_row:
            continue

        # Assign each word to whichever column anchor its *center* sits
        # closest to, rather than a strict left-edge band membership test.
        # Numeric columns (Debit/Credit/Balance) are conventionally
        # right-aligned, so a shorter value's left edge can land a few
        # pixels short of its own column's header — "5000.00" sitting 1px
        # left of the "Debit" header's anchor used to fall just outside
        # the debit band and get silently misfiled under "Narration"
        # instead, meaning a "mask this column" selection would miss it.
        # Nearest-anchor-by-center has no such hard edge to fall short of.
        cell_words = {col: [] for _, col in anchors}
        for idx in line["word_idxs"]:
            wx = (words[idx]["left"] + words[idx]["right"]) / 2
            _, col = min(anchors, key=lambda a: abs(a[0] - wx))
            cell_words[col].append(idx)

        for col, idxs in cell_words.items():
            if not idxs:
                continue
            claimed.update(idxs)
            value = " ".join(words[i]["text"] for i in idxs)
            left = min(words[i]["left"] for i in idxs)
            top = min(words[i]["top"] for i in idxs)
            right = max(words[i]["right"] for i in idxs)
            bottom = max(words[i]["bottom"] for i in idxs)
            bbox = pad_bbox(left, top, right, bottom, img_w, img_h)
            instances.append({
                "id": counter.next(),
                "field_type": f"table:{col}",
                "display_label": COLUMN_LABELS[col],
                "category": "table",
                "value": value,
                "page": page,
                "bbox": bbox,
            })
    return instances, claimed
