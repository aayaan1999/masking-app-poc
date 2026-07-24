"""
engine/ocr.py
Converts a PDF into page images and runs Tesseract OCR, normalizing the
raw parallel-array output into easy-to-work-with `Word` / `Line` objects
that every detector in this package builds on.

Runs multilingual: English + Urdu + Arabic by default, so GCC identity
documents (which are commonly bilingual, and whose expat-labor
paperwork often carries Urdu alongside English/Arabic) are actually
legible to OCR instead of coming back blank/garbled for the non-Latin
script. Requires the corresponding Tesseract language packs — see
Dockerfile (`tesseract-ocr-urd`, `tesseract-ocr-ara`). If they aren't
installed, this falls back to English-only automatically rather than
failing the whole request.
"""

import pytesseract
from pdf2image import convert_from_path

DPI = 300
PREFERRED_LANGS = ["eng", "urd", "ara"]

_active_lang_string = None

# Arabic-Indic (٠-٩) and Extended Arabic-Indic/Persian (۰-۹) digits, mapped
# to plain Western 0-9. When Tesseract's Arabic model is active it can
# render printed Western-numeral dates/IDs using either digit form —
# every numeric regex in this app (dates, Emirates ID, phone numbers...)
# only matches ASCII 0-9, so without this normalization those fields
# would silently fail to be recognized at all despite OCR reading the
# characters correctly.
_DIGIT_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _normalize_digits(text: str) -> str:
    return text.translate(_DIGIT_TRANS)


def _installed_langs():
    try:
        return set(pytesseract.get_languages(config=""))
    except Exception:
        return {"eng"}


def active_ocr_langs():
    """
    Returns the '+'-joined language string Tesseract will actually use,
    computed once and cached — e.g. 'eng+urd+ara', or just 'eng' if the
    extra language packs aren't installed in this environment.
    """
    global _active_lang_string
    if _active_lang_string is None:
        installed = _installed_langs()
        usable = [l for l in PREFERRED_LANGS if l in installed] or ["eng"]
        _active_lang_string = "+".join(usable)
    return _active_lang_string


def pdf_to_images(pdf_path: str, dpi: int = DPI):
    """Returns a list of PIL Images, one per page."""
    return convert_from_path(pdf_path, dpi=dpi)


def _cluster_into_lines(words):
    """
    Groups word indices into visual rows by y-position, rather than
    trusting Tesseract's own block/par/line numbering. In sparse-text
    mode (psm 11, used here because ID-card layouts scatter text in
    boxes) Tesseract frequently assigns words that are visibly on the
    same printed row to different "line" numbers, which silently broke
    every row-aware feature (table headers, same-line name/value
    pairs, and RTL label/value pairing). Clustering by vertical center
    is layout-mode- and reading-direction-independent.
    """
    if not words:
        return []
    order = sorted(range(len(words)), key=lambda i: (words[i]["top"], words[i]["left"]))
    clusters, current = [], [order[0]]
    for i in order[1:]:
        avg_top = sum(words[j]["top"] for j in current) / len(current)
        avg_h = sum(words[j]["height"] for j in current) / len(current)
        if abs(words[i]["top"] - avg_top) <= max(8, avg_h * 0.6):
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)
    return clusters


def ocr_page(image):
    """
    Runs Tesseract on a single page image and returns:
      words: list of dicts {text,left,top,width,height,right,bottom,conf,line_key}
      lines: list of dicts {key,text,left,top,right,bottom,word_idxs}
    line_key groups words by visual row (see _cluster_into_lines).

    Note on line `text`: words are joined left-to-right by pixel
    position for *display* purposes only. For a right-to-left Urdu/
    Arabic line this is not correct reading order — detectors that need
    to know "does this line mention concept X" or "mask everything on
    this line except the label" don't care about reading order and are
    unaffected; only human-facing preview text may read reversed.
    """
    raw = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT,
        lang=active_ocr_langs(), config="--psm 6 --oem 3",
    )
    words = []
    n = len(raw["text"])
    for i in range(n):
        text = raw["text"][i]
        if not text or not text.strip():
            continue
        text = _normalize_digits(text)
        left, top = raw["left"][i], raw["top"][i]
        width, height = raw["width"][i], raw["height"][i]
        words.append({
            "text": text,
            "left": left, "top": top, "width": width, "height": height,
            "right": left + width, "bottom": top + height,
            "conf": int(float(raw["conf"][i])) if raw["conf"][i] not in ("-1", -1) else -1,
        })

    clusters = _cluster_into_lines(words)
    for line_idx, idxs in enumerate(clusters):
        for i in idxs:
            words[i]["line_key"] = line_idx

    lines = []
    for line_idx, idxs in enumerate(clusters):
        idxs = sorted(idxs, key=lambda i: words[i]["left"])
        text = " ".join(words[i]["text"] for i in idxs)
        left = min(words[i]["left"] for i in idxs)
        top = min(words[i]["top"] for i in idxs)
        right = max(words[i]["right"] for i in idxs)
        bottom = max(words[i]["bottom"] for i in idxs)
        lines.append({
            "key": line_idx, "text": text, "left": left, "top": top,
            "right": right, "bottom": bottom, "word_idxs": idxs,
        })
    lines.sort(key=lambda l: (l["top"], l["left"]))

    return words, lines


def pad_bbox(left, top, right, bottom, img_w, img_h, pad=6):
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(img_w, right + pad),
        min(img_h, bottom + pad),
    )


def words_bbox(words, idxs, img_w, img_h, pad=6):
    left = min(words[i]["left"] for i in idxs)
    top = min(words[i]["top"] for i in idxs)
    right = max(words[i]["right"] for i in idxs)
    bottom = max(words[i]["bottom"] for i in idxs)
    return pad_bbox(left, top, right, bottom, img_w, img_h, pad)
