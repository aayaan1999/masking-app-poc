"""
engine/pipeline.py
Ties the whole thing together:
  extract_fields()  PDF -> page images + every detected field instance
  group_for_ui()     instances -> UI-friendly groups (checkbox list)
  render_masked_pdf() page images + chosen instances -> masked PDF file
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import ocr, detectors, tables, ner, custom, gcc_ids
from .detectors import InstanceCounter
from .gemini_detector import detect_gemini_fields
from .masking import apply_redactions

CATEGORY_ORDER = ["identity", "contact", "financial", "table", "generic", "custom"]
CATEGORY_LABELS = {
    "identity": "Identity fields", "contact": "Contact fields",
    "financial": "Financial fields", "table": "Statement / table columns",
    "generic": "Other detected fields", "custom": "Custom matches",
}


def _bbox_overlaps(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def _union_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _overlap_ratio(a, b):
    """Intersection area as a fraction of the smaller box's area (0 if disjoint)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    if iw == 0 or ih == 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return (iw * ih) / min(area_a, area_b)


def _field_priority(inst):
    """
    Which detector's naming to trust when two overlapping instances
    disagree on what to call the same physical field. Highest wins:
      3 - a specific, precisely-classified field (every regex/GCC/table
          detector, plus Gemini's own normalized types like dob,
          date_of_issue, person_name, occupation, ...) — these already
          carry a real, human-meaningful field name.
      2 - the generic "label:<text>" detector — its name comes directly
          from whatever label is actually printed on the document, so
          it's still a genuine label, just not one the app specifically
          recognizes.
      1 - spaCy NER ("entity:person" etc.) — a free-text mention with no
          real printed label backing it, the least specific of the four
          detectors, kept only as a last resort.
    """
    ft = inst["field_type"]
    if ft.startswith("entity:"):
        return 1
    if ft.startswith("label:"):
        return 2
    return 3


def _merge_overlapping_instances(instances):
    """
    The same physical field is routinely found more than once — by the
    regex/OCR-line detector, the Gemini vision detector, the generic
    "any Label: Value" fallback, and spaCy NER — each under its own
    field_type/display_label and sometimes only a partial bbox. Grouping
    by page and field_type only merges instances that happen to already
    agree on a name (or fall under a small hand-picked equivalence map),
    which is too narrow: a field Gemini calls "occupation" and the
    generic detector independently finds as "label:occupation" describe
    the exact same printed text but never matched on field_type, so they
    used to show up as two separate, confusing checkbox entries for one
    field. Bucketing by page and merging on *physical overlap* instead
    catches this regardless of what each detector happens to call it,
    and _field_priority picks the most specific/real label as the name
    kept for the merged instance — so a field always shows up once,
    under its actual label when it has one, not once per detector.
    """
    by_page = {}
    for inst in instances:
        by_page.setdefault(inst["page"], []).append(inst)

    merged = []
    for group in by_page.values():
        used = [False] * len(group)
        for i, inst in enumerate(group):
            if used[i]:
                continue
            used[i] = True
            current = dict(inst)
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(group):
                    if used[j]:
                        continue
                    if _overlap_ratio(current["bbox"], other["bbox"]) >= 0.3:
                        current["bbox"] = _union_bbox(current["bbox"], other["bbox"])
                        if len(other["value"]) > len(current["value"]):
                            current["value"] = other["value"]
                        if _field_priority(other) > _field_priority(current):
                            current["field_type"] = other["field_type"]
                            current["display_label"] = other["display_label"]
                            current["category"] = other["category"]
                        used[j] = True
                        changed = True
            merged.append(current)
    return merged


def extract_fields(pdf_path: str, use_ner: bool = True):
    """
    Runs OCR + every detector on every page.
    Returns (page_images, instances, ocr_cache):
      instances: flat list of {id, field_type, display_label, category,
                 value, page, bbox}
      ocr_cache: list of (words, lines, img_w, img_h) per page, so a
                 later custom-text search doesn't need to re-run OCR.
    """
    page_images = ocr.pdf_to_images(pdf_path)
    counter = InstanceCounter()
    all_instances = []
    ocr_cache = []
    page_ocr = []  # (words, lines, img_w, img_h) per page, for the Gemini pass below

    for page_idx, image in enumerate(page_images):
        img_w, img_h = image.size
        words, lines = ocr.ocr_page(image)
        ocr_cache.append((words, lines, img_w, img_h))
        page_ocr.append((words, lines, img_w, img_h))

        known, claimed = detectors.run_known_detectors(words, lines, page_idx, img_w, img_h, counter)
        known = detectors.reassociate_unlabelled_dates(known, lines, words)

        gcc_instances, gcc_claimed = gcc_ids.run_gcc_detectors(
            words, lines, page_idx, img_w, img_h, counter, claimed)
        claimed |= gcc_claimed

        table_instances = tables.detect_table_columns(words, lines, page_idx, img_w, img_h, counter)

        # Any bare date match (DOB, issue, expiry, or unlabelled) is
        # ambiguous with a transaction/statement date — if it sits
        # inside a cell of a table already found on this page, the
        # table's own "Statement Date" column already represents it,
        # so drop the duplicate here rather than offering the same
        # date twice under two different group names.
        _DATE_TYPES = {"dob", "date_of_issue", "date_of_expiry", "date_unlabelled"}
        if table_instances:
            known = [inst for inst in known
                     if not (inst["field_type"] in _DATE_TYPES
                             and any(_bbox_overlaps(inst["bbox"], t["bbox"]) for t in table_instances))]

        all_instances += known
        all_instances += gcc_instances
        all_instances += table_instances

        generic = detectors.detect_generic_labels(words, lines, page_idx, img_w, img_h, counter, claimed)
        all_instances += generic

        if use_ner:
            entity_instances = ner.detect_entities(words, lines, page_idx, img_w, img_h, counter, claimed)
            all_instances += entity_instances

    # A Gemini call is dominated by network round-trip time, not CPU, so a
    # multi-page document doesn't need to pay for each page's latency one
    # after another — running them concurrently means a 3-page PDF takes
    # roughly as long as its slowest single page instead of the sum of
    # all three, which matters a lot on a host with tight request-time
    # budgets. Skipped entirely (no thread pool spun up) when no API key
    # is configured, since every call would be an instant no-op anyway.
    if os.getenv("GOOGLE_API_KEY"):
        with ThreadPoolExecutor(max_workers=min(4, len(page_images) or 1)) as pool:
            futures = [
                pool.submit(detect_gemini_fields, image, words, lines, page_idx, img_w, img_h, counter)
                for page_idx, (image, (words, lines, img_w, img_h)) in enumerate(zip(page_images, page_ocr))
            ]
            for fut in as_completed(futures):
                all_instances += fut.result()

    all_instances = _merge_overlapping_instances(all_instances)
    return page_images, all_instances, ocr_cache


def run_custom_search(pdf_words_lines, text):
    """
    pdf_words_lines: list of (words, lines, img_w, img_h) per page, as
    produced while extracting. text: free-text instruction string.
    Returns a list of new instances for whatever terms it finds.
    
    Supports:
    - Simple words: "Product" → finds and masks "Product"
    - Quoted terms: "mask this term"
    - Label:Value format: "Product Code: ABC123" → masks the value
    """
    targets = custom.extract_custom_targets(text)
    label_value_pairs = _extract_label_value_pairs(text)
    
    if not targets and not label_value_pairs:
        return []
    
    counter = InstanceCounter()
    counter.n = 900000  # keep custom ids from colliding with extract-time ids
    found = []
    
    for page_idx, (words, lines, img_w, img_h) in enumerate(pdf_words_lines):
        # Handle simple word/phrase targets
        for term, mode in targets:
            found += custom.find_custom_target_instances(
                words, lines, page_idx, img_w, img_h, term, mode, counter)
        
        # Handle label:value patterns
        for label, value in label_value_pairs:
            found += custom.find_custom_label_value_instances(
                words, lines, page_idx, img_w, img_h, label, value, counter)
    
    return found


def _extract_label_value_pairs(text: str):
    """
    Extracts label:value or label = value patterns from instructions.
    Returns list of (label, value) tuples.
    """
    pairs = []
    # Match patterns like "Product Code: ABC123" or "Product = XYZ"
    for m in re.finditer(r'([A-Za-z][A-Za-z\s]+?)\s*[:=]\s*([A-Za-z0-9\-\s]+?)(?:[,;\n]|$)', text):
        label = m.group(1).strip()
        value = m.group(2).strip()
        if label and value and len(label) > 1 and len(value) > 1:
            pairs.append((label, value))
    return pairs


def group_for_ui(instances):
    """
    Groups instances by (category, field_type, display_label) so the UI
    shows one checkbox per distinct field kind, e.g. "Phone Number (2
    found)" rather than one row per individual match.
    """
    groups = {}
    for inst in instances:
        key = (inst["category"], inst["field_type"], inst["display_label"])
        if key not in groups:
            groups[key] = {
                "group_id": f"{inst['category']}::{inst['field_type']}::{inst['display_label']}",
                "category": inst["category"],
                "category_label": CATEGORY_LABELS.get(inst["category"], inst["category"].title()),
                "field_type": inst["field_type"],
                "display_label": inst["display_label"],
                "count": 0,
                "sample_values": [],
                "instance_ids": [],
            }
        g = groups[key]
        g["count"] += 1
        g["instance_ids"].append(inst["id"])
        if len(g["sample_values"]) < 3:
            g["sample_values"].append(inst["value"])

    grouped = list(groups.values())
    grouped.sort(key=lambda g: (CATEGORY_ORDER.index(g["category"])
                                 if g["category"] in CATEGORY_ORDER else 99,
                                 -g["count"]))
    return grouped


def render_masked_pdf(page_images, instances, output_path):
    by_page = {}
    for inst in instances:
        by_page.setdefault(inst["page"], []).append(inst)

    masked_pages = []
    for idx, image in enumerate(page_images):
        page_instances = by_page.get(idx, [])
        masked_pages.append(apply_redactions(image, page_instances) if page_instances else image.copy())

    first = masked_pages[0].convert("RGB")
    rest = [p.convert("RGB") for p in masked_pages[1:]]
    # Without an explicit resolution, Pillow assumes 72 DPI when writing
    # the PDF page size — since our page images are rendered at
    # ocr.DPI (300), that silently produced PDF pages ~4x too large
    # physically (a 300dpi 2481x3508px page written out as a
    # 2481x3508-*point* page). Passing the real resolution keeps the
    # output page the same physical size as the source document.
    first.save(output_path, save_all=True, append_images=rest, resolution=ocr.DPI)
