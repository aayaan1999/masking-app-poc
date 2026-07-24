"""
engine/pipeline.py
Ties the whole thing together:
  extract_fields()  PDF -> page images + every detected field instance
  group_for_ui()     instances -> UI-friendly groups (checkbox list)
  render_masked_pdf() page images + chosen instances -> masked PDF file
"""

from . import ocr, detectors, tables, ner, custom, gcc_ids
from .detectors import InstanceCounter
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

    for page_idx, image in enumerate(page_images):
        img_w, img_h = image.size
        words, lines = ocr.ocr_page(image)
        ocr_cache.append((words, lines, img_w, img_h))

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

    return page_images, all_instances, ocr_cache


def run_custom_search(pdf_words_lines, text):
    """
    pdf_words_lines: list of (words, lines, img_w, img_h) per page, as
    produced while extracting. text: free-text instruction string.
    Returns a list of new instances for whatever terms it finds.
    """
    targets = custom.extract_custom_targets(text)
    if not targets:
        return []
    counter = InstanceCounter()
    counter.n = 900000  # keep custom ids from colliding with extract-time ids
    found = []
    for page_idx, (words, lines, img_w, img_h) in enumerate(pdf_words_lines):
        for term, mode in targets:
            found += custom.find_custom_target_instances(
                words, lines, page_idx, img_w, img_h, term, mode, counter)
    return found


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
