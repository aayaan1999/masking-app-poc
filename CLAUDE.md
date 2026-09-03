# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Flask app that takes an uploaded PDF (IDs, bank statements, forms — heavy focus on GCC/expat identity
documents), OCRs it, auto-detects PII fields, lets the user pick which fields (or free-text terms) to
redact, and returns a masked PDF with solid black boxes over the chosen regions. No database — job state
lives on disk under `jobs/<job_id>/` for the life of a session.

## Running locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional, enables NER-based detection
python app.py                              # dev server on :8080 (PORT env var to override)
```

Requires **Tesseract OCR** (with `urd` and `ara` language packs — see `Dockerfile`) and **poppler**
(`pdf2image` shells out to `pdftoppm`) installed on the host/PATH. Without the extra Tesseract language
packs, OCR silently falls back to English-only (`engine/ocr.py: active_ocr_langs`).

Production runs via gunicorn: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 180`.

Optional: set `GOOGLE_API_KEY` to enable the Gemini-based field detector (`engine/gemini_detector.py`);
without it, `detect_gemini_fields` is a no-op.

## Tests

```bash
python -m pytest tests/
python -m pytest tests/test_detection.py::test_generic_label_with_whitespace_separator_is_detected
```

Tests construct synthetic OCR `words`/`lines` dicts directly (see helpers at the top of each test file)
rather than running real Tesseract — this is the pattern to follow for new detector tests.

## Request flow (two-phase, poll-based)

1. `POST /extract` — saves the upload, kicks off `_run_extraction_job` on a **background thread**, and
   returns a `job_id` immediately (202). OCR over a multi-page multilingual PDF at 300 DPI can run past
   what a reverse proxy holds a request open for, so this never blocks the HTTP response — see the
   docstring on `_run_extraction_job` in `app.py`.
2. Client polls `GET /extract/status/<job_id>` until `status` is `done` or `error`. Always returns HTTP
   200; state is carried in the JSON `status` field, not the status code, so a slow-but-alive job is
   never mistaken for a transport failure.
3. `POST /mask` — takes the `job_id`, a list of selected `group_ids`, and optional free-text
   `instructions`, re-renders the previously extracted page images with redactions burned in, and streams
   the resulting PDF back. Job data is deleted from disk right after (`jobs.cleanup_job`) whether masking
   succeeds or fails.

All per-job state (page images as PNGs, OCR cache, detected instances) is persisted to
`jobs/<job_id>/` between the two requests by `engine/jobs.py`, and swept by
`jobs.cleanup_stale_jobs` (30 min TTL) at the start of every `/extract` call. The original uploaded PDF is
deleted the moment OCR finishes, successful or not — it's never kept.

## Detection pipeline (`engine/pipeline.py: extract_fields`)

For each page: OCR runs once (`engine/ocr.py`), and every detector below runs over the same
`words`/`lines` output, each returning a list of `instance` dicts:
`{id, field_type, display_label, category, value, page, bbox}`.

Detectors run in this order, with later ones aware of what earlier ones already claimed (a `claimed` set
of word indices, threaded through to avoid double-detecting the same text under two labels):

1. `detectors.run_known_detectors` — Aadhaar, PAN, phone, email, card numbers, PIN, IFSC, account number,
   DOB, address, name, plus a generic "any `Label: Value`" fallback for fields with no dedicated detector.
2. `gcc_ids.run_gcc_detectors` — GCC passport/national-ID numbers (UAE Emirates ID, Saudi, Qatar, Kuwait,
   Bahrain, Oman). Emirates ID matches on pattern alone (`784-...` is distinctive); every other GCC ID
   pattern is just N digits, so it only counts as a match when a recognized passport/national-ID label
   (English/Urdu/Arabic, via `i18n_labels`) appears on the same OCR'd line.
3. `tables.detect_table_columns` — finds header rows on bank-statement-style tabular pages (Date /
   Narration / Debit / Credit / Balance) and emits one instance per (row, column) cell, so the UI can
   offer "mask this whole column."
4. `gemini_detector.detect_gemini_fields` — optional, calls the Gemini API (needs `GOOGLE_API_KEY`) with
   the page image for AI-based extraction of fields the regex detectors miss.
5. `detectors.detect_generic_labels` — catch-all `Label: Value` pattern for anything not already claimed.
6. `ner.detect_entities` — optional (spaCy `en_core_web_sm`), catches free-text PERSON/ORG/GPE/MONEY
   mentions (e.g. a name mid-sentence). Degrades silently to `[]` if spaCy isn't installed.

Bare-date instances (`dob`, `date_of_issue`, `date_of_expiry`, unlabelled) that overlap a table cell on
the same page are dropped, since the table's own date column already represents them.

`group_for_ui` collapses instances into one checkbox group per
`(category, field_type, display_label)` for the frontend. `run_custom_search` (triggered by free-text
`instructions` on `/mask`) re-parses the cached OCR data (no re-OCR) to find arbitrary
terms/names/`label:value` patterns the fixed detectors don't cover — see `engine/custom.py`.

## Multilingual / GCC-specific handling

This app is built around bilingual GCC identity documents (English + Urdu + Arabic), not just plain
English forms:

- `engine/ocr.py` runs Tesseract with `eng+urd+ara` when those language packs are installed, and
  normalizes Arabic-Indic/Persian digit forms to ASCII (every numeric regex in the app only matches
  ASCII `0-9`).
- `engine/i18n_labels.py` is the single source of truth mapping a *concept* (e.g. "date of birth") to its
  recognized keyword in every supported language/script, including Arabic alef-letterform and diacritic
  normalization. Add new label keywords here, not in individual detectors.
- Label/value matching never assumes left-to-right reading order — it only checks "the value is
  somewhere else on this same OCR'd line" (lines are clustered by y-position in `ocr._cluster_into_lines`,
  not by Tesseract's own line numbering, which is unreliable in the sparse-layout PSM mode used for ID
  cards).

## Rendering the masked PDF

`masking.apply_redactions` draws solid filled rectangles (not overlays) directly onto the page image —
the underlying pixels are destroyed, making this a real redaction. `pipeline.render_masked_pdf` must pass
`resolution=ocr.DPI` when saving via Pillow, or the output PDF page size is wrong (Pillow assumes 72 DPI
otherwise, producing pages ~4x too large since images are rendered at 300 DPI).
