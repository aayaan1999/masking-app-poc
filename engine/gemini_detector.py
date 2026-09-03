import base64
import json
import os
import re
import time
import traceback
import urllib.request
import urllib.error
from io import BytesIO

from .ocr import words_bbox
from .detectors import DATE_CONCEPT_LABELS

# Verified live against the project's actual API key (see chat/session
# notes) as of 2026-09: "gemini-3.6-flash" is the model Google's own API
# error message names as the current replacement for retired flash models,
# and responds reliably. "gemini-flash-latest" is Google's rolling alias
# for whatever the current flash model is, kept as the fallback so a
# future model retirement/rename doesn't require another manual fix here —
# but it was seen intermittently 503-ing (capacity, not an error in this
# code) so it isn't the primary. Both overridable via env var.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-latest")

SYSTEM_PROMPT = r"""
You are an expert document data extraction system. Scan and process every page of the provided document thoroughly from start to finish. Extract all visible identification, demographic, and financial fields and visible values. Do not omit any clearly visible data fields.

Return your response strictly as JSON. Do not include markdown fences, explanatory text, or any wrapper text.

If you return a single object, wrap it as an array under a key such as "fields" or "data". The final result must be a JSON array of objects.

Each object must contain these exact keys:
- "field_type": String. Use one of: "email", "phone", "person_name", "date_of_birth", "issue_date", "expiry_date", "date", "account_number", "routing_number", "id_number", "nationality", "sex", "occupation", "organization", "location", "issuing_authority", "place_of_issue".
- "value": String. The exact visible text for the field.
- "bbox": Array of four integers [x, y, width, height] in pixel coordinates relative to the page image.
- "label": String or null. The nearest contextual label or header text.
- "page_number": Integer. The 1-indexed page number where the field appears.

A document commonly has more than one date on it — do not lump them
together. Classify every date by what it actually represents:
- "date_of_birth" for a birth date (DOB).
- "issue_date" for when the document/card was issued.
- "expiry_date" for when it expires/is valid until.
- "date" ONLY for a date that is none of the above (e.g. a transaction
  or statement date) — never use "date" for a birth, issue, or expiry
  date just because its label is ambiguous; use the surrounding context
  (nearby labels, document type) to decide which of the three it is.

For GCC identity documents, include fields such as ID Number, Name, Date of Birth, Nationality, Sex, Occupation, Employer, Place of Issue, Issuing Authority, Issue Date, and Expiry Date when they are visible.

If a field is not clearly visible or present, skip it. If no fields are found, return an empty array: []
"""

FIELD_LABELS = {
    # Existing fields
    "email": "Email Address",
    "phone": "Phone Number",
    "person_name": "Name",
    "date": "Date",
    "account_number": "Account Number",
    "routing_number": "Routing Number",
    "id_number": "ID Number",
    # New GCC Document fields
    "nationality": "Nationality",
    "sex": "Gender",
    "occupation": "Occupation / Job Title",
    "organization": "Sponsor / Employer / Company",
    "location": "Place of Issue / Address / City",
    "issuing_authority": "Issuing Authority",
    "place_of_issue": "Place of Issue",
    # Reuse the regex detector's own field_type names/labels for
    # dob/date_of_issue/date_of_expiry (see _normalize_field_type) so a
    # date found by both detectors is recognized as the same field
    # instead of showing up as two separately-labeled groups.
    **DATE_CONCEPT_LABELS,
}

FIELD_CATEGORIES = {
    # Existing categories
    "email": "contact",
    "phone": "contact",
    "person_name": "identity",
    "date": "identity",
    "account_number": "financial",
    "routing_number": "financial",
    "id_number": "identity",
    # New GCC Document categories
    "nationality": "identity",
    "sex": "identity",
    "occupation": "employment",
    "organization": "employment",
    "location": "geographic",
    "issuing_authority": "employment",
    "place_of_issue": "geographic",
    "dob": "identity",
    "date_of_issue": "identity",
    "date_of_expiry": "identity",
}



def _call_gemini_once(model, api_key, mime_type, data, prompt):
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": data,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
        return body.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")


def _call_gemini(image_or_bytes, prompt):
    """
    Calls Gemini with either a page image or raw PDF bytes. Every failure
    used to be swallowed into a silent "" return, which made a bad model
    name / expired key / quota error indistinguishable from "Gemini just
    found nothing" — printing here is what lets that actually surface in
    server logs instead of masquerading as a detection gap.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return ""

    try:
        if hasattr(image_or_bytes, "save"):
            buffer = BytesIO()
            image_or_bytes.save(buffer, format="PNG")
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            mime_type = "image/png"
        else:
            data = base64.b64encode(image_or_bytes).decode("ascii")
            mime_type = "application/pdf"
    except Exception:
        traceback.print_exc()
        return ""

    _TRANSIENT_CODES = {429, 500, 502, 503, 504}

    for model in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL):
        attempts = 2 if model == GEMINI_MODEL else 1
        move_to_fallback = False
        for attempt in range(attempts):
            try:
                return _call_gemini_once(model, api_key, mime_type, data, prompt)
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                print(f"[gemini_detector] {model} request failed: HTTP {exc.code} {body}")
                transient = exc.code in _TRANSIENT_CODES
                retriable = exc.code == 404 or transient
            except (urllib.error.URLError, ValueError, json.JSONDecodeError, OSError) as exc:
                # A cold/slow model can simply time out (OSError covers
                # socket.timeout) rather than return an HTTP error code —
                # that deserves the same retry-then-fallback treatment as
                # a 503, not an immediate give-up.
                print(f"[gemini_detector] {model} request failed: {exc!r}")
                transient, retriable = True, True

            if transient and attempt + 1 < attempts:
                time.sleep(1.5)
                continue  # model is momentarily overloaded/rate-limited/slow — brief retry
            if retriable and model != GEMINI_FALLBACK_MODEL:
                move_to_fallback = True
                break  # model name is wrong/retired, or still unavailable — try the fallback model
            return ""
        if not move_to_fallback:
            return ""
    return ""


def _call_gemini_pdf_bytes(pdf_bytes, prompt):
    """Backward-compatible wrapper for PDF byte input."""
    return _call_gemini(pdf_bytes, prompt)


def _parse_json(text):
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict):
        for key in ("fields", "data", "results", "items", "output"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return candidate
        # If the object itself looks like one field, return it as a single-item list.
        if "field_type" in payload and "value" in payload:
            return [payload]
        return []
    if isinstance(payload, list):
        return payload
    return []


def _normalize_field_type(field_type):
    normalized = str(field_type or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "name": "person_name",
        "full_name": "person_name",
        "customer_name": "person_name",
        # Match the regex detector's field_type names (dob /
        # date_of_issue / date_of_expiry — see detectors.py
        # DATE_CONCEPT_LABELS) rather than Gemini's own vocabulary, so a
        # date found by both detectors merges into one group instead of
        # showing up twice under different names.
        "dob": "dob",
        "date_of_birth": "dob",
        "birth_date": "dob",
        "expiry_date": "date_of_expiry",
        "expiration_date": "date_of_expiry",
        "date_of_expiry": "date_of_expiry",
        "issue_date": "date_of_issue",
        "issued_at": "date_of_issue",
        "date_of_issue": "date_of_issue",
        "email_address": "email",
        "phone_number": "phone",
        "mobile_number": "phone",
        "account": "account_number",
        "account_no": "account_number",
        "routing": "routing_number",
        "routing_no": "routing_number",
        "id": "id_number",
        "passport_number": "id_number",
        "national_id": "id_number",
        "civil_id": "id_number",
        "emirates_id": "id_number",
        "gender": "sex",
        "job": "occupation",
        "employer": "organization",
        "company": "organization",
        "authority": "organization",
        "place_of_issue": "place_of_issue",
        "issued_by": "issuing_authority",
        "issuing_authority": "issuing_authority",
        "address": "location",
        "city": "location",
    }
    return aliases.get(normalized, normalized)


def _coerce_bbox(raw_bbox, img_w, img_h):
    if isinstance(raw_bbox, dict):
        x = raw_bbox.get("x", raw_bbox.get("left", 0))
        y = raw_bbox.get("y", raw_bbox.get("top", 0))
        width = raw_bbox.get("width", raw_bbox.get("w", 0))
        height = raw_bbox.get("height", raw_bbox.get("h", 0))
        raw_bbox = [x, y, width, height]

    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return (0, 0, img_w, img_h)

    values = [float(v) for v in raw_bbox]
    if any(v > max(img_w, img_h) for v in values):
        ymin, xmin, ymax, xmax = values
        px_x = (xmin / 1000.0) * img_w
        px_y = (ymin / 1000.0) * img_h
        px_w = ((xmax - xmin) / 1000.0) * img_w
        px_h = ((ymax - ymin) / 1000.0) * img_h
        left, top = int(max(0, px_x)), int(max(0, px_y))
        right, bottom = int(max(left, px_x + px_w)), int(max(top, px_y + px_h))
        return (left, top, right, bottom)

    x, y, width, height = values
    left, top = int(max(0, x)), int(max(0, y))
    right = int(max(left, x + width))
    bottom = int(max(top, y + height))
    return (left, top, right, bottom)


def _bbox_from_words(words, value, near_cx, near_cy, img_w, img_h):
    """
    Fallback box for when Gemini didn't give us a usable bbox: finds the
    OCR words nearest to Gemini's approximate location whose text plausibly
    matches the value, rather than scanning the whole page. Whole-page
    substring matching used to pick up any word anywhere that happened to
    share a substring with the (space-stripped) value — e.g. a stray "id"
    matching into "id number" fields elsewhere on the page — which produced
    boxes that didn't actually cover the real text. Restricting the
    candidate pool to words near Gemini's own coordinates keeps a
    same-token collision elsewhere on the page from hijacking the box.
    """
    if not words:
        return (0, 0, img_w, img_h)

    value_tokens = [re.sub(r"\W+", "", t.lower()) for t in (value or "").split()]
    value_tokens = [t for t in value_tokens if t]
    max_dist = max(img_w, img_h) * 0.12  # stay local to Gemini's reported spot

    candidates = []
    for i, w in enumerate(words):
        wx = (w["left"] + w["right"]) / 2
        wy = (w["top"] + w["bottom"]) / 2
        dist = abs(wx - near_cx) + abs(wy - near_cy)
        if dist <= max_dist:
            candidates.append((dist, i))
    candidates.sort()

    idxs = []
    if value_tokens:
        remaining = list(value_tokens)
        for _, i in candidates:
            token = re.sub(r"\W+", "", words[i]["text"].lower())
            if not token:
                continue
            for j, vt in enumerate(remaining):
                if token == vt or (len(token) > 2 and (token in vt or vt in token)):
                    idxs.append(i)
                    remaining.pop(j)
                    break
            if not remaining:
                break

    if not idxs:
        idxs = [candidates[0][1]] if candidates else [
            min(range(len(words)), key=lambda i: abs(((words[i]["left"] + words[i]["right"]) / 2) - near_cx)
                + abs(((words[i]["top"] + words[i]["bottom"]) / 2) - near_cy))
        ]
    return words_bbox(words, idxs, img_w, img_h, pad=6)


def detect_gemini_fields(image, words, lines, page, img_w, img_h, counter):
    ocr_context = "\n".join(line.get("text", "").strip() for line in lines if line.get("text", "").strip())
    prompt = SYSTEM_PROMPT
    if ocr_context:
        prompt = f"{prompt}\n\nUse this OCR text from the page exactly as additional context:\n{ocr_context[:6000]}"

    text = _call_gemini(image, prompt)
    if not text:
        return []
    try:
        items = _parse_json(text)
    except Exception:
        return []

    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
            
        field_type = _normalize_field_type(item.get("field_type") or item.get("type") or item.get("label") or "")
        value = str(item.get("value") or item.get("text") or item.get("content") or "").strip()
        if not value:
            continue
            
        bbox = item.get("bbox") or item.get("bounding_box") or item.get("box") or [0, 0, 0, 0]
        bbox = _coerce_bbox(bbox, img_w, img_h)
        left, top, right, bottom = bbox

        # Gemini looked at the actual pixels, so its own bbox is more
        # trustworthy than reconstructing one from Tesseract's OCR text —
        # especially for names/values Tesseract misread. Only fall back to
        # word-matching (constrained to near Gemini's reported location)
        # when Gemini didn't give us a usable box at all.
        if right <= left or bottom <= top or (right - left) * (bottom - top) < 4:
            near_cx = left if right > left else left
            near_cy = top if bottom > top else top
            left, top, right, bottom = _bbox_from_words(words, value, near_cx, near_cy, img_w, img_h)
        else:
            pad = 4
            left = max(0, left - pad)
            top = max(0, top - pad)
            right = min(img_w, right + pad)
            bottom = min(img_h, bottom + pad)

        label = str(item.get("label") or FIELD_LABELS.get(field_type, field_type.replace("_", " ").title()))
        category = FIELD_CATEGORIES.get(field_type, "generic")
        
        out.append({
            "id": counter.next(),
            "field_type": field_type,
            "display_label": label,
            "category": category,
            "value": value,
            "page": page,
            "bbox": (left, top, right, bottom)
        })
        
    return out