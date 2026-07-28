import base64
import json
import os
import re
import urllib.request
import urllib.error
from io import BytesIO

from .ocr import words_bbox

SYSTEM_PROMPT = r"""
You are an expert document data extraction system. Scan and process every page of the provided document thoroughly from start to finish. Extract all visible identification, demographic, and financial fields and visible values. Do not omit any clearly visible data fields.

Return your response strictly as JSON. Do not include markdown fences, explanatory text, or any wrapper text.

If you return a single object, wrap it as an array under a key such as "fields" or "data". The final result must be a JSON array of objects.

Each object must contain these exact keys:
- "field_type": String. Use one of: "email", "phone", "person_name", "date", "account_number", "routing_number", "id_number", "nationality", "sex", "occupation", "organization", "location", "issue_date", "expiry_date", "issuing_authority", "place_of_issue".
- "value": String. The exact visible text for the field.
- "bbox": Array of four integers [x, y, width, height] in pixel coordinates relative to the page image.
- "label": String or null. The nearest contextual label or header text.
- "page_number": Integer. The 1-indexed page number where the field appears.

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
    "issue_date": "Issue Date",
    "expiry_date": "Expiry Date",
    "issuing_authority": "Issuing Authority",
    "place_of_issue": "Place of Issue",
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
    "issue_date": "identity",
    "expiry_date": "identity",
    "issuing_authority": "employment",
    "place_of_issue": "geographic",
}



def _call_gemini(image_or_bytes, prompt):
    """Calls Gemini with either a page image or raw PDF bytes."""
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

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
            return body.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
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
        "dob": "date",
        "date_of_birth": "date",
        "birth_date": "date",
        "expiry_date": "expiry_date",
        "issue_date": "issue_date",
        "issued_at": "issue_date",
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


def _bbox_from_words(words, value, bbox, img_w, img_h):
    if not words:
        return (0, 0, img_w, img_h)
    target = re.sub(r"\W+", "", (value or "").lower())
    idxs = []
    for i, w in enumerate(words):
        token = re.sub(r"\W+", "", w["text"].lower())
        if target and (target in token or token in target):
            idxs.append(i)
    if not idxs:
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        idxs = [min(range(len(words)), key=lambda i: abs(((words[i]["left"] + words[i]["right"]) / 2) - cx) + abs(((words[i]["top"] + words[i]["bottom"]) / 2) - cy))]
    return words_bbox(words, idxs, img_w, img_h, pad=0)


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

        if right <= left or bottom <= top:
            left, top, right, bottom = _bbox_from_words(words, value, (left, top, img_w, img_h), img_w, img_h)
        else:
            left, top, right, bottom = _bbox_from_words(words, value, (left, top, right - left, bottom - top), img_w, img_h)

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