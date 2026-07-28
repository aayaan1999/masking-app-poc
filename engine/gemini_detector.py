import base64
import json
import os
import re
import urllib.request
import urllib.error
from io import BytesIO

from .ocr import words_bbox

SYSTEM_PROMPT = r"""
You are an expert document data extraction system. Scan and process every page of the provided document thoroughly from start to finish. Extract all visible identification, demographic, and financial fields. Do not omit any data fields regardless of document length.

Return your response strictly as a JSON array of objects. Do not include markdown code block formatting (such as ```json), introductory text, or explanatory text. 

Each object in the array must contain these exact keys:
- "field_type": String. Must strictly be one of: "email", "phone", "person_name", "date", "account_number", "routing_number", "id_number", "nationality", "sex", "occupation", "organization", "location".
- "value": String. The exact text extracted from the document.
- "bbox": Array of four integers [x, y, width, height] representing the bounding box in pixel coordinates.
- "label": String. The contextual text label or header near the field (e.g., "ID Number / رقم الهوية", "Occupation:"). If no explicit label exists, use null.
- "page_number": Integer. The 1-indexed page number where the field appears.

Validation Rules:
1. "field_type" categorization mapping:
   - Use "id_number" for identity card numbers, document numbers, or card serial numbers.
   - Use "organization" for employer names, companies, or authorities.
   - Use "location" for addresses, cities of issuance, or places of birth.
   - Use "sex" for gender indicators (e.g., M, F, Male, Female).
2. If a field spans multiple lines, treat it as a single extraction with a bounding box that encompasses the entire text string.
3. If no valid fields are found, return an empty array: []
"""

FIELD_LABELS = {
    "email": "Email Address",
    "phone": "Phone Number",
    "person_name": "Name",
    "date": "Date",
    "account_number": "Account Number",
    "routing_number": "Routing Number",
    "id_number": "ID Number",
}

FIELD_CATEGORIES = {
    "email": "contact",
    "phone": "contact",
    "person_name": "identity",
    "date": "identity",
    "account_number": "financial",
    "routing_number": "financial",
    "id_number": "identity",
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
    return json.loads(text)


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
    text = _call_gemini(image, SYSTEM_PROMPT)
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
            
        field_type = str(item.get("field_type") or "").strip().lower().replace(" ", "_")
        value = str(item.get("value") or "").strip()
        if not value:
            continue
            
        bbox = item.get("bbox") or [0, 0, 0, 0]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        # Convert normalized 0-1000 scale [ymin, xmin, ymax, xmax] to pixel [x, y, w, h]
        ymin, xmin, ymax, xmax = [float(v) for v in bbox]
        
        # Scale back to original image dimensions
        px_x = (xmin / 1000.0) * img_w
        px_y = (ymin / 1000.0) * img_h
        px_w = ((xmax - xmin) / 1000.0) * img_w
        px_h = ((ymax - ymin) / 1000.0) * img_h

        left, top, right, bottom = int(max(0, px_x)), int(max(0, px_y)), int(max(0, px_x + px_w)), int(max(0, px_y + px_h))

        # Perform exact word-boundary refinement using OCR words
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