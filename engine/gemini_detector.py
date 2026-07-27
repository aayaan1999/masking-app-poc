import base64
import json
import os
import re
import urllib.request

SYSTEM_PROMPT = (
    "You are a document field extractor. Find only clearly visible PII or financial fields. "
    "Return strict JSON as an array of objects with keys: field_type, value, bbox, label. "
    "bbox must be [x, y, width, height] in pixel coordinates. "
    "Use field_type values: email, phone, person_name, date, account_number, routing_number, id_number."
)

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


def _call_gemini_pdf(pdf_bytes, prompt):
    """Sends raw PDF bytes directly to the Gemini API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is not set.")
        return None

    encoded_pdf = base64.b64encode(pdf_bytes).decode("ascii")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": encoded_pdf,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0.0},
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
            text = (
                body.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return text.strip() if isinstance(text, str) else None
    except urllib.error.HTTPError as e:
        # Prints full error details from Google API
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def _parse_json(text):
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    return json.loads(text)


def detect_gemini_fields_pdf(pdf_bytes, page_num, img_w, img_h, counter):
    """Accepts PDF bytes directly, extracts fields via Gemini, and formats output."""
    text = _call_gemini_pdf(pdf_bytes, SYSTEM_PROMPT)
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

        x, y, w, h = [float(v) for v in bbox]
        left, top, right, bottom = (
            int(max(0, x)),
            int(max(0, y)),
            int(max(0, x + w)),
            int(max(0, y + h)),
        )

        label = str(
            item.get("label")
            or FIELD_LABELS.get(field_type, field_type.replace("_", " ").title())
        )
        category = FIELD_CATEGORIES.get(field_type, "generic")

        out.append(
            {
                "id": counter.next(),
                "field_type": field_type,
                "display_label": label,
                "category": category,
                "value": value,
                "page": page_num,
                "bbox": (left, top, right, bottom),
            }
        )
    return out