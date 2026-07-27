import base64, json, os, re, urllib.request
from io import BytesIO

from .ocr import words_bbox

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


def _call_gemini(image, prompt):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=92)
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode("ascii")}}]}],
        "generationConfig": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    text = body.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    return text.strip() if isinstance(text, str) else None


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
        x, y, w, h = [float(v) for v in bbox]
        left, top, right, bottom = int(max(0, x)), int(max(0, y)), int(max(0, x + w)), int(max(0, y + h))
        if right <= left or bottom <= top:
            left, top, right, bottom = _bbox_from_words(words, value, (left, top, right, bottom), img_w, img_h)
        else:
            left, top, right, bottom = _bbox_from_words(words, value, (left, top, right - left, bottom - top), img_w, img_h)
        label = str(item.get("label") or FIELD_LABELS.get(field_type, field_type.replace("_", " ").title()))
        category = FIELD_CATEGORIES.get(field_type, "generic")
        out.append({"id": counter.next(), "field_type": field_type, "display_label": label, "category": category, "value": value, "page": page, "bbox": (left, top, right, bottom)})
    return out
