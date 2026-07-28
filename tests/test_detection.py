from PIL import Image

from engine import detectors, gemini_detector


def _make_word(text, left, top, right, bottom, conf=90, line_key=0):
    return {
        "text": text,
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
        "right": right,
        "bottom": bottom,
        "conf": conf,
        "line_key": line_key,
    }


def _make_line(key, text, left, top, right, bottom, word_idxs):
    return {"key": key, "text": text, "left": left, "top": top, "right": right, "bottom": bottom, "word_idxs": word_idxs}


def test_generic_label_with_whitespace_separator_is_detected():
    words = [
        _make_word("Account", 10, 0, 70, 20, 90, 0),
        _make_word("Number", 75, 0, 135, 20, 90, 0),
        _make_word("1234567890", 140, 0, 220, 20, 90, 0),
    ]
    lines = [_make_line(0, "Account Number 1234567890", 10, 0, 220, 20, [0, 1, 2])]

    instances = detectors.detect_generic_labels(words, lines, 0, 300, 200, detectors.InstanceCounter(), set())

    assert len(instances) == 1
    assert instances[0]["display_label"] == "Account Number"
    assert instances[0]["value"] == "1234567890"


def test_generic_label_on_next_line_is_detected():
    words = [
        _make_word("Father's", 10, 0, 80, 16, 90, 0),
        _make_word("Name", 82, 0, 120, 16, 90, 0),
        _make_word("John", 10, 24, 45, 40, 90, 1),
        _make_word("Smith", 48, 24, 95, 40, 90, 1),
    ]
    lines = [
        _make_line(0, "Father's Name", 10, 0, 120, 16, [0, 1]),
        _make_line(1, "John Smith", 10, 24, 95, 40, [2, 3]),
    ]

    instances = detectors.detect_generic_labels(words, lines, 0, 300, 200, detectors.InstanceCounter(), set())

    assert len(instances) == 1
    assert instances[0]["display_label"] == "Father's Name"
    assert instances[0]["value"] == "John Smith"


def test_gemini_detector_parses_json_and_converts_bbox(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(gemini_detector, "_call_gemini", lambda image, prompt: '[{"field_type": "email", "value": "john@example.com", "bbox": [10, 20, 80, 15], "label": "Email Address"}]')

    words = [_make_word("john@example.com", 10, 20, 90, 35, 90, 0)]
    lines = [_make_line(0, "john@example.com", 10, 20, 90, 35, [0])]

    instances = gemini_detector.detect_gemini_fields(
        Image.new("RGB", (100, 100)), words, lines, 0, 100, 100, detectors.InstanceCounter(),
    )

    assert len(instances) == 1
    assert instances[0]["field_type"] == "email"
    assert instances[0]["display_label"] == "Email Address"
    assert instances[0]["bbox"] == (10, 20, 90, 35)


def test_gemini_prompt_includes_document_and_ocr_guidance(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    captured = {}

    def fake_call(image, prompt):
        captured["prompt"] = prompt
        return '[]'

    monkeypatch.setattr(gemini_detector, "_call_gemini", fake_call)

    gemini_detector.detect_gemini_fields(
        Image.new("RGB", (100, 100)),
        [_make_word("Email", 10, 10, 40, 24, 90, 0)],
        [_make_line(0, "Email", 10, 10, 40, 24, [0])],
        0, 100, 100, detectors.InstanceCounter(),
    )

    prompt = captured["prompt"]
    assert "document" in prompt.lower()
    assert "ocr" in prompt.lower()
    assert "visible values" in prompt.lower()


def test_gemini_detector_parses_wrapped_json_payload(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(gemini_detector, "_call_gemini", lambda image, prompt: '{"fields": [{"field_type": "email", "value": "john@example.com", "bbox": [10, 20, 80, 15], "label": "Email Address", "page_number": 1}]}')

    words = [_make_word("john@example.com", 10, 20, 90, 35, 90, 0)]
    lines = [_make_line(0, "john@example.com", 10, 20, 90, 35, [0])]

    instances = gemini_detector.detect_gemini_fields(
        Image.new("RGB", (100, 100)), words, lines, 0, 100, 100, detectors.InstanceCounter(),
    )

    assert len(instances) == 1
    assert instances[0]["field_type"] == "email"


def test_gemini_pdf_bytes_wrapper_returns_empty_when_api_unavailable(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = gemini_detector._call_gemini_pdf_bytes(b"%PDF-1.4\n%test", "prompt")

    assert result == ""
