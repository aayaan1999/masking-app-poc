from PIL import Image

from engine import detectors, gemini_detector, pipeline


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


def test_name_detector_does_not_pull_in_same_row_sibling_column():
    # Two "lines" at nearly the same y-position (as produced by
    # ocr._cluster_into_lines splitting a bilingual row into an English
    # column and a far-off Arabic column) are adjacent in the
    # (top, left)-sorted `lines` list. detect_name's "value wraps onto
    # the next line" fallback must not treat that same-row sibling as a
    # continuation of this line's value — only an actual row below.
    words = [
        _make_word("Name", 10, 712, 90, 742),
        _make_word("John", 100, 712, 150, 742),
        _make_word("Smith", 160, 712, 220, 742),
        # A same-row sibling column, far to the right (simulates the
        # Arabic mirror column on a real bilingual document).
        _make_word("Unrelated", 1500, 712, 1600, 742),
        _make_word("Column", 1610, 715, 1700, 745),
    ]
    lines = [
        _make_line(0, "Name John Smith", 10, 712, 220, 742, [0, 1, 2]),
        _make_line(1, "Unrelated Column", 1500, 712, 1700, 745, [3, 4]),
    ]

    instances, _ = detectors.detect_name(words, lines, 0, 2000, 1000, detectors.InstanceCounter())

    assert len(instances) == 1
    assert instances[0]["value"] == "John Smith"


def test_merge_does_not_fuse_adjacent_rows_that_barely_overlap_vertically():
    # Real bboxes from a real Emirates ID scan: a "Name" row and the
    # "Date of Birth" row directly below it are both wide (near full
    # card width), and Arabic diacritics on the Name row extend its
    # bbox down just enough to overlap the DOB row's bbox by ~26px out
    # of the DOB row's own ~113px height. The combined-area overlap
    # ratio alone crosses 0.3 purely from the huge x-overlap, wrongly
    # fusing two entirely different fields ("Abdul Rasheed..." and
    # "01/06/1999") into one nonsense box spanning both rows.
    name_inst = {
        "id": "i1", "field_type": "person_name", "display_label": "Name",
        "category": "identity", "value": "Abdu Rasheed Pothakkaran Kabeer Kabeer Pothakkaran",
        "page": 0, "bbox": (1008, 1637, 2430, 1807),
    }
    dob_inst = {
        "id": "i2", "field_type": "dob", "display_label": "Date of Birth",
        "category": "identity", "value": "01/06/1999",
        "page": 0, "bbox": (1452, 1769, 1790, 1885),
    }

    merged = pipeline._merge_overlapping_instances([name_inst, dob_inst])

    assert len(merged) == 2
    field_types = {m["field_type"] for m in merged}
    assert field_types == {"person_name", "dob"}


def test_merge_still_fuses_genuine_duplicate_detections():
    # Two detectors finding the *same* printed field under different
    # names (Gemini's "occupation" vs. the generic "label:occupation")
    # produce near-identical bboxes — this must still merge into one.
    a = {
        "id": "i1", "field_type": "occupation", "display_label": "Occupation",
        "category": "generic", "value": "Engineer",
        "page": 0, "bbox": (100, 100, 400, 140),
    }
    b = {
        "id": "i2", "field_type": "label:occupation", "display_label": "Occupation",
        "category": "generic", "value": "Engineer",
        "page": 0, "bbox": (98, 102, 402, 138),
    }

    merged = pipeline._merge_overlapping_instances([a, b])

    assert len(merged) == 1
    assert merged[0]["field_type"] == "occupation"


def test_card_number_rejects_low_confidence_digit_run():
    # A real Emirates ID scan's decorative security-pattern watermark
    # (no actual printed digits anywhere in that region) got misread by
    # Tesseract as a clean 13-digit "word". A bare digit run has no
    # other structural signal to lean on, so this detector must reject
    # it below a reasonably high confidence bar — matching every other
    # numeric PII pattern in this file (phone/email require > 35).
    bbox_args = (1346, 2059, 2310, 2151)
    low_conf_words = [_make_word("8141999147907", *bbox_args, conf=25, line_key=0)]
    low_conf_lines = [_make_line(0, "8141999147907", *bbox_args, [0])]
    rejected, _ = detectors.detect_card_number(low_conf_words, low_conf_lines, 0, 3000, 3500, detectors.InstanceCounter())
    assert rejected == []

    high_conf_words = [_make_word("8141999147907", *bbox_args, conf=90, line_key=0)]
    high_conf_lines = [_make_line(0, "8141999147907", *bbox_args, [0])]
    accepted, _ = detectors.detect_card_number(high_conf_words, high_conf_lines, 0, 3000, 3500, detectors.InstanceCounter())
    assert len(accepted) == 1


def test_generic_label_rejects_low_confidence_garbage_prefix():
    # Simulates OCR noise picked up from a noisy/photo-textured region of
    # the scan: a low-confidence garbage token ("Ss") sits right before a
    # real recognized keyword ("Occupation"). The line's *average*
    # confidence is still decent (helped by the high-confidence real
    # word), so this must be caught by checking each label word's own
    # confidence, not just the line average.
    words = [
        _make_word("Ss", 10, 0, 30, 20, conf=12, line_key=0),
        _make_word("Occupation", 34, 0, 120, 20, conf=92, line_key=0),
        _make_word("Engineer", 124, 0, 200, 20, conf=88, line_key=0),
    ]
    lines = [_make_line(0, "Ss Occupation Engineer", 10, 0, 200, 20, [0, 1, 2])]

    instances = detectors.detect_generic_labels(words, lines, 0, 300, 200, detectors.InstanceCounter(), set())

    assert instances == []


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
    # Gemini's own bbox is now trusted directly (padded slightly) instead
    # of being discarded and reconstructed from OCR word matches.
    assert instances[0]["bbox"] == (6, 16, 94, 39)


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


def test_gemini_detector_normalizes_gcc_fields(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        gemini_detector,
        "_call_gemini",
        lambda image, prompt: '[{"field_type": "Issue Date", "value": "01-01-2025", "bbox": [10, 20, 80, 15], "label": "Issue Date"},'
                             + '{"field_type": "Expiry Date", "value": "01-01-2030", "bbox": [10, 40, 80, 15]},'
                             + '{"field_type": "Nationality", "value": "UAE", "bbox": [10, 60, 80, 15]}]'
    )

    words = [
        _make_word("01-01-2025", 10, 20, 90, 35, 90, 0),
        _make_word("01-01-2030", 10, 40, 90, 55, 90, 0),
        _make_word("UAE", 10, 60, 40, 75, 90, 0),
    ]
    lines = [
        _make_line(0, "01-01-2025", 10, 20, 90, 35, [0]),
        _make_line(1, "01-01-2030", 10, 40, 90, 55, [1]),
        _make_line(2, "UAE", 10, 60, 40, 75, [2]),
    ]

    instances = gemini_detector.detect_gemini_fields(
        Image.new("RGB", (100, 100)), words, lines, 0, 100, 100, detectors.InstanceCounter(),
    )

    # field_type is normalized to match the regex detector's own naming
    # (date_of_issue/date_of_expiry, not Gemini's issue_date/expiry_date)
    # so the same physical date found by either detector merges into one
    # group instead of showing up twice under two different names.
    assert {inst["field_type"] for inst in instances} == {"date_of_issue", "date_of_expiry", "nationality"}
    # The first item supplies its own "label" (used verbatim); the second
    # has none, so it falls back to the shared DATE_CONCEPT_LABELS text
    # ("Date of Expiry") that the regex detector also uses for this
    # field_type, rather than a separately-drifting "Expiry Date" string.
    assert any(inst["display_label"] == "Issue Date" and inst["value"] == "01-01-2025" for inst in instances)
    assert any(inst["display_label"] == "Date of Expiry" and inst["value"] == "01-01-2030" for inst in instances)
    assert any(inst["display_label"] == "Nationality" and inst["value"] == "UAE" for inst in instances)


def test_gemini_pdf_bytes_wrapper_returns_empty_when_api_unavailable(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = gemini_detector._call_gemini_pdf_bytes(b"%PDF-1.4\n%test", "prompt")

    assert result == ""
