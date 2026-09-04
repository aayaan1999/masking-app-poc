from engine import custom, ocr


def _word(text, left, top, right, bottom, conf=90):
    return {
        "text": text, "left": left, "top": top, "width": right - left,
        "height": bottom - top, "right": right, "bottom": bottom, "conf": conf,
    }


def test_cluster_into_lines_splits_wide_bilingual_columns():
    # A bilingual document (English label/value column, then a mirrored
    # Arabic column) prints both at the same y-position, separated by a
    # wide blank gutter — e.g. a real Dubai trade license's "Trade Name"
    # row: English column ends around x=1010, Arabic column starts
    # around x=1476 (a ~466px gutter), vs. a normal ~10-20px gap between
    # words within one column. Without treating that gutter as a column
    # break, both columns were merged into one "line" spanning the
    # entire page width.
    words = [
        _word("Trade", 156, 712, 241, 742),
        _word("Name", 252, 712, 337, 742),
        _word("DOE", 512, 712, 584, 742),
        _word("GLOBAL", 606, 712, 748, 742),
        _word("L.L.C", 933, 712, 1010, 742),
        _word("Arabic1", 1476, 712, 1610, 742),
        _word("Arabic2", 1623, 712, 1727, 742),
    ]
    clusters = ocr._cluster_into_lines(words)

    assert len(clusters) == 2
    left_col, right_col = sorted(clusters, key=lambda c: min(words[i]["left"] for i in c))
    assert [words[i]["text"] for i in left_col] == ["Trade", "Name", "DOE", "GLOBAL", "L.L.C"]
    assert [words[i]["text"] for i in right_col] == ["Arabic1", "Arabic2"]


def test_cluster_into_lines_keeps_normal_label_value_gap_together():
    # A genuinely wide label-to-value gap within a single column (as
    # seen on the same document, ~175-290px) must NOT be split — only
    # gaps wide enough to be an actual column gutter should be.
    words = [
        _word("Office", 158, 2202, 248, 2238),
        _word("No.", 259, 2202, 307, 2238),
        _word("1205", 520, 2202, 595, 2238),
    ]
    clusters = ocr._cluster_into_lines(words)
    assert len(clusters) == 1


def test_extract_custom_targets_from_description_format():
    text = "Please mask the name 'John Smith' in the description and redact the full row"
    targets = custom.extract_custom_targets(text)
    assert targets == [("John Smith", "row")]


def test_extract_custom_targets_from_plain_name_instruction():
    text = "Mask the name Sarah Ahmed everywhere"
    targets = custom.extract_custom_targets(text)
    assert targets == [("Sarah Ahmed", "token")]


def test_extract_custom_targets_multiple_commands():
    text = 'Mask ZHANG LIU and redact "Acme Corp" everywhere'
    targets = custom.extract_custom_targets(text)
    assert targets == [("ZHANG LIU", "token"), ("Acme Corp", "token")]


def test_extract_custom_targets_lowercase_name():
    text = 'mask zhang liu'
    targets = custom.extract_custom_targets(text)
    assert targets == [("zhang liu", "token")]


def test_extract_custom_targets_for_clause():
    text = 'mask all records for sarah ahmed'
    targets = custom.extract_custom_targets(text)
    assert targets == [("sarah ahmed", "token")]
