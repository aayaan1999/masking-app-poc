from engine import custom


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
