from engine import custom


def test_extract_custom_targets_from_description_format():
    text = "Please mask the name 'John Smith' in the description and redact the full row"
    targets = custom.extract_custom_targets(text)
    assert targets == [("John Smith", "row")]


def test_extract_custom_targets_from_plain_name_instruction():
    text = "Mask the name Sarah Ahmed everywhere"
    targets = custom.extract_custom_targets(text)
    assert targets == [("Sarah Ahmed", "token")]
