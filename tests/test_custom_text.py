from engine import custom


def test_extract_custom_targets_mask_name():
    result = custom.extract_custom_targets("mask ZHANG LIU")
    assert result == [("ZHANG LIU", "token")]


def test_extract_custom_targets_mask_name_sentence():
    result = custom.extract_custom_targets("Please mask ZHANG LIU from the document")
    assert result == [("ZHANG LIU", "token")]


def test_extract_custom_targets_redact_name():
    result = custom.extract_custom_targets("redact the name John Doe")
    assert result == [("John Doe", "token")]
