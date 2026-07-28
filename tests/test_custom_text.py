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


def test_extract_custom_targets_redact_bare_name():
    result = custom.extract_custom_targets("redact the name")
    assert result == [("name", "token")]


def test_extract_custom_targets_mask_sex():
    result = custom.extract_custom_targets("mask sex")
    assert result == [("sex", "token")]


def test_extract_custom_targets_mask_issue_dates():
    result = custom.extract_custom_targets("mask issue dates")
    assert result == [("issue date", "token")]
