from personica.jsonutil import extract_json


def test_plain_json():
    assert extract_json('{"facts": ["a", "b"]}') == {"facts": ["a", "b"]}


def test_fenced_json():
    text = 'Here you go:\n```json\n{"action": "add"}\n```\nDone.'
    assert extract_json(text) == {"action": "add"}


def test_fence_without_language_tag():
    assert extract_json('```\n{"x": 1}\n```') == {"x": 1}


def test_json_embedded_in_prose():
    text = 'Sure! The answer is {"action": "skip", "replace": []} as requested.'
    assert extract_json(text) == {"action": "skip", "replace": []}


def test_invalid_returns_none():
    assert extract_json("no json here at all") is None


def test_malformed_json_returns_none():
    assert extract_json('{"unclosed": ') is None
