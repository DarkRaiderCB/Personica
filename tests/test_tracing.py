import json

from personica.tracing import Tracer, save_transcript


def read_events(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_tracer_writes_jsonl_events(tmp_path):
    tracer = Tracer(str(tmp_path), "sess-1")
    tracer.event("session_start", model="m")
    tracer.new_turn()
    tracer.event("turn_start", user_chars=5)
    tracer.new_turn()
    tracer.event("turn_start", user_chars=7)

    events = read_events(tmp_path / "sess-1.jsonl")
    assert [e["event"] for e in events] == [
        "session_start", "turn_start", "turn_start"]
    assert [e["turn"] for e in events] == [0, 1, 2]
    assert all(e["session_id"] == "sess-1" for e in events)
    assert all("ts_utc" in e for e in events)
    assert events[0]["model"] == "m"


def test_disabled_tracer_writes_nothing(tmp_path):
    tracer = Tracer(str(tmp_path), "sess-2", enabled=False)
    tracer.event("session_start")
    assert not (tmp_path / "sess-2.jsonl").exists()


def test_tracer_serializes_non_json_values(tmp_path):
    tracer = Tracer(str(tmp_path), "sess-3")
    tracer.event("weird", value={1, 2})  # sets aren't JSON — default=str
    events = read_events(tmp_path / "sess-3.jsonl")
    assert events[0]["event"] == "weird"


def test_save_transcript(tmp_path):
    turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    path = save_transcript(str(tmp_path / "transcripts"), "sess-4", turns)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["session_id"] == "sess-4"
    assert payload["turns"] == turns
    assert "created_at_utc" in payload
