import pytest

from personica.memory.short_term import ShortTermMemory


def fill(stm: ShortTermMemory, pairs: int) -> None:
    for i in range(pairs):
        stm.add("user", f"user message {i}")
        stm.add("assistant", f"assistant reply {i}")


def test_rejects_invalid_window():
    with pytest.raises(ValueError):
        ShortTermMemory(keep_last_turns=0)


def test_no_summarize_below_window():
    stm = ShortTermMemory(keep_last_turns=3)
    fill(stm, 3)  # exactly the window: 6 messages
    assert stm.maybe_summarize(lambda p: "summary") is False
    assert stm.summary == ""
    assert len(stm.turns) == 6


def test_summarize_folds_old_turns():
    stm = ShortTermMemory(keep_last_turns=2)
    fill(stm, 4)  # 8 messages; window is 4
    assert stm.maybe_summarize(lambda p: "the summary") is True
    assert stm.summary == "the summary"
    assert len(stm.turns) == 4
    # the newest pair must survive verbatim
    assert stm.turns[-1].content == "assistant reply 3"
    assert stm.turns[-2].content == "user message 3"


def test_summarize_keeps_pairs_intact():
    stm = ShortTermMemory(keep_last_turns=2)
    fill(stm, 5)
    stm.maybe_summarize(lambda p: "s")
    # window starts on a user message when pairs are complete
    assert stm.turns[0].role == "user"
    assert stm.turns[-1].role == "assistant"


def test_failed_summarization_loses_nothing():
    stm = ShortTermMemory(keep_last_turns=2)
    fill(stm, 4)

    def boom(prompt):
        raise RuntimeError("LLM down")

    assert stm.maybe_summarize(boom) is False
    assert stm.summary == ""
    assert len(stm.turns) == 8  # nothing dropped


def test_empty_summary_result_loses_nothing():
    stm = ShortTermMemory(keep_last_turns=2)
    fill(stm, 4)
    assert stm.maybe_summarize(lambda p: "   ") is False
    assert len(stm.turns) == 8


def test_existing_summary_is_folded_into_prompt():
    stm = ShortTermMemory(keep_last_turns=1)
    stm.summary = "user lives in Boston"
    fill(stm, 2)
    prompts = []

    def capture(prompt):
        prompts.append(prompt)
        return "updated summary"

    assert stm.maybe_summarize(capture) is True
    assert "user lives in Boston" in prompts[0]
    assert "user message 0" in prompts[0]  # old turn in transcript
    assert "user message 1" not in prompts[0]  # recent turn not summarized


def test_as_messages_orders_summary_first():
    stm = ShortTermMemory(keep_last_turns=2)
    stm.summary = "past context"
    stm.add("user", "hi")
    stm.add("assistant", "hello")
    msgs = stm.as_messages()
    assert msgs[0]["role"] == "system"
    assert "past context" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"]


def test_as_messages_without_summary():
    stm = ShortTermMemory()
    stm.add("user", "hi")
    msgs = stm.as_messages()
    assert msgs == [{"role": "user", "content": "hi"}]
