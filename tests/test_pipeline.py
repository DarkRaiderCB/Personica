"""TurnPipeline tests — the full turn flow with fake LLM/store doubles."""

import json

import pytest

from conftest import FakeLLM, FakeStore, make_memory, make_settings
from personica.llm import LLMError
from personica.memory.short_term import ShortTermMemory
from personica.pipeline import (
    TurnPipeline,
    build_memory_block,
    extract_session_facts,
    format_timestamp,
)
from personica.tracing import Tracer


def make_pipeline(llm=None, store=None, stm=None, settings=None, tracer=None):
    return TurnPipeline(
        llm=llm or FakeLLM(),
        store=store or FakeStore(),
        short_term=stm or ShortTermMemory(),
        settings=settings or make_settings(),
        tracer=tracer,
    )


def scripted_llm(**overrides):
    responses = {
        "query_rewrite": "keywords for search",
        "relevance_check": "RELEVANT",
    }
    responses.update(overrides)
    return FakeLLM(chat_response="the answer", complete_responses=responses)


# ── build_memory_block ──────────────────────────────────────────────────────

def test_memory_block_tags_newest_as_latest():
    old = make_memory(text="lives in Boston", epoch=1.0)
    new = make_memory(text="lives in Chicago", epoch=2.0)
    block = build_memory_block([old, new])
    assert "lives in Chicago" in block
    # the marker appears inside the metadata brackets as "[LATEST]]";
    # the block footer also mentions "[LATEST]" so count the bracketed form
    assert block.count("[LATEST]]") == 1
    lines = block.splitlines()
    latest_idx = next(
        i for i, line in enumerate(lines) if "[LATEST]]" in line)
    # the marker is on the metadata line following the newest memory's text
    assert "lives in Chicago" in lines[latest_idx - 1]


def test_memory_block_single_memory_has_no_latest_tag():
    block = build_memory_block([make_memory()])
    assert "[LATEST]]" not in block


def test_format_timestamp():
    assert format_timestamp("2026-07-05T10:00:00+00:00") == "2026-07-05 10:00:00"


# ── TurnPipeline.run ────────────────────────────────────────────────────────

def test_turn_injects_relevant_memories():
    memories = [make_memory(text="name is Sanyog", epoch=1.0),
                make_memory(text="loves pizza", epoch=2.0)]
    llm = scripted_llm()
    store = FakeStore(default_results=memories)
    stm = ShortTermMemory()
    pipeline = make_pipeline(llm=llm, store=store, stm=stm)

    result = pipeline.run("what's my name?")

    assert result.assistant == "the answer"
    assert result.memories_injected is True
    assert result.retrieval_query == "keywords for search"
    # memory block was injected as a system message
    sent = llm.chat_calls[0]["messages"]
    memory_msgs = [m for m in sent
                   if m["role"] == "system" and "LONG-TERM MEMORIES" in m["content"]]
    assert len(memory_msgs) == 1
    assert "name is Sanyog" in memory_msgs[0]["content"]
    assert "[LATEST]" in memory_msgs[0]["content"]
    # short-term memory recorded the pair
    assert [t.role for t in stm.turns] == ["user", "assistant"]


def test_gatekeeper_blocks_irrelevant_memories():
    llm = scripted_llm(relevance_check="IRRELEVANT")
    store = FakeStore(default_results=[make_memory()])
    pipeline = make_pipeline(llm=llm, store=store)

    result = pipeline.run("how do I bake bread?")

    assert result.memories_injected is False
    sent = llm.chat_calls[0]["messages"]
    assert not any("LONG-TERM MEMORIES" in m["content"] for m in sent)


def test_no_memories_skips_relevance_check():
    llm = scripted_llm()
    pipeline = make_pipeline(llm=llm, store=FakeStore())  # empty store

    result = pipeline.run("hello")

    assert result.memories_injected is False
    purposes = [c["purpose"] for c in llm.complete_calls]
    assert "relevance_check" not in purposes


def test_rewrite_failure_falls_back_to_raw_message():
    llm = scripted_llm(query_rewrite=LLMError("rewrite model down"))
    store = FakeStore(default_results=[])
    pipeline = make_pipeline(llm=llm, store=store)

    pipeline.run("what's my name?")

    # rewrite failed → only the raw message is searched (no duplicate query)
    assert store.search_calls == ["what's my name?"]


def test_retrieval_unions_rewritten_and_raw_queries():
    memory = make_memory(text="name is Sanyog")
    llm = scripted_llm(query_rewrite="unrelated keywords")
    store = FakeStore(
        results_by_query={"unrelated keywords": [],
                          "what's my name?": [memory]})
    pipeline = make_pipeline(llm=llm, store=store)

    result = pipeline.run("what's my name?")

    # both queries are searched; the raw-query hit survives the union
    assert store.search_calls == ["unrelated keywords", "what's my name?"]
    assert result.memories_injected is True


def test_retrieval_union_dedupes_and_ranks():
    shared = make_memory(text="works in Berlin", score=0.4)
    kw_only = make_memory(text="name is Sanyog", score=0.5)
    raw_only = make_memory(text="loves sushi", score=0.3)
    llm = scripted_llm(query_rewrite="keywords")
    store = FakeStore(results_by_query={
        "keywords": [kw_only, shared],
        "who am I?": [shared, raw_only],
    })
    pipeline = make_pipeline(llm=llm, store=store)

    result = pipeline.run("who am I?")

    texts = [m.text for m in result.retrieved]
    assert sorted(texts) == ["loves sushi", "name is Sanyog", "works in Berlin"]
    assert len(texts) == 3  # shared memory appears once


def test_relevance_check_failure_defaults_to_relevant():
    llm = scripted_llm(relevance_check=LLMError("gatekeeper down"))
    store = FakeStore(default_results=[make_memory()])
    pipeline = make_pipeline(llm=llm, store=store)

    result = pipeline.run("what do you know about me?")
    assert result.memories_injected is True


def test_chat_failure_raises_and_keeps_user_turn():
    llm = scripted_llm()
    llm.chat_response = LLMError("chat model down")
    stm = ShortTermMemory()
    pipeline = make_pipeline(llm=llm, stm=stm)

    with pytest.raises(LLMError):
        pipeline.run("hello")

    # user turn recorded, no dangling assistant turn
    assert [t.role for t in stm.turns] == ["user"]


def test_rolling_summary_triggered_when_window_exceeded():
    llm = scripted_llm(rolling_summary="compressed history")
    stm = ShortTermMemory(keep_last_turns=1)
    pipeline = make_pipeline(
        llm=llm, stm=stm, settings=make_settings(keep_last_turns=1))

    pipeline.run("turn one")
    assert stm.summary == ""  # exactly at window, nothing folded
    pipeline.run("turn two")
    assert stm.summary == "compressed history"
    assert len(stm.turns) == 2


def test_trace_events_written_for_full_turn(tmp_path):
    tracer = Tracer(str(tmp_path), "sess-t")
    llm = scripted_llm()
    store = FakeStore(default_results=[make_memory()])
    pipeline = make_pipeline(llm=llm, store=store, tracer=tracer)

    pipeline.run("what's my name?")

    with open(tmp_path / "sess-t.jsonl", encoding="utf-8") as f:
        events = [json.loads(line)["event"] for line in f if line.strip()]
    assert events == [
        "turn_start", "query_rewrite", "retrieval", "gatekeeper",
        "turn_complete",
    ]


# ── extract_session_facts ───────────────────────────────────────────────────

def test_fact_extraction_returns_atomic_facts():
    llm = FakeLLM(complete_responses={
        "fact_extraction": '{"facts": ["Name is Sanyog", "Loves pizza"]}'})
    facts = extract_session_facts(
        llm, [{"role": "user", "content": "I'm Sanyog and I love pizza"}])
    assert facts == ["Name is Sanyog", "Loves pizza"]
    # extraction must use the full-quality model
    assert llm.complete_calls[0]["use_utility_model"] is False


def test_fact_extraction_handles_fenced_json():
    llm = FakeLLM(complete_responses={
        "fact_extraction": '```json\n{"facts": ["Plays guitar"]}\n```'})
    facts = extract_session_facts(llm, [{"role": "user", "content": "x"}])
    assert facts == ["Plays guitar"]


def test_fact_extraction_empty_facts_returns_empty_list():
    llm = FakeLLM(complete_responses={"fact_extraction": '{"facts": []}'})
    assert extract_session_facts(
        llm, [{"role": "user", "content": "what's my name?"}]) == []


def test_fact_extraction_drops_blank_and_non_string_entries():
    llm = FakeLLM(complete_responses={
        "fact_extraction": '{"facts": ["  real fact  ", "", 42, null]}'})
    assert extract_session_facts(
        llm, [{"role": "user", "content": "x"}]) == ["real fact"]


def test_fact_extraction_unparseable_returns_empty_list():
    llm = FakeLLM(complete_responses={
        "fact_extraction": "I could not produce JSON, sorry."})
    assert extract_session_facts(
        llm, [{"role": "user", "content": "x"}]) == []


def test_fact_extraction_failure_returns_empty_list():
    llm = FakeLLM(complete_responses={
        "fact_extraction": LLMError("model down")})
    assert extract_session_facts(
        llm, [{"role": "user", "content": "hi"}]) == []
