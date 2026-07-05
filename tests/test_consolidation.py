"""Consolidation tests — add/skip/replace decisions with fake LLM/store."""

import json

from conftest import FakeLLM, FakeStore, make_memory
from personica.consolidation import consolidate_fact, consolidate_facts
from personica.llm import LLMError


def decision(action, replace=None, text=""):
    return json.dumps(
        {"action": action, "replace": replace or [], "text": text})


def test_no_similar_memories_adds_directly():
    llm = FakeLLM()  # any complete() call would raise: none expected
    store = FakeStore(default_results=[])

    report = consolidate_fact(llm, store, "Plays guitar", "s1")

    assert report["action"] == "add"
    assert store.added == [
        {"text": "Plays guitar", "kind": "fact", "session_id": "s1"}]
    assert llm.complete_calls == []  # no LLM call needed


def test_duplicate_fact_is_skipped():
    llm = FakeLLM(complete_responses={"consolidation": decision("skip")})
    store = FakeStore(default_results=[make_memory(text="Loves pizza")])

    report = consolidate_fact(llm, store, "Favorite food is pizza", "s1")

    assert report["action"] == "skip"
    assert store.added == []
    assert store.deleted == []


def test_superseding_fact_replaces_stale_memory():
    stale = make_memory(text="Lives in Boston")
    llm = FakeLLM(complete_responses={
        "consolidation": decision(
            "replace", replace=[1],
            text="Lives in San Francisco (previously Boston)"),
    })
    store = FakeStore(default_results=[stale])

    report = consolidate_fact(llm, store, "Moved to San Francisco", "s2")

    assert report["action"] == "replace"
    assert store.deleted == [stale.id]
    assert store.added[0]["text"] == "Lives in San Francisco (previously Boston)"


def test_out_of_range_replace_indices_are_ignored():
    stale = make_memory(text="Lives in Boston")
    llm = FakeLLM(complete_responses={
        "consolidation": decision("replace", replace=[1, 5, 0], text="merged"),
    })
    store = FakeStore(default_results=[stale])

    consolidate_fact(llm, store, "Moved to SF", "s2")

    assert store.deleted == [stale.id]  # only the valid index 1


def test_llm_failure_defaults_to_add():
    llm = FakeLLM(complete_responses={"consolidation": LLMError("down")})
    store = FakeStore(default_results=[make_memory()])

    report = consolidate_fact(llm, store, "New fact", "s1")

    assert report["action"] == "add"
    assert store.added[0]["text"] == "New fact"
    assert store.deleted == []


def test_malformed_decision_defaults_to_add():
    llm = FakeLLM(complete_responses={"consolidation": "not json at all"})
    store = FakeStore(default_results=[make_memory()])

    report = consolidate_fact(llm, store, "New fact", "s1")

    assert report["action"] == "add"
    assert store.added[0]["text"] == "New fact"


def test_invalid_action_defaults_to_add():
    llm = FakeLLM(complete_responses={
        "consolidation": '{"action": "explode", "replace": [], "text": ""}'})
    store = FakeStore(default_results=[make_memory()])

    report = consolidate_fact(llm, store, "New fact", "s1")
    assert report["action"] == "add"


def test_consolidate_facts_processes_each_fact():
    llm = FakeLLM(complete_responses={"consolidation": decision("skip")})
    store = FakeStore(default_results=[])

    reports = consolidate_facts(
        llm, store, ["fact one", "fact two"], "s1")

    assert [r["action"] for r in reports] == ["add", "add"]
    assert len(store.added) == 2
