"""Shared fixtures and test doubles. No test in this suite calls a real API."""

from __future__ import annotations

import uuid

import pytest

from personica.config import Settings
from personica.llm import LLMError
from personica.memory.long_term import MemoryItem


def make_settings(**overrides) -> Settings:
    base = dict(
        data_dir="./personica_data",
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        api_key="test-key",
        chat_model="test/chat-model",
        utility_model="test/utility-model",
        base_url="https://example.invalid/api/v1",
        keep_last_turns=5,
        retrieval_top_k=5,
        retrieval_min_score=0.20,
        relevance_weight=0.7,
        recency_half_life_days=30.0,
        timezone="UTC",
        log_level="INFO",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(data_dir=str(tmp_path))


def make_memory(
    text: str = "User likes pizza",
    epoch: float = 1.0,
    score: float = 0.9,
    kind: str = "session_summary",
) -> MemoryItem:
    return MemoryItem(
        id=str(uuid.uuid4()),
        text=text,
        kind=kind,
        session_id="test-session",
        created_at_utc="2026-07-05T10:00:00+00:00",
        epoch=epoch,
        score=score,
        metadata={},
    )


class FakeLLM:
    """Scripted LLM double.

    `complete()` responses are dispatched by purpose; a value may be a string,
    an exception instance (raised), or a callable receiving the prompt.
    All calls are recorded for assertions.
    """

    def __init__(self, chat_response="(fake reply)", complete_responses=None):
        self.chat_response = chat_response
        self.complete_responses = dict(complete_responses or {})
        self.chat_calls: list[dict] = []
        self.complete_calls: list[dict] = []

    def chat(self, messages, temperature=0.2, purpose="chat"):
        self.chat_calls.append(
            {"messages": messages, "temperature": temperature})
        if isinstance(self.chat_response, Exception):
            raise self.chat_response
        return self.chat_response

    def complete(self, prompt, *, use_utility_model=True, temperature=0.0,
                 purpose="utility"):
        self.complete_calls.append({
            "prompt": prompt,
            "purpose": purpose,
            "use_utility_model": use_utility_model,
        })
        if purpose not in self.complete_responses:
            raise AssertionError(
                f"FakeLLM got unscripted complete() purpose: {purpose!r}")
        handler = self.complete_responses[purpose]
        if isinstance(handler, Exception):
            raise handler
        if callable(handler):
            return handler(prompt)
        return handler


class FakeStore:
    """In-memory stand-in for ChromaMemoryStore."""

    def __init__(self, results_by_query=None, default_results=None):
        self.results_by_query = dict(results_by_query or {})
        self.default_results = list(default_results or [])
        self.search_calls: list[str] = []
        self.added: list[dict] = []
        self.deleted: list[str] = []

    def search(self, query, top_k=5, min_score=0.30, where=None,
               relevance_weight=0.7, recency_half_life_days=30.0):
        self.search_calls.append(query)
        return self.results_by_query.get(query, self.default_results)

    def add_memory(self, text, kind, session_id, metadata=None):
        self.added.append(
            {"text": text, "kind": kind, "session_id": session_id})
        return f"fake-mem-{len(self.added)}"

    def delete_memories(self, ids):
        self.deleted.extend(ids)

    def count(self):
        return len(self.added)


__all__ = ["make_settings", "make_memory", "FakeLLM", "FakeStore", "LLMError"]
