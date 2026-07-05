"""ChromaMemoryStore tests — real ChromaDB on disk (tmp_path), but with a
deterministic word-overlap embedding function so no model download or network
is needed."""

import hashlib
import time

import pytest
from chromadb.api.types import EmbeddingFunction

from personica.memory.long_term import ChromaMemoryStore, hybrid_score

DIM = 64


class FakeEmbeddingFunction(EmbeddingFunction):
    """Deterministic bag-of-words hashing embeddings: texts sharing words are
    close in cosine space, disjoint texts are orthogonal."""

    def __init__(self):
        pass  # chroma's base __init__ only emits a deprecation warning

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbeddingFunction()

    @staticmethod
    def _embed(text):
        vec = [0.0] * DIM
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            vec[bucket] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def name():
        return "fake-embeddings"


@pytest.fixture
def store(tmp_path):
    return ChromaMemoryStore(
        persist_directory=str(tmp_path / "chroma"),
        embedding_fn=FakeEmbeddingFunction(),
    )


def test_empty_store_returns_nothing(store):
    assert store.count() == 0
    assert store.search("anything") == []


def test_add_and_search_roundtrip(store):
    mem_id = store.add_memory(
        "User loves pizza and pasta", kind="session_summary", session_id="s1")
    assert store.count() == 1

    results = store.search("pizza")
    assert len(results) == 1
    m = results[0]
    assert m.id == mem_id
    assert m.text == "User loves pizza and pasta"
    assert m.kind == "session_summary"
    assert m.session_id == "s1"
    assert m.epoch > 0
    assert 0.0 <= m.score <= 1.0
    assert m.created_at_utc  # ISO timestamp recorded


def test_min_score_filters_unrelated_memories(store):
    store.add_memory("User loves pizza", kind="fact", session_id="s1")
    # completely disjoint vocabulary → orthogonal fake embedding → score ~0
    assert store.search("quarterly finance meeting agenda") == []


def test_recency_first_ordering(store):
    store.add_memory("User lives in Boston", kind="fact", session_id="s1")
    time.sleep(0.01)  # ensure distinct epochs
    store.add_memory("User lives in Chicago", kind="fact", session_id="s2")

    results = store.search("lives user city Boston Chicago", min_score=0.0)
    assert len(results) == 2
    assert results[0].text == "User lives in Chicago"  # newest first
    assert results[0].epoch > results[1].epoch


def test_top_k_limits_results(store):
    for i in range(4):
        store.add_memory(f"pizza fact number {i}", kind="fact", session_id="s")
    results = store.search("pizza", top_k=2, min_score=0.0)
    assert len(results) == 2


def test_where_filter_by_kind(store):
    store.add_memory("pizza preference", kind="preference", session_id="s")
    store.add_memory("pizza meeting", kind="event", session_id="s")
    results = store.search(
        "pizza", min_score=0.0, where={"kind": "preference"})
    assert [m.kind for m in results] == ["preference"]


def test_rejects_empty_memory(store):
    with pytest.raises(ValueError):
        store.add_memory("   ", kind="fact", session_id="s")


def test_blank_query_returns_nothing(store):
    store.add_memory("something", kind="fact", session_id="s")
    assert store.search("   ") == []


def test_delete_memories(store):
    mem_id = store.add_memory("User plays chess", kind="fact", session_id="s")
    assert store.count() == 1
    store.delete_memories([mem_id])
    assert store.count() == 0
    store.delete_memories([])  # no-op, must not raise


# ── hybrid_score ────────────────────────────────────────────────────────────

DAY = 86400.0


def test_hybrid_score_fresh_memory_gets_full_recency():
    # age 0 → recency 1.0 → score = w*sim + (1-w)
    assert hybrid_score(0.8, 1000.0, 1000.0, 0.7, 30.0) == pytest.approx(
        0.7 * 0.8 + 0.3)


def test_hybrid_score_halves_recency_at_half_life():
    now = 100 * DAY
    fresh = hybrid_score(0.0, now, now, 0.0, 30.0)          # pure recency
    at_half_life = hybrid_score(0.0, now - 30 * DAY, now, 0.0, 30.0)
    assert fresh == pytest.approx(1.0)
    assert at_half_life == pytest.approx(0.5)


def test_hybrid_score_decays_monotonically():
    now = 100 * DAY
    scores = [hybrid_score(0.5, now - d * DAY, now) for d in (0, 10, 50, 200)]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_score_relevance_can_beat_recency():
    # a much more relevant old memory outranks a barely-relevant fresh one
    now = 100 * DAY
    old_relevant = hybrid_score(0.9, now - 60 * DAY, now, 0.7, 30.0)
    fresh_irrelevant = hybrid_score(0.35, now, now, 0.7, 30.0)
    assert old_relevant > fresh_irrelevant


def test_hybrid_score_future_epoch_clamped():
    # clock skew must not produce recency > 1
    assert hybrid_score(0.0, 2000.0, 1000.0, 0.0, 30.0) == pytest.approx(1.0)


def test_persistence_across_reopen(tmp_path):
    path = str(tmp_path / "chroma")
    store1 = ChromaMemoryStore(
        persist_directory=path, embedding_fn=FakeEmbeddingFunction())
    store1.add_memory("User plays guitar", kind="fact", session_id="s1")

    store2 = ChromaMemoryStore(
        persist_directory=path, embedding_fn=FakeEmbeddingFunction())
    assert store2.count() == 1
    results = store2.search("guitar", min_score=0.0)
    assert results[0].text == "User plays guitar"
