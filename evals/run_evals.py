"""Memory-recall eval harness for Personica.

Runs scripted multi-session scenarios end-to-end against the REAL pipeline
(real LLM via OpenRouter, real ChromaDB vector store in a temp directory) and
scores whether facts survive across sessions:

- cross-session recall:  a fact stated in session 1 must be answered in session 2
- supersession:          an updated fact must override the stale one
- multi-fact recall:     several facts from one session must all be retrievable
- no-fabrication:        with an empty memory, the assistant must not invent facts

Requires OPENROUTER_API_KEY (costs a few cents per run). This is deliberately
separate from the unit tests, which are fully offline.

Usage: uv run python evals/run_evals.py
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import tempfile
import uuid

from personica.config import Settings
from personica.consolidation import consolidate_facts
from personica.llm import LLMClient
from personica.memory.long_term import ChromaMemoryStore
from personica.memory.short_term import ShortTermMemory
from personica.pipeline import TurnPipeline, extract_session_facts

SCENARIOS = [
    {
        "name": "cross-session recall",
        "sessions": [
            ["Hi! My name is Alice and I work as a data scientist in Berlin."],
        ],
        "probe": "What's my name and where do I work?",
        "expect": ["alice", "berlin"],
    },
    {
        "name": "fact update (supersession)",
        "sessions": [
            ["I live in Boston."],
            ["Quick update - I moved to San Francisco last week."],
        ],
        "probe": "Which city do I live in right now? Answer with just the city.",
        "expect": ["san francisco"],
    },
    {
        "name": "multi-fact recall",
        "sessions": [
            ["I love playing guitar.", "Also, my favorite food is sushi."],
        ],
        "probe": "What instrument do I play, and what's my favorite food?",
        "expect": ["guitar", "sushi"],
    },
    {
        "name": "no fabrication on empty memory",
        "sessions": [],
        "probe": (
            "What's my favorite color? If you don't know, say exactly: "
            "\"I don't know\"."
        ),
        "expect": ["don't know"],
    },
]


def run_session(settings, store, llm, user_messages) -> str:
    """Simulate one full session: turns, then fact extraction + consolidation."""
    stm = ShortTermMemory(keep_last_turns=settings.keep_last_turns)
    pipeline = TurnPipeline(
        llm=llm, store=store, short_term=stm, settings=settings)
    full_log = []
    reply = ""
    for msg in user_messages:
        result = pipeline.run(msg)
        reply = result.assistant
        full_log.append({"role": "user", "content": msg})
        full_log.append({"role": "assistant", "content": reply})
    if full_log:
        facts = extract_session_facts(llm, full_log)
        consolidate_facts(llm, store, facts, session_id=str(uuid.uuid4()))
    return reply


def run_scenario(scenario, base_settings, embedding_fn=None) -> tuple[bool, str, list]:
    with tempfile.TemporaryDirectory() as tmp:
        settings = dataclasses.replace(base_settings, data_dir=tmp)
        llm = LLMClient(settings)
        store = ChromaMemoryStore(
            persist_directory=os.path.join(tmp, "chroma"),
            embed_model=settings.embed_model,
            embedding_fn=embedding_fn,
        )

        for session_messages in scenario["sessions"]:
            run_session(settings, store, llm, session_messages)

        # probe in a fresh session (fresh short-term memory, same store)
        stm = ShortTermMemory(keep_last_turns=settings.keep_last_turns)
        pipeline = TurnPipeline(
            llm=llm, store=store, short_term=stm, settings=settings)
        answer = pipeline.run(scenario["probe"]).assistant

        low = answer.lower()
        missing = [term for term in scenario["expect"] if term not in low]
        return not missing, answer, missing


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    base_settings = Settings.from_env()
    if not base_settings.api_key:
        print("OPENROUTER_API_KEY is not set — evals need a real LLM.",
              file=sys.stderr)
        sys.exit(2)

    # load the embedding model once and share it across scenarios
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=base_settings.embed_model)

    print(f"Running {len(SCENARIOS)} memory eval scenarios "
          f"(chat={base_settings.chat_model})...\n")

    passed = 0
    for scenario in SCENARIOS:
        ok, answer, missing = run_scenario(
            scenario, base_settings, embedding_fn)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {scenario['name']}")
        print(f"       probe:  {scenario['probe']}")
        print(f"       answer: {answer}")
        if missing:
            print(f"       missing expected terms: {missing}")
        print()

    print(f"{'=' * 60}")
    print(f"RESULT: {passed}/{len(SCENARIOS)} scenarios passed")
    sys.exit(0 if passed == len(SCENARIOS) else 1)


if __name__ == "__main__":
    main()
