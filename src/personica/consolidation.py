"""Write-time memory consolidation.

Append-only memory degrades: duplicates accumulate and contradictions are
only resolved at prompt time. Instead, each newly extracted fact is checked
against the most similar existing memories and an LLM decides whether to
add it, skip it (duplicate), or replace the stale memories it supersedes —
the write-time consolidation approach used by systems like MemGPT and mem0.

Failure policy: if the decision step fails or returns garbage, the fact is
added anyway — losing a memory is worse than a temporary duplicate (which a
later consolidation pass can still collapse).
"""

from __future__ import annotations

import logging
from typing import Any

from personica.jsonutil import extract_json
from personica.llm import LLMError
from personica.memory.long_term import MemoryItem
from personica.prompts import build_consolidation_prompt
from personica.tracing import Tracer

logger = logging.getLogger(__name__)

SIMILAR_TOP_K = 3
SIMILAR_MIN_SCORE = 0.35
FACT_KIND = "fact"

VALID_ACTIONS = {"add", "skip", "replace"}


def _parse_decision(raw: str, num_existing: int) -> dict[str, Any] | None:
    """Validate the LLM's consolidation decision; None if unusable."""
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return None
    action = parsed.get("action")
    if action not in VALID_ACTIONS:
        return None
    replace_refs = parsed.get("replace") or []
    if not isinstance(replace_refs, list):
        replace_refs = []
    indices = [
        int(i) for i in replace_refs
        if isinstance(i, (int, float)) and 1 <= int(i) <= num_existing
    ]
    text = parsed.get("text")
    return {
        "action": action,
        "replace_indices": indices,
        "text": text.strip() if isinstance(text, str) and text.strip() else None,
    }


def consolidate_fact(
    llm,
    store,
    fact: str,
    session_id: str,
    tracer: Tracer | None = None,
) -> dict[str, Any]:
    """Integrate one fact into the store. Returns a report dict."""

    def trace(event: str, **fields) -> None:
        if tracer is not None:
            tracer.event(event, **fields)

    similar: list[MemoryItem] = store.search(
        fact, top_k=SIMILAR_TOP_K, min_score=SIMILAR_MIN_SCORE)

    if not similar:
        mem_id = store.add_memory(fact, kind=FACT_KIND, session_id=session_id)
        trace("fact_consolidated", action="add", fact=fact, memory_id=mem_id)
        return {"fact": fact, "action": "add", "memory_id": mem_id}

    decision = None
    try:
        raw = llm.complete(
            build_consolidation_prompt(fact, similar), purpose="consolidation")
        decision = _parse_decision(raw, len(similar))
        if decision is None:
            logger.warning(
                "Consolidation decision unparseable for %r; defaulting to add", fact)
    except LLMError as exc:
        logger.warning(
            "Consolidation decision failed for %r (%s); defaulting to add",
            fact, exc)
    if decision is None:
        decision = {"action": "add", "replace_indices": [], "text": None}

    action = decision["action"]
    if action == "skip":
        logger.info("[consolidation] skip (duplicate): %s", fact)
        trace("fact_consolidated", action="skip", fact=fact,
              duplicate_of=[m.id for m in similar])
        return {"fact": fact, "action": "skip"}

    replaced_ids: list[str] = []
    if action == "replace":
        replaced_ids = [similar[i - 1].id for i in decision["replace_indices"]]
        if replaced_ids:
            store.delete_memories(replaced_ids)
            logger.info(
                "[consolidation] replacing %d stale memories for: %s",
                len(replaced_ids), fact)

    text = decision["text"] or fact
    mem_id = store.add_memory(text, kind=FACT_KIND, session_id=session_id)
    trace("fact_consolidated", action=action, fact=fact,
          memory_id=mem_id, replaced_ids=replaced_ids, stored_text=text)
    return {
        "fact": fact,
        "action": action,
        "memory_id": mem_id,
        "replaced_ids": replaced_ids,
        "stored_text": text,
    }


def consolidate_facts(
    llm,
    store,
    facts: list[str],
    session_id: str,
    tracer: Tracer | None = None,
) -> list[dict[str, Any]]:
    """Integrate each extracted fact into long-term memory."""
    return [
        consolidate_fact(llm, store, fact, session_id, tracer)
        for fact in facts
    ]
