"""The per-turn pipeline: rewrite → retrieve → gate → assemble → respond.

Separated from the CLI so the whole flow is testable with fake LLM/store
doubles. Every step is logged and, when a tracer is attached, recorded as a
structured trace event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from personica.config import Settings
from personica.jsonutil import extract_json
from personica.llm import LLMError
from personica.memory.long_term import MemoryItem
from personica.prompts import (
    SYSTEM_PROMPT,
    build_fact_extraction_prompt,
    build_query_rewrite_prompt,
    build_relevance_prompt,
)
from personica.time_context import build_time_system_message
from personica.tracing import Tracer

logger = logging.getLogger(__name__)


def format_timestamp(timestamp_str: str) -> str:
    """Format an ISO UTC timestamp for display."""
    return timestamp_str[:19].replace("T", " ")


def build_memory_block(retrieved: list[MemoryItem]) -> str:
    """Render retrieved memories as a system message, tagging the newest [LATEST]."""
    most_recent = max(retrieved, key=lambda m: m.epoch)
    lines = []
    for m in retrieved:
        marker = " [LATEST]" if m is most_recent and len(retrieved) > 1 else ""
        lines.append(
            f"• {m.text}\n  [Stored: {format_timestamp(m.created_at_utc)}, Type: {m.kind}{marker}]"
        )
    block = "\n".join(lines)
    return (
        "═══ LONG-TERM MEMORIES (USE THESE!) ═══\n"
        f"{block}\n"
        "══════════════════════════════════════════════════\n"
        "These are facts about the user. Use them to answer questions.\n"
        "[LATEST] tag indicates the most recent memory - prefer it when facts conflict."
    )


@dataclass
class TurnResult:
    assistant: str
    retrieval_query: str
    retrieved: list[MemoryItem]
    memories_injected: bool

    def retrieved_display(self) -> list[str]:
        return [
            f"{m.text[:80]}... (score={m.score:.2f}, kind={m.kind}, "
            f"time={format_timestamp(m.created_at_utc)})"
            for m in self.retrieved
        ]


class TurnPipeline:
    def __init__(
        self,
        llm,
        store,
        short_term,
        settings: Settings,
        tracer: Tracer | None = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.short_term = short_term
        self.settings = settings
        self.tracer = tracer

    def _trace(self, event: str, **fields) -> None:
        if self.tracer is not None:
            self.tracer.event(event, **fields)

    def _search(self, query: str) -> list[MemoryItem]:
        return self.store.search(
            query,
            top_k=self.settings.retrieval_top_k,
            min_score=self.settings.retrieval_min_score,
            relevance_weight=self.settings.relevance_weight,
            recency_half_life_days=self.settings.recency_half_life_days,
        )

    def rewrite_query(self, user_msg: str, recent_context: str) -> str:
        """Rewrite the user query + context into an expanded retrieval key.

        Falls back to the raw user message if the rewrite fails.
        """
        try:
            return self.llm.complete(
                build_query_rewrite_prompt(user_msg, recent_context),
                purpose="query_rewrite",
            ).lower()
        except LLMError as exc:
            logger.warning("Query rewrite failed (%s); using the raw message", exc)
            self._trace("query_rewrite_failed", error=str(exc))
            return user_msg

    def check_relevance(
        self, user_msg: str, recent_context: str, memories: list[str],
    ) -> bool:
        """Gatekeeper: decide whether retrieved memories should be injected.

        Defaults to relevant if the check fails.
        """
        try:
            verdict = self.llm.complete(
                build_relevance_prompt(user_msg, recent_context, memories),
                purpose="relevance_check",
            )
        except LLMError as exc:
            logger.warning("Relevance check failed (%s); assuming relevant", exc)
            return True
        logger.info("[gatekeeper] relevance verdict: %s", verdict)
        return "IRRELEVANT" not in verdict.upper()

    def run(self, user: str) -> TurnResult:
        """Run one conversation turn. Raises LLMError if the reply fails."""
        if self.tracer is not None:
            self.tracer.new_turn()
        self._trace("turn_start", user_chars=len(user))

        recent_ctx = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in self.short_term.turns)

        # ── LLM-based query rewrite for long-term retrieval ──
        query = self.rewrite_query(user, recent_ctx)
        logger.info("[retrieval] query: %s", query)
        self._trace("query_rewrite", query=query)

        # ── Multi-query retrieval: union of the rewritten keyword query and
        # the raw message. A single keyword query biases toward one sub-topic
        # of a multi-part question; the raw phrasing balances differently.
        # Merge by id keeping the best-ranked copy, then take the top_k.
        kw_results = self._search(query)
        raw_results = self._search(user) if query != user else []
        by_id: dict[str, MemoryItem] = {}
        for m in kw_results + raw_results:
            if m.id not in by_id or m.rank_score > by_id[m.id].rank_score:
                by_id[m.id] = m
        retrieved = sorted(
            by_id.values(), key=lambda m: (m.rank_score, m.epoch), reverse=True,
        )[:self.settings.retrieval_top_k]
        self._trace(
            "retrieval",
            result_count=len(retrieved),
            keyword_query_hits=len(kw_results),
            raw_query_hits=len(raw_results),
            results=[
                {
                    "id": m.id,
                    "kind": m.kind,
                    "similarity": round(m.score, 3),
                    "rank_score": round(m.rank_score, 3),
                }
                for m in retrieved
            ],
        )

        # ── Relevance gatekeeper ──
        injected = False
        if retrieved:
            injected = self.check_relevance(
                user, recent_ctx, [m.text for m in retrieved])
            self._trace("gatekeeper", relevant=injected)
            if not injected:
                logger.info(
                    "[gatekeeper] memories deemed irrelevant; skipping injection")
        else:
            logger.info("[retrieval] no memories found")

        # ── Assemble messages ──
        self.short_term.add("user", user)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system",
             "content": build_time_system_message(self.settings.timezone)},
        ]
        if injected:
            messages.append(
                {"role": "system", "content": build_memory_block(retrieved)})
        messages.extend(self.short_term.as_messages())

        try:
            assistant = self.llm.chat(messages, temperature=0.2)
        except LLMError as exc:
            self._trace("turn_failed", error=str(exc))
            raise
        self.short_term.add("assistant", assistant)

        # Fold older turns into the rolling summary after the pair completes,
        # so a user/assistant pair is never split across summary and window.
        folded = self.short_term.maybe_summarize(
            lambda p: self.llm.complete(p, purpose="rolling_summary"))
        if folded:
            self._trace(
                "summary_folded", summary_chars=len(self.short_term.summary))

        self._trace(
            "turn_complete",
            assistant_chars=len(assistant),
            memories_injected=injected,
        )
        return TurnResult(
            assistant=assistant,
            retrieval_query=query,
            retrieved=retrieved,
            memories_injected=injected,
        )


def extract_session_facts(llm, full_log: list[dict[str, str]]) -> list[str]:
    """Extract new/updated user-stated facts from the session transcript.

    Returns a list of atomic fact strings; empty when there is nothing worth
    remembering, extraction fails, or the output cannot be parsed (the
    transcript itself is always saved separately, so nothing is lost).
    """
    transcript = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in full_log)
    try:
        raw = llm.complete(
            build_fact_extraction_prompt(transcript),
            use_utility_model=False,
            purpose="fact_extraction",
        )
    except LLMError as exc:
        logger.error("Fact extraction failed: %s", exc)
        return []

    parsed = extract_json(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list):
        logger.warning("Fact extraction returned unparseable output: %.200s", raw)
        return []
    return [f.strip() for f in parsed["facts"]
            if isinstance(f, str) and f.strip()]
