"""Short-term conversational memory.

Keeps the last N user/assistant pairs verbatim (a sliding window) and folds
anything older into a rolling LLM-generated summary so the prompt stays
bounded without losing context.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from personica.prompts import build_rolling_summary_prompt

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    role: str
    content: str


class ShortTermMemory:
    def __init__(self, keep_last_turns: int = 5) -> None:
        if keep_last_turns < 1:
            raise ValueError("keep_last_turns must be >= 1")
        self.keep_last_turns = keep_last_turns
        self.summary: str = ""
        self.turns: list[Turn] = []

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))

    def maybe_summarize(self, summarize_fn: Callable[[str], str]) -> bool:
        """Fold turns older than the sliding window into the rolling summary.

        Returns True if the summary was updated. If summarization fails, the
        old turns are kept so nothing is lost — the next call retries.
        """
        # keep_last_turns means "last N pairs", i.e. N*2 individual messages.
        window = self.keep_last_turns * 2
        if len(self.turns) <= window:
            return False

        old, recent = self.turns[:-window], self.turns[-window:]
        transcript = "\n".join(f"{t.role.upper()}: {t.content}" for t in old)
        try:
            new_summary = summarize_fn(
                build_rolling_summary_prompt(self.summary, transcript)).strip()
        except Exception as exc:
            logger.warning(
                "Rolling summarization failed; keeping %d older turns to retry later: %s",
                len(old), exc,
            )
            return False

        if not new_summary:
            return False

        self.summary = new_summary
        self.turns = recent
        logger.debug(
            "Folded %d older turns into the rolling summary (%d chars)",
            len(old), len(new_summary),
        )
        return True

    def as_messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self.summary:
            msgs.append({
                "role": "system",
                "content": f"Conversation summary so far:\n{self.summary}",
            })
        msgs.extend({"role": t.role, "content": t.content} for t in self.turns)
        return msgs
