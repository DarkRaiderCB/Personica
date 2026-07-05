"""Traceability: per-session JSONL pipeline traces and transcript records.

Every session produces:
- a trace file   (<data_dir>/traces/<session_id>.jsonl) — one JSON event per
  pipeline step (query rewrite, retrieval, gatekeeper, LLM calls, ...)
- a transcript   (<data_dir>/transcripts/<session_id>.json) — the full
  conversation, saved at session end for auditing.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class Tracer:
    """Appends structured JSONL events for one session.

    Never raises: tracing failures are logged and swallowed so traceability
    problems can't take down the assistant.
    """

    def __init__(self, trace_dir: str, session_id: str, enabled: bool = True) -> None:
        self.session_id = session_id
        self.turn = 0
        self.enabled = enabled
        self.path: str | None = None
        if enabled:
            try:
                os.makedirs(trace_dir, exist_ok=True)
                self.path = os.path.join(trace_dir, f"{session_id}.jsonl")
            except OSError as exc:
                logger.warning("Could not create trace directory: %s", exc)
                self.enabled = False

    def new_turn(self) -> int:
        self.turn += 1
        return self.turn

    def event(self, event: str, **fields) -> None:
        if not self.enabled or self.path is None:
            return
        record = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "turn": self.turn,
            "event": event,
            **fields,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("Failed to write trace event %r: %s", event, exc)


def save_transcript(
    transcript_dir: str, session_id: str, turns: list[dict[str, str]],
) -> str:
    """Save the complete conversation as JSON for auditing. Returns the path."""
    os.makedirs(transcript_dir, exist_ok=True)
    path = os.path.join(transcript_dir, f"{session_id}.json")
    payload = {
        "session_id": session_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "turns": turns,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
