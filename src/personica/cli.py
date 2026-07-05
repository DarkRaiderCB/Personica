"""Personica interactive CLI: the REPL and session lifecycle."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import uuid

from personica import __version__
from personica.config import Settings
from personica.consolidation import consolidate_facts
from personica.llm import LLMClient, LLMError
from personica.logging_setup import setup_logging
from personica.memory.long_term import ChromaMemoryStore
from personica.memory.short_term import ShortTermMemory
from personica.pipeline import TurnPipeline, TurnResult, extract_session_facts
from personica.tracing import Tracer, save_transcript

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="personica",
        description="Personica — a personal AI assistant with short- and long-term memory.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete all stored long-term memories before starting.")
    parser.add_argument(
        "--version", action="version", version=f"personica {__version__}")
    return parser.parse_args()


def reset_long_term_memory(data_dir: str) -> None:
    chroma_dir = os.path.join(data_dir, "chroma")
    if os.path.isdir(chroma_dir):
        shutil.rmtree(chroma_dir)
        print(f"[reset] Cleared all long-term memories from {chroma_dir}\n")
    else:
        print("[reset] No memories found to clear.\n")


def handle_remember(store: ChromaMemoryStore, tracer: Tracer,
                    session_id: str, text: str) -> None:
    if not text:
        print("Usage: /remember <fact to store>\n")
        return
    mem_id = store.add_memory(text, kind="user_note", session_id=session_id)
    tracer.event("memory_stored", memory_id=mem_id, kind="user_note",
                 source="explicit_command")
    print(f"[memory] Remembered: {text}\n")


def handle_forget(store: ChromaMemoryStore, tracer: Tracer, query: str) -> None:
    if not query:
        print("Usage: /forget <what to forget>\n")
        return
    matches = store.search(query, top_k=3, min_score=0.20)
    if not matches:
        print("[memory] No matching memories found.\n")
        return
    print("Matching memories:")
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m.text}")
    choice = input("Forget which? [numbers/all/n]: ").strip().lower()
    if choice in ("", "n", "no"):
        print("[memory] Nothing forgotten.\n")
        return
    if choice == "all":
        selected = matches
    else:
        indices = {int(tok) for tok in choice.replace(",", " ").split()
                   if tok.isdigit() and 1 <= int(tok) <= len(matches)}
        selected = [matches[i - 1] for i in sorted(indices)]
    if not selected:
        print("[memory] Nothing forgotten.\n")
        return
    store.delete_memories([m.id for m in selected])
    tracer.event("memory_forgotten", ids=[m.id for m in selected],
                 texts=[m.text for m in selected])
    print(f"[memory] Forgot {len(selected)} memory(ies).\n")


def print_memory_debug(
    last_result: TurnResult | None, stm: ShortTermMemory,
) -> None:
    if last_result is None or not last_result.retrieved:
        print("(No retrieved memories last turn.)\n")
    else:
        print("Last retrieved memories:")
        for item in last_result.retrieved_display():
            print(f"  - {item}")
        print()
    if stm.summary:
        print("Short-term summary (older turns):")
        print(f"  {stm.summary}\n")


def finalize_session(
    settings: Settings,
    store: ChromaMemoryStore,
    llm: LLMClient,
    tracer: Tracer,
    session_id: str,
    full_log: list[dict[str, str]],
) -> None:
    """Persist the transcript and consolidate new facts into long-term memory."""
    tracer.event(
        "session_end",
        turns_logged=len(full_log),
        prompt_tokens=llm.total_prompt_tokens,
        completion_tokens=llm.total_completion_tokens,
        cost_usd=round(llm.total_cost_usd, 6),
    )
    if not full_log:
        return

    path = save_transcript(
        os.path.join(settings.data_dir, "transcripts"), session_id, full_log)
    print(f"[session] Transcript saved to {path}")

    facts = extract_session_facts(llm, full_log)
    if not facts:
        tracer.event("memory_skipped", reason="nothing_notable_or_error")
        print("[session] Nothing notable to store in long-term memory.")
    else:
        reports = consolidate_facts(llm, store, facts, session_id, tracer)
        print(f"[session] Consolidated {len(facts)} fact(s) into long-term memory:")
        for r in reports:
            suffix = ""
            if r.get("replaced_ids"):
                suffix = f" (superseded {len(r['replaced_ids'])} stale memories)"
            print(f"  - [{r['action']}] {r['fact']}{suffix}")

    if llm.total_prompt_tokens or llm.total_completion_tokens:
        print(
            f"[session] LLM usage: {llm.total_prompt_tokens} prompt + "
            f"{llm.total_completion_tokens} completion tokens"
            f" (~${llm.total_cost_usd:.4f})"
        )


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    session_id = (
        os.getenv("PERSONICA_SESSION_ID")
        or os.getenv("ASSISTANT_SESSION_ID")
        or str(uuid.uuid4())
    )
    setup_logging(settings.log_level, settings.data_dir, session_id)

    if args.reset:
        reset_long_term_memory(settings.data_dir)

    tracer = Tracer(os.path.join(settings.data_dir, "traces"), session_id)
    llm = LLMClient(settings, tracer=tracer)
    if not llm.is_configured:
        print(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        sys.exit(1)

    store = ChromaMemoryStore(
        persist_directory=os.path.join(settings.data_dir, "chroma"),
        embed_model=settings.embed_model,
    )
    stm = ShortTermMemory(keep_last_turns=settings.keep_last_turns)
    pipeline = TurnPipeline(
        llm=llm, store=store, short_term=stm, settings=settings, tracer=tracer)

    full_log: list[dict[str, str]] = []  # complete audit trail
    last_result: TurnResult | None = None

    print(f"Personica {__version__} — session {session_id}")
    print("Commands: /exit, /mem (last retrieved), "
          "/remember <fact>, /forget <query>\n")
    tracer.event(
        "session_start",
        chat_model=settings.chat_model,
        utility_model=settings.utility_model,
        memories_stored=store.count(),
    )

    try:
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user:
                continue
            command = user.lower()
            if command == "/exit":
                break
            if command == "/mem":
                print_memory_debug(last_result, stm)
                continue
            if command.startswith("/remember"):
                handle_remember(
                    store, tracer, session_id, user[len("/remember"):].strip())
                continue
            if command.startswith("/forget"):
                handle_forget(store, tracer, user[len("/forget"):].strip())
                continue

            full_log.append({"role": "user", "content": user})
            try:
                last_result = pipeline.run(user)
            except LLMError as exc:
                logger.error("Could not generate a response: %s", exc)
                print("Assistant: (sorry, I could not reach the language "
                      "model — please try again)\n")
                continue
            full_log.append(
                {"role": "assistant", "content": last_result.assistant})
            print(f"\nAssistant: {last_result.assistant}\n")
    finally:
        # Runs on /exit, Ctrl+C, and Ctrl+D alike, so no session is lost.
        finalize_session(settings, store, llm, tracer, session_id, full_log)


if __name__ == "__main__":
    main()
