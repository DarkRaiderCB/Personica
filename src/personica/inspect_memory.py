"""Debug utility: dump every stored long-term memory with its metadata.

Run with `personica-inspect` or `python -m personica.inspect_memory`.
"""

from __future__ import annotations

import os

from personica.config import Settings
from personica.memory.long_term import ChromaMemoryStore


def main() -> None:
    settings = Settings.from_env()
    store = ChromaMemoryStore(
        persist_directory=os.path.join(settings.data_dir, "chroma"),
        embed_model=settings.embed_model,
    )

    total = store.count()
    print(f"{'=' * 80}")
    print(f"TOTAL MEMORIES STORED: {total}")
    print(f"{'=' * 80}\n")

    if total == 0:
        print("No memories found in storage.\n")
        return

    results = store.col.get(include=["documents", "metadatas"])
    for i, (mem_id, text, meta) in enumerate(
        zip(results["ids"], results["documents"],
            results["metadatas"], strict=True), 1,
    ):
        print(f"[{i}] ID: {mem_id[:16]}...")
        print(f"    Kind: {meta.get('kind', 'N/A')}")
        print(f"    Session: {str(meta.get('session_id', 'N/A'))[:16]}...")
        print(f"    Created: {meta.get('created_at_utc', 'N/A')}")
        print(f"    Epoch: {meta.get('epoch', 'N/A')}")
        print(f"    Text: {text}")
        print()


if __name__ == "__main__":
    main()
