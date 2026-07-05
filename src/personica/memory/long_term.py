"""Long-term memory backed by a persistent ChromaDB collection.

Documents are embedded with a sentence-transformers model and searched by
cosine similarity; each memory carries kind/session/timestamp metadata.

Retrieval ranks by a hybrid score blending semantic similarity with an
exponential recency decay (the retrieval formula from Park et al.,
"Generative Agents", 2023) so that fresher memories win among comparably
relevant ones without recency completely overriding relevance.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

DEFAULT_RELEVANCE_WEIGHT = 0.7
DEFAULT_RECENCY_HALF_LIFE_DAYS = 30.0


def hybrid_score(
    similarity: float,
    epoch: float,
    now_epoch: float,
    relevance_weight: float = DEFAULT_RELEVANCE_WEIGHT,
    recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
) -> float:
    """Blend semantic similarity with exponential recency decay.

    recency = 0.5 ** (age_days / half_life): a memory loses half its recency
    contribution every `recency_half_life_days`. Both components are in
    [0, 1], combined as a weighted sum.
    """
    age_days = max(0.0, now_epoch - epoch) / 86400.0
    if recency_half_life_days > 0:
        recency = 0.5 ** (age_days / recency_half_life_days)
    else:
        recency = 0.0
    return relevance_weight * similarity + (1.0 - relevance_weight) * recency


@dataclass
class MemoryItem:
    id: str
    text: str
    kind: str
    session_id: str
    created_at_utc: str
    epoch: float
    score: float  # raw cosine similarity in [0, 1]
    metadata: dict[str, Any]
    rank_score: float = field(default=0.0)  # hybrid similarity+recency score


class ChromaMemoryStore:
    """Long-term memory using Chroma PersistentClient.

    - stores documents + metadata
    - supports semantic search
    - persists to disk (persist_directory)

    An `embedding_fn` can be injected (e.g. a lightweight fake in tests);
    by default a sentence-transformers model is used.
    """

    def __init__(
        self,
        persist_directory: str = "./personica_data/chroma",
        collection_name: str = "personica_memories",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_fn: Any | None = None,
    ) -> None:
        os.makedirs(persist_directory, exist_ok=True)

        if embedding_fn is None:
            # Imported lazily: pulls in torch / sentence-transformers.
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=embed_model)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_fn
        self.col = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.col.count()

    def add_memory(
        self,
        text: str,
        kind: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("Cannot store an empty memory")

        now = datetime.now(UTC)
        metadata = dict(metadata or {})
        metadata["kind"] = kind
        metadata["session_id"] = session_id
        metadata["created_at_utc"] = now.isoformat()
        metadata["epoch"] = now.timestamp()

        mem_id = str(uuid.uuid4())
        self.col.add(
            ids=[mem_id],
            documents=[text.strip()],
            metadatas=[metadata],
        )
        logger.info("Stored %s memory %s (%d chars)", kind, mem_id, len(text))
        return mem_id

    def delete_memories(self, ids: list[str]) -> None:
        if not ids:
            return
        self.col.delete(ids=list(ids))
        logger.info("Deleted %d memories: %s", len(ids), ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.20,
        where: dict[str, Any] | None = None,
        relevance_weight: float = DEFAULT_RELEVANCE_WEIGHT,
        recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    ) -> list[MemoryItem]:
        """Semantic search over stored memories, ranked by hybrid score.

        Chroma returns cosine distances (lower = closer); these are converted
        to a similarity-like score = 1 - distance, clamped to [0, 1]. Results
        below `min_score` similarity are dropped. Survivors are re-ranked by
        `hybrid_score` (similarity blended with exponential recency decay,
        epoch as tie-break) and the top_k best are returned. A wider candidate
        pool than top_k is fetched so re-ranking has room to reorder.
        """
        total = self.col.count()
        if total == 0 or top_k < 1 or not query.strip():
            return []

        candidate_pool = min(total, max(top_k * 3, 10))
        res = self.col.query(
            query_texts=[query],
            n_results=candidate_pool,
            where=where,  # e.g. {"kind": "fact"}
            include=["documents", "metadatas", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        now_epoch = datetime.now(UTC).timestamp()
        out: list[MemoryItem] = []
        for mem_id, text, meta, dist in zip(ids, docs, metas, dists, strict=True):
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            if score < float(min_score):
                continue
            epoch = float(meta.get("epoch", 0.0))
            out.append(
                MemoryItem(
                    id=mem_id,
                    text=text,
                    kind=str(meta.get("kind", "")),
                    session_id=str(meta.get("session_id", "")),
                    created_at_utc=str(meta.get("created_at_utc", "")),
                    epoch=epoch,
                    score=score,
                    metadata=dict(meta),
                    rank_score=hybrid_score(
                        score, epoch, now_epoch,
                        relevance_weight, recency_half_life_days,
                    ),
                )
            )

        out.sort(key=lambda m: (m.rank_score, m.epoch), reverse=True)
        logger.debug(
            "Search %r matched %d/%d candidates above min_score=%.2f",
            query[:60], len(out), len(ids), min_score,
        )
        return out[:top_k]
