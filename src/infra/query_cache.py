"""
Semantic query cache for FAQ-style banking queries.

Uses embedding similarity to match semantically equivalent queries
(e.g. "几点开门" ≈ "营业时间是什么"), avoiding full pipeline re-runs
for answers that rarely change.

Cache entries are persisted to disk so they survive restarts.
Invalidated on index rebuild (documents changed → answers may change).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from src.infra.model_registry import get_embedding_model


class QueryCache:
    """Semantic cache keyed by query embedding cosine similarity.

    Usage::

        cache = QueryCache("./cache/query_cache.json")
        hit = cache.lookup("营业时间是几点")
        if hit:
            return hit  # ~50ms
        # ... run full pipeline ...
        cache.store("营业时间是几点", answer)
    """

    def __init__(
        self,
        cache_path: str = "./cache/query_cache.json",
        similarity_threshold: float = 0.92,
        max_entries: int = 500,
    ):
        self.cache_path = Path(cache_path)
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

        self._entries: List[dict] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, dim)

        # Shared embedding model (single instance across whole pipeline)
        self.model = get_embedding_model()

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, query: str) -> Optional[str]:
        """Return cached answer if a semantically similar query exists.

        Returns None on miss (cache empty, or no entry above threshold).
        """
        if not query or not query.strip():
            return None

        if self._embeddings is None or len(self._entries) == 0:
            return None

        q_vec = self.model.encode(
            [query.strip()], normalize_embeddings=True
        )[0]

        sims = np.dot(self._embeddings, q_vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.similarity_threshold:
            entry = self._entries[best_idx]
            entry["hit_count"] = entry.get("hit_count", 0) + 1
            entry["last_hit"] = time.time()
            self._save()
            return entry["answer"]

        return None

    def store(self, query: str, answer: str) -> None:
        """Persist a query-answer pair for future lookups."""
        if not query or not query.strip():
            return
        if not answer or not answer.strip():
            return

        q_vec = self.model.encode(
            [query.strip()], normalize_embeddings=True
        )[0]

        entry = {
            "query": query.strip(),
            "answer": answer.strip(),
            "embedding": q_vec.tolist(),
            "hit_count": 0,
            "created_at": time.time(),
            "last_hit": None,
        }

        self._entries.append(entry)

        if self._embeddings is None:
            self._embeddings = np.array([q_vec])
        else:
            self._embeddings = np.vstack([self._embeddings, q_vec])

        # LRU eviction: drop the least recently hit entry
        if len(self._entries) > self.max_entries:
            self._evict_one()

        self._save()

    def clear(self) -> None:
        """Invalidate all cache entries (call after index rebuild)."""
        self._entries = []
        self._embeddings = None
        self._save()

    @property
    def size(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        """Remove the least recently hit (or never-hit) entry."""
        if not self._entries:
            return

        # Sort by last_hit (None = never hit, oldest first)
        def _sort_key(i: int) -> float:
            lh = self._entries[i].get("last_hit")
            return lh if lh is not None else 0.0

        worst_idx = min(range(len(self._entries)), key=_sort_key)
        self._entries.pop(worst_idx)
        self._embeddings = np.delete(self._embeddings, worst_idx, axis=0)

    def _load(self) -> None:
        """Restore cache from disk."""
        if not self.cache_path.exists():
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        entries = data.get("entries", [])
        if not entries:
            return

        embeddings = []
        for e in entries:
            emb = e.get("embedding", [])
            if len(emb) > 0:
                embeddings.append(emb)
            else:
                # Entry without embedding — re-encode
                vec = self.model.encode(
                    [e["query"]], normalize_embeddings=True
                )[0]
                embeddings.append(vec.tolist())
                e["embedding"] = vec.tolist()

        self._entries = entries
        self._embeddings = np.array(embeddings, dtype=np.float32)

    def _save(self) -> None:
        """Persist cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "entries": self._entries,
        }

        # Write to temp file, then atomic rename
        tmp = self.cache_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.cache_path)
