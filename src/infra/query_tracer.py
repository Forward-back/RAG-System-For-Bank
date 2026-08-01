"""
Structured query tracing for the RAG pipeline.

Collects data at key pipeline stages and flushes one JSON line per query
to a date-rotated file.  Thread-safe via a module-level write lock.

Output format (one line per query)::

    {
      "timestamp": "2026-07-16T14:30:00.123456",
      "query": "转账手续费是多少",
      "classification": {"risk_level": "caution", "doc_type": "product", ...},
      "cache_hit": false,
      "rewritten_query": null,
      "sql": {"used": false},
      "retrieval": {"num_raw": 15, "num_unique": 8, ...},
      "answer": {"length": 342, "is_rejection": false},
      "safety": {"L1_flags": 0, "L2_unsupported_ratio": 0.0, "L3_triggered": false},
      "latency_ms": {"total": 1240, "retrieval": 180, "generation": 450, "safety": 85}
    }
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Module-level lock — all QueryTracer instances in the same process share it
_write_lock = threading.Lock()

# How many chars of the answer to store (enough for analysis, not the full text)
_MAX_ANSWER_CHARS = 500


class QueryTracer:
    """Collect structured telemetry for a single query and flush to disk.

    Usage::

        tracer = QueryTracer(query="转账手续费", logs_dir="./logs")
        tracer.classification = {"risk_level": "caution", ...}
        tracer.cache_hit = True
        tracer.flush(answer_text)
    """

    def __init__(
        self,
        query: str,
        logs_dir: str = "./logs",
        enabled: bool = True,
    ):
        self._query = query.strip()[:200] if query else ""
        self._logs_dir = Path(logs_dir)
        self._enabled = enabled
        self._t0 = time.perf_counter()

        # Public fields — set during pipeline execution
        self.classification: Optional[Dict[str, Any]] = None
        self.cache_hit: bool = False
        self.rewritten_query: Optional[str] = None
        self.sql: Dict[str, Any] = {"used": False}
        self.retrieval: Dict[str, Any] = {}
        self.safety: Dict[str, Any] = {}

        # Stage latencies (filled by record_stage)
        self._stage_times: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Stage latency
    # ------------------------------------------------------------------

    def record_stage(self, name: str, elapsed_ms: float) -> None:
        """Record elapsed time for a named pipeline stage."""
        self._stage_times[name] = round(elapsed_ms, 1)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self, answer: str) -> None:
        """Write the complete trace as one JSON line to the log file."""
        if not self._enabled:
            return

        total_ms = (time.perf_counter() - self._t0) * 1000

        trace: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": self._query,
            "classification": self.classification,
            "cache_hit": self.cache_hit,
            "rewritten_query": self.rewritten_query,
            "sql": self.sql,
            "retrieval": self.retrieval,
            "answer": {
                "length": len(answer) if answer else 0,
                "is_rejection": self._is_rejection(answer),
                "preview": (answer or "")[:_MAX_ANSWER_CHARS],
            },
            "safety": self.safety,
            "latency_ms": {
                "total": round(total_ms, 1),
                **self._stage_times,
            },
        }

        self._write_line(trace)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rejection(answer: str) -> bool:
        if not answer:
            return True
        patterns = [
            "No relevant documents found",
            "I cannot find this information",
            "未查询到符合条件",
            "很抱歉",
            "找不到",
            "无法查询到",
        ]
        return any(p in answer for p in patterns)

    def _write_line(self, trace: dict) -> None:
        """Thread-safe append of one JSON line to today's trace file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._logs_dir / f"traces_{date_str}.jsonl"

        line = json.dumps(trace, ensure_ascii=False) + "\n"

        with _write_lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(line)
