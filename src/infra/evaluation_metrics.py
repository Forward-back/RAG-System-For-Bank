"""
RAG evaluation metrics — retrieval quality + per-stage latency.

Metrics:
  - Hit@k  — fraction of queries where a relevant chunk appears in the top-k
  - MRR    — mean reciprocal rank of the first relevant chunk
  - Stage-level latency percentiles (p50, p95, p99)
  - Anomaly detection: flag queries whose total_ms exceeds p95 + 2×IQR

Relevance is judged by substring match: a retrieved chunk is "relevant" if
its page_content contains ALL of the relevant_contents strings.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """
    Lightweight per-stage stopwatch.

    Usage::

        tracker = LatencyTracker()
        with tracker.stage("rewrite"):
            ...
        with tracker.stage("retrieval"):
            ...
        tracker.report()
    """

    def __init__(self):
        self._stages: Dict[str, List[float]] = defaultdict(list)
        self._current_stage: Optional[str] = None
        self._t0: float = 0.0

    def stage(self, name: str):
        """Context manager that records elapsed time for *name*."""
        return _StageContext(self, name)

    def record(self, name: str, elapsed_ms: float) -> None:
        self._stages[name].append(elapsed_ms)

    def start(self, name: str) -> None:
        self._current_stage = name
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._current_stage:
            elapsed = (time.perf_counter() - self._t0) * 1000
            self._stages[self._current_stage].append(elapsed)
            self._current_stage = None

    def clear(self) -> None:
        self._stages.clear()

    def summary(self) -> Dict[str, Any]:
        """Return per-stage stats: count, avg, p50, p95, p99, min, max."""
        result: Dict[str, Any] = {}
        stage_totals_per_query: Dict[int, float] = defaultdict(float)

        for stage_name, times in sorted(self._stages.items()):
            if not times:
                continue
            sorted_t = sorted(times)
            n = len(sorted_t)
            result[stage_name] = {
                "count": n,
                "avg":   round(statistics.mean(sorted_t), 1),
                "p50":   round(_percentile(sorted_t, 0.50), 1),
                "p95":   round(_percentile(sorted_t, 0.95), 1),
                "p99":   round(_percentile(sorted_t, 0.99), 1),
                "min":   round(sorted_t[0], 1),
                "max":   round(sorted_t[-1], 1),
            }
            for i, t in enumerate(times):
                stage_totals_per_query[i] += t

        if stage_totals_per_query:
            totals_sorted = sorted(stage_totals_per_query.values())
            n = len(totals_sorted)
            p50 = _percentile(totals_sorted, 0.50)
            p95 = _percentile(totals_sorted, 0.95)
            iqr = p95 - p50
            anomaly_threshold = p95 + 2 * iqr if iqr > 0 else p95 * 2

            result["_total"] = {
                "count": n,
                "avg":   round(statistics.mean(totals_sorted), 1),
                "p50":   round(p50, 1),
                "p95":   round(p95, 1),
                "p99":   round(_percentile(totals_sorted, 0.99), 1),
                "min":   round(totals_sorted[0], 1),
                "max":   round(totals_sorted[-1], 1),
                "anomaly_threshold_ms": round(anomaly_threshold, 1),
            }

        return result

    def anomalies(self) -> List[Tuple[int, float]]:
        """Return [(query_index, total_ms)] for queries whose total exceeds the anomaly threshold."""
        s = self.summary()
        total_info = s.get("_total", {})
        threshold = total_info.get("anomaly_threshold_ms", float("inf"))

        stage_totals_per_query: Dict[int, float] = defaultdict(float)
        for times in self._stages.values():
            for i, t in enumerate(times):
                stage_totals_per_query[i] += t

        return [
            (i, round(total, 1))
            for i, total in sorted(stage_totals_per_query.items())
            if total > threshold
        ]

    def report(self) -> str:
        """Return a formatted multi-line latency report."""
        s = self.summary()
        lines = ["=" * 65, "LATENCY REPORT", "=" * 65]

        total = s.pop("_total", {})
        stage_order = [
            "rewrite", "classification", "text_to_sql", "retrieval",
            "rerank", "compression", "generation", "fact_check", "other",
        ]

        for name in stage_order:
            if name in s:
                info = s[name]
                total_ms = info["avg"] * info["count"]
                pct = ""
                if total:
                    denom = total.get("avg", 1) * total.get("count", 1)
                    if denom > 0:
                        pct = f"({total_ms / denom * 100:.0f}%)"
                lines.append(
                    f"  {name:<18} avg={info['avg']:>7.1f}ms  "
                    f"p50={info['p50']:>7.1f}ms  p95={info['p95']:>7.1f}ms  "
                    f"p99={info['p99']:>7.1f}ms  {pct}"
                )

        for name, info in s.items():
            lines.append(
                f"  {name:<18} avg={info['avg']:>7.1f}ms  "
                f"p50={info['p50']:>7.1f}ms  p95={info['p95']:>7.1f}ms"
            )

        if total:
            lines.append("  " + "-" * 55)
            lines.append(
                f"  {'TOTAL':<18} avg={total['avg']:>7.1f}ms  "
                f"p50={total['p50']:>7.1f}ms  p95={total['p95']:>7.1f}ms  "
                f"p99={total['p99']:>7.1f}ms"
            )
            anomalies = self.anomalies()
            if anomalies:
                lines.append(
                    f"\n  ANOMALIES ({len(anomalies)} queries over "
                    f"{total['anomaly_threshold_ms']:.0f}ms):"
                )
                for idx, ms in anomalies[:10]:
                    lines.append(f"    query #{idx}: {ms:.0f}ms")

        lines.append("=" * 65)
        return "\n".join(lines)


class _StageContext:
    def __init__(self, tracker: LatencyTracker, name: str):
        self.tracker = tracker
        self.name = name
        self.t0: float = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.t0) * 1000
        self.tracker.record(self.name, elapsed)


def _percentile(sorted_data: List[float], p: float) -> float:
    """Return the p-th percentile of sorted_data (linear interpolation)."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    k = (n - 1) * p
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]


# ---------------------------------------------------------------------------
# Retrieval evaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """
    Evaluate retrieval quality against a ground-truth set.

    Usage::

        gt = RetrievalEvaluator.load_ground_truth("eval_queries.json")
        evaluator = RetrievalEvaluator(gt)
        report = evaluator.evaluate(pipeline, k_values=[3, 5])
    """

    def __init__(self, ground_truth: List[dict]):
        self.ground_truth = ground_truth

    # ------------------------------------------------------------------
    # Ground-truth loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_ground_truth(path: str) -> List[dict]:
        """Load ground-truth queries from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, item in enumerate(data):
            if "query" not in item:
                raise ValueError(f"Item {i}: missing 'query'")
            if "relevant_contents" not in item:
                raise ValueError(f"Item {i}: missing 'relevant_contents'")
        return data

    @staticmethod
    def save_ground_truth(gt: List[dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Relevance judgment
    # ------------------------------------------------------------------

    @staticmethod
    def is_relevant(doc: Document, relevant_contents: List[str]) -> bool:
        """
        A chunk is relevant if its page_content contains ALL of the
        relevant_contents substrings (case-insensitive).
        """
        content_lower = doc.page_content.lower()
        return all(
            term.lower() in content_lower
            for term in relevant_contents
        )

    @staticmethod
    def find_first_rank(
        docs: List[Document],
        relevant_contents: List[str],
    ) -> Optional[int]:
        """Return the 1-based rank of the first relevant document, or None."""
        for rank, doc in enumerate(docs, start=1):
            if RetrievalEvaluator.is_relevant(doc, relevant_contents):
                return rank
        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pipeline,
        k_values: Tuple[int, ...] = (3, 5, 10),
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all ground-truth queries through the pipeline's retrieval and
        compute Hit@k and MRR.
        """
        hits: Dict[int, int] = {k: 0 for k in k_values}
        reciprocal_ranks: List[float] = []
        per_query: List[dict] = []
        n = len(self.ground_truth)

        for i, item in enumerate(self.ground_truth):
            query = item["query"]
            relevant = item["relevant_contents"]

            retrieved = pipeline.retrieve_for_evaluation(
                query, k=max(k_values)
            )
            docs = [doc for doc, _ in retrieved] if retrieved else []

            rank = self.find_first_rank(docs, relevant)
            rr = 1.0 / rank if rank else 0.0
            reciprocal_ranks.append(rr)

            for k in k_values:
                if rank is not None and rank <= k:
                    hits[k] += 1

            per_query.append({
                "query": query,
                "first_rank": rank,
                "rr": round(rr, 4),
                "hit@3": rank is not None and rank <= 3,
                "hit@5": rank is not None and rank <= 5,
            })

            if verbose and (i + 1) % 10 == 0:
                print(f"  ... evaluated {i + 1}/{n} queries")

        report = {
            "num_queries": n,
            "metrics": {},
            "per_query": per_query,
        }

        for k in k_values:
            report["metrics"][f"hit@{k}"] = round(hits[k] / n, 4) if n else 0.0
        report["metrics"]["mrr"] = round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0

        return report

    @staticmethod
    def report_text(report: Dict[str, Any]) -> str:
        """Format an evaluation report as a readable string."""
        m = report["metrics"]
        n = report["num_queries"]
        lines = [
            "=" * 50,
            "RETRIEVAL QUALITY REPORT",
            "=" * 50,
            f"  Queries evaluated: {n}",
        ]
        for metric, value in m.items():
            if "hit" in metric:
                k = metric.split("@")[1]
                lines.append(f"  {metric:<10} {value:.2%}  ({int(value * n)}/{n})")
            else:
                lines.append(f"  {metric:<10} {value:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined evaluation runner
# ---------------------------------------------------------------------------

class EvaluationRunner:
    """
    Run retrieval evaluation + latency tracking in one pass.

    Usage::

        runner = EvaluationRunner(ground_truth)
        report = runner.run(pipeline, k_values=(3, 5))
        print(report["quality_text"])
        print(report["latency_text"])
    """

    def __init__(self, ground_truth: List[dict]):
        self.evaluator = RetrievalEvaluator(ground_truth)

    def run(
        self,
        pipeline,
        k_values: Tuple[int, ...] = (3, 5, 10),
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run both quality and latency evaluation."""
        tracker = LatencyTracker()

        if verbose:
            print("\n[EVAL] Running retrieval quality evaluation...")

        if verbose:
            print("[EVAL] Running latency benchmarks...")

        sample_for_latency = self.evaluator.ground_truth[:min(15, len(self.evaluator.ground_truth))]

        for item in sample_for_latency:
            query = item["query"]
            t0 = time.perf_counter()

            try:
                pipeline.run(query)
            except Exception as e:
                if verbose:
                    print(f"  [WARN] query '{query[:30]}...' failed: {e}")

            total_ms = (time.perf_counter() - t0) * 1000
            tracker.record("end_to_end", total_ms)

        quality_report = self.evaluator.evaluate(
            pipeline, k_values=k_values, verbose=verbose
        )

        return {
            "quality": quality_report,
            "quality_text": RetrievalEvaluator.report_text(quality_report),
            "latency": tracker.summary(),
            "latency_text": tracker.report(),
            "anomalies": tracker.anomalies(),
        }
