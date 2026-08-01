#!/usr/bin/env python3
"""
Analyse RAG pipeline trace logs.

Reads one or more traces_*.jsonl files and prints a summary report.
Usage::

    python scripts/analyze_traces.py ./logs/               # latest day
    python scripts/analyze_traces.py ./logs/traces_2026-07-16.jsonl
    python scripts/analyze_traces.py ./logs/ --days 7      # last 7 days
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_traces(paths: List[Path]) -> List[dict]:
    """Load all JSONL trace files into a list of dicts."""
    traces: List[dict] = []
    for p in paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return traces


def compute_stats(traces: List[dict]) -> Dict[str, Any]:
    """Compute aggregate statistics from trace records."""
    if not traces:
        return {}

    total = len(traces)

    # ── Basic counts ──
    cache_hits = sum(1 for t in traces if t.get("cache_hit"))
    rejections = sum(1 for t in traces if t.get("answer", {}).get("is_rejection"))
    sql_used = sum(1 for t in traces if t.get("sql", {}).get("used"))

    # ── Latency (ms) ──
    latencies = [t.get("latency_ms", {}).get("total", 0) for t in traces]
    latencies_sorted = sorted(latencies)
    avg_latency = sum(latencies) / total if total else 0
    p50 = latencies_sorted[int(total * 0.50)] if total else 0
    p95 = latencies_sorted[int(total * 0.95)] if total else 0
    p99 = latencies_sorted[int(total * 0.99)] if total else 0

    # Stage breakdown
    stage_totals: Dict[str, float] = defaultdict(float)
    for t in traces:
        for stage, ms in t.get("latency_ms", {}).items():
            if stage != "total":
                stage_totals[stage] += ms
    stage_avg = {k: round(v / total, 1) for k, v in stage_totals.items()}

    # ── Breakdown by classification ──
    risk_dist: Counter = Counter()
    doctype_dist: Counter = Counter()
    action_dist: Counter = Counter()
    for t in traces:
        cls = t.get("classification") or {}
        if cls.get("risk_level"):
            risk_dist[cls["risk_level"]] += 1
        if cls.get("doc_type"):
            doctype_dist[cls["doc_type"]] += 1
        if cls.get("action"):
            action_dist[cls["action"]] += 1

    # ── Rejection rate by doc_type ──
    rejection_by_doctype: Dict[str, float] = {}
    for dt in doctype_dist:
        dt_total = sum(1 for t in traces
                       if (t.get("classification") or {}).get("doc_type") == dt)
        dt_reject = sum(1 for t in traces
                        if (t.get("classification") or {}).get("doc_type") == dt
                        and t.get("answer", {}).get("is_rejection"))
        rejection_by_doctype[dt] = round(dt_reject / dt_total * 100, 1) if dt_total else 0

    # ── Latency by risk_level ──
    latency_by_risk: Dict[str, float] = {}
    for rl in risk_dist:
        rl_lats = [t.get("latency_ms", {}).get("total", 0) for t in traces
                   if (t.get("classification") or {}).get("risk_level") == rl]
        latency_by_risk[rl] = round(sum(rl_lats) / len(rl_lats), 1) if rl_lats else 0

    # ── Safety cascade ──
    safety_stats = {
        "L1_triggered": sum(1 for t in traces
                            if t.get("safety", {}).get("L1_flags", 0) > 0),
        "L2_avg_unsupported": sum(t.get("safety", {}).get("L2_unsupported_ratio", 0)
                                  for t in traces) / total if total else 0,
        "L3_triggered": sum(1 for t in traces
                            if t.get("safety", {}).get("L3_triggered")),
    }

    # ── Top slowest queries ──
    slowest = sorted(traces, key=lambda t: t.get("latency_ms", {}).get("total", 0),
                     reverse=True)[:10]
    slowest_queries = [
        {"query": t["query"][:80],
         "latency_ms": t.get("latency_ms", {}).get("total", 0)}
        for t in slowest
    ]

    # ── Top rejection queries ──
    rejection_traces = [t for t in traces if t.get("answer", {}).get("is_rejection")]
    top_rejections = Counter(
        t["query"] for t in rejection_traces
    ).most_common(10)

    return {
        "total_queries": total,
        "cache_hit_rate": round(cache_hits / total * 100, 1),
        "rejection_rate": round(rejections / total * 100, 1),
        "sql_usage_rate": round(sql_used / total * 100, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "stage_avg_ms": stage_avg,
        "risk_distribution": dict(risk_dist.most_common()),
        "doctype_distribution": dict(doctype_dist.most_common()),
        "action_distribution": dict(action_dist.most_common()),
        "rejection_by_doctype_pct": rejection_by_doctype,
        "latency_by_risk_ms": latency_by_risk,
        "safety": safety_stats,
        "slowest_queries": slowest_queries,
        "top_rejection_queries": top_rejections,
    }


def print_report(stats: Dict[str, Any]) -> None:
    """Print a human-readable summary report."""
    if not stats:
        print("No trace data found.")
        return

    print("=" * 60)
    print("  RAG Pipeline Trace Analysis")
    print("=" * 60)

    print(f"\n  Total queries:       {stats['total_queries']}")
    print(f"  Cache hit rate:      {stats['cache_hit_rate']}%")
    print(f"  Rejection rate:      {stats['rejection_rate']}%")
    print(f"  SQL usage rate:      {stats['sql_usage_rate']}%")

    print(f"\n── Latency (ms) ──")
    print(f"  Avg:  {stats['avg_latency_ms']}")
    print(f"  P50:  {stats['p50_latency_ms']}")
    print(f"  P95:  {stats['p95_latency_ms']}")
    print(f"  P99:  {stats['p99_latency_ms']}")

    stage = stats.get("stage_avg_ms", {})
    if stage:
        print(f"\n── Stage breakdown (avg ms) ──")
        for name in sorted(stage):
            print(f"  {name:20s} {stage[name]:8.1f}")

    risk = stats.get("risk_distribution", {})
    if risk:
        print(f"\n── Risk level distribution ──")
        for k, v in sorted(risk.items()):
            lats = stats.get("latency_by_risk_ms", {}).get(k, 0)
            print(f"  {k:12s} {v:5d}  (avg {lats}ms)")

    dt = stats.get("doctype_distribution", {})
    if dt:
        print(f"\n── Doc type distribution ──")
        for k, v in sorted(dt.items()):
            rej = stats.get("rejection_by_doctype_pct", {}).get(k, 0)
            print(f"  {k:14s} {v:5d}  (rejection rate {rej}%)")

    action = stats.get("action_distribution", {})
    if action:
        print(f"\n── Action distribution ──")
        for k, v in sorted(action.items()):
            print(f"  {k:30s} {v:5d}")

    safety = stats.get("safety", {})
    if safety:
        print(f"\n── Safety cascade ──")
        print(f"  L1 (NumericGuard) triggered:  {safety['L1_triggered']} queries")
        print(f"  L2 (NLIVerifier) avg ratio:   {safety['L2_avg_unsupported']:.3f}")
        print(f"  L3 (FactChecker) triggered:   {safety['L3_triggered']} queries")

    slow = stats.get("slowest_queries", [])
    if slow:
        print(f"\n── Top 5 slowest queries ──")
        for i, s in enumerate(slow[:5], 1):
            print(f"  {i}. [{s['latency_ms']:.0f}ms] {s['query']}")

    rej = stats.get("top_rejection_queries", [])
    if rej:
        print(f"\n── Top 5 rejection queries ──")
        for i, (q, n) in enumerate(rej[:5], 1):
            print(f"  {i}. (x{n}) {q[:80]}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyse RAG trace logs")
    parser.add_argument(
        "path", nargs="?", default="./logs",
        help="Path to traces file or directory (default: ./logs)"
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="Number of recent days to load (default: 1)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output stats as JSON instead of a human-readable report"
    )
    args = parser.parse_args()

    base = Path(args.path)

    # Resolve files
    if base.is_dir():
        files = sorted(base.glob("traces_*.jsonl"), reverse=True)[:args.days]
    else:
        files = [base]

    if not files:
        print(f"No trace files found in {args.path}", file=sys.stderr)
        sys.exit(1)

    traces = load_traces(files)
    stats = compute_stats(traces)

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print(f"\nSource: {len(files)} file(s)")
        print_report(stats)


if __name__ == "__main__":
    main()
