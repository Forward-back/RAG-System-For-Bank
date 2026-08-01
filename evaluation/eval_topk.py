"""
top_k ablation experiment: retrieval quality at k=5,10,15,20.
"""
import json
import os
import sys
import time
from statistics import mean

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from evaluation.eval_baseline import (
    load_eval_dataset, DenseOnlyRetriever, compute_hit_at_k, compute_mrr,
    get_page,
)


def run_topk_ablation(retriever, dataset, k):
    """Evaluate with specific k, computing hit@k and MRR."""
    hit_k = 0
    mrr_values = []
    latencies = []
    per_query = []

    for item in dataset:
        query = item["query"]
        gt_pages = item.get("ground_truth_pages", [])

        t0 = time.perf_counter()
        results = retriever.retrieve(query, k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        docs = [doc for doc, _ in results] if results else []
        hk = compute_hit_at_k(docs, gt_pages, k)
        mrr = compute_mrr(docs, gt_pages)

        if hk:
            hit_k += 1
        mrr_values.append(mrr)

        per_query.append({
            "query": query[:60],
            "type": item["type"],
            f"hit@{k}": hk,
            "mrr": round(mrr, 3),
            "latency_ms": round(elapsed_ms, 1),
            "pages": [get_page(d) for d in docs],
        })

    n = len(dataset)
    lats = sorted(latencies)
    return {
        "k": k,
        "hit@k": round(hit_k / n, 3),
        "mrr": round(mean(mrr_values), 3) if mrr_values else 0,
        "latency_avg_ms": round(mean(lats), 1),
        "latency_p50_ms": round(lats[n // 2], 1),
        "latency_p95_ms": round(lats[int(n * 0.95)], 1),
        "per_query": per_query,
    }


def main():
    os.environ.setdefault("TESSDATA_PREFIX", os.path.join(BASE_DIR, "tessdata"))

    dataset = load_eval_dataset()
    persist_dir = os.path.join(BASE_DIR, "chroma_db")
    retriever = DenseOnlyRetriever(persist_dir=persist_dir)

    print(f"\n{'='*65}")
    print(f"  TOP-K ABLATION EXPERIMENT")
    print(f"  k = 5, 10, 15, 20 | Dense-only (ChromaDB cosine)")
    print(f"{'='*65}\n")

    results = []
    for k in [5, 10, 15, 20]:
        r = run_topk_ablation(retriever, dataset, k)
        results.append(r)
        print(f"  k={r['k']:<3}  Hit@{r['k']}={r['hit@k']:.3f}  MRR={r['mrr']:.3f}  "
              f"Lat(avg)={r['latency_avg_ms']:.1f}ms  p95={r['latency_p95_ms']:.1f}ms")

    # Comparison table
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"  {'k':<6} {'Hit@k':<8} {'MRR':<8} {'Lat(avg)':<12} {'Lat(p95)':<12}")
    print(f"  {'-'*46}")
    for r in results:
        print(f"  {r['k']:<6} {r['hit@k']:<8.3f} {r['mrr']:<8.3f} "
              f"{r['latency_avg_ms']:<12.1f} {r['latency_p95_ms']:<12.1f}")

    # Per-query breakdown at each k
    print(f"\n  Per-query Hit@k:")
    header = f"  {'Q#':<4} {'Type':<10}"
    for k in [5, 10, 15, 20]:
        header += f" {'k='+str(k):<8}"
    print(header)
    print(f"  {'-'*50}")
    for i, item in enumerate(dataset):
        line = f"  {i+1:<4} {item['type']:<10}"
        for r in results:
            key = f"hit@{r['k']}"
            hk = r['per_query'][i][key]
            line += f" {str(hk):<8}"
        print(line)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "topk_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
