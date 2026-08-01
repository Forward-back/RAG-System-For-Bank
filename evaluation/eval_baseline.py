"""
RAG Evaluation Framework — Page-based ground truth.

Instead of annotating per-retrieval-result positions (which breaks when
changing retriever configs), we annotate which PAGES contain the answer.
Any retrieved chunk from a ground-truth page counts as a hit.

Supports:
  - Naive baseline (dense-only)
  - Hybrid retrieval (dense + BM25, with/without rerank)
  - top_k ablation, chunking comparison, query expansion
"""
import json
import os
import sys
import time
from statistics import mean
from typing import List, Dict, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from langchain_core.documents import Document
from src.infra.model_registry import SharedEmbeddings


# ── Load dataset ──
def load_eval_dataset() -> List[dict]:
    path = os.path.join(os.path.dirname(__file__), "eval_queries.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Page-based metrics ──
import re

_PAGE_IN_TEXT = re.compile(r'=== PAGE (\d+) ===')

def get_page(doc: Document) -> int:
    """Extract page number from document metadata or text content."""
    # Try metadata first
    meta = doc.metadata
    for key in ("page", "first_page", "last_page"):
        val = meta.get(key)
        if val is not None and str(val) != "na":
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    # Fallback: look for "=== PAGE N ===" in text
    m = _PAGE_IN_TEXT.search(doc.page_content)
    if m:
        return int(m.group(1))
    # Try extracting from text like "\n5\n" at start
    text_start = doc.page_content.strip()[:10]
    try:
        return int(text_start)
    except ValueError:
        pass
    return -1


def compute_hit_at_k(
    retrieved_docs: List[Document],
    ground_truth_pages: List[int],
    k: int,
) -> bool:
    """True if any top-k doc comes from a ground-truth page."""
    for doc in retrieved_docs[:k]:
        page = get_page(doc)
        if page in ground_truth_pages:
            return True
    return False


def compute_mrr(
    retrieved_docs: List[Document],
    ground_truth_pages: List[int],
) -> float:
    """Reciprocal rank of first doc from a ground-truth page."""
    for rank, doc in enumerate(retrieved_docs, 1):
        if get_page(doc) in ground_truth_pages:
            return 1.0 / rank
    return 0.0


# ── Retrievers ──
class DenseOnlyRetriever:
    """ChromaDB vector similarity only — no BM25."""

    def __init__(self, persist_dir: str):
        from langchain_community.vectorstores import Chroma
        self.embeddings = SharedEmbeddings()
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name="rag_collection",
        )

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        results = self.db.similarity_search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in results]


# ── Evaluation runner ──
def run_eval(
    retriever,
    dataset: List[dict],
    k: int = 5,
    label: str = "Unnamed",
) -> dict:
    hit_3 = hit_5 = 0
    mrr_values = []
    latencies = []
    per_query = []
    n = len(dataset)

    for item in dataset:
        query = item["query"]
        gt_pages = item.get("ground_truth_pages", [])

        t0 = time.perf_counter()
        results = retriever.retrieve(query, k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        docs = [doc for doc, _ in results] if results else []

        h3 = compute_hit_at_k(docs, gt_pages, 3)
        h5 = compute_hit_at_k(docs, gt_pages, 5)
        mrr = compute_mrr(docs, gt_pages)

        if h3:
            hit_3 += 1
        if h5:
            hit_5 += 1
        mrr_values.append(mrr)

        per_query.append({
            "query": query,
            "type": item["type"],
            "hit@3": h3,
            "hit@5": h5,
            "mrr": round(mrr, 3),
            "latency_ms": round(elapsed_ms, 1),
            "retrieved_pages": [get_page(d) for d in docs],
            "gt_pages": gt_pages,
        })

    total = n if n > 0 else 1
    lats = sorted(latencies)

    return {
        "label": label,
        "k": k,
        "queries": n,
        "hit@3": round(hit_3 / total, 3),
        "hit@5": round(hit_5 / total, 3),
        "mrr": round(mean(mrr_values), 3) if mrr_values else 0.0,
        "latency_avg_ms": round(mean(lats), 1),
        "latency_p50_ms": round(lats[int(total * 0.50)], 1),
        "latency_p95_ms": round(lats[int(total * 0.95)], 1),
        "per_query": per_query,
    }


# ── Print report ──
def print_report(result: dict) -> None:
    print(f"  Config:      {result['label']}")
    print(f"  top_k:       {result['k']}")
    print(f"  Hit@3:       {result['hit@3']:.3f}")
    print(f"  Hit@5:       {result['hit@5']:.3f}")
    print(f"  MRR:         {result['mrr']:.3f}")
    print(f"  Latency avg: {result['latency_avg_ms']:.1f} ms")
    print(f"  Latency p50: {result['latency_p50_ms']:.1f} ms")
    print(f"  Latency p95: {result['latency_p95_ms']:.1f} ms")
    print()
    print(f"  {'Q#':<4} {'Type':<10} {'Hit@5':<6} {'MRR':<6} {'Pages':<20} {'GT':<15}")
    print(f"  {'-'*4} {'-'*10} {'-'*6} {'-'*6} {'-'*20} {'-'*15}")
    for pq in result["per_query"]:
        idx = result["per_query"].index(pq) + 1
        print(f"  {idx:<4} {pq['type']:<10} {str(pq['hit@5']):<6} {pq['mrr']:<6.3f} {str(pq['retrieved_pages']):<20} {str(pq['gt_pages']):<15}")


# ── Main ──
def main():
    os.environ.setdefault("TESSDATA_PREFIX", os.path.join(BASE_DIR, "tessdata"))

    print("=" * 65)
    print("  NAIVE BASELINE: Dense-Only Retrieval (page-based GT)")
    print("=" * 65)

    dataset = load_eval_dataset()
    print(f"  Queries: {len(dataset)}")
    print(f"  Config:  ChromaDB cosine similarity, top_k=5")
    print(f"           No BM25 / No rerank / No compression / No expansion")
    print()

    persist_dir = os.path.join(BASE_DIR, "chroma_db")
    retriever = DenseOnlyRetriever(persist_dir=persist_dir)

    result = run_eval(retriever, dataset, k=5, label="Naive Baseline (Dense-Only)")

    print_report(result)
    print("=" * 65)

    out_path = os.path.join(os.path.dirname(__file__), "baseline_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
