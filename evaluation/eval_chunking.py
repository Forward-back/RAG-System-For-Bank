"""
Chunking strategy comparison — per-page preservation.

Keeps page boundaries intact: chunks each page separately so
page metadata is preserved for ground-truth evaluation.
"""
import json
import os
import re
import sys
import time
from statistics import mean

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from langchain_core.documents import Document
from src.infra.model_registry import SharedEmbeddings
from src.ingestion.chunking import Chunking
from evaluation.eval_baseline import (
    load_eval_dataset, compute_hit_at_k, compute_mrr, get_page,
)


def extract_per_page_docs(persist_dir: str) -> list:
    """Extract text per page, return list of (page_num, text)."""
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.list_collections()[0]
    results = col.get(include=["documents", "metadatas"])

    page_texts = {}
    for text, meta in zip(results["documents"], results["metadatas"]):
        page = meta.get("page", None)
        if page is not None and str(page) != "na":
            try:
                p = int(page)
            except (ValueError, TypeError):
                continue
            cleaned = re.sub(r'=== PAGE \d+ ===', '', text).strip()
            if cleaned:
                page_texts[p] = cleaned

    docs = []
    for p in sorted(page_texts.keys()):
        docs.append(Document(
            page_content=page_texts[p],
            metadata={"page": p, "first_page": p, "last_page": p,
                       "source": "招商银行规章.pdf"}
        ))
    return docs


class MinDoc:
    """Minimal object with .metadata for get_page() compatibility."""
    def __init__(self, meta):
        self.metadata = meta
        self.page_content = ""


def evaluate_strategy(chunks: list, dataset: list, embeddings, k: int = 5) -> dict:
    """Evaluate one chunking strategy using in-memory ChromaDB."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    col_name = f"eval_{int(time.time()*1000)}"
    collection = client.create_collection(col_name)

    texts = [c.page_content for c in chunks]
    embeds = embeddings.embed_documents(texts)
    ids = [str(i) for i in range(len(texts))]
    metadatas = [c.metadata for c in chunks]
    collection.add(embeddings=embeds, documents=texts, metadatas=metadatas, ids=ids)

    hit_k = 0
    mrr_values = []
    latencies = []
    for item in dataset:
        query = item["query"]
        gt_pages = item.get("ground_truth_pages", [])

        t0 = time.perf_counter()
        q_embed = embeddings.embed_query(query)
        results = collection.query(query_embeddings=[q_embed], n_results=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        retrieved_metadatas = results["metadatas"][0] if results["metadatas"] else []
        docs = [MinDoc(m) for m in retrieved_metadatas]
        hk = compute_hit_at_k(docs, gt_pages, k)
        mrr = compute_mrr(docs, gt_pages)
        if hk:
            hit_k += 1
        mrr_values.append(mrr)

    client.delete_collection(col_name)
    n = len(dataset)
    lats = sorted(latencies)
    return {
        "chunk_count": len(chunks),
        "hit@k": round(hit_k / n, 3) if n else 0,
        "mrr": round(mean(mrr_values), 3) if mrr_values else 0,
        "latency_avg_ms": round(mean(lats), 1),
        "latency_p50_ms": round(lats[n // 2], 1),
        "latency_p95_ms": round(lats[int(n * 0.95)], 1),
    }


def main():
    os.environ.setdefault("TESSDATA_PREFIX", os.path.join(BASE_DIR, "tessdata"))
    persist_dir = os.path.join(BASE_DIR, "chroma_db")
    dataset = load_eval_dataset()

    print(f"\n{'='*70}")
    print(f"  CHUNKING STRATEGY COMPARISON (per-page preservation)")
    print(f"{'='*70}")

    print("  Extracting per-page documents...")
    page_docs = extract_per_page_docs(persist_dir)
    print(f"  Found {len(page_docs)} pages with text")
    total_chars = sum(len(d.page_content) for d in page_docs)
    print(f"  Total text: {total_chars:,} chars")

    embeddings = SharedEmbeddings()
    results = []
    strategies = ["recursive", "semantic", "element_aware", "hierarchical"]

    for strategy in strategies:
        chunk_func = {
            "recursive": Chunking.recursive_chunking,
            "semantic": Chunking.semantic_chunking,
            "element_aware": Chunking.element_aware_chunking,
            "hierarchical": Chunking.hierarchical_chunking,
        }[strategy]

        print(f"\n  [{strategy}] Chunking {len(page_docs)} pages...")
        t0 = time.perf_counter()

        try:
            if strategy == "semantic":
                chunks = chunk_func(page_docs, chunk_size=1000)
            else:
                chunks = chunk_func(page_docs, chunk_size=1000, chunk_overlap=200)
        except Exception as e:
            print(f"    ERROR: {e}, falling back to recursive")
            chunks = Chunking.recursive_chunking(page_docs, chunk_size=1000, chunk_overlap=200)

        chunk_time = (time.perf_counter() - t0) * 1000
        print(f"    {len(chunks)} chunks ({chunk_time:.0f}ms)")

        # Clean metadata: remove empty lists/None values (ChromaDB rejects them)
        for chunk in chunks:
            for key in list(chunk.metadata.keys()):
                val = chunk.metadata[key]
                if val is None or (isinstance(val, list) and len(val) == 0):
                    del chunk.metadata[key]

        # Check page metadata preservation
        pages_with_meta = sum(1 for c in chunks
                              if c.metadata.get("page") is not None
                              and str(c.metadata.get("page")) != "na")
        print(f"    Pages with metadata: {pages_with_meta}/{len(chunks)}")

        r = evaluate_strategy(chunks, dataset, embeddings, k=5)
        r["label"] = strategy
        r["chunk_time_ms"] = round(chunk_time, 0)
        r["k"] = 5
        results.append(r)

        print(f"    Hit@5={r['hit@k']:.3f}  MRR={r['mrr']:.3f}  "
              f"Lat(avg)={r['latency_avg_ms']:.1f}ms")

    # Summary
    print(f"\n{'='*70}")
    print(f"  CHUNKING STRATEGY SUMMARY")
    print(f"  {'Strategy':<16} {'Chunks':<8} {'Hit@5':<8} {'MRR':<8} "
          f"{'Lat(avg)':<12} {'p95':<10} {'Time':<10}")
    print(f"  {'-'*64}")
    for r in results:
        print(f"  {r['label']:<16} {r['chunk_count']:<8} {r['hit@k']:<8.3f} {r['mrr']:<8.3f} "
              f"{r['latency_avg_ms']:<12.1f} {r['latency_p95_ms']:<10.1f} {r['chunk_time_ms']:<10.0f}ms")

    out_path = os.path.join(os.path.dirname(__file__), "chunking_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
