"""
Context compression evaluation — compares full context vs compressed context
on token savings and generation speed.
"""

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
sys.path.append(BASE_DIR)

from src.rag_pipeline import RAGPipeline
from src.infra.evaluation_metrics import LatencyTracker
from evaluation.eval_dataset import EVALUATION_DATASET


def run_compression_eval():
    print("Initializing pipeline...\n")

    pipeline = RAGPipeline(
        data_paths=[DATA_DIR],
        persist_dir=DB_DIR,
        chunking_mode="recursive",
        enable_compression=True,
        enable_rerank=True,
        enable_query_classification=False,
        enable_text_to_sql=False,
        enable_fact_check=False,
        enable_query_rewriting=False,
        top_k=5,
        verbose=False,
    )

    pipeline.build_index(rebuild=False)
    pipeline.load_models()

    tracker = LatencyTracker()
    total_orig_chars = 0
    total_comp_chars = 0
    n = len(EVALUATION_DATASET)

    print(f"Evaluating {n} queries...\n")

    for idx, item in enumerate(EVALUATION_DATASET, 1):
        print(f"Processing query [{idx}/{n}]...", end="\r", flush=True)

        query = item["query"]
        retrieved = pipeline.retrieve_for_evaluation(query, k=5)
        original_docs = [doc for doc, _ in retrieved]

        if not original_docs:
            continue

        orig_chars = sum(len(doc.page_content) for doc in original_docs)
        total_orig_chars += orig_chars

        # Compress
        compressed_docs = pipeline.compressor.compress_documents(
            query=query, documents=original_docs
        )
        comp_chars = sum(len(doc.page_content) for doc in compressed_docs)
        total_comp_chars += comp_chars

        # Gen time (uncompressed)
        with tracker.stage("gen_full"):
            pipeline.generator.generate_with_citations(
                query=query, context_docs=original_docs
            )

        # Gen time (compressed)
        with tracker.stage("gen_compressed"):
            pipeline.generator.generate_with_citations(
                query=query, context_docs=compressed_docs
            )

    print(" " * 50, end="\r")

    saved_pct = (
        ((total_orig_chars - total_comp_chars) / total_orig_chars * 100)
        if total_orig_chars > 0 else 0
    )
    tokens_saved = (total_orig_chars - total_comp_chars) // 4

    latency = tracker.summary()
    gen_full = latency.get("gen_full", {})
    gen_comp = latency.get("gen_compressed", {})

    print("=" * 60)
    print("         COMPRESSION EVALUATION REPORT")
    print("=" * 60)
    print(f"  Original Context Size : {total_orig_chars} chars")
    print(f"  Compressed Context    : {total_comp_chars} chars")
    print(f"  Tokens Saved          : ~{tokens_saved} tokens")
    print(f"  Overall Cost Reduction: {saved_pct:.2f}%")
    print("-" * 60)
    print(f"  Gen Time (Full)       : avg={gen_full.get('avg', 0):.1f}ms  p95={gen_full.get('p95', 0):.1f}ms")
    print(f"  Gen Time (Compressed) : avg={gen_comp.get('avg', 0):.1f}ms  p95={gen_comp.get('p95', 0):.1f}ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_compression_eval()
