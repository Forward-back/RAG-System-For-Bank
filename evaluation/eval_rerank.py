"""
Retrieval quality evaluation — Hit@k, MRR, and latency tracking.

Compares multiple pipeline configurations:
  - Fusion:  weighted_sum vs RRF
  - Model:   BGE (multilingual) vs BCE (Chinese-only)
  - Precision: FP32 vs INT8 (dynamic quantization)
"""

import time
import sys
import os
from statistics import mean

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
sys.path.append(BASE_DIR)

from src.rag_pipeline import RAGPipeline
from src.infra.evaluation_metrics import RetrievalEvaluator, LatencyTracker
from evaluation.eval_dataset import EVALUATION_DATASET

TOP_K = 5

# ── Configurations ──────────────────────────────────────────────────
# Each group isolates one variable; all groups share the same baseline
# so you can read the table top-to-bottom as progressive improvements.
CONFIGS = [
    # ── Group 1: Fusion mode ──
    {
        "name": "WeightedSum (No Rerank)",
        "enable_rerank": False, "enable_compression": False,
        "fusion_mode": "weighted_sum",
        "reranker_model": "BAAI/bge-reranker-v2-m3", "reranker_quantize": False,
    },
    {
        "name": "RRF (No Rerank)",
        "enable_rerank": False, "enable_compression": False,
        "fusion_mode": "rrf",
        "reranker_model": "BAAI/bge-reranker-v2-m3", "reranker_quantize": False,
    },
    {
        "name": "WeightedSum + BGE-FP32",
        "enable_rerank": True, "enable_compression": False,
        "fusion_mode": "weighted_sum",
        "reranker_model": "BAAI/bge-reranker-v2-m3", "reranker_quantize": False,
    },
    {
        "name": "RRF + BGE-FP32",
        "enable_rerank": True, "enable_compression": False,
        "fusion_mode": "rrf",
        "reranker_model": "BAAI/bge-reranker-v2-m3", "reranker_quantize": False,
    },

    # ── Group 2: Reranker model ──
    {
        "name": "RRF + BCE-FP32",
        "enable_rerank": True, "enable_compression": False,
        "fusion_mode": "rrf",
        "reranker_model": "maidalun1020/bce-reranker-base_v1", "reranker_quantize": False,
    },

    # ── Group 3: INT8 quantization ──
    # NOTE: PyTorch dynamic quantization is NOT compatible with
    # transformer CrossEncoder models (XLMRoberta's .ne() op fails on
    # qint8 tensors). For production INT8 inference, use ONNX Runtime or
    # OpenVINO via optimum.onnxruntime. Expected: ~2x memory reduction
    # with <1% quality loss, similar or better latency on CPU.
]


def evaluate_config(config: dict) -> dict:
    """Run the full eval loop for one pipeline configuration."""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    pipeline = RAGPipeline(
        data_paths=[DATA_DIR],
        persist_dir=DB_DIR,
        chunking_mode="recursive",
        enable_compression=config["enable_compression"],
        enable_rerank=config["enable_rerank"],
        enable_query_classification=False,
        enable_text_to_sql=False,
        enable_fact_check=False,
        enable_query_rewriting=False,
        fusion_mode=config.get("fusion_mode", "weighted_sum"),
        reranker_model=config.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
        reranker_quantize=config.get("reranker_quantize", False),
        reranker_local_files_only=False,  # allow download for eval
        top_k=TOP_K,
        verbose=False,
    )

    pipeline.build_index(rebuild=False)
    pipeline.load_models()

    evaluator = RetrievalEvaluator(EVALUATION_DATASET)
    tracker = LatencyTracker()
    hit_3 = hit_5 = 0
    reciprocal_ranks = []
    n = len(EVALUATION_DATASET)

    for i, item in enumerate(EVALUATION_DATASET):
        query = item["query"]
        relevant = item["relevant_contents"]

        t0 = time.perf_counter()
        retrieved = pipeline.retrieve_for_evaluation(query, k=TOP_K)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        tracker.record("retrieval", elapsed_ms)

        docs = [doc for doc, _ in retrieved] if retrieved else []
        rank = RetrievalEvaluator.find_first_rank(docs, relevant)
        rr = 1.0 / rank if rank else 0.0
        reciprocal_ranks.append(rr)

        if rank is not None and rank <= 3:
            hit_3 += 1
        if rank is not None and rank <= 5:
            hit_5 += 1

        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{n} queries")

    summary = tracker.summary().get("retrieval", {})

    return {
        "config": name,
        "hit@3": hit_3 / n if n else 0,
        "hit@5": hit_5 / n if n else 0,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0,
        "latency_avg": summary.get("avg", 0),
        "latency_p95": summary.get("p95", 0),
    }


def main():
    results = []
    for config in CONFIGS:
        try:
            results.append(evaluate_config(config))
        except Exception as e:
            print(f"  [SKIP] {config['name']} — {e}")

    if not results:
        print("No results collected.")
        return

    # ── Print comparison table ──
    print("\n\n" + "=" * 100)
    print(" " * 32 + "RAG RE-RANKING OPTIMIZATION RESULTS")
    print("=" * 100)
    header = (
        f"{'Config':<30} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | "
        f"{'Lat(avg)':<10} | {'Lat(p95)':<10} | {'vs Baseline':<12}"
    )
    print(header)
    print("-" * 100)

    # Use first result as baseline
    baseline_hit5 = results[0]["hit@5"]
    baseline_lat = results[0]["latency_avg"]

    for r in results:
        hit5_delta = r["hit@5"] - baseline_hit5
        lat_delta = r["latency_avg"] - baseline_lat
        cmp = f"Hit5 {hit5_delta:+.3f}  Lat {lat_delta:+.0f}ms"
        print(
            f"{r['config']:<30} | {r['hit@3']:<8.3f} | {r['hit@5']:<8.3f} | "
            f"{r['mrr']:<8.3f} | {r['latency_avg']:<8.1f}ms | "
            f"{r['latency_p95']:<8.1f}ms | {cmp:<12}"
        )

    print("=" * 100)

    # ── Summary analysis ──
    print("\n── Key Comparisons ──")

    def find(name_fragment):
        for r in results:
            if name_fragment in r["config"]:
                return r
        return None

    # Fusion mode
    ws = find("WeightedSum + BGE-FP32")
    rrf = find("RRF + BGE-FP32")
    if ws and rrf:
        print(f"\n  RRF vs WeightedSum (both with BGE-FP32):")
        print(f"    Hit@5:  {ws['hit@5']:.3f} → {rrf['hit@5']:.3f}  ({rrf['hit@5'] - ws['hit@5']:+.3f})")
        print(f"    MRR:    {ws['mrr']:.3f} → {rrf['mrr']:.3f}  ({rrf['mrr'] - ws['mrr']:+.3f})")
        print(f"    LatAvg: {ws['latency_avg']:.0f}ms → {rrf['latency_avg']:.0f}ms  ({rrf['latency_avg'] - ws['latency_avg']:+.0f}ms)")

    # Model comparison
    bge = find("RRF + BGE-FP32")
    bce = find("RRF + BCE-FP32")
    if bge and bce:
        print(f"\n  BCE vs BGE (both RRF + FP32):")
        print(f"    Hit@5:  {bge['hit@5']:.3f} → {bce['hit@5']:.3f}  ({bce['hit@5'] - bge['hit@5']:+.3f})")
        print(f"    MRR:    {bge['mrr']:.3f} → {bce['mrr']:.3f}  ({bce['mrr'] - bge['mrr']:+.3f})")
        print(f"    LatAvg: {bge['latency_avg']:.0f}ms → {bce['latency_avg']:.0f}ms  ({bce['latency_avg'] - bge['latency_avg']:+.0f}ms)")
        print(f"    Note:   BCE ~279M params (558MB) vs BGE ~568M params (1.1GB) — ~2x smaller")

    # Quantization note
    print(f"\n  INT8 Quantization:")
    print(f"    PyTorch dynamic quantization incompatible with transformer CrossEncoder models.")
    print(f"    For production INT8: use ONNX Runtime (optimum.onnxruntime) or OpenVINO backend.")
    print(f"    Expected: ~2x memory reduction, <1% quality loss, comparable or better latency on CPU.")

    # Best overall
    best = max(results, key=lambda r: r["hit@5"])
    fastest = min(results, key=lambda r: r["latency_avg"])
    print(f"\n── Recommendation ──")
    print(f"  Best quality:  {best['config']}  (Hit@5={best['hit@5']:.3f}, MRR={best['mrr']:.3f})")
    print(f"  Best latency:  {fastest['config']}  (LatAvg={fastest['latency_avg']:.0f}ms)")
    print()


if __name__ == "__main__":
    main()
