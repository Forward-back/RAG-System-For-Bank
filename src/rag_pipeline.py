from typing import Dict, List, Optional, Tuple
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from langchain_core.documents import Document

from src.ingestion.chunking import Chunking
from src.retrieval.context_compression import ContextCompressor
from src.ingestion.documents_ingestion import DataIngestion
from src.safety.fact_checker import FactChecker
from src.safety.numeric_guard import NumericGuard
from src.safety.nli_verifier import NLIVerifier
from src.generation.generator_with_citations import RAGGenerator
from src.ingestion.hierarchy_parser import build_hierarchy_index, expand_context
from src.generation.llm_client import DeepSeekLLM
from src.query.query_classifier import BANK_SEED_SAMPLES, QueryClassifier
from src.query.query_rewriter import QueryRewriter
from src.retrieval.query_transformer import QueryTransformer
from src.retrieval.reranker import ReRanker
from src.retrieval.retrieval import HybridRetriever
from src.ingestion.table_processor import TableProcessor
from src.query.text_to_sql import TextToSQL
from src.retrieval.vector_embedding import EmbeddingStore
from src.infra.query_cache import QueryCache
from src.infra.query_tracer import QueryTracer
from src.infra.web_search import WebSearcher, WebSearchResponse

logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(
        self,
        data_paths: List[str],
        persist_dir: str = "./chroma_db",

        chunking_mode: str = "recursive",
        enable_rerank: bool = True,
        enable_compression: bool = True,
        enable_table_processing: bool = True,
        enable_query_classification: bool = True,
        enable_text_to_sql: bool = True,
        enable_fact_check: bool = True,
        enable_numeric_guard: bool = True,
        enable_nli_verifier: bool = True,
        enable_query_rewriting: bool = True,
        query_expansion_mode: str = "template",
        enable_query_cache: bool = True,
        cache_path: str = "./cache/query_cache.json",
        enable_tracing: bool = True,
        trace_logs_dir: str = "./logs",
        top_k: int = 5,
        verbose: bool = True,
        pdf_strategy: str = "auto_detect",
        fusion_mode: str = "weighted_sum",
        reranker_model: str = "E:/models/bge-reranker-v2-m3",
        reranker_quantize: bool = False,
        reranker_local_files_only: bool = True,
        retrieval_quality_threshold: float = 0.0,
        enable_web_search: bool = False,
        web_search_quality_threshold: float = 0.0,
        web_search_max_results: int = 5,
        web_search_timeout: float = 10.0,
    ):
        self.data_paths = data_paths
        self.persist_dir = persist_dir

        self.chunking_mode = chunking_mode.lower()
        self.enable_rerank = enable_rerank
        self.enable_compression = enable_compression
        self.enable_table_processing = enable_table_processing
        self.enable_query_classification = enable_query_classification
        self.enable_text_to_sql = enable_text_to_sql
        self.enable_fact_check = enable_fact_check
        self.enable_numeric_guard = enable_numeric_guard
        self.enable_nli_verifier = enable_nli_verifier
        self.enable_query_rewriting = enable_query_rewriting
        self.query_expansion_mode = query_expansion_mode.lower()
        self.enable_query_cache = enable_query_cache
        self.cache_path = cache_path
        self.enable_tracing = enable_tracing
        self.trace_logs_dir = trace_logs_dir
        self.top_k = top_k
        self.verbose = verbose
        self.pdf_strategy = pdf_strategy
        self.fusion_mode = fusion_mode
        self.reranker_model = reranker_model
        self.reranker_quantize = reranker_quantize
        self.reranker_local_files_only = reranker_local_files_only
        self.retrieval_quality_threshold = retrieval_quality_threshold
        self.enable_web_search = enable_web_search
        self.web_search_quality_threshold = web_search_quality_threshold
        self.web_search_max_results = web_search_max_results
        self.web_search_timeout = web_search_timeout

        # Runtime objects
        self.chunks = None
        self.vector_db = None
        self.retriever = None
        self.reranker = None
        self.web_searcher: Optional[WebSearcher] = None

        self.query_transformer = QueryTransformer(mode=self.query_expansion_mode)
        self.llm = None
        self.compressor = None
        self.generator = None
        self.query_classifier: Optional[QueryClassifier] = None
        self.text_to_sql: Optional[TextToSQL] = None
        self.fact_checker: Optional[FactChecker] = None
        self.numeric_guard: Optional[NumericGuard] = None
        self.nli_verifier: Optional[NLIVerifier] = None
        self.query_rewriter: Optional[QueryRewriter] = None
        self.query_cache: Optional[QueryCache] = None
        self._tracer: Optional[QueryTracer] = None  # per-query, set in run()

        # Incremental-indexing state: {absolute_path: sha256}
        self._file_hashes: Dict[str, str] = {}

        # Hierarchy index for O(1) context expansion (only when chunking_mode=="hierarchical")
        self._hierarchy_index: Optional[dict] = None

        if self.verbose:
            logger.info("RAG pipeline config initialized.")

    # ------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------
    def _try_load_chunks(self):
        """Load chunks from the persisted vector store if it exists."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            try:
                embedding_store = EmbeddingStore()
                temp_db = embedding_store.create_or_load_db(
                    chunks=[], persist_directory=self.persist_dir, rebuild=False
                )
                self.chunks = self._load_chunks_from_store(temp_db)
            except Exception:
                pass

    def _load_chunks_from_store(self, store=None) -> List[Document]:
        """Reconstruct chunk documents from an existing ChromaDB collection."""
        db = store or self.vector_db
        try:
            data = db.get(include=["documents", "metadatas"])
            chunks = []
            for i, doc_text in enumerate(data.get("documents", [])):
                meta = data["metadatas"][i] if data.get("metadatas") else {}
                chunks.append(Document(page_content=doc_text, metadata=meta or {}))
            if self.verbose:
                logger.info("[INDEX] Loaded %d chunks from existing vector store.", len(chunks))
            return chunks
        except Exception as e:
            logger.warning("[INDEX] Failed to load chunks from store: %s", e)
            return []

    def _hash_file_path(self) -> str:
        return os.path.join(self.persist_dir, "file_hashes.json")

    def _load_file_hashes(self) -> Dict[str, str]:
        p = self._hash_file_path()
        if os.path.exists(p):
            try:
                import json
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_file_hashes(self):
        import json
        p = self._hash_file_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._file_hashes, f, ensure_ascii=False)

    def build_index(self, rebuild: bool = False):

        if self.verbose:
            logger.info("[INDEX] Building / Loading index...")
            logger.info("Loading documents...")

        if rebuild:
            self.vector_db = None
            self.retriever = None
            self._file_hashes = {}
            self._hierarchy_index = None
            self.chunks = None
        else:
            # Ensure chunks are always available for BM25
            if self.chunks is None:
                self._try_load_chunks()

        # Restore persisted hashes so restarts don't re-ingest unchanged files
        if not self._file_hashes:
            self._file_hashes = self._load_file_hashes()
            logger.info("[INDEX] Loaded %d file hashes from disk.", len(self._file_hashes))

        # Incremental: compare on-disk state against last known hashes
        skip_hashes: Optional[Dict[str, str]] = None
        sources_to_cleanup: List[str] = []  # changed + removed sources to delete from DB
        is_incremental = False
        removed_set: set = set()
        changed_set: set = set()

        if not rebuild and self._file_hashes:
            logger.info(
                "[INDEX] Entering diff block, rebuild=%s hashes=%d",
                rebuild, len(self._file_hashes)
            )
            diff = DataIngestion.diff_files(self.data_paths, self._file_hashes)
            logger.info(
                "[INDEX] Diff: new=%d changed=%d removed=%d",
                len(diff["new"]), len(diff["changed"]), len(diff["removed"])
            )
            if self.verbose:
                logger.info(
                    "[INDEX] Incremental: %d new, %d changed, %d removed",
                    len(diff["new"]), len(diff["changed"]), len(diff["removed"])
                )
            sources_to_cleanup = diff["removed"] + diff["changed"]
            removed_set = set(diff["removed"])
            changed_set = set(diff["changed"])

            if not diff["new"] and not diff["changed"]:
                # Only removals (no new/changed) — clean up stale chunks
                if diff["removed"]:
                    if self.vector_db is None:
                        embedding_store = EmbeddingStore()
                        self.vector_db = embedding_store.create_or_load_db(
                            chunks=[], persist_directory=self.persist_dir, rebuild=False
                        )
                    for src in diff["removed"]:
                        self.vector_db.delete(where={"source": src})
                    # Filter self.chunks
                    if self.chunks:
                        self.chunks = [
                            c for c in self.chunks
                            if c.metadata.get("source") not in removed_set
                        ]
                    self._file_hashes = diff["current_hashes"]
                    # Refresh retriever with filtered chunks
                    if self.retriever and self.chunks is not None:
                        self.retriever.refresh_documents(self.chunks)
                    if self.verbose:
                        logger.info(
                            "[INDEX] Cleaned up %d removed file(s) from vector store.",
                            len(diff["removed"])
                        )
                else:
                    if self.verbose:
                        logger.info("[INDEX] No changes detected — index is up to date.")
                if self.vector_db is None:
                    embedding_store = EmbeddingStore()
                    self.vector_db = embedding_store.create_or_load_db(
                        chunks=[], persist_directory=self.persist_dir, rebuild=False
                    )
                if self.retriever is None and self.vector_db is not None:
                    # Load chunks from vector store so BM25 can be built.
                    # self.chunks is None on first load from a persisted index.
                    if self.chunks is None:
                        self.chunks = self._load_chunks_from_store()
                    self.retriever = HybridRetriever(self.vector_db, self.chunks or [])
                logger.info("[INDEX] No changes detected — skipping document ingestion.")
                return

            is_incremental = True
            # Only pass the files that haven't changed as skip_hashes
            skip_hashes = {
                p: h for p, h in self._file_hashes.items()
                if p not in diff["new"] and p not in diff["changed"] and p not in diff["removed"]
            }
            # Update hashes to the new snapshot
            self._file_hashes = diff["current_hashes"]
        else:
            # Full build — snapshot all hashes
            self._file_hashes = DataIngestion.scan_files(self.data_paths)

        docs = DataIngestion.ingest(
            self.data_paths,
            pdf_strategy=self.pdf_strategy,
            skip_hashes=skip_hashes,
        )

        if self.verbose:
            logger.info("Chunking mode: %s", self.chunking_mode)

        if self.chunking_mode == "recursive":
            new_chunks = Chunking.recursive_chunking(docs)

        elif self.chunking_mode == "semantic":
            new_chunks = Chunking.semantic_chunking(docs)

        elif self.chunking_mode == "element_aware":
            new_chunks = Chunking.element_aware_chunking(docs)

        elif self.chunking_mode == "hierarchical":
            new_chunks = Chunking.hierarchical_chunking(docs)

        else:
            raise ValueError(f"Invalid chunking mode: {self.chunking_mode}")

        # Extract keywords for table chunks (post-chunking, table boundaries respected)
        if self.enable_table_processing:
            new_chunks = TableProcessor.process(new_chunks, verbose=self.verbose)

        if self.verbose:
            logger.info("Generated %d new chunks.", len(new_chunks))

        embedding_store = EmbeddingStore()

        if is_incremental:
            # ── Incremental update ──
            # Load existing DB
            self.vector_db = embedding_store.create_or_load_db(
                chunks=[], persist_directory=self.persist_dir, rebuild=False
            )
            # Delete stale chunks for changed + removed sources
            if sources_to_cleanup and self.vector_db:
                for src in sources_to_cleanup:
                    self.vector_db.delete(where={"source": src})
                if self.verbose:
                    logger.info(
                        "[INDEX] Cleaned up %d stale source(s) from vector store.",
                        len(sources_to_cleanup)
                    )
            # Add new chunks to existing DB
            if new_chunks:
                valid = EmbeddingStore.validate_chunks(new_chunks)
                if valid:
                    self.vector_db.add_documents(documents=valid)
                    if self.verbose:
                        logger.info("[INDEX] Added %d chunks to vector store.", len(valid))
            # Merge chunks for BM25: keep unchanged, drop stale, add new
            if self.chunks:
                stale = removed_set | changed_set
                kept = [
                    c for c in self.chunks
                    if c.metadata.get("source") not in stale
                ]
                self.chunks = kept + new_chunks
            else:
                self.chunks = new_chunks
            # Refresh retriever
            if self.retriever:
                self.retriever.refresh_documents(self.chunks)
            else:
                self.retriever = HybridRetriever(self.vector_db, self.chunks)
        else:
            # ── Full build ──
            self.vector_db = embedding_store.create_or_load_db(
                chunks=new_chunks,
                persist_directory=self.persist_dir,
                rebuild=rebuild,
            )
            self.chunks = new_chunks
            if self.verbose:
                logger.info("Building hybrid retriever...")
            self.retriever = HybridRetriever(self.vector_db, self.chunks)

        # Hierarchy index (after self.chunks is final)
        if self.chunking_mode == "hierarchical" and self.chunks:
            self._hierarchy_index = build_hierarchy_index(self.chunks)

        if self.verbose:
            logger.info("Total chunks in index: %d", len(self.chunks))
            logger.info("[INDEX] Ready.")

        # Persist file hashes so restarts don't re-ingest unchanged files
        self._save_file_hashes()

    def rebuild_index(self):
        if self.verbose:
            logger.info("[INDEX] Rebuilding index...")
        self.build_index(rebuild=True)
        if self.query_cache:
            self.query_cache.clear()
            if self.verbose:
                logger.info("[CACHE] Cleared after index rebuild.")

    # ------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------
    def load_models(self):

        if self.verbose:
            logger.info("[MODELS] Loading models...")

        if self.enable_rerank:
            self.reranker = ReRanker(
                model_name=self.reranker_model,
                use_quantization=self.reranker_quantize,
                local_files_only=self.reranker_local_files_only,
            )
            if self.verbose:
                logger.info("Re-ranker enabled.")
        else:
            self.reranker = None
            if self.verbose:
                logger.info("Re-ranker disabled.")

        self.llm = DeepSeekLLM()

        # Wire LLM into QueryTransformer if using llm expansion mode
        if self.query_expansion_mode == "llm":
            self.query_transformer.llm = self.llm
            if self.verbose:
                logger.info("Query expansion mode: LLM")

        if self.enable_compression:
            self.compressor = ContextCompressor(self.llm)
            if self.verbose:
                logger.info("Context compression enabled.")
        else:
            self.compressor = None
            if self.verbose:
                logger.info("Context compression disabled.")

        self.generator = RAGGenerator()
        self.generator.llm=self.llm

        # Query classifier (lazy init: reuses existing embedding infra)
        if self.enable_query_classification:
            self.query_classifier = QueryClassifier()
            self.query_classifier.load_samples(BANK_SEED_SAMPLES)
            if self.verbose:
                logger.info("Query classifier enabled.")
        else:
            self.query_classifier = None
            if self.verbose:
                logger.info("Query classifier disabled.")

        # Text-to-SQL for structured data queries
        if self.enable_text_to_sql:
            self.text_to_sql = TextToSQL(self.llm)
            if self.verbose:
                logger.info("Text-to-SQL enabled.")
        else:
            self.text_to_sql = None
            if self.verbose:
                logger.info("Text-to-SQL disabled.")

        # Post-generation fact-checker (Layer 3 — LLM fallback)
        if self.enable_fact_check:
            self.fact_checker = FactChecker(self.llm)
            if self.verbose:
                logger.info("Fact-checker (L3 LLM) enabled.")
        else:
            self.fact_checker = None
            if self.verbose:
                logger.info("Fact-checker (L3) disabled.")

        # Numeric guard (Layer 1 — deterministic rules)
        if self.enable_numeric_guard:
            self.numeric_guard = NumericGuard()
            if self.verbose:
                logger.info("NumericGuard (L1 rules) enabled.")
        else:
            self.numeric_guard = None

        # NLI verifier (Layer 2 — cross-encoder, shares reranker model)
        if self.enable_nli_verifier:
            shared_model = self.reranker.model if self.reranker else None
            self.nli_verifier = NLIVerifier(model=shared_model)
            if self.verbose:
                logger.info("NLIVerifier (L2 cross-encoder) enabled (shared=%s).",
                            "yes" if shared_model else "no (standalone)")
        else:
            self.nli_verifier = None

        # Query rewriter (pre-retrieval)
        if self.enable_query_rewriting:
            self.query_rewriter = QueryRewriter(self.llm)
            if self.verbose:
                logger.info("Query rewriter enabled.")
        else:
            self.query_rewriter = None
            if self.verbose:
                logger.info("Query rewriter disabled.")

        # Query cache (semantic dedup for FAQ-style queries)
        if self.enable_query_cache:
            self.query_cache = QueryCache(cache_path=self.cache_path)
            if self.verbose:
                logger.info("Query cache enabled (%d entries).", self.query_cache.size)
        else:
            self.query_cache = None
            if self.verbose:
                logger.info("Query cache disabled.")

        # Web searcher (KB fallback)
        if self.enable_web_search:
            self.web_searcher = WebSearcher(
                max_results=self.web_search_max_results,
                timeout=self.web_search_timeout,
            )
            if self.verbose:
                logger.info("Web searcher enabled (threshold=%.2f).",
                            self.web_search_quality_threshold)
        else:
            self.web_searcher = None
            if self.verbose:
                logger.info("Web searcher disabled.")

        if self.verbose:
            logger.info("[MODELS] Ready.")

    # ------------------------------------------------
    # Safety check
    # ------------------------------------------------
    def ready(self):

        if self.retriever is None or self.vector_db is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        if self.generator is None or self.llm is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if self.verbose:
            logger.info("Pipeline is ready for queries.")

    # ------------------------------------------------
    # Query execution
    # ------------------------------------------------
    def run(self, query: str, enable_web_search: Optional[bool] = None) -> str:

        self.ready()

        _use_web = (
            enable_web_search if enable_web_search is not None
            else self.enable_web_search
        )

        tracer = QueryTracer(
            query=query, logs_dir=self.trace_logs_dir,
            enabled=self.enable_tracing,
        )

        if self.verbose:
            logger.info("User Query: %s", query)

        # ── Step 0: classify & route ──
        route = None
        if self.query_classifier:
            route = self.query_classifier.route(query)
            tracer.classification = {
                "risk_level": route["risk_level"],
                "doc_type": route["doc_type"],
                "action": route["action"],
                "computation_needed": route["computation_needed"],
                "confidence": route["confidence"],
            }
            if self.verbose:
                logger.info(
                    "[ROUTE] risk=%s doc_type=%s action=%s computation=%s confidence=%.2f",
                    route["risk_level"], route["doc_type"], route["action"],
                    route["computation_needed"], route["confidence"]
                )

            if route["action"] == "reject_and_escalate":
                rejection_msg = (
                    f"很抱歉，您咨询的问题涉及个性化金融建议或存在合规风险，"
                    f"建议您联系我行客服热线或前往网点咨询客户经理。\n\n"
                    f"（系统提示：{route['reason']}）"
                )
                tracer.flush(rejection_msg)
                return rejection_msg

        # ── Step 0.4: Semantic cache lookup ──
        if self.query_cache:
            cached = self.query_cache.lookup(query)
            if cached:
                if self.verbose:
                    logger.info("[CACHE] Hit — returning cached answer")
                tracer.cache_hit = True
                tracer.flush(cached)
                return cached

        # ── Step 0.5: Query rewriting (clarify ambiguous queries) ──
        rewritten_query: Optional[str] = None
        if self.query_rewriter:
            # Pass embedding model for semantic similarity gates
            sim_model = self.query_classifier.model if self.query_classifier else None
            rewrite_result = self.query_rewriter.rewrite(query, similarity_model=sim_model)
            if rewrite_result["is_rewritten"]:
                rewritten_query = rewrite_result["rewritten"]
                tracer.rewritten_query = rewritten_query
                logger.info("[REWRITE] 改写: '%s' → '%s'", query, rewritten_query)
            else:
                logger.info("[REWRITE] 未改写 (%s): '%s'",
                            rewrite_result.get("reject_reason", "未知原因"), query)

        # Use rewritten query for retrieval if available
        retrieval_query = rewritten_query or query

        # ── Step 0.6: Text-to-SQL for computation queries ──
        sql_result: Optional[dict] = None
        if route and route["computation_needed"] and self.text_to_sql:
            if self.verbose:
                logger.info("[SQL] Attempting Text-to-SQL...")
            sql_result = self.text_to_sql.query(query)
            if self.verbose and sql_result.get("sql"):
                logger.info("[SQL] Generated: %s", sql_result["sql"])
                logger.info("[SQL] Returned %d rows", sql_result["row_count"])
            if sql_result.get("error") and self.verbose:
                logger.warning("[SQL] Error: %s", sql_result["error"])

        tracer.sql = {
            "used": sql_result is not None and sql_result.get("sql") is not None,
            "row_count": sql_result.get("row_count", 0) if sql_result else 0,
            "has_error": bool(sql_result and sql_result.get("error")),
        }

        # ── Step 1: Query transformation ──
        queries = self.query_transformer.expand(retrieval_query)

        if self.verbose:
            logger.info("Expanded Queries: %s", queries)

        # Build retrieval filter from classifier prediction.
        # NOTE: classifier doc_type labels (regulation/procedure/product/faq)
        # are in English but chunk metadata "domain" uses Chinese directory
        # names (规章制度/...), so we skip domain filtering for now.
        # To enable: map doc_type ↔ domain explicitly, or store both.
        retrieval_filter: Optional[Dict[str, str]] = None

        # ── Launch web search in parallel (waiting pool) ──
        # Web search runs concurrently with KB retrieval so results are
        # ready by the time we check the quality gate.  If KB quality is
        # sufficient the results are discarded.
        web_future: Optional[Future] = None
        _web_executor: Optional[ThreadPoolExecutor] = None
        if _use_web and self.web_searcher:
            _web_executor = ThreadPoolExecutor(max_workers=1)
            web_future = _web_executor.submit(self.web_searcher.search, query)
            if self.verbose:
                logger.info("[WEB] Web search launched in parallel (waiting pool).")

        # Retrieval + rerank + compression (timed)
        t_retrieval = time.time()

        # Retrieval
        retrieved: List[Tuple[Document, float]] = []

        for q in queries:
            retrieved.extend(
                self.retriever.retrieve(q, k=self.top_k, fusion_mode=self.fusion_mode, filter_dict=retrieval_filter)
            )

        if not retrieved:
            tracer.retrieval = {"num_raw": 0, "num_unique": 0}
            tracer.flush("No relevant documents found.")
            return "No relevant documents found."

        # Deduplicate
        unique_docs = {}
        for doc, score in retrieved:
            key = doc.page_content[:200]
            if key not in unique_docs:
                unique_docs[key] = (doc, score)

        retrieved_docs = [v[0] for v in unique_docs.values()]

        # Hierarchy expansion: for regulation chunks, pull in parent + siblings
        if self.chunking_mode == "hierarchical" and self.chunks:
            expanded = list(retrieved_docs)
            for doc in retrieved_docs:
                if doc.metadata.get("hierarchy_path"):
                    extra = expand_context(doc, self.chunks, direction="all", index=self._hierarchy_index)
                    for e in extra:
                        if e.page_content[:200] not in {
                            d.page_content[:200] for d in expanded
                        }:
                            expanded.append(e)
            retrieved_docs = expanded
            if self.verbose and len(expanded) > len(retrieved_docs):
                logger.info(
                    "Hierarchy expansion: %d → %d documents",
                    len(retrieved_docs), len(expanded)
                )

        if self.verbose:
            logger.info("Retrieved %d unique documents.", len(retrieved_docs))

        # Re-ranking
        if self.enable_rerank and self.reranker:
            reranked = self.reranker.rerank(
                retrieval_query, retrieved_docs, top_n=self.top_k
            )
            reranked_docs = [doc for doc, _ in reranked]
            if self.verbose:
                logger.info("Reranking applied.")
        else:
            reranked = []
            reranked_docs = retrieved_docs[:self.top_k]
            if self.verbose:
                logger.info("Reranking skipped.")

        # ── Retrieval quality gate (with web fallback) ──
        # Two thresholds serve different purposes:
        #   retrieval_quality_threshold → "not found" hard stop (no web search)
        #   web_search_quality_threshold  → trigger web fallback before giving up
        # When web search is enabled and KB quality is poor, wake web results
        # from the waiting pool and generate from them instead.
        _is_web_fallback = False
        _effective_threshold = (
            self.web_search_quality_threshold if (_use_web and self.web_searcher)
            else self.retrieval_quality_threshold
        )

        if self.enable_rerank and self.reranker and reranked:
            top_score = reranked[0][1]

            if top_score < _effective_threshold:
                # ── KB quality insufficient — try web fallback ──
                web_docs: List[Document] = []
                if web_future is not None:
                    try:
                        web_response = web_future.result(timeout=self.web_search_timeout)
                        if web_response.results:
                            web_docs = self._web_results_to_documents(web_response)
                            _is_web_fallback = True
                            if self.verbose:
                                logger.info(
                                    "[WEB] Fallback activated: KB score %.4f < "
                                    "threshold %.2f, using %d web results.",
                                    top_score, _effective_threshold, len(web_docs),
                                )
                        elif self.verbose:
                            logger.info(
                                "[WEB] No web results found — falling through to "
                                "'not found' message."
                            )
                    except Exception as e:
                        logger.warning("[WEB] Web search failed: %s", e)
                    finally:
                        if _web_executor:
                            _web_executor.shutdown(wait=False)
                            _web_executor = None

                if _is_web_fallback:
                    # Route to web-based generation (skip context compression
                    # on KB docs — web docs are already compact snippets)
                    compressed_docs = web_docs
                else:
                    # No web results either — return clean "not found"
                    if self.verbose:
                        logger.info(
                            "Retrieval quality gate: top score %.4f < threshold "
                            "%.2f — no relevant documents found.",
                            top_score, _effective_threshold,
                        )
                    msg = (
                        "您咨询的问题在现行制度中未找到相关内容。"
                        "建议您致电我行客服热线或前往就近网点，"
                        "由工作人员为您详细解答。"
                    )
                    tracer.record_stage("retrieval", (time.time() - t_retrieval) * 1000)
                    tracer.flush(msg)
                    if self.query_cache:
                        self.query_cache.store(query, msg)
                    return msg
            else:
                # ── KB quality sufficient — discard web results ──
                if web_future is not None:
                    web_future.cancel()
                if _web_executor:
                    _web_executor.shutdown(wait=False)
                    _web_executor = None
                if self.verbose and _use_web:
                    logger.info(
                        "[WEB] KB quality sufficient (%.4f >= %.2f) — "
                        "web results discarded.",
                        top_score, _effective_threshold,
                    )
        elif _web_executor:
            # Reranker disabled — discard web results, continue normally
            _web_executor.shutdown(wait=False)
            _web_executor = None
            if web_future:
                web_future.cancel()

        # Context compression (KB path only — web docs are already compact)
        if not _is_web_fallback and self.enable_compression and self.compressor:
            compressed_docs = self.compressor.compress_documents(
                query=retrieval_query,
                documents=reranked_docs,
            )
            if self.verbose:
                logger.info("Compression applied.")
        elif not _is_web_fallback:
            compressed_docs = reranked_docs
            if self.verbose:
                logger.info("Compression skipped.")

        # Record retrieval stage
        retrieval_ms = (time.time() - t_retrieval) * 1000
        tracer.record_stage("retrieval", retrieval_ms)
        tracer.retrieval = {
            "num_raw": len(retrieved),
            "num_unique": len(retrieved_docs),
            "num_final": len(compressed_docs),
            "rerank_applied": bool(self.enable_rerank and self.reranker),
            "compression_applied": bool(self.enable_compression and self.compressor),
            "top_sources": list(dict.fromkeys(
                d.metadata.get("source", "unknown")
                for d in compressed_docs
            ))[:5],
        }

        # Generation — include SQL results when available
        context_docs = compressed_docs

        if sql_result and sql_result["rows"]:
            # Build a synthetic document from SQL results for the generator
            sql_context = (
                f"[数据库查询结果]\n"
                f"SQL: {sql_result['sql']}\n"
                f"结果行数: {sql_result['row_count']}\n"
                f"数据:\n{sql_result['answer']}"
            )
            # Prepend SQL results as the first "document" for citation
            sql_doc = Document(
                page_content=sql_context,
                metadata={"source": "structured_database", "is_sql_result": True},
            )
            context_docs = [sql_doc] + list(compressed_docs)

        t_generation = time.time()
        if _is_web_fallback:
            answer = self.generator.generate_from_web(
                query=query,
                web_docs=compressed_docs,
            )
        else:
            answer = self.generator.generate_with_citations(
                query=query,
                context_docs=context_docs,
            )
        tracer.record_stage("generation", (time.time() - t_generation) * 1000)

        # ── Post-generation safety cascade (L1 → L2 → L3) ──
        # For web-fallback answers: L1 still runs (rules-based, model-agnostic),
        # but L2/L3 are skipped — the cross-encoder and fact-checker are trained
        # on internal documents and produce unreliable results on web snippets.
        t_safety = time.time()
        fc_result = None  # populated if L3 is triggered

        if _is_web_fallback:
            # L1 only — deterministic rules, no model dependency
            numeric_flags: List[dict] = []
            if self.numeric_guard and answer:
                numeric_flags = self.numeric_guard.verify(
                    answer=answer,
                    sql_result=None,
                    source_docs=compressed_docs,
                )
                if self.verbose and numeric_flags:
                    logger.info(
                        "[SAFETY-L1] NumericGuard flagged %d issue(s): %s",
                        len(numeric_flags),
                        ", ".join(f["rule"] for f in numeric_flags),
                    )
            nli_result = None
            needs_llm = False
            if self.verbose:
                logger.info("[SAFETY] Web fallback — L2/L3 skipped, L1 only.")
        else:
            # ── Standard KB safety cascade (L1 → L2 → L3) ──

            # Layer 1: Numeric guard — deterministic rules (0ms, 0 API)
            numeric_flags: List[dict] = []
            if self.numeric_guard and answer:
                numeric_flags = self.numeric_guard.verify(
                    answer=answer,
                    sql_result=sql_result,
                    source_docs=compressed_docs,
                )
                if self.verbose and numeric_flags:
                    logger.info(
                        "[SAFETY-L1] NumericGuard flagged %d issue(s): %s",
                        len(numeric_flags),
                        ", ".join(f["rule"] for f in numeric_flags),
                    )

            # Layer 2: NLI verification — cross-encoder (<100ms, 0 API)
            nli_result = None
            if self.nli_verifier and answer:
                nli_result = self.nli_verifier.verify(answer, compressed_docs)
                if self.verbose:
                    logger.info(
                        "[SAFETY-L2] NLIVerifier: %d/%d claims verified, ratio=%.2f",
                        nli_result["verified_count"], nli_result["total_claims"],
                        nli_result["unsupported_ratio"],
                    )

            # Decide: escalate to Layer 3 (LLM)?
            # Skip L3 when answer is a retrieval-failure rejection — no factual
            # claims to verify, just wasted API call.
            _is_rejection = answer and (
                "cannot find" in answer.lower()
                or "no relevant documents" in answer.lower()
                or "无法找到" in answer
                or "未找到相关" in answer
            )
            needs_llm = (
                not _is_rejection
                and (bool(numeric_flags) or (nli_result and nli_result["needs_llm_review"]))
            )

            if needs_llm and self.fact_checker and answer:
                # Layer 3: LLM deep review (~500ms, 1 API call)
                fc_result = self.fact_checker.verify(
                    question=query,
                    answer=answer,
                    source_docs=compressed_docs,
                )
                if self.verbose:
                    logger.info(
                        "[SAFETY-L3] FactChecker: %d claims, %d replaced, safe_to_show=%s",
                        fc_result["total_claims"], fc_result["replaced_count"],
                        fc_result["safe_to_show"],
                    )
                if fc_result["replaced_count"] > 0:
                    answer = fc_result["safe_answer"]
            elif nli_result and nli_result["unsupported_ratio"] > 0 and not _is_rejection:
                # L2 found issues but not severe enough for L3 — rewrite inline
                answer = self._rewrite_uncertain_claims(answer, nli_result, numeric_flags)
                if self.verbose:
                    logger.info("[SAFETY-L2] Rewrote uncertain claims inline (no LLM).")
            elif self.verbose:
                logger.info("[SAFETY] All claims verified — answer is clean.")

        # Record safety stage
        tracer.record_stage("safety", (time.time() - t_safety) * 1000)
        tracer.safety = {
            "L1_flags": len(numeric_flags),
            "L2_unsupported_ratio": nli_result["unsupported_ratio"] if nli_result else 0.0,
            "L2_total_claims": nli_result["total_claims"] if nli_result else 0,
            "L3_triggered": needs_llm and self.fact_checker is not None,
            "L3_replaced": fc_result.get("replaced_count", 0) if fc_result else 0,
            "web_fallback": _is_web_fallback,
        }

        # Prepend confirmation line when query was rewritten
        if rewritten_query:
            answer = QueryRewriter.confirmation_line(rewritten_query) + "\n\n" + answer

        # Append source transparency note for web-fallback answers
        if _is_web_fallback:
            answer += (
                "\n\n---\n"
                "**信息来源说明**：以上回答基于互联网公开信息整理，非本行内部制度原文。"
                "如需获取准确信息，请咨询我行客服热线或前往就近网点。"
            )

        # Append disclaimer for caution-level queries (KB path only)
        if not _is_web_fallback and route and route["action"] == "answer_with_disclaimer":
            answer += (
                "\n\n---\n"
                "**免责声明**：以上信息基于我行现行规定，仅供参考。"
                "具体业务办理请以网点工作人员告知或最新公告为准。"
                "如有疑问，请拨打客服热线咨询。"
            )

        # Store in semantic cache for future lookups
        if self.query_cache and answer:
            self.query_cache.store(query, answer)

        tracer.flush(answer)
        return answer

    @staticmethod
    def _rewrite_uncertain_claims(
        answer: str,
        nli_result: dict,
        numeric_flags: List[dict],
    ) -> str:
        """Replace claims flagged by L1/L2 with CS-redirect text inline.

        Called when L2 finds issues but they're not severe enough to
        warrant an LLM round-trip (L3).  When most claims are flagged,
        replaces the entire answer with a clean redirect instead of
        producing a garbled mix of parenthetical notes.
        """
        # Collect flagged claim texts from NLI + NumericGuard
        flagged_texts: List[str] = []
        if nli_result:
            for v in nli_result.get("verdicts", []):
                if v.get("verdict") == "uncertain":
                    flagged_texts.append(v["claim"])
        for f in numeric_flags:
            flagged_texts.append(f.get("claim", ""))

        if not flagged_texts:
            return answer

        total = nli_result.get("total_claims", 0) if nli_result else 0
        unsupported_ratio = nli_result.get("unsupported_ratio", 0.0) if nli_result else 0.0

        # When >50% of claims are unsupported (or all claims are flagged),
        # the retrieved docs are likely irrelevant.  Replace the entire
        # answer with a clean redirect instead of per-claim parenthesis spam.
        if unsupported_ratio > 0.5 or (total > 0 and len(flagged_texts) >= total):
            # Build a short summary of what was asked about
            topics = "、".join(
                c[:20] for c in list(dict.fromkeys(flagged_texts))[:3]
            )
            return (
                f"关于{'该问题' if not topics else topics}，"
                f"建议您拨打我行客服热线或前往就近网点咨询客户经理获取准确信息。"
            )

        # Deduplicate
        seen = set()
        unique = []
        for t in flagged_texts:
            if t not in seen:
                unique.append(t)
                seen.add(t)

        rewritten = answer
        for claim in unique[:3]:  # Cap at 3 replacements to avoid garbling
            topic = claim[:30]
            replacement = (
                f"（关于{topic}的具体信息，建议您咨询我行客服获取准确答复）"
            )
            if claim in rewritten:
                rewritten = rewritten.replace(claim, replacement)
            elif claim[:20] in rewritten:
                rewritten = rewritten.replace(claim[:20], replacement)

        # If no replacement happened (claims not found verbatim), append a note
        if rewritten == answer:
            topics = "、".join(c[:20] for c in unique[:3])
            rewritten += (
                f"\n\n关于{topics}的详细信息，建议您咨询我行客服获取准确答复。"
            )

        return rewritten

    @staticmethod
    def _web_results_to_documents(response: WebSearchResponse) -> List[Document]:
        """Convert web search results to LangChain Documents for generation.

        Each document contains title + URL + snippet — the LLM only sees
        these summaries, never full page content.
        """
        docs = []
        for r in response.results:
            content = (
                f"[网页标题] {r.title}\n"
                f"[网页链接] {r.url}\n"
                f"[内容摘要] {r.snippet}"
            )
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": r.url,
                    "title": r.title,
                    "is_web_result": True,
                },
            ))
        return docs

    def retrieve_for_evaluation(self, query: str, k: int):

        retrieved = self.retriever.retrieve(query, k=20, fusion_mode=self.fusion_mode)

        docs = [doc for doc, _ in retrieved]
        
        if self.enable_rerank and self.reranker:
            reranked = self.reranker.rerank(query, docs, top_n=k)
            return reranked

        return retrieved[:k]
