from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Optional
import logging
import numpy as np
import jieba

logger = logging.getLogger(__name__)

class HybridRetriever:

    def __init__(self,vectorstore,documents:List[Document]):

        self.vectorstore=vectorstore
        self.documents=documents
        self.bm25 = None
        if documents:
            self._build_bm25(documents)

        logger.info("Hybrid Retriever initialized with %d documents", len(documents))


    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for BM25. Uses jieba for Chinese, whitespace for English."""
        if not text:
            return []
        # Detect Chinese characters
        has_chinese = any('一' <= c <= '鿿' for c in text)
        if has_chinese:
            return [w for w in jieba.lcut(text) if w.strip()]
        return text.lower().split()

    @staticmethod
    def _doc_uid(doc:Document)->str:

        src=doc.metadata.get("source","unknown")
        page=doc.metadata.get("page","na")
        return f"{src}::page={page}::hash={hash(doc.page_content)}"



    @staticmethod
    def _normalize(scores:np.ndarray)->np.ndarray:
        if len(scores)==0:
            return scores

        min_s,max_s=scores.min(),scores.max()
        if max_s>min_s:
            return (scores-min_s)/(max_s-min_s)

        return scores


    def _build_bm25(self,documents:List[Document])->None:
        tokenized_docs=[
            self._tokenize(doc.page_content)
            for doc in documents
        ]
        self.bm25=BM25Okapi(tokenized_docs)


    def refresh_documents(self,documents:List[Document])->None:

        self.documents=documents
        self._build_bm25(documents)


    @staticmethod
    def _rrf_fuse(
        dense_docs: List[Document],
        bm25_docs: List[Document],
        bm25_indices: List[int],
        all_documents: List[Document],
        k: int,
        rrf_k: int = 60,
    ) -> List[Tuple[Document, float]]:
        """Fuse dense and sparse rankings via Reciprocal Rank Fusion (RRF).

        RRF is insensitive to score distribution differences between retrievers
        and requires no normalization — it uses only rank positions.
        """
        score_map: Dict[str, Dict] = {}

        for rank, doc in enumerate(dense_docs):
            uid = HybridRetriever._doc_uid(doc)
            score_map[uid] = {"doc": doc, "score": 1.0 / (rrf_k + rank + 1)}

        for rank, idx in enumerate(bm25_indices):
            doc = bm25_docs[idx] if bm25_docs else all_documents[idx]
            uid = HybridRetriever._doc_uid(doc)
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if uid in score_map:
                score_map[uid]["score"] += rrf_score
            else:
                score_map[uid] = {"doc": doc, "score": rrf_score}

        ranked = sorted(
            score_map.values(),
            key=lambda x: (x["score"], x["doc"].page_content),
            reverse=True,
        )[:k]

        return [(item["doc"], float(item["score"])) for item in ranked]

    @staticmethod
    def _weighted_fuse(
        dense_docs: List[Document],
        dense_scores: np.ndarray,
        bm25_docs: List[Document],
        bm25_indices: List[int],
        bm25_scores: np.ndarray,
        all_documents: List[Document],
        k: int,
        alpha: float,
    ) -> List[Tuple[Document, float]]:
        """Fuse dense and sparse rankings via weighted-sum fusion."""
        score_map: Dict[str, Dict] = {}

        for i, doc in enumerate(dense_docs):
            uid = HybridRetriever._doc_uid(doc)
            score_map[uid] = {"doc": doc, "score": alpha * dense_scores[i]}

        for rank, idx in enumerate(bm25_indices):
            doc = bm25_docs[idx] if bm25_docs else all_documents[idx]
            uid = HybridRetriever._doc_uid(doc)
            sparse_score = (1 - alpha) * bm25_scores[rank]
            if uid in score_map:
                score_map[uid]["score"] += sparse_score
            else:
                score_map[uid] = {"doc": doc, "score": sparse_score}

        ranked = sorted(
            score_map.values(),
            key=lambda x: (x["score"], x["doc"].page_content),
            reverse=True,
        )[:k]

        return [(item["doc"], float(item["score"])) for item in ranked]

    def retrieve(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5,
        bm25_k: int = 50,
        fusion_mode: str = "weighted_sum",
        filter_dict: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float]]:

        if not query or not query.strip():
            return []

        if not self.documents:
            return []

        # Build ChromaDB where clause from filter_dict (e.g. {"domain": "regulation"})
        where_clause = filter_dict if filter_dict else None

        # Dense retrieval — optionally scoped by domain/doc_type
        dense_results = self.vectorstore.similarity_search_with_score(
            query,
            k=k * 2,
            filter=where_clause,
        )

        dense_docs = [doc for doc, _ in dense_results]
        dense_distances = np.array([score for _, score in dense_results])

        if len(dense_distances) == 0:
            return []

        dense_scores = 1 / (1 + dense_distances)
        dense_scores = self._normalize(dense_scores)

        # BM25 retrieval — pre-filter when a domain filter is active
        if self.bm25 is None:
            return []

        if filter_dict:
            filter_field, filter_value = next(iter(filter_dict.items()))
            filtered_docs = [
                doc for doc in self.documents
                if doc.metadata.get(filter_field) == filter_value
            ]
            if not filtered_docs:
                pass
            else:
                temp_bm25 = BM25Okapi([self._tokenize(d.page_content) for d in filtered_docs])
                tokenized_query = self._tokenize(query)
                bm25_raw = temp_bm25.get_scores(tokenized_query)
                top_idx = np.argsort(bm25_raw)[::-1][:bm25_k]
                bm25_scores_norm = self._normalize(bm25_raw[top_idx])

                if fusion_mode == "rrf":
                    return self._rrf_fuse(
                        dense_docs, filtered_docs, top_idx, self.documents, k
                    )
                else:
                    return self._weighted_fuse(
                        dense_docs, dense_scores,
                        filtered_docs, top_idx, bm25_scores_norm,
                        self.documents, k, alpha,
                    )

        # ── Unfiltered path ──
        tokenized_query = self._tokenize(query)
        bm25_raw_score = self.bm25.get_scores(tokenized_query)

        top_bm25_idx = np.argsort(bm25_raw_score)[::-1][:bm25_k]
        bm25_scores_norm = self._normalize(bm25_raw_score[top_bm25_idx])

        if fusion_mode == "rrf":
            return self._rrf_fuse(
                dense_docs, self.documents, top_bm25_idx, self.documents, k
            )
        else:
            return self._weighted_fuse(
                dense_docs, dense_scores,
                self.documents, top_bm25_idx, bm25_scores_norm,
                self.documents, k, alpha,
            )

