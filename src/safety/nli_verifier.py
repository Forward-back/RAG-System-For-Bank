"""
Layer 2 — NLI Verifier (cross-encoder, <100ms, 0 API calls).

Uses a cross-encoder model (shared with the ReRanker) as a zero-shot
Natural Language Inference scorer. Unlike Layer 3 (LLM-based FactChecker),
the verifier runs a **different model architecture** from the generator
(DeepSeek), breaking the self-check cycle.

Claim decomposition is done via regex sentence splitting — no LLM needed.
Each claim is scored against every source document; the best score per
claim determines whether it is supported.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class NLIVerifier:
    """Cross-encoder based claim verification.

    Usage::

        # Reuse existing reranker model (recommended)
        verifier = NLIVerifier(model=reranker.model)

        # Or load independently
        verifier = NLIVerifier()

        result = verifier.verify(answer, source_docs)
        # → {"verdicts": [...], "unsupported_ratio": 0.2, ...}
    """

    # Chinese + English sentence boundary pattern
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?\n])\s*")

    # Lines that are not factual claims (skip during decomposition)
    _SKIP_PATTERNS = [
        re.compile(r"^(免责声明|请注意|温馨提示|以上信息|具体)"),
        re.compile(r"^(---|\*\*.*\*\*)$"),
        re.compile(r"^[\(（]?[Ss]ource \d+[\)）]?"),
    ]

    def __init__(
        self,
        model=None,
        model_name: str = "E:/models/bge-reranker-v2-m3",
        device: Optional[str] = None,
        threshold: float = 0.25,
        batch_size: int = 32,
    ):
        """
        Args:
            model:      Existing CrossEncoder instance (e.g. from ReRanker).
                        If None, loads a new one from *model_name*.
            model_name: HuggingFace model ID (only used when *model* is None).
            device:     "cuda" / "cpu". Auto-detect if None.
            threshold:  Cross-encoder score below which a claim is considered
                        unsupported (0.0–1.0). Lower = stricter.
            batch_size: Max pairs per predict() call.
        """
        if model is not None:
            self.model = model
        else:
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CrossEncoder(model_name, device=device, local_files_only=True)

        self.threshold = threshold
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        answer: str,
        source_docs: List[Document],
    ) -> dict:
        """Verify *answer* claims against *source_docs*.

        Returns::

            {
                "verdicts": [
                    {"claim": str, "verdict": "verified|uncertain",
                     "best_score": float, "best_source": str},
                    ...
                ],
                "total_claims":       int,
                "verified_count":     int,
                "uncertain_count":    int,
                "unsupported_ratio":  float,   # 0.0–1.0
                "needs_llm_review":   bool,    # True if ratio > 0.3
            }
        """
        if not answer or not answer.strip():
            return self._empty_result()

        if not source_docs:
            return self._empty_result()

        claims = self._decompose_claims(answer)
        if not claims:
            return self._empty_result()

        # Prepare source texts (truncated to reasonable length)
        source_texts = [doc.page_content[:2000] for doc in source_docs]

        verdicts = []
        for claim in claims:
            best_score, best_source = self._score_claim(claim, source_texts)
            verdict = "verified" if best_score >= self.threshold else "uncertain"
            verdicts.append({
                "claim": claim,
                "verdict": verdict,
                "best_score": round(float(best_score), 4),
                "best_source": best_source[:80],
            })

        verified = sum(1 for v in verdicts if v["verdict"] == "verified")
        uncertain = len(verdicts) - verified
        ratio = uncertain / len(verdicts) if verdicts else 0.0

        return {
            "verdicts": verdicts,
            "total_claims": len(verdicts),
            "verified_count": verified,
            "uncertain_count": uncertain,
            "unsupported_ratio": round(ratio, 4),
            "needs_llm_review": ratio > 0.5,
        }

    # ------------------------------------------------------------------
    # Claim decomposition (regex, not LLM)
    # ------------------------------------------------------------------

    @classmethod
    def _decompose_claims(cls, text: str) -> List[str]:
        """Split *text* into atomic claims on sentence boundaries.

        Filters out:
          - Lines under 8 characters (too short to be a claim)
          - Disclaimer / meta lines matching skip patterns
        """
        raw = cls._SENTENCE_BOUNDARY.split(text)
        claims = []
        for line in raw:
            line = line.strip()
            if len(line) < 8:
                continue
            if any(p.search(line) for p in cls._SKIP_PATTERNS):
                continue
            claims.append(line)
        return claims

    # ------------------------------------------------------------------
    # Claim scoring
    # ------------------------------------------------------------------

    def _score_claim(
        self, claim: str, source_texts: List[str]
    ) -> Tuple[float, str]:
        """Return (best_score, best_source_snippet) for a single claim.

        Builds (claim, source_text) pairs and batch-scores them through
        the cross-encoder. The highest score across all sources wins.
        """
        if not source_texts:
            return 0.0, ""

        pairs = [[claim, src] for src in source_texts]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        scores = np.asarray(scores, dtype=float)

        best_idx = int(np.argmax(scores))
        return float(scores[best_idx]), source_texts[best_idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict:
        return {
            "verdicts": [],
            "total_claims": 0,
            "verified_count": 0,
            "uncertain_count": 0,
            "unsupported_ratio": 0.0,
            "needs_llm_review": False,
        }
