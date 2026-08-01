from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List, Tuple, Optional
import logging
import torch
import os

logger = logging.getLogger(__name__)


class ReRanker:
    """Cross-encoder re-ranker for improving retrieval precision.

    Default model is ``BAAI/bge-reranker-v2-m3``, a multilingual reranker
    with strong Chinese performance. Alternative: ``maidalun1020/bce-reranker-base_v1``
    (Chinese-only, smaller, faster).

    Set ``use_quantization=True`` to apply PyTorch dynamic INT8 quantization.
    This reduces model size ~2x and can improve CPU inference speed.
    Quantized models run on CPU only.
    """

    def __init__(
        self,
        model_name: str = "E:/models/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 16,
        use_quantization: bool = False,
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.use_quantization = use_quantization

        if use_quantization:
            device = "cpu"
            logger.info("INT8 quantization requested — forcing CPU device.")
        elif device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading re-ranker model on %s: %s", device, model_name)
        self.model = CrossEncoder(
            model_name,
            device=device,
            local_files_only=local_files_only,
        )
        self.batch_size = batch_size

        # Log model size on disk (cache dir)
        self._log_model_size()

        # Apply dynamic INT8 quantization to Linear layers
        if use_quantization:
            self._apply_quantization()

        logger.info("Re-Ranker ready")

    def _log_model_size(self):
        """Log the model size on disk."""
        try:
            model_dir = self.model_name
            if not os.path.isabs(model_dir):
                cache_root = os.path.expanduser(
                    os.environ.get("HF_HOME", "~/.cache/huggingface")
                )
                model_dir = os.path.join(cache_root, "hub", f"models--{model_dir.replace('/', '--')}")
            if os.path.isdir(model_dir):
                total = 0
                for root, _, files in os.walk(model_dir):
                    for f in files:
                        total += os.path.getsize(os.path.join(root, f))
                logger.info("Model disk size: %.1f MB", total / (1024 * 1024))
        except Exception:
            pass

    def _apply_quantization(self):
        """Apply PyTorch dynamic quantization to Linear layers."""
        try:
            original_params = sum(
                p.numel() * p.element_size() for p in self.model.model.parameters()
            )
            self.model.model = torch.quantization.quantize_dynamic(
                self.model.model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            logger.info(
                "INT8 quantization applied — param memory: %.0f MB → ~%.0f MB",
                original_params / (1024 * 1024),
                original_params / (2 * 1024 * 1024),
            )
        except Exception as e:
            logger.warning("INT8 quantization failed (%s). Continuing with FP32.", e)
            self.use_quantization = False

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[Tuple[Document, float]]:

        if not query or not query.strip():
            return []

        if not documents:
            return []

        pairs = []
        valid_docs = []

        for doc in documents:
            text = doc.page_content.strip()
            if not text:
                continue
            pairs.append([query, text])
            valid_docs.append(doc)

        if not pairs:
            return []

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scored_docs = list(zip(valid_docs, scores))

        scored_docs.sort(
            key=lambda x: (float(x[1]), x[0].page_content),
            reverse=True,
        )

        return scored_docs[:top_n]
