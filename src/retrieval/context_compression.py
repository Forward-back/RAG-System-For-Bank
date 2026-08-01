from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from typing import List
import copy
import re

from src.generation.llm_client import DeepSeekLLM


class ContextCompressor:
    """
    Compresses documents using an LLM.
    Extracts only the most relevant sentences for a query.

    Table chunks (doc.metadata["is_table"] == True) are passed through
    uncompressed — their structure carries meaning that sentence-level
    extraction destroys, and they are already compact by nature.
    """

    def __init__(self, llm: DeepSeekLLM, max_chars: int = 3000, max_workers: int = 2,
                 min_chars_for_compression: int = 1000):
        self.llm = llm
        self.max_chars = max_chars
        self.max_workers = max_workers
        self.min_chars_for_compression = min_chars_for_compression

    @staticmethod
    def _extract_text(text: str) -> str:
        if not text:
            return ""
        return str(text).strip()

    def _compress_one(self, query: str, prompt_template: str, doc: Document) -> Document:
        """Compress a single document. Called by worker threads."""
        text = doc.page_content.strip()
        if not text:
            return None

        prompt = prompt_template.format(query=query, text=text[:self.max_chars])

        try:
            response = self.llm.generate(prompt)
            relevant_text = self._extract_text(response)

            if not relevant_text:
                return Document(
                    page_content=text[:self.max_chars],
                    metadata=copy.deepcopy(doc.metadata),
                )

            clean = relevant_text.strip().lower()

            if clean.startswith("none") or clean.startswith("无") or clean == "n/a":
                return None  # LLM says nothing relevant

            return Document(
                page_content=relevant_text,
                metadata=copy.deepcopy(doc.metadata),
            )

        except Exception:
            return Document(
                page_content=text[:self.max_chars],
                metadata=copy.deepcopy(doc.metadata),
            )

    def compress_documents(
        self,
        query: str,
        documents: List[Document],
        max_docs: int = 5,
    ) -> List[Document]:

        if not query or not query.strip():
            return []

        if not documents:
            return []

        # Split: tables pass through, short text passes through, long text gets compressed
        tables: List[Document] = []
        texts: List[Document] = []
        for doc in documents[:max_docs]:
            if doc.metadata.get("is_table"):
                tables.append(doc)
            elif doc.page_content.strip():
                # Skip compression for short docs — LLM overhead not worth it
                if len(doc.page_content) < self.min_chars_for_compression:
                    tables.append(doc)  # pass through like tables
                else:
                    texts.append(doc)

        has_chinese = bool(re.search(r'[一-鿿]', query))

        if has_chinese:
            prompt_template = """
从以下文档中提取与问题相关的所有句子。
如果文档中没有相关内容，请回复：无。
不要添加解释或额外文字。

问题：
{query}

文档：
{text}

相关句子：
""".strip()
        else:
            prompt_template = """
Extract only the sentences that directly answer or are relevant to the question below.
If no sentence is relevant, output exactly: None.
Do not add explanations or extra text.

Question:
{query}

Document:
{text}

Relevant sentences:
""".strip()

        compressed: List[Document] = []

        # Compress text documents in parallel
        if texts:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(texts))) as executor:
                futures = {
                    executor.submit(self._compress_one, query, prompt_template, doc): idx
                    for idx, doc in enumerate(texts)
                }
                results = {}  # idx → Document | None
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception:
                        # If a worker fails, keep the original
                        results[idx] = Document(
                            page_content=texts[idx].page_content[:self.max_chars],
                            metadata=copy.deepcopy(texts[idx].metadata),
                        )
                compressed.extend(
                    results[i] for i in sorted(results) if results[i] is not None
                )

        # Tables pass through unchanged (preserving text_as_html for the generator)
        compressed.extend(tables)

        if not compressed:
            return documents[:max_docs]

        return compressed
