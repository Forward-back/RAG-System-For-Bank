from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import re

from src.ingestion.hierarchy_parser import HierarchyParser


class Chunking:

    # ------------------------------------------------------------------
    # Public chunking strategies
    # ------------------------------------------------------------------

    @staticmethod
    def recursive_chunking(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "？", "！", "；", "，", ".", "!", "?", ";", ",", " ", ""],
        )

        chunks = splitter.split_documents(documents)
        return [Chunking._normalize_page_meta(c) for c in chunks]

    @staticmethod
    def semantic_chunking(
        documents: List[Document],
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1000,
        similarity_percentile=30,
    ) -> List[Document]:

        model = SentenceTransformer(model_name, local_files_only=True)
        chunks = []

        for doc in documents:
            text = doc.page_content.strip()
            if not text:
                continue

            sentences = re.split(r"(?<=[.!?。！？])\s+", text)

            if len(sentences) <= 2:
                chunks.append(Chunking._normalize_page_meta(doc))
                continue

            embeddings = model.encode(sentences, normalize_embeddings=True)
            similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)

            threshold = np.percentile(similarities, similarity_percentile)

            current_chunk = []
            current_length = 0

            for i, sentence in enumerate(sentences):
                current_chunk.append(sentence)
                current_length += len(sentence)

                split_here = i < len(similarities) and similarities[i] < threshold
                size_exceed = current_length >= chunk_size

                if split_here or size_exceed:
                    chunk_text = " ".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(Document(
                            page_content=chunk_text,
                            metadata=doc.metadata.copy(),
                        ))
                    current_chunk = []
                    current_length = 0

            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata=doc.metadata.copy(),
                    ))

        return [Chunking._normalize_page_meta(c) for c in chunks]

    @staticmethod
    def element_aware_chunking(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_section_len: int = 100,
    ) -> List[Document]:
        """
        Chunking strategy that respects element types from unstructured.

        Design rules:
        - Title elements are prepended to the following content for heading context.
        - Table elements are never split — they emit as atomic chunks.
        - Consecutive ListItems are grouped into a single chunk.
        - NarrativeText is accumulated and split recursively at chunk_size boundaries.

        Falls back to recursive_chunking when no documents carry element_type metadata.
        """
        if not any(d.metadata.get("element_type") for d in documents):
            return Chunking.recursive_chunking(documents, chunk_size, chunk_overlap)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "？", "！", "；", "，", ".", "!", "?", ";", ",", " ", ""],
        )

        # Group by source so each file is processed in reading order
        sources: dict = {}
        for doc in documents:
            src = doc.metadata.get("source", "__unknown__")
            sources.setdefault(src, []).append(doc)

        result = []
        for src_docs in sources.values():
            result.extend(
                Chunking._element_chunk_one_file(src_docs, splitter, chunk_size, min_section_len)
            )

        return result

    @staticmethod
    def hierarchical_chunking(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """
        Chunk documents by their native heading hierarchy.

        Designed for Chinese regulatory documents with 第X章 / 第Y条 / (Z)
        numbering.  Parses the document structure, splits along article /
        paragraph boundaries, and tags every chunk with its full ancestry
        path in metadata.

        Segments that still exceed *chunk_size* are further split with
        recursive_chunking, but never across a hierarchy boundary.
        """
        from src.ingestion.hierarchy_parser import HierarchyParser

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "？", "！", "；", "，", ".", "!", "?", ";", ",", " ", ""],
        )

        # Group by source file — each file's pages are concatenated
        sources: dict = {}
        for doc in documents:
            src = doc.metadata.get("source", "__unknown__")
            sources.setdefault(src, []).append(doc)

        all_chunks: List[Document] = []

        for src, src_docs in sources.items():
            # Merge all pages from this source into one text, preserving page info
            full_text = "\n".join(d.page_content for d in src_docs)
            base_meta = src_docs[0].metadata.copy() if src_docs else {}

            segments = HierarchyParser.parse(full_text)
            if not segments:
                # No hierarchy detected — fall back to recursive
                all_chunks.extend(
                    Chunking.recursive_chunking(src_docs, chunk_size, chunk_overlap)
                )
                continue

            seg_docs = HierarchyParser.to_documents(segments, base_metadata=base_meta)

            for seg_doc in seg_docs:
                if len(seg_doc.page_content) <= chunk_size:
                    all_chunks.append(Chunking._normalize_page_meta(seg_doc))
                else:
                    # Oversized segment — split but preserve hierarchy metadata
                    sub_chunks = splitter.split_documents([seg_doc])
                    for sc in sub_chunks:
                        # Inherit hierarchy metadata from parent segment
                        for key in ("hierarchy_path", "hierarchy_level",
                                    "hierarchy_heading", "parent_heading",
                                    "article_heading"):
                            if key in seg_doc.metadata:
                                sc.metadata[key] = seg_doc.metadata[key]
                        all_chunks.append(Chunking._normalize_page_meta(sc))

        return all_chunks

    # ------------------------------------------------------------------
    # Page metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_page(page_val) -> int | None:
        """Parse a page value into an integer, or None if unavailable."""
        if page_val is None or page_val == "na" or page_val == "":
            return None
        try:
            return int(page_val)
        except (ValueError, TypeError):
            # Could be a range like "3-5" — take the first number
            m = re.match(r"(\d+)", str(page_val))
            return int(m.group(1)) if m else None

    @staticmethod
    def _collect_pages(docs: List[Document]) -> tuple:
        """Return (first_page, last_page) across a list of documents."""
        pages = []
        for d in docs:
            p = Chunking._parse_page(d.metadata.get("page"))
            if p is not None:
                pages.append(p)
            # Also check first_page / last_page from a prior merge
            fp = Chunking._parse_page(d.metadata.get("first_page"))
            lp = Chunking._parse_page(d.metadata.get("last_page"))
            if fp is not None:
                pages.append(fp)
            if lp is not None:
                pages.append(lp)

        if not pages:
            return ("na", "na")
        return (str(min(pages)), str(max(pages)))

    @staticmethod
    def _normalize_page_meta(doc: Document) -> Document:
        """
        Ensure every chunk carries first_page + last_page.

        Also normalises the old 'page' key so downstream code that reads
        'page' still works — set to first_page for single-page chunks,
        or 'first-last' for multi-page chunks.
        """
        first, last = Chunking._collect_pages([doc])

        doc.metadata["first_page"] = first
        doc.metadata["last_page"] = last

        if first == last:
            doc.metadata["page"] = first
        elif first != "na" and last != "na":
            doc.metadata["page"] = f"{first}-{last}"

        return doc

    # ------------------------------------------------------------------
    # Internal helpers for element-aware chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _element_chunk_one_file(
        docs: List[Document],
        splitter: RecursiveCharacterTextSplitter,
        chunk_size: int,
        min_section_len: int,
    ) -> List[Document]:
        """Chunk documents from a single source file in page/element order."""
        result = []
        pending_title = ""
        pending_title_doc: Document | None = None
        buffer: List[Document] = []
        buffer_len = 0

        for doc in docs:
            etype = doc.metadata.get("element_type", "")
            text = doc.page_content

            # ── Title: start a new section ──
            if etype == "Title":
                result.extend(
                    Chunking._flush_buffer(buffer, pending_title, pending_title_doc,
                                           splitter, min_section_len)
                )
                buffer = []
                buffer_len = 0
                pending_title = text
                pending_title_doc = doc
                continue

            # ── Table: emit standalone (never split) ──
            if etype == "Table":
                result.extend(
                    Chunking._flush_buffer(buffer, pending_title, pending_title_doc,
                                           splitter, min_section_len)
                )
                buffer = []
                buffer_len = 0

                # page_content = title + plain text (for BM25 retrieval)
                # text_as_html = raw HTML (for LLM generation)
                table_text = f"{pending_title}\n{text}" if pending_title else text
                html = doc.metadata.get("text_as_html", "")

                # Page range: title page + table page
                page_docs = [doc]
                if pending_title_doc:
                    page_docs.append(pending_title_doc)
                first, last = Chunking._collect_pages(page_docs)

                result.append(Document(
                    page_content=table_text,
                    metadata={
                        **doc.metadata,
                        "is_table": True,
                        "text_as_html": html or doc.metadata.get("text_as_html", ""),
                        "first_page": first,
                        "last_page": last,
                        "page": first if first == last else f"{first}-{last}",
                    },
                ))
                pending_title = ""
                pending_title_doc = None
                continue

            # ── NarrativeText / ListItem / UncategorizedText ──
            buffer.append(doc)
            buffer_len += len(text)

            if buffer_len >= chunk_size:
                result.extend(
                    Chunking._flush_buffer(buffer, pending_title, pending_title_doc,
                                           splitter, min_section_len)
                )
                buffer = []
                buffer_len = 0

        # Flush remaining buffer
        result.extend(
            Chunking._flush_buffer(buffer, pending_title, pending_title_doc,
                                   splitter, min_section_len)
        )

        return result

    @staticmethod
    def _flush_buffer(
        buffer: List[Document],
        title: str,
        title_doc: Document | None,
        splitter: RecursiveCharacterTextSplitter,
        min_section_len: int,
    ) -> List[Document]:
        """Merge buffer elements, prepend title, split if oversized."""
        if not buffer:
            return []

        merged_text = "\n\n".join(d.page_content for d in buffer)
        if title:
            merged_text = f"{title}\n{merged_text}"

        # Compute page range across buffer + optional title
        page_sources = list(buffer)
        if title_doc:
            page_sources.append(title_doc)
        first, last = Chunking._collect_pages(page_sources)

        base_meta = buffer[0].metadata.copy()
        base_meta["first_page"] = first
        base_meta["last_page"] = last
        base_meta["page"] = first if first == last else f"{first}-{last}"

        merged_doc = Document(
            page_content=merged_text,
            metadata=base_meta,
        )

        if len(merged_text) <= splitter._chunk_size:
            return [merged_doc]

        # When splitting, child chunks inherit the page range
        sub_chunks = splitter.split_documents([merged_doc])
        for sc in sub_chunks:
            sc.metadata["first_page"] = first
            sc.metadata["last_page"] = last
            sc.metadata["page"] = first if first == last else f"{first}-{last}"
        return sub_chunks
