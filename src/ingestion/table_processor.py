"""
Table keyword extraction for BM25-friendly retrieval.

Tables are kept as atomic chunks by element_aware chunking.  This module
replaces the old NL-conversion approach with lightweight keyword extraction:
the raw HTML is preserved in ``text_as_html`` for the generator, while
``page_content`` gets a keyword-dense summary optimised for BM25 matching.

Only one strategy: **rule** (deterministic, no API cost).
"""

from __future__ import annotations

import csv
import io
import logging
import re
from html.parser import HTMLParser
from typing import Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight HTML table parser
# ---------------------------------------------------------------------------

class _TableHTMLParser(HTMLParser):
    """Pull rows/cells out of a simple <table> fragment."""

    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._current_cell = ""
        self._in_td = False

    def handle_starttag(self, tag: str, attrs):
        if tag in ("td", "th"):
            self._in_td = True
            self._current_cell = ""

    def handle_endtag(self, tag: str):
        if tag in ("td", "th"):
            self._in_td = False
            cell = self._current_cell.strip()
            self._current_row.append(cell)
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data: str):
        if self._in_td:
            self._current_cell += data


# ---------------------------------------------------------------------------
# Table processor (keyword extraction, post-chunking)
# ---------------------------------------------------------------------------

class TableProcessor:
    """
    Post-chunking keyword extraction for table chunks.

    Runs AFTER chunking so that table boundaries are already respected.
    Only processes documents with ``metadata["is_table"] == True``.

    Usage::

        chunks = Chunking.element_aware_chunking(docs)
        chunks = TableProcessor.process(chunks)
    """

    @staticmethod
    def process(
        documents: List[Document],
        verbose: bool = True,
    ) -> List[Document]:
        """Extract keywords from table chunks.  Non-table docs pass through unchanged."""
        updated = []
        table_count = 0
        csv_count = 0

        for doc in documents:
            is_html_table = doc.metadata.get("is_table") and not TableProcessor._is_csv_doc(doc)
            is_csv = TableProcessor._is_csv_doc(doc)

            if is_csv:
                csv_text = doc.page_content
                keywords = TableProcessor._extract_csv_keywords(csv_text)
                if not keywords:
                    updated.append(doc)
                    continue

                csv_count += 1
                updated.append(Document(
                    page_content=keywords,
                    metadata={
                        **doc.metadata,
                        "is_table": True,
                        "source_format": "csv",
                    },
                ))
                continue

            if not is_html_table:
                updated.append(doc)
                continue

            html = doc.metadata.get("text_as_html", "")
            keywords = TableProcessor._extract_table_keywords(html)

            if not keywords:
                updated.append(doc)
                continue

            table_count += 1
            updated.append(Document(
                page_content=keywords,
                metadata={
                    **doc.metadata,
                    # text_as_html kept from original chunk metadata for generator
                },
            ))

        if verbose and (table_count or csv_count):
            logger.info(
                "Extracted keywords for %d HTML tables + %d CSV files",
                table_count, csv_count
            )

        return updated

    # ------------------------------------------------------------------
    # Keyword extraction (rule-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_table_keywords(html: str) -> str:
        """Parse HTML table and produce a keyword-dense string for BM25 retrieval.

        Output format::

            表格列: 产品名称 | 利率 | 期限
            活期存款 0.35% 无 一年定存 1.75% 12个月 ...
        """
        if not html or "<table" not in html.lower():
            return ""

        parser = _TableHTMLParser()
        try:
            parser.feed(html)
        except Exception:
            return ""

        rows = parser.rows
        if not rows:
            return ""

        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []

        parts = []

        if header:
            parts.append("表格列: " + " | ".join(h for h in header if h))

        if not data_rows:
            parts.append(" | ".join(h for h in header if h))
            return "\n".join(parts).strip()

        # Flatten cell values into a keyword-dense block
        all_values: List[str] = []
        for row in data_rows:
            for cell in row:
                cell = cell.strip()
                if cell and cell not in all_values:
                    all_values.append(cell)

        if all_values:
            parts.append(" ".join(all_values))

        return "\n".join(parts).strip()

    # ------------------------------------------------------------------
    # CSV detection & keyword extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _is_csv_doc(doc: Document) -> bool:
        """Return True if *doc* originated from a CSV file."""
        source = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "")
        return source.endswith(".csv") or filename.endswith(".csv")

    @staticmethod
    def _parse_csv_text(text: str) -> List[List[str]]:
        """Parse CSV-formatted text into rows of cells."""
        if not text:
            return []
        try:
            reader = csv.reader(io.StringIO(text))
            rows = [row for row in reader if any(cell.strip() for cell in row)]
            return rows
        except Exception:
            lines = [l for l in text.strip().split("\n") if l.strip()]
            return [re.split(r"\s*[,，]\s*", l) for l in lines]

    @staticmethod
    def _extract_csv_keywords(text: str) -> str:
        """Convert CSV text into a keyword-dense block for BM25 retrieval."""
        rows = TableProcessor._parse_csv_text(text)
        if not rows or len(rows) < 1:
            return ""

        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []

        parts = []
        if header:
            parts.append("表格列: " + " | ".join(h for h in header if h))

        if not data_rows:
            parts.append(" | ".join(h for h in header if h))
            return "\n".join(parts).strip()

        all_values: List[str] = []
        for row in data_rows:
            for cell in row:
                cell = cell.strip()
                if cell and cell not in all_values:
                    all_values.append(cell)

        if all_values:
            parts.append(" ".join(all_values))

        return "\n".join(parts).strip()
