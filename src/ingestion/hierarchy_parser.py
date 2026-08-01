"""
Parse the heading/level hierarchy of Chinese regulatory documents.

Chinese banking regulations follow a standard numbering scheme::

    第一章 总则              ← level 1  (chapter)
      第一节 适用范围         ← level 2  (section, optional)
        第一条 为了...        ← level 3  (article)
          (一) 商业银行...    ← level 4  (paragraph)
            1. 具体事项...    ← level 5  (sub-paragraph)
          (二) ...
        第二条 ...

The parser uses a stack to track the current path and splits the
document into segments, each tagged with its full ancestry path.

This enables:
  - Chunk boundaries that respect document structure (never split mid-article)
  - Retrieval-time context expansion (walk up the path to the parent article)
  - Citation provenance ("《零售业务规范》第三章 第12条 (三)")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Numbering conversion
# ---------------------------------------------------------------------------

_CN_NUM = "零一二三四五六七八九十百千"
_CN_NUM_MAP: Dict[str, int] = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000,
}
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_ROMAN = "ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ"


def _cn_to_int(s: str) -> int:
    """Convert a Chinese number string to int.

    >>> _cn_to_int("十二") → 12
    >>> _cn_to_int("一百二十三") → 123
    """
    s = s.strip()
    if not s:
        return 0

    # Handle "十" at the start → "一十"
    if s.startswith("十"):
        s = "一" + s

    total = 0
    section = 0
    for ch in s:
        if ch not in _CN_NUM_MAP:
            continue
        v = _CN_NUM_MAP[ch]
        if v >= 10:
            section = (section or 1) * v
            if v >= 100:
                total += section
                section = 0
        else:
            section = v
    total += section
    return total


# ---------------------------------------------------------------------------
# Hierarchy patterns
# ---------------------------------------------------------------------------

@dataclass
class _LevelDef:
    """Definition of one heading level in the hierarchy."""
    level: int
    name: str                          # "chapter", "article", etc.
    pattern: str                       # compiled regex
    is_optional: bool = False          # can this level be skipped?


# Patterns are anchored to line start and must be preceded by whitespace or BOS.
# This avoids matching "第一条" in the middle of a sentence like
# "这是第一条重要规则。"

_HIERARCHY_LEVELS: List[_LevelDef] = [
    _LevelDef(1, "chapter",    r"第[一二三四五六七八九十百千]+章"),
    _LevelDef(2, "section",    r"第[一二三四五六七八九十百千]+节"),
    _LevelDef(3, "article",    r"第[一二三四五六七八九十百千]+条"),
    _LevelDef(4, "paragraph",  r"[（(][一二三四五六七八九十]+[）)]"),
    _LevelDef(5, "sub_para",   r"\d+[\.、．]"),
    _LevelDef(6, "item",       r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]"),
]

# Compile each pattern with the line-start anchor
for _ld in _HIERARCHY_LEVELS:
    _ld.pattern = re.compile(r"^(\s*)" + _ld.pattern)


# ---------------------------------------------------------------------------
# Segment dataclass
# ---------------------------------------------------------------------------

@dataclass
class HierarchySegment:
    """A contiguous piece of text with its full hierarchy path."""
    content: str
    path: List[str] = field(default_factory=list)   # e.g. ["第一章 总则", "第一条", "(一)"]
    level: int = 0
    heading: str = ""                                # the heading text of this segment
    start_line: int = 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class HierarchyParser:
    """
    Parse a Chinese regulation document into hierarchy-tagged segments.

    Usage::

        segments = HierarchyParser.parse(document_text)
        for seg in segments:
            print(seg.level, " → ".join(seg.path))
            print(seg.content[:80])
    """

    @staticmethod
    def parse(text: str) -> List[HierarchySegment]:
        """
        Split *text* along chapter / article / paragraph boundaries.

        Returns a list of HierarchySegment covering the entire document.
        Segments are ordered as they appear in the source.
        """
        if not text or not text.strip():
            return []

        lines = text.split("\n")
        segments: List[HierarchySegment] = []

        # Stack tracks the current path: [(level, heading), ...]
        stack: List[Tuple[int, str]] = []
        current_content: List[str] = []
        current_heading = "前言"
        current_level = 0
        seg_start_line = 0

        for line_no, raw_line in enumerate(lines):
            matched = HierarchyParser._match_line(raw_line)

            if matched is not None:
                lvl, heading = matched

                # Flush current segment
                body = "\n".join(current_content).strip()
                if body:
                    path = [h for _, h in stack]  # snapshot before popping
                    segments.append(HierarchySegment(
                        content=body,
                        path=list(path),
                        level=current_level,
                        heading=current_heading,
                        start_line=seg_start_line,
                    ))

                # Pop stack: remove entries at this level or deeper
                while stack and stack[-1][0] >= lvl:
                    stack.pop()

                # Push new heading
                stack.append((lvl, heading))

                current_content = [raw_line]
                current_level = lvl
                current_heading = heading
                seg_start_line = line_no
            else:
                current_content.append(raw_line)

        # Flush final segment
        body = "\n".join(current_content).strip()
        if body:
            path = [h for _, h in stack]
            segments.append(HierarchySegment(
                content=body,
                path=list(path),
                level=current_level,
                heading=current_heading,
                start_line=seg_start_line,
            ))

        return segments

    @staticmethod
    def _match_line(line: str) -> Optional[Tuple[int, str]]:
        """
        Try to match *line* against each hierarchy level, highest priority first.

        Returns (level, heading_text) or None if no match.
        """
        for ld in _HIERARCHY_LEVELS:
            m = ld.pattern.match(line)
            if m:
                heading = line[m.end():].strip()
                full_heading = m.group(0).strip()
                if heading:
                    full_heading += " " + heading
                return (ld.level, full_heading)
        return None

    # ------------------------------------------------------------------
    # Convert to LangChain Documents (for downstream chunking / indexing)
    # ------------------------------------------------------------------

    @staticmethod
    def to_documents(
        segments: List[HierarchySegment],
        base_metadata: Optional[dict] = None,
    ) -> List[Document]:
        """
        Convert a list of HierarchySegments into LangChain Documents.

        Each document's metadata carries::

            {
                "hierarchy_path":    ["第一章 总则", "第一条"],
                "hierarchy_level":   3,
                "hierarchy_heading": "第一条 为了...",
                "parent_heading":    "第一章 总则",     # immediate parent
                "article_heading":   "第一条 为了...",   # nearest article ancestor
            }
        """
        base = dict(base_metadata or {})
        docs = []

        for seg in segments:
            meta = dict(base)
            meta["hierarchy_path"] = seg.path
            meta["hierarchy_level"] = seg.level
            meta["hierarchy_heading"] = seg.heading

            # Immediate parent = last entry in path before this segment
            if len(seg.path) >= 1:
                meta["parent_heading"] = seg.path[-1]
            else:
                meta["parent_heading"] = ""

            # Nearest article ancestor (level 3)
            article = ""
            for h in reversed(seg.path):
                if re.match(r"第[一二三四五六七八九十百千]+条", h):
                    article = h
                    break
            meta["article_heading"] = article

            docs.append(Document(
                page_content=seg.content,
                metadata=meta,
            ))

        return docs


# ---------------------------------------------------------------------------
# Context expansion utilities (for retrieval-time use)
# ---------------------------------------------------------------------------

def build_hierarchy_index(chunks: List[Document]) -> dict:
    """Build O(1) lookup indices for hierarchy-aware context expansion.

    Returns a dict with:
        ``"by_heading"``: ``{(source, heading): [chunks]}`` — parent lookup
        ``"by_parent"``:  ``{(source, parent_path, level): [chunks]}`` — sibling/child lookup
    """
    by_heading: dict = {}
    by_parent: dict = {}

    for doc in chunks:
        source = doc.metadata.get("source", "")
        heading = doc.metadata.get("hierarchy_heading", "")
        path = tuple(doc.metadata.get("hierarchy_path", []))
        level = doc.metadata.get("hierarchy_level", 0)

        if heading:
            by_heading.setdefault((source, heading), []).append(doc)
        if path:
            by_parent.setdefault((source, path[:-1], level), []).append(doc)

    return {"by_heading": by_heading, "by_parent": by_parent}


def expand_context(
    doc: Document,
    all_docs: List[Document],
    direction: str = "parent",
    index: Optional[dict] = None,
) -> List[Document]:
    """
    Expand a retrieved document's context using hierarchy metadata.

    Args:
        doc:       The retrieved document.
        all_docs:  All documents (fallback when *index* is None).
        direction: "parent"  → return parent article (one level up)
                   "children" → return all children of the same parent
                   "siblings" → return siblings under the same parent
                   "all"      → parent + siblings
        index:     Optional pre-built index from :func:`build_hierarchy_index`.
                   When provided, lookups are O(1) instead of O(n).

    Returns:
        Additional documents to include in the context window.
    """
    path = doc.metadata.get("hierarchy_path", [])
    if not path:
        return []

    source = doc.metadata.get("source", "")

    # ── Fast path: use pre-built index ──
    if index is not None:
        by_heading = index["by_heading"]
        by_parent = index["by_parent"]

        if direction == "parent":
            parent_heading = path[-1] if path else ""
            return list(by_heading.get((source, parent_heading), [])[:1])

        if direction == "siblings":
            pkey = (source, tuple(path[:-1]), doc.metadata.get("hierarchy_level", 0))
            candidates = by_parent.get(pkey, [])
            return [d for d in candidates if d.page_content != doc.page_content]

        if direction == "children":
            child_level = doc.metadata.get("hierarchy_level", 0) + 1
            pkey = (source, tuple(path), child_level)
            return list(by_parent.get(pkey, []))

        if direction == "all":
            return (
                expand_context(doc, all_docs, "parent", index=index)
                + expand_context(doc, all_docs, "siblings", index=index)
            )

        return []

    # ── Slow path: linear scan (no index available) ──
    if direction == "parent":
        parent_heading = path[-1] if path else ""
        for d in all_docs:
            if d.metadata.get("source") != source:
                continue
            if d.metadata.get("hierarchy_heading") == parent_heading:
                return [d]
        return []

    if direction == "siblings":
        parent = path[-1] if path else ""
        siblings = []
        for d in all_docs:
            if d.metadata.get("source") != source:
                continue
            dp = d.metadata.get("hierarchy_path", [])
            if len(dp) == len(path) and dp[:-1] == path[:-1] and d.page_content != doc.page_content:
                siblings.append(d)
        return siblings

    if direction == "children":
        children = []
        for d in all_docs:
            if d.metadata.get("source") != source:
                continue
            dp = d.metadata.get("hierarchy_path", [])
            if len(dp) == len(path) + 1 and dp[:len(path)] == path:
                children.append(d)
        return children

    if direction == "all":
        return (
            expand_context(doc, all_docs, "parent")
            + expand_context(doc, all_docs, "siblings")
        )

    return []
