import logging
import re
from typing import List, Optional, Literal

logger = logging.getLogger(__name__)


class QueryTransformer:
    """Query expansion for retrieval coverage.

    Two modes:
      - ``"template"``: deterministic template-based expansion (zero latency)
      - ``"llm"``:      LLM-generated semantically diverse queries (better recall, ~0.5s)

    Usage::

        # Template mode (default, no LLM needed)
        qt = QueryTransformer(mode="template")
        queries = qt.expand("转账手续费是多少")

        # LLM mode
        qt = QueryTransformer(mode="llm", llm=groq_llm)
        queries = qt.expand("转账手续费是多少")
    """

    def __init__(
        self,
        mode: Literal["template", "llm"] = "template",
        llm=None,
    ):
        self.mode = mode
        self.llm = llm
        logger.info("QueryTransformer initialized (mode=%s)", mode)

    # ------------------------------------------------------------------
    # Public entry point — dispatches based on self.mode
    # ------------------------------------------------------------------

    def expand(self, original_query: str, num_queries: int = 3) -> List[str]:
        """Expand a query into multiple retrieval queries.

        Dispatches to the active expansion strategy based on ``self.mode``.
        """
        if self.mode == "llm":
            if self.llm is None:
                logger.warning("LLM mode requested but no LLM provided; falling back to template")
                return self.multi_query(original_query, num_queries)
            return self._llm_multi_query(original_query, num_queries)
        return self.multi_query(original_query, num_queries)

    # ------------------------------------------------------------------
    # Template-based expansion (deterministic, zero latency)
    # ------------------------------------------------------------------

    def multi_query(self, original_query: str, num_queries: int = 3) -> List[str]:
        if not original_query or not original_query.strip():
            return []

        base = original_query.strip()

        # Detect if query is primarily Chinese
        has_chinese = bool(re.search(r'[一-鿿]', base))

        if has_chinese:
            candidates = [
                base,
                f"{base} 的相关规定是什么？",
                f"关于 {base} 的公司政策",
                f"{base} 的管理办法",
                f"公司对 {base} 有哪些要求？",
            ]
        else:
            candidates = [
                base,
                f"Explain {base}",
                f"What are the rules related to {base}?",
                f"Policy regarding {base}",
                f"Company guidelines for {base}",
            ]

        seen = set()
        unique = []
        for q in candidates:
            key = q.lower().strip()
            if key not in seen:
                unique.append(q)
                seen.add(key)

        return unique[:num_queries]

    def hyde(self, query: str) -> str:
        if not query or not query.strip():
            return ""

        has_chinese = bool(re.search(r'[一-鿿]', query))

        if has_chinese:
            return (
                f"该查询涉及与 {query} 相关的组织政策、规章制度和文件化流程。"
                f"答案预期存在于政策或人力资源文档中。"
            )
        else:
            return (
                f"This query refers to official organizational policies, rules, "
                f"and documented procedures related to {query}. "
                f"The answer is expected to be found in policy or HR documentation."
            )

    def step_back(self, query: str) -> str:
        if not query or not query.strip():
            return ""

        # Preserve Chinese characters in the query
        clean = re.sub(r"[^\w\s一-鿿㐀-䶿]", "", query).strip()

        has_chinese = bool(re.search(r'[一-鿿]', clean))

        if has_chinese:
            return f"与 {clean} 相关的一般性组织政策是什么？"
        else:
            return f"What are the general organizational policies related to {clean}?"

    # ------------------------------------------------------------------
    # LLM-based expansion (semantically diverse, ~0.5s latency)
    # ------------------------------------------------------------------

    def _llm_multi_query(self, original_query: str, num_queries: int = 3) -> List[str]:
        """Use the LLM to generate semantically diverse retrieval queries.

        Unlike template expansion which appends fixed suffixes, this asks the
        LLM to rephrase the query from different angles — varying keywords,
        granularity, and perspective — so that dense and sparse retrievers
        each surface complementary document sets.
        """
        if not original_query or not original_query.strip():
            return []

        if num_queries < 2:
            return [original_query.strip()]

        prompt = self._build_llm_expansion_prompt(original_query, num_queries)

        try:
            raw = self.llm.generate(prompt)
        except Exception:
            logger.exception("LLM multi-query expansion failed; falling back to template")
            return self.multi_query(original_query, num_queries)

        queries = self._parse_expansion_response(raw, original_query, num_queries)
        return queries

    @staticmethod
    def _build_llm_expansion_prompt(query: str, num_queries: int) -> str:
        return f"""你是一个银行知识库的检索优化助手。

请将以下问题从不同角度改写为 {num_queries} 个检索查询。
每个查询应该：
- 使用不同的关键词和表达方式
- 从不同粒度提问（如：具体数值、政策条款、操作流程）
- 保持原问题的核心意图不变
- 一行一个查询，不要编号、不要解释

原问题：{query}

改写后的 {num_queries} 个查询："""

    @staticmethod
    def _parse_expansion_response(
        raw: str, fallback_query: str, num_queries: int
    ) -> List[str]:
        """Parse LLM output into a deduplicated query list.

        Falls back to a single-element list (the original query) if parsing
        produces nothing usable.
        """
        if not raw:
            return [fallback_query]

        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]

        # Strip common numbering prefixes ("1. ", "1、", "1)", "- ")
        cleaned = []
        for line in lines:
            line = re.sub(r"^[\d]+[\.\、\)\)]\s*", "", line)
            line = re.sub(r"^[-•·]\s*", "", line)
            line = line.strip().strip("\"'\"'")
            if line:
                cleaned.append(line)

        # Deduplicate while preserving order; always include the original
        seen = set()
        unique = []
        for q in cleaned:
            key = q.lower().strip()
            if key not in seen:
                unique.append(q)
                seen.add(key)

        if not unique:
            return [fallback_query]

        # Ensure the original query is first
        if fallback_query not in unique:
            unique.insert(0, fallback_query)

        return unique[:num_queries]
