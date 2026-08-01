"""
Layer 1 — Numeric Guard (rule-based, 0ms, 0 API calls).

Deterministic checks for the most dangerous hallucination patterns
in banking answers: wrong interest rates, missing fee conditions,
phantom "free" claims, and numbers fabricated from nowhere.

Each rule returns a list of ``(claim_text, verdict, reason)`` tuples.
An empty list means the rule passed.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document


class NumericGuard:
    """Fast deterministic verification of numeric claims.

    Usage::

        guard = NumericGuard()
        flags = guard.verify(
            answer="一年定期利率1.75%，同行转账免费",
            sql_result={"rows": [{"annual_rate": 0.0175}]},
            source_docs=retrieved_docs,
        )
        # flags → [{"claim": "...", "rule": "absolute_claim", ...}, ...]
    """

    # Patterns for extracting numbers from Chinese text
    _PCT_PATTERN = re.compile(
        r"([\d]+\.?[\d]*)\s*%"
    )  # "1.75%", "0.035%"
    _AMOUNT_PATTERN = re.compile(
        r"([\d,]+\.?\d*)\s*(?:元|块|万|亿)"
    )  # "50元", "20万"
    _ABSOLUTE_WORDS = re.compile(
        r"(免费|无手续费|不收|不收取|全免|没有任何|无论|随时|任意|所有|全部|一律)"
    )
    _CONDITION_WORDS = re.compile(
        r"(仅限|限于|条件|前提|需|须|须要|要求|满|达到|超过|不低于|"
        r"首年|次年|境内|境外|柜台|网银|手机银行|特定|指定|部分|"
        r"工作日|营业时间|工作日|除|除非|不包括)"
    )

    def verify(
        self,
        answer: str,
        sql_result: Optional[dict] = None,
        source_docs: Optional[List[Document]] = None,
    ) -> List[dict]:
        """Run all numeric guard rules against *answer*.

        Returns:
            List of flagged claims, each::

                {
                    "claim":  str,    # the offending text span
                    "rule":   str,    # rule name that fired
                    "reason": str,    # human-readable explanation
                }
        """
        if not answer or not answer.strip():
            return []

        flags: List[dict] = []

        flags.extend(self._check_sql_consistency(answer, sql_result))
        flags.extend(self._check_absolute_claims(answer, source_docs or []))
        flags.extend(self._check_orphan_numbers(answer, source_docs or []))

        return flags

    # ------------------------------------------------------------------
    # Rule 1: SQL consistency
    # ------------------------------------------------------------------

    def _check_sql_consistency(
        self, answer: str, sql_result: Optional[dict]
    ) -> List[dict]:
        """Numbers in the answer must match SQL query results."""
        if not sql_result or not sql_result.get("rows"):
            return []

        flags: List[dict] = []
        rows = sql_result["rows"]

        for pct_match in self._PCT_PATTERN.finditer(answer):
            answer_pct = float(pct_match.group(1))
            found = False
            for row in rows:
                for val in row.values():
                    if isinstance(val, (int, float)):
                        # SQL stores rates as decimals (0.0175 = 1.75%)
                        row_pct = float(val) * 100
                        if abs(answer_pct - row_pct) < 0.1:
                            found = True
                            break
                        # Also check direct match
                        if abs(answer_pct - float(val)) < 0.1:
                            found = True
                            break
                if found:
                    break
            if not found:
                span = answer[max(0, pct_match.start() - 10):pct_match.end() + 10]
                flags.append({
                    "claim": span.strip(),
                    "rule": "sql_consistency",
                    "reason": f"答案中的数值 {pct_match.group()} 在数据库查询结果中未找到",
                })

        return flags

    # ------------------------------------------------------------------
    # Rule 2: Absolute claims — "free" must have conditions
    # ------------------------------------------------------------------

    def _check_absolute_claims(
        self, answer: str, source_docs: List[Document]
    ) -> List[dict]:
        """Absolute words like "免费" must be qualified in source documents."""
        flags: List[dict] = []

        for match in self._ABSOLUTE_WORDS.finditer(answer):
            word = match.group()
            # Extract the sentence containing this word
            span_start = max(0, answer.rfind("。", 0, match.start()) + 1)
            span_end = answer.find("。", match.end())
            if span_end == -1:
                span_end = len(answer)
            sentence = answer[span_start:span_end].strip()

            # Check if any source doc contains condition words near this claim
            has_condition = any(
                self._CONDITION_WORDS.search(doc.page_content)
                for doc in source_docs
            )

            if has_condition:
                # The source HAS conditions, but the answer may have dropped them.
                # Check if the answer sentence itself includes a condition word.
                if not self._CONDITION_WORDS.search(sentence):
                    flags.append({
                        "claim": sentence[:120],
                        "rule": "absolute_claim",
                        "reason": (
                            f"答案声称'{word}'，但参考文档中存在条件限定"
                            f"（如'仅限''需''满...条件'），答案未包含这些限定"
                        ),
                    })

        return flags

    # ------------------------------------------------------------------
    # Rule 3: Orphan numbers — every number must exist in sources
    # ------------------------------------------------------------------

    def _check_orphan_numbers(
        self, answer: str, source_docs: List[Document]
    ) -> List[dict]:
        """Numbers in the answer should appear (approximately) in at least one source doc."""
        flags: List[dict] = []

        # Build a set of all numbers from source docs
        source_numbers: set = set()
        for doc in source_docs:
            for m in re.finditer(r"[\d]+\.?[\d]*", doc.page_content):
                try:
                    source_numbers.add(float(m.group()))
                except ValueError:
                    pass

        if not source_numbers:
            return flags

        for m in re.finditer(r"[\d]+\.?[\d]*", answer):
            try:
                num = float(m.group())
            except ValueError:
                continue

            # Skip small integers (page numbers, list indices, years)
            if num < 1 or (num == int(num) and 2000 <= num <= 2100):
                continue

            # Check if any source number is within ±5% of this answer number
            close = any(
                abs(num - sn) / max(abs(sn), 0.01) < 0.05
                for sn in source_numbers
            )
            if not close:
                span = answer[max(0, m.start() - 15):m.end() + 15]
                flags.append({
                    "claim": span.strip(),
                    "rule": "orphan_number",
                    "reason": f"数值 {m.group()} 在参考文档中未找到 (±5%)",
                })

        return flags
