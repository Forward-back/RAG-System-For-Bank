"""
Layer 3 — LLM Fact-Checker (DeepSeek API, ~500ms, fallback only).

This is the LAST line of defence. It is only invoked when Layer 1
(NumericGuard) and/or Layer 2 (NLIVerifier) flag issues that cannot
be resolved deterministically.

Unlike Layer 2, this uses the LLM (DeepSeek) — a different capability
class than the cross-encoder — to perform deep semantic reasoning on
claims that require understanding nuance, multi-sentence entailment,
or domain-specific judgment.

Design:
  1. Take the generated answer + the source documents used to produce it.
  2. Ask the LLM to decompose the answer into claims and verify each.
  3. Rewrite the answer: keep verified claims, replace uncertain/
     contradicted ones with a CS-redirect sentence.
  4. If the entire answer is flagged as uncertain/contradicted, replace
     completely with a CS-redirect.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FactChecker:
    """
    Verify generated answers against source documents.

    Usage::

        checker = FactChecker(llm)
        result = checker.verify(
            question="转账手续费是多少",
            answer="同行转账免费，跨行转账收费2元...",
            source_docs=retrieved_docs,
        )
        if result["safe_to_show"]:
            return result["safe_answer"]
        else:
            return "请咨询人工客服..."
    """

    def __init__(self, llm):
        """
        Args:
            llm: A DeepSeekLLM instance (or any object with .generate(prompt)).
        """
        self.llm = llm

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def verify(
        self,
        question: str,
        answer: str,
        source_docs: List[Document],
        max_source_chars: int = 8000,
    ) -> Dict[str, Any]:
        """
        Verify *answer* against *source_docs* and return a safe version.

        Returns::

            {
                "safe_answer":   str,           # rewritten answer (unsafe parts replaced)
                "safe_to_show":  bool,          # False if the entire answer is unreliable
                "verdicts": [
                    {"claim": str, "verdict": "verified|uncertain|contradicted", "evidence": str},
                    ...
                ],
                "replaced_count": int,          # how many claims were replaced
                "total_claims":   int,
                "raw_response":   str,          # raw LLM output for debugging
            }

        If *answer* is empty or trivially short, it is returned as-is
        (assumed to be a rejection / "I don't know" message).
        """
        # Trivial answers pass through — they are already rejections
        if not answer or len(answer.strip()) < 15:
            return {
                "safe_answer": answer,
                "safe_to_show": True,
                "verdicts": [],
                "replaced_count": 0,
                "total_claims": 0,
                "raw_response": "",
            }

        # Build a compact source text block
        source_text = self._format_sources(source_docs, max_source_chars)

        prompt = self._build_prompt(question, answer, source_text)

        raw = self.llm.generate(prompt)

        parsed = self._parse_response(raw, answer)

        safe_answer = self._rewrite_answer(parsed, answer)

        replaced = sum(
            1 for v in parsed.get("verdicts", [])
            if v.get("verdict") in ("uncertain", "contradicted")
        )
        total = len(parsed.get("verdicts", []))

        safe_to_show = not (total > 0 and replaced == total)

        return {
            "safe_answer": safe_answer,
            "safe_to_show": safe_to_show,
            "verdicts": parsed.get("verdicts", []),
            "replaced_count": replaced,
            "total_claims": total,
            "raw_response": raw,
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(question: str, answer: str, source_text: str) -> str:
        return f"""你是银行客服系统的答案审核员。请审核以下AI生成的回答是否准确。

【客户问题】
{question}

【AI回答】
{answer}

【参考文档】（回答必须基于以下文档中的信息）
{source_text}

审核步骤：
1. 将AI回答拆解为原子事实陈述（每个陈述一句，如"同行转账免费""跨行转账收费2元"）
2. 逐一在参考文档中查找证据
3. 对每个陈述标注：
   - "verified": 参考文档中有明确原文支持
   - "uncertain": 文档中提到相关主题，但没有足够细节确认该陈述
   - "contradicted": 文档中的信息与该陈述矛盾

4. 输出JSON格式（只输出JSON，不要其他文字）：

{{
  "verdicts": [
    {{"claim": "事实陈述原文", "verdict": "verified", "evidence": "文档中对应的原文片段"}},
    {{"claim": "事实陈述原文", "verdict": "uncertain", "evidence": "文档未提供足够信息"}},
    {{"claim": "事实陈述原文", "verdict": "contradicted", "evidence": "文档中矛盾的原文片段"}}
  ],
  "overall_assessment": "all_verified | partial_issues | fully_unreliable"
}}

重要提示：
- 宁可标注为uncertain，也不要放过任何没有文档支撑的陈述
- 免责声明、礼貌用语等非事实性内容不需要审核
- evidence字段请引用文档原文，不要自己概括
"""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str, fallback_answer: str) -> dict:
        """Extract JSON from the LLM response. Fall back to marking all as uncertain."""
        if not raw:
            return {
                "verdicts": [],
                "overall_assessment": "fully_unreliable",
            }

        # Strip markdown code fences
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Try to find the JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: treat the whole answer as uncertain
        return {
            "verdicts": [
                {"claim": fallback_answer[:200], "verdict": "uncertain",
                 "evidence": "无法解析审核结果"}
            ],
            "overall_assessment": "fully_unreliable",
        }

    # ------------------------------------------------------------------
    # Answer rewriting
    # ------------------------------------------------------------------

    @staticmethod
    def _rewrite_answer(parsed: dict, original_answer: str) -> str:
        """
        Replace uncertain/contradicted claims with CS-redirect sentences.

        Strategy:
          - For each uncertain/contradicted claim, extract the topic
            and replace with a redirect sentence.
          - If all claims are flagged, replace the entire answer.
        """
        verdicts = parsed.get("verdicts", [])

        if not verdicts:
            return original_answer

        # Count flagged claims
        flagged = [v for v in verdicts if v.get("verdict") in ("uncertain", "contradicted")]
        verified = [v for v in verdicts if v.get("verdict") == "verified"]

        # All (or nearly all) flagged → clean full redirect
        # Trigger when zero verified, or when <20% of claims pass verification.
        _mostly_flagged = (
            len(verified) == 0
            or (len(verdicts) > 0 and len(flagged) / len(verdicts) > 0.6)
        )
        if _mostly_flagged and len(flagged) > 0:
            topics = "、".join(
                FactChecker._extract_topic(v.get("claim", ""))
                for v in flagged[:3]
            )
            return (
                f"关于{'该问题' if not topics else topics}，建议您拨打我行客服热线"
                f"或前往就近网点咨询客户经理获取准确信息。"
            )

        # Partial flagged → replace flagged claims inline
        rewritten = original_answer
        for v in flagged:
            claim = v.get("claim", "")
            if claim and len(claim) > 5:
                topic = FactChecker._extract_topic(claim)
                replacement = (
                    f"（关于{topic}的具体信息，建议您咨询我行客服获取准确答复）"
                )
                # Try to replace the claim text in the answer
                # If not found verbatim, append the note instead
                if claim in rewritten:
                    rewritten = rewritten.replace(claim, replacement)
                else:
                    # Search for a substring match
                    short = claim[:30]
                    if short in rewritten:
                        rewritten = rewritten.replace(short, replacement)

        # If rewriting didn't change anything but we have flagged claims,
        # append a note
        if rewritten == original_answer and flagged:
            topics = "、".join(
                FactChecker._extract_topic(v.get("claim", ""))
                for v in flagged[:3]
            )
            rewritten += (
                f"\n\n关于{topics}的详细信息，建议您咨询我行客服获取准确答复。"
            )

        return rewritten

    @staticmethod
    def _extract_topic(claim: str) -> str:
        """Extract the topic phrase from a claim sentence."""
        # Try to get the first meaningful phrase
        claim = claim.strip().rstrip("。，；！？,.!?;；")
        # Take first 15 chars as topic (Chinese characters)
        if len(claim) <= 15:
            return claim
        # Try to find a natural break point
        for sep in ["的", "是", "为", "：", ":", "，", ","]:
            idx = claim.find(sep, 5, 20)
            if idx > 0:
                return claim[:idx+1]
        return claim[:15]

    # ------------------------------------------------------------------
    # Source formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_sources(docs: List[Document], max_chars: int) -> str:
        """Build a compact text block of source documents for the prompt."""
        parts = []
        total = 0
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "unknown")
            content = doc.page_content
            block = f"[文档{i+1} 来源: {src}]\n{content}\n"
            total += len(block)
            if total > max_chars:
                # Truncate last block to fit
                available = max_chars - (total - len(block))
                if available > 200:
                    block = block[:available] + "\n...(内容过长，已截断)"
                    parts.append(block)
                break
            parts.append(block)
        return "\n".join(parts)
