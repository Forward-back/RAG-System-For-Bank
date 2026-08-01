"""
查询改写器 —— 将客户口语化的模糊问题转换为结构化的检索查询。

银行客户的问题往往高度口语化、充满指代和省略：
  "我想把那个钱拿出来" → "定期存款提前支取流程及利息计算规则"
  "转账要钱吗"         → "银行转账手续费收取标准"
  "那个卡丢了咋整"     → "银行卡挂失补办流程"

改写器在检索之前调用 LLM 消除歧义，并在最终回答前附加确认语句，
让客户验证系统理解是否正确，避免"答非所问"。
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    将模糊的用户查询改写为适合检索引擎的结构化问题。

    两阶段设计：
      1. 前置门禁 _is_vague — 只有真正模糊的查询才调用 LLM
      2. 后置校验 _validate_rewrite — 多信号验证改写是否偏离原意

    使用示例::

        rewriter = QueryRewriter(llm)
        result = rewriter.rewrite("我想把钱取出来")
        # → {"original": "我想把钱取出来",
        #    "rewritten": "定期存款提前支取流程及利息计算规则",
        #    "is_rewritten": True}
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self, llm):
        """
        参数：
            llm: DeepSeekLLM 实例（或任何实现了 generate(prompt) → str 的对象）
        """
        self.llm = llm

    # ------------------------------------------------------------------
    # 前置门禁 — 模糊度检测
    # ------------------------------------------------------------------

    # 模糊指示词 / 口语化表达 — 命中任一即认为可能需要改写
    _VAGUE_PATTERNS: List[str] = [
        # 指代不明
        r"那个", r"这个", r"它", r"这", r"那",
        r"这些", r"那些", r"其中", r"以上", r"以下",
        # 口语化
        r"咋", r"咋整", r"咋办", r"咋搞",
        r"搞一下", r"弄一下", r"整一下",
        r"怎么办", r"怎么搞", r"怎么弄",
        r"什么样", r"啥", r"啥样",
        # 模糊动词短语（缺宾语/主语）
        r"想[把给跟和]", r"想把",
        r"钱取出来", r"钱拿出来",
        r"丢了咋", r"不见了",
        r"要钱吗", r"收费吗",
        # 毫无上下文的单个动词
        r"^[办弄整搞]$",
    ]

    # 问句类型前缀 — 用于检测意图偏移
    _QUESTION_TYPE_PATTERNS = [
        (r"^(什么(是|样|叫)|何为|啥是|啥叫|啥样)", "definition"),   # 定义
        (r"^(如何|怎么|怎样|咋[整办搞]|咋样|为啥|怎么会)", "procedure"),  # 流程
        (r"^(哪里|哪儿|在哪|去哪|哪个网点|什么地方)", "location"),   # 地点
        (r"^(谁|哪位|什么人)", "person"),                          # 人物
        (r"^(为什么|为啥|原因|怎么回事)", "reason"),                # 原因
        (r"^(什么时候|何时|几点|多久|多长时间)", "time"),           # 时间
        (r"^(多少|多少钱|几[多]?[钱块元])", "amount"),              # 金额
    ]

    @classmethod
    def _is_vague(cls, query: str) -> bool:
        """检查查询是否包含需要消歧的模糊指代或高度口语化表达。

        两条件同时满足才认为需要改写：
          1. 命中模糊词（有歧义空间）
          2. 不是"自带具体关键词"的查询（有明确检索目标的不改）
        """
        has_vague_signal = any(re.search(p, query) for p in cls._VAGUE_PATTERNS)
        if not has_vague_signal:
            return False

        # 有具体关键词 → 即使含模糊词也不改，关键词本身就是检索目标
        has_concrete = cls._has_concrete_keywords(query)
        return not has_concrete

    @classmethod
    def _has_concrete_keywords(cls, query: str) -> bool:
        """检测查询中是否包含具体检索关键词（专有名词、政策术语等）。

        规则（命中任一即认为有具体目标）：
          - 连续 3+ 个汉字且不包含"怎么/如何/什么"等疑问词
          - 包含引号内的精确短语
        """
        # 疑问功能词 — 不算具体关键词
        question_words = {
            "什么", "怎么", "如何", "哪里", "哪个", "为什么", "为何",
            "多少", "多久", "什么时候", "是否", "有没有",
            "请问", "可以", "能不能", "可不可以",
            "是", "的", "吗", "呢", "吧", "啊",
            "我想", "想要", "想", "要", "帮", "帮我", "给我",
        }

        # 用 jieba 分词提取名词/专有名词
        try:
            import jieba.posseg as pseg
            words = pseg.lcut(query)
            concrete = [
                w.word for w in words
                if w.flag in ("n", "nr", "ns", "nt", "nz", "eng")
                and w.word not in question_words
                and len(w.word) >= 2
            ]
            if len(concrete) >= 1:
                return True
        except Exception:
            pass

        # Fallback: 用正则提取连续汉字块
        han_blocks = re.findall(r"[一-鿿]{3,}", query)
        for block in han_blocks:
            if block not in question_words:
                return True

        return False

    # ------------------------------------------------------------------
    # 问句类型提取
    # ------------------------------------------------------------------

    @classmethod
    def _intent_type(cls, query: str) -> Optional[str]:
        """提取问句类型。``None`` 表示无法识别。"""
        for pattern, itype in cls._QUESTION_TYPE_PATTERNS:
            if re.match(pattern, query):
                return itype
        return None

    # ------------------------------------------------------------------
    # 实体提取
    # ------------------------------------------------------------------

    @classmethod
    def _extract_key_nouns(cls, query: str) -> List[str]:
        """提取查询中的关键名词（长度 >= 2 的名词/专有名词）。"""
        try:
            import jieba.posseg as pseg
            words = pseg.lcut(query)
            return [w.word for w in words if w.flag in ("n", "nr", "ns", "nt", "nz") and len(w.word) >= 2]
        except Exception:
            # jieba 不可用时退化为去停用词的长词
            return re.findall(r"[一-鿿]{2,}", query)

    # ------------------------------------------------------------------
    # 后置校验 — 多信号验证
    # ------------------------------------------------------------------

    @classmethod
    def _validate_rewrite(cls, original: str, rewritten: str) -> tuple:
        """验证改写是否偏离原意。

        Returns:
            (is_valid: bool, reject_reason: str | None)
        """
        # ── 信号 1: 意图偏移 ──
        orig_intent = cls._intent_type(original)
        new_intent = cls._intent_type(rewritten)
        if orig_intent and new_intent and orig_intent != new_intent:
            return False, f"意图偏移: {orig_intent} → {new_intent}"

        # ── 信号 2: 核心实体丢失 ──
        orig_nouns = cls._extract_key_nouns(original)
        new_nouns = set(cls._extract_key_nouns(rewritten))
        lost = [n for n in orig_nouns if n not in new_nouns and n not in rewritten]
        if lost:
            return False, f"核心实体丢失: {', '.join(lost[:3])}"

        # ── 信号 3: 长度突增 ──
        if len(rewritten) > len(original) * 2.5:
            return False, f"长度突增: {len(original)} → {len(rewritten)}"

        # ── 信号 4: 语义兜底（需要有 embedding 模型） ──
        # 由调用方注入，此处跳过（_validate_rewrite 本身不依赖外部模型）
        # 外部通过 _compute_similarity 处理

        return True, None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def rewrite(self, query: str, similarity_model=None) -> dict:
        """将查询改写为清晰、适合检索的形式。

        前置门禁 + 后置校验双保险：只有真正模糊且改写不偏离原意的
        查询才会被改写。

        Args:
            query:            用户原始查询
            similarity_model: SentenceTransformer 实例，用于后置语义校验。
                             传 None 则跳过语义相似度校验。

        返回::
            {
                "original":       str,   # 用户原始查询
                "rewritten":      str,   # 改写后的查询（用于检索）
                "is_rewritten":   bool,  # 是否实际发生了改写
                "reject_reason":  str,   # 改写被拒绝的原因（is_rewritten=False 时）
                "raw_response":   str,   # LLM 原始输出，便于调试
            }
        """
        if not query or not query.strip():
            return {
                "original": query,
                "rewritten": query,
                "is_rewritten": False,
                "reject_reason": "empty query",
                "raw_response": "",
            }

        # ── 阶段 1: 前置门禁 ──
        if not self._is_vague(query):
            return {
                "original": query,
                "rewritten": query,
                "is_rewritten": False,
                "reject_reason": "前置门禁: 查询已足够清晰",
                "raw_response": "",
            }

        # ── 阶段 2: LLM 改写 ──
        prompt = self._build_prompt(query)
        raw = self.llm.generate(prompt)
        rewritten = self._parse_response(raw, query)

        if rewritten.strip() == query.strip():
            return {
                "original": query,
                "rewritten": query,
                "is_rewritten": False,
                "reject_reason": "LLM 原样返回（无需改写）",
                "raw_response": raw,
            }

        # ── 阶段 3: 后置校验 ──
        valid, reason = self._validate_rewrite(query, rewritten)
        if not valid:
            logger.info("[REWRITE] Rejected: %s | original='%s' → rewritten='%s'",
                        reason, query, rewritten)
            return {
                "original": query,
                "rewritten": query,
                "is_rewritten": False,
                "reject_reason": f"后置校验: {reason}",
                "raw_response": raw,
            }

        # ── 信号 4: 语义兜底校验（外部注入模型） ──
        if similarity_model is not None:
            sim = self._compute_similarity(query, rewritten, similarity_model)
            if sim < 0.65:
                logger.info("[REWRITE] Rejected: 语义相似度过低 (%.3f) | '%s' → '%s'",
                            sim, query, rewritten)
                return {
                    "original": query,
                    "rewritten": query,
                    "is_rewritten": False,
                    "reject_reason": f"后置校验: 语义相似度过低 ({sim:.3f})",
                    "raw_response": raw,
                }

        return {
            "original": query,
            "rewritten": rewritten,
            "is_rewritten": True,
            "reject_reason": None,
            "raw_response": raw,
        }

    @staticmethod
    def _compute_similarity(query: str, rewritten: str, model) -> float:
        """计算原始查询与改写查询的余弦相似度。"""
        import numpy as np
        vecs = model.encode([query, rewritten], normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))

    # ------------------------------------------------------------------
    # 确认语句
    # ------------------------------------------------------------------

    @staticmethod
    def confirmation_line(rewritten_query: str) -> str:
        """
        生成回答开头的确认语句，让用户验证系统的理解是否准确。

        示例：
            "请问您想了解的是：定期存款提前支取流程及利息计算规则 吗？"
        """
        return f"请问您想了解的是：{rewritten_query} 吗？"

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(query: str) -> str:
        """
        构建 LLM 改写提示词。

        核心约束：
          - 消除模糊指代（"那个""这个"→ 具体事物）
          - 口语转正式（"咋整"→"如何办理"）
          - 补充隐含的业务上下文
          - 保持原意，不添加客户未提及的信息
          - 已经清晰的查询原样返回
        """
        return f"""你是一个银行客服系统的查询改写器。

客户提出的问题可能模糊、口语化、包含指代不明的词（如"那个""这个""它"），
你需要将问题改写为一个清晰、具体、结构化的检索查询。

改写规则：
- 消除模糊指代：将"那个""这个"替换为具体指代的事物
- 扩展口语和缩写：将"咋整""咋办"改为"如何办理"等正式表达
- 补充隐含上下文：如果问题隐含了某个业务流程，补充完整
- 保持原意：不要添加客户没问的信息，不要改变问题的核心意图
- 如果原问题已经足够清晰，直接返回原问题
- 只输出改写后的问题，一行，不要任何解释或标记

客户原问题：{query}

改写后的问题："""

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str, fallback: str) -> str:
        """从 LLM 原始输出中提取改写后的查询文本。"""
        if not raw:
            return fallback

        cleaned = raw.strip()

        for prefix in ["改写后的问题：", "改写后：", "问题：", "查询："]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        if len(cleaned) > 200:
            lines = cleaned.split("\n")
            cleaned = lines[0].strip()

        if len(cleaned) > 200 or not cleaned:
            return fallback

        return cleaned
