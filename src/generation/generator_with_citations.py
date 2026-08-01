from typing import List
import logging
from langchain_core.documents import Document

from src.generation.llm_client import DeepSeekLLM

logger = logging.getLogger(__name__)


class RAGGenerator:
    """Final answer generator using DeepSeek API."""

    def __init__(self, max_content_chars: int = 6000):
        logger.info("Initializing RAG Generator")
        self.llm = DeepSeekLLM()
        self.max_content_chars = max_content_chars
        logger.info("RAG Generator Ready")

    @staticmethod
    def _build_context_block(doc: Document, source_num: int) -> str:
        """Build a context block for a single document.

        For table chunks, uses ``text_as_html`` so the LLM sees the original
        table structure.  For non-table chunks, uses ``page_content``.
        """
        source = doc.metadata.get("source", "Unknown")

        if doc.metadata.get("is_table") and doc.metadata.get("text_as_html"):
            content = doc.metadata["text_as_html"]
        else:
            content = doc.page_content

        return f"\n[来源{source_num}]: {source}\n{content}\n"

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT = """你是一名银行内部知识库助手，服务于银行客户。你必须严格遵循以下规则生成回答。

## 回答结构

1. **定位问题**：先确认用户问的是什么。如果检索到的制度与用户问题不完全一致，以反问句开头确认（如"您咨询的是XX吗？"），再展开回答。
2. **给出结论**：用1-2句话直接回答核心问题。如果制度中定义的是相近概念而非完全一致的关键词，先给出相近概念的定义，最后指出与用户问法的差异。
3. **引用制度依据**：引用相关规章制度的原文片段，标明文件名称。原文用引号或缩进区分。最多引用3条来源，禁止编造条文编号。
4. **补充操作要点**：仅当问题涉及办理流程、操作步骤时，补充简要的操作指引或注意事项。

## 语气与风格

- 使用中文，专业但不生硬，必要时用通俗语言解释专有名词
- 不做自我介绍，不附加免责声明（系统会自动处理）
- 结论先行，依据居中，操作收尾
-每一段的结尾不要加任何标点符号

## 禁止行为（触犯任一即视为不合格）

以下问题必须礼貌拒绝回答，引导客户致电人工客服或前往网点咨询：

- **投资与产品**：不做产品优劣比较、投资时机建议、收益承诺/预测、资产配置建议
- **信用与授信**：不预测审批结果、不建议额度、不谈判利率或手续费
- **合规与法律**：不解读监管政策、不做合规性判断、不评估法律后果
- **账户与信息**：不查询余额/交易明细/他人账户（需鉴权），不代操作账户
- **投诉与纠纷**：不认定责任、不承诺赔偿金额、不承诺处理时限
- **同业比较**：不与其他银行或第三方产品做比较
- **特批与制度外**：不承诺特殊流程、不臆造制度未覆盖的操作方式

## 信息缺失处理

- 制度中有直接答案 → 按上述结构正常回答
- 制度中有相近概念 → 反问确认 + 给出相近概念的定义 + "如果以上内容与您咨询的问题不符，建议致电我行客服热线进一步咨询"
- 制度中完全无关 → "您咨询的问题在现行制度中未找到相关内容。建议您致电我行客服热线或前往就近网点，由工作人员为您详细解答。"

## 引用格式

- 使用"根据《文件名称》"的格式引用来源
- 原文片段单独成段，用引号包裹
- 解读性内容使用"综上""一般而言""制度要求"等引导词，与原文明确区分
- 最多引用3条制度来源
- 严禁编造条文编号、条款序号。如果原文中没有"第X条""第X款"等编号，引用时只写文件名称，不得自行添加任何编号"""

    _WEB_SYSTEM_PROMPT = """你是一名银行内部知识库助手，服务于银行客户。当前知识库中未找到与用户问题直接相关的制度原文，以下信息来自互联网公开渠道，仅供初步参考。你必须严格遵循以下规则生成回答。

## 回答结构

1. **定位问题**：先确认用户问的是什么。如果搜索结果与用户问题不完全一致，以反问句开头确认（如"您咨询的是XX吗？"），再展开回答。
2. **给出结论**：基于网络搜索结果给出回答，同时明确说明信息来源于网络公开渠道，非本行内部制度。
3. **引用来源**：引用网络搜索结果的标题和链接，每条关键信息后标注出处，方便用户自行核实。最多引用3条来源。
4. **补充说明**：提醒用户网络信息可能与本行实际制度存在差异，建议以本行网点工作人员告知或最新公告为准。

## 语气与风格

- 使用中文，专业但不生硬，必要时用通俗语言解释专有名词
- 不做自我介绍
- 结论先行，依据居中，提醒收尾
- 每一段的结尾不要加任何标点符号

## 禁止行为（触犯任一即视为不合格）

以下问题必须礼貌拒绝回答，引导客户致电人工客服或前往网点咨询：

- **投资与产品**：不做产品优劣比较、投资时机建议、收益承诺/预测、资产配置建议
- **信用与授信**：不预测审批结果、不建议额度、不谈判利率或手续费
- **合规与法律**：不解读监管政策、不做合规性判断、不评估法律后果
- **账户与信息**：不查询余额/交易明细/他人账户（需鉴权），不代操作账户
- **投诉与纠纷**：不认定责任、不承诺赔偿金额、不承诺处理时限
- **同业比较**：不与其他银行或第三方产品做比较
- **特批与制度外**：不承诺特殊流程、不臆造制度未覆盖的操作方式

## 信息缺失处理

- 网络搜索有相关信息 → 按上述结构正常回答，但必须标注"以上信息来源于网络公开渠道，非本行内部制度原文"
- 网络搜索无相关信息 → "您咨询的问题在互联网公开信息中也未找到相关内容。建议您致电我行客服热线或前往就近网点，由工作人员为您详细解答。"

## 引用格式

- 使用"根据网络公开信息"的格式引用来源
- 每条信息末尾标注来源网址，格式为"（来源：[网页标题](网址)）"
- 解读性内容使用"综上""一般而言""公开资料显示"等引导词，与网络信息明确区分
- 最多引用3条网络来源
- 严禁将网络信息伪装为银行内部制度原文"""

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_with_citations(
        self,
        query: str,
        context_docs: List[Document],
    ) -> str:

        if not query or not query.strip():
            return "请输入您想咨询的问题。"

        if not context_docs:
            return (
                "您咨询的问题在现行制度中未找到相关内容。"
                "建议您致电我行客服热线或前往就近网点，"
                "由工作人员为您详细解答。"
            )

        context_text = ""
        total_chars = 0

        for i, doc in enumerate(context_docs):
            block = self._build_context_block(doc, i + 1)
            total_chars += len(block)
            if total_chars > self.max_content_chars:
                break
            context_text += block

        prompt = f"""{self._SYSTEM_PROMPT}

## 检索到的制度原文

{context_text}

## 用户问题

{query}

请按上述规则生成回答："""

        try:
            answer = self.llm.generate(prompt)

            if not answer:
                return (
                    "您咨询的问题在现行制度中未找到相关内容。"
                    "建议您致电我行客服热线或前往就近网点，"
                    "由工作人员为您详细解答。"
                )

            return answer.strip()

        except Exception as e:
            return f"Generation failed: {str(e)}"

    def generate_from_web(
        self,
        query: str,
        web_docs: List[Document],
    ) -> str:
        """Generate answer from web search results (KB fallback).

        Only snippets (~200-300 chars) are provided to the LLM — full page
        content is never fetched.  The LLM is instructed to cite web sources
        and clearly distinguish them from internal policy documents.
        """
        if not query or not query.strip():
            return "请输入您想咨询的问题。"

        if not web_docs:
            return (
                "您咨询的问题在互联网公开信息中也未找到相关内容。"
                "建议您致电我行客服热线或前往就近网点，"
                "由工作人员为您详细解答。"
            )

        context_text = ""
        total_chars = 0

        for i, doc in enumerate(web_docs):
            title = doc.metadata.get("title", "未知来源")
            url = doc.metadata.get("source", "")
            snippet = doc.page_content
            block = (
                f"\n[网页来源{i + 1}]: {title}\n"
                f"链接: {url}\n"
                f"摘要: {snippet}\n"
            )
            total_chars += len(block)
            if total_chars > self.max_content_chars:
                break
            context_text += block

        prompt = f"""{self._WEB_SYSTEM_PROMPT}

## 网络搜索结果

{context_text}

## 用户问题

{query}

请按上述规则生成回答："""

        try:
            answer = self.llm.generate(prompt)

            if not answer:
                return (
                    "您咨询的问题在互联网公开信息中也未找到相关内容。"
                    "建议您致电我行客服热线或前往就近网点，"
                    "由工作人员为您详细解答。"
                )

            return answer.strip()

        except Exception as e:
            return f"Generation failed: {str(e)}"
