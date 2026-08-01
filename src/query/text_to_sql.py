"""
Text-to-SQL —— 将自然语言问题转换为 MySQL 查询。

通过 LLM（DeepSeek）结合完整的数据库 schema 上下文，
生成安全的只读 SQL。面向银行客户查询场景，涵盖数值比较、聚合、模糊搜索等。

安全机制：
  - 仅允许 SELECT 语句，拒绝所有写操作和 DDL
  - 禁止跨库访问（db.table 写法）
  - 表名白名单校验，只能查 bank_rag 库中实际存在的表
  - 执行前验证 SQL，失败时附带错误信息重试一次
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.infra.database import execute_query, get_schema_info, schema_to_prompt_text


# ---------------------------------------------------------------------------
# SQL 安全校验
# ---------------------------------------------------------------------------

# 禁止出现的关键字，覆盖写操作、DDL、权限变更、文件操作、跨库切换
_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "LOAD", "INTO OUTFILE", "INTO DUMPFILE",
    "USE ",       # 防止切换数据库上下文
]

# 问题最大长度（安全阀，防止 prompt 注入）
_MAX_QUESTION_LENGTH = 200

# 禁止出现在问题中的危险字符/模式（prompt 注入防护）
_DANGEROUS_PATTERNS = [
    "```",              # markdown 代码块标记
    "DROP TABLE",
    "DELETE FROM",
    "INSERT INTO",
    "UPDATE ",
    "--",
    "/*",
    "*/",
    "UNION SELECT",
    "1=1",
    "1 = 1",
    "OR 1=",
    "SLEEP(",
    "BENCHMARK(",
]


def _sanitize_question(question: str) -> str:
    """清洗用户问题：去除控制字符、限制长度、拦截危险模式。"""
    import unicodedata

    # 去除控制字符（保留换行和制表符）
    cleaned = "".join(
        ch for ch in question
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")
    )
    cleaned = cleaned.strip()

    if len(cleaned) > _MAX_QUESTION_LENGTH:
        cleaned = cleaned[:_MAX_QUESTION_LENGTH]

    # 检查危险模式
    upper = cleaned.upper()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.upper() in upper:
            raise ValueError(f"查询包含不允许的内容")

    return cleaned


def _validate_sql(sql: str, allowed_tables: Optional[set] = None) -> Tuple[bool, str]:
    """
    校验 SQL 安全性，返回 (是否安全, 原因)。

    拒绝以下情况：
      - 包含上述禁止关键字
      - 非 SELECT 语句（不以 SELECT 开头）
      - 跨库引用（FROM db.table 或 JOIN db.table）
      - 堆叠查询（分号分隔的多条语句）
      - SQL 注释（行注释 -- 和块注释 /* */）
      - 引用的表不在 allowlist 中
    """
    upper = sql.strip().upper()

    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(r"\b" + kw + r"\b", upper):
            return False, f"禁止的关键字: {kw}"

    if not upper.startswith("SELECT"):
        return False, "仅允许 SELECT 语句"

    # 拦截堆叠查询
    if ";" in sql.strip().rstrip(";"):
        return False, "禁止堆叠查询"

    # 拦截 SQL 注释
    if "--" in sql or "/*" in sql or "*/" in sql:
        return False, "禁止 SQL 注释"

    # 拦截跨库访问：FROM bank_rag.xxx 或 JOIN other_db.xxx
    if re.search(r"\bFROM\s+\w+\.\w+", upper) or re.search(r"\bJOIN\s+\w+\.\w+", upper):
        return False, "禁止跨库访问"

    # 表名白名单校验
    if allowed_tables is not None:
        table_refs = re.findall(r"\b(?:FROM|JOIN)\s+`?(\w+)`?", upper)
        for tbl in table_refs:
            if tbl.lower() not in allowed_tables:
                return False, f"表不在白名单中: {tbl}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Text-to-SQL 转换器
# ---------------------------------------------------------------------------

class TextToSQL:
    """
    将自然语言问题转换为 MySQL 查询并执行。

    使用示例::

        t2sql = TextToSQL(llm)
        result = t2sql.query("年化收益率超过3.5%的理财产品有哪些")
        # → {"sql": "SELECT ...", "rows": [...], "answer": "..."}
    """

    def __init__(self, llm):
        """
        参数：
            llm: DeepSeekLLM 实例（或任何实现了 generate(prompt) → str 的对象）
        """
        self.llm = llm
        self._schema_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Schema 管理
    # ------------------------------------------------------------------

    @property
    def schema(self) -> Dict[str, Any]:
        """数据库 schema 信息，首次访问时从 MySQL 读取并缓存。"""
        if self._schema_cache is None:
            self._schema_cache = get_schema_info()
        return self._schema_cache

    @property
    def allowed_tables(self) -> set:
        """当前 bank_rag 库中存在的表名集合（小写），用于 SQL 白名单校验。"""
        return set(
            t.lower() for t in self.schema.get("tables", {}).keys()
        )

    def refresh_schema(self) -> None:
        """重新读取数据库 schema（DDL 变更后调用）。"""
        self._schema_cache = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def query(self, question: str, max_retries: int = 1) -> dict:
        """
        将自然语言问题转为 SQL、执行、并返回结构化结果。

        流程：
        1. 清洗问题（去除控制字符、拦截危险模式）
        2. 获取 schema 文本作为 prompt 上下文
        3. 调用 LLM 生成 SQL
        4. 安全校验 → 执行 → 格式化回答
        5. 失败时附带错误信息重试（最多 max_retries 次）

        返回::

            {
                "sql":          str,            # 生成的 SQL
                "rows":         list[dict],     # 查询结果（最多 50 行）
                "column_names": list[str],      # 列名列表
                "row_count":    int,            # 结果行数
                "answer":       str,            # 人类可读的中文摘要
                "error":        str | None,     # 失败时的错误信息，成功时为 None
            }
        """
        schema_text = schema_to_prompt_text(self.schema)

        if not schema_text:
            return {
                "sql": "", "rows": [], "column_names": [],
                "row_count": 0, "answer": "", "error": "数据库中没有表。",
            }

        # 清洗问题，防止 prompt 注入
        try:
            question = _sanitize_question(question)
        except ValueError as e:
            return {
                "sql": "", "rows": [], "column_names": [],
                "row_count": 0, "answer": "很抱歉，暂时无法查询到相关信息，请联系客服。",
                "error": str(e),
            }

        last_error: Optional[str] = None

        for attempt in range(max_retries + 1):
            sql = self._generate_sql(question, schema_text, last_error)

            if not sql:
                last_error = "LLM 返回了空 SQL"
                continue

            safe, reason = _validate_sql(sql, self.allowed_tables)
            if not safe:
                last_error = f"SQL 被拒绝: {reason}"
                continue

            try:
                rows = execute_query(sql)
            except Exception as e:
                last_error = str(e)
                continue

            # 成功
            column_names = list(rows[0].keys()) if rows else []
            answer = self._format_answer(question, sql, rows)

            return {
                "sql": sql,
                "rows": rows[:50],
                "column_names": column_names,
                "row_count": len(rows),
                "answer": answer,
                "error": None,
            }

        # 所有重试均失败
        return {
            "sql": "", "rows": [], "column_names": [],
            "row_count": 0,
            "answer": "很抱歉，暂时无法查询到相关信息，请联系客服。",
            "error": last_error or "Text-to-SQL 在全部重试后仍失败",
        }

    # ------------------------------------------------------------------
    # SQL 生成
    # ------------------------------------------------------------------

    def _generate_sql(
        self,
        question: str,
        schema_text: str,
        previous_error: Optional[str] = None,
    ) -> str:
        """
        调用 LLM 将问题转换为 SQL。

        参数：
            question: 用户自然语言问题
            schema_text: 格式化后的数据库 schema 文本
            previous_error: 上一次尝试的错误信息（用于重试时的自我修正）

        返回：LLM 生成的纯 SQL 文本（已去除 markdown 代码块和末尾分号）
        """

        error_block = ""
        if previous_error:
            error_block = (
                f"\n上一次尝试产生了以下错误：\n"
                f"  {previous_error}\n"
                f"请修正 SQL 并重试。\n"
            )

        prompt = f"""你是一个面向中国招商银行 MySQL 数据库的 SQL 专家。
根据给定的数据库 schema 和客户问题，编写一条 SELECT 查询。

数据库：bank_rag（唯一允许查询的数据库）
允许的表：{', '.join(sorted(self.allowed_tables))}

关键规则：
- 只能查询上述列出的表，不存在其他表
- 不要使用 db.table 写法（如 bank_rag.xxx），直接使用表名
- 只允许 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DDL
- 用户按名称搜索时，使用中文友好的 LIKE 模式
- 结果限制 50 行，除非用户明确要求更多
- 百分比值以小数存储时（如 0.035 表示 3.5%），注意比较方式
- 只返回 SQL，不要任何解释、不要 markdown 格式

数据库 Schema：
{schema_text}
{error_block}
问题：{question}

SQL:"""

        result = self.llm.generate(prompt)

        # 去除 markdown 代码块标记和末尾分号
        sql = result.strip()
        sql = re.sub(r"^```(?:sql)?\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        sql = sql.strip().rstrip(";")

        return sql

    # ------------------------------------------------------------------
    # 回答格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _format_answer(question: str, sql: str, rows: List[dict]) -> str:
        """
        将查询结果格式化为中文可读文本。

        单行结果：列出所有字段的键值对
        多行结果：表格形式，最多展示前 20 行，超出部分显示剩余条数
        空结果：提示无匹配数据
        """
        if not rows:
            return "未查询到符合条件的数据。"

        if len(rows) == 1:
            parts = [f"{k}: {v}" for k, v in rows[0].items()]
            return "查询结果：\n" + "\n".join(parts)

        cols = list(rows[0].keys())
        header = " | ".join(cols)
        lines = [f"共 {len(rows)} 条结果：", "", header, "-" * len(header)]
        for row in rows[:20]:
            lines.append(" | ".join(str(row.get(c, "")) for c in cols))
        if len(rows) > 20:
            lines.append(f"... 以及另外 {len(rows) - 20} 条结果")
        return "\n".join(lines)
