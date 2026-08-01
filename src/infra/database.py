"""
MySQL connection management for the bank RAG system.

Manages connections to the local MySQL instance that stores structured
financial product data (interest rates, fees, product metadata, etc.).
This is the "ground truth" source for numerical queries that vector
search cannot handle.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG: Dict[str, Any] = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE", "bank_rag"),
    "charset":  "utf8mb4",
    "use_unicode": True,
}


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection():
    """Return a new MySQL connection pinned to the bank_rag database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        # Explicitly pin to bank_rag — prevents stray USE statements
        cursor = conn.cursor()
        cursor.execute("USE bank_rag")
        cursor.close()
        return conn
    except Error as e:
        print(f"[DB] Connection failed: {e}")
        raise


@contextmanager
def get_cursor(commit: bool = True):
    """Context manager yielding a (connection, cursor) pair."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        yield conn, cursor
        if commit:
            conn.commit()
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def execute_query(sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """
    Execute a SELECT query and return results as a list of dicts.

    Raises ValueError for any non-SELECT statement (safety gate).
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT") and not sql_stripped.startswith("SHOW") and not sql_stripped.startswith("DESCRIBE"):
        raise ValueError(f"Only SELECT/SHOW/DESCRIBE allowed. Rejected: {sql[:80]}...")

    with get_cursor(commit=False) as (conn, cursor):
        cursor.execute(sql, params or ())
        return cursor.fetchall()


def execute_ddl(sql: str) -> None:
    """Execute a DDL statement (CREATE TABLE, INSERT, etc.)."""
    with get_cursor(commit=True) as (conn, cursor):
        # Split by semicolons for multi-statement
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                cursor.execute(stmt)


def get_schema_info() -> Dict[str, Any]:
    """
    Introspect the database and return schema metadata suitable for
    inclusion in a Text-to-SQL prompt.
    """
    tables = execute_query("SHOW TABLES")
    schema: Dict[str, Any] = {"tables": {}}

    for row in tables:
        # The key name varies by MySQL version — try common variants
        tbl = row.get("Tables_in_bank_rag") or list(row.values())[0]
        cols = execute_query(f"DESCRIBE `{tbl}`")
        schema["tables"][tbl] = [
            {"name": c["Field"], "type": c["Type"], "key": c["Key"], "extra": c["Extra"]}
            for c in cols
        ]

    return schema


def schema_to_prompt_text(schema: Dict[str, Any]) -> str:
    """Convert schema dict into a compact text block for LLM prompts."""
    lines = []
    for tbl, cols in schema.get("tables", {}).items():
        lines.append(f"  {tbl}(")
        for c in cols:
            flags = []
            if c["key"] == "PRI":
                flags.append("PRIMARY KEY")
            if c.get("extra") == "auto_increment":
                flags.append("AUTO_INCREMENT")
            flag_str = ("  -- " + ", ".join(flags)) if flags else ""
            lines.append(f"    {c['name']} {c['type']}{flag_str}")
        lines.append("  )")
    return "\n".join(lines)
