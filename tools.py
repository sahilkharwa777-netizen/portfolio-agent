"""The two agent tools.

1. run_sql              - validated, read-only SQL execution
2. exposure_calculator  - equity sector exposure for one portfolio
"""
import logging
import re
import sqlite3
import time
from typing import Any, Dict

log = logging.getLogger("tools")

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|truncate)\b", re.I)


class SQLValidationError(ValueError):
    """The SQL is not an allowed read-only single SELECT statement."""


def validate_sql(sql: str) -> str:
    """Return the cleaned SQL or raise SQLValidationError."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("Empty SQL query.")
    cleaned = sql.strip().rstrip(";").strip()
    if ";" in cleaned:
        raise SQLValidationError("Only one SQL statement is allowed.")
    if not re.match(r"^(select|with)\b", cleaned, re.I):
        raise SQLValidationError("Only SELECT queries are allowed.")
    if FORBIDDEN.search(cleaned):
        raise SQLValidationError("Query contains a forbidden keyword (read-only access).")
    return cleaned


def run_sql(conn: sqlite3.Connection, sql: str, max_rows: int = 200) -> Dict[str, Any]:
    """Validate and execute SQL.

    Returns {ok, sql, columns, rows, truncated} on success, or
    {ok: False, error, error_type} where error_type is 'validation' or 'database'.
    Only 'database' errors (bad column, syntax ...) are worth sending back to the LLM for repair.
    """
    start = time.perf_counter()
    try:
        cleaned = validate_sql(sql)
    except SQLValidationError as e:
        log.warning("sql rejected: %s | %r", e, sql)
        return {"ok": False, "sql": sql, "error": str(e), "error_type": "validation"}
    try:
        cur = conn.execute(cleaned)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.Error as e:
        log.warning("sql failed: %s | %s", e, cleaned)
        return {"ok": False, "sql": cleaned, "error": str(e), "error_type": "database"}
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    log.info("run_sql ok rows=%d truncated=%s %.1fms | %s",
             len(rows), truncated, (time.perf_counter() - start) * 1000, cleaned)
    return {"ok": True, "sql": cleaned, "columns": cols, "rows": rows, "truncated": truncated}


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def find_portfolio(conn: sqlite3.Connection, name: str) -> Dict[str, Any]:
    """Resolve a user-supplied name to exactly one portfolio.

    exact (case-insensitive) match -> unique partial match -> else ambiguous / not found.
    Never silently picks one of several candidates.
    """
    name = (name or "").strip()
    if not name:
        return {"ok": False, "error_type": "invalid_input", "error": "No portfolio name was given."}
    row = conn.execute(
        "SELECT portfolio_id, portfolio_name FROM portfolios WHERE LOWER(portfolio_name) = LOWER(?)",
        (name,)).fetchone()
    if row:
        return {"ok": True, "portfolio_id": row[0], "portfolio_name": row[1]}
    rows = conn.execute(
        "SELECT portfolio_id, portfolio_name FROM portfolios "
        "WHERE LOWER(portfolio_name) LIKE ? ESCAPE '\\' ORDER BY portfolio_name",
        (f"%{_escape_like(name.lower())}%",)).fetchall()
    if len(rows) == 1:
        return {"ok": True, "portfolio_id": rows[0][0], "portfolio_name": rows[0][1]}
    if len(rows) > 1:
        return {"ok": False, "error_type": "ambiguous", "candidates": [r[1] for r in rows],
                "error": f"'{name}' matches several portfolios."}
    available = [r[0] for r in conn.execute("SELECT portfolio_name FROM portfolios ORDER BY portfolio_name")]
    return {"ok": False, "error_type": "not_found", "available": available,
            "error": f"Portfolio '{name}' not found."}


def exposure_calculator(conn: sqlite3.Connection, portfolio_name: str) -> Dict[str, Any]:
    """Sector exposure of one portfolio, equities only (bonds are ignored).

    Two clearly defined metrics are returned for every sector:
      portfolio_weight_pct : sum of current_weight of the sector's stocks, as % of the WHOLE portfolio
      exposure_pct         : that weight divided by the total equity weight, i.e. % of the EQUITY part.
                             These add up to 100%.
    """
    start = time.perf_counter()
    found = find_portfolio(conn, portfolio_name)
    if not found["ok"]:
        return found
    pid, pname = found["portfolio_id"], found["portfolio_name"]
    try:
        rows = conn.execute(
            """SELECT sec.sector_name, SUM(h.current_weight)
               FROM holdings h
               JOIN securities s  ON s.security_id = h.security_id
               JOIN sectors sec   ON sec.sector_id = s.sector_id
               WHERE h.portfolio_id = ? AND s.asset_type = 'Stock'
               GROUP BY sec.sector_name""", (pid,)).fetchall()
    except sqlite3.Error as e:
        log.exception("exposure query failed")
        return {"ok": False, "error_type": "database", "error": str(e)}
    equity_weight = sum(w for _, w in rows)
    result: Dict[str, Any] = {"ok": True, "portfolio": pname,
                              "equity_weight_pct": round(equity_weight * 100, 2), "exposures": []}
    if equity_weight <= 0:
        result["note"] = "This portfolio has no equity holdings, so there is no sector exposure."
    else:
        result["exposures"] = sorted(
            ({"sector": s,
              "exposure_pct": round(w / equity_weight * 100, 2),
              "portfolio_weight_pct": round(w * 100, 2)} for s, w in rows),
            key=lambda d: (-d["exposure_pct"], d["sector"]))
    log.info("exposure_calculator portfolio=%r sectors=%d %.1fms", pname,
             len(result["exposures"]), (time.perf_counter() - start) * 1000)
    return result
