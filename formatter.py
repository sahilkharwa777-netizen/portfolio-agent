"""Deterministic answer formatting.

Answers are built directly from tool output (no second LLM call), so what the user
reads can never contradict the data, and each question costs one API call instead of two.
"""
from typing import Any, Dict, List


def _cell(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    return str(v)


def markdown_table(columns: List[str], rows: List[tuple]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(_cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def format_sql_result(res: Dict[str, Any]) -> str:
    rows, cols = res["rows"], res["columns"]
    if not rows:
        return "No rows matched your question."
    if len(rows) == 1 and len(cols) == 1:
        out = f"Result: {_cell(rows[0][0])}"
    else:
        out = f"{len(rows)} row(s):\n\n" + markdown_table(cols, rows)
    if res.get("truncated"):
        out += f"\n\n(Output truncated to the first {len(rows)} rows; refine the question to narrow it.)"
    return out


def format_exposure(res: Dict[str, Any]) -> str:
    head = f"Sector exposure for {res['portfolio']} (equities only, bonds excluded)"
    if not res["exposures"]:
        return f"{head}\n\n{res.get('note', 'No equity holdings.')}"
    table = markdown_table(
        ["Sector", "Exposure (% of equities)", "Weight (% of whole portfolio)"],
        [(e["sector"], f"{e['exposure_pct']:.2f}%", f"{e['portfolio_weight_pct']:.2f}%") for e in res["exposures"]])
    return (f"{head}\n\nEquities make up {res['equity_weight_pct']:.2f}% of the portfolio.\n\n{table}\n\n"
            "Exposure % = the sector's share of the equity holdings (adds up to 100%). "
            "Weight % = the sector's share of the entire portfolio.")


def format_error(res: Dict[str, Any]) -> str:
    kind = res.get("error_type")
    if kind == "ambiguous":
        return (f"{res['error']} Which one did you mean?\n- " + "\n- ".join(res["candidates"]))
    if kind == "not_found":
        return f"{res['error']} Available portfolios:\n- " + "\n- ".join(res["available"])
    if kind == "validation":
        return f"I can only run read-only SELECT queries, so I could not run that: {res['error']}"
    return f"The tool failed: {res.get('error', 'unknown error')}"
