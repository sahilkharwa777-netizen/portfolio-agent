"""Turns the LLM's routing reply into a validated Decision (tool + arguments)."""
import json
import re
from dataclasses import dataclass
from typing import Optional

TOOLS = ("sql_query", "exposure_calculator", "none")

# JSON schema handed to Gemini so it is *constrained* to this shape.
DECISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tool": {"type": "STRING", "enum": list(TOOLS)},
        "sql": {"type": "STRING"},
        "portfolio_name": {"type": "STRING"},
        "reply": {"type": "STRING"},
    },
    "required": ["tool"],
}

ROUTER_PROMPT = """You are a portfolio analytics assistant with two tools.

TOOL "sql_query": answers any data question with ONE read-only SQLite SELECT. Database schema:
{schema}
Notes: portfolios.status is 'Active' or 'Passive'; target_risk_level is 'Low','Medium' or 'High';
securities.asset_type is 'Stock' or 'Bond'. Value of a holding = quantity * current_price.

SQL style rules:
- Return only the columns the question asks for: the name/symbol columns plus the requested measures.
  Do NOT add id columns, status, asset type, market cap or other extras.
- Exception: when listing individual holdings or transactions (e.g. "top 5 holdings"), put the
  record's own id first (holding_id / transaction_id), then symbol and company name, then quantity
  and the measure asked for (e.g. cost_basis).
- "Total portfolio value" means SUM(quantity * current_price) over that portfolio's holdings.
  Use portfolios.total_aum only when the question says AUM.
- When asked for a part and its share of a total, return the name, the part, the total and the
  percentage, in that order. ROUND percentages and ratios to 2 decimals. Compute the part and the
  whole separately (CTEs or subqueries) and list only the groups where the part exists, so groups
  with no matching rows are left out. Do not use CASE WHEN ... ELSE 0 to include empty groups.
- Ordering: if the question computes a percentage, ratio or average per group, ORDER BY that value
  DESC. For other grouped numeric results, ORDER BY the main measure DESC. For plain lists
  (names, securities), no ORDER BY.
- Naming: if the result is one single value, do not alias it (write SELECT COUNT(*), SELECT SUM(total_aum)).
  If the result has several columns, give every computed column a short descriptive snake_case alias.
- Do not join risk_metrics or portfolio_performance unless the question is about risk or performance.
  Compute "diversification" from holdings and sectors (sector count, number of holdings, and their ratio).

TOOL "exposure_calculator": sector exposure breakdown (equities only) for ONE portfolio.
Use it when the user asks for sector exposure / sector breakdown / sector allocation of a portfolio.
Set portfolio_name to the portfolio's full name. Existing portfolios: {portfolios}

TOOL "none": the question is unrelated to portfolio data; put a short reply in "reply".

Answer ONLY with JSON: {{"tool": ..., "sql": ..., "portfolio_name": ..., "reply": ...}} (omit unused fields)."""

class DecisionError(Exception):
    """The LLM reply could not be turned into a valid tool decision."""


@dataclass
class Decision:
    tool: str
    sql: Optional[str] = None
    portfolio_name: Optional[str] = None
    reply: Optional[str] = None


def strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json|sql)?\s*|\s*```$", "", (text or "").strip(), flags=re.I).strip()


def parse_decision(text: str) -> Decision:
    """Parse + validate. Raises DecisionError with a message the LLM can use to self-correct."""
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, TypeError) as e:
        raise DecisionError(f"Reply was not valid JSON ({e}).") from e
    if not isinstance(data, dict):
        raise DecisionError("Reply must be a JSON object.")
    tool = data.get("tool")
    if tool not in TOOLS:
        raise DecisionError(f"Unknown tool {tool!r}; must be one of {list(TOOLS)}.")
    dec = Decision(tool=tool, sql=data.get("sql"), portfolio_name=data.get("portfolio_name"),
                   reply=data.get("reply"))
    if tool == "sql_query" and not (isinstance(dec.sql, str) and dec.sql.strip()):
        raise DecisionError("Tool 'sql_query' requires a non-empty 'sql' string.")
    if tool == "exposure_calculator" and not (isinstance(dec.portfolio_name, str) and dec.portfolio_name.strip()):
        raise DecisionError("Tool 'exposure_calculator' requires a 'portfolio_name' string.")
    if tool == "none" and not (isinstance(dec.reply, str) and dec.reply.strip()):
        dec.reply = "I can only answer questions about the portfolio data (funds, holdings, securities, sectors)."
    return dec