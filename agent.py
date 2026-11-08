"""Portfolio Agent (CLI).

Flow:  question -> LLM routes to a tool (validated JSON) -> tool runs on the DB
       -> answer formatted deterministically from the tool result.
Errors are handled by type: LLM/API, routing/parsing, validation and database.
"""
import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from db import build_database, get_schema_text
from formatter import format_error, format_exposure, format_sql_result
from llm import LLMError
from router import (DECISION_SCHEMA, ROUTER_PROMPT, Decision, DecisionError, parse_decision,
                    strip_fences)
from tools import exposure_calculator, run_sql

log = logging.getLogger("agent")


@dataclass
class AgentResponse:
    answer: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True


class PortfolioAgent:
    def __init__(self, conn, llm, max_rows: int = 200):
        self.conn, self.llm, self.max_rows = conn, llm, max_rows
        names = [r[0] for r in conn.execute("SELECT portfolio_name FROM portfolios ORDER BY portfolio_id")]
        self.system = ROUTER_PROMPT.format(schema=get_schema_text(), portfolios=", ".join(names))

    # ---- step 1: routing ------------------------------------------------
    def route(self, question: str) -> Decision:
        """Ask the LLM for a decision; if the reply is invalid, tell it why and retry once."""
        prompt = question
        for attempt in (1, 2):
            raw = self.llm.generate_json(prompt, self.system, DECISION_SCHEMA)
            try:
                return parse_decision(raw)
            except DecisionError as e:
                log.warning("invalid routing reply (attempt %d): %s | %r", attempt, e, raw[:200])
                if attempt == 2:
                    raise
                prompt = f"{question}\n\nYour previous reply was rejected: {e} Reply with valid JSON only."
        raise DecisionError("unreachable")  # pragma: no cover

    # ---- step 2: tool execution -----------------------------------------
    def _run_sql_with_repair(self, sql: str) -> Dict[str, Any]:
        res = run_sql(self.conn, sql, self.max_rows)
        if not res["ok"] and res["error_type"] == "database":  # only DB errors are worth an LLM repair
            log.info("asking LLM to repair SQL: %s", res["error"])
            fixed = self.llm.generate_text(
                f"This SQLite query failed.\nError: {res['error']}\nQuery: {res['sql']}\n"
                "Return ONLY the corrected SELECT query, nothing else.", self.system)
            res = run_sql(self.conn, strip_fences(fixed), self.max_rows)  # repaired SQL is re-validated
            res["repaired"] = True
        return res

    # ---- full pipeline ----------------------------------------------------
    def answer_question(self, question: str) -> AgentResponse:
        question = (question or "").strip()
        if not question:
            return AgentResponse("Please type a question.", ok=False)
        start = time.perf_counter()
        try:
            decision = self.route(question)
        except LLMError as e:
            log.error("LLM failure kind=%s: %s", e.kind, e)
            return AgentResponse(f"AI service problem: {e}", ok=False)
        except DecisionError as e:
            log.error("routing failed: %s", e)
            return AgentResponse("I couldn't work out how to answer that. Try rephrasing the question.", ok=False)
        log.info("decision tool=%s args=%s", decision.tool,
                 {k: v for k, v in vars(decision).items() if v and k != "tool"})

        if decision.tool == "none":
            return AgentResponse(decision.reply, tool="none")
        try:
            if decision.tool == "sql_query":
                res = self._run_sql_with_repair(decision.sql)
                args = {"sql": res.get("sql", decision.sql)}
                fmt = format_sql_result
            else:
                res = exposure_calculator(self.conn, decision.portfolio_name)
                args = {"portfolio_name": decision.portfolio_name}
                fmt = format_exposure
        except LLMError as e:  # raised by the SQL-repair call
            return AgentResponse(f"AI service problem: {e}", tool=decision.tool, ok=False)
        log.info("answered in %.0fms ok=%s", (time.perf_counter() - start) * 1000, res.get("ok"))
        answer = fmt(res) if res.get("ok") else format_error(res)
        return AgentResponse(answer, tool=decision.tool, args=args, result=res, ok=bool(res.get("ok")))


def main() -> None:
    from config import ConfigError, load_settings
    from llm import GeminiClient

    parser = argparse.ArgumentParser(description="Portfolio analytics agent")
    parser.add_argument("question", nargs="?", help="Ask one question and exit (omit for interactive mode)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log tool choices, SQL and timings")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings()
        llm = GeminiClient(settings)
        llm.check_model()
    except ConfigError as e:
        raise SystemExit(f"Configuration error: {e}")
    agent = PortfolioAgent(build_database(), llm, settings.max_rows)

    def ask(q: str) -> None:
        out = agent.answer_question(q)
        print(f"\n[tool: {out.tool}] {out.args if out.args else ''}\n{out.answer}")

    if args.question:
        ask(args.question)
        return
    print("Portfolio Agent ready. Type a question, or 'exit'.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            ask(q)


if __name__ == "__main__":
    main()
