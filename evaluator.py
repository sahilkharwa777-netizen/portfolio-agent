"""Runs the ground-truth questions through the agent and checks the answers.

text2sql questions:
    The agent's SQL is executed and its RESULT SET is compared with the result of the
    ground-truth SQL (values, not SQL text; row order only matters if the truth query has
    ORDER BY; numbers compared with a small tolerance). COLUMN NAMES are compared too:
    with --strict a name mismatch fails the question and extra columns are not allowed;
    without it, a name mismatch (and extra columns) still passes but is reported as a warning.

exposure_calculator questions:
    Compared with expected_exposures.json, which is produced INDEPENDENTLY with pandas
    (scripts/make_expected_exposures.py), not by the function under test. Checks the tool,
    the portfolio, the exact set of sectors, every sector's percentage, and the equity
    weight (which proves bonds were excluded).

Final answer text (both types):
    The natural-language answer the user actually reads is checked against the independent
    expected data: every expected value / sector / percentage must appear in it, in the
    expected response format (single value, row count + table, or exposure table + explanation).
    A question passes only if the data AND the final answer are correct.

Usage:  python evaluator.py [--strict] [--ids 1 9] [--delay 15] [--report eval_report.json]
        --delay waits N seconds between questions (useful on rate-limited free API tiers).
"""
import argparse
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from db import ROOT, build_database
from tools import run_sql

GT_PATH = os.path.join(ROOT, "ground_truth_dataset.json")
EXPECTED_EXPOSURE_PATH = os.path.join(ROOT, "expected_exposures.json")
NUM_TOL = 0.011      # tolerance for SQL numeric values (ROUND(x, 2) differences)
EXPOSURE_TOL = 0.02  # tolerance for exposure percentages (both sides are rounded to 2 dp)


# ---------------------------------------------------------------- comparisons
def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= NUM_TOL
    return a == b


def _row_eq(r1: tuple, r2: tuple) -> bool:
    return len(r1) == len(r2) and all(_eq(a, b) for a, b in zip(r1, r2))


def _same_rows(truth: List[tuple], got: List[tuple], ordered: bool) -> bool:
    if len(truth) != len(got):
        return False
    if ordered:
        return all(_row_eq(t, g) for t, g in zip(truth, got))
    remaining = list(got)
    for t in truth:
        for i, g in enumerate(remaining):
            if _row_eq(t, g):
                del remaining[i]
                break
        else:
            return False
    return True


def _names(cols: List[str]) -> List[str]:
    return [str(c).strip().lower() for c in cols]


def rows_match(truth_rows: List[tuple], got_rows: List[tuple], ordered: bool, strict: bool = False,
               truth_cols: Optional[List[str]] = None, got_cols: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Compare two result sets: values, row order (if `ordered`) and, when column lists are
    given, column names (case-insensitive).

    strict=True : same column count AND same column names, otherwise fail.
    strict=False: extra agent columns are accepted if a projection of them matches the truth;
                  a column-name mismatch passes but is returned as a 'warning:' message.
    """
    nt = len(truth_cols) if truth_cols is not None else (len(truth_rows[0]) if truth_rows else None)
    ng = len(got_cols) if got_cols is not None else (len(got_rows[0]) if got_rows else None)
    if bool(truth_rows) != bool(got_rows):
        return False, f"row count differs (truth {len(truth_rows)}, agent {len(got_rows)})"

    # --- find which of the agent's columns line up with the truth columns -----------------
    if nt is not None and ng is not None and nt == ng:
        projection = tuple(range(ng))
        if truth_rows and not _same_rows(truth_rows, got_rows, ordered):
            return False, "values differ"
    elif nt is not None and ng is not None and ng > nt and not strict and ng <= 8 and truth_rows:
        wanted = _names(truth_cols) if truth_cols is not None and got_cols is not None else None
        projection, fallback = None, None
        for cols in itertools.permutations(range(ng), nt):
            if _same_rows(truth_rows, [tuple(r[i] for i in cols) for r in got_rows], ordered):
                if wanted is None or _names([got_cols[i] for i in cols]) == wanted:
                    projection = cols          # values AND names match: best candidate
                    break
                fallback = fallback or cols    # values match, names do not
        projection = projection or fallback
        if projection is None:
            return False, f"column count differs (truth {nt}, agent {ng}) and no column subset matches"
    else:
        return False, f"column count differs (truth {nt}, agent {ng})"

    # --- column-name validation -------------------------------------------------------------
    notes = []
    if projection != tuple(range(ng or 0)) or (ng is not None and nt is not None and ng != nt):
        notes.append("extra columns tolerated")
    if truth_cols is not None and got_cols is not None:
        truth_names, got_names = _names(truth_cols), _names([got_cols[i] for i in projection])
        if truth_names != got_names:
            detail = f"column names differ (truth {list(truth_cols)}, agent {[got_cols[i] for i in projection]})"
            if strict:
                return False, detail
            notes.append(f"warning: {detail}")
    return True, "; ".join(notes)


def compare_exposure(expected: Dict[str, Any], got: Dict[str, Any], tol: float = EXPOSURE_TOL) -> Tuple[bool, str]:
    """Strict, per-sector comparison against the independently computed expected answer."""
    exp_sectors = expected["exposure_pct"]
    got_sectors = {e["sector"]: e["exposure_pct"] for e in got.get("exposures", [])}
    if set(exp_sectors) != set(got_sectors):
        return False, f"sectors differ: expected {sorted(exp_sectors)}, got {sorted(got_sectors)}"
    for sector, pct in exp_sectors.items():
        if abs(pct - got_sectors[sector]) > tol:
            return False, f"{sector}: expected {pct}%, got {got_sectors[sector]}%"
    got_weights = {e["sector"]: e["portfolio_weight_pct"] for e in got.get("exposures", [])}
    for sector, pct in expected["portfolio_weight_pct"].items():
        if abs(pct - got_weights.get(sector, -1)) > tol:
            return False, f"{sector} portfolio weight: expected {pct}%, got {got_weights.get(sector)}%"
    if abs(expected["equity_weight_pct"] - got.get("equity_weight_pct", -1)) > tol:
        return False, (f"equity weight: expected {expected['equity_weight_pct']}%, "
                       f"got {got.get('equity_weight_pct')}% (bonds not excluded correctly?)")
    return True, ""


# ---------------------------------------------------------------- final-answer checks
def _value_forms(v: Any) -> List[str]:
    """Acceptable text renderings of a value inside an answer."""
    if v is None:
        return ["NULL"]
    forms = {str(v)}
    if isinstance(v, float):
        forms |= {f"{v:,.2f}", f"{v:.2f}"}
    elif isinstance(v, int) and not isinstance(v, bool):
        forms.add(f"{v:,}")
    return sorted(forms)


def check_sql_answer(truth_rows: List[tuple], answer: str) -> Tuple[bool, str]:
    """The answer text must present exactly the expected data in the expected format."""
    low = answer.lower()
    if not truth_rows:
        return ("no rows" in low), "" if "no rows" in low else "expected a 'no rows' answer"
    if "no rows matched" in low:
        return False, "answer says no rows matched but data exists"
    if len(truth_rows) == 1 and len(truth_rows[0]) == 1:
        ok = any(f"Result: {f}" in answer for f in _value_forms(truth_rows[0][0]))
        return ok, "" if ok else f"answer does not state 'Result: {truth_rows[0][0]}'"
    if f"{len(truth_rows)} row(s)" not in answer:
        return False, f"answer does not state the row count ({len(truth_rows)} row(s))"
    table = "\n".join(l for l in answer.splitlines() if l.startswith("|"))
    if not table:
        return False, "answer has no table"
    for row in truth_rows:
        for v in row:
            if not any(f in table for f in _value_forms(v)):
                return False, f"value {v!r} is missing from the answer table"
    return True, ""


def check_exposure_answer(name: str, expected: Dict[str, Any], answer: str) -> Tuple[bool, str]:
    """Answer must name the portfolio, every sector with both percentages, the equity share,
    and say bonds are excluded (or that there are no equities)."""
    low = answer.lower()
    if name not in answer:
        return False, f"answer does not mention the portfolio {name!r}"
    if not expected["exposure_pct"]:
        return ("no equity" in low), "" if "no equity" in low else "answer should say there are no equity holdings"
    if "bonds" not in low:
        return False, "answer does not say that bonds are excluded"
    if f"{expected['equity_weight_pct']:.2f}%" not in answer:
        return False, f"answer does not state the equity share ({expected['equity_weight_pct']:.2f}%)"
    rows = [l for l in answer.splitlines() if l.startswith("|")]
    for sector, pct in expected["exposure_pct"].items():
        line = next((l for l in rows if f"| {sector} |" in l), None)
        if line is None:
            return False, f"sector {sector!r} is missing from the answer table"
        weight = expected["portfolio_weight_pct"][sector]
        for label, value in (("exposure", pct), ("portfolio weight", weight)):
            if f"{value:.2f}%" not in line:
                return False, f"{sector}: {label} {value:.2f}% not shown in the answer"
    if "whole portfolio" not in low and "entire portfolio" not in low:
        return False, "answer does not explain what the two percentages are measured against"
    return True, ""


# ---------------------------------------------------------------- evaluation
def _outcome(passed: bool, message: str, tool: Optional[str], result_ok: bool, answer_ok: Optional[bool]) -> Dict[str, Any]:
    return {"passed": passed, "message": message, "tool": tool,
            "result_correct": result_ok, "answer_correct": answer_ok}


def evaluate_item(agent, conn, item: Dict[str, Any], golden: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """Run one question. Passes only if the tool result AND the final answer text are correct."""
    gt, question = item["ground_truth"], item["question"]
    out = agent.answer_question(question)
    if item["type"] == "text2sql":
        if out.tool != "sql_query":
            return _outcome(False, f"expected tool sql_query, agent used {out.tool}", out.tool, False, None)
        if not out.result.get("ok"):
            return _outcome(False, f"agent SQL failed: {out.result.get('error')}", out.tool, False, None)
        truth = run_sql(conn, gt["sql_query"], max_rows=10_000)
        ordered = "order by" in gt["sql_query"].lower()
        ok, msg = rows_match(truth["rows"], out.result["rows"], ordered, strict,
                             truth["columns"], out.result["columns"])
        if not ok:
            msg += (f"\n      truth: {truth['rows'][:3]}\n      agent: {out.result['rows'][:3]}"
                    f"\n      sql:   {out.args.get('sql')}")
            return _outcome(False, msg, out.tool, False, None)
        ans_ok, ans_msg = check_sql_answer(truth["rows"], out.answer)
    else:  # exposure_calculator
        if out.tool != "exposure_calculator":
            return _outcome(False, f"expected tool exposure_calculator, agent used {out.tool}", out.tool, False, None)
        name = gt["parameters"]["portfolio_name"]
        if not out.result.get("ok"):
            return _outcome(False, f"tool failed: {out.result.get('error')}", out.tool, False, None)
        if out.result["portfolio"] != name:
            return _outcome(False, f"wrong portfolio: expected {name!r}, got {out.result['portfolio']!r}", out.tool, False, None)
        ok, msg = compare_exposure(golden[name], out.result)
        if not ok:
            return _outcome(False, msg, out.tool, False, None)
        ans_ok, ans_msg = check_exposure_answer(name, golden[name], out.answer)
    if not ans_ok:
        return _outcome(False, f"final answer text is wrong: {ans_msg}", out.tool, True, False)
    return _outcome(True, msg, out.tool, True, True)  # msg may carry warnings (e.g. column names differ)


def run_evaluation(agent, conn, items: List[Dict[str, Any]], golden: Dict[str, Any], strict: bool = False,
                   delay: float = 0.0) -> Dict[str, Any]:  # NEW: delay parameter
    results = []
    for idx, it in enumerate(items):
        start = time.perf_counter()
        try:
            outcome = evaluate_item(agent, conn, it, golden, strict)
        except Exception as e:  # one broken question must not stop the whole run
            outcome = _outcome(False, f"evaluator error: {type(e).__name__}: {e}", None, False, None)
        results.append({"id": it["id"], "type": it["type"], "difficulty": it["difficulty"],
                        "question": it["question"], **outcome,
                        "seconds": round(time.perf_counter() - start, 2)})
        if delay > 0 and idx < len(items) - 1:  # NEW: pause between questions (rate limits)
            time.sleep(delay)
    passed = sum(r["passed"] for r in results)
    return {"passed": passed, "total": len(results), "results": results}


def main() -> None:
    from agent import PortfolioAgent
    from config import ConfigError, load_settings
    from llm import GeminiClient

    p = argparse.ArgumentParser(description="Evaluate the portfolio agent on the ground-truth dataset")
    p.add_argument("--strict", action="store_true", help="require identical column counts for SQL answers")
    p.add_argument("--ids", type=int, nargs="*", help="only run these question ids")
    p.add_argument("--report", default="eval_report.json", help="where to write the JSON report")
    p.add_argument("--delay", type=float, default=0.0,  # NEW
                   help="seconds to wait between questions (avoids free-tier rate limits)")
    args = p.parse_args()
    try:
        settings = load_settings()
        llm = GeminiClient(settings)
        llm.check_model()
    except ConfigError as e:
        raise SystemExit(f"Configuration error: {e}")
    conn = build_database()
    agent = PortfolioAgent(conn, llm, settings.max_rows)
    with open(GT_PATH, encoding="utf-8") as f:
        items = json.load(f)["questions"]
    if args.ids:
        items = [i for i in items if i["id"] in args.ids]
    with open(EXPECTED_EXPOSURE_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    report = run_evaluation(agent, conn, items, golden, args.strict, args.delay)  # NEW: pass delay
    for r in report["results"]:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] Q{r['id']} ({r['type']}, {r['difficulty']}, {r['seconds']}s) {r['question']}")
        if r["message"]:
            print(f"    -> {r['message']}")
    by_type: Dict[str, List[bool]] = {}
    for r in report["results"]:
        by_type.setdefault(r["type"], []).append(r["passed"])
    print("\n" + "  ".join(f"{t}: {sum(v)}/{len(v)}" for t, v in by_type.items()))
    answered = [r for r in report["results"] if r["result_correct"]]
    print(f"Final answer text correct: {sum(bool(r['answer_correct']) for r in answered)}/{len(answered)} "
          f"(of questions whose data was correct)")
    print(f"Score: {report['passed']}/{report['total']} ({report['passed'] / max(report['total'], 1):.0%})")
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report written to {args.report}")
    sys.exit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()