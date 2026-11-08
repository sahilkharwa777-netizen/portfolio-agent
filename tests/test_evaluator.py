import json
import os
import unittest

from agent import PortfolioAgent
from db import ROOT, build_database
from agent import AgentResponse
from evaluator import (check_exposure_answer, check_sql_answer, compare_exposure, rows_match,
                       run_evaluation)
from formatter import format_exposure, format_sql_result
from tools import exposure_calculator, run_sql
from tests.fakes import OracleLLM

with open(os.path.join(ROOT, "expected_exposures.json"), encoding="utf-8") as f:
    GOLDEN = json.load(f)
with open(os.path.join(ROOT, "ground_truth_dataset.json"), encoding="utf-8") as f:
    ITEMS = json.load(f)["questions"]


def as_result(sector_to_pct, equity=100.0):
    return {"equity_weight_pct": equity,
            "exposures": [{"sector": s, "exposure_pct": p, "portfolio_weight_pct": p * equity / 100}
                          for s, p in sector_to_pct.items()]}


class CompareExposureTests(unittest.TestCase):
    EXPECTED = {"equity_weight_pct": 100.0,
                "exposure_pct": {"Technology": 40.0, "Healthcare": 35.0, "Financials": 25.0},
                "portfolio_weight_pct": {"Technology": 40.0, "Healthcare": 35.0, "Financials": 25.0}}

    def test_correct_answer_passes(self):
        ok, _ = compare_exposure(self.EXPECTED, as_result({"Technology": 40, "Healthcare": 35, "Financials": 25}))
        self.assertTrue(ok)

    def test_wrong_distribution_that_still_sums_to_100_fails(self):
        """The exact weakness found in review: 10/10/80 also sums to 100."""
        bad = as_result({"Technology": 10, "Healthcare": 10, "Financials": 80})
        self.assertAlmostEqual(sum(e["exposure_pct"] for e in bad["exposures"]), 100)
        ok, msg = compare_exposure(self.EXPECTED, bad)
        self.assertFalse(ok)
        self.assertIn("Technology", msg)

    def test_wrong_sector_names_fail(self):
        ok, _ = compare_exposure(self.EXPECTED, as_result({"Technology": 40, "Energy": 35, "Financials": 25}))
        self.assertFalse(ok)

    def test_missing_sector_fails(self):
        ok, _ = compare_exposure(self.EXPECTED, as_result({"Technology": 60, "Healthcare": 40}))
        self.assertFalse(ok)

    def test_floating_point_noise_tolerated(self):
        ok, _ = compare_exposure(self.EXPECTED, as_result({"Technology": 40.004, "Healthcare": 34.996, "Financials": 25}))
        self.assertTrue(ok)

    def test_bonds_counted_as_equity_fails(self):
        ok, msg = compare_exposure(self.EXPECTED, as_result({"Technology": 40, "Healthcare": 35, "Financials": 25}, equity=64.0))
        self.assertFalse(ok)


class RowsMatchTests(unittest.TestCase):
    def test_unordered(self):
        self.assertTrue(rows_match([(1,), (2,)], [(2,), (1,)], ordered=False)[0])

    def test_ordered_detects_wrong_order(self):
        self.assertFalse(rows_match([(1,), (2,)], [(2,), (1,)], ordered=True)[0])

    def test_numeric_tolerance(self):
        self.assertTrue(rows_match([("a", 1.005)], [("a", 1.0)], ordered=True)[0])
        self.assertFalse(rows_match([("a", 1.5)], [("a", 1.0)], ordered=True)[0])

    def test_wrong_values_fail(self):
        self.assertFalse(rows_match([(13,)], [(12,)], ordered=False)[0])

    def test_extra_columns_tolerated_unless_strict(self):
        truth, got = [("Growth",), ("Income",)], [(1, "Growth"), (2, "Income")]
        self.assertTrue(rows_match(truth, got, False)[0])
        self.assertFalse(rows_match(truth, got, False, strict=True)[0])

    def test_extra_columns_with_wrong_values_still_fail(self):
        self.assertFalse(rows_match([("Growth",)], [(1, "Other")], False)[0])


class EndToEndEvaluatorTests(unittest.TestCase):
    """Whole pipeline with an oracle LLM: validates the evaluator + agent plumbing, not Gemini."""

    def test_oracle_scores_10_of_10(self):
        conn = build_database()
        agent = PortfolioAgent(conn, OracleLLM(ITEMS))
        report = run_evaluation(agent, conn, ITEMS, GOLDEN)
        failures = [r for r in report["results"] if not r["passed"]]
        self.assertEqual(failures, [])
        self.assertEqual((report["passed"], report["total"]), (10, 10))

    def test_evaluator_catches_wrong_portfolio(self):
        conn = build_database()

        class WrongPortfolio(OracleLLM):
            def generate_json(self, prompt, system, schema):
                return json.dumps({"tool": "exposure_calculator", "portfolio_name": "Growth Equity Fund"})

        agent = PortfolioAgent(conn, WrongPortfolio(ITEMS))
        report = run_evaluation(agent, conn, [i for i in ITEMS if i["id"] == 9], GOLDEN)
        self.assertEqual(report["passed"], 0)

    def test_evaluator_catches_wrong_tool(self):
        conn = build_database()

        class AlwaysSql(OracleLLM):
            def generate_json(self, prompt, system, schema):
                return json.dumps({"tool": "sql_query", "sql": "SELECT COUNT(*) FROM portfolios"})

        agent = PortfolioAgent(conn, AlwaysSql(ITEMS))
        report = run_evaluation(agent, conn, [i for i in ITEMS if i["id"] in (4, 9)], GOLDEN)
        self.assertEqual(report["passed"], 0)


class ColumnNameTests(unittest.TestCase):
    def test_same_values_different_names_fails_in_strict_mode(self):
        ok, msg = rows_match([(13,)], [(13,)], False, strict=True,
                             truth_cols=["COUNT(*)"], got_cols=["total_portfolios"])
        self.assertFalse(ok)
        self.assertIn("column names differ", msg)

    def test_same_values_different_names_passes_with_warning_by_default(self):
        ok, msg = rows_match([(13,)], [(13,)], False, truth_cols=["COUNT(*)"], got_cols=["total_portfolios"])
        self.assertTrue(ok)
        self.assertTrue(msg.startswith("warning:"))

    def test_matching_names_are_case_insensitive(self):
        ok, msg = rows_match([("a", 1)], [("a", 1)], True, strict=True,
                             truth_cols=["Name", "N"], got_cols=["name", "n"])
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_swapped_column_order_fails(self):
        ok, _ = rows_match([("a", 1)], [(1, "a")], True, strict=True, truth_cols=["name", "n"], got_cols=["n", "name"])
        self.assertFalse(ok)

    def test_extra_column_prefers_projection_with_matching_names(self):
        truth_cols, got_cols = ["portfolio_name"], ["portfolio_id", "portfolio_name"]
        ok, msg = rows_match([("Growth",)], [(1, "Growth")], False, False, truth_cols, got_cols)
        self.assertTrue(ok)
        self.assertIn("extra columns tolerated", msg)
        self.assertNotIn("warning", msg)

    def test_extra_columns_never_allowed_in_strict_mode(self):
        ok, _ = rows_match([("Growth",)], [(1, "Growth")], False, True, ["portfolio_name"], ["portfolio_id", "portfolio_name"])
        self.assertFalse(ok)

    def test_empty_results_still_compare_names_in_strict_mode(self):
        self.assertFalse(rows_match([], [], False, True, ["a"], ["b"])[0])
        self.assertTrue(rows_match([], [], False, True, ["a"], ["A"])[0])


class FinalAnswerTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = build_database()

    def sql_answer(self, sql):
        res = run_sql(self.conn, sql)
        return res["rows"], format_sql_result(res)

    def test_all_ground_truth_sql_answers_pass_the_text_check(self):
        for it in ITEMS:
            if it["type"] == "text2sql":
                with self.subTest(id=it["id"]):
                    rows, text = self.sql_answer(it["ground_truth"]["sql_query"])
                    self.assertEqual(check_sql_answer(rows, text), (True, ""))

    def test_single_value_must_be_stated(self):
        rows, _ = self.sql_answer("SELECT COUNT(*) FROM portfolios")
        self.assertTrue(check_sql_answer(rows, "Result: 13")[0])
        self.assertFalse(check_sql_answer(rows, "Result: 12")[0])
        self.assertFalse(check_sql_answer(rows, "There are thirteen portfolios")[0])

    def test_table_answer_missing_a_value_fails(self):
        rows, text = self.sql_answer("SELECT portfolio_name FROM portfolios WHERE status='Active'")
        self.assertTrue(check_sql_answer(rows, text)[0])
        self.assertFalse(check_sql_answer(rows, text.replace("Growth Equity Fund", "Something Else"))[0])

    def test_wrong_row_count_or_no_rows_claim_fails(self):
        rows, text = self.sql_answer("SELECT portfolio_name FROM portfolios WHERE status='Active'")
        self.assertFalse(check_sql_answer(rows, text.replace("10 row(s)", "9 row(s)"))[0])
        self.assertFalse(check_sql_answer(rows, "No rows matched your question.")[0])

    def test_empty_truth_requires_a_no_rows_answer(self):
        self.assertTrue(check_sql_answer([], "No rows matched your question.")[0])
        self.assertFalse(check_sql_answer([], "Result: 5")[0])

    def test_exposure_answers_pass_for_all_portfolios(self):
        for name, exp in GOLDEN.items():
            with self.subTest(portfolio=name):
                text = format_exposure(exposure_calculator(self.conn, name))
                self.assertEqual(check_exposure_answer(name, exp, text), (True, ""))

    def test_exposure_answer_with_wrong_percentage_fails(self):
        name = "Balanced Portfolio"
        text = format_exposure(exposure_calculator(self.conn, name))
        ok, msg = check_exposure_answer(name, GOLDEN[name], text.replace("61.11%", "16.11%"))
        self.assertFalse(ok)
        self.assertIn("Technology", msg)

    def test_exposure_answer_missing_sector_or_explanation_fails(self):
        name = "Balanced Portfolio"
        text = format_exposure(exposure_calculator(self.conn, name))
        no_sector = "\n".join(l for l in text.splitlines() if "| Healthcare |" not in l)
        self.assertFalse(check_exposure_answer(name, GOLDEN[name], no_sector)[0])
        no_note = text.split("Exposure % =")[0].replace("whole portfolio", "x")
        self.assertFalse(check_exposure_answer(name, GOLDEN[name], no_note)[0])

    def test_bond_only_answer_must_say_no_equities(self):
        name = "Fixed Income Plus"
        text = format_exposure(exposure_calculator(self.conn, name))
        self.assertTrue(check_exposure_answer(name, GOLDEN[name], text)[0])
        self.assertFalse(check_exposure_answer(name, GOLDEN[name], f"Sector exposure for {name}: Technology 100%")[0])


class EvaluatorUsesColumnsAndAnswerTextTests(unittest.TestCase):
    """End to end: correct DATA is not enough, names (strict) and the answer text count too."""

    def run_one(self, llm_cls, question_id, strict=False, agent_cls=PortfolioAgent):
        conn = build_database()
        agent = agent_cls(conn, llm_cls(ITEMS))
        return run_evaluation(agent, conn, [i for i in ITEMS if i["id"] == question_id], GOLDEN, strict)["results"][0]

    def test_oracle_passes_all_10_in_strict_mode(self):
        conn = build_database()
        report = run_evaluation(PortfolioAgent(conn, OracleLLM(ITEMS)), conn, ITEMS, GOLDEN, strict=True)
        self.assertEqual((report["passed"], report["total"]), (10, 10))
        self.assertTrue(all(r["result_correct"] and r["answer_correct"] for r in report["results"]))

    def test_aliased_column_name_fails_strict_but_passes_default_with_warning(self):
        class Aliased(OracleLLM):
            def generate_json(self, prompt, system, schema):
                return json.dumps({"tool": "sql_query", "sql": "SELECT COUNT(*) AS total_portfolios FROM portfolios"})
        strict = self.run_one(Aliased, 1, strict=True)
        self.assertFalse(strict["passed"])
        self.assertIn("column names differ", strict["message"])
        lenient = self.run_one(Aliased, 1, strict=False)
        self.assertTrue(lenient["passed"])
        self.assertIn("warning", lenient["message"])

    def test_correct_data_but_wrong_answer_text_fails(self):
        class LyingAgent(PortfolioAgent):
            def answer_question(self, question):
                out = super().answer_question(question)
                return AgentResponse("The answer is 42.", out.tool, out.args, out.result, out.ok)
        for qid in (1, 9):   # one SQL question, one exposure question
            with self.subTest(question=qid):
                r = self.run_one(OracleLLM, qid, agent_cls=LyingAgent)
                self.assertTrue(r["result_correct"])
                self.assertFalse(r["answer_correct"])
                self.assertFalse(r["passed"])
                self.assertIn("final answer text is wrong", r["message"])


if __name__ == "__main__":
    unittest.main()
