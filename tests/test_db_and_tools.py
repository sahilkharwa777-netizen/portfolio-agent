import json
import os
import sqlite3
import unittest

from db import ROOT, build_database
from tools import SQLValidationError, exposure_calculator, find_portfolio, run_sql, validate_sql

with open(os.path.join(ROOT, "expected_exposures.json"), encoding="utf-8") as f:
    GOLDEN = json.load(f)


class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = build_database()

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def test_tables_loaded(self):
        self.assertEqual(self.count("portfolios"), 13)
        self.assertEqual(self.count("securities"), 30)
        self.assertEqual(self.count("holdings"), 95)
        self.assertEqual(self.count("sectors"), 10)

    def test_connection_is_read_only_even_without_validator(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.conn.execute("DELETE FROM portfolios")


class SqlValidationTests(unittest.TestCase):
    def test_allows_select_and_with(self):
        self.assertEqual(validate_sql("SELECT 1;"), "SELECT 1")
        validate_sql("WITH x AS (SELECT 1) SELECT * FROM x")

    def test_rejects_bad_sql(self):
        for bad in ["", "   ", None, "DELETE FROM portfolios", "DROP TABLE portfolios",
                    "SELECT 1; DROP TABLE portfolios", "UPDATE portfolios SET status='x'",
                    "PRAGMA table_info(portfolios)", "SELECT 1; SELECT 2"]:
            with self.subTest(sql=bad), self.assertRaises(SQLValidationError):
                validate_sql(bad)


class RunSqlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = build_database()

    def test_count(self):
        res = run_sql(self.conn, "SELECT COUNT(*) FROM portfolios;")
        self.assertTrue(res["ok"])
        self.assertEqual(res["rows"], [(13,)])
        self.assertFalse(res["truncated"])

    def test_join(self):
        res = run_sql(self.conn, "SELECT s.symbol FROM securities s JOIN sectors sec "
                                 "ON s.sector_id = sec.sector_id WHERE sec.sector_name = 'Technology'")
        self.assertTrue(res["ok"])
        self.assertGreater(len(res["rows"]), 0)

    def test_truncation_flag(self):
        res = run_sql(self.conn, "SELECT * FROM holdings", max_rows=10)
        self.assertEqual(len(res["rows"]), 10)
        self.assertTrue(res["truncated"])

    def test_error_types(self):
        self.assertEqual(run_sql(self.conn, "DELETE FROM holdings")["error_type"], "validation")
        self.assertEqual(run_sql(self.conn, "SELECT nope FROM holdings")["error_type"], "database")

    def test_ground_truth_queries_all_run(self):
        with open(os.path.join(ROOT, "ground_truth_dataset.json"), encoding="utf-8") as f:
            items = json.load(f)["questions"]
        for it in items:
            if it["type"] == "text2sql":
                with self.subTest(id=it["id"]):
                    self.assertTrue(run_sql(self.conn, it["ground_truth"]["sql_query"])["ok"])


class FindPortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = build_database()

    def test_exact_case_insensitive(self):
        self.assertEqual(find_portfolio(self.conn, "tech innovation fund")["portfolio_name"], "Tech Innovation Fund")

    def test_unique_partial(self):
        self.assertEqual(find_portfolio(self.conn, "international equity")["portfolio_name"], "International Equity Fund")

    def test_ambiguous_is_never_guessed(self):
        res = find_portfolio(self.conn, "international")
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_type"], "ambiguous")
        self.assertIn("International Equity Fund", res["candidates"])
        self.assertIn("Total International Index Fund", res["candidates"])

    def test_not_found_and_empty(self):
        self.assertEqual(find_portfolio(self.conn, "Nonexistent")["error_type"], "not_found")
        self.assertEqual(find_portfolio(self.conn, "  ")["error_type"], "invalid_input")

    def test_like_wildcards_are_not_wildcards(self):
        self.assertEqual(find_portfolio(self.conn, "%")["error_type"], "not_found")
        self.assertEqual(find_portfolio(self.conn, "_")["error_type"], "not_found")


class ExposureCalculatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = build_database()

    def test_matches_independent_golden_for_every_portfolio(self):
        """Golden file is built with pandas from the raw CSVs, not with this code."""
        for name, exp in GOLDEN.items():
            with self.subTest(portfolio=name):
                res = exposure_calculator(self.conn, name)
                self.assertTrue(res["ok"])
                got = {e["sector"]: e["exposure_pct"] for e in res["exposures"]}
                self.assertEqual(set(got), set(exp["exposure_pct"]))
                for sector, pct in exp["exposure_pct"].items():
                    self.assertAlmostEqual(got[sector], pct, delta=0.02)
                self.assertAlmostEqual(res["equity_weight_pct"], exp["equity_weight_pct"], delta=0.02)

    def test_hand_checked_values(self):
        res = exposure_calculator(self.conn, "Balanced Portfolio")
        got = {e["sector"]: e["exposure_pct"] for e in res["exposures"]}
        self.assertEqual(got, {"Technology": 61.11, "Financials": 22.22, "Healthcare": 16.67})
        self.assertEqual(res["equity_weight_pct"], 36.0)     # 64% of this portfolio is bonds

    def test_bonds_are_excluded(self):
        res = exposure_calculator(self.conn, "Conservative Income Fund")
        self.assertEqual(res["equity_weight_pct"], 32.0)
        self.assertAlmostEqual(sum(e["exposure_pct"] for e in res["exposures"]), 100, delta=0.05)

    def test_bond_only_portfolio(self):
        res = exposure_calculator(self.conn, "Fixed Income Plus")
        self.assertTrue(res["ok"])
        self.assertEqual(res["exposures"], [])
        self.assertIn("no equity", res["note"])

    def test_sorted_descending(self):
        pcts = [e["exposure_pct"] for e in exposure_calculator(self.conn, "Total Stock Market Index Fund")["exposures"]]
        self.assertEqual(pcts, sorted(pcts, reverse=True))

    def test_ambiguous_name_asks_for_clarification(self):
        res = exposure_calculator(self.conn, "index fund")
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_type"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
