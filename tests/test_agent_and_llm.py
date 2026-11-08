import unittest

from agent import PortfolioAgent
from db import build_database
from formatter import format_exposure, format_sql_result
from llm import LLMError, call_with_retry, classify_error
from router import DecisionError, parse_decision
from tests.fakes import ScriptedLLM, api_error


class FakeHttpError(Exception):
    def __init__(self, code):
        super().__init__(f"http {code}")
        self.code = code


class ClassifyAndRetryTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(classify_error(FakeHttpError(429)), "rate_limit")
        self.assertEqual(classify_error(FakeHttpError(503)), "server")
        self.assertEqual(classify_error(FakeHttpError(403)), "auth")
        self.assertEqual(classify_error(FakeHttpError(404)), "bad_request")
        self.assertEqual(classify_error(TimeoutError()), "timeout")
        self.assertEqual(classify_error(ValueError("x")), "unknown")

    def test_retries_then_succeeds(self):
        calls, sleeps = [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise FakeHttpError(429)
            return "ok"
        self.assertEqual(call_with_retry(flaky, max_retries=3, base_delay=1, sleep=sleeps.append), "ok")
        self.assertEqual(sleeps, [1, 2])   # exponential backoff

    def test_gives_up_after_max_retries(self):
        with self.assertRaises(LLMError) as ctx:
            call_with_retry(lambda: (_ for _ in ()).throw(FakeHttpError(503)), max_retries=2, sleep=lambda s: None)
        self.assertEqual(ctx.exception.kind, "server")

    def test_auth_errors_are_not_retried(self):
        calls = []

        def bad_key():
            calls.append(1)
            raise FakeHttpError(403)
        with self.assertRaises(LLMError) as ctx:
            call_with_retry(bad_key, max_retries=3, sleep=lambda s: None)
        self.assertEqual((len(calls), ctx.exception.kind), (1, "auth"))


class ParseDecisionTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_decision('{"tool":"sql_query","sql":"SELECT 1"}').sql, "SELECT 1")
        self.assertEqual(parse_decision('```json\n{"tool":"exposure_calculator","portfolio_name":"X"}\n```').portfolio_name, "X")

    def test_invalid(self):
        for bad in ["not json", "[]", '{"tool":"drop_tables"}', '{"tool":"sql_query"}',
                    '{"tool":"exposure_calculator"}', '{"tool":"sql_query","sql":"  "}', "", None]:
            with self.subTest(reply=bad), self.assertRaises(DecisionError):
                parse_decision(bad)

    def test_none_tool_gets_default_reply(self):
        self.assertTrue(parse_decision('{"tool":"none"}').reply)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = build_database()

    def agent(self, llm):
        return PortfolioAgent(self.conn, llm)

    def test_sql_question_uses_exactly_one_llm_call(self):
        llm = ScriptedLLM([{"tool": "sql_query", "sql": "SELECT COUNT(*) FROM portfolios;"}])
        out = self.agent(llm).answer_question("How many portfolios?")
        self.assertTrue(out.ok)
        self.assertEqual(out.answer, "Result: 13")
        self.assertEqual(len(llm.json_calls) + len(llm.text_calls), 1)

    def test_exposure_question(self):
        llm = ScriptedLLM([{"tool": "exposure_calculator", "portfolio_name": "Tech Innovation Fund"}])
        out = self.agent(llm).answer_question("Sector exposure of tech innovation?")
        self.assertEqual(out.tool, "exposure_calculator")
        self.assertIn("Technology", out.answer)
        self.assertIn("100.00%", out.answer)

    def test_malformed_json_is_retried_once_with_feedback(self):
        llm = ScriptedLLM(["this is not json", {"tool": "sql_query", "sql": "SELECT COUNT(*) FROM portfolios"}])
        out = self.agent(llm).answer_question("How many portfolios?")
        self.assertTrue(out.ok)
        self.assertIn("rejected", llm.json_calls[1])

    def test_persistent_garbage_fails_gracefully(self):
        llm = ScriptedLLM(["garbage", "still garbage"])
        out = self.agent(llm).answer_question("anything")
        self.assertFalse(out.ok)
        self.assertIn("rephras", out.answer)

    def test_invalid_tool_name_is_rejected(self):
        llm = ScriptedLLM([{"tool": "delete_everything"}, {"tool": "delete_everything"}])
        self.assertFalse(self.agent(llm).answer_question("x").ok)

    def test_bad_sql_is_repaired_and_repair_is_validated(self):
        llm = ScriptedLLM([{"tool": "sql_query", "sql": "SELECT COUNT(*) FROM portfolio"}],
                          ["```sql\nSELECT COUNT(*) FROM portfolios\n```"])
        out = self.agent(llm).answer_question("How many portfolios?")
        self.assertTrue(out.ok)
        self.assertEqual(out.result["rows"], [(13,)])
        self.assertTrue(out.result["repaired"])

    def test_malicious_repair_is_blocked(self):
        llm = ScriptedLLM([{"tool": "sql_query", "sql": "SELECT * FROM nope"}], ["DROP TABLE portfolios"])
        out = self.agent(llm).answer_question("x")
        self.assertFalse(out.ok)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0], 13)

    def test_destructive_sql_is_rejected_without_asking_llm_to_fix_it(self):
        llm = ScriptedLLM([{"tool": "sql_query", "sql": "DELETE FROM portfolios"}])
        out = self.agent(llm).answer_question("delete all portfolios")
        self.assertFalse(out.ok)
        self.assertEqual(llm.text_calls, [])
        self.assertIn("read-only", out.answer)

    def test_api_failures_are_reported_not_raised(self):
        for kind in ("rate_limit", "timeout", "auth"):
            with self.subTest(kind=kind):
                out = self.agent(ScriptedLLM([api_error(kind)])).answer_question("x")
                self.assertFalse(out.ok)
                self.assertIn("AI service problem", out.answer)

    def test_ambiguous_portfolio_asks_user(self):
        llm = ScriptedLLM([{"tool": "exposure_calculator", "portfolio_name": "international"}])
        out = self.agent(llm).answer_question("exposure of international")
        self.assertFalse(out.ok)
        self.assertIn("Which one did you mean", out.answer)

    def test_unrelated_question_and_empty_input(self):
        llm = ScriptedLLM([{"tool": "none", "reply": "I only answer portfolio questions."}])
        self.assertEqual(self.agent(llm).answer_question("weather?").tool, "none")
        self.assertFalse(self.agent(ScriptedLLM()).answer_question("   ").ok)


class FormatterTests(unittest.TestCase):
    def test_truncation_notice(self):
        res = {"ok": True, "columns": ["a"], "rows": [(1,), (2,)], "truncated": True}
        self.assertIn("truncated", format_sql_result(res))

    def test_empty_result(self):
        self.assertIn("No rows", format_sql_result({"ok": True, "columns": ["a"], "rows": [], "truncated": False}))

    def test_exposure_explains_both_percentages(self):
        res = {"portfolio": "P", "equity_weight_pct": 36.0, "exposures": [
            {"sector": "Technology", "exposure_pct": 100.0, "portfolio_weight_pct": 36.0}]}
        text = format_exposure(res)
        self.assertIn("% of equities", text)
        self.assertIn("whole portfolio", text)


if __name__ == "__main__":
    unittest.main()
