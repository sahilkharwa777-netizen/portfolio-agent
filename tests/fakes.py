"""Test doubles that stand in for Gemini (no network / API key needed)."""
import json
from typing import List

from llm import LLMError


class ScriptedLLM:
    """Returns pre-written replies in order and records every call."""

    def __init__(self, json_replies: List = None, text_replies: List = None):
        self.json_replies = list(json_replies or [])
        self.text_replies = list(text_replies or [])
        self.json_calls, self.text_calls = [], []

    def generate_json(self, prompt, system, schema):
        self.json_calls.append(prompt)
        reply = self.json_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    def generate_text(self, prompt, system=None):
        self.text_calls.append(prompt)
        reply = self.text_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class OracleLLM:
    """Always routes exactly as the ground-truth dataset says. Used to test the whole
    evaluator pipeline offline (it says nothing about real Gemini accuracy)."""

    def __init__(self, items):
        self.by_question = {i["question"]: i for i in items}

    def generate_json(self, prompt, system, schema):
        gt = self.by_question[prompt]["ground_truth"]
        if "sql_query" in gt:
            return json.dumps({"tool": "sql_query", "sql": gt["sql_query"]})
        return json.dumps({"tool": "exposure_calculator", "portfolio_name": gt["parameters"]["portfolio_name"]})

    def generate_text(self, prompt, system=None):
        raise AssertionError("oracle never needs SQL repair")


def api_error(kind="rate_limit"):
    return LLMError(kind, f"simulated {kind}")
