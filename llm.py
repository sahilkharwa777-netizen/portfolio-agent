"""Thin Gemini wrapper: structured JSON output, timeouts, and retry with backoff.

Errors are classified so the agent can tell the user what actually went wrong
(rate limit vs. timeout vs. bad key vs. server outage).
"""
import logging
import time
from typing import Any, Callable, Dict, Optional

from config import ConfigError, Settings

log = logging.getLogger("llm")

RETRYABLE = {"rate_limit", "timeout", "server"}


class LLMError(Exception):
    """Gemini call failed. `kind` is one of rate_limit, timeout, server, auth, bad_request, unknown."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def classify_error(exc: Exception) -> str:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        if code == 429:
            return "rate_limit"
        if code in (401, 403):
            return "auth"
        if code in (400, 404):
            return "bad_request"
        if code >= 500:
            return "server"
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "timeout"
    return "unknown"


_FRIENDLY = {
    "rate_limit": "Gemini rate limit reached. Wait a moment and try again.",
    "timeout": "The Gemini request timed out.",
    "server": "Gemini is temporarily unavailable.",
    "auth": "Gemini rejected the API key (check GEMINI_API_KEY).",
    "bad_request": "Gemini rejected the request (check GEMINI_MODEL).",
    "unknown": "Unexpected error while calling Gemini.",
}


def call_with_retry(fn: Callable[[], Any], max_retries: int = 3, base_delay: float = 1.0,
                    sleep: Callable[[float], None] = time.sleep) -> Any:
    """Run fn(); retry rate-limit / timeout / 5xx errors with exponential backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:  # SDK raises several exception types; classify them all
            kind = classify_error(e)
            if kind == "rate_limit" and "PerDay" in str(e):
                raise LLMError(kind, f"{_FRIENDLY[kind]} (daily free quota used up)") from e
            if kind in RETRYABLE and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                log.warning("Gemini %s error (attempt %d/%d), retrying in %.1fs: %s",
                            kind, attempt + 1, max_retries, delay, e)
                sleep(delay)
                attempt += 1
                continue
            raise LLMError(kind, f"{_FRIENDLY[kind]} ({e})") from e


class GeminiClient:
    """generate_json() -> raw JSON text constrained by a schema; generate_text() -> plain text."""

    def __init__(self, settings: Settings):
        try:
            from google import genai  # imported lazily so the rest of the project is testable offline
            from google.genai import types
        except ImportError as e:
            raise ConfigError("google-genai is not installed. Run: pip install -r requirements.txt") from e
        self._types = types
        self.settings = settings
        self.client = genai.Client(
            api_key=settings.api_key,
            http_options=types.HttpOptions(timeout=int(settings.timeout_s * 1000)))

    def check_model(self) -> None:
        """Fail fast with a clear message if the configured model does not exist."""
        try:
            self.client.models.get(model=self.settings.model)
        except Exception as e:
            kind = classify_error(e)
            if kind == "bad_request":
                raise ConfigError(f"Model '{self.settings.model}' is not available. "
                                  "Set GEMINI_MODEL in .env to a valid model.") from e
            if kind == "auth":
                raise ConfigError("Gemini rejected the API key (check GEMINI_API_KEY).") from e
            log.warning("Could not verify model availability: %s", e)

    def _generate(self, prompt: str, config: Any) -> str:
        def call():
            resp = self.client.models.generate_content(
                model=self.settings.model, contents=prompt, config=config)
            return resp.text or ""
        return call_with_retry(call, self.settings.max_retries)

    def generate_json(self, prompt: str, system: str, schema: Dict[str, Any]) -> str:
        t = self._types
        try:
            cfg = t.GenerateContentConfig(system_instruction=system, temperature=0,
                                          response_mime_type="application/json", response_schema=schema)
        except (ValueError, TypeError):  # schema not accepted by this SDK version -> prompt-only JSON
            cfg = t.GenerateContentConfig(system_instruction=system, temperature=0,
                                          response_mime_type="application/json")
        return self._generate(prompt, cfg)

    def generate_text(self, prompt: str, system: Optional[str] = None) -> str:
        t = self._types
        return self._generate(prompt, t.GenerateContentConfig(system_instruction=system, temperature=0))
