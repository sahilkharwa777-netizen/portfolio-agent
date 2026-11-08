"""Central configuration + validation (API key, model name, limits)."""
import os
from dataclasses import dataclass

try:  # python-dotenv is optional at import time so tests run without it
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


DEFAULT_MODEL = "gemini-3.8-flash"  # gemini-2.0-flash was shut down on 2026-06-01


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str = DEFAULT_MODEL
    timeout_s: float = 30.0
    max_retries: int = 3
    max_rows: int = 200


def load_settings() -> Settings:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key or api_key == "your_key_here":
        raise ConfigError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and paste your key "
            "from https://aistudio.google.com/apikey")
    model = (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    try:
        timeout_s = float(os.getenv("GEMINI_TIMEOUT_S", "30"))
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
        max_rows = int(os.getenv("MAX_ROWS", "200"))
    except ValueError as e:
        raise ConfigError(f"Invalid numeric setting in environment: {e}") from e
    if timeout_s <= 0 or max_retries < 0 or max_rows <= 0:
        raise ConfigError("GEMINI_TIMEOUT_S and MAX_ROWS must be > 0, GEMINI_MAX_RETRIES >= 0.")
    return Settings(api_key=api_key, model=model, timeout_s=timeout_s,
                    max_retries=max_retries, max_rows=max_rows)
