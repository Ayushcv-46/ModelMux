"""
Credential and sensitive data sanitization for ModelMux.
Ensures API keys, bearer tokens, passwords, and authorization headers
are never leaked in attempt logs, database records, or error messages.
"""

import re
from typing import Any

# Regular expression patterns for common credentials and sensitive tokens
SENSITIVE_PATTERNS = [
    # Authorization header / Bearer token (e.g. "Bearer ya29...", "Bearer sk-...")
    (re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", re.IGNORECASE), r"\1[REDACTED]"),
    # Standard OpenAI / Groq / Anthropic API keys (e.g. sk-..., gsk-..., gsk_...)
    (re.compile(r"\b((?:g?sk[-_])[A-Za-z0-9_\-\.]{8,})\b", re.IGNORECASE), r"[REDACTED_API_KEY]"),
    # Google Gemini API key format (e.g. AIzaSy...)
    (re.compile(r"\b(AIza[0-9A-Za-z\-_]{20,})\b"), r"[REDACTED_API_KEY]"),
    # URI query parameters and headers: key=..., api_key=..., token=..., secret=...
    (re.compile(r"((?:api[_-]?key|key|token|secret|password)=)[^&\s]+", re.IGNORECASE), r"\1[REDACTED]"),
    # Generic Authorization: header lines
    (re.compile(r"(Authorization:\s*)[^\r\n]+", re.IGNORECASE), r"\1[REDACTED]"),
]


def sanitize_error_message(message: str) -> str:
    """Scrub sensitive credentials, tokens, and keys from an error string."""
    if not message:
        return ""
    sanitized = str(message)
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_error(err: Any) -> str | None:
    """
    Produce a clean error category and sanitized error message.
    Never exposes raw secrets or credentials.
    """
    if err is None:
        return None
    if isinstance(err, str):
        return sanitize_error_message(err)
    category = type(err).__name__
    msg = str(err).strip()
    if not msg:
        return category
    return f"{category}: {sanitize_error_message(msg)}"
