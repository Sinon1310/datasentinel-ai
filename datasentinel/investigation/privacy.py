"""Privacy controls for anything that leaves the process (e.g. LLM prompts).

Two modes:

* ``metadata_only`` (default) — no raw cell values. Category labels become
  deterministic tokens (``<region#3>``) and offending values are described by their
  *shape signature* (``"12,99"`` → ``"99,99"``), which reveals format problems without content.
* ``masked_samples`` (explicit opt-in) — values are included after masking emails,
  phone numbers and long digit runs, truncation, and redaction of instruction-like text.

Tokens are rehydrated to real values only for local display (:meth:`PrivacyGuard.rehydrate`).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

PrivacyMode = Literal["metadata_only", "masked_samples"]
PRIVACY_MODES = ("metadata_only", "masked_samples")

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_LONG_DIGITS = re.compile(r"\d{9,}")
_TOKEN = re.compile(r"<([^<>#\n]{1,80})#(\d{1,6})>")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# Heuristic detector for prompt-injection payloads hidden in data values.
_INSTRUCTION_LIKE = re.compile(
    r"(ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above)|system\s*prompt|you\s+are\s+now|"
    r"disregard\s+|new\s+instructions?|act\s+as\s+|<\s*/?\s*(system|assistant|user)\s*>|\bapprove\b.*\bfix)",
    re.IGNORECASE,
)

MAX_VALUE_LENGTH = 64
MAX_IDENTIFIER_LENGTH = 64


def shape_signature(value: Any, max_length: int = 24) -> str:
    """Describe a value's format without its content: letters → a/A, digits → 9."""
    text = _CONTROL.sub(" ", str(value))
    shape = "".join("9" if ch.isdigit() else ("A" if ch.isupper() else "a") if ch.isalpha() else ch for ch in text)
    if len(shape) > max_length:
        return f"{shape[:max_length]}…(len={len(text)})"
    return shape


def safe_identifier(name: Any) -> str:
    """Sanitise a column name (which comes from the file and is untrusted) for prompts."""
    text = _CONTROL.sub(" ", str(name)).strip()
    if _INSTRUCTION_LIKE.search(text):
        return "<redacted-column-name>"
    return text[:MAX_IDENTIFIER_LENGTH]


def mask_text(value: Any, max_length: int = MAX_VALUE_LENGTH) -> str:
    """Mask PII-like patterns, neutralise instruction-like text and truncate."""
    text = _CONTROL.sub(" ", str(value))
    if _INSTRUCTION_LIKE.search(text):
        return "<redacted:instruction-like-text>"
    text = _EMAIL.sub("<email>", text)
    text = _LONG_DIGITS.sub("<number>", text)
    text = _PHONE.sub("<phone>", text)
    return text if len(text) <= max_length else text[:max_length] + "…"


class PrivacyGuard:
    """Apply the run's privacy mode to dataset-derived values.

    Args:
        mode: ``metadata_only`` or ``masked_samples``.
        raw: The raw DataFrame, used to build deterministic category tokens.
    """

    def __init__(self, mode: str, raw: pd.DataFrame) -> None:
        if mode not in PRIVACY_MODES:
            raise ValueError(f"Unknown privacy mode '{mode}'. Expected one of {PRIVACY_MODES}.")
        self.mode: PrivacyMode = mode  # type: ignore[assignment]
        self._raw = raw
        self._forward: Dict[str, Dict[str, int]] = {}
        self._reverse: Dict[str, Dict[int, str]] = {}

    @property
    def allows_values(self) -> bool:
        """True when masked sample values may be shared."""
        return self.mode == "masked_samples"

    def _tokens_for(self, column: str) -> Dict[str, int]:
        if column not in self._forward:
            values = sorted({str(v) for v in self._raw[column].dropna().tolist()}) if column in self._raw else []
            self._forward[column] = {v: i + 1 for i, v in enumerate(values)}
            self._reverse[column] = {i + 1: v for i, v in enumerate(values)}
        return self._forward[column]

    def label(self, column: str, value: Any) -> str:
        """A category label safe to share: a token, or a masked value in opt-in mode."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "<missing>"
        if self.allows_values:
            return mask_text(value)
        index = self._tokens_for(column).get(str(value))
        return f"<{safe_identifier(column)}#{index}>" if index else "<unknown>"

    def describe_value(self, value: Any) -> Dict[str, Any]:
        """Describe a single (possibly offending) cell value according to the mode."""
        described: Dict[str, Any] = {"shape": shape_signature(value), "python_type": type(value).__name__}
        if self.allows_values:
            described["value"] = mask_text(value)
        return described

    def describe_values(self, values: List[Any]) -> List[Dict[str, Any]]:
        """Describe several values."""
        return [self.describe_value(v) for v in values]

    def rehydrate(self, text: Optional[str]) -> Optional[str]:
        """Replace tokens such as ``<region#3>`` with real values — for local display only."""
        if not text:
            return text

        def _swap(match: re.Match) -> str:
            column, index = match.group(1), int(match.group(2))
            if column not in self._raw.columns:
                return match.group(0)
            self._tokens_for(column)
            return self._reverse.get(column, {}).get(index, match.group(0))

        return _TOKEN.sub(_swap, text)
