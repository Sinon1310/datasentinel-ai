"""Write-path redaction: everything free-form is sanitised BEFORE it can be persisted.

This is the only production code that seals values as :class:`~datasentinel.observability.models.Sanitized`,
which is the only form the store accepts for free-form content.

What it does:

* **Numbers** in mappings: kept under measurement keys (rows, counts, bytes, durations, timestamps, …);
  replaced with ``"<number>"`` under identifier keys (account, card, phone, customer, …) and, under any
  other key, when they have 9 or more digits — so IDs, card and phone numbers passed as integers are
  redacted like their text forms.
* **Mappings** (parameters, metadata, failure context): recursive copy, at most 6 levels deep,
  100 items per dict/list, strings of at most 1,024 characters, and at most 64 KB once encoded. Values
  under sensitive key names (``password``, ``token``, ``api_key``, ``dsn``, ``cookie``, …) become
  ``"[REDACTED]"``. Non-JSON objects become ``"<TypeName>"`` (dates, enums, decimals, paths and UUIDs
  keep a string form, which is still pattern-redacted).
* **Text** anywhere: credentials in URLs, ``Bearer``/``Basic`` values, ``key=value`` secrets, known key
  formats (``gsk_``, ``sk-``, ``AKIA``/``ASIA``, ``ghp_``, Slack tokens, JWTs, PEM blocks), e-mails,
  phone numbers and digit runs of 9+ are replaced. Unlike the prompt-oriented ``privacy.mask_text``, it
  does not truncate to 64 characters or rewrite instruction-like text.
* **Exceptions**: qualified type name, redacted message (at most 2,048 characters), up to 5 cause/context
  links (type and redacted message), and the innermost 20 stack frames as ``module``/``function``/
  ``lineno`` only. **Source lines are never read and local variables are never accessed.**

Redaction is pattern-based: it removes known secret shapes and everything under sensitive keys, but an
unusual secret format passed as an ordinary value could still slip through (see the Phase 2 report).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import decimal
import enum
import json
import math
import re
import uuid
from pathlib import PurePath
from types import TracebackType
from typing import Any, List, Mapping, Optional

from datasentinel.observability.errors import ObservabilityValidationError
from datasentinel.observability.models import Sanitized, _seal_sanitized

REDACTED = "[REDACTED]"
MAX_DEPTH = 6
MAX_ITEMS = 100
MAX_STRING_CHARS = 1_024
MAX_KEY_CHARS = 128
MAX_PAYLOAD_CHARS = 65_536
MAX_MESSAGE_CHARS = 2_048
MAX_ACTOR_CHARS = 128
MAX_REASON_CHARS = 1_024
MAX_CHAIN_LINKS = 5
MAX_STACK_FRAMES = 20
TRUNCATION_MARK = "…[truncated]"

_SENSITIVE_KEY = re.compile(
    r"pass(word|wd)?|pwd|secret|token|api[_-]?key|apikey|auth|credential|private[_-]?key|\bdsn\b|^dsn$|"
    r"conn(ection)?[_-]?str(ing)?|access[_-]?key|session|cookie|signature|salt", re.IGNORECASE)

_PEM = re.compile(r"-----BEGIN [A-Z0-9 ]+-----[\s\S]*?(-----END [A-Z0-9 ]+-----|$)")
_URL_CREDENTIALS = re.compile(r"\b([A-Za-z][A-Za-z0-9+.\-]*://)[^\s/@:]+(?::[^\s/@]*)?@")
_AUTH_HEADER = re.compile(r"\b(Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=\-]{6,}", re.IGNORECASE)
_KEY_VALUE = re.compile(
    r"\b(password|passwd|pwd|secret|client[_-]?secret|token|access[_-]?token|refresh[_-]?token|api[_-]?key|"
    r"apikey|access[_-]?key|private[_-]?key|auth)\b(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;&]+)", re.IGNORECASE)
_KNOWN_TOKENS = re.compile(
    r"\b(gsk_[A-Za-z0-9]{10,}|sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{16,}|(?:AKIA|ASIA)[0-9A-Z]{16}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9\-]{10,}|"
    r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_LONG_DIGITS = re.compile(r"\d{9,}")
_PHONE = re.compile(r"\+?\d[\d\s().\-]{7,}\d")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_ALL_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.<>]")


_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}(?::\d{2}(?::\d{2}(?:\.\d+)?)?)?)?\s?")


def _phone_or_timestamp(match: "re.Match[str]") -> str:
    """ISO dates/times look like phone numbers to the pattern; keep them (they are not personal data)."""
    return match.group(0) if _TIMESTAMP.fullmatch(match.group(0)) else "<phone>"


# Numeric values are judged by the key they sit under (list items inherit their parent's key).
_MEASUREMENT_TOKENS = frozenset({
    "row", "rows", "count", "counts", "total", "totals", "num", "n", "size", "sizes", "bytes", "kb", "mb", "gb",
    "ms", "us", "ns", "millis", "sec", "secs", "seconds", "min", "mins", "minutes", "hours", "duration",
    "elapsed", "latency", "timeout", "attempt", "attempts", "retry", "retries", "version", "port", "year",
    "month", "day", "limit", "offset", "batch", "threshold", "score", "pct", "percent", "ratio", "rate",
    "records", "partitions", "files", "chunks", "max", "mean", "avg", "median", "p50", "p95", "p99", "epoch",
    "timestamp", "ts", "at", "time", "date", "width", "height", "depth", "level", "workers", "threads",
    "memory", "cpu", "lines", "errors", "warnings", "items", "length", "len", "capacity", "quota", "step",
    "steps", "index", "page", "shard", "shards", "replicas", "link"})
_IDENTIFIER_TOKENS = frozenset({
    "account", "acct", "card", "cc", "pan", "ssn", "sin", "nin", "social", "phone", "mobile", "msisdn", "tel",
    "telephone", "fax", "iban", "bic", "swift", "routing", "customer", "client", "user", "member", "patient",
    "employee", "passport", "license", "licence", "tax", "tin", "vat", "national", "contact", "subscriber"})
_KEY_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_DIGIT_STRING = re.compile(r"\+?[\d][\d\s\-().]{3,}")
NUMBER_PLACEHOLDER = "<number>"
LONG_NUMBER_THRESHOLD = 100_000_000  # 9+ digits, the same threshold as the text rule


def _key_tokens(key: Optional[str]) -> frozenset:
    if not key:
        return frozenset()
    tokens = [t for t in _KEY_TOKEN_SPLIT.split(_CAMEL_BOUNDARY.sub("_", str(key)).lower()) if t]
    # Plurals match their singular ("accounts", "phones", "customers").
    return frozenset(tokens + [t[:-1] for t in tokens if len(t) > 3 and t.endswith("s")])


def numeric_is_sensitive(value: float, key: Optional[str]) -> bool:
    """Whether a number under ``key`` may be a personal or account identifier.

    Measurements (rows, counts, bytes, durations, timestamps, versions, ...) are kept. Numbers under
    identifier-like keys (account, card, phone, customer, ...) are redacted at any size. Any other number
    with 9 or more digits is redacted, like digit runs in text.
    """
    tokens = _key_tokens(key)
    if tokens & _MEASUREMENT_TOKENS:
        return False
    if tokens & _IDENTIFIER_TOKENS:
        return True
    return abs(value) >= LONG_NUMBER_THRESHOLD


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - len(TRUNCATION_MARK)] + TRUNCATION_MARK


def _qualified_name(cls: type) -> str:
    """``module.QualName`` restricted to identifier characters (fits the store's error_type rule)."""
    name = f"{getattr(cls, '__module__', '') or 'unknown'}.{getattr(cls, '__qualname__', cls.__name__)}"
    name = re.sub(r"[^A-Za-z0-9_.]", "_", name).strip(".") or "unknown"
    if not re.match(r"[A-Za-z_]", name):
        name = "_" + name
    return name[:200]


def is_sensitive_key(key: str) -> bool:
    """Whether a mapping key names a credential-like value."""
    return bool(_SENSITIVE_KEY.search(str(key)))


class Redactor:
    """Deterministic, side-effect-free redaction for every observability write path."""

    def redact_text(self, value: Any, *, max_chars: int = MAX_STRING_CHARS, keep_newlines: bool = True) -> str:
        """Pattern-redact one string, remove control characters and cap its length."""
        text = str(value)
        text = (_CONTROL if keep_newlines else _ALL_CONTROL).sub(" ", text)
        text = _PEM.sub("[REDACTED:private-key]", text)
        text = _URL_CREDENTIALS.sub(lambda m: f"{m.group(1)}{REDACTED}@", text)
        text = _AUTH_HEADER.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
        text = _KEY_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
        text = _KNOWN_TOKENS.sub("[REDACTED:token]", text)
        text = _EMAIL.sub("<email>", text)
        text = _LONG_DIGITS.sub("<number>", text)
        text = _PHONE.sub(_phone_or_timestamp, text)
        return _truncate(text, max_chars)

    # ------------------------------------------------------------------------------------ structures

    def _value(self, value: Any, depth: int, key: Optional[str] = None) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return NUMBER_PLACEHOLDER if numeric_is_sensitive(value, key) else value
        if isinstance(value, float):
            if not math.isfinite(value):
                return f"<{value}>"
            return NUMBER_PLACEHOLDER if value.is_integer() and numeric_is_sensitive(value, key) else value
        if isinstance(value, str):
            if (_key_tokens(key) & _IDENTIFIER_TOKENS and not _key_tokens(key) & _MEASUREMENT_TOKENS
                    and _DIGIT_STRING.fullmatch(value.strip())):
                return NUMBER_PLACEHOLDER
            return self.redact_text(value)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"<bytes len={len(value)}>"
        if isinstance(value, enum.Enum):
            return self._value(value.value, depth, key)
        if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
            return value.isoformat()
        if isinstance(value, decimal.Decimal):
            if value.is_finite() and value == value.to_integral_value() and numeric_is_sensitive(value, key):
                return NUMBER_PLACEHOLDER
            return self.redact_text(str(value))
        if isinstance(value, (uuid.UUID, PurePath)):
            return self.redact_text(str(value))
        if depth >= MAX_DEPTH:
            return "<max-depth>"
        if isinstance(value, Mapping):
            return self._mapping(value, depth + 1)
        if isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
            if isinstance(value, (set, frozenset)):
                items = sorted(items, key=repr)
            out = [self._value(v, depth + 1, key) for v in items[:MAX_ITEMS]]  # items inherit the key
            if len(items) > MAX_ITEMS:
                out.append(f"<{len(items) - MAX_ITEMS} more items>")
            return out
        return f"<{type(value).__name__}>"

    def _mapping(self, value: Mapping, depth: int) -> dict:
        out: dict = {}
        items = list(value.items())
        for key, item in items[:MAX_ITEMS]:
            safe_key = self.redact_text(key, max_chars=MAX_KEY_CHARS, keep_newlines=False)
            while safe_key in out:  # distinct keys can collide after redaction/truncation
                safe_key += "_"
            out[safe_key] = REDACTED if is_sensitive_key(str(key)) else self._value(item, depth, str(key))
        if len(items) > MAX_ITEMS:
            out["<truncated>"] = f"{len(items) - MAX_ITEMS} more items"
        return out

    def sanitize_mapping(self, value: Optional[Mapping[str, Any]]) -> Sanitized[dict]:
        """Sanitise parameters, metadata or failure context (``None`` becomes ``{}``)."""
        if value is None:
            return _seal_sanitized({})
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        if not isinstance(value, Mapping):
            raise ObservabilityValidationError("parameters, metadata and context must be mappings")
        cleaned = self._mapping(value, 1)
        if len(json.dumps(cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) > MAX_PAYLOAD_CHARS:
            cleaned = {"<truncated>": "payload exceeded 64 KB after redaction",
                       "keys": [k for k in list(cleaned)[:50]]}
        return _seal_sanitized(cleaned)

    # ------------------------------------------------------------------------------------- scalars

    def sanitize_text(self, value: Any, *, max_chars: int = MAX_REASON_CHARS) -> Sanitized[str]:
        """Sanitise a free-text value such as a retry or cancellation reason."""
        return _seal_sanitized(self.redact_text(value, max_chars=max_chars).strip())

    def sanitize_actor(self, value: Any) -> Sanitized[str]:
        """Sanitise an actor name: single line, redacted, at most 128 characters, never empty."""
        text = " ".join(self.redact_text(value, max_chars=MAX_ACTOR_CHARS * 2, keep_newlines=False).split())
        return _seal_sanitized(_truncate(text, MAX_ACTOR_CHARS) or "unknown")

    # ---------------------------------------------------------------------------------- exceptions

    def exception_message(self, exc: BaseException) -> str:
        """Redacted, length-capped message of one exception (never raises)."""
        try:
            text = str(exc)
        except Exception:  # noqa: BLE001 - a broken __str__ must not break failure recording
            text = "<unprintable exception message>"
        return self.redact_text(text, max_chars=MAX_MESSAGE_CHARS)

    def capture_exception(self, exc: BaseException) -> "CapturedException":
        """Structured, redacted capture of an exception, its cause chain and its stack (no source, no locals)."""
        chain: List[dict] = []
        seen = {id(exc)}
        link: Optional[BaseException] = exc
        while link is not None and len(chain) < MAX_CHAIN_LINKS:
            if link.__cause__ is not None:
                nxt, relation = link.__cause__, "cause"
            elif link.__context__ is not None and not link.__suppress_context__:
                nxt, relation = link.__context__, "context"
            else:
                break
            if id(nxt) in seen:
                break
            seen.add(id(nxt))
            chain.append({"type": _qualified_name(type(nxt)), "relation": relation,
                          "message": self.redact_text(self.exception_message(nxt), max_chars=MAX_STRING_CHARS)})
            link = nxt
        return CapturedException(
            error_type=_qualified_name(type(exc)),
            message=_seal_sanitized(self.exception_message(exc)),
            exception_chain=_seal_sanitized(chain),
            stack=_seal_sanitized(stack_frames(exc.__traceback__)),
        )


def stack_frames(tb: Optional[TracebackType], limit: int = MAX_STACK_FRAMES) -> List[dict]:
    """Innermost ``limit`` frames as ``{module, function, lineno}``.

    Reads only the code object name, the module's ``__name__`` and the line number. It never calls
    ``linecache``/``traceback`` (no source lines) and never touches ``f_locals``.
    """
    frames: List[dict] = []
    while tb is not None:
        frame = tb.tb_frame
        module = frame.f_globals.get("__name__", "<unknown>")
        function = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        frames.append({"module": _SAFE_NAME.sub("_", str(module))[:200],
                       "function": _SAFE_NAME.sub("_", str(function))[:200],
                       "lineno": int(tb.tb_lineno or 0)})
        tb = tb.tb_next
    return frames[-limit:]


@dataclasses.dataclass(frozen=True)
class CapturedException:
    """A redacted, source-free description of an exception, ready for the store."""

    error_type: str
    message: Sanitized[str]
    exception_chain: Sanitized[list]
    stack: Sanitized[list]
