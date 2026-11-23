"""Write-path redaction: secrets, PII, sensitive keys, size and depth caps, and source-free exception capture."""

from __future__ import annotations

import datetime as dt
import enum
import json
from pathlib import Path

import pytest

from datasentinel.observability import ObservabilityValidationError, Sanitized
from datasentinel.observability.redaction import (
    MAX_CHAIN_LINKS, MAX_ITEMS, MAX_MESSAGE_CHARS, MAX_PAYLOAD_CHARS, MAX_STACK_FRAMES, REDACTED, Redactor,
    is_sensitive_key, stack_frames,
)

R = Redactor()

SECRETS = {
    "groq key": "gsk_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
    "openai key": "sk-proj-" + "abcdefghijklmnopqrstuvwx",
    "aws key": "AKIA" + "ABCDEFGHIJKLMNOP",
    "github token": "ghp_" + "abcdefghijklmnopqrstuvwxyz0123",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
}


@pytest.mark.parametrize("name,secret", list(SECRETS.items()))
def test_known_secret_formats_are_removed(name: str, secret: str) -> None:
    assert secret not in R.redact_text(f"failed with {secret} in request")


@pytest.mark.parametrize("text,leak", [
    ("postgres://admin:hunter2@db.internal:5432/sales", "hunter2"),
    ("https://user@host/x", "user@"),
    ("Authorization: Bearer abc.def-ghi_jkl", "abc.def"),
    ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ("password=hunter2&x=1", "hunter2"),
    ('api_key: "two words"', "two words"),
    ("client_secret=s3cr3t;", "s3cr3t"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----", "MIIEow"),
    ("contact jane.doe@example.com", "jane.doe@example.com"),
    ("call +44 20 7946 0958", "7946"),
    ("account 123456789012", "123456789012"),
])
def test_credentials_and_pii_are_redacted(text: str, leak: str) -> None:
    assert leak not in R.redact_text(text)


def test_harmless_text_and_timestamps_survive() -> None:
    text = "Quality gate failed at 2026-11-23 10:00:05 for stage transform (score 58.7, 12 rows)"
    assert R.redact_text(text) == text


def test_control_characters_and_length_are_bounded() -> None:
    assert "\x00" not in R.redact_text("a\x00b\x1bc")
    assert "\n" in R.redact_text("line1\nline2")
    long = R.redact_text("x" * 5000, max_chars=100)
    assert len(long) == 100 and long.endswith("…[truncated]")


@pytest.mark.parametrize("key", ["password", "DB_PASSWORD", "apiKey", "api-key", "x_auth_header", "dsn", "secret",
                                 "connection_string", "session_cookie", "private_key", "aws_access_key_id"])
def test_sensitive_keys_are_recognised(key: str) -> None:
    assert is_sensitive_key(key)


@pytest.mark.parametrize("key", ["date", "region", "rows", "batch_size", "table"])
def test_ordinary_keys_are_kept(key: str) -> None:
    assert not is_sensitive_key(key)


def test_mappings_are_sanitised_recursively() -> None:
    class Colour(enum.Enum):
        RED = "red"

    value = R.sanitize_mapping({
        "password": "hunter2", "date": dt.date(2026, 11, 23), "colour": Colour.RED, "path": Path("/tmp/x"),
        "nested": {"token": "abc", "dsn": "postgres://u:p@h/db", "url": "mysql://root:pw@h/db",
                   "notes": ["mail a@b.co", 123456789012]},
        "nan": float("nan"), "blob": b"\x00\x01", "obj": object(), "set": {3, 1, 2},
    })
    assert isinstance(value, Sanitized)
    v = value.value
    assert v["password"] == REDACTED and v["nested"]["token"] == REDACTED and v["nested"]["dsn"] == REDACTED
    assert v["nested"]["url"] == f"mysql://{REDACTED}@h/db"
    assert v["nested"]["notes"] == ["mail <email>", "<number>"]  # 9+ digit integers under neutral keys
    assert (v["date"], v["colour"], v["path"]) == ("2026-11-23", "red", "/tmp/x")
    assert (v["nan"], v["blob"], v["obj"], v["set"]) == ("<nan>", "<bytes len=2>", "<object>", [1, 2, 3])


def test_depth_items_and_total_size_are_capped() -> None:
    deep: dict = {}
    cursor = deep
    for _ in range(10):
        cursor["d"] = {}
        cursor = cursor["d"]
    flattened = json.dumps(R.sanitize_mapping({"deep": deep}).value)
    assert "<max-depth>" in flattened
    wide = R.sanitize_mapping({f"k{i}": i for i in range(150)}).value
    assert len(wide) == MAX_ITEMS + 1 and wide["<truncated>"] == "50 more items"
    big = R.sanitize_mapping({f"k{i}": "x" * 1000 for i in range(90)}).value
    assert "<truncated>" in big and len(json.dumps(big)) < MAX_PAYLOAD_CHARS
    listy = R.sanitize_mapping({"l": list(range(500))}).value["l"]
    assert len(listy) == MAX_ITEMS + 1 and listy[-1] == "<400 more items>"


def test_mapping_keys_are_redacted_and_never_collide() -> None:
    value = R.sanitize_mapping({"a@b.co": 1, "c@d.co": 2}).value
    assert set(value) == {"<email>", "<email>_"}


def test_non_mappings_are_rejected() -> None:
    with pytest.raises(ObservabilityValidationError):
        R.sanitize_mapping(["not", "a", "mapping"])  # type: ignore[arg-type]
    assert R.sanitize_mapping(None).value == {}


def test_actor_and_text_sanitisation() -> None:
    assert R.sanitize_actor("  jane.doe@example.com \n ops ").value == "<email> ops"
    assert R.sanitize_actor("").value == "unknown"
    assert len(R.sanitize_actor("x" * 500).value) == 128
    assert R.sanitize_text("retry: token=abc123").value == f"retry: token={REDACTED}"


# ------------------------------------------------------------------------------------------ exceptions

def _raise_with_secret_local() -> None:
    api_key = SECRETS["groq key"]  # noqa: F841 - a local that must never be captured
    raise ValueError(f"upstream rejected key {SECRETS['groq key']} for jane.doe@example.com")


def test_exception_capture_has_no_source_lines_locals_or_secrets() -> None:
    try:
        _raise_with_secret_local()
    except ValueError as exc:
        captured = R.capture_exception(exc)
    dumped = json.dumps([captured.error_type, captured.message.value, captured.exception_chain.value,
                         captured.stack.value])
    assert captured.error_type == "builtins.ValueError"
    assert SECRETS["groq key"] not in dumped and "jane.doe" not in dumped
    assert "api_key =" not in dumped and "raise ValueError" not in dumped  # no source text
    frame = captured.stack.value[-1]
    assert set(frame) == {"module", "function", "lineno"} and frame["function"] == "_raise_with_secret_local"
    assert frame["module"] == __name__


def test_exception_chain_is_bounded_and_redacted() -> None:
    def nest(depth: int) -> None:
        if depth == 0:
            raise KeyError("password=hunter2")
        try:
            nest(depth - 1)
        except Exception as exc:
            raise RuntimeError(f"level {depth}") from exc

    try:
        nest(8)
    except RuntimeError as exc:
        captured = R.capture_exception(exc)
    chain = captured.exception_chain.value
    assert len(chain) == MAX_CHAIN_LINKS and all(link["relation"] == "cause" for link in chain)
    try:
        nest(1)
    except RuntimeError as exc:
        short = R.capture_exception(exc).exception_chain.value
    assert short[-1]["type"] == "builtins.KeyError" and "hunter2" not in short[-1]["message"]


def test_implicit_context_and_suppressed_context() -> None:
    try:
        try:
            raise OSError("disk")
        except OSError:
            raise ValueError("while handling")
    except ValueError as exc:
        assert R.capture_exception(exc).exception_chain.value[0]["relation"] == "context"
    try:
        try:
            raise OSError("disk")
        except OSError:
            raise ValueError("clean") from None
    except ValueError as exc:
        assert R.capture_exception(exc).exception_chain.value == []


def test_stack_keeps_only_innermost_frames() -> None:
    def recurse(n: int) -> None:
        if n == 0:
            raise RuntimeError("deep")
        recurse(n - 1)

    try:
        recurse(40)
    except RuntimeError as exc:
        frames = stack_frames(exc.__traceback__)
    assert len(frames) == MAX_STACK_FRAMES and all(f["function"].endswith("recurse") for f in frames)


def test_long_and_unprintable_messages() -> None:
    assert len(R.capture_exception(ValueError("x" * 10_000)).message.value) == MAX_MESSAGE_CHARS

    class Broken(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no")

    assert R.capture_exception(Broken()).message.value == "<unprintable exception message>"


def test_local_classes_get_valid_type_names() -> None:
    class Local(Exception):
        pass

    name = R.capture_exception(Local()).error_type
    assert "<" not in name and name.endswith("Local")
