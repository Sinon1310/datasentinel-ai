"""Centralised logging configuration.

Modules obtain loggers with ``logging.getLogger(__name__)``; this module attaches a
single handler to the ``datasentinel`` logger namespace so we never clobber the root
logger owned by host frameworks such as Streamlit or Uvicorn.
"""

from __future__ import annotations

import logging
from typing import Optional

from datasentinel.config import get_settings

_LOGGER_NAMESPACE = "datasentinel"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure the ``datasentinel`` logger. Safe to call repeatedly.

    Args:
        level: Log level name (e.g. ``"DEBUG"``). Defaults to ``Settings.log_level``.

    Returns:
        The configured package-level logger.
    """
    logger = logging.getLogger(_LOGGER_NAMESPACE)
    resolved = (level or get_settings().log_level).upper()
    logger.setLevel(getattr(logging, resolved, logging.INFO))

    if not any(getattr(h, "_datasentinel", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._datasentinel = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False
    return logger
