"""Append-only JSON Lines audit trail for approvals and executions."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class AuditLog:
    """Append-only audit log. Entries are never modified or deleted by the application."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, event: str, **fields: Any) -> Dict[str, Any]:
        """Append an event with a UTC timestamp and return the written entry."""
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **fields}
        line = json.dumps(entry, default=str, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return entry

    def read(self) -> List[Dict[str, Any]]:
        """All entries in order."""
        if not self.path.is_file():
            return []
        with open(self.path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
