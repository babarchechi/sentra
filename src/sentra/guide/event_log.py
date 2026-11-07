"""Privacy-minimal event log: one JSON line per alert, plus a session summary. Never images.

Written to outputs/guide_events_<YYYYmmdd_HHMMSS>[_<label>].jsonl (outputs/ is git-ignored).
Each alert line is GuideAlert.to_dict(): time, frame, label, direction, distance validity, confidence,
severity, the spoken text and why it fired. docs/VALIDATION.md explains how to use these files.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sentra.guide.schema import SCHEMA_VERSION, GuideAlert

DEFAULT_OUTPUT_DIR = Path("outputs")


class GuideEventLog:
    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR, label: str | None = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"guide_events_{datetime.now():%Y%m%d_%H%M%S}"
        suffix = f"_{label}" if label else ""
        self.path = output_dir / f"{stem}{suffix}.jsonl"
        n = 2
        while self.path.exists():  # two runs started in the same second must not overwrite each other
            self.path = output_dir / f"{stem}_{n}{suffix}.jsonl"
            n += 1
        self._file = self.path.open("x", encoding="utf-8")
        self.alerts = 0

    def write_alert(self, alert: GuideAlert, *, spoken: bool) -> None:
        """`spoken` is False when the voice was muted (the alert was only shown)."""
        record = alert.to_dict()
        record["wall_time"] = datetime.now().isoformat(timespec="milliseconds")
        record["spoken"] = spoken
        self._write(record)
        self.alerts += 1

    def write_summary(self, stats: dict[str, Any]) -> None:
        self._write({"schema": SCHEMA_VERSION, "type": "session_summary",
                     "wall_time": datetime.now().isoformat(timespec="milliseconds"), **stats})

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
