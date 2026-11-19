"""
Agentic Monitor & Auto-Labeling Utilities for SerenityPC / Training Station.
Provides standardized 001_YYYY-MM-DD run/bank labeling and background health monitoring
for active models, error logs, and custom Wringer test banks.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


def format_auto_label(index: int, dt: Optional[datetime] = None, suffix: str = "") -> str:
    """Formats a standardized sequential label like `001_2026-11-16` (optionally with slug suffix)."""
    target_dt = dt or datetime.now()
    base = f"{max(1, int(index)):03d}_{target_dt.strftime('%Y-%m-%d')}"
    clean_suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(suffix or "").strip()).strip("_")
    return f"{base}_{clean_suffix}" if clean_suffix else base


def next_auto_label_for_dir(
    directory: str,
    suffix: str = "",
    dt: Optional[datetime] = None,
) -> str:
    """Scans `directory` for `NNN_YYYY-MM-DD*` items and returns the next 3-digit sequential label."""
    max_idx = 0
    if directory and os.path.isdir(directory):
        try:
            for name in os.listdir(directory):
                m = re.match(r"^(\d{3})_\d{4}-\d{2}-\d{2}", name)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        except OSError:
            pass
    return format_auto_label(max_idx + 1, dt=dt, suffix=suffix)


def is_text_readable(text: str, min_ratio: float = 0.85) -> bool:
    """Validates text is clean readable UTF-8 without mojibake, NULs, or binary garbage."""
    if not text or not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < 4:
        return False
    if "\ufffd" in stripped or "\x00" in stripped:
        return False
    readable = sum(
        1 for ch in stripped
        if (32 <= ord(ch) <= 126) or ch in "\n\r\t" or (0x00C0 <= ord(ch) <= 0x024F)
    )
    return (readable / max(1, len(stripped))) >= min_ratio


class AgenticMonitor:
    """Monitors runtime model health, recent tracebacks, and custom Wringer test bank integrity."""

    def __init__(
        self,
        app: Any = None,
        workspace_dir: Optional[str] = None,
        custom_banks_dir: Optional[str] = None,
        poll_interval_sec: float = 30.0,
    ):
        self.app = app
        self.workspace_dir = workspace_dir or getattr(app, "script_dir", None) or os.getcwd()
        self.custom_banks_dir = custom_banks_dir or os.path.join(
            self.workspace_dir, "benchmarks", "wringer", "custom_test_banks"
        )
        self.poll_interval_sec = max(2.0, float(poll_interval_sec))
        self.running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_report: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []

    def start(self) -> bool:
        """Starts the periodic background monitor thread."""
        if self.running and self._thread and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> bool:
        """Signals the background monitor thread to stop."""
        was_running = self.running
        self.running = False
        self._stop_event.set()
        return was_running

    def _run_loop(self) -> None:
        while self.running and not self._stop_event.is_set():
            try:
                self.scan_once()
            except Exception:
                pass
            self._stop_event.wait(self.poll_interval_sec)

    def scan_once(self) -> Dict[str, Any]:
        """Performs a synchronous health & asset scan and records the report."""
        model_loaded = bool(getattr(self.app, "model", None)) if self.app is not None else False
        active_level = getattr(self.app, "active_persona_level", None) if self.app is not None else None
        current_tier = getattr(self.app, "current_model_tier", None) if self.app is not None else None
        is_generating = bool((getattr(self.app, "state", {}) or {}).get("running", False)) if self.app is not None else False

        err_count = 0
        err_log = getattr(self.app, "error_log_file", None) if self.app is not None else None
        if not err_log:
            err_log = os.path.join(self.workspace_dir, "System", "error_log.txt")
        if err_log and os.path.isfile(err_log):
            try:
                with open(err_log, "r", encoding="utf-8", errors="replace") as f:
                    tail = f.read()[-16000:]
                err_count = tail.count("Traceback (most recent call last):")
            except OSError:
                pass

        valid_banks: List[str] = []
        corrupt_banks: List[str] = []
        if self.custom_banks_dir and os.path.isdir(self.custom_banks_dir):
            for fn in sorted(os.listdir(self.custom_banks_dir)):
                if not fn.endswith(".json"):
                    continue
                fp = os.path.join(self.custom_banks_dir, fn)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    items: List[Any] = []
                    if isinstance(data, dict):
                        if isinstance(data.get("questions"), list):
                            items.extend(data["questions"])
                        else:
                            for k, v in data.items():
                                if re.match(r"^lvl\d+$", str(k)) and isinstance(v, list):
                                    items.extend(v)
                    elif isinstance(data, list):
                        items = data
                    if not items:
                        corrupt_banks.append(fn)
                        continue
                    bad_q = any(
                        not is_text_readable(str(q.get("prompt") or q.get("question") or ""))
                        for q in items[:40]
                        if isinstance(q, dict)
                    )
                    if bad_q:
                        corrupt_banks.append(fn)
                    else:
                        valid_banks.append(fn)
                except Exception:
                    corrupt_banks.append(fn)

        health = "OK" if not corrupt_banks and err_count == 0 else "WARNING"
        report = {
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "model_loaded": model_loaded,
            "active_level": active_level,
            "current_tier": current_tier,
            "generating": is_generating,
            "recent_tracebacks": err_count,
            "valid_banks": valid_banks,
            "corrupt_banks": corrupt_banks,
        }
        report["summary"] = (
            f"[MONITOR {health}] Model: {'Loaded' if model_loaded else 'Idle'} "
            f"(Lvl {active_level if active_level is not None else '?'}, tier={current_tier or 'none'}) | "
            f"Valid Banks: {len(valid_banks)}, Corrupt Banks: {len(corrupt_banks)} | "
            f"Recent Tracebacks: {err_count}"
        )
        self.last_report = report
        self.history.append(report)
        if len(self.history) > 50:
            self.history = self.history[-50:]
        return report

    def status(self) -> Dict[str, Any]:
        """Returns current monitor daemon state and latest scan summary."""
        return {
            "running": self.running,
            "checks_count": len(self.history),
            "last_report": self.last_report,
        }
