#!/usr/bin/env python3
"""
Minimal single-line progress utility for large datasets.

- Prints one updatable line with a progress bar
- Every `update_every` updates, shows samples/sec and ETA
- Avoids flooding logs when processing millions of rows
"""

from __future__ import annotations

import sys
import time
from typing import Optional


class SingleLineProgress:
    def __init__(
        self,
        total: int,
        *,
        desc: str = "Progress",
        bar_width: int = 30,
        update_every: int = 200,
        stream = sys.stdout,
        enabled: bool = True,
    ) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.bar_width = max(10, int(bar_width))
        self.update_every = max(1, int(update_every))
        self.stream = stream
        self.enabled = enabled and (self.total > 0)
        self.start_time = time.time()
        self.last_rate: Optional[float] = None
        self._last_print_len = 0

    def _format_eta(self, remaining: int, rate: float) -> str:
        if rate <= 0:
            return "ETA: --"
        seconds = remaining / rate
        if seconds < 60:
            return f"ETA: {seconds:.0f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"ETA: {minutes:.1f}m"
        hours = minutes / 60
        return f"ETA: {hours:.1f}h"

    def _render(self, completed: int, rate: Optional[float]) -> str:
        pct = (completed / self.total) if self.total else 1.0
        filled = int(self.bar_width * pct)
        bar = "#" * filled + "." * (self.bar_width - filled)
        pct_str = f"{pct*100:5.1f}%"
        rate = rate if rate is not None else self.last_rate or 0.0
        self.last_rate = rate
        eta = self._format_eta(self.total - completed, rate)
        return f"{self.desc} [{bar}] {completed}/{self.total} ({pct_str}) | {rate:.1f}/s {eta}"

    def update(self, completed: int) -> None:
        if not self.enabled:
            return

        now = time.time()
        elapsed = max(1e-9, now - self.start_time)
        # Refresh rate every update_every steps
        rate: Optional[float] = None
        if (completed % self.update_every == 0) or (completed >= self.total):
            rate = completed / elapsed

        line = self._render(completed, rate)
        # Pad to clear previous longer line
        pad = max(0, self._last_print_len - len(line))
        self._last_print_len = len(line)
        self.stream.write("\r" + line + (" " * pad))
        self.stream.flush()

    def finish(self) -> None:
        if not self.enabled:
            return
        self.update(self.total)
        self.stream.write("\n")
        self.stream.flush()


