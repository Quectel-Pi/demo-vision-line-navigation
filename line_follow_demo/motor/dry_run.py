from __future__ import annotations

import time

from .base import MotorDriver


class DryRunDriver(MotorDriver):
    def __init__(self, print_interval_s: float = 0.2):
        self._print_interval_s = float(print_interval_s)
        self._last_print = 0.0

    def open(self) -> None:
        self._last_print = 0.0

    def set_wheel_duty(self, *, fl: float, fr: float, rl: float, rr: float) -> None:
        now = time.monotonic()
        if now - self._last_print < self._print_interval_s:
            return
        self._last_print = now
        print(f"[dry-run] duty fl={fl:+.1f} fr={fr:+.1f} rl={rl:+.1f} rr={rr:+.1f}")
