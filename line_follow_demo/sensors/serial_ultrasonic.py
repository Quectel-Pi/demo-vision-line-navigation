from __future__ import annotations

import re
import threading
import time

import serial

from .base import DistanceSensor


_FRAME_RE = re.compile(r"\$(\d+)(?:,\d+)?\$")
_LINE_RE = re.compile(r"(?:Distance\s*:\s*)?(\d+)\s*(?:mm|MM)?")


class SerialUltrasonicSensor(DistanceSensor):
    def __init__(self, device: str, baudrate: int = 9600, timeout_s: float = 0.1):
        self._device = device
        self._baudrate = int(baudrate)
        self._timeout_s = float(timeout_s)
        self._ser: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_mm: int | None = None
        self._last_ts = 0.0
        self._buf = ""

    def open(self) -> None:
        self._ser = serial.Serial(self._device, self._baudrate, timeout=self._timeout_s)
        self._stop.clear()
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()

    def _update(self, mm: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_mm = int(mm)
            self._last_ts = now

    def _parse_text(self, text: str) -> None:
        # 支持两种常见格式：
        # 1) $123,xxxx$（参考 miniAuto app_control.ino）
        # 2) Distance: 123 mm
        for m in _FRAME_RE.finditer(text):
            self._update(int(m.group(1)))

        # 作为兜底：尝试逐行找数字
        for line in text.splitlines():
            m = _LINE_RE.search(line.strip())
            if m:
                self._update(int(m.group(1)))

    def _rx_loop(self) -> None:
        assert self._ser is not None
        while not self._stop.is_set():
            try:
                data = self._ser.read(256)
            except Exception:
                time.sleep(0.05)
                continue

            if not data:
                continue

            try:
                chunk = data.decode("utf-8", errors="ignore")
            except Exception:
                continue

            self._buf += chunk
            if len(self._buf) > 4096:
                self._buf = self._buf[-2048:]

            # 解析一把
            self._parse_text(self._buf)

    def read_distance_mm(self) -> int | None:
        with self._lock:
            return self._last_mm

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None

        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
