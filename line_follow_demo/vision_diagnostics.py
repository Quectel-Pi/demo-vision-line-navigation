"""Bounded, asynchronous raw-frame evidence for the existing vision algorithm."""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import queue
import threading
from typing import Any
from uuid import uuid4

import cv2


@dataclass(frozen=True)
class VisionDiagnosticConfig:
    enabled: bool = False
    sample_hz: float = 5.0
    pre_frames: int = 5
    post_s: float = 1.0
    motion_rate_dps: float = 2.0
    max_frames: int = 300
    directory: str = "../logs/vision_diagnostics"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("vision_diagnostics.enabled must be a boolean")
        if not all(math.isfinite(value) for value in (self.sample_hz, self.post_s, self.motion_rate_dps)) or not (
            0 < self.sample_hz <= 10 and 0 <= self.post_s <= 5 and self.motion_rate_dps > 0
        ):
            raise ValueError("invalid vision diagnostic capture timing")
        for value, maximum in ((self.pre_frames, 20), (self.max_frames, 3000)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError("invalid diagnostic frame/memory limit")
        if not self.directory:
            raise ValueError("diagnostic directory must not be empty")


class VisionDiagnosticRecorder:
    """Keep pre-event frames; save motion/invalid-path frames and their metadata.

    PNG is lossless and has no debug overlays. Only copied frames enter the
    bounded queue. Encoding, source snapshots and disk writes run off-loop.
    Capture faults are exposed separately and do not change path decisions.
    """

    def __init__(self, config: VisionDiagnosticConfig, *, config_directory: Path, vision_source: Path):
        self.config = config
        self._config_directory = config_directory
        self._vision_source = vision_source
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=16)
        self._before: deque = deque(maxlen=config.pre_frames)
        self._directory: Path | None = None
        self._last_sample_s = -math.inf
        self._save_until_s = -math.inf
        self._sequence = 0
        self._last_queued = 0
        self._accepted = self._saved = self._dropped = 0
        self._error = ""
        self._state = "STOPPED" if config.enabled else "DISABLED"

    def start(self) -> None:
        if not self.config.enabled or (self._worker is not None and self._worker.is_alive()):
            return
        self._stop.clear()
        self._before.clear()
        self._queue = queue.Queue(maxsize=16)
        self._last_sample_s = self._save_until_s = -math.inf
        self._sequence = self._last_queued = self._accepted = self._saved = self._dropped = 0
        self._error = ""
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        self._directory = (self._config_directory / self.config.directory / suffix).resolve()
        self._state = "READY"
        self._worker = threading.Thread(target=self._write, name="vision-evidence", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=2.0)
        self._before.clear()
        with self._lock:
            if self._state != "ERROR":
                self._state = "STOPPED" if self.config.enabled else "DISABLED"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "vision_capture_state": self._state,
                "vision_capture_saved": self._saved,
                "vision_capture_accepted": self._accepted,
                "vision_capture_dropped": self._dropped,
                "vision_capture_error": self._error,
                "vision_capture_directory": None if self._directory is None else str(self._directory),
            }

    def offer(self, frame, *, status: dict, imu: dict, features, vision_config: dict) -> None:
        if not self.config.enabled or self._stop.is_set():
            return
        timestamp = status["vision_timestamp_monotonic_s"]
        if timestamp - self._last_sample_s < 1.0 / self.config.sample_hz:
            return
        with self._lock:
            if self._accepted >= self.config.max_frames or self._state == "ERROR":
                return
        self._last_sample_s = timestamp
        self._sequence += 1
        rate = imu.get("imu_yaw_rate_dps")
        moving = rate is not None and abs(rate) >= self.config.motion_rate_dps
        # Idle preview can remain LOST for minutes while the car is positioned.
        # Preserve the bounded disk budget for an armed run or actual rotation;
        # the deque still keeps recent preview frames before either event.
        if status.get("armed") or moving:
            self._save_until_s = timestamp + self.config.post_s
        metadata = {
            "sequence": self._sequence,
            "enqueued_at_utc": datetime.now(timezone.utc).isoformat(),
            "vision_timestamp_monotonic_s": timestamp,
            "features": asdict(features),
            "vision_config": copy.deepcopy(vision_config),
            "status": copy.deepcopy(status), "imu": copy.deepcopy(imu),
        }
        self._before.append((self._sequence, frame.copy(), metadata))
        if timestamp > self._save_until_s:
            return
        for item in self._before:
            sequence, _, _ = item
            if sequence <= self._last_queued:
                continue
            with self._lock:
                if self._accepted >= self.config.max_frames:
                    self._state = "LIMIT_REACHED"
                    return
            self._last_queued = sequence
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                with self._lock:
                    self._dropped += 1
            else:
                with self._lock:
                    self._accepted += 1
                    if self._accepted >= self.config.max_frames:
                        self._state = "LIMIT_REACHED"

    def _write(self) -> None:
        try:
            assert self._directory is not None
            self._directory.mkdir(parents=True, exist_ok=False)
            source = self._vision_source.read_bytes()
            (self._directory / "vision_source.py").write_bytes(source)
            (self._directory / "manifest.json").write_text(json.dumps({
                "vision_source_sha256": hashlib.sha256(source).hexdigest(),
                "capture_config": asdict(self.config),
                "image_format": "lossless PNG, raw camera frame, no debug overlays",
                "opencv_version": cv2.__version__,
                "timestamps": "host monotonic clock, not exposure timestamps",
            }, indent=2), encoding="utf-8")
            with (self._directory / "frames.jsonl").open("x", encoding="utf-8") as handle:
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        sequence, frame, metadata = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    name = f"frame_{sequence:08d}.png"
                    ok, data = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    if not ok:
                        raise OSError("raw frame PNG encoding failed")
                    (self._directory / name).write_bytes(data.tobytes())
                    metadata["image"] = name
                    handle.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    with self._lock:
                        self._saved += 1
        except Exception as exc:
            with self._lock:
                self._state, self._error = "ERROR", str(exc)
            logging.getLogger(__name__).error("Vision diagnostic capture failed: %s", exc)
            self._stop.set()
