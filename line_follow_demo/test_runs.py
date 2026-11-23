"""Per-Start offline evidence, written on the board without a browser poller."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import threading
import time
from uuid import uuid4

import cv2


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TestRunConfig:
    enabled: bool = False
    directory: str = "../logs/test_runs"
    image_hz: float = 5.0
    max_images: int = 3000
    max_run_mb: int = 512
    reserve_free_mb: int = 256

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not self.directory:
            raise ValueError("invalid test recording enable/directory")
        if not math.isfinite(self.image_hz) or not 0 < self.image_hz <= 10:
            raise ValueError("test recording image_hz must be within (0, 10]")
        for key, low, high in (("max_images", 1, 20000), ("max_run_mb", 1, 4096), ("reserve_free_mb", 1, 16384)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"invalid test recording {key}")


class TestRunRecorder:
    """One unique folder per arm; bounded queue and no disk IO in control calls.

    Stops and faults explicitly close a run. Rebooted unfinished runs are marked
    interrupted, never passed. Disk limits never delete earlier evidence.
    """

    def __init__(self, config: TestRunConfig, *, config_directory: Path):
        self.config = config
        self.directory = (config_directory / config.directory).resolve()
        self._source_directory = Path(__file__).parent
        self._lock = threading.RLock()
        self._queue = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._worker = None
        self._active = None
        self._last_id = None
        self._state = "STOPPED" if config.enabled else "DISABLED"
        self._error = ""
        self._saved = self._rows = self._dropped = 0
        self._sensor_rows = self._sensor_dropped = 0
        self._limits = []
        self._last_image_s = -math.inf

    def start(self):
        if not self.config.enabled:
            return
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._queue = queue.Queue(maxsize=256)
        self._active = None
        self._error = ""
        self._worker = threading.Thread(target=self._write, name="offline-test-records", daemon=True)
        self._worker.start()
        self._ready.wait(3)

    def stop(self):
        self.finish("service_stop", {})
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3)

    def snapshot(self):
        with self._lock:
            return {"test_record_state": self._state, "test_record_error": self._error,
                    "test_run_id": self._last_id, "test_record_active": self._active is not None,
                    "test_record_directory": str(self.directory) if self.config.enabled else None,
                    "test_record_images": self._saved, "test_record_rows": self._rows,
                    "test_record_dropped": self._dropped, "test_record_limits": list(self._limits),
                    "test_record_line_sensor_rows": self._sensor_rows,
                    "test_record_line_sensor_dropped": self._sensor_dropped}

    def _enqueue_control(self, item):
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            self._state, self._error = "ERROR", "Test record queue full; event could not be saved"
            raise RuntimeError(self._error) from exc

    def begin(self, config, status):
        if not self.config.enabled:
            return
        with self._lock:
            if self._active is not None:
                return  # Duplicate Start must not split an active run.
            if self._state not in {"READY", "COMPLETE"}:
                raise RuntimeError(self._error or "Offline test recorder is not ready")
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:12]
            self._enqueue_control(("begin", run_id, copy.deepcopy(config), copy.deepcopy(status), utc_now(), time.monotonic()))
            self._active = self._last_id = run_id
            self._state = "RECORDING"
            self._saved = self._rows = self._dropped = 0
            self._sensor_rows = self._sensor_dropped = 0
            self._limits = []
            self._last_image_s = -math.inf

    def finish(self, reason, status):
        if not self.config.enabled:
            return
        with self._lock:
            if self._active is None:
                return
            run_id = self._active
            self._active = None
            try:
                self._enqueue_control(("end", run_id, reason, copy.deepcopy(status), utc_now(), time.monotonic(), self._dropped, self._sensor_dropped))
            except RuntimeError:
                return  # A record error must never prevent a motor Stop request.
            if self._state == "RECORDING":
                self._state = "FINALIZING"

    def offer(self, frame, status, *, run_id):
        if not self.config.enabled:
            return
        with self._lock:
            if run_id is None or run_id != self._active or self._state != "RECORDING":
                return
            if self._queue.qsize() >= 248:  # Leave slots for Stop and end reasons.
                self._dropped += 1
                return
            stamp = status["vision_timestamp_monotonic_s"]
            image = None
            if stamp - self._last_image_s >= 1.0 / self.config.image_hz:
                self._last_image_s = stamp
                image = frame.copy()
            self._queue.put_nowait(("frame", run_id, copy.deepcopy(status), image, utc_now()))

    def offer_line_sensor_sample(self, sample, *, run_id):
        if not self.config.enabled:
            return
        with self._lock:
            if run_id is None or run_id != self._active or self._state != "RECORDING":
                return
            if self._queue.qsize() >= 248:
                self._dropped += 1
                self._sensor_dropped += 1
                return
            self._queue.put_nowait(("line_sample", run_id, asdict(sample), utc_now()))

    @staticmethod
    def _json(path, value):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def _write(self):
        handle = None
        run = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            for old in self.directory.iterdir():
                if old.is_dir() and (old / "manifest.json").exists() and not (old / "summary.json").exists():
                    self._json(old / "summary.json", {"run_id": old.name, "result": "INTERRUPTED",
                        "end_reason": "previous service exited without a completed end record", "detected_utc": utc_now(),
                        "physical_stop_confirmed": False})
            if shutil.disk_usage(self.directory).free < self.config.reserve_free_mb * 1024**2:
                raise OSError("Insufficient free space for offline recording")
            with self._lock:
                self._state = "READY"
            self._ready.set()
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                kind, run_id, *data = item
                if kind == "begin":
                    config, status, stamp, monotonic = data
                    folder = self.directory / run_id
                    folder.mkdir(exist_ok=False)
                    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in self._source_directory.glob("*.py")}
                    source_folder = folder / "source"
                    source_folder.mkdir()
                    for source in self._source_directory.glob("*.py"):
                        shutil.copy2(source, source_folder / source.name)
                    self._json(folder / "manifest.json", {"run_id": run_id, "started_utc": stamp,
                        "started_monotonic_s": monotonic, "config": config, "status_before": status,
                        "source_sha256": sources, "opencv_version": cv2.__version__,
                        "line_sensor_format": "timestamped line_sensor_sample events independent of camera frames",
                        "image_format": "raw PNG without overlays", "software_commands_only": True})
                    handle = (folder / "telemetry.jsonl").open("x", encoding="utf-8")
                    run = {"run_id": run_id, "folder": folder, "start": monotonic,
                           "images": 0, "rows": 0, "sensor_rows": 0, "bytes": 0, "limits": [], "flush": 0.0}
                    handle.write(json.dumps({"event": "start", "timestamp_utc": stamp}) + "\n")
                elif run is not None and run["run_id"] == run_id and kind in {"frame", "line_sample"}:
                    if kind == "frame":
                        status, image, stamp = data
                    else:
                        sample, stamp = data
                    if run["bytes"] >= self.config.max_run_mb * 1024**2:
                        if "run_size_limit" not in run["limits"]:
                            run["limits"].append("run_size_limit")
                        with self._lock:
                            self._state = "LIMIT_REACHED"
                            self._limits = list(run["limits"])
                        continue
                    if shutil.disk_usage(self.directory).free < self.config.reserve_free_mb * 1024**2:
                        if "free_space_reserve" not in run["limits"]:
                            run["limits"].append("free_space_reserve")
                        with self._lock:
                            self._state = "LIMIT_REACHED"
                            self._limits = list(run["limits"])
                        continue
                    if kind == "line_sample":
                        row = {"event": "line_sensor_sample", "timestamp_utc": stamp, "sample": sample}
                        run["sensor_rows"] += 1
                    else:
                        row = {"timestamp_utc": stamp, "status": status}
                        if image is not None and run["images"] < self.config.max_images:
                            ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                            if not ok:
                                raise OSError("Test image encoding failed")
                            name = f"frame_{run['images']:06d}.png"
                            (run["folder"] / name).write_bytes(encoded.tobytes())
                            run["bytes"] += encoded.nbytes
                            run["images"] += 1
                            row["image"] = name
                        elif image is not None and "image_count_limit" not in run["limits"]:
                            run["limits"].append("image_count_limit")
                    line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    handle.write(line)
                    run["bytes"] += len(line.encode("utf-8"))
                    run["rows"] += kind == "frame"
                    if time.monotonic() - run["flush"] >= 1:
                        handle.flush()
                        os.fsync(handle.fileno())
                        run["flush"] = time.monotonic()
                    with self._lock:
                        if self._last_id == run_id:
                            self._saved, self._rows = run["images"], run["rows"]
                            self._sensor_rows = run["sensor_rows"]
                            self._limits = list(run["limits"])
                elif run is not None and run["run_id"] == run_id and kind == "end":
                    reason, status, stamp, monotonic, dropped, sensor_dropped = data
                    handle.write(json.dumps({"event": "end", "reason": reason, "timestamp_utc": stamp,
                                             "status": status}, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                    handle = None
                    self._json(run["folder"] / "summary.json", {"run_id": run_id, "result": "RECORDED",
                        "end_reason": reason, "ended_utc": stamp, "duration_s": monotonic - run["start"],
                        "frames": run["images"], "telemetry_rows": run["rows"], "dropped": dropped,
                        "line_sensor_samples": run["sensor_rows"], "line_sensor_dropped": sensor_dropped,
                        "limits": run["limits"], "evidence_complete": not dropped and not run["limits"],
                        "end_status": status, "physical_stop_confirmed": False,
                        "note": "Recorded evidence does not certify track performance or physical wheel stop"})
                    run = None
                    with self._lock:
                        if self._active is None:
                            self._state = "COMPLETE"
        except Exception as exc:
            with self._lock:
                self._state, self._error = "ERROR", str(exc)
            self._ready.set()
        finally:
            if handle is not None:
                handle.close()
