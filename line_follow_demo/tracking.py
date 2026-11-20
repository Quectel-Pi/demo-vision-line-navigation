from __future__ import annotations

from dataclasses import dataclass
import threading

from .vision import LineFeatures


@dataclass(frozen=True)
class TemporalAssessment:
    state: str
    center_x: float | None
    confidence: float
    reacquire_count: int
    missed_frames: int


class TemporalPathFilter:
    def __init__(
        self,
        *,
        max_center_shift_norm_per_s: float,
        confirm_frames: int,
        max_uncertain_frames: int,
    ):
        self._max_center_shift_norm_per_s = max(0.0, float(max_center_shift_norm_per_s))
        self._confirm_frames = max(1, int(confirm_frames))
        self._max_uncertain_frames = max(0, int(max_uncertain_frames))
        self._lock = threading.Lock()
        self._last_valid_x: float | None = None
        self._pending_x: float | None = None
        self._pending_count = 0
        self._missed_frames = 0

    def reset(self) -> None:
        with self._lock:
            self._last_valid_x = None
            self._pending_x = None
            self._pending_count = 0
            self._missed_frames = 0

    def update(self, features: LineFeatures, *, frame_width: int, dt: float) -> TemporalAssessment:
        with self._lock:
            if features.state != "VALID" or features.x_near is None:
                self._missed_frames += 1
                self._pending_x = None
                self._pending_count = 0
                if self._missed_frames > self._max_uncertain_frames:
                    state = "LOST"
                    self._last_valid_x = None
                else:
                    state = "UNCERTAIN"
                return TemporalAssessment(
                    state=state,
                    center_x=None,
                    confidence=features.confidence,
                    reacquire_count=0,
                    missed_frames=self._missed_frames,
                )

            candidate_x = float(features.x_near)
            max_shift = max(
                2.0,
                frame_width * self._max_center_shift_norm_per_s * max(0.001, dt),
            )

            if (
                self._last_valid_x is not None
                and self._missed_frames == 0
                and abs(candidate_x - self._last_valid_x) <= max_shift
            ):
                self._last_valid_x = candidate_x
                self._pending_x = None
                self._pending_count = 0
                return TemporalAssessment(
                    state="VALID",
                    center_x=candidate_x,
                    confidence=features.confidence,
                    reacquire_count=self._confirm_frames,
                    missed_frames=0,
                )

            pending_limit = max(max_shift, frame_width * 0.02)
            if self._pending_x is None or abs(candidate_x - self._pending_x) > pending_limit:
                self._pending_x = candidate_x
                self._pending_count = 1
            else:
                self._pending_x = (self._pending_x * self._pending_count + candidate_x) / (self._pending_count + 1)
                self._pending_count += 1

            self._missed_frames = 0
            if self._pending_count >= self._confirm_frames:
                self._last_valid_x = self._pending_x
                self._pending_x = None
                self._pending_count = 0
                return TemporalAssessment(
                    state="VALID",
                    center_x=candidate_x,
                    confidence=features.confidence,
                    reacquire_count=self._confirm_frames,
                    missed_frames=0,
                )

            return TemporalAssessment(
                state="UNCERTAIN",
                center_x=None,
                confidence=features.confidence * self._pending_count / self._confirm_frames,
                reacquire_count=self._pending_count,
                missed_frames=0,
            )
