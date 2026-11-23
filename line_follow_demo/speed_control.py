from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class CurveSpeedConfig:
    enabled: bool = False
    minimum_scale: float = 1.0
    maximum_scale: float = 1.0
    straight_scale: float = 1.0
    curve_scale: float = 1.0
    full_curve_px_inv: float = 0.0
    acceleration_scale_per_s: float = 0.0
    deceleration_scale_per_s: float = 0.0


def speed_control_config_from_dict(
    values: Any, *, curve_threshold_px_inv: float
) -> CurveSpeedConfig:
    defaults = CurveSpeedConfig()
    if not isinstance(values, dict):
        raise ValueError("control.speed_control must be an object")

    enabled = values.get("enabled", defaults.enabled)
    if not isinstance(enabled, bool):
        raise ValueError("speed_control.enabled must be a boolean")

    numeric_values = {
        "minimum_scale": (defaults.minimum_scale, 0.0, 1.0),
        "maximum_scale": (defaults.maximum_scale, 0.0, 1.0),
        "straight_scale": (defaults.straight_scale, 0.0, 1.0),
        "curve_scale": (defaults.curve_scale, 0.0, 1.0),
        "full_curve_px_inv": (defaults.full_curve_px_inv, 0.0, 0.02),
        "acceleration_scale_per_s": (
            defaults.acceleration_scale_per_s,
            0.0,
            5.0,
        ),
        "deceleration_scale_per_s": (
            defaults.deceleration_scale_per_s,
            0.0,
            5.0,
        ),
    }
    parsed: dict[str, float] = {}
    for name, (default, lower, upper) in numeric_values.items():
        value = values.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"speed_control.{name} must be numeric")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"speed_control.{name} must be numeric") from exc
        if not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"speed_control.{name} must be between {lower} and {upper}")
        parsed[name] = number

    if not (
        parsed["minimum_scale"]
        <= parsed["curve_scale"]
        <= parsed["straight_scale"]
        <= parsed["maximum_scale"]
    ):
        raise ValueError(
            "speed scales must satisfy minimum <= curve <= straight <= maximum"
        )

    if enabled:
        if parsed["full_curve_px_inv"] <= curve_threshold_px_inv:
            raise ValueError(
                "speed_control.full_curve_px_inv must exceed curve_threshold_px_inv"
            )
        if (
            parsed["acceleration_scale_per_s"] <= 0.0
            or parsed["deceleration_scale_per_s"] <= 0.0
        ):
            raise ValueError(
                "enabled speed control requires positive acceleration and deceleration rates"
            )

    return CurveSpeedConfig(enabled=enabled, **parsed)


class CurveSpeedController:
    def __init__(self, config: CurveSpeedConfig | None = None):
        self._config = config or CurveSpeedConfig()
        self._scale = self._config.minimum_scale if self._config.enabled else 1.0

    @property
    def config(self) -> CurveSpeedConfig:
        return self._config

    @property
    def current_scale(self) -> float:
        return self._scale

    def reset(self) -> None:
        self._scale = self._config.minimum_scale if self._config.enabled else 1.0

    def target_scale(
        self,
        curvature_px_inv: float | None,
        *,
        curve_threshold_px_inv: float,
        path_valid: bool,
    ) -> float:
        if not self._config.enabled:
            return 1.0
        if not path_valid or curvature_px_inv is None:
            return self._config.minimum_scale
        try:
            curvature = float(curvature_px_inv)
            threshold = float(curve_threshold_px_inv)
        except (OverflowError, TypeError, ValueError):
            return self._config.minimum_scale
        if not math.isfinite(curvature) or not math.isfinite(threshold):
            return self._config.minimum_scale
        span = self._config.full_curve_px_inv - threshold
        if threshold < 0.0 or span <= 0.0:
            return self._config.minimum_scale

        severity = max(0.0, min(1.0, (abs(curvature) - threshold) / span))
        target = self._config.straight_scale + severity * (
            self._config.curve_scale - self._config.straight_scale
        )
        return max(self._config.minimum_scale, min(self._config.maximum_scale, target))

    def update(self, target_scale: float, dt: float) -> float:
        if not self._config.enabled:
            self._scale = 1.0
            return self._scale
        try:
            target = float(target_scale)
            elapsed = float(dt)
        except (OverflowError, TypeError, ValueError):
            return self._scale
        if not math.isfinite(target) or not math.isfinite(elapsed) or elapsed <= 0.0:
            return self._scale

        target = max(self._config.minimum_scale, min(self._config.maximum_scale, target))
        elapsed = min(elapsed, 1.0)
        if target < self._scale:
            rate = self._config.deceleration_scale_per_s
            self._scale = max(target, self._scale - rate * elapsed)
        else:
            rate = self._config.acceleration_scale_per_s
            self._scale = min(target, self._scale + rate * elapsed)
        return self._scale


def scale_wheel_commands(
    values: tuple[float, float, float, float], scale: float
) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("exactly four wheel commands are required")
    factor = float(scale)
    if not math.isfinite(factor):
        raise ValueError("speed scale must be finite")
    factor = max(0.0, min(1.0, factor))
    return tuple(float(value) * factor for value in values)
