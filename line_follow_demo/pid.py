from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float
    integral_limit: float
    output_limit: float
    derivative_filter_alpha: float = 1.0
    output_smoothing_alpha: float = 1.0
    large_error_threshold: float = 1.0
    large_error_gain: float = 1.0


class PID:
    def __init__(self, cfg: PIDConfig):
        self._cfg = cfg
        self._integral = 0.0
        self._prev_error: float | None = None
        self._filtered_derivative: float | None = None
        self._prev_output: float | None = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = None
        self._prev_output = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        previous_integral = self._integral
        self._integral += error * dt
        if self._cfg.integral_limit > 0:
            self._integral = max(-self._cfg.integral_limit, min(self._integral, self._cfg.integral_limit))

        if self._prev_error is None:
            derivative = 0.0
        else:
            raw_derivative = (error - self._prev_error) / dt
            alpha = self._bounded_alpha(self._cfg.derivative_filter_alpha)
            if self._filtered_derivative is None:
                derivative = raw_derivative
            else:
                derivative = alpha * raw_derivative + (1.0 - alpha) * self._filtered_derivative

        self._prev_error = error
        self._filtered_derivative = derivative

        gain = self._cfg.large_error_gain if abs(error) >= self._cfg.large_error_threshold else 1.0
        output = gain * (self._cfg.kp * error + self._cfg.kd * derivative) + self._cfg.ki * self._integral
        if self._cfg.output_limit > 0:
            limit = self._cfg.output_limit
            integral_drive = self._cfg.ki * error
            if (output > limit and integral_drive > 0.0) or (
                output < -limit and integral_drive < 0.0
            ):
                self._integral = previous_integral
                output = gain * (self._cfg.kp * error + self._cfg.kd * derivative) + self._cfg.ki * self._integral
            output = max(-limit, min(output, limit))

        alpha = self._bounded_alpha(self._cfg.output_smoothing_alpha)
        if self._prev_output is not None:
            output = alpha * output + (1.0 - alpha) * self._prev_output
        self._prev_output = output
        return output

    @staticmethod
    def _bounded_alpha(value: float) -> float:
        if not math.isfinite(value):
            return 1.0
        return max(0.0, min(1.0, float(value)))
