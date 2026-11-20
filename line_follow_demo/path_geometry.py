from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PathGeometryConfig:
    smoothing_alpha: float = 0.35
    curve_threshold_px_inv: float = 0.0005
    min_points: int = 5
    sample_count: int = 13
    min_heading_points: int = 3
    min_span_height_ratio: float = 0.06
    max_fit_rms_width_ratio: float = 0.015


@dataclass(frozen=True)
class PathGeometry:
    valid: bool = False
    heading_deg: float | None = None
    curvature_px_inv: float | None = None
    curve_direction: str = "UNKNOWN"
    fit_rms_px: float | None = None
    centerline: tuple[tuple[int, int], ...] = ()
    heading_source: str = "NONE"


def path_geometry_config_from_dict(values: Any) -> PathGeometryConfig:
    defaults = PathGeometryConfig()
    if not isinstance(values, dict):
        return defaults

    try:
        smoothing_alpha = float(values.get("smoothing_alpha", defaults.smoothing_alpha))
        curve_threshold = float(values.get("curve_threshold_px_inv", defaults.curve_threshold_px_inv))
        min_points = int(values.get("min_points", defaults.min_points))
        sample_count = int(values.get("sample_count", defaults.sample_count))
        min_heading_points = int(values.get("min_heading_points", defaults.min_heading_points))
        min_span = float(values.get("min_span_height_ratio", defaults.min_span_height_ratio))
        max_rms = float(values.get("max_fit_rms_width_ratio", defaults.max_fit_rms_width_ratio))
    except (OverflowError, TypeError, ValueError):
        return defaults

    if not all(math.isfinite(value) for value in (smoothing_alpha, curve_threshold, min_span, max_rms)):
        return defaults
    return PathGeometryConfig(
        smoothing_alpha=max(0.05, min(1.0, smoothing_alpha)),
        curve_threshold_px_inv=max(0.0, min(0.02, curve_threshold)),
        min_points=max(3, min(41, min_points)),
        sample_count=max(5, min(41, sample_count)),
        min_heading_points=max(3, min(41, min_heading_points)),
        min_span_height_ratio=max(0.01, min(1.0, min_span)),
        max_fit_rms_width_ratio=max(0.001, min(0.1, max_rms)),
    )


class PathGeometryEstimator:
    def __init__(self, config: PathGeometryConfig | None = None):
        self._config = config or PathGeometryConfig()
        self._smoothed_coefficients: np.ndarray | None = None
        self._fit_degree: int | None = None

    def reset(self) -> None:
        self._smoothed_coefficients = None
        self._fit_degree = None

    def update(
        self,
        centerline: tuple[tuple[int, int], ...] | list[tuple[int, int]],
        *,
        frame_width: int,
        frame_height: int,
    ) -> PathGeometry:
        if frame_width <= 0 or frame_height <= 0:
            self.reset()
            return PathGeometry()

        points_by_y: dict[float, list[float]] = {}
        for point in centerline:
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points_by_y.setdefault(y, []).append(x)

        if len(points_by_y) < self._config.min_heading_points:
            self.reset()
            return PathGeometry()

        near_y = max(points_by_y)
        far_y = min(points_by_y)
        span_px = near_y - far_y
        if span_px < max(1.0, frame_height * self._config.min_span_height_ratio):
            self.reset()
            return PathGeometry()

        y_values = np.asarray(sorted(points_by_y), dtype=np.float64)
        x_values = np.asarray(
            [sum(points_by_y[y]) / len(points_by_y[y]) for y in y_values],
            dtype=np.float64,
        )
        # A fixed frame coordinate makes coefficients comparable when the far
        # path disappears. A coordinate relative to the visible span changes
        # scale/origin between frames and corrupts the smoothed tangent.
        forward_position = (frame_height - 1 - y_values) / frame_height
        degree = 2 if len(points_by_y) >= self._config.min_points else 1
        design = np.vander(forward_position, N=degree + 1)
        raw_coefficients, _, rank, _ = np.linalg.lstsq(
            design, x_values / frame_width, rcond=None
        )
        if rank < degree + 1 or not np.all(np.isfinite(raw_coefficients)):
            self.reset()
            return PathGeometry()

        raw_residual_px = (
            design @ raw_coefficients - x_values / frame_width
        ) * frame_width
        fit_rms_px = float(np.sqrt(np.mean(raw_residual_px**2)))
        if fit_rms_px > frame_width * self._config.max_fit_rms_width_ratio:
            self.reset()
            return PathGeometry()

        if self._smoothed_coefficients is None or self._fit_degree != degree:
            self._smoothed_coefficients = raw_coefficients
        else:
            alpha = self._config.smoothing_alpha
            self._smoothed_coefficients = (
                alpha * raw_coefficients + (1.0 - alpha) * self._smoothed_coefficients
            )
        self._fit_degree = degree

        if degree == 2:
            quadratic, linear, offset = self._smoothed_coefficients
        else:
            linear, offset = self._smoothed_coefficients
            quadratic = 0.0
        far_position = (frame_height - 1 - far_y) / frame_height
        near_position = (frame_height - 1 - near_y) / frame_height
        far_slope_px_per_px = frame_width * (2.0 * quadratic * far_position + linear) / frame_height
        heading_deg = math.degrees(math.atan(far_slope_px_per_px))

        middle_slope_px_per_px = frame_width * (quadratic * (far_position + near_position) + linear) / frame_height
        second_derivative_px_inv = 2.0 * quadratic * frame_width / (frame_height**2)
        curvature_px_inv = (
            second_derivative_px_inv / (1.0 + middle_slope_px_per_px**2) ** 1.5
            if degree == 2 else None
        )

        if curvature_px_inv is None:
            curve_direction = "UNKNOWN"
        elif abs(curvature_px_inv) <= self._config.curve_threshold_px_inv:
            curve_direction = "STRAIGHT"
        elif curvature_px_inv > 0.0:
            curve_direction = "RIGHT"
        else:
            curve_direction = "LEFT"

        sample_positions = np.linspace(near_position, far_position, self._config.sample_count)
        sample_x = frame_width * (
            quadratic * sample_positions**2 + linear * sample_positions + offset
        )
        sample_y = frame_height - 1 - sample_positions * frame_height
        smooth_centerline = tuple(
            (
                int(round(max(0.0, min(frame_width - 1.0, x)))),
                int(round(max(0.0, min(frame_height - 1.0, y)))),
            )
            for x, y in zip(sample_x, sample_y)
        )

        return PathGeometry(
            valid=True,
            heading_deg=heading_deg,
            curvature_px_inv=curvature_px_inv,
            curve_direction=curve_direction,
            fit_rms_px=fit_rms_px,
            centerline=smooth_centerline,
            heading_source="QUADRATIC" if degree == 2 else "NEAR_LINEAR",
        )
