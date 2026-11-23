"""Measured camera-to-ground mapping and bounded front-reference prediction.

Calibration points must be collected from physical ground marks relative to the
front-edge centre. Empty/default calibration is deliberately unusable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class GroundPathEstimate:
    state: str = "UNCERTAIN"
    reason: str = "calibration_unavailable"
    mapped_points_m: tuple[tuple[float, float], ...] = ()
    fit_points_m: tuple[tuple[float, float], ...] = ()
    predicted_point_m: tuple[float, float] | None = None
    nearest_observation_m: float | None = None
    lateral_error_m: float | None = None
    heading_error_deg: float | None = None
    fit_rms_m: float | None = None
    confidence: float | None = None
    extrapolation_m: float | None = None
    predicted: bool = False


def load_ground_calibration(path: str | Path) -> dict[str, Any]:
    """Load the independent ground calibration; missing files mean disabled."""
    calibration_path = Path(path)
    if not calibration_path.exists():
        return {"enabled": False, "_load_error": "calibration_file_missing"}
    try:
        values = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"enabled": False, "_load_error": f"calibration_file_invalid: {exc}"}
    if not isinstance(values, dict):
        return {"enabled": False, "_load_error": "calibration_root_must_be_object"}
    return values


class GroundPlaneMapper:
    """Map measured image points to the chassis plane without extrapolating.

    The homography is restricted to the convex hull of supplied image points.
    Front-reference prediction is separately gated by a metric distance limit
    and a configured maximum fit residual.
    """

    def __init__(self, values: Any):
        self._values = values if isinstance(values, dict) else {}
        self._reason = str(self._values.get("_load_error", ""))
        self._enabled = self._values.get("enabled") is True
        self._matrix: np.ndarray | None = None
        self._image_hull: np.ndarray | None = None
        self._frame_size: tuple[int, int] | None = None
        self._camera_pose: tuple[int, int] | None = None
        self._ground_points_m: np.ndarray | None = None
        self._row_distance_samples: tuple[tuple[float, float], ...] = ()
        if not self._reason:
            self._prepare()

    @property
    def available(self) -> bool:
        return self._enabled and self._matrix is not None and not self._reason

    @property
    def reason(self) -> str:
        return self._reason or ("calibration_disabled" if not self._enabled else "")

    @property
    def ground_plot_bounds_m(self) -> tuple[float, float, float, float] | None:
        """Measured X/Y plot limits including the defined vehicle origin."""
        if self._ground_points_m is None:
            return None
        x_values = self._ground_points_m[:, 0]
        y_values = self._ground_points_m[:, 1]
        return (
            min(0.0, float(np.min(x_values))),
            max(0.0, float(np.max(x_values))),
            0.0,
            float(np.max(y_values)),
        )

    def _prepare(self) -> None:
        if not self._enabled:
            self._reason = "calibration_disabled"
            return
        try:
            resolution = self._values["reference_resolution"]
            width = int(resolution["width"])
            height = int(resolution["height"])
            pose = self._values["camera_pose_pwm_us"]
            pan = int(pose["pan"])
            tilt = int(pose["tilt"])
            image_points = np.asarray(self._values["image_points_px"], dtype=np.float64)
            ground_points = np.asarray(self._values["ground_points_m"], dtype=np.float64)
            raw_row_samples = self._values.get("row_distance_samples", [])
        except (KeyError, TypeError, ValueError, OverflowError):
            self._reason = "calibration_fields_invalid"
            return

        if width <= 0 or height <= 0:
            self._reason = "reference_resolution_invalid"
            return
        if image_points.ndim != 2 or image_points.shape[1:] != (2,) or len(image_points) < 4:
            self._reason = "at_least_four_measured_image_points_required"
            return
        if ground_points.shape != image_points.shape:
            self._reason = "image_and_ground_point_counts_must_match"
            return
        if not np.all(np.isfinite(image_points)) or not np.all(np.isfinite(ground_points)):
            self._reason = "calibration_points_must_be_finite"
            return
        if np.any(image_points[:, 0] < 0) or np.any(image_points[:, 0] > width - 1):
            self._reason = "image_calibration_point_out_of_frame"
            return
        if np.any(image_points[:, 1] < 0) or np.any(image_points[:, 1] > height - 1):
            self._reason = "image_calibration_point_out_of_frame"
            return
        if np.any(ground_points[:, 1] < 0):
            self._reason = "ground_points_must_be_ahead_of_front_reference"
            return
        if not isinstance(raw_row_samples, list):
            self._reason = "row_distance_samples_must_be_a_list"
            return
        try:
            row_samples = tuple(
                (float(sample["row_px"]), float(sample["distance_m"]))
                for sample in raw_row_samples
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            self._reason = "row_distance_samples_invalid"
            return
        if any(
            not math.isfinite(row) or not math.isfinite(distance)
            or not 0.0 <= row <= height - 1 or distance < 0.0
            for row, distance in row_samples
        ):
            self._reason = "row_distance_samples_out_of_range"
            return
        row_values = [row for row, _ in row_samples]
        distance_values = [distance for _, distance in row_samples]
        if row_values != sorted(set(row_values)):
            self._reason = "row_distance_sample_rows_must_be_sorted_unique"
            return
        if len(distance_values) > 1 and not (
            all(a < b for a, b in zip(distance_values, distance_values[1:]))
            or all(a > b for a, b in zip(distance_values, distance_values[1:]))
        ):
            self._reason = "row_distance_samples_must_be_monotonic"
            return

        matrix, _ = cv2.findHomography(image_points, ground_points, method=0)
        if matrix is None or not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < 1e-12:
            self._reason = "ground_homography_singular"
            return

        hull = cv2.convexHull(image_points.astype(np.float32))
        if hull is None or len(hull) < 4 or abs(float(cv2.contourArea(hull))) < 1.0:
            self._reason = "image_calibration_points_not_spatially_distributed"
            return

        self._matrix = matrix
        self._image_hull = hull
        self._frame_size = (width, height)
        self._camera_pose = (pan, tilt)
        self._ground_points_m = ground_points
        self._row_distance_samples = row_samples

    def _context_reason(
        self,
        *,
        frame_width: int,
        frame_height: int,
        pan_pulse_us: int | None,
        tilt_pulse_us: int | None,
    ) -> str | None:
        if not self.available:
            return self.reason or "calibration_unavailable"
        if self._frame_size != (frame_width, frame_height):
            return "frame_resolution_does_not_match_calibration"
        if pan_pulse_us is None or tilt_pulse_us is None:
            return "camera_pose_command_unknown"
        if self._camera_pose != (int(pan_pulse_us), int(tilt_pulse_us)):
            return "camera_pose_command_does_not_match_calibration"
        return None

    def map_image_point(
        self,
        x_px: float,
        y_px: float,
        *,
        frame_width: int,
        frame_height: int,
        pan_pulse_us: int | None,
        tilt_pulse_us: int | None,
    ) -> tuple[float, float] | None:
        """Return (lateral-right m, forward m) inside the measured image hull."""
        if self._context_reason(
            frame_width=frame_width,
            frame_height=frame_height,
            pan_pulse_us=pan_pulse_us,
            tilt_pulse_us=tilt_pulse_us,
        ) is not None:
            return None
        if not math.isfinite(float(x_px)) or not math.isfinite(float(y_px)):
            return None
        assert self._matrix is not None and self._image_hull is not None
        point = (float(x_px), float(y_px))
        if cv2.pointPolygonTest(self._image_hull, point, False) < 0:
            return None
        source = np.asarray([[[point[0], point[1]]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(source, self._matrix)[0, 0]
        x_m, y_m = float(mapped[0]), float(mapped[1])
        if not math.isfinite(x_m) or not math.isfinite(y_m) or y_m < 0.0:
            return None
        return x_m, y_m

    def distance_for_image_row(
        self,
        row_px: float,
        *,
        frame_width: int,
        frame_height: int,
        pan_pulse_us: int | None,
        tilt_pulse_us: int | None,
    ) -> float | None:
        """Interpolate the measured centreline row LUT; never extrapolate."""
        if self._context_reason(
            frame_width=frame_width,
            frame_height=frame_height,
            pan_pulse_us=pan_pulse_us,
            tilt_pulse_us=tilt_pulse_us,
        ) is not None or len(self._row_distance_samples) < 2:
            return None
        if not math.isfinite(float(row_px)):
            return None
        samples = self._row_distance_samples
        if row_px < samples[0][0] or row_px > samples[-1][0]:
            return None
        if row_px == samples[0][0]:
            return samples[0][1]
        if row_px == samples[-1][0]:
            return samples[-1][1]
        for (row0, distance0), (row1, distance1) in zip(samples, samples[1:]):
            if row0 <= row_px <= row1:
                ratio = (row_px - row0) / (row1 - row0)
                return distance0 + ratio * (distance1 - distance0)
        return None

    def estimate_front_reference_path(
        self,
        centerline: tuple[tuple[int, int], ...] | list[tuple[int, int]],
        *,
        frame_width: int,
        frame_height: int,
        pan_pulse_us: int | None,
        tilt_pulse_us: int | None,
        path_confidence: float,
        path_fit_valid: bool,
    ) -> GroundPathEstimate:
        """Estimate path x/heading at chassis y=0, with strict quality limits."""
        context_reason = self._context_reason(
            frame_width=frame_width,
            frame_height=frame_height,
            pan_pulse_us=pan_pulse_us,
            tilt_pulse_us=tilt_pulse_us,
        )
        if context_reason is not None:
            return GroundPathEstimate(reason=context_reason)
        if not path_fit_valid:
            return GroundPathEstimate(reason="image_path_fit_invalid")
        if not math.isfinite(path_confidence):
            return GroundPathEstimate(reason="path_confidence_invalid")
        min_confidence = self._finite_config_float("min_path_confidence", 0.65)
        if min_confidence is None or not 0.0 <= min_confidence <= 1.0:
            return GroundPathEstimate(reason="min_path_confidence_invalid")
        if path_confidence < min_confidence:
            return GroundPathEstimate(reason="path_confidence_below_limit", confidence=path_confidence)

        mapped: list[tuple[float, float]] = []
        for x_px, y_px in centerline:
            point = self.map_image_point(
                x_px,
                y_px,
                frame_width=frame_width,
                frame_height=frame_height,
                pan_pulse_us=pan_pulse_us,
                tilt_pulse_us=tilt_pulse_us,
            )
            if point is not None:
                mapped.append(point)
        if len(mapped) < 3 or len({round(point[1], 9) for point in mapped}) < 3:
            return GroundPathEstimate(
                reason="too_few_measured_ground_path_points",
                mapped_points_m=tuple(mapped),
                confidence=path_confidence,
            )

        mapped.sort(key=lambda point: point[1])
        x_values = np.asarray([point[0] for point in mapped], dtype=np.float64)
        y_values = np.asarray([point[1] for point in mapped], dtype=np.float64)
        degree = 2 if len(mapped) >= 5 else 1
        design = np.vander(y_values, N=degree + 1)
        coefficients, _, rank, _ = np.linalg.lstsq(design, x_values, rcond=None)
        if rank < degree + 1 or not np.all(np.isfinite(coefficients)):
            return GroundPathEstimate(
                reason="ground_path_fit_singular",
                mapped_points_m=tuple(mapped),
                confidence=path_confidence,
            )
        fit_rms = float(np.sqrt(np.mean((design @ coefficients - x_values) ** 2)))
        nearest = float(y_values[0])
        extrapolation = max(0.0, nearest)
        max_fit_rms = self._finite_config_float("max_fit_rms_m", None)
        max_extrapolation = self._finite_config_float("max_extrapolation_m", 0.0)
        if max_fit_rms is None or max_fit_rms <= 0.0:
            return GroundPathEstimate(
                reason="max_fit_rms_m_not_configured",
                mapped_points_m=tuple(mapped),
                nearest_observation_m=nearest,
                fit_rms_m=fit_rms,
                confidence=path_confidence,
                extrapolation_m=extrapolation,
            )
        if fit_rms > max_fit_rms:
            return GroundPathEstimate(
                reason="ground_path_fit_rms_exceeds_limit",
                mapped_points_m=tuple(mapped),
                nearest_observation_m=nearest,
                fit_rms_m=fit_rms,
                confidence=path_confidence,
                extrapolation_m=extrapolation,
            )
        if max_extrapolation is None or max_extrapolation < 0.0:
            return GroundPathEstimate(
                reason="max_extrapolation_m_invalid",
                mapped_points_m=tuple(mapped),
                nearest_observation_m=nearest,
                fit_rms_m=fit_rms,
                confidence=path_confidence,
                extrapolation_m=extrapolation,
            )
        if y_values[0] > 0.0 and extrapolation > max_extrapolation:
            return GroundPathEstimate(
                reason="front_reference_prediction_exceeds_limit",
                mapped_points_m=tuple(mapped),
                nearest_observation_m=nearest,
                fit_rms_m=fit_rms,
                confidence=path_confidence,
                extrapolation_m=extrapolation,
            )

        if degree == 2:
            lateral_error = float(coefficients[2])
            slope = float(coefficients[1])
        else:
            lateral_error = float(coefficients[1])
            slope = float(coefficients[0])
        heading_error = math.degrees(math.atan(slope))
        if not math.isfinite(lateral_error) or not math.isfinite(heading_error):
            return GroundPathEstimate(reason="ground_path_prediction_non_finite")
        fit_y_values = np.linspace(float(y_values[0]), float(y_values[-1]), 17)
        fit_x_values = np.polyval(coefficients, fit_y_values)
        fit_points = tuple(
            (float(x), float(y)) for x, y in zip(fit_x_values, fit_y_values)
        )
        return GroundPathEstimate(
            state="VALID",
            reason="ok",
            mapped_points_m=tuple(mapped),
            fit_points_m=fit_points,
            predicted_point_m=(lateral_error, 0.0),
            nearest_observation_m=nearest,
            lateral_error_m=lateral_error,
            heading_error_deg=heading_error,
            fit_rms_m=fit_rms,
            confidence=path_confidence,
            extrapolation_m=extrapolation if y_values[0] > 0.0 else 0.0,
            predicted=not (y_values[0] <= 0.0 <= y_values[-1]),
        )

    def _finite_config_float(self, name: str, default: float | None) -> float | None:
        value = self._values.get(name, default)
        if value is None or isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None
