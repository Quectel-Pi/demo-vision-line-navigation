from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, width: int, height: int) -> "ROI":
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        w = max(1, min(self.w, width - x))
        h = max(1, min(self.h, height - y))
        return ROI(x=x, y=y, w=w, h=h)


@dataclass(frozen=True)
class VisionConfig:
    threshold: int = 60
    blur_ksize: int = 5
    morph_ksize: int = 5
    max_width_ratio: float = 0.35
    reference_width: int = 640
    reference_height: int = 480
    ground_roi_norm: tuple[tuple[float, float], ...] = ()
    tape_width_mm: float = 15.0
    tape_width_min_mm: float = 15.0
    tape_width_max_mm: float = 15.0
    tape_width_profile: tuple[tuple[float, float], ...] = ()
    ground_calibration_confirmed: bool = False
    width_tolerance_ratio: float = 0.5
    scanline_count: int = 11
    scanline_positions_norm: tuple[float, ...] = ()
    min_valid_scanlines: int = 7
    min_near_valid_scanlines: int = 3
    max_scanline_gap: int = 1
    max_lateral_slope: float = 1.25
    max_width_change_ratio: float = 0.8
    min_confidence: float = 0.65
    center_transition_weight: float = 0.3
    width_transition_weight: float = 0.2
    confidence_width_fit_weight: float = 0.55
    confidence_smoothness_weight: float = 0.25
    confidence_coverage_weight: float = 0.2
    debug_visualization: bool = True

    # ROI 使用像素；默认按 640x480 经验值给一个“近端+远端”框，首次上车需要标定
    roi_near: ROI = ROI(x=0, y=320, w=640, h=120)
    roi_far: ROI = ROI(x=0, y=200, w=640, h=80)

    @property
    def calibration_ready(self) -> bool:
        if (
            not self.ground_calibration_confirmed
            or len(self.ground_roi_norm) != 4
            or len(self.tape_width_profile) < 3
            or self.tape_width_mm <= 0
            or self.tape_width_min_mm <= 0
            or self.tape_width_min_mm > 100
            or self.tape_width_max_mm < self.tape_width_min_mm
            or self.tape_width_max_mm > 100
        ):
            return False
        points = self.ground_roi_norm
        crosses = []
        for index in range(4):
            first = points[index]
            second = points[(index + 1) % 4]
            third = points[(index + 2) % 4]
            crosses.append(
                (second[0] - first[0]) * (third[1] - second[1])
                - (second[1] - first[1]) * (third[0] - second[0])
            )
        if any(abs(cross) < 1e-6 for cross in crosses) or not (
            all(cross > 0 for cross in crosses) or all(cross < 0 for cross in crosses)
        ):
            return False
        sample_rows = [sample[0] for sample in self.tape_width_profile]
        roi_top = min(y for _, y in points)
        roi_bottom = max(y for _, y in points)
        return (
            all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
            and all(0.0 < y < 1.0 and 0.0 < width <= 1.0 for y, width in self.tape_width_profile)
            and sample_rows == sorted(set(sample_rows))
            and sample_rows[0] <= roi_top
            and sample_rows[-1] >= roi_bottom
        )

    def expected_tape_width_px(self, y_norm: float, frame_width: int) -> float | None:
        if not self.tape_width_profile:
            return None
        samples = self.tape_width_profile
        if y_norm < samples[0][0] or y_norm > samples[-1][0]:
            return None
        if y_norm == samples[0][0]:
            normalized_width = samples[0][1]
        elif y_norm == samples[-1][0]:
            normalized_width = samples[-1][1]
        else:
            normalized_width = samples[-1][1]
            for (row0, width0), (row1, width1) in zip(samples, samples[1:]):
                if row0 <= y_norm <= row1:
                    ratio = (y_norm - row0) / (row1 - row0)
                    normalized_width = width0 + ratio * (width1 - width0)
                    break
        return normalized_width * frame_width

    def expected_tape_width_range_px(
        self, y_norm: float, frame_width: int
    ) -> tuple[float, float] | None:
        reference_width = self.expected_tape_width_px(y_norm, frame_width)
        if reference_width is None or self.tape_width_mm <= 0:
            return None
        return (
            reference_width * self.tape_width_min_mm / self.tape_width_mm,
            reference_width * self.tape_width_max_mm / self.tape_width_mm,
        )


@dataclass(frozen=True)
class LineFeatures:
    ok_near: bool
    ok_far: bool
    x_near: float | None
    x_far: float | None
    roi_near: ROI
    roi_far: ROI
    consistent: bool = False
    state: str = "LOST"
    confidence: float = 0.0
    valid_scanlines: int = 0
    candidate_scanlines: int = 0
    far_candidate_scanlines: int = 0
    middle_candidate_scanlines: int = 0
    near_candidate_scanlines: int = 0
    near_path_scanlines: int = 0
    near_path_confidence: float = 0.0
    centerline: tuple[tuple[int, int], ...] = ()
    expected_width_near_px: float | None = None
    expected_width_near_min_px: float | None = None
    expected_width_near_max_px: float | None = None


@dataclass(frozen=True)
class _ScanCandidate:
    left: int
    right: int
    width: int
    center: float
    expected_width: float | None
    fit_score: float


def _roi_from_dict(d: dict, default: ROI) -> ROI:
    try:
        return ROI(
            x=int(d.get("x", default.x)),
            y=int(d.get("y", default.y)),
            w=int(d.get("w", default.w)),
            h=int(d.get("h", default.h)),
        )
    except Exception:
        return default


def _normalized_points(values: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(values, list) or len(values) != 4:
        return ()
    try:
        points = tuple((float(point[0]), float(point[1])) for point in values)
    except (TypeError, ValueError, IndexError):
        return ()
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in points):
        return ()
    return points


def _width_profile(values: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(values, list):
        return ()
    samples = []
    try:
        for sample in values:
            samples.append((float(sample["y_norm"]), float(sample["width_norm"])))
    except (TypeError, ValueError, KeyError):
        return ()
    return tuple(samples)


def _scanline_positions(values: Any) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        return ()
    try:
        positions = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return ()
    if (
        not 3 <= len(positions) <= 41
        or any(not 0.0 <= value <= 1.0 for value in positions)
        or list(positions) != sorted(set(positions))
    ):
        return ()
    return positions


def vision_config_from_dict(d: dict) -> VisionConfig:
    cfg = VisionConfig()
    reference_resolution = d.get("reference_resolution", {})
    if not isinstance(reference_resolution, dict):
        reference_resolution = {}
    return VisionConfig(
        threshold=int(d.get("threshold", cfg.threshold)),
        blur_ksize=int(d.get("blur_ksize", cfg.blur_ksize)),
        morph_ksize=int(d.get("morph_ksize", cfg.morph_ksize)),
        max_width_ratio=float(d.get("max_width_ratio", cfg.max_width_ratio)),
        reference_width=max(1, int(reference_resolution.get("width", cfg.reference_width))),
        reference_height=max(1, int(reference_resolution.get("height", cfg.reference_height))),
        ground_roi_norm=_normalized_points(d.get("ground_roi_norm", [])),
        tape_width_mm=float(d.get("tape_width_mm", cfg.tape_width_mm)),
        tape_width_min_mm=float(
            d.get("tape_width_min_mm", d.get("tape_width_mm", cfg.tape_width_min_mm))
        ),
        tape_width_max_mm=float(
            d.get("tape_width_max_mm", d.get("tape_width_mm", cfg.tape_width_max_mm))
        ),
        tape_width_profile=_width_profile(d.get("tape_width_profile", [])),
        ground_calibration_confirmed=d.get("ground_calibration_confirmed", cfg.ground_calibration_confirmed) is True,
        width_tolerance_ratio=float(d.get("width_tolerance_ratio", cfg.width_tolerance_ratio)),
        scanline_count=min(41, max(3, int(d.get("scanline_count", cfg.scanline_count)))),
        scanline_positions_norm=_scanline_positions(d.get("scanline_positions_norm", [])),
        min_valid_scanlines=min(41, max(2, int(d.get("min_valid_scanlines", cfg.min_valid_scanlines)))),
        min_near_valid_scanlines=min(41, max(1, int(d.get("min_near_valid_scanlines", cfg.min_near_valid_scanlines)))),
        max_scanline_gap=min(1, max(0, int(d.get("max_scanline_gap", cfg.max_scanline_gap)))),
        max_lateral_slope=max(0.01, float(d.get("max_lateral_slope", cfg.max_lateral_slope))),
        max_width_change_ratio=max(0.01, float(d.get("max_width_change_ratio", cfg.max_width_change_ratio))),
        min_confidence=max(0.0, min(1.0, float(d.get("min_confidence", cfg.min_confidence)))),
        center_transition_weight=max(0.0, float(d.get("center_transition_weight", cfg.center_transition_weight))),
        width_transition_weight=max(0.0, float(d.get("width_transition_weight", cfg.width_transition_weight))),
        confidence_width_fit_weight=max(0.0, float(d.get("confidence_width_fit_weight", cfg.confidence_width_fit_weight))),
        confidence_smoothness_weight=max(0.0, float(d.get("confidence_smoothness_weight", cfg.confidence_smoothness_weight))),
        confidence_coverage_weight=max(0.0, float(d.get("confidence_coverage_weight", cfg.confidence_coverage_weight))),
        debug_visualization=d.get("debug_visualization", cfg.debug_visualization) is True,
        roi_near=_roi_from_dict(d.get("roi_near", {}), cfg.roi_near),
        roi_far=_roi_from_dict(d.get("roi_far", {}), cfg.roi_far),
    )


def _scale_roi(roi: ROI, cfg: VisionConfig, width: int, height: int) -> ROI:
    scale_x = width / cfg.reference_width
    scale_y = height / cfg.reference_height
    return ROI(
        x=round(roi.x * scale_x),
        y=round(roi.y * scale_y),
        w=max(1, round(roi.w * scale_x)),
        h=max(1, round(roi.h * scale_y)),
    ).clamp(width, height)


def _scanline_candidates(
    row: np.ndarray,
    expected_width: float | None,
    cfg: VisionConfig,
) -> list[_ScanCandidate]:
    black_pixels = np.flatnonzero(row)
    if black_pixels.size == 0:
        return []

    split_points = np.flatnonzero(np.diff(black_pixels) > 1) + 1
    intervals = np.split(black_pixels, split_points)
    max_width = max(1, int(row.size * cfg.max_width_ratio))
    tolerance = max(0.01, cfg.width_tolerance_ratio)
    max_angle_scale = math.sqrt(1.0 + cfg.max_lateral_slope**2)
    candidates = []
    for interval in intervals:
        left = int(interval[0])
        right = int(interval[-1])
        width = right - left + 1
        if width > max_width:
            continue
        if expected_width is None:
            fit_score = 0.5
        else:
            expected_min = expected_width * cfg.tape_width_min_mm / cfg.tape_width_mm
            expected_max = expected_width * cfg.tape_width_max_mm / cfg.tape_width_mm
            measurement_margin = expected_width * tolerance
            if (
                width < expected_min - measurement_margin
                or width > expected_max * max_angle_scale + measurement_margin
            ):
                continue
            fit_score = 1.0
        candidates.append(
            _ScanCandidate(
                left=left,
                right=right,
                width=width,
                center=(left + right) / 2.0,
                expected_width=expected_width,
                fit_score=fit_score,
            )
        )
    return candidates


def _normalized_candidate_width(
    candidate: _ScanCandidate, slope: float
) -> float | None:
    if candidate.expected_width is None or candidate.expected_width <= 0:
        return None
    angle_scale = math.sqrt(1.0 + slope**2)
    return candidate.width / (candidate.expected_width * angle_scale)


def _candidate_width_fit_score(
    candidate: _ScanCandidate, slope: float, cfg: VisionConfig
) -> float:
    ratio = _normalized_candidate_width(candidate, slope)
    if ratio is None:
        return candidate.fit_score
    minimum = cfg.tape_width_min_mm / cfg.tape_width_mm
    maximum = cfg.tape_width_max_mm / cfg.tape_width_mm
    if minimum <= ratio <= maximum:
        return 1.0
    tolerance = max(0.01, cfg.width_tolerance_ratio)
    difference = minimum - ratio if ratio < minimum else ratio - maximum
    return max(0.0, 1.0 - difference / tolerance)


def _scanline_path(
    rows: list[tuple[int, list[_ScanCandidate]]],
    cfg: VisionConfig,
) -> tuple[list[tuple[int, _ScanCandidate]], float]:
    states: list[list[tuple[float, list[tuple[int, _ScanCandidate]]]]] = []
    for row_index, (y, candidates) in enumerate(rows):
        row_states = []
        for candidate in candidates:
            best_score = _candidate_width_fit_score(candidate, 0.0, cfg)
            best_path = [(y, candidate)]
            # One bad scanrow (e.g. glare on tape) may be bridged using actual
            # candidates on either side. It never substitutes a stale frame.
            for previous_row in range(row_index - 1, max(-1, row_index - cfg.max_scanline_gap - 2), -1):
                previous_y, previous_candidates = rows[previous_row]
                dy = max(1, y - previous_y)
                for previous_index, previous in enumerate(previous_candidates):
                    previous_score, previous_path = states[previous_row][previous_index]
                    slope = abs(candidate.center - previous.center) / dy
                    if slope > cfg.max_lateral_slope:
                        continue
                    if candidate.expected_width and previous.expected_width:
                        candidate_ratio = _normalized_candidate_width(candidate, slope)
                        previous_ratio = _normalized_candidate_width(previous, slope)
                        minimum_ratio = (
                            cfg.tape_width_min_mm / cfg.tape_width_mm
                            - cfg.width_tolerance_ratio
                        )
                        maximum_ratio = (
                            cfg.tape_width_max_mm / cfg.tape_width_mm
                            + cfg.width_tolerance_ratio
                        )
                        if (
                            candidate_ratio is None
                            or previous_ratio is None
                            or not minimum_ratio <= candidate_ratio <= maximum_ratio
                            or not minimum_ratio <= previous_ratio <= maximum_ratio
                        ):
                            continue
                        width_change = abs(candidate_ratio - previous_ratio)
                    else:
                        width_change = abs(candidate.width - previous.width) / max(candidate.width, previous.width, 1)
                    if width_change > cfg.max_width_change_ratio:
                        continue
                    transition_penalty = (
                        cfg.center_transition_weight * slope / cfg.max_lateral_slope
                        + cfg.width_transition_weight * width_change / cfg.max_width_change_ratio
                    )
                    fit_score = _candidate_width_fit_score(candidate, slope, cfg)
                    score = previous_score + fit_score - transition_penalty
                    if score > best_score:
                        best_score = score
                        best_path = [*previous_path, (y, candidate)]
            row_states.append((best_score, best_path))
        states.append(row_states)

    completed_paths = [item for row_states in states for item in row_states]
    if not completed_paths:
        return [], 0.0
    _, best_path = max(
        completed_paths,
        key=lambda item: (len(item[1]), item[0] / max(1, len(item[1]))),
    )
    if not best_path:
        return [], 0.0

    fit_scores = []
    for index, (y, candidate) in enumerate(best_path):
        if len(best_path) == 1:
            local_slope = 0.0
        elif index == 0:
            next_y, next_candidate = best_path[index + 1]
            local_slope = abs(next_candidate.center - candidate.center) / max(1, next_y - y)
        else:
            previous_y, previous_candidate = best_path[index - 1]
            local_slope = abs(candidate.center - previous_candidate.center) / max(1, y - previous_y)
        fit_scores.append(_candidate_width_fit_score(candidate, local_slope, cfg))
    fit_score = sum(fit_scores) / len(fit_scores)
    if len(best_path) > 1:
        # Smoothness measures changes in direction, not distance from vertical.
        # The slope bound above still rejects disconnected candidate jumps.
        slopes = [
            (candidate1.center - candidate0.center) / max(1, y1 - y0)
            for (y0, candidate0), (y1, candidate1) in zip(best_path, best_path[1:])
        ]
        smoothness = [
            max(0.0, 1.0 - abs(slope1 - slope0) / cfg.max_lateral_slope)
            for slope0, slope1 in zip(slopes, slopes[1:])
        ]
        smooth_score = sum(smoothness) / len(smoothness) if smoothness else 1.0
    else:
        smooth_score = 0.0
    coverage = len(best_path) / max(1, len(rows))
    weights = (
        cfg.confidence_width_fit_weight,
        cfg.confidence_smoothness_weight,
        cfg.confidence_coverage_weight,
    )
    weight_sum = sum(weights)
    confidence = (
        sum(weight * score for weight, score in zip(weights, (fit_score, smooth_score, coverage))) / weight_sum
        if weight_sum > 0.0
        else 0.0
    )
    return best_path, confidence


def detect_line(frame_bgr: np.ndarray, cfg: VisionConfig) -> tuple[LineFeatures, np.ndarray]:
    height, width = frame_bgr.shape[:2]

    roi_near = _scale_roi(cfg.roi_near, cfg, width, height)
    roi_far = _scale_roi(cfg.roi_far, cfg, width, height)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    blur_ksize = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1
    if blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    _, binary = cv2.threshold(gray, int(cfg.threshold), 255, cv2.THRESH_BINARY_INV)

    ground_points = None
    if len(cfg.ground_roi_norm) == 4:
        ground_points = np.asarray(
            [[round(x * (width - 1)), round(y * (height - 1))] for x, y in cfg.ground_roi_norm],
            dtype=np.int32,
        )
        ground_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(ground_mask, [ground_points], 255)
        binary = cv2.bitwise_and(binary, binary, mask=ground_mask)

    if cfg.morph_ksize >= 3:
        k = int(cfg.morph_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if ground_points is not None:
        scan_start = int(min(point[1] for point in ground_points))
        scan_end = int(max(point[1] for point in ground_points))
    else:
        scan_start = min(roi_far.y, roi_near.y)
        scan_end = max(roi_far.y + roi_far.h - 1, roi_near.y + roi_near.h - 1)
    if cfg.scanline_positions_norm:
        scan_rows = sorted(
            set(
                int(round(scan_start + position * (scan_end - scan_start)))
                for position in cfg.scanline_positions_norm
            )
        )
    else:
        scan_rows = sorted(
            set(
                int(round(y))
                for y in np.linspace(scan_start, scan_end, max(3, cfg.scanline_count))
            )
        )
    row_candidates = []
    for y in scan_rows:
        expected_width = cfg.expected_tape_width_px(y / max(1, height - 1), width)
        row_candidates.append((y, _scanline_candidates(binary[y], expected_width, cfg)))

    path, full_confidence = _scanline_path(row_candidates, cfg)
    candidate_scanlines = sum(bool(candidates) for _, candidates in row_candidates)
    near_rows = [
        (y, candidates)
        for y, candidates in row_candidates
        if roi_near.y <= y < roi_near.y + roi_near.h
    ]
    near_only_path, near_confidence = _scanline_path(near_rows, cfg)
    near_candidates = [
        y for y, candidates in row_candidates
        if candidates and roi_near.y <= y < roi_near.y + roi_near.h
    ]
    far_candidates = [
        y for y, candidates in row_candidates
        if candidates and roi_far.y <= y < roi_far.y + roi_far.h
    ]
    far_zone_candidates = [
        y for y, candidates in row_candidates
        if candidates and y < roi_far.y + roi_far.h
    ]
    middle_candidates = [
        y for y, candidates in row_candidates
        if candidates
        and roi_far.y + roi_far.h <= y < roi_near.y
    ]
    near_path = [candidate for y, candidate in path if roi_near.y <= y < roi_near.y + roi_near.h]
    far_path = [candidate for y, candidate in path if roi_far.y <= y < roi_far.y + roi_far.h]
    covers_both_rois = bool(near_path and far_path)
    full_path_valid = (
        len(path) >= cfg.min_valid_scanlines
        and covers_both_rois
        and full_confidence >= cfg.min_confidence
    )
    near_path_valid = (
        len(near_only_path) >= cfg.min_near_valid_scanlines
        and near_confidence >= cfg.min_confidence
    )

    if candidate_scanlines == 0:
        state = "LOST"
        selected_path = []
        confidence = 0.0
        selected_near_path = []
        selected_far_path = []
    elif not cfg.calibration_ready:
        state = "UNCERTAIN"
        selected_path = path
        confidence = full_confidence
        selected_near_path = near_path
        selected_far_path = far_path
    elif full_path_valid:
        state = "VALID"
        selected_path = path
        confidence = full_confidence
        selected_near_path = near_path
        selected_far_path = far_path
    elif near_path_valid:
        state = "VALID"
        selected_path = near_only_path
        confidence = near_confidence
        selected_near_path = [candidate for _, candidate in near_only_path]
        selected_far_path = []
    else:
        state = "UNCERTAIN"
        selected_path = path
        confidence = full_confidence
        selected_near_path = near_path
        selected_far_path = far_path

    candidate_x_near = (
        sum(candidate.center for candidate in selected_near_path) / len(selected_near_path)
        if selected_near_path
        else None
    )
    candidate_x_far = (
        sum(candidate.center for candidate in selected_far_path) / len(selected_far_path)
        if selected_far_path
        else None
    )
    x_near = candidate_x_near if state == "VALID" else None
    x_far = candidate_x_far if state == "VALID" else None
    centerline = tuple((int(round(candidate.center)), y) for y, candidate in selected_path)
    expected_width_near = cfg.expected_tape_width_px(
        (roi_near.y + roi_near.h / 2.0) / max(1, height - 1),
        width,
    )
    expected_width_near_range = cfg.expected_tape_width_range_px(
        (roi_near.y + roi_near.h / 2.0) / max(1, height - 1),
        width,
    )
    feats = LineFeatures(
        ok_near=bool(near_candidates),
        ok_far=bool(far_candidates),
        x_near=x_near,
        x_far=x_far,
        roi_near=roi_near,
        roi_far=roi_far,
        consistent=state == "VALID",
        state=state,
        confidence=confidence,
        valid_scanlines=len(selected_path),
        candidate_scanlines=candidate_scanlines,
        far_candidate_scanlines=len(far_zone_candidates),
        middle_candidate_scanlines=len(middle_candidates),
        near_candidate_scanlines=len(near_candidates),
        near_path_scanlines=len(near_only_path),
        near_path_confidence=near_confidence,
        centerline=centerline,
        expected_width_near_px=expected_width_near,
        expected_width_near_min_px=(
            None if expected_width_near_range is None else expected_width_near_range[0]
        ),
        expected_width_near_max_px=(
            None if expected_width_near_range is None else expected_width_near_range[1]
        ),
    )

    if not cfg.debug_visualization:
        return feats, frame_bgr.copy()

    debug = frame_bgr.copy()
    cv2.rectangle(debug, (roi_near.x, roi_near.y), (roi_near.x + roi_near.w, roi_near.y + roi_near.h), (0, 255, 255), 2)
    cv2.rectangle(debug, (roi_far.x, roi_far.y), (roi_far.x + roi_far.w, roi_far.y + roi_far.h), (255, 255, 0), 2)
    if ground_points is not None:
        cv2.polylines(debug, [ground_points], True, (255, 0, 255), 2)

    x_ref = width // 2
    cv2.line(debug, (x_ref, 0), (x_ref, height - 1), (255, 0, 255), 1)
    for y, candidates in row_candidates:
        if y >= roi_near.y:
            zone_color = (0, 255, 255)  # near: yellow
        elif y < roi_far.y + roi_far.h:
            zone_color = (255, 255, 0)  # far: cyan
        else:
            zone_color = (255, 0, 255)  # middle: magenta
        cv2.line(debug, (0, y), (width - 1, y), zone_color, 1)
        for candidate in candidates:
            cv2.line(debug, (candidate.left, y), (candidate.right, y), (0, 165, 255), 2)
            cv2.circle(debug, (int(round(candidate.center)), y), 2, zone_color, -1)
    if len(centerline) > 1:
        cv2.polylines(
            debug,
            [np.asarray(centerline, dtype=np.int32)],
            False,
            (0, 220, 0) if state == "VALID" else (0, 165, 255),
            2,
        )
    if candidate_x_near is not None and selected_near_path:
        cv2.circle(debug, (int(candidate_x_near), roi_near.y + roi_near.h // 2), 6, (0, 0, 255), -1)
    if candidate_x_far is not None and selected_far_path:
        cv2.circle(debug, (int(candidate_x_far), roi_far.y + roi_far.h // 2), 6, (255, 255, 0), -1)
    cv2.putText(
        debug,
        f"{state} conf={confidence:.2f} rows={len(path)}/{len(row_candidates)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 0) if state == "VALID" else (0, 165, 255),
        2,
    )
    if expected_width_near_range is not None:
        expected_width_text = (
            f"expected near width={expected_width_near_range[0]:.1f}-"
            f"{expected_width_near_range[1]:.1f}px"
        )
    else:
        expected_width_text = "expected near width=uncalibrated"

    binary_debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.putText(binary_debug, f"BINARY threshold={cfg.threshold}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(
        binary_debug,
        f"scan zones far={len(far_zone_candidates)} mid={len(middle_candidates)} near={len(near_candidates)} "
        f"near_path={len(near_only_path)} near_conf={near_confidence:.2f}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(binary_debug, expected_width_text, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
    return feats, np.concatenate((debug, binary_debug), axis=1)


def normalized_error(x_line: float, x_ref: float, half_width: float) -> float:
    if half_width <= 1e-6:
        return 0.0
    e = (x_line - x_ref) / half_width
    return float(max(-1.5, min(1.5, e)))
