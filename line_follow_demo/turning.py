from __future__ import annotations

import math


def path_lookahead_error(centerline, *, near_y, far_y, far_line_weight,
                         reference_x, half_width):
    points = sorted((float(y), float(x)) for x, y in centerline)
    target_y = near_y + max(0.0, min(0.5, far_line_weight)) * (far_y-near_y)
    target_x = points[-1][1]
    if target_y <= points[0][0]:
        target_x = points[0][1]
    else:
        for (y0, x0), (y1, x1) in zip(points, points[1:]):
            if y0 <= target_y <= y1:
                target_x = x0 + (x1-x0)*(target_y-y0)/(y1-y0)
                break
    return (target_x-reference_x)/half_width


def combine_line_errors(
    near_error: float,
    far_error: float | None,
    far_line_weight: float,
) -> float:
    if not math.isfinite(near_error):
        raise ValueError("near_error must be finite")
    if far_error is None or not math.isfinite(far_error):
        return near_error

    weight = float(far_line_weight)
    if not math.isfinite(weight):
        weight = 0.0
    weight = max(0.0, min(0.5, weight))
    return near_error + weight * (far_error - near_error)


def line_offset_to_turn_input(line_offset: float) -> float:
    if not math.isfinite(line_offset):
        raise ValueError("line_offset must be finite")
    return -float(line_offset)


def heading_to_turn_input(heading_deg: float | None) -> float:
    if heading_deg is None:
        return 0.0
    if not math.isfinite(heading_deg):
        raise ValueError("heading_deg must be finite")
    return -math.radians(float(heading_deg))


def combine_pid_and_heading(
    pid_output: float,
    heading_deg: float | None,
    heading_gain: float,
    heading_feedback_limit: float = 1.0,
) -> tuple[float, float]:
    if not math.isfinite(pid_output):
        raise ValueError("pid_output must be finite")
    if not math.isfinite(heading_gain) or heading_gain < 0.0:
        raise ValueError("heading_gain must be finite and non-negative")
    if (
        not math.isfinite(heading_feedback_limit)
        or heading_feedback_limit < 0.0
        or heading_feedback_limit > 1.0
    ):
        raise ValueError("heading_feedback_limit must be between 0 and 1")

    raw_heading_feedback = heading_to_turn_input(heading_deg) * float(heading_gain)
    heading_feedback = max(
        -heading_feedback_limit,
        min(heading_feedback_limit, raw_heading_feedback),
    )
    steering_output = max(-1.0, min(1.0, float(pid_output) + heading_feedback))
    return steering_output, heading_feedback


def mix_forward_turn_duty(base_duty: float, turn_duty: float) -> tuple[float, float, float, float]:
    left = base_duty - turn_duty
    right = base_duty + turn_duty
    return left, right, left, right


def mix_forward_turn_duty_scaled(
    base_duty: float,
    turn_duty: float,
    forward_scale: float,
    *,
    allow_inner_reverse: bool = False,
) -> tuple[float, float, float, float]:
    values = (float(base_duty), float(turn_duty), float(forward_scale))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("base duty, turn duty and forward scale must be finite")
    scaled_base = values[0] * values[2]
    if not 0.0 <= values[2] <= 1.0:
        raise ValueError("forward_scale must be between 0 and 1")
    bounded_turn = values[1] if allow_inner_reverse else max(-abs(scaled_base), min(abs(scaled_base), values[1]))
    return mix_forward_turn_duty(scaled_base, bounded_turn)
