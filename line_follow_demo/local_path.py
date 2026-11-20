"""Geometry-only local path assessment; never authorizes motor motion.

Vehicle coordinates use +X right and +Y forward. Positive curvature and
turn angles mean right turns.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from bisect import bisect_right
import math


@dataclass(frozen=True)
class LocalPathPlan:
    state: str = "UNCERTAIN"
    reason: str = "path_unavailable"
    mode: str = "STOP"
    lookahead_point_m: tuple[float, float] | None = None
    steering_curvature_m_inv: float | None = None
    target_heading_deg: float | None = None
    corner_point_m: tuple[float, float] | None = None
    corner_distance_m: float | None = None
    corner_curvature_m_inv: float | None = None
    corner_turn_deg: float | None = None
    nearest_observed_point_m: tuple[float, float] | None = None
    lateral_error_m: float | None = None
    unobserved_distance_m: float | None = None
    mapped_path_complete: bool = True
    near_heading_deg: float | None = None
    near_heading_support_m: float | None = None
    near_heading_variation_deg: float | None = None


@dataclass(frozen=True)
class GroundCornerGeometry:
    state: str = "UNCERTAIN"
    reason: str = "corner_unavailable"
    corner_point_m: tuple[float, float] | None = None
    turn_deg: float | None = None
    incoming_heading_deg: float | None = None
    outgoing_heading_deg: float | None = None
    fit_rms_m: float | None = None
    incoming_span_m: float | None = None
    outgoing_span_m: float | None = None


def _fit_leg(points, max_fit_rms_m):
    if len(points) < 3 or math.dist(points[0], points[-1]) < 1e-6:
        return None
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    xx = sum((x-cx)**2 for x, _ in points)
    yy = sum((y-cy)**2 for _, y in points)
    xy = sum((x-cx)*(y-cy) for x, y in points)
    angle = .5 * math.atan2(2*xy, xx-yy)
    dx, dy = math.cos(angle), math.sin(angle)
    if dx*(points[-1][0]-points[0][0]) + dy*(points[-1][1]-points[0][1]) < 0:
        dx, dy = -dx, -dy
    rms = math.sqrt(sum(((x-cx)*dy-(y-cy)*dx)**2 for x, y in points)/len(points))
    span = math.dist(points[0], points[-1])
    return (cx, cy, dx, dy, rms, span) if rms <= max_fit_rms_m else None


def estimate_ground_corner(incoming_px, outgoing_px, *, mapper, frame_width,
                           frame_height, pan_pulse_us, tilt_pulse_us,
                           min_leg_span_m, max_fit_rms_m) -> GroundCornerGeometry:
    """Fit two observed legs inside the measured hull and intersect them."""
    if min_leg_span_m <= 0 or max_fit_rms_m <= 0:
        raise ValueError("corner fit limits must be positive")
    legs = []
    for pixels in (incoming_px, outgoing_px):
        mapped = []
        for a, b in zip(pixels, pixels[1:]):
            steps = max(1, math.ceil(math.dist(a, b)))
            for index in range(steps + 1):
                fraction = index/steps
                point = mapper.map_image_point(
                    a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1]),
                    frame_width=frame_width, frame_height=frame_height,
                    pan_pulse_us=pan_pulse_us, tilt_pulse_us=tilt_pulse_us)
                if point is not None and (not mapped or math.dist(mapped[-1], point) > 1e-6):
                    mapped.append(point)
        fit = _fit_leg(mapped, max_fit_rms_m)
        if fit is None:
            return GroundCornerGeometry(reason="insufficient_calibrated_leg_support")
        legs.append(fit)
    ax, ay, ux, uy, rms_a, span_a = legs[0]
    bx, by, vx, vy, rms_b, span_b = legs[1]
    cross = ux*vy-uy*vx
    if abs(cross) < 1e-6:
        return GroundCornerGeometry(reason="legs_parallel")
    distance = ((bx-ax)*vy-(by-ay)*vx)/cross
    corner = ax+distance*ux, ay+distance*uy
    vertex_px = incoming_px[-1]
    observed = mapper.map_image_point(*vertex_px, frame_width=frame_width,
        frame_height=frame_height, pan_pulse_us=pan_pulse_us, tilt_pulse_us=tilt_pulse_us)
    if observed is None or math.dist(corner, observed) > max_fit_rms_m:
        return GroundCornerGeometry(reason="intersection_not_at_observed_vertex")
    enough_support = min(span_a, span_b) >= min_leg_span_m
    return GroundCornerGeometry(
        state="VALID" if enough_support else "UNCERTAIN",
        reason="ok" if enough_support else "short_calibrated_leg_support",
        corner_point_m=corner,
        turn_deg=math.degrees(math.atan2(uy*vx-ux*vy, ux*vx+uy*vy)),
        incoming_heading_deg=math.degrees(math.atan2(ux, uy)),
        outgoing_heading_deg=math.degrees(math.atan2(vx, vy)),
        fit_rms_m=max(rms_a, rms_b), incoming_span_m=span_a, outgoing_span_m=span_b)


def plan_local_path(
    path_points_m: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    *,
    branch_count: int,
    lookahead_m: float,
    max_track_curvature_m_inv: float,
    max_anchor_offset_m: float,
    max_segment_m: float,
    max_unobserved_distance_m: float = 0.0,
) -> LocalPathPlan:
    if branch_count != 1:
        return LocalPathPlan(reason="branch_ambiguous" if branch_count > 1 else "path_unavailable")
    limits = (lookahead_m, max_track_curvature_m_inv, max_anchor_offset_m, max_segment_m)
    if not all(math.isfinite(value) and value > 0 for value in limits):
        raise ValueError("local path limits must be positive finite numbers")
    if not math.isfinite(max_unobserved_distance_m) or max_unobserved_distance_m < 0:
        raise ValueError("unobserved distance limit must be finite and nonnegative")

    points: list[tuple[float, float]] = []
    for point in path_points_m:
        if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
            return LocalPathPlan(reason="path_point_invalid")
        x, y = float(point[0]), float(point[1])
        if y < 0:
            return LocalPathPlan(reason="path_behind_vehicle")
        if not points or math.dist(points[-1], (x, y)) > 1e-6:
            points.append((x, y))
    if len(points) < 3:
        return LocalPathPlan(reason="path_too_short")

    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    if any(length > max_segment_m for length in lengths):
        return LocalPathPlan(reason="path_disconnected")
    progress = [0.0]
    for length in lengths:
        progress.append(progress[-1] + length)

    nearest = (math.inf, 0.0, 0, (0.0, 0.0), 0.0)
    for index, (a, b) in enumerate(zip(points, points[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        raw_fraction = -(a[0] * dx + a[1] * dy) / lengths[index] ** 2
        fraction = max(0.0, min(1.0, raw_fraction))
        projection = (a[0] + fraction * dx, a[1] + fraction * dy)
        offset = math.hypot(*projection)
        if offset < nearest[0]:
            nearest = offset, progress[index] + fraction * lengths[index], index, projection, raw_fraction
    _, anchor_progress, anchor_index, anchor_point, raw_fraction = nearest
    a, b = points[anchor_index:anchor_index + 2]
    lateral_error = (a[0]*(b[1]-a[1])-a[1]*(b[0]-a[0]))/lengths[anchor_index]
    # The unseen prefix remains an unobserved distance, even when its
    # supporting straight line passes exactly through the control centre.
    unobserved = max(0.0, -raw_fraction*lengths[0]) if anchor_index == 0 else 0.0
    evidence = dict(nearest_observed_point_m=anchor_point, lateral_error_m=lateral_error,
                    unobserved_distance_m=unobserved)

    def at_distance(distance):
        index = min(len(lengths)-1, max(0, bisect_right(progress, distance)-1))
        fraction = (distance-progress[index])/lengths[index]
        a, b = points[index:index+2]
        return (a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1])), index

    target_progress = anchor_progress + max(0.0, lookahead_m-unobserved)
    target, target_segment = at_distance(min(target_progress, progress[-1]))
    radius_squared = target[0] ** 2 + target[1] ** 2
    if radius_squared < 1e-8:
        return LocalPathPlan(reason="lookahead_at_vehicle", **evidence)
    steering_curvature = 2.0 * target[0] / radius_squared
    tangent = (points[target_segment + 1][0] - points[target_segment][0],
               points[target_segment + 1][1] - points[target_segment][1])
    target_heading = math.degrees(math.atan2(tangent[0], tangent[1]))

    sharpest = (0.0, None, None, None, None)
    first_corner = None
    window = lookahead_m/3
    for index in range(1, len(points) - 1):
        if not anchor_progress < progress[index]:
            continue
        previous, b, following = points[index - 1:index + 2]
        local_cross = ((b[1]-previous[1])*(following[0]-b[0])
                       -(b[0]-previous[0])*(following[1]-b[1]))
        if abs(local_cross) <= 1e-10*lengths[index-1]*lengths[index]:
            continue
        local_dot = ((b[0]-previous[0])*(following[0]-b[0])
                     +(b[1]-previous[1])*(following[1]-b[1]))
        local_turn = math.atan2(local_cross, local_dot)
        a, _ = at_distance(max(0.0, progress[index]-window))
        c, _ = at_distance(min(progress[-1], progress[index]+window))
        cross = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        chord = math.dist(a, c)
        if chord < 1e-6:
            continue
        curvature = 2.0 * cross / (math.dist(a, b) * math.dist(b, c) * chord)
        # Fixed arc-length support avoids changing sharpness when a segment
        # is subdivided. Distinguish the actual heading discontinuity from
        # nearby low-angle vertices whose windows also straddle the bend.
        local_strength = 2*abs(math.sin(local_turn/2))/window
        if (first_corner is None and abs(curvature) > max_track_curvature_m_inv
                and local_strength > max_track_curvature_m_inv):
            first_corner = (curvature, b, progress[index]-anchor_progress+unobserved,
                            following, math.degrees(local_turn))
        if abs(curvature) > abs(sharpest[0]):
            dot = ((b[0] - a[0]) * (c[0] - b[0])
                   + (b[1] - a[1]) * (c[1] - b[1]))
            sharpest = (curvature, b, progress[index] - anchor_progress + unobserved,
                        c, math.degrees(math.atan2(cross, dot)))

    near_end = min(anchor_progress+window, progress[-1])
    if first_corner is not None:
        near_end = min(near_end, anchor_progress+max(0., first_corner[2]-unobserved))
    near_a, _ = at_distance(anchor_progress)
    near_b, _ = at_distance(near_end)
    near_support = math.dist(near_a, near_b)
    if near_support > 1e-6:
        near_heading = math.degrees(math.atan2(near_b[0]-near_a[0], near_b[1]-near_a[1]))
        variations = []
        for i, (a, b) in enumerate(zip(points, points[1:])):
            if progress[i] < near_end and progress[i+1] > anchor_progress:
                heading = math.degrees(math.atan2(b[0]-a[0], b[1]-a[1]))
                variations.append(abs((heading-near_heading+180) % 360-180))
        evidence.update(near_heading_deg=near_heading, near_heading_support_m=near_support,
                        near_heading_variation_deg=max(variations, default=0.))

    corner_curvature, corner_point, corner_distance, outgoing_point, corner_turn = first_corner or sharpest
    pivot = (abs(corner_curvature) > max_track_curvature_m_inv
             and corner_distance <= lookahead_m+1e-9)
    if pivot:
        outgoing = (outgoing_point[0] - corner_point[0], outgoing_point[1] - corner_point[1])
        target_heading = math.degrees(math.atan2(outgoing[0], outgoing[1]))
    reason = "ok"
    if abs(lateral_error) > max_anchor_offset_m:
        reason = "path_not_at_vehicle"
    elif unobserved > max_unobserved_distance_m+1e-9:
        reason = "unobserved_path_exceeds_limit"
    elif unobserved >= lookahead_m:
        reason = "lookahead_not_observed"
    elif target_progress > progress[-1]+1e-9:
        reason = "lookahead_not_visible"
    return LocalPathPlan(
        state="VALID" if reason == "ok" else "UNCERTAIN",
        reason=reason,
        mode=("PIVOT_REQUIRED" if pivot else "TRACK") if reason == "ok" else "STOP",
        lookahead_point_m=target,
        steering_curvature_m_inv=steering_curvature,
        target_heading_deg=target_heading,
        corner_point_m=corner_point,
        corner_distance_m=corner_distance,
        corner_curvature_m_inv=corner_curvature if corner_point is not None else None,
        corner_turn_deg=corner_turn,
        **evidence,
    )


def plan_image_path(
    path_points_px: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    *,
    mapper,
    frame_width: int,
    frame_height: int,
    pan_pulse_us: int | None,
    tilt_pulse_us: int | None,
    branch_count: int,
    lookahead_m: float,
    max_track_curvature_m_inv: float,
    max_anchor_offset_m: float,
    max_segment_m: float,
    front_to_control_center_m: float | None = None,
    max_unobserved_distance_m: float = 0.0,
) -> LocalPathPlan:
    if not path_points_px:
        return LocalPathPlan(reason="path_unavailable")
    if front_to_control_center_m is None:
        return LocalPathPlan(reason="vehicle_reference_unconfigured")
    if (isinstance(front_to_control_center_m, bool)
            or not math.isfinite(front_to_control_center_m) or front_to_control_center_m < 0):
        raise ValueError("front-to-control-centre offset must be finite and nonnegative")
    mapped = []
    missing_after_start = False
    complete = True
    for x, y in path_points_px:
        point = mapper.map_image_point(
            x, y, frame_width=frame_width, frame_height=frame_height,
            pan_pulse_us=pan_pulse_us, tilt_pulse_us=tilt_pulse_us,
        )
        if point is None:
            complete = False
            missing_after_start |= bool(mapped)
            continue
        if missing_after_start:
            return LocalPathPlan(reason="calibrated_path_split", mapped_path_complete=False)
        mapped.append((point[0], point[1]+front_to_control_center_m))
    if not mapped:
        return LocalPathPlan(reason="ground_mapping_unavailable", mapped_path_complete=False)
    plan = plan_local_path(
        mapped, branch_count=branch_count, lookahead_m=lookahead_m,
        max_track_curvature_m_inv=max_track_curvature_m_inv,
        max_anchor_offset_m=max_anchor_offset_m, max_segment_m=max_segment_m,
        max_unobserved_distance_m=max_unobserved_distance_m,
    )
    return replace(plan, mapped_path_complete=complete)
