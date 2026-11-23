"""One motion owner per frame; no hardware access."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NavigationConfig:
    track_speed_mps: float
    approach_speed_mps: float
    approach_lookahead_m: float
    max_input_age_s: float
    execution_enabled: bool = False

    def __post_init__(self):
        for name in ("track_speed_mps", "approach_speed_mps", "approach_lookahead_m", "max_input_age_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid navigation.{name}")
        if type(self.execution_enabled) is not bool:
            raise ValueError("navigation.execution_enabled must be boolean")


@dataclass(frozen=True)
class NavigationRequest:
    owner: str = "STOP"
    action: str = "STOP"
    reason: str = "navigation_unconfigured"
    timestamp_s: float | None = None
    track_id: int | None = None
    forward_speed_mps: float = 0.
    yaw_rate_dps: float = 0.
    fault: bool = False


def select_navigation(*, now_s, config, armed, calibration_ready, manual_stop,
                      path, corner_candidate, motion, approach, transition, bypass):
    def request(owner="STOP", action="STOP", reason="", **values):
        return NavigationRequest(owner=owner, action=action, reason=reason, timestamp_s=now_s, **values)

    if manual_stop or not armed:
        return request(reason="navigation_not_armed")
    if not calibration_ready:
        return request(reason="calibration_required")
    if bypass.phase == "FAILED":
        return request(reason=bypass.reason or "obstacle_bypass_failed", fault=True)
    if bypass.stop or (bypass.active and bypass.phase == "STOP"):
        return request(owner="OBSTACLE", reason=bypass.reason or "obstacle_stop")
    if bypass.active:
        return request(owner="BYPASS", action="BYPASS", reason=bypass.reason)
    if transition.phase == "FAILED":
        return request(reason=transition.reason, fault=True)
    if transition.phase in {"TURN", "SETTLE", "EXIT"}:
        if transition.timestamp_s is None or not 0 <= now_s-transition.timestamp_s <= config.max_input_age_s:
            return request(reason="transition_not_fresh", fault=True)
        return request(owner=transition.phase,
            action="SETPOINT" if transition.forward_speed_mps or transition.target_yaw_rate_dps else "STOP",
            reason=transition.reason, track_id=transition.track_id,
            forward_speed_mps=transition.forward_speed_mps, yaw_rate_dps=transition.target_yaw_rate_dps)
    completed = transition.phase == "COMPLETE" and approach.track_id == transition.track_id
    associated_corner = motion.state in {"OBSERVED", "TRACKING", "PREDICTED"} and motion.track_id is not None
    if (corner_candidate or associated_corner) and not completed:
        if (approach.timestamp_s is None or not 0 <= now_s-approach.timestamp_s <= config.max_input_age_s
                or approach.track_id is None or approach.track_id != motion.track_id):
            return request(owner="APPROACH", reason="approach_not_fresh_or_associated")
        if approach.phase != "ADVANCE":
            return request(owner="APPROACH", reason=transition.reason if approach.phase == "TURN_READY"
                           else approach.reason, track_id=approach.track_id)
        if (approach.advance_margin_m is None
                or config.approach_speed_mps*(config.max_input_age_s+min(.01, config.max_input_age_s/4))
                    > approach.advance_margin_m):
            return request(owner="APPROACH", reason="advance_margin_below_next_update_travel",
                           track_id=approach.track_id)
        if motion.incoming_heading_deg is None or motion.corner_position_m is None:
            return request(owner="APPROACH", reason="incoming_path_unavailable")
        theta = math.radians(motion.incoming_heading_deg)
        ux, uy = math.sin(theta), math.cos(theta)
        if uy <= 0:
            return request(owner="APPROACH", reason="incoming_path_not_forward")
        x, y = motion.corner_position_m
        along = x*ux+y*uy
        if along <= 0:
            return request(owner="APPROACH", reason="corner_not_ahead")
        step = min(config.approach_lookahead_m, along)
        tx, ty = x+(step-along)*ux, y+(step-along)*uy
        curvature = 2*tx/(tx*tx+ty*ty)
        speed = config.approach_speed_mps
        return request(owner="APPROACH", action="SETPOINT", reason="bounded_incoming_path",
            track_id=approach.track_id, forward_speed_mps=speed,
            yaw_rate_dps=-math.degrees(speed*curvature))
    if path.state != "VALID" or path.mode != "TRACK" or path.steering_curvature_m_inv is None:
        return request(reason="tracking_path_unavailable")
    # A distance-based speed row must remain measured; obstacle slowing can
    # request another row, but the mapper may reject an unmeasured speed.
    speed = config.track_speed_mps*bypass.speed_limit_scale
    return request(owner="TRACK", action="SETPOINT", reason="ordered_path_tracking",
        forward_speed_mps=speed, yaw_rate_dps=-math.degrees(speed*path.steering_curvature_m_inv))
