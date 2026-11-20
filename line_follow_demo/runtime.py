from __future__ import annotations

import copy
import json
import logging
import math
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import Camera, CameraConfig
from .gimbal import (
    PWM_PULSE_MAX_US,
    PWM_PULSE_MIN_US,
    TILT_PULSE_MAX_US,
    TILT_PULSE_MIN_US,
    PwmGimbalController,
    gimbal_config_from_dict,
    validate_gimbal_position,
)
from .mecanum import MecanumGeometry, mecanum_inverse, normalize_wheels
from .motor import MotorDriver, create_motor_driver
from .imu import IMUError
from .imu_observer import IMUObserver, imu_observer_config_from_dict
from .ground_calibration import GroundPlaneMapper, load_ground_calibration
from .line_sensor_sampling import LineSamplingConfig, LineSensorSampler
from .corner_motion import CornerMotionConfig, CornerMotionEstimate, CornerMotionTracker
from .corner_drive import CornerDriveConfig, CornerDriveMapper, CornerDriveCommand
from .corner_position import CornerPositionConfig, CornerPositionEstimate, fuse_corner_position
from .navigation import NavigationConfig, NavigationRequest, select_navigation
from .obstacle_bypass import ObstacleBypassCommand
from .motor import model_a_config_from_dict
from .motor.hiwonder_model_a import model_a_wheel_commands
from .corner_approach import CornerApproachConfig, CornerApproachDecision, decide_corner_approach
from .corner_transition import (CornerTransitionConfig, CornerTransitionController,
                                CornerTransitionDecision, exit_path_evidence)
from .position_evidence import (LinePoseConfig, LinePoseEvidence, SensorContactConfig,
                                SensorContactEvidence, SensorContactObserver, estimate_image_line_pose)
from .local_path import GroundCornerGeometry, LocalPathPlan, estimate_ground_corner, plan_image_path
from .path_geometry import PathGeometryEstimator, path_geometry_config_from_dict
from .pid import PID, PIDConfig
from .sensors import DistanceSensor, I2CLineFollowerSensor, create_distance_sensor, create_line_sensor
from .speed_control import (
    CurveSpeedController,
    speed_control_config_from_dict,
)
from .tracking import TemporalPathFilter
from .obstacle_bypass import (
    DistanceFilter,
    ObstacleBypassConfig,
    ObstacleBypassController,
)
from .right_angle import (
    RightAngleConfig,
    RightAngleController,
    CornerCommand,
    detect_right_angle,
    CornerObservation,
)
from .stop_marker import StopMarkerController, StopMarkerObservation, StopMarkerCommand, detect_stop_marker
from .test_runs import TestRunConfig, TestRunRecorder
from .turning import (
    combine_line_errors,
    path_lookahead_error,
    combine_pid_and_heading,
    heading_to_turn_input,
    line_offset_to_turn_input,
    mix_forward_turn_duty_scaled,
    mix_forward_turn_duty,
)
from .vision import detect_line, normalized_error, vision_config_from_dict
from .vision_diagnostics import VisionDiagnosticConfig, VisionDiagnosticRecorder
from .yaw_rate_feedback import (
    YawFeedbackError, YawFeedbackResult, YawRateFeedback, forward_motor_profile_key, yaw_feedback_config_from_dict,
)


log = logging.getLogger("line_follow_demo.runtime")


def _ground_debug_panel(estimate, bounds, *, width: int, height: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(panel, "GROUND PLANE / m", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, f"{estimate.state}: {estimate.reason.replace('_', ' ')[:22]}",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                (0, 220, 0) if estimate.state == "VALID" else (0, 165, 255), 1)
    if bounds is None:
        cv2.putText(panel, "NO MEASURED MAP", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 165, 255), 1)
        cv2.putText(panel, "X: chassis right", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
        cv2.putText(panel, "Y: chassis forward", (18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
        return panel

    x_min, x_max, y_min, y_max = bounds
    if x_max <= x_min or y_max <= y_min:
        cv2.putText(panel, "INVALID MEASURED EXTENT", (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
        return panel
    left, top, right, bottom = 28, 72, width - 16, height - 48
    cv2.rectangle(panel, (left, top), (right, bottom), (90, 90, 90), 1)

    def to_pixel(point: tuple[float, float]) -> tuple[int, int]:
        x_value, y_value = point
        x_pixel = left + round((x_value - x_min) / (x_max - x_min) * (right - left))
        y_pixel = bottom - round((y_value - y_min) / (y_max - y_min) * (bottom - top))
        return x_pixel, y_pixel

    origin = to_pixel((0.0, 0.0))
    cv2.drawMarker(panel, origin, (255, 120, 0), cv2.MARKER_CROSS, 10, 2)
    cv2.putText(panel, "front", (min(width - 42, origin[0] + 4), max(64, origin[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 160, 40), 1)

    if len(estimate.mapped_points_m) > 1:
        observed = np.asarray([to_pixel(point) for point in estimate.mapped_points_m], dtype=np.int32)
        cv2.polylines(panel, [observed], False, (0, 165, 255), 1)
    for point in estimate.mapped_points_m:
        cv2.circle(panel, to_pixel(point), 3, (0, 220, 255), -1)
    if estimate.state == "VALID" and len(estimate.fit_points_m) > 1:
        fit = np.asarray([to_pixel(point) for point in estimate.fit_points_m], dtype=np.int32)
        cv2.polylines(panel, [fit], False, (0, 220, 0), 2)
    if estimate.predicted_point_m is not None:
        cv2.drawMarker(panel, to_pixel(estimate.predicted_point_m), (0, 0, 255), cv2.MARKER_DIAMOND, 12, 2)
        cv2.putText(panel, "front path", (8, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(panel, f"X {x_min:+.2f}..{x_max:+.2f}", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
    cv2.putText(panel, f"Y 0..{y_max:.2f}", (8, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
    return panel


def _camera_config(data: dict[str, Any]) -> CameraConfig:
    device = data.get("device", data.get("index", 0))
    if not isinstance(device, str) or device.isdigit():
        device = int(device)
    return CameraConfig(
        device=device,
        width=None if data.get("width") is None else int(data.get("width", 640)),
        height=None if data.get("height") is None else int(data.get("height", 480)),
        fps=None if data.get("fps") is None else int(data.get("fps", 30)),
        reconnect_interval_s=max(1.0, float(data.get("reconnect_interval_s", 2.0))),
    )


def _pid_config(data: dict[str, Any]) -> PIDConfig:
    return PIDConfig(
        kp=float(data.get("kp", 1.0)),
        ki=float(data.get("ki", 0.0)),
        kd=float(data.get("kd", 0.15)),
        integral_limit=float(data.get("integral_limit", 0.5)),
        output_limit=float(data.get("output_limit", 1.0)),
        derivative_filter_alpha=float(data.get("derivative_filter_alpha", 1.0)),
        output_smoothing_alpha=float(data.get("output_smoothing_alpha", 1.0)),
        large_error_threshold=float(data.get("large_error_threshold", 1.0)),
        large_error_gain=float(data.get("large_error_gain", 1.0)),
    )


class TrackingRuntime:
    def __init__(self, config_path: str | Path):
        self._config_path = Path(config_path).resolve()
        self._config = json.loads(self._config_path.read_text(encoding="utf-8-sig"))
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._motor_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._running = False
        self._armed = False
        self._command_generation = 0
        self._tracking_active = False
        self._acquire_count = 0
        self._near_lost_frames = 0
        self._tracking_seen_since_arm = False
        self._frame_id = 0
        self._jpeg: bytes | None = None
        self._raw_snapshot = None
        self._camera: Camera | None = None
        self._motor: MotorDriver | None = None
        self._distance_sensor: DistanceSensor | None = None
        self._line_sensor: I2CLineFollowerSensor | None = None
        self._line_sampler: LineSensorSampler | None = None
        sampling_config = self._config.get("line_sensor", {}).get("sampling")
        self._line_sampling_config = LineSamplingConfig(**sampling_config) if sampling_config is not None else None
        if self._line_sampling_config is not None and not self._config.get("line_sensor", {}).get("enabled"):
            raise ValueError("Independent line sampling requires an enabled line sensor")
        camera_values = self._config.get("camera", {})
        self._line_frame_size = (camera_values.get("width", 640), camera_values.get("height", 480))
        self._sensor_contact_latest = SensorContactEvidence()
        self._motor_active = False
        self._last_drive_trace_at = 0.0
        self._battery_voltage_v: float | None = None
        self._battery_voltage_updated_at: float | None = None
        self._temporal_tracker = self._new_temporal_tracker(self._config)
        path_geometry_config = path_geometry_config_from_dict(
            self._config.get("path_geometry", {})
        )
        self._path_geometry = PathGeometryEstimator(path_geometry_config)
        self._curve_speed_controller = CurveSpeedController(
            speed_control_config_from_dict(
                self._config.get("control", {}).get("speed_control", {}),
                curve_threshold_px_inv=path_geometry_config.curve_threshold_px_inv,
            )
        )
        self._gimbal = PwmGimbalController(
            gimbal_config_from_dict(self._config.get("gimbal", {}))
        )
        self._pid = PID(_pid_config(self._config.get("control", {}).get("pid", {})))
        self._imu = IMUObserver(
            imu_observer_config_from_dict(self._config.get("imu", {})),
            config_directory=self._config_path.parent,
        )
        self._manual_stop_pending = False
        self._corner = RightAngleController(RightAngleConfig(**self._config.get("right_angle", {})))
        self._stop_marker = StopMarkerController(self._config.get("stop_marker", {}))
        if self._corner.config.enabled and (not self._imu.config.enabled or self._config.get("motor", {}).get("mixer", "duty") != "duty"):
            raise ValueError("Right-angle turns require verified IMU sampling and the duty mixer")
        self._bypass_config = ObstacleBypassConfig.from_obstacle_config(
            self._config.get("obstacle", {})
        )
        if (self._config.get("obstacle", {}).get("enabled", False)
                and self._bypass_config.enabled and not self._imu.config.enabled):
            raise ValueError("Obstacle bypass requires verified IMU sampling")
        self._bypass = ObstacleBypassController(self._bypass_config)
        self._distance_filter = DistanceFilter(
            min_valid_mm=self._bypass_config.min_valid_distance_mm,
            max_spread_mm=self._bypass_config.max_spread_mm,
            required_samples=2,
        )
        self._last_bypass_phase = "IDLE"
        self._test_runs = TestRunRecorder(TestRunConfig(**self._config.get("test_recording", {})),
                                         config_directory=self._config_path.parent)
        ground_calibration_path = self._config_path.with_name("ground_distance_calibration.json")
        self._ground_mapper = GroundPlaneMapper(load_ground_calibration(ground_calibration_path))
        motion_config = self._config.get("corner_motion")
        self._corner_motion = CornerMotionTracker(CornerMotionConfig(**motion_config)) if motion_config else None
        approach_config = self._config.get("corner_approach")
        self._corner_approach_config = CornerApproachConfig(**approach_config) if approach_config else None
        if self._corner_approach_config is not None and self._corner_motion is None:
            raise ValueError("Corner approach decisions require the corner motion observer")
        transition_config = self._config.get("corner_transition")
        self._corner_transition = (CornerTransitionController(CornerTransitionConfig(**transition_config))
                                   if transition_config else None)
        if self._corner_transition is not None and (
                self._corner_approach_config is None or not self._config.get("local_path")):
            raise ValueError("Corner transition requires approach decisions and local_path planning")
        drive_config = self._config.get("corner_drive")
        self._corner_drive = CornerDriveMapper(CornerDriveConfig.from_dict(drive_config)) if drive_config else None
        if self._corner_drive is not None and self._corner_transition is None:
            raise ValueError("Corner drive mapping requires corner transition decisions")
        navigation_config = self._config.get("navigation")
        self._navigation_config = NavigationConfig(**navigation_config) if navigation_config else None
        if self._navigation_config is not None and (self._corner_drive is None or self._corner_transition is None):
            raise ValueError("Unified navigation requires transition and drive mapping")
        self._navigation_latest = NavigationRequest()
        self._navigation_lost_frames = 0
        self._navigation_watchdog = None
        self._navigation_dispatched_at_s = None
        evidence_config = self._config.get("position_evidence", {})
        sensor_config = evidence_config.get("line_sensor")
        line_pose_config = evidence_config.get("line_pose")
        self._sensor_contact = SensorContactObserver(SensorContactConfig(**sensor_config)) if sensor_config else None
        fusion_config = evidence_config.get("fusion")
        self._corner_position_config = CornerPositionConfig(**fusion_config) if fusion_config else None
        if self._corner_position_config is not None:
            if self._corner_motion is None or self._sensor_contact is None:
                raise ValueError("Position fusion requires corner motion and line sensor geometry")
            if self._line_sampling_config is None:
                raise ValueError("Position fusion requires independently timestamped line sampling")
            if self._corner_position_config.max_yaw_rate_dps < self._corner.config.max_rate_dps:
                raise ValueError("Position fusion yaw bound must cover sensor sampling yaw bound")
        self._line_pose_config = LinePoseConfig(**line_pose_config) if line_pose_config else None
        if self._sensor_contact is not None and self._corner_motion is None:
            raise ValueError("Sensor contact geometry requires the corner motion observer")
        feedback_path = self._config_path.with_name("yaw_rate_feedback.json")
        feedback_values = json.loads(feedback_path.read_text(encoding="utf-8-sig")) if feedback_path.exists() else {}
        self._yaw_feedback = YawRateFeedback(yaw_feedback_config_from_dict(feedback_values))
        if self._yaw_feedback.config.enabled != (self._imu.config.mode == "feedback"):
            raise ValueError("Enable yaw_rate_feedback.json and imu.mode=feedback together; calibration is required")
        if self._yaw_feedback.config.enabled and self._config.get("motor", {}).get("mixer", "duty") != "duty":
            raise ValueError("Yaw feedback currently supports only the verified duty mixer")
        self._vision_capture = VisionDiagnosticRecorder(
            VisionDiagnosticConfig(**self._config.get("vision_diagnostics", {})),
            config_directory=self._config_path.parent,
            vision_source=Path(__file__).with_name("vision.py"),
        )
        self._status: dict[str, Any] = {
            "state": "STOPPED",
            "armed": False,
            "calibration_ready": False,
            "camera_ok": False,
            "line_near": False,
            "line_far": False,
            "x_near": None,
            "x_far": None,
            "path_midpoint_x": None,
            "path_midpoint_y": None,
            "error": None,
            "track_heading_deg": None,
            "path_heading_source": "NONE",
            "heading_feedback": None,
            "path_curvature_px_inv": None,
            "curve_direction": "UNKNOWN",
            "path_fit_rms_px": None,
            "curve_speed_enabled": self._curve_speed_controller.config.enabled,
            "curve_speed_scale": self._curve_speed_controller.current_scale,
            "curve_speed_target_scale": self._curve_speed_controller.current_scale,
            "gimbal_enabled": self._gimbal.config.enabled,
            "gimbal_state": "NOT_HOMED" if self._gimbal.config.enabled else "DISABLED",
            "gimbal_message": "",
            "gimbal_pan_pulse_us": self._gimbal.config.pan_pulse_us,
            "gimbal_tilt_pulse_us": self._gimbal.config.tilt_pulse_us,
            "ground_path_state": "UNCERTAIN",
            "ground_path_reason": self._ground_mapper.reason,
            "ground_nearest_observation_m": None,
            "ground_lateral_error_m": None,
            "ground_heading_error_deg": None,
            "ground_fit_rms_m": None,
            "ground_extrapolation_m": None,
            "ground_path_points_m": [],
            "ground_path_predicted": False,
            "ground_corner_geometry": asdict(GroundCornerGeometry()),
            "corner_motion": asdict(CornerMotionEstimate()),
            "corner_approach": asdict(CornerApproachDecision()),
            "corner_transition": asdict(CornerTransitionDecision()),
            "corner_drive": asdict(CornerDriveCommand()),
            "corner_position": asdict(CornerPositionEstimate()),
            "navigation": asdict(self._navigation_latest),
            "sensor_contact": asdict(SensorContactEvidence()),
            "line_pose": asdict(LinePoseEvidence()),
            "local_path_plan": asdict(LocalPathPlan(reason="planner_limits_unconfigured")),
            "local_path_state": "UNCERTAIN",
            "local_path_reason": "planner_limits_unconfigured",
            "local_path_mode": "STOP",
            "local_path_corner_distance_m": None,
            "local_path_corner_turn_deg": None,
            "local_path_steering_curvature_m_inv": None,
            "distance_mm": None,
            "distance_mm_filtered": None,
            "obstacle_bypass_mode": "INACTIVE",
            "obstacle_bypass_phase": "IDLE",
            "obstacle_bypass_progress_deg": None,
            "obstacle_bypass_reason": "",
            "obstacle_bypass_attempts": 0,
            "obstacle_bypass_line_candidate": False,
            "line_sensor_channels": None,
            "line_sensor_samples": [],
            "line_sensor_dropped_samples": 0,
            "sensor_contact_samples": [],
            "battery_voltage_v": None,
            "battery_voltage_age_s": None,
            "fps": 0.0,
            "motor_stop_error": None,
            "forward_duty_command": 0.0,
            "turn_duty_command": 0.0,
            "wheel_duty_commands": {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0},
            "imu_feedback_enabled": False,
            "visual_turn": None,
            "target_yaw_rate_dps": None,
            "yaw_rate_error_dps": None,
            "imu_turn_correction": None,
            "message": "Not started",
        }

    @property
    def config(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._config)

    @staticmethod
    def _new_temporal_tracker(config: dict[str, Any]) -> TemporalPathFilter:
        control = config.get("control", {})
        return TemporalPathFilter(
            max_center_shift_norm_per_s=float(control.get("temporal_max_center_shift_norm_per_s", 0.35)),
            confirm_frames=max(1, int(control.get("line_acquire_frames", 3))),
            max_uncertain_frames=max(0, int(control.get("temporal_max_uncertain_frames", 2))),
        )

    def start(self, *, auto_start: bool = False) -> None:
        with self._lock:
            if self._running:
                return

        if self._line_sampler is not None:
            self._close_devices()
            if self._line_sampler is not None:
                raise RuntimeError("Previous line-sensor read is still stopping")
        self._camera = Camera(_camera_config(self._config.get("camera", {})))
        self._motor = create_motor_driver(self._config.get("motor", {}))
        ultrasonic_config = self._config.get("ultrasonic", {})
        self._distance_sensor = create_distance_sensor(
            ultrasonic_config if ultrasonic_config.get("enabled", False) else self._config.get("obstacle", {})
        )
        self._line_sensor = create_line_sensor(self._config.get("line_sensor", {}))

        try:
            self._camera.open()
            self._motor.open()
            self._motor.stop()
            gimbal_state = self._move_gimbal_home_at_start()
            self._distance_sensor.open()
            if self._line_sensor is not None:
                self._line_sensor.open()
        except Exception:
            self._close_devices()
            raise

        calibration_ready = vision_config_from_dict(self._config.get("vision", {})).calibration_ready
        self._imu.start()
        self._vision_capture.start()
        self._test_runs.start()
        with self._lock:
            self._running = True
            self._armed = bool(auto_start and calibration_ready)
            self._command_generation += 1
            self._tracking_active = False
            self._acquire_count = 0
            self._near_lost_frames = 0
            self._tracking_seen_since_arm = False
            self._temporal_tracker.reset()
            self._reset_obstacle_bypass("runtime_started")
            self._status.update(
                state=("WAITING_LINE" if self._armed else "PREVIEW") if calibration_ready else "CALIBRATION_REQUIRED",
                armed=self._armed,
                calibration_ready=calibration_ready,
                camera_ok=True,
                path_midpoint_x=None,
                path_midpoint_y=None,
                forward_duty_command=0.0,
                turn_duty_command=0.0,
                wheel_duty_commands={"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0},
                gimbal_enabled=self._gimbal.config.enabled,
                gimbal_state=gimbal_state,
                gimbal_message=(
                    "Startup PWM target sent; mechanical camera angle is not measured"
                    if gimbal_state == "HOME_COMMAND_SENT"
                    else ""
                ),
                message=(
                    "Waiting for a stable line"
                    if self._armed
                    else "Preview only; press Start to arm"
                )
                if calibration_ready
                else "Ground ROI and configured tape-width calibration required; motors locked",
            )
            self._stop_event.clear()

        if self._line_sensor is not None and self._line_sampling_config is not None:
            self._line_sampler = LineSensorSampler(self._line_sensor, self._line_sampling_config,
                imu_snapshot=self._imu.snapshot, context_snapshot=self._line_sampling_context,
                on_sample=lambda sample: self._test_runs.offer_line_sensor_sample(sample, run_id=sample.context[-1]))
            self._line_sampler.start()
        self._worker = threading.Thread(target=self._run, name="line-follow-control", daemon=True)
        self._worker.start()
        if (self._navigation_config is not None and self._navigation_config.execution_enabled
                and self._corner_drive.config.execution_enabled):
            self._navigation_watchdog = threading.Thread(target=self._run_navigation_watchdog,
                name="navigation-command-lease", daemon=True)
            self._navigation_watchdog.start()
        log.info("Tracking runtime started; auto_start=%s", auto_start)

    def stop(self) -> None:
        with self._lock:
            if (not self._running and self._camera is None and self._motor is None
                    and self._line_sampler is None and self._line_sensor is None):
                return
            self._running = False
            self._armed = False
            self._command_generation += 1
            self._tracking_active = False
            self._acquire_count = 0
            self._temporal_tracker.reset()
            self._curve_speed_controller.reset()
            self._stop_event.set()
            self._reset_obstacle_bypass("runtime_stopped")
            self._reset_corner_motion("runtime_stopped")
            self._status.update(
                state="STOPPED",
                armed=False,
                curve_speed_enabled=self._curve_speed_controller.config.enabled,
                curve_speed_scale=self._curve_speed_controller.current_scale,
                curve_speed_target_scale=self._curve_speed_controller.current_scale,
                message="Runtime stopped",
            )
            self._frame_condition.notify_all()

        self._force_stop()
        self._test_runs.finish("service_stop", self.get_status())
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=2.0)
        self._worker = None
        if self._navigation_watchdog is not None and self._navigation_watchdog is not threading.current_thread():
            self._navigation_watchdog.join(timeout=2.)
        self._navigation_watchdog = None
        self._close_devices()

    def arm(self) -> dict[str, Any]:
        with self._lock:
            if self._armed:
                return self.get_status()
        self._force_stop()
        with self._lock:
            if self._armed:
                return self.get_status()
            if not self._running:
                raise RuntimeError("Camera runtime is not running")
            if self._corner.config.enabled and (
                self._corner.config.pivot_duty > float(self._config.get("control", {}).get("turn_duty_max", 25.0))
                or float(self._config.get("control", {}).get("base_duty", 35.0)) <= 0
            ):
                raise RuntimeError("Corner pivot duty must fit the configured turn limit and base duty must be positive")
            if not vision_config_from_dict(self._config.get("vision", {})).calibration_ready:
                self._armed = False
                self._status.update(
                    state="CALIBRATION_REQUIRED",
                    armed=False,
                    calibration_ready=False,
                    message="Ground ROI and configured tape-width calibration required; motors locked",
                )
                raise RuntimeError("Ground ROI and tape-width calibration must be completed before arming")
            if (
                self._gimbal.config.enabled
                and self._status.get("gimbal_state") != "HOME_COMMAND_SENT"
            ):
                raise RuntimeError("Move the camera gimbal to its configured home position before arming")
            self._test_runs.begin({"runtime": copy.deepcopy(self._config),
                                   "yaw_rate_feedback": asdict(self._yaw_feedback.config)}, self.get_status())
            self._corner.reset()
            self._stop_marker.reset()
            self._reset_obstacle_bypass("new_navigation_session")
            self._reset_corner_motion("new_navigation_session")
            self._armed = True
            self._command_generation += 1
            self._tracking_active = False
            self._acquire_count = 0
            self._near_lost_frames = 0
            self._tracking_seen_since_arm = False
            self._reset_steering_controllers()
            self._curve_speed_controller.reset()
            self._temporal_tracker.reset()
            self._status.update(
                state="WAITING_LINE",
                armed=True,
                corner_state="IDLE",
                corner_direction=0,
                corner_progress_deg=None,
                corner_reason="",
                curve_speed_enabled=self._curve_speed_controller.config.enabled,
                curve_speed_scale=self._curve_speed_controller.current_scale,
                curve_speed_target_scale=self._curve_speed_controller.current_scale,
                message="Waiting for a stable line",
            )
        return self.get_status()

    def disarm(self) -> dict[str, Any]:
        with self._lock:
            self._manual_stop_pending = True
            self._corner.reset()
            self._stop_marker.reset()
            self._reset_obstacle_bypass("disarmed")
            self._reset_corner_motion("disarmed")
            self._armed = False
            self._command_generation += 1
            self._tracking_active = False
            self._acquire_count = 0
            self._near_lost_frames = 0
            self._tracking_seen_since_arm = False
            self._reset_steering_controllers()
            self._curve_speed_controller.reset()
            self._temporal_tracker.reset()
            self._status.update(state="STOPPING", armed=False, message="Stop command pending")
        try:
            self._force_stop()
        except Exception:
            self._test_runs.finish("manual_stop_failed", self.get_status())
            self._manual_stop_pending = False
            raise
        with self._lock:
            self._status.update(
                state="PREVIEW",
                armed=False,
                curve_speed_enabled=self._curve_speed_controller.config.enabled,
                curve_speed_scale=self._curve_speed_controller.current_scale,
                curve_speed_target_scale=self._curve_speed_controller.current_scale,
                message="Zero-duty command sent; physical wheel stop has no feedback",
            )
            self._test_runs.finish("manual_stop", self.get_status())
            self._manual_stop_pending = False
        return self.get_status()

    def home_gimbal(self) -> dict[str, Any]:
        with self._motor_lock:
            with self._lock:
                if not self._running or self._motor is None:
                    raise RuntimeError("Camera and Model A driver must be running")
                if self._armed or self._motor_active:
                    raise RuntimeError("Stop wheel tracking before moving the camera gimbal")
                controller = self._gimbal
                motor = self._motor
                if not controller.config.enabled:
                    raise RuntimeError("Enable and configure gimbal homing first")
                self._status.update(
                    gimbal_state="MOVING",
                    gimbal_message="Sending configured home position",
                )
            try:
                controller.command_home(motor)
                time.sleep(controller.config.move_duration_s + controller.config.settle_time_s)
            except Exception as exc:
                with self._lock:
                    self._status.update(
                        gimbal_state="ERROR",
                        gimbal_message=str(exc),
                    )
                raise RuntimeError(f"Gimbal home command failed: {exc}") from exc
            with self._lock:
                self._status.update(
                    gimbal_state="HOME_COMMAND_SENT",
                    gimbal_message="PWM target sent; mechanical camera angle is not measured",
                    gimbal_pan_pulse_us=controller.config.pan_pulse_us,
                    gimbal_tilt_pulse_us=controller.config.tilt_pulse_us,
                )
                return self.get_status()

    def move_gimbal(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        pan_pulse_us, tilt_pulse_us = validate_gimbal_position(
            payload.get("pan_pulse_us"), payload.get("tilt_pulse_us")
        )

        with self._motor_lock:
            with self._lock:
                if not self._running or self._motor is None:
                    raise RuntimeError("Camera runtime and Model A driver must be running")
                if self._armed or self._motor_active:
                    raise RuntimeError("Stop wheel tracking before moving the camera gimbal")
                controller = self._gimbal
                motor = self._motor
                if not controller.config.enabled:
                    raise RuntimeError("Enable and configure gimbal control first")
                self._invalidate_ground_calibration_locked()
                self._status.update(
                    gimbal_state="MOVING",
                    gimbal_message="Sending manual PWM target",
                )
            try:
                pan_pulse_us, tilt_pulse_us = controller.command_position(
                    motor,
                    pan_pulse_us=pan_pulse_us,
                    tilt_pulse_us=tilt_pulse_us,
                )
            except Exception as exc:
                with self._lock:
                    self._status.update(gimbal_state="ERROR", gimbal_message=str(exc))
                raise RuntimeError(f"Gimbal move command failed: {exc}") from exc

            with self._lock:
                self._status.update(
                    gimbal_state="MANUAL_POSITION",
                    gimbal_message=(
                        "Manual PWM target sent; mechanical angle is not measured. "
                        "Ground calibration must be confirmed again."
                    ),
                    gimbal_pan_pulse_us=pan_pulse_us,
                    gimbal_tilt_pulse_us=tilt_pulse_us,
                )
                return self.get_status()

    def _invalidate_ground_calibration_locked(self) -> None:
        vision = self._config.get("vision", {})
        if not isinstance(vision, dict) or not vision.get("ground_calibration_confirmed"):
            return

        candidate = copy.deepcopy(self._config)
        candidate["vision"]["ground_calibration_confirmed"] = False
        temp_path = self._config_path.with_suffix(self._config_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp_path.replace(self._config_path)
        self._config = candidate
        self._status.update(calibration_ready=False)

    def _move_gimbal_home_at_start(self) -> str:
        controller = self._gimbal
        if not controller.config.enabled:
            return "DISABLED"
        if self._motor is None:
            raise RuntimeError("Model A driver is unavailable for gimbal homing")

        with self._motor_lock:
            controller.command_home(self._motor)
        time.sleep(controller.config.move_duration_s + controller.config.settle_time_s)
        log.info("Gimbal startup home command sent; camera settling completed")
        return "HOME_COMMAND_SENT"

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            calibration_ready = vision_config_from_dict(self._config.get("vision", {})).calibration_ready
            return {
                **self._status,
                **self._imu.snapshot(),
                **self._vision_capture.snapshot(),
                **self._test_runs.snapshot(),
                "line_sensor_sampling": self._line_sampler.snapshot() if self._line_sampler is not None else {"state": "FRAME_SYNC"},
                "sensor_contact_age_s": (None if self._sensor_contact_latest.timestamp_s is None else
                                         max(0., time.monotonic()-self._sensor_contact_latest.timestamp_s)),
                "right_angle_enabled": self._corner.config.enabled,
                "obstacle_enabled": bool(self._config.get("obstacle", {}).get("enabled", False)),
                "obstacle_bypass_enabled": bool(
                    self._config.get("obstacle", {}).get("enabled", False) and self._bypass_config.enabled
                ),
                "imu_feedback_enabled": bool(self._status.get("imu_feedback_enabled") and self._motor_active),
                "armed": self._armed,
                "calibration_ready": calibration_ready,
                "motor_active": self._motor_active,
                "running": self._running,
            }

    def next_frame(self, previous_id: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        with self._frame_condition:
            if self._frame_id == previous_id and self._running:
                self._frame_condition.wait(timeout=max(0.0, timeout))
            return self._frame_id, self._jpeg

    def raw_frame_png(self):
        with self._lock:
            if (not self._running or self._armed or self._motor_active
                    or not self._status.get("camera_ok") or self._raw_snapshot is None):
                return None
            frame, info = self._raw_snapshot
            if time.monotonic() - info["timestamp_monotonic_s"] > 0.5:
                return None
            frame, info = frame.copy(), dict(info)
        ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return (info, encoded.tobytes()) if ok else None

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        if any(key in payload for key in ("imu", "vision_diagnostics", "yaw_rate_feedback", "right_angle", "test_recording")):
            raise ValueError("Edit IMU/diagnostic settings in the config file and restart while stopped")
        candidate = self.config
        vision = payload.get("vision", {})
        control = payload.get("control", {})
        path_geometry = payload.get("path_geometry", {})
        gimbal = payload.get("gimbal", {})
        obstacle = payload.get("obstacle", {})

        if (
            not isinstance(vision, dict)
            or not isinstance(control, dict)
            or not isinstance(path_geometry, dict)
            or not isinstance(gimbal, dict)
            or not isinstance(obstacle, dict)
        ):
            raise ValueError("vision, control, path_geometry, gimbal, and obstacle must be objects")

        if set(obstacle) - {"enabled"}:
            raise ValueError("Only obstacle.enabled can be changed from the Web page")
        if "enabled" in obstacle:
            if not isinstance(obstacle["enabled"], bool):
                raise ValueError("obstacle.enabled must be a boolean")
            candidate.setdefault("obstacle", {})["enabled"] = obstacle["enabled"]

        candidate_vision = candidate.setdefault("vision", {})
        candidate_control = candidate.setdefault("control", {})
        candidate_path_geometry = candidate.setdefault("path_geometry", {})
        candidate_gimbal = candidate.setdefault("gimbal", {})
        if not isinstance(candidate_gimbal, dict):
            raise ValueError("gimbal configuration must be an object")
        candidate_speed_control = candidate_control.setdefault("speed_control", {})
        if not isinstance(candidate_speed_control, dict):
            raise ValueError("control.speed_control must be an object")
        if "speed_control" in control and not isinstance(control["speed_control"], dict):
            raise ValueError("control.speed_control must be an object")

        if "smoothing_alpha" in path_geometry:
            candidate_path_geometry["smoothing_alpha"] = self._bounded_float(
                path_geometry["smoothing_alpha"], "smoothing_alpha", 0.05, 1.0
            )
        if "curve_threshold_px_inv" in path_geometry:
            candidate_path_geometry["curve_threshold_px_inv"] = self._bounded_float(
                path_geometry["curve_threshold_px_inv"],
                "curve_threshold_px_inv",
                0.0,
                0.02,
            )

        if "enabled" in gimbal:
            if not isinstance(gimbal["enabled"], bool):
                raise ValueError("gimbal.enabled must be a boolean")
            candidate_gimbal["enabled"] = gimbal["enabled"]
        for key in ("pan_servo_id", "tilt_servo_id"):
            if key in gimbal:
                candidate_gimbal[key] = self._bounded_int(gimbal[key], key, 1, 4)
        for key, lower, upper in (
            ("pan_pulse_us", PWM_PULSE_MIN_US, PWM_PULSE_MAX_US),
            ("tilt_pulse_us", TILT_PULSE_MIN_US, TILT_PULSE_MAX_US),
        ):
            if key in gimbal:
                candidate_gimbal[key] = self._bounded_int(gimbal[key], key, lower, upper)
        if "move_duration_s" in gimbal:
            candidate_gimbal["move_duration_s"] = self._bounded_float(
                gimbal["move_duration_s"], "move_duration_s", 0.05, 10.0
            )
        if "settle_time_s" in gimbal:
            candidate_gimbal["settle_time_s"] = self._bounded_float(
                gimbal["settle_time_s"], "settle_time_s", 0.0, 5.0
            )
        gimbal_config = gimbal_config_from_dict(candidate_gimbal)

        if "threshold" in vision:
            candidate_vision["threshold"] = self._bounded_int(vision["threshold"], "threshold", 0, 255)
        if "width_tolerance_ratio" in vision:
            candidate_vision["width_tolerance_ratio"] = self._bounded_float(vision["width_tolerance_ratio"], "width_tolerance_ratio", 0.05, 1.0)
        if "tape_width_mm" in vision:
            candidate_vision["tape_width_mm"] = self._bounded_float(vision["tape_width_mm"], "tape_width_mm", 1.0, 100.0)
        for key in ("tape_width_min_mm", "tape_width_max_mm"):
            if key in vision:
                candidate_vision[key] = self._bounded_float(vision[key], key, 1.0, 100.0)
        width_reference = candidate_vision.get("tape_width_mm", 15.0)
        width_min = candidate_vision.get("tape_width_min_mm", width_reference)
        width_max = candidate_vision.get("tape_width_max_mm", width_reference)
        if width_min > width_max:
            raise ValueError("tape_width_min_mm must not exceed tape_width_max_mm")
        if "ground_roi_norm" in vision:
            polygon = vision["ground_roi_norm"]
            if not isinstance(polygon, list) or (polygon and len(polygon) != 4):
                raise ValueError("ground_roi_norm must be empty or contain four normalized points")
            validated_polygon = []
            for point in polygon:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("each ground ROI point must contain normalized x and y")
                validated_polygon.append([
                    self._bounded_float(point[0], "ground_roi_norm.x", 0.0, 1.0),
                    self._bounded_float(point[1], "ground_roi_norm.y", 0.0, 1.0),
                ])
            candidate_vision["ground_roi_norm"] = validated_polygon
        if "tape_width_profile" in vision:
            profile = vision["tape_width_profile"]
            if not isinstance(profile, list) or (profile and len(profile) < 3):
                raise ValueError("tape_width_profile must be empty or contain at least three samples")
            validated_profile = []
            for sample in profile:
                if not isinstance(sample, dict):
                    raise ValueError("each tape width sample must be an object")
                validated_profile.append({
                    "y_norm": self._bounded_float(sample.get("y_norm"), "tape_width_profile.y_norm", 0.0, 1.0),
                    "width_norm": self._bounded_float(sample.get("width_norm"), "tape_width_profile.width_norm", 0.0, 1.0),
                })
            sample_rows = [sample["y_norm"] for sample in validated_profile]
            if sample_rows != sorted(set(sample_rows)):
                raise ValueError("tape_width_profile rows must be sorted and unique")
            candidate_vision["tape_width_profile"] = validated_profile
        if "ground_calibration_confirmed" in vision:
            if not isinstance(vision["ground_calibration_confirmed"], bool):
                raise ValueError("ground_calibration_confirmed must be a boolean")
            candidate_vision["ground_calibration_confirmed"] = vision["ground_calibration_confirmed"]
        if "scanline_count" in vision:
            candidate_vision["scanline_count"] = self._bounded_int(vision["scanline_count"], "scanline_count", 3, 41)
        if "scanline_positions_norm" in vision:
            positions = vision["scanline_positions_norm"]
            if not isinstance(positions, list) or (positions and not 3 <= len(positions) <= 41):
                raise ValueError("scanline_positions_norm must be empty or contain between three and 41 positions")
            validated_positions = [
                self._bounded_float(value, "scanline_positions_norm", 0.0, 1.0)
                for value in positions
            ]
            if validated_positions != sorted(set(validated_positions)):
                raise ValueError("scanline_positions_norm must be sorted and unique")
            candidate_vision["scanline_positions_norm"] = validated_positions
        if "min_valid_scanlines" in vision:
            candidate_vision["min_valid_scanlines"] = self._bounded_int(vision["min_valid_scanlines"], "min_valid_scanlines", 2, 41)
        if "min_near_valid_scanlines" in vision:
            candidate_vision["min_near_valid_scanlines"] = self._bounded_int(
                vision["min_near_valid_scanlines"], "min_near_valid_scanlines", 1, 41
            )
        if "max_lateral_slope" in vision:
            candidate_vision["max_lateral_slope"] = self._bounded_float(vision["max_lateral_slope"], "max_lateral_slope", 0.01, 5.0)
        if "max_width_change_ratio" in vision:
            candidate_vision["max_width_change_ratio"] = self._bounded_float(vision["max_width_change_ratio"], "max_width_change_ratio", 0.01, 2.0)
        if "min_confidence" in vision:
            candidate_vision["min_confidence"] = self._bounded_float(vision["min_confidence"], "min_confidence", 0.0, 1.0)
        score_weight_limits = {
            "center_transition_weight": (0.0, 5.0),
            "width_transition_weight": (0.0, 5.0),
            "confidence_width_fit_weight": (0.0, 5.0),
            "confidence_smoothness_weight": (0.0, 5.0),
            "confidence_coverage_weight": (0.0, 5.0),
        }
        for key, (lower, upper) in score_weight_limits.items():
            if key in vision:
                candidate_vision[key] = self._bounded_float(vision[key], key, lower, upper)
        if "debug_visualization" in vision:
            if not isinstance(vision["debug_visualization"], bool):
                raise ValueError("debug_visualization must be a boolean")
            candidate_vision["debug_visualization"] = vision["debug_visualization"]
        for roi_name in ("roi_near", "roi_far"):
            if roi_name not in vision:
                continue
            roi_values = vision[roi_name]
            if not isinstance(roi_values, dict):
                raise ValueError(f"{roi_name} must be an object")
            roi = candidate_vision.setdefault(roi_name, {})
            for key, lower, upper in (("x", 0, 4095), ("y", 0, 4095), ("w", 1, 4096), ("h", 1, 4096)):
                if key in roi_values:
                    roi[key] = self._bounded_int(roi_values[key], f"{roi_name}.{key}", lower, upper)

        if "base_duty" in control:
            candidate_control["base_duty"] = self._bounded_float(control["base_duty"], "base_duty", 0.0, 100.0)
        if "turn_duty_max" in control:
            candidate_control["turn_duty_max"] = self._bounded_float(control["turn_duty_max"], "turn_duty_max", 0.0, 100.0)
        if "far_line_weight" in control:
            candidate_control["far_line_weight"] = self._bounded_float(
                control["far_line_weight"], "far_line_weight", 0.0, 0.5
            )
        if "heading_gain" in control:
            candidate_control["heading_gain"] = self._bounded_float(
                control["heading_gain"], "heading_gain", 0.0, 5.0
            )
        if "heading_feedback_limit" in control:
            candidate_control["heading_feedback_limit"] = self._bounded_float(
                control["heading_feedback_limit"], "heading_feedback_limit", 0.0, 1.0
            )
        if "near_only_speed_scale" in control:
            candidate_control["near_only_speed_scale"] = self._bounded_float(
                control["near_only_speed_scale"], "near_only_speed_scale", 0.1, 1.0
            )
        if "near_lost_stop_frames" in control:
            candidate_control["near_lost_stop_frames"] = self._bounded_int(
                control["near_lost_stop_frames"], "near_lost_stop_frames", 1, 60
            )
        if "line_acquire_frames" in control:
            candidate_control["line_acquire_frames"] = self._bounded_int(control["line_acquire_frames"], "line_acquire_frames", 1, 30)
        if "temporal_max_center_shift_norm_per_s" in control:
            candidate_control["temporal_max_center_shift_norm_per_s"] = self._bounded_float(
                control["temporal_max_center_shift_norm_per_s"],
                "temporal_max_center_shift_norm_per_s",
                0.01,
                2.0,
            )
        if "temporal_max_uncertain_frames" in control:
            candidate_control["temporal_max_uncertain_frames"] = self._bounded_int(
                control["temporal_max_uncertain_frames"],
                "temporal_max_uncertain_frames",
                0,
                30,
            )
        if "pid" in control:
            pid_values = control["pid"]
            if not isinstance(pid_values, dict):
                raise ValueError("control.pid must be an object")
            pid_config = candidate_control.setdefault("pid", {})
            limits = {
                "kp": (0.0, 100.0),
                "ki": (0.0, 100.0),
                "kd": (0.0, 100.0),
                "integral_limit": (0.0, 100.0),
                "output_limit": (0.01, 2.0),
                "derivative_filter_alpha": (0.05, 1.0),
                "output_smoothing_alpha": (0.05, 1.0),
                "large_error_threshold": (0.0, 1.0),
                "large_error_gain": (1.0, 2.0),
            }
            for key, (lower, upper) in limits.items():
                if key in pid_values:
                    pid_config[key] = self._bounded_float(pid_values[key], key, lower, upper)

        speed_control_values = control.get("speed_control", {})
        if "enabled" in speed_control_values:
            if not isinstance(speed_control_values["enabled"], bool):
                raise ValueError("speed_control.enabled must be a boolean")
            candidate_speed_control["enabled"] = speed_control_values["enabled"]
        speed_control_limits = {
            "minimum_scale": (0.0, 1.0),
            "maximum_scale": (0.0, 1.0),
            "straight_scale": (0.0, 1.0),
            "curve_scale": (0.0, 1.0),
            "full_curve_px_inv": (0.0, 0.02),
            "acceleration_scale_per_s": (0.0, 5.0),
            "deceleration_scale_per_s": (0.0, 5.0),
        }
        for key, (lower, upper) in speed_control_limits.items():
            if key in speed_control_values:
                candidate_speed_control[key] = self._bounded_float(
                    speed_control_values[key], f"speed_control.{key}", lower, upper
                )
        curve_speed_config = speed_control_config_from_dict(
            candidate_speed_control,
            curve_threshold_px_inv=path_geometry_config_from_dict(
                candidate_path_geometry
            ).curve_threshold_px_inv,
        )

        effective_scanline_count = len(candidate_vision.get("scanline_positions_norm", [])) or int(candidate_vision.get("scanline_count", 11))
        if int(candidate_vision.get("min_valid_scanlines", 7)) > effective_scanline_count:
            raise ValueError("min_valid_scanlines cannot exceed the effective scanline count")
        if int(candidate_vision.get("min_near_valid_scanlines", 3)) > effective_scanline_count:
            raise ValueError("min_near_valid_scanlines cannot exceed the effective scanline count")

        candidate_bypass_config = ObstacleBypassConfig.from_obstacle_config(
            candidate.get("obstacle", {})
        )
        if (candidate.get("obstacle", {}).get("enabled", False)
                and candidate_bypass_config.enabled and not self._imu.config.enabled):
            raise ValueError("Obstacle bypass requires verified IMU sampling")

        with self._lock:
            temp_path = self._config_path.with_suffix(self._config_path.suffix + ".tmp")
            temp_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp_path.replace(self._config_path)
            self._config = candidate
            self._bypass_config = candidate_bypass_config
            self._bypass = ObstacleBypassController(candidate_bypass_config)
            self._distance_filter = DistanceFilter(
                min_valid_mm=candidate_bypass_config.min_valid_distance_mm,
                max_spread_mm=candidate_bypass_config.max_spread_mm,
                required_samples=2,
            )
            self._pid = PID(_pid_config(candidate_control.get("pid", {})))
            self._temporal_tracker = self._new_temporal_tracker(candidate)
            self._path_geometry = PathGeometryEstimator(
                path_geometry_config_from_dict(candidate.get("path_geometry", {}))
            )
            self._curve_speed_controller = CurveSpeedController(curve_speed_config)
            if gimbal:
                self._gimbal = PwmGimbalController(gimbal_config)
            self._command_generation += 1
            self._acquire_count = 0
            self._tracking_active = False
            self._armed = False
            self._reset_obstacle_bypass("configuration_changed")
            self._reset_corner_motion("configuration_changed")
            self._near_lost_frames = 0
            self._tracking_seen_since_arm = False
            self._reset_steering_controllers()
            self._status.update(
                state="PREVIEW" if vision_config_from_dict(candidate.get("vision", {})).calibration_ready else "CALIBRATION_REQUIRED",
                armed=False,
                calibration_ready=vision_config_from_dict(candidate.get("vision", {})).calibration_ready,
                curve_speed_enabled=self._curve_speed_controller.config.enabled,
                curve_speed_scale=self._curve_speed_controller.current_scale,
                curve_speed_target_scale=self._curve_speed_controller.current_scale,
                gimbal_enabled=gimbal_config.enabled,
                gimbal_state=(("NOT_HOMED" if gimbal_config.enabled else "DISABLED")
                              if gimbal else self._status.get("gimbal_state", "DISABLED")),
                gimbal_message="" if gimbal else self._status.get("gimbal_message", ""),
                message="Settings applied; disarmed. Review preview before starting again",
            )

        self._force_stop()
        return self.get_debug_config()

    def get_debug_config(self) -> dict[str, Any]:
        with self._lock:
            vision = self._config.get("vision", {})
            control = self._config.get("control", {})
            gimbal = self._gimbal.config
            return {
                "obstacle": {
                    "enabled": bool(self._config.get("obstacle", {}).get("enabled", False)),
                    "stop_distance_mm": int(self._config.get("obstacle", {}).get("stop_distance_mm", 250)),
                    "hard_stop_distance_mm": int(self._config.get("obstacle", {}).get("hard_stop_distance_mm", 60)),
                    "pre_slow_distance_mm": int(
                        self._config.get("obstacle", {}).get("bypass", {}).get("pre_slow_distance_mm", 250)
                    ),
                    "pre_slow_scale": float(
                        self._config.get("obstacle", {}).get("bypass", {}).get("pre_slow_scale", 0.4)
                    ),
                    "bypass_enabled": bool(
                        self._config.get("obstacle", {}).get("bypass", {}).get("enabled", False)
                    ),
                    "bypass_effective_enabled": bool(
                        self._config.get("obstacle", {}).get("enabled", False) and self._bypass_config.enabled
                    ),
                    "bypass_phase": self._status.get("obstacle_bypass_phase", "IDLE"),
                },
                "imu": copy.deepcopy(self._config.get("imu", {})),
                "vision": {
                    "threshold": vision.get("threshold", 60),
                    "reference_resolution": copy.deepcopy(vision.get("reference_resolution", {})),
                    "ground_roi_norm": copy.deepcopy(vision.get("ground_roi_norm", [])),
                    "tape_width_mm": vision.get("tape_width_mm", 15.0),
                    "tape_width_min_mm": vision.get(
                        "tape_width_min_mm", vision.get("tape_width_mm", 15.0)
                    ),
                    "tape_width_max_mm": vision.get(
                        "tape_width_max_mm", vision.get("tape_width_mm", 15.0)
                    ),
                    "tape_width_profile": copy.deepcopy(vision.get("tape_width_profile", [])),
                    "ground_calibration_confirmed": vision_config_from_dict(vision).ground_calibration_confirmed,
                    "width_tolerance_ratio": vision.get("width_tolerance_ratio", 0.5),
                    "calibration_ready": vision_config_from_dict(vision).calibration_ready,
                    "scanline_count": vision.get("scanline_count", 11),
                    "scanline_positions_norm": copy.deepcopy(vision.get("scanline_positions_norm", [])),
                    "min_valid_scanlines": vision.get("min_valid_scanlines", 7),
                    "min_near_valid_scanlines": vision.get("min_near_valid_scanlines", 3),
                    "max_lateral_slope": vision.get("max_lateral_slope", 1.25),
                    "max_width_change_ratio": vision.get("max_width_change_ratio", 0.8),
                    "min_confidence": vision.get("min_confidence", 0.65),
                    "center_transition_weight": vision.get("center_transition_weight", 0.3),
                    "width_transition_weight": vision.get("width_transition_weight", 0.2),
                    "confidence_width_fit_weight": vision.get("confidence_width_fit_weight", 0.55),
                    "confidence_smoothness_weight": vision.get("confidence_smoothness_weight", 0.25),
                    "confidence_coverage_weight": vision.get("confidence_coverage_weight", 0.2),
                    "debug_visualization": vision.get("debug_visualization", True),
                    "roi_near": copy.deepcopy(vision.get("roi_near", {})),
                    "roi_far": copy.deepcopy(vision.get("roi_far", {})),
                },
                "control": {
                    "base_duty": control.get("base_duty", 35.0),
                    "turn_duty_max": control.get("turn_duty_max", 25.0),
                    "far_line_weight": control.get("far_line_weight", 0.0),
                    "heading_gain": control.get("heading_gain", 0.25),
                    "heading_feedback_limit": control.get("heading_feedback_limit", 0.35),
                    "near_only_speed_scale": control.get("near_only_speed_scale", 0.5),
                    "near_lost_stop_frames": control.get("near_lost_stop_frames", 5),
                    "speed_control": {
                        "enabled": self._curve_speed_controller.config.enabled,
                        "minimum_scale": self._curve_speed_controller.config.minimum_scale,
                        "maximum_scale": self._curve_speed_controller.config.maximum_scale,
                        "straight_scale": self._curve_speed_controller.config.straight_scale,
                        "curve_scale": self._curve_speed_controller.config.curve_scale,
                        "full_curve_px_inv": self._curve_speed_controller.config.full_curve_px_inv,
                        "acceleration_scale_per_s": self._curve_speed_controller.config.acceleration_scale_per_s,
                        "deceleration_scale_per_s": self._curve_speed_controller.config.deceleration_scale_per_s,
                    },
                    "line_acquire_frames": control.get("line_acquire_frames", 3),
                    "temporal_max_center_shift_norm_per_s": control.get("temporal_max_center_shift_norm_per_s", 0.35),
                    "temporal_max_uncertain_frames": control.get("temporal_max_uncertain_frames", 2),
                    "pid": copy.deepcopy(control.get("pid", {})),
                },
                "path_geometry": {
                    "smoothing_alpha": self._config.get("path_geometry", {}).get(
                        "smoothing_alpha", 0.35
                    ),
                    "curve_threshold_px_inv": self._config.get("path_geometry", {}).get(
                        "curve_threshold_px_inv", 0.0005
                    ),
                },
                "gimbal": {
                    "enabled": gimbal.enabled,
                    "pan_servo_id": gimbal.pan_servo_id,
                    "tilt_servo_id": gimbal.tilt_servo_id,
                    "pan_pulse_us": gimbal.pan_pulse_us,
                    "tilt_pulse_us": gimbal.tilt_pulse_us,
                    "move_duration_s": gimbal.move_duration_s,
                    "settle_time_s": gimbal.settle_time_s,
                },
            }

    @staticmethod
    def _bounded_int(value: Any, name: str, lower: int, upper: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        number = int(value)
        if number < lower or number > upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
        return number

    @staticmethod
    def _bounded_float(value: Any, name: str, lower: float, upper: float) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < lower or number > upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
        return number

    def _run(self) -> None:
        last_time = time.monotonic()
        next_camera_reconnect_at = 0.0
        next_battery_poll_at = 0.0
        camera_reconnect_interval_s = max(
            1.0, float(self._config.get("camera", {}).get("reconnect_interval_s", 2.0))
        )
        while not self._stop_event.is_set():
            try:
                now = time.monotonic()
                if now >= next_battery_poll_at:
                    next_battery_poll_at = now + 1.0
                    self._refresh_battery_status(now)
                try:
                    frame = self._camera.read() if self._camera is not None else None
                except Exception as exc:
                    log.warning("Camera read failed: %s", exc)
                    frame = None
                now = time.monotonic()
                dt = max(0.001, now - last_time)
                last_time = now
                if frame is None:
                    self._camera_failed()
                    if self._camera is not None and now >= next_camera_reconnect_at:
                        next_camera_reconnect_at = now + camera_reconnect_interval_s
                        try:
                            self._camera.reopen()
                        except Exception as exc:
                            log.warning("Camera reopen failed; retrying: %s", exc)
                        else:
                            log.info("Camera reopened; tracking remains disarmed")
                    self._stop_event.wait(0.1)
                    continue
                self._process_frame(frame, dt)
            except Exception as exc:
                log.exception("Tracking loop failed")
                with self._lock:
                    self._armed = False
                    self._command_generation += 1
                    self._tracking_active = False
                    self._acquire_count = 0
                    self._temporal_tracker.reset()
                    self._path_geometry.reset()
                    self._reset_obstacle_bypass("runtime_error")
                    self._reset_corner_motion("runtime_error")
                    self._status.update(
                        state="ERROR",
                        armed=False,
                        camera_ok=False,
                        line_near=False,
                        line_far=False,
                        line_consistent=False,
                        vision_state="LOST",
                        track_state="LOST",
                        confidence=0.0,
                        valid_scanlines=0,
                        candidate_scanlines=0,
                        reacquire_count=0,
                        missed_frames=0,
                        expected_width_near_px=None,
                        expected_width_near_min_px=None,
                        expected_width_near_max_px=None,
                        candidate_x_near=None,
                        candidate_x_far=None,
                        x_near=None,
                        x_far=None,
                        path_midpoint_x=None,
                        path_midpoint_y=None,
                        error=None,
                        track_heading_deg=None,
                        path_heading_source="NONE",
                        path_curvature_px_inv=None,
                        curve_direction="UNKNOWN",
                        path_fit_rms_px=None,
                        ground_corner_geometry=asdict(GroundCornerGeometry(reason="runtime_error")),
                        local_path_plan=asdict(LocalPathPlan(reason="runtime_error")),
                        local_path_state="UNCERTAIN",
                        local_path_reason="runtime_error",
                        local_path_mode="STOP",
                        local_path_corner_distance_m=None,
                        local_path_corner_turn_deg=None,
                        local_path_steering_curvature_m_inv=None,
                        fps=0.0,
                        message=str(exc),
                    )
                try:
                    self._force_stop()
                except Exception:
                    log.critical("Tracking failed and motor stop could not be dispatched", exc_info=True)
                self._corner.reset()
                self._stop_marker.reset()
                self._test_runs.finish("runtime_error", self.get_status())
                self._stop_event.wait(0.2)

    def _process_frame(self, frame, dt: float) -> None:
        frame_timestamp_s = time.monotonic()
        raw_frame = frame.copy()
        imu_status = self._imu.snapshot()
        imu_unavailable = self._imu.config.enabled and not imu_status["imu_healthy"]
        with self._lock:
            self._line_frame_size = (frame.shape[1], frame.shape[0])
            config = copy.deepcopy(self._config)
            armed = self._armed
            was_tracking = self._tracking_active
            motor_was_active = self._motor_active
            command_generation = self._command_generation
            temporal_tracker = self._temporal_tracker
            path_geometry_estimator = self._path_geometry
            curve_speed_controller = self._curve_speed_controller
            test_run_id = self._test_runs.snapshot()["test_run_id"]
            gimbal_pose_is_commanded = self._status.get("gimbal_state") in {
                "HOME_COMMAND_SENT", "MANUAL_POSITION"
            }
            gimbal_pan_pulse_us = (
                self._status.get("gimbal_pan_pulse_us") if gimbal_pose_is_commanded else None
            )
            gimbal_tilt_pulse_us = (
                self._status.get("gimbal_tilt_pulse_us") if gimbal_pose_is_commanded else None
            )

        sample_context, sample_active = self._line_sampling_context()
        sensor_samples, dropped_samples = (self._line_sampler.drain_before(frame_timestamp_s)
                                           if self._line_sampler is not None else ((), 0))
        vision_config = vision_config_from_dict(config.get("vision", {}))
        calibration_ready = vision_config.calibration_ready
        features, debug = detect_line(frame, vision_config)
        temporal = temporal_tracker.update(features, frame_width=frame.shape[1], dt=dt)
        if features.state == "VALID" and temporal.state == "VALID":
            path_geometry = path_geometry_estimator.update(
                features.centerline,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )
        else:
            path_geometry = path_geometry_estimator.update(
                (), frame_width=frame.shape[1], frame_height=frame.shape[0]
            )
        path_midpoint = (
            path_geometry.centerline[len(path_geometry.centerline) // 2]
            if path_geometry.valid and path_geometry.centerline
            else None
        )
        ground_path = self._ground_mapper.estimate_front_reference_path(
            features.centerline,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            pan_pulse_us=gimbal_pan_pulse_us,
            tilt_pulse_us=gimbal_tilt_pulse_us,
            path_confidence=features.confidence,
            path_fit_valid=(
                features.state == "VALID"
                and temporal.state == "VALID"
                and path_geometry.valid
            ),
        )
        if self._line_sampler is None:
            line_sensor_channels = self._line_sensor.read_channels() if self._line_sensor is not None else None
        else:
            latest_sample = sensor_samples[-1] if sensor_samples else None
            line_sensor_channels = (latest_sample.channels if latest_sample is not None
                and latest_sample.context == sample_context
                and 0 <= frame_timestamp_s-latest_sample.started_s <= self._line_sampler.config.stale_after_s else None)
        corner_observation = detect_right_angle(frame, vision_config, tracking_corner=self._corner.state in {"CONFIRM", "APPROACH"}) if (self._navigation_config is not None or self._corner.config.enabled or not armed) else CornerObservation()
        ground_corner = GroundCornerGeometry()
        if corner_observation.direction and corner_observation.path_points_px:
            pixels = corner_observation.path_points_px
            vertex = (corner_observation.x_norm*frame.shape[1],
                      corner_observation.y_norm*frame.shape[0])
            vertex_index = min(range(len(pixels)), key=lambda i: math.dist(pixels[i], vertex))
            ground_corner = estimate_ground_corner(
                pixels[:vertex_index+1], pixels[vertex_index:],
                mapper=self._ground_mapper,
                frame_width=frame.shape[1], frame_height=frame.shape[0],
                pan_pulse_us=gimbal_pan_pulse_us, tilt_pulse_us=gimbal_tilt_pulse_us,
                min_leg_span_m=2*vision_config.tape_width_mm/1000,
                max_fit_rms_m=vision_config.tape_width_mm/2000)
        local_path_config = config.get("local_path", {})
        local_path_limits = ("lookahead_m", "max_track_curvature_m_inv",
                             "max_anchor_offset_m", "max_segment_m",
                             "front_to_control_center_m", "max_unobserved_distance_m")
        if not self._ground_mapper.available:
            local_path = LocalPathPlan(reason=self._ground_mapper.reason)
        elif not all(key in local_path_config for key in local_path_limits):
            local_path = LocalPathPlan(reason="planner_limits_unconfigured")
        else:
            path_points_px = (corner_observation.path_points_px
                              if corner_observation.direction else tuple(reversed(features.centerline)))
            local_path = plan_image_path(
                path_points_px, mapper=self._ground_mapper,
                frame_width=frame.shape[1], frame_height=frame.shape[0],
                pan_pulse_us=gimbal_pan_pulse_us, tilt_pulse_us=gimbal_tilt_pulse_us,
                branch_count=1 if corner_observation.direction or (
                    features.state == "VALID" and temporal.state == "VALID") else 0,
                **{key: float(local_path_config[key]) for key in local_path_limits},
            )
        line_pose = LinePoseEvidence()
        if self._line_pose_config is not None:
            line_pose = (estimate_image_line_pose(
                tuple(reversed(features.centerline)), mapper=self._ground_mapper, config=self._line_pose_config,
                observation_unique=features.state == "VALID" and temporal.state == "VALID",
                frame_width=frame.shape[1], frame_height=frame.shape[0],
                pan_pulse_us=gimbal_pan_pulse_us, tilt_pulse_us=gimbal_tilt_pulse_us)
                if calibration_ready else LinePoseEvidence(reason="calibration_unavailable"))
        exit_path_available = (
            temporal.state == "VALID" and features.state == "VALID"
            and features.consistent and features.ok_near
            and path_geometry.valid and path_geometry.heading_deg is not None
            and abs(path_geometry.heading_deg) < self._corner.config.exit_max_heading_deg
            and temporal.center_x is not None and bool(features.centerline))
        obstacle_config = config.get("obstacle", {})
        obstacle_enabled = bool(obstacle_config.get("enabled", False))
        raw_distances: list[int | None] = []
        if self._distance_sensor is not None:
            try:
                raw_distances = [self._distance_sensor.read_distance_mm() for _ in range(3)]
            except Exception as exc:
                log.warning("Distance sensor read failed: %s", exc)
        distance_mm = (
            self._distance_filter.update(raw_distances)
            if self._distance_sensor is not None else None
        )
        line_heading_deg = path_geometry.heading_deg if path_geometry.valid else None
        line_valid = (
            temporal.state == "VALID" and features.state == "VALID"
            and features.consistent and features.ok_near
            and temporal.center_x is not None and line_heading_deg is not None
        )
        line_center_norm = (
            (temporal.center_x - frame.shape[1] / 2.0) / frame.shape[1]
            if temporal.center_x is not None else None
        )
        line_near = bool(features.ok_near)
        line_far = bool(features.ok_far)
        corner_running = self._corner.state in {
            "APPROACH", "ADVANCE", "SETTLE", "TURN", "SETTLE_TURN"
        }
        if self._navigation_config is not None:
            corner_running = self._navigation_latest.owner in {"APPROACH", "TURN", "SETTLE", "EXIT"}
        stop_marker_running = self._stop_marker.state in {"CONFIRM", "APPROACH", "ADVANCE"}
        bypass = self._bypass.update(
            now=frame_timestamp_s, dt=dt, distance_mm=distance_mm,
            obstacle_enabled=obstacle_enabled and not stop_marker_running,
            armed=armed and command_generation == self._command_generation,
            corner_active=corner_running,
            tracking_ready=bool(was_tracking or line_valid),
            imu=imu_status, line_valid=line_valid,
            line_center_norm=line_center_norm, line_heading_deg=line_heading_deg,
            line_sensor_channels=line_sensor_channels,
            line_near=line_near, line_far=line_far,
            camera_ok=True, user_stop=self._manual_stop_pending,
        )
        if bypass.phase != self._last_bypass_phase:
            log.info(
                "OBSTACLE_BYPASS phase=%s mode=%s reason=%s progress_deg=%s "
                "distance_mm=%s attempts=%s",
                bypass.phase, bypass.mode, bypass.reason,
                None if bypass.progress_deg is None else round(bypass.progress_deg, 2),
                distance_mm, bypass.attempts,
            )
            self._last_bypass_phase = bypass.phase
        with self._lock:
            corner_motion = CornerMotionEstimate()
            if self._corner_motion is not None:
                if command_generation != self._command_generation or not calibration_ready:
                    self._reset_corner_motion("frame_context_invalidated")
                    corner_motion = CornerMotionEstimate(reason="frame_context_invalidated")
                elif bypass.active or bypass.stop or self._bypass.failed:
                    self._reset_corner_motion("obstacle_bypass_owns_control")
                    corner_motion = CornerMotionEstimate(reason="obstacle_bypass_owns_control")
                else:
                    corner_motion = self._corner_motion.update(
                        ground_corner, timestamp_s=frame_timestamp_s, imu=imu_status,
                        context=(frame.shape[1], frame.shape[0], gimbal_pan_pulse_us,
                                 gimbal_tilt_pulse_us, id(self._ground_mapper), command_generation))
            contact_samples = []
            sensor_contact = SensorContactEvidence()
            if self._sensor_contact is not None:
                if command_generation != self._command_generation or not calibration_ready:
                    self._sensor_contact.reset()
                    sensor_contact = SensorContactEvidence(reason="frame_context_invalidated")
                elif self._line_sampler is None:
                    sensor_contact = self._sensor_contact.update(
                        line_sensor_channels, corner_motion, timestamp_s=frame_timestamp_s, imu=imu_status,
                        active=sample_active, context=sample_context)
                else:
                    if dropped_samples:
                        self._sensor_contact.invalidate_samples()
                        self._sensor_contact_latest = SensorContactEvidence(reason="sensor_buffer_gap")
                    sensor_contact = self._sensor_contact_latest
                    for sample in sensor_samples:
                        if sample.context != sample_context:
                            self._sensor_contact.invalidate_samples()
                            sensor_contact = SensorContactEvidence(reason="sample_context_changed",
                                                                   timestamp_s=sample.completed_s)
                        else:
                            sensor_contact = self._sensor_contact.update(
                                sample.channels, None, timestamp_s=sample.completed_s, imu=sample.imu,
                                active=sample.active, context=sample.context, sample_started_s=sample.started_s,
                                yaw_rate_bound_dps=self._corner.config.max_rate_dps)
                        contact_samples.append({"sequence": sample.sequence, "evidence": asdict(sensor_contact)})
                    if (sensor_contact.timestamp_s is not None and not
                            0 <= frame_timestamp_s-sensor_contact.timestamp_s+sensor_contact.sample_duration_s <= self._line_sampler.config.stale_after_s):
                        self._sensor_contact.invalidate_samples()
                        sensor_contact = replace(sensor_contact, state="UNAVAILABLE", reason="sensor_samples_stale",
                            corner_y_interval_m=None, contact_band_y_interval_m=None, prior_corner_y_interval_m=None)
                    if sample_active:
                        self._sensor_contact.observe_corner(corner_motion, imu=imu_status, context=sample_context)
                self._sensor_contact_latest = sensor_contact
            corner_position = CornerPositionEstimate()
            if bypass.active or bypass.stop or self._bypass.failed:
                corner_position = CornerPositionEstimate(reason="obstacle_bypass_owns_control")
            elif self._corner_position_config is not None:
                fusion_contact = (sensor_contact if self._line_sampler is not None else
                                  SensorContactEvidence(reason="timestamped_contact_unavailable"))
                corner_position = fuse_corner_position(corner_motion, fusion_contact,
                    now_s=frame_timestamp_s, imu=imu_status, config=self._corner_position_config,
                    motion_config=self._corner_motion.config, contact_config=self._sensor_contact.config)
            corner_approach = CornerApproachDecision()
            if bypass.active or bypass.stop or self._bypass.failed:
                corner_approach = CornerApproachDecision(reason="obstacle_bypass_owns_control")
            elif self._corner_approach_config is not None:
                corner_approach = decide_corner_approach(corner_motion,
                    self._corner_approach_config, now_s=frame_timestamp_s, yaw_rate_dps=imu_status.get("imu_yaw_rate_dps"),
                    position=corner_position if self._corner_position_config is not None else None)
            corner_transition = CornerTransitionDecision()
            if bypass.active or bypass.stop or self._bypass.failed:
                corner_transition = CornerTransitionDecision(reason="obstacle_bypass_owns_control")
            elif self._corner_transition is not None:
                exit_evidence = exit_path_evidence(local_path, timestamp_s=frame_timestamp_s,
                    observation_unique=(temporal.state == "VALID" and features.state == "VALID"
                                        and features.consistent and features.ok_near),
                    position_error_m=self._corner_transition.config.position_error_m)
                corner_transition = self._corner_transition.update(corner_approach,
                    now_s=frame_timestamp_s, imu=imu_status, exit_path=exit_evidence,
                    context=(command_generation, frame.shape[1], frame.shape[0],
                             gimbal_pan_pulse_us, gimbal_tilt_pulse_us, id(self._ground_mapper)))
            navigation = NavigationRequest()
            if self._navigation_config is not None:
                navigation = select_navigation(now_s=frame_timestamp_s, config=self._navigation_config,
                    armed=self._armed and command_generation == self._command_generation,
                    calibration_ready=calibration_ready, manual_stop=self._manual_stop_pending,
                    path=local_path, corner_candidate=bool(corner_observation.direction)
                        or local_path.mode == "PIVOT_REQUIRED", motion=corner_motion,
                    approach=corner_approach, transition=corner_transition, bypass=bypass)
                self._navigation_latest = navigation
            corner_drive = CornerDriveCommand()
            if bypass.active or bypass.stop or self._bypass.failed:
                corner_drive = CornerDriveCommand(reason="obstacle_bypass_owns_control")
            elif self._navigation_config is not None:
                corner_drive = self._corner_drive.map_setpoint(
                    navigation.forward_speed_mps, navigation.yaw_rate_dps,
                    now_s=frame_timestamp_s, requested_at_s=navigation.timestamp_s,
                    imu=imu_status, motor_config=config.get("motor", {}),
                    turn_duty_max=float(config.get("control", {}).get("turn_duty_max", 25.)))
            elif self._corner_drive is not None:
                corner_drive = self._corner_drive.map_setpoint(
                    corner_transition.forward_speed_mps, corner_transition.target_yaw_rate_dps,
                    now_s=frame_timestamp_s, requested_at_s=corner_transition.timestamp_s,
                    imu=imu_status, motor_config=config.get("motor", {}),
                    turn_duty_max=float(config.get("control", {}).get("turn_duty_max", 25.)))
            if self._navigation_config is not None or self._bypass.active or bypass.stop or self._bypass.failed:
                self._corner.reset()
                self._stop_marker.reset()
                corner = CornerCommand()
            else:
                corner = self._corner.update(
                    corner_observation, now=frame_timestamp_s, dt=dt,
                    armed=self._armed and calibration_ready, ground_corner=ground_corner,
                    imu=imu_status, line_sensor_channels=line_sensor_channels,
                    exit_path_available=exit_path_available,
                    line_aligned=(exit_path_available and
                        abs(features.centerline[-1][0]-frame.shape[1]/2) <
                        frame.shape[1]*self._corner.config.exit_max_center_offset_norm))
            if corner.failed:
                self._reset_obstacle_bypass("corner_failed")
                self._armed = armed = False
                self._command_generation += 1
                self._tracking_active = False
        marker_config = config.get("stop_marker", {})
        marker_observation = StopMarkerObservation()
        if marker_config.get("enabled", False):
            marker_observation = detect_stop_marker(
                frame, vision_config, self._ground_mapper,
                pan_pulse_us=gimbal_pan_pulse_us, tilt_pulse_us=gimbal_tilt_pulse_us,
                crossbar_m=marker_config["crossbar_m"],
                tolerance_m=marker_config["crossbar_tolerance_m"],
            )
        if bypass.active or bypass.stop or self._bypass.failed:
            self._stop_marker.reset()
            marker = StopMarkerCommand()
        else:
            marker = self._stop_marker.update(
                marker_observation, now=frame_timestamp_s, dt=dt,
                armed=armed and command_generation == self._command_generation,
                imu=imu_status, channels=line_sensor_channels,
            )
        if marker.active or marker.done or marker.failed:
            self._corner.reset()
            corner = CornerCommand()
        if marker.done or marker.failed:
            with self._lock:
                if self._armed:
                    self._armed = armed = False
                    self._command_generation += 1
                self._tracking_active = False
        control_config = config.get("control", {})
        near_lost_stop_frames = max(1, int(control_config.get("near_lost_stop_frames", 5)))
        near_path_available = (
            features.state == "VALID"
            and features.x_near is not None
            and features.consistent
        )
        near_path_available = near_path_available or corner.active
        if self._navigation_config is not None and navigation.owner in {"APPROACH", "TURN", "SETTLE", "EXIT"}:
            near_path_available = True
        end_of_line = False
        with self._lock:
            if (self._navigation_config is not None or features.state != "LOST" or not self._armed or not self._tracking_seen_since_arm
                    or near_path_available or self._bypass.active):
                self._near_lost_frames = 0
            else:
                self._near_lost_frames += 1
                if self._near_lost_frames >= near_lost_stop_frames:
                    self._armed = False
                    self._command_generation += 1
                    self._tracking_active = False
                    self._acquire_count = 0
                    self._near_lost_frames = 0
                    self._tracking_seen_since_arm = False
                    self._reset_steering_controllers()
                    self._curve_speed_controller.reset()
                    self._temporal_tracker.reset()
                    armed = False
                    end_of_line = True
        path_geometry_config = path_geometry_config_from_dict(
            config.get("path_geometry", {})
        )
        curve_speed_target_scale = curve_speed_controller.target_scale(
            path_geometry.curvature_px_inv,
            curve_threshold_px_inv=path_geometry_config.curve_threshold_px_inv,
            path_valid=path_geometry.valid,
        )
        curve_speed_scale = curve_speed_controller.current_scale
        near_error = None
        far_error = None
        steering_error = None
        steering_output = None
        visual_output = None
        feedback_result = None
        pid_output = None
        heading_feedback = None
        heading_gain = float(control_config.get("heading_gain", 0.25))
        heading_feedback_limit = float(control_config.get("heading_feedback_limit", 0.35))
        active_far_line_weight = 0.0
        if (
            features.state == "VALID"
            and temporal.state == "VALID"
            and temporal.center_x is not None
            and features.ok_near
            and features.consistent
        ):
            x_reference = frame.shape[1] / 2.0
            near_error = normalized_error(
                temporal.center_x, x_reference, features.roi_near.w / 2.0
            )
            if features.ok_far and features.x_far is not None:
                far_error = normalized_error(
                    features.x_far, x_reference, features.roi_far.w / 2.0
                )
            if far_error is not None:
                active_far_line_weight = float(control_config.get("far_line_weight", 0.0))
            if features.centerline:
                steering_error = path_lookahead_error(
                    features.centerline,
                    near_y=features.roi_near.y + features.roi_near.h/2,
                    far_y=features.roi_far.y + features.roi_far.h/2,
                    far_line_weight=float(control_config.get("far_line_weight", 0.0)),
                    reference_x=x_reference, half_width=features.roi_near.w/2,
                )
            else:
                steering_error = combine_line_errors(
                    near_error, far_error, active_far_line_weight
                )
            if path_geometry.valid and path_geometry.heading_deg is not None:
                raw_heading_feedback = (
                    heading_to_turn_input(path_geometry.heading_deg) * heading_gain
                )
                heading_feedback = max(
                    -heading_feedback_limit,
                    min(heading_feedback_limit, raw_heading_feedback),
                )

        if marker.done or marker.failed:
            state = "STOP_MARKER_REACHED" if marker.done else "STOP_MARKER_FAILED"
            message = marker.reason
            can_track = False
        elif marker.active:
            state = "STOP_MARKER_" + marker.state
            message = "Short T confirmed; positioning four-wheel centre at crossbar"
            can_track = True
            with self._lock:
                self._tracking_active = self._tracking_seen_since_arm = True
        elif corner.failed:
            state = "CORNER_FAILED"
            message = corner.reason + "; stopped and disarmed"
            can_track = False
        elif bypass.active:
            state = "OBSTACLE_BYPASS_" + bypass.phase
            message = bypass.message or bypass.reason
            can_track = True
            with self._lock:
                self._tracking_active = self._tracking_seen_since_arm = True
        elif bypass.mode == "STOP":
            state = ("OBSTACLE_BYPASS_FAILED" if bypass.phase == "FAILED"
                     else "OBSTACLE_STOP")
            message = bypass.message or bypass.reason
            can_track = False
        elif end_of_line:
            state = "END_OF_LINE"
            message = "Near-field path unavailable for the configured frames; stopped and disarmed (endpoint not identified)"
            can_track = False
        elif not calibration_ready:
            state = "CALIBRATION_REQUIRED"
            message = "Ground ROI and configured tape-width calibration required; motors locked"
            can_track = False
        elif not armed:
            state = "PREVIEW"
            message = "Preview only; press Start to arm"
            can_track = False
        elif corner.active:
            state = "CORNER_" + corner.state
            message = "Following confirmed L corner; visual exit reacquisition required"
            can_track = True
            with self._lock:
                self._tracking_active = self._tracking_seen_since_arm = True
        elif (
            temporal.state != "VALID"
            or temporal.center_x is None
            or features.state != "VALID"
            or not features.ok_near
            or not features.consistent
        ):
            if temporal.state == "UNCERTAIN" and features.state == "VALID":
                state = "ACQUIRING_LINE"
                confirm_frames = max(1, int(config.get("control", {}).get("line_acquire_frames", 3)))
                message = f"Confirming temporal path: {temporal.reacquire_count}/{confirm_frames} frames"
            elif temporal.state == "UNCERTAIN" or features.state == "UNCERTAIN":
                state = "UNCERTAIN"
                message = "Candidate path is uncertain; steering target rejected"
            else:
                state = "LINE_LOST" if was_tracking else "WAITING_LINE"
                message = "Line lost; stopped" if was_tracking else "Waiting for a valid tape path"
            can_track = False
        else:
            with self._lock:
                if not self._armed:
                    state = "PREVIEW"
                    message = "Stopped; preview only"
                    can_track = False
                else:
                    if not self._tracking_active:
                        self._reset_steering_controllers()
                        log.info("Spatial and temporal line validation passed; tracking started")
                    self._tracking_active = True
                    self._tracking_seen_since_arm = True
                    state = "TRACKING"
                    message = (
                        "Following the near tape only; far preview unavailable, speed reduced"
                        if features.x_far is None
                        else "Following the validated near-and-far tape path"
                    )
                    can_track = True

        if can_track and imu_unavailable and not corner.active and not bypass.active:
            message += "; IMU unavailable, using visual steering"

        if self._navigation_config is not None:
            state = "NAV_"+navigation.owner
            message = navigation.reason
            can_track = navigation.action in {"SETPOINT", "BYPASS"}
            with self._lock:
                self._tracking_active = can_track
                self._tracking_seen_since_arm |= can_track
            dispatched = self._dispatch_navigation(navigation, corner_drive, bypass, command_generation)
            with self._lock:
                if navigation is self._navigation_latest and command_generation == self._command_generation:
                    self._tracking_active = dispatched
                    self._navigation_lost_frames = (self._navigation_lost_frames+1
                        if self._armed and navigation.reason == "tracking_path_unavailable" else 0)
                    if self._navigation_lost_frames >= near_lost_stop_frames:
                        self._armed = False
                        self._command_generation += 1
                        self._tracking_active = False
                        self._reset_corner_motion("navigation_line_lost")
        elif not can_track:
            with self._lock:
                curve_speed_controller.reset()
                curve_speed_scale = curve_speed_controller.current_scale
                self._tracking_active = False if state in {"ACQUIRING_LINE", "UNCERTAIN", "LINE_LOST", "END_OF_LINE", "OBSTACLE_STOP", "OBSTACLE_BYPASS_FAILED", "WAIT_DISTANCE"} else self._tracking_active
                if state in {"ACQUIRING_LINE", "UNCERTAIN", "LINE_LOST", "END_OF_LINE", "OBSTACLE_STOP", "OBSTACLE_BYPASS_FAILED", "WAIT_DISTANCE", "PREVIEW"}:
                    self._acquire_count = 0
                    self._reset_steering_controllers()
            if was_tracking or motor_was_active or end_of_line or corner.failed or marker.done or marker.failed or state in {"OBSTACLE_STOP", "OBSTACLE_BYPASS_FAILED", "WAIT_DISTANCE"}:
                self._force_stop()
        else:
            with self._lock:
                current_config = copy.deepcopy(self._config)
                if bypass.active:
                    self._reset_steering_controllers()
                    steering_output = 0.0
                    curve_speed_scale = bypass.forward_scale
                elif not self._armed:
                    state = "PREVIEW"
                    message = "Stopped; preview only"
                    can_track = False
                elif marker.active:
                    self._reset_steering_controllers()
                    steering_output = 0.0
                    curve_speed_scale = 0.0
                elif steering_error is None and not corner.active:
                    state = "UNCERTAIN"
                    message = "Validated path unavailable to the turning controller"
                    can_track = False
                elif corner.active:
                    self._reset_steering_controllers()
                    steering_output = corner.turn_input if not corner.pivot_duty else corner.pivot_duty / float(control_config.get("turn_duty_max", 25.0))
                    curve_speed_scale = corner.forward_scale
                else:
                    line_error = steering_error
                    pid_output = self._pid.update(line_offset_to_turn_input(line_error), dt)
                    steering_output, heading_feedback = combine_pid_and_heading(
                        pid_output,
                        path_geometry.heading_deg,
                        heading_gain,
                        heading_feedback_limit,
                    )
                    curve_speed_scale = curve_speed_controller.update(
                        curve_speed_target_scale, dt
                    )

            if bypass.active and can_track:
                if bypass.phase == "STOP":
                    self._force_stop()
                else:
                    self._send_wheels(
                        self._obstacle_bypass_wheels(bypass, current_config),
                        command_generation,
                    )
            elif can_track and marker.active:
                if marker.duty:
                    marker_turn = max(-2.0, min(2.0, -(steering_error or 0.0)*4.0))
                    self._send_wheels(mix_forward_turn_duty(marker.duty, marker_turn), command_generation)
                else:
                    self._stop_if_active()
            elif can_track and corner.active and not corner.forward_scale and not corner.pivot_duty:
                self._stop_if_active()
            elif can_track:
                control_config = current_config.get("control", {})
                turn_limit = float(control_config.get("turn_duty_max", 25.0))
                applied_speed_scale = corner.forward_scale if corner.active else curve_speed_scale
                if features.x_far is None and not corner.active:
                    applied_speed_scale *= float(
                        control_config.get("near_only_speed_scale", 0.5)
                    )
                applied_speed_scale *= bypass.speed_limit_scale
                visual_output = steering_output
                if self._yaw_feedback.config.enabled and not corner.active:
                    imu_status = self._imu.snapshot()
                    feedback_result = self._apply_yaw_feedback(
                        steering_output, dt=dt, imu_status=imu_status,
                        forward_duty=float(control_config.get("base_duty", 35.0)) * applied_speed_scale,
                        turn_limit=turn_limit, current_config=current_config,
                    )
                    steering_output = feedback_result.final_turn
                if corner.active and corner.pivot_duty:
                    wheel_values = mix_forward_turn_duty(0.0, corner.pivot_duty)
                elif str(current_config.get("motor", {}).get("mixer", "duty")) == "mecanum":
                    geometry_config = current_config.get("motor", {}).get("mecanum", {})
                    geometry = MecanumGeometry(
                        wheel_radius_m=float(geometry_config.get("wheel_radius_m", 0.03)),
                        wheelbase_x_m=float(geometry_config.get("wheelbase_x_m", 0.10)),
                        wheelbase_y_m=float(geometry_config.get("wheelbase_y_m", 0.10)),
                    )
                    wheels = mecanum_inverse(
                        vx_mps=float(control_config.get("vx_mps", 0.25)) * applied_speed_scale,
                        vy_mps=0.0,
                        wz_rps=steering_output,
                        geom=geometry,
                    )
                    wheels = normalize_wheels(wheels, max_abs=float(control_config.get("wheel_omega_limit", 20.0)))
                    wheel_values = tuple(value * float(current_config.get("motor", {}).get("omega_to_duty", 8.0)) for value in (wheels.fl, wheels.fr, wheels.rl, wheels.rr))
                else:
                    turn = steering_output * turn_limit
                    base_duty = float(control_config.get("base_duty", 35.0))
                    sharp_curve = (
                        not corner.active
                        and path_geometry.valid
                        and path_geometry.heading_deg is not None
                        and path_geometry.curvature_px_inv is not None
                        and abs(path_geometry.heading_deg) >= 45.0
                        and abs(path_geometry.curvature_px_inv)
                        >= 1.15 * curve_speed_controller.config.full_curve_px_inv
                        and turn * path_geometry.heading_deg < 0.0
                        and abs(turn) > base_duty * applied_speed_scale
                    )
                    wheel_values = mix_forward_turn_duty_scaled(
                        base_duty,
                        turn,
                        applied_speed_scale,
                        allow_inner_reverse=sharp_curve,
                    )
                self._send_wheels(wheel_values, command_generation)
                motor_config = current_config.get("motor", {})
                if bool(motor_config.get("trace_commands", True)):
                    trace_now = time.monotonic()
                    trace_interval = max(
                        0.25, float(motor_config.get("trace_interval_s", 0.5))
                    )
                    if trace_now - self._last_drive_trace_at >= trace_interval:
                        self._last_drive_trace_at = trace_now
                        mapping = motor_config.get("mapping", {}) or {}
                        inversion = motor_config.get("invert", {}) or {}
                        wheel_names = ("FL", "FR", "RL", "RR")
                        wheel_text = ",".join(
                            f"{name}:{value:+.1f}"
                            for name, value in zip(wheel_names, wheel_values)
                        )
                        mapping_text = ",".join(
                            f"{name}:M{mapping.get(key, default)}"
                            for name, key, default in (
                                ("FL", "fl", 1),
                                ("FR", "fr", 2),
                                ("RL", "rl", 3),
                                ("RR", "rr", 4),
                            )
                        )
                        inversion_text = ",".join(
                            f"{name}:{inversion.get(key, False)}"
                            for name, key in (
                                ("FL", "fl"),
                                ("FR", "fr"),
                                ("RL", "rl"),
                                ("RR", "rr"),
                            )
                        )
                        log.info(
                            "DRIVE_TRACE state=%s near=%s far=%s heading_deg=%s lateral=%s body_lateral_m=%s body_heading_deg=%s ground_state=%s pid=%s heading_feedback=%s output=%s wheels=[%s] mapping=[%s] invert=[%s] mixer=%s battery_v=%s",
                            state,
                            near_error,
                            far_error,
                            path_geometry.heading_deg,
                            steering_error,
                            ground_path.lateral_error_m,
                            ground_path.heading_error_deg,
                            ground_path.state,
                            pid_output,
                            heading_feedback,
                            steering_output,
                            wheel_text,
                            mapping_text,
                            inversion_text,
                            motor_config.get("mixer", "duty"),
                            self._battery_voltage_v,
                        )
            else:
                state = "PREVIEW"
                message = "Stopped; preview only"

        if vision_config.debug_visualization:
            if path_geometry.valid and len(path_geometry.centerline) > 1:
                cv2.polylines(
                    debug,
                    [np.asarray(path_geometry.centerline, dtype=np.int32)],
                    False,
                    (255, 0, 255),
                    2,
                )
            if path_midpoint is not None:
                cv2.drawMarker(
                    debug,
                    path_midpoint,
                    (255, 0, 0),
                    markerType=cv2.MARKER_DIAMOND,
                    markerSize=12,
                    thickness=2,
                )
            overlay_color = (0, 220, 0) if state == "TRACKING" else (0, 180, 255)
            cv2.putText(debug, f"{state}: {message}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, overlay_color, 2)
            if distance_mm is not None:
                cv2.putText(debug, f"distance={distance_mm} mm", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            heading_text = (
                "--"
                if path_geometry.heading_deg is None
                else f"{path_geometry.heading_deg:+.1f} deg"
            )
            curve_text = (
                "UNKNOWN"
                if path_geometry.curvature_px_inv is None
                else f"{path_geometry.curve_direction} {path_geometry.curvature_px_inv:+.6f} px^-1"
            )
            cv2.putText(
                debug,
                f"path heading={heading_text}",
                (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 0, 255),
                1,
            )
            cv2.putText(
                debug,
                f"image curve={curve_text}",
                (10, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 0, 255),
                1,
            )
            cv2.putText(
                debug,
                f"curve speed target={curve_speed_target_scale:.2f} output={curve_speed_scale:.2f}",
                (10, 142),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 255),
                1,
            )
            nearest_text = (
                "--" if ground_path.nearest_observation_m is None
                else f"{ground_path.nearest_observation_m * 100.0:.1f}cm"
            )
            lateral_text = (
                "--" if ground_path.lateral_error_m is None
                else f"{ground_path.lateral_error_m * 100.0:+.1f}cm"
            )
            ground_heading_text = (
                "--" if ground_path.heading_error_deg is None
                else f"{ground_path.heading_error_deg:+.1f}deg"
            )
            ground_rms_text = (
                "--" if ground_path.fit_rms_m is None
                else f"{ground_path.fit_rms_m * 100.0:.2f}cm"
            )
            ground_color = (0, 220, 0) if ground_path.state == "VALID" else (0, 165, 255)
            cv2.putText(
                debug,
                f"ground {ground_path.state} near={nearest_text} x@front={lateral_text} "
                f"heading={ground_heading_text} rms={ground_rms_text}",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                ground_color,
                1,
            )

        if vision_config.debug_visualization and steering_error is not None:
            far_text = "--" if far_error is None else f"{far_error:+.3f}"
            cv2.putText(
                debug,
                f"near={near_error:+.3f} far={far_text} control={steering_error:+.3f} heading_fb={heading_feedback if heading_feedback is not None else 0.0:+.3f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 255),
                1,
            )

        if vision_config.debug_visualization and self._imu.config.enabled:
            rate, angle = imu_status["imu_yaw_rate_dps"], imu_status["imu_window_yaw_deg"]
            text = (
                f"IMU {imu_status['imu_mode'].upper()} {imu_status['imu_state']} rate={rate:+.2f} deg/s window={angle:+.2f} deg"
                if rate is not None and angle is not None
                else f"IMU {imu_status['imu_mode'].upper()} {imu_status['imu_state']} (no valid data)"
            )
            cv2.putText(debug, text, (10, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 1)

        if self._corner.config.enabled:
            cv2.putText(debug, f"L corner {corner.state} yaw={corner.progress_deg if corner.progress_deg is not None else '--'}",
                        (10, 208), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 180, 255), 1)
            if corner_observation.direction:
                point = (round(corner_observation.x_norm*frame.shape[1]), round(corner_observation.y_norm*frame.shape[0]))
                cv2.circle(debug, point, 9, (0, 180, 255), 2)
                cv2.arrowedLine(debug, point, (point[0]-corner_observation.direction*70, point[1]), (0, 180, 255), 2)
        if vision_config.debug_visualization:
            ground_panel = _ground_debug_panel(
                ground_path,
                self._ground_mapper.ground_plot_bounds_m,
                width=240,
                height=frame.shape[0],
            )
            debug = np.concatenate((debug, ground_panel), axis=1)
        encoded_ok, encoded = cv2.imencode(".jpg", debug, [cv2.IMWRITE_JPEG_QUALITY, 78])
        status = {
            "stop_marker_state": marker.state,
            "stop_marker_observation": asdict(marker_observation),
            "stop_marker_reason": marker.reason,
            "corner_state": corner.state, "corner_direction": corner.direction,
            "corner_progress_deg": corner.progress_deg, "corner_reason": corner.reason,
            "corner_observation": asdict(corner_observation),
            "vision_timestamp_monotonic_s": frame_timestamp_s,
            "visual_turn_hint": None if steering_error is None else line_offset_to_turn_input(steering_error),
            "state": state,
            "armed": self._armed,
            "calibration_ready": calibration_ready,
            "tape_width_min_mm": vision_config.tape_width_min_mm,
            "tape_width_max_mm": vision_config.tape_width_max_mm,
            "camera_ok": True,
            "line_near": features.ok_near,
            "line_far": features.ok_far,
            "line_consistent": features.consistent,
            "vision_state": features.state,
            "track_state": temporal.state,
            "confidence": round(min(features.confidence, temporal.confidence), 3),
            "valid_scanlines": features.valid_scanlines,
            "candidate_scanlines": features.candidate_scanlines,
            "far_candidate_scanlines": features.far_candidate_scanlines,
            "middle_candidate_scanlines": features.middle_candidate_scanlines,
            "near_candidate_scanlines": features.near_candidate_scanlines,
            "near_path_scanlines": features.near_path_scanlines,
            "near_path_confidence": round(features.near_path_confidence, 3),
            "reacquire_count": temporal.reacquire_count,
            "missed_frames": temporal.missed_frames,
            "expected_width_near_px": (
                None if features.expected_width_near_px is None else round(features.expected_width_near_px, 1)
            ),
            "expected_width_near_min_px": (
                None
                if features.expected_width_near_min_px is None
                else round(features.expected_width_near_min_px, 1)
            ),
            "expected_width_near_max_px": (
                None
                if features.expected_width_near_max_px is None
                else round(features.expected_width_near_max_px, 1)
            ),
            "candidate_x_near": None if features.x_near is None else round(features.x_near, 1),
            "candidate_x_far": None if features.x_far is None else round(features.x_far, 1),
            "path_midpoint_x": None if path_midpoint is None else path_midpoint[0],
            "path_midpoint_y": None if path_midpoint is None else path_midpoint[1],
            "x_near": None if not can_track or temporal.center_x is None else round(temporal.center_x, 1),
            "x_far": None if not can_track or features.x_far is None else round(features.x_far, 1),
            "error": None if near_error is None else round(near_error, 4),
            "far_error": None if far_error is None else round(far_error, 4),
            "steering_error": None if steering_error is None else round(steering_error, 4),
            "steering_output": None if steering_output is None else round(steering_output, 4),
            "visual_turn": None if visual_output is None else round(visual_output, 4),
            "imu_feedback_enabled": feedback_result is not None and feedback_result.feedback_active,
            "target_yaw_rate_dps": None if feedback_result is None else feedback_result.target_yaw_rate_dps,
            "yaw_rate_error_dps": None if feedback_result is None else feedback_result.yaw_rate_error_dps,
            "imu_turn_correction": None if feedback_result is None else feedback_result.correction,
            "heading_feedback": None if heading_feedback is None else round(heading_feedback, 4),
            "far_line_weight_active": round(active_far_line_weight, 3),
            "track_heading_deg": (
                None if not path_geometry.valid else round(path_geometry.heading_deg, 2)
            ),
            "path_heading_source": path_geometry.heading_source,
            "path_curvature_px_inv": (
                None
                if path_geometry.curvature_px_inv is None
                else round(path_geometry.curvature_px_inv, 7)
            ),
            "curve_direction": path_geometry.curve_direction,
            "curve_speed_enabled": curve_speed_controller.config.enabled,
            "curve_speed_scale": round(curve_speed_scale, 3),
            "curve_speed_target_scale": round(curve_speed_target_scale, 3),
            "path_fit_rms_px": (
                None if not path_geometry.valid else round(path_geometry.fit_rms_px, 2)
            ),
            "ground_path_state": ground_path.state,
            "ground_path_reason": ground_path.reason,
            "ground_nearest_observation_m": (
                None if ground_path.nearest_observation_m is None
                else round(ground_path.nearest_observation_m, 5)
            ),
            "ground_lateral_error_m": (
                None if ground_path.lateral_error_m is None
                else round(ground_path.lateral_error_m, 5)
            ),
            "ground_heading_error_deg": (
                None if ground_path.heading_error_deg is None
                else round(ground_path.heading_error_deg, 2)
            ),
            "ground_fit_rms_m": (
                None if ground_path.fit_rms_m is None else round(ground_path.fit_rms_m, 5)
            ),
            "ground_extrapolation_m": (
                None if ground_path.extrapolation_m is None
                else round(ground_path.extrapolation_m, 5)
            ),
            "ground_path_points_m": [
                [round(x, 5), round(y, 5)] for x, y in ground_path.mapped_points_m
            ],
            "ground_path_fit_points_m": [
                [round(x, 5), round(y, 5)] for x, y in ground_path.fit_points_m
            ],
            "ground_predicted_point_m": (
                None if ground_path.predicted_point_m is None
                else [round(ground_path.predicted_point_m[0], 5), round(ground_path.predicted_point_m[1], 5)]
            ),
            "ground_path_predicted": ground_path.predicted,
            "ground_corner_geometry": asdict(ground_corner),
            "corner_motion": asdict(corner_motion),
            "corner_approach": asdict(corner_approach),
            "corner_transition": asdict(corner_transition),
            "corner_drive": asdict(corner_drive),
            "corner_position": asdict(corner_position),
            "navigation": asdict(navigation),
            "sensor_contact": asdict(sensor_contact),
            "sensor_contact_samples": contact_samples,
            "line_sensor_samples": [asdict(sample) for sample in sensor_samples],
            "line_sensor_dropped_samples": dropped_samples,
            "line_pose": asdict(line_pose),
            "local_path_plan": asdict(local_path),
            "local_path_state": local_path.state,
            "local_path_reason": local_path.reason,
            "local_path_mode": local_path.mode,
            "local_path_corner_distance_m": local_path.corner_distance_m,
            "local_path_corner_turn_deg": local_path.corner_turn_deg,
            "local_path_steering_curvature_m_inv": local_path.steering_curvature_m_inv,
            "obstacle_enabled": obstacle_enabled,
            "distance_mm": distance_mm,
            "distance_mm_filtered": distance_mm,
            "obstacle_bypass_mode": bypass.mode,
            "obstacle_bypass_phase": bypass.phase,
            "obstacle_bypass_progress_deg": (
                None if bypass.progress_deg is None else round(bypass.progress_deg, 2)
            ),
            "obstacle_bypass_reason": bypass.reason,
            "obstacle_bypass_attempts": bypass.attempts,
            "obstacle_bypass_line_candidate": bypass.line_candidate,
            "line_sensor_channels": list(line_sensor_channels) if line_sensor_channels is not None else None,
            "fps": round(1.0 / dt, 1),
            "message": message,
        }
        with self._frame_condition:
            if not self._armed and status["state"].startswith("CORNER_") and not corner.failed:
                status["state"], status["message"] = self._status["state"], self._status["message"]
            if (not self._armed and status["state"].startswith("OBSTACLE_BYPASS_")
                    and bypass.phase != "FAILED"):
                status["state"], status["message"] = self._status["state"], self._status["message"]
            if command_generation != self._command_generation and (self._corner_motion is not None
                    or self._sensor_contact is not None or self._line_pose_config is not None):
                self._reset_corner_motion("command_generation_changed")
                status["corner_motion"] = self._status["corner_motion"]
                status["corner_approach"] = self._status["corner_approach"]
                status["corner_transition"] = self._status["corner_transition"]
                status["corner_drive"] = self._status["corner_drive"]
                status["corner_position"] = self._status["corner_position"]
                status["navigation"] = self._status["navigation"]
                status["sensor_contact"] = self._status["sensor_contact"]
                status["sensor_contact_samples"] = []
                status["line_pose"] = self._status["line_pose"]
            self._status.update(status)
            if encoded_ok:
                self._jpeg = encoded.tobytes()
                self._frame_id += 1
                self._raw_snapshot = (raw_frame, {
                    "frame_id": self._frame_id,
                    "timestamp_monotonic_s": frame_timestamp_s,
                    "gimbal_pan_pulse_us": gimbal_pan_pulse_us,
                    "gimbal_tilt_pulse_us": gimbal_tilt_pulse_us,
                }) if not self._armed and not self._motor_active else None
            self._frame_condition.notify_all()
            log_status = dict(self._status)
        self._imu.record_frame(log_status, imu_status)
        self._vision_capture.offer(
            frame, status=log_status, imu=imu_status,
            features=features, vision_config=config.get("vision", {}),
        )
        self._test_runs.offer(frame, {**log_status, **imu_status}, run_id=test_run_id)
        if not self._armed and not self._manual_stop_pending:
            self._test_runs.finish(state.lower(), self.get_status())

    def _line_sampling_context(self):
        with self._lock:
            commanded = self._status.get("gimbal_state") in {"HOME_COMMAND_SENT", "MANUAL_POSITION"}
            recording = self._test_runs.snapshot()
            run_id = recording["test_run_id"] if recording["test_record_active"] else None
            key = (self._command_generation, self._line_frame_size,
                   self._status.get("gimbal_pan_pulse_us") if commanded else None,
                   self._status.get("gimbal_tilt_pulse_us") if commanded else None,
                   id(self._ground_mapper), self._config.get("vision", {}).get("ground_calibration_confirmed", False), run_id)
            active = not self._armed or self._corner.state in {"IDLE", "CONFIRM", "APPROACH", "ADVANCE", "SETTLE"}
            return key, active

    def _reset_obstacle_bypass(self, reason: str = "") -> None:
        self._bypass.reset()
        self._last_bypass_phase = "IDLE"
        self._status.update(
            obstacle_bypass_mode="INACTIVE",
            obstacle_bypass_phase="IDLE",
            obstacle_bypass_progress_deg=None,
            obstacle_bypass_reason=reason,
            obstacle_bypass_attempts=0,
            obstacle_bypass_line_candidate=False,
        )

    def _reset_corner_motion(self, reason: str) -> None:
        self._navigation_lost_frames = 0
        self._navigation_latest = NavigationRequest(reason=reason)
        self._status["navigation"] = asdict(self._navigation_latest)
        if self._corner_motion is not None:
            self._corner_motion.reset()
        self._status["corner_motion"] = asdict(CornerMotionEstimate(reason=reason))
        self._status["corner_position"] = asdict(CornerPositionEstimate(reason=reason))
        self._status["corner_approach"] = asdict(CornerApproachDecision(reason=reason))
        if self._corner_transition is not None:
            self._corner_transition.reset(reason)
        self._status["corner_transition"] = asdict(CornerTransitionDecision(reason=reason))
        if self._corner_drive is not None:
            self._corner_drive.reset()
        self._status["corner_drive"] = asdict(CornerDriveCommand(reason=reason))
        if self._sensor_contact is not None:
            self._sensor_contact.reset()
        self._sensor_contact_latest = SensorContactEvidence(reason=reason)
        self._status["sensor_contact"] = asdict(self._sensor_contact_latest)
        self._status["sensor_contact_samples"] = []
        self._status["line_pose"] = asdict(LinePoseEvidence(reason=reason))

    def _reset_steering_controllers(self) -> None:
        self._pid.reset()
        self._yaw_feedback.reset()

    def _apply_yaw_feedback(self, visual_turn: float, *, dt: float, imu_status: dict,
                            forward_duty: float, turn_limit: float, current_config: dict):
        rate = imu_status.get("imu_yaw_rate_dps")
        age = imu_status.get("imu_age_s")
        feedback_config = self._yaw_feedback.config
        if (imu_status.get("imu_healthy") is not True
                or not isinstance(rate, (int, float)) or not math.isfinite(rate)
                or abs(rate) > feedback_config.max_actual_yaw_rate_dps
                or not isinstance(age, (int, float)) or not math.isfinite(age)
                or not 0 <= age <= feedback_config.stale_after_s):
            self._yaw_feedback.reset()
            return YawFeedbackResult(visual_turn, visual_turn)
        if not feedback_config.forward_duty_min <= forward_duty <= feedback_config.forward_duty_max:
            self._yaw_feedback.reset()
            return YawFeedbackResult(visual_turn, visual_turn)
        try:
            result = self._yaw_feedback.update(
                visual_turn, imu_status.get("imu_yaw_rate_dps"), dt_s=dt,
                imu_age_s=imu_status.get("imu_age_s"), imu_healthy=imu_status.get("imu_healthy"),
                path_state="VALID", forward_duty=forward_duty, turn_duty_max=turn_limit,
                motor_profile_key=forward_motor_profile_key(current_config.get("motor", {}), turn_limit),
            )
            if result.stop_required:
                raise YawFeedbackError("Yaw feedback requires a stop")
            return result
        except YawFeedbackError as exc:
            with self._lock:
                self._armed = False
                self._command_generation += 1
                self._tracking_active = False
                self._status.update(state="IMU_ERROR", armed=False, imu_feedback_enabled=False,
                                    target_yaw_rate_dps=None, yaw_rate_error_dps=None,
                                    imu_turn_correction=None, message=str(exc))
            self._force_stop()
            raise

    def _refresh_battery_status(self, now: float) -> None:
        try:
            voltage = self._motor.read_battery_voltage_v() if self._motor is not None else None
        except Exception as exc:
            log.debug("Battery voltage read failed: %s", exc)
            voltage = None

        with self._lock:
            if voltage is not None and math.isfinite(voltage) and voltage > 0:
                self._battery_voltage_v = float(voltage)
                self._battery_voltage_updated_at = now
            age = (
                None
                if self._battery_voltage_updated_at is None
                else max(0.0, now - self._battery_voltage_updated_at)
            )
            fresh_voltage = age is not None and age <= 5.0
            self._status.update(
                battery_voltage_v=round(self._battery_voltage_v, 2) if fresh_voltage else None,
                battery_voltage_age_s=None if age is None else round(age, 1),
            )

    def _camera_failed(self) -> None:
        self._corner.reset()
        self._stop_marker.reset()
        with self._lock:
            self._reset_obstacle_bypass("camera_error")
            self._reset_corner_motion("camera_error")
            was_armed = self._armed
            camera_was_ok = bool(self._status.get("camera_ok"))
            motor_was_active = self._motor_active
            self._armed = False
            self._command_generation += 1
            self._tracking_active = False
            self._acquire_count = 0
            self._near_lost_frames = 0
            self._tracking_seen_since_arm = False
            self._reset_steering_controllers()
            self._curve_speed_controller.reset()
            self._temporal_tracker.reset()
            self._path_geometry.reset()
            self._status.update(
                state="CAMERA_ERROR",
                armed=False,
                curve_speed_enabled=self._curve_speed_controller.config.enabled,
                curve_speed_scale=self._curve_speed_controller.current_scale,
                curve_speed_target_scale=self._curve_speed_controller.current_scale,
                camera_ok=False,
                line_near=False,
                line_far=False,
                line_consistent=False,
                vision_state="LOST",
                track_state="LOST",
                confidence=0.0,
                valid_scanlines=0,
                candidate_scanlines=0,
                reacquire_count=0,
                missed_frames=0,
                expected_width_near_px=None,
                expected_width_near_min_px=None,
                expected_width_near_max_px=None,
                candidate_x_near=None,
                candidate_x_far=None,
                x_near=None,
                x_far=None,
                path_midpoint_x=None,
                path_midpoint_y=None,
                error=None,
                track_heading_deg=None,
                path_heading_source="NONE",
                path_curvature_px_inv=None,
                curve_direction="UNKNOWN",
                path_fit_rms_px=None,
                ground_path_state="UNCERTAIN",
                ground_path_reason="camera_error",
                ground_nearest_observation_m=None,
                ground_lateral_error_m=None,
                ground_heading_error_deg=None,
                ground_fit_rms_m=None,
                ground_extrapolation_m=None,
                ground_path_points_m=[],
                ground_path_predicted=False,
                ground_corner_geometry=asdict(GroundCornerGeometry(reason="camera_error")),
                local_path_plan=asdict(LocalPathPlan(reason="camera_error")),
                local_path_state="UNCERTAIN",
                local_path_reason="camera_error",
                local_path_mode="STOP",
                local_path_corner_distance_m=None,
                local_path_corner_turn_deg=None,
                local_path_steering_curvature_m_inv=None,
                fps=0.0,
                message="Camera read failed; disarmed",
            )
        if was_armed or camera_was_ok or motor_was_active:
            self._force_stop()
        if was_armed:
            log.error("Camera read failed; tracking disarmed")
        self._test_runs.finish("camera_error", self.get_status())

    @staticmethod
    def _obstacle_bypass_wheels(bypass, current_config: dict[str, Any]) -> tuple[float, float, float, float]:
        obstacle = current_config.get("obstacle", {})
        if (not obstacle.get("enabled", False) or not obstacle.get("bypass", {}).get("enabled", False)
                or not bypass.active or bypass.phase == "STOP"):
            return (0.0,) * 4
        control = current_config.get("control", {})
        base_duty = float(control.get("base_duty", 35.0))
        turn_limit = float(control.get("turn_duty_max", 25.0))
        if bypass.phase == "REVERSE":
            duty = -base_duty * float(bypass.reverse_scale)
            return (duty, duty, duty, duty)
        if bypass.phase in {"LEFT_ARC", "RIGHT_ARC", "SEEK_RECOVER"}:
            turn = max(-turn_limit, min(turn_limit, float(bypass.turn_duty)))
            if bypass.forward_scale == 0:
                return mix_forward_turn_duty(0.0, turn)
            return mix_forward_turn_duty_scaled(
                base_duty, turn, float(bypass.forward_scale), allow_inner_reverse=False
            )
        return mix_forward_turn_duty_scaled(
            base_duty, 0.0, float(bypass.forward_scale), allow_inner_reverse=False
        )

    def _run_navigation_watchdog(self):
        period = min(.01, self._navigation_config.max_input_age_s/4)
        while not self._stop_event.wait(period):
            self._check_navigation_deadline(time.monotonic())

    def _check_navigation_deadline(self, now_s):
        with self._lock:
            cfg, request = self._navigation_config, self._navigation_latest
            if (cfg is None or not cfg.execution_enabled or not self._running or not self._armed
                    or not self._motor_active or self._navigation_dispatched_at_s is None
                    or now_s-self._navigation_dispatched_at_s <= min(cfg.max_input_age_s, self._corner_drive.config.max_command_age_s)):
                return False
            generation = self._command_generation
            expired_at = self._navigation_dispatched_at_s
        try:
            return self._dispatch_navigation(request, CornerDriveCommand(state="FAULT",
                reason="navigation_command_deadline_exceeded"), ObstacleBypassCommand(), generation,
                expired_dispatched_at_s=expired_at)
        except Exception:
            log.exception("Navigation command deadline exceeded; stop attempted")
        return True

    def _dispatch_navigation(self, request, command, bypass, command_generation, *, expired_dispatched_at_s=None):
        try:
            with self._motor_lock:
                with self._lock:
                    cfg, mapper = self._navigation_config, self._corner_drive
                    if (cfg is None or request is not self._navigation_latest
                            or command_generation != self._command_generation
                            or not self._running or self._manual_stop_pending):
                        return False
                    if (expired_dispatched_at_s is not None
                            and self._navigation_dispatched_at_s != expired_dispatched_at_s):
                        return False
                    execute = cfg.execution_enabled and mapper.config.execution_enabled
                    if request.fault or command.state == "FAULT":
                        raise ValueError(request.reason if request.fault else command.reason)
                    move = execute and self._armed and request.action != "STOP"
                    if move:
                        now = time.monotonic()
                        if request.timestamp_s is None or not 0 <= now-request.timestamp_s <= cfg.max_input_age_s:
                            raise ValueError("Navigation request expired before dispatch")
                        current = self._config
                        if request.action == "SETPOINT":
                            if (command.requested_at_s != request.timestamp_s
                                    or command.target_yaw_rate_dps != request.yaw_rate_dps
                                    or command.requested_forward_speed_mps != request.forward_speed_mps):
                                if command.state != "HOLD":
                                    raise ValueError("Navigation wheel command differs from current request")
                            if command.state == "READY":
                                mapper.validate_dispatch(command, now_s=now, motor_config=current.get("motor", {}),
                                    turn_duty_max=float(current.get("control", {}).get("turn_duty_max", 25.)))
                                values = command.wheels
                            else:
                                values = (0.,)*4
                        elif request.action == "BYPASS":
                            values = self._obstacle_bypass_wheels(bypass, current)
                            _, clipped = model_a_wheel_commands(model_a_config_from_dict(current.get("motor", {})), values)
                            if clipped or max(abs(v) for v in values) > mapper.config.max_logical_wheel_duty:
                                raise ValueError("Bypass wheel command exceeds unified drive bounds")
                        else:
                            raise ValueError("Unknown navigation action")
                        if any(values):
                            sample = self._imu.snapshot()
                            now = time.monotonic()
                            keys = ("imu_age_s", "imu_timestamp_monotonic_s", "imu_yaw_rate_dps")
                            if (not sample.get("imu_healthy") or not all(isinstance(sample.get(k), (int, float))
                                    and math.isfinite(sample[k]) for k in keys)
                                    or not 0 <= sample["imu_age_s"] <= mapper.config.max_imu_age_s
                                    or abs(now-sample["imu_timestamp_monotonic_s"]) > mapper.config.max_imu_age_s
                                    or abs(sample["imu_yaw_rate_dps"]) > mapper.config.max_actual_yaw_rate_dps):
                                raise IMUError("Fresh bounded IMU required at navigation dispatch")
                            if (request is not self._navigation_latest or command_generation != self._command_generation
                                    or not self._armed or self._manual_stop_pending):
                                return False
                            if not 0 <= now-request.timestamp_s <= min(cfg.max_input_age_s, mapper.config.max_command_age_s):
                                raise ValueError("Navigation request expired before packet write")
                            self._motor_active = True
                            self._motor.set_wheel_duty(fl=values[0], fr=values[1], rl=values[2], rr=values[3])
                            self._navigation_dispatched_at_s = request.timestamp_s
                            self._status.update(forward_duty_command=sum(values)/4,
                                turn_duty_command=(values[1]+values[3]-values[0]-values[2])/4,
                                wheel_duty_commands=dict(zip(("FL", "FR", "RL", "RR"), values)))
                            return True
                    if self._motor_active:
                        self._motor.stop()
                        self._motor_active = False
                        self._navigation_dispatched_at_s = None
                        self._status.update(forward_duty_command=0., turn_duty_command=0.,
                            wheel_duty_commands=dict.fromkeys(("FL", "FR", "RL", "RR"), 0.))
                    return False
        except Exception as exc:
            with self._lock:
                if request is not self._navigation_latest or command_generation != self._command_generation:
                    return False
                self._armed = False
                self._command_generation += 1
                self._tracking_active = False
                self._reset_corner_motion("navigation_error")
                self._status.update(state="NAVIGATION_ERROR", armed=False, message=str(exc))
            self._force_stop()
            raise

    def _dispatch_corner_drive(self, command, command_generation):
        with self._lock:
            mapper = self._corner_drive
            if (self._navigation_config is not None or mapper is None or not mapper.config.execution_enabled
                    or self._corner_transition is None
                    or self._corner_transition.phase not in {"TURN", "SETTLE", "EXIT"}
                    or (command.requested_at_s is not None
                        and command.requested_at_s != self._corner_transition.timestamp_s)
                    or not self._armed or not self._running or self._manual_stop_pending
                    or command_generation != self._command_generation
                    or self._bypass.active or self._bypass.failed
                    or self._status.get("state") in {"OBSTACLE_STOP", "WAIT_DISTANCE", "OBSTACLE_BYPASS_FAILED"}
                    or self._corner.state in {"CONFIRM", "APPROACH", "ADVANCE", "SETTLE", "TURN", "SETTLE_TURN"}):
                return False
            current = copy.deepcopy(self._config)
        if command.state != "FAULT" and (command.state != "READY" or not any(command.wheels)):
            self._stop_if_active(command_generation=command_generation,
                                 corner_requested_at_s=command.requested_at_s)
            return False
        try:
            if command.state == "FAULT":
                raise ValueError("Corner drive mapping fault: "+command.reason)
            reference = self._corner_transition.latest_decision
            if (command.target_yaw_rate_dps != reference.target_yaw_rate_dps
                    or command.requested_forward_speed_mps != reference.forward_speed_mps):
                raise ValueError("Corner drive command differs from current transition setpoint")
            expected_mode = "pivot" if self._corner_transition.phase == "TURN" else "forward"
            if (command.mode != expected_mode
                    or command.requested_at_s != self._corner_transition.timestamp_s):
                raise ValueError("Corner drive setpoint no longer matches transition state")
            mapper.validate_dispatch(command, now_s=time.monotonic(),
                motor_config=current.get("motor", {}),
                turn_duty_max=float(current.get("control", {}).get("turn_duty_max", 25.)))
            return self._send_wheels(command.wheels, command_generation,
                imu_limits=(mapper.config.max_imu_age_s, mapper.config.max_actual_yaw_rate_dps),
                corner_requested_at_s=command.requested_at_s,
                corner_max_age_s=mapper.config.max_command_age_s)
        except Exception as exc:
            with self._lock:
                if (command_generation != self._command_generation
                        or self._manual_stop_pending or self._bypass.active or self._bypass.failed
                        or self._status.get("state") in {"OBSTACLE_STOP", "WAIT_DISTANCE", "OBSTACLE_BYPASS_FAILED"}
                        or self._corner.state in {"CONFIRM", "APPROACH", "ADVANCE", "SETTLE", "TURN", "SETTLE_TURN"}
                        or self._corner_transition.timestamp_s != command.requested_at_s):
                    return False
                self._armed = False
                self._command_generation += 1
                self._tracking_active = False
                self._reset_corner_motion("corner_drive_error")
                self._status.update(state="CORNER_DRIVE_ERROR", armed=False,
                    corner_drive=asdict(CornerDriveCommand(state="FAULT", reason=str(exc))),
                    message=str(exc))
            self._force_stop()
            raise

    def _send_wheels(self, values: tuple[float, float, float, float], command_generation: int,
                     *, imu_limits=None, corner_requested_at_s=None, corner_max_age_s=None) -> bool:
        with self._lock:
            pivoting = self._corner.state == "TURN"
        if pivoting and not self._imu.snapshot()["imu_healthy"]:
            with self._lock:
                self._armed = False
                self._command_generation += 1
                self._tracking_active = False
            self._force_stop()
            raise IMUError("IMU angle unavailable before corner pivot dispatch")
        with self._motor_lock:
            with self._lock:
                if (
                    not self._armed
                    or not self._running
                    or self._motor is None
                    or command_generation != self._command_generation
                ):
                    return False
                if imu_limits is not None:
                    sample = self._imu.snapshot()
                    now = time.monotonic()
                    age_limit, rate_limit = imu_limits
                    keys = ("imu_age_s", "imu_timestamp_monotonic_s", "imu_yaw_rate_dps")
                    if (not sample.get("imu_healthy") or not all(isinstance(sample.get(k), (int, float))
                            and math.isfinite(sample[k]) for k in keys)
                            or not 0 <= sample["imu_age_s"] <= age_limit
                            or abs(now-sample["imu_timestamp_monotonic_s"]) > age_limit
                            or abs(sample["imu_yaw_rate_dps"]) > rate_limit):
                        raise IMUError("Fresh bounded IMU required at corner drive dispatch")
                    if (self._manual_stop_pending or self._bypass.active or self._bypass.failed
                            or self._status.get("state") in {"OBSTACLE_STOP", "WAIT_DISTANCE", "OBSTACLE_BYPASS_FAILED"}
                            or self._corner.state in {"CONFIRM", "APPROACH", "ADVANCE", "SETTLE", "TURN", "SETTLE_TURN"}
                            or command_generation != self._command_generation or not self._armed
                            or self._corner_transition.timestamp_s != corner_requested_at_s
                            or self._corner_transition.phase not in {"TURN", "EXIT"}):
                        return False
                    if not 0 <= now-corner_requested_at_s <= corner_max_age_s:
                        raise ValueError("Corner drive command expired before packet write")
                motor = self._motor
                self._motor_active = True
                if imu_limits is not None:
                    motor.set_wheel_duty(fl=values[0], fr=values[1], rl=values[2], rr=values[3])
            if imu_limits is None:
                motor.set_wheel_duty(fl=values[0], fr=values[1], rl=values[2], rr=values[3])
            forward_duty = sum(values) / 4.0
            turn_duty = (values[1] + values[3] - values[0] - values[2]) / 4.0
            with self._lock:
                self._status.update(
                    forward_duty_command=round(forward_duty, 2),
                    turn_duty_command=round(turn_duty, 2),
                    wheel_duty_commands={
                        name: round(value, 2)
                        for name, value in zip(("FL", "FR", "RL", "RR"), values)
                    },
                )

        return True

    def _stop_if_active(self, *, command_generation=None, corner_requested_at_s=None) -> None:
        with self._lock:
            if not self._motor_active:
                return
        self._force_stop(command_generation=command_generation, corner_requested_at_s=corner_requested_at_s)

    def _force_stop(self, *, command_generation=None, corner_requested_at_s=None) -> None:
        with self._motor_lock:
            with self._lock:
                if command_generation is not None and (
                        command_generation != self._command_generation
                        or self._bypass.active or self._bypass.failed
                        or self._status.get("state") in {"OBSTACLE_STOP", "WAIT_DISTANCE", "OBSTACLE_BYPASS_FAILED"}
                        or self._corner_transition is None
                        or self._corner_transition.timestamp_s != corner_requested_at_s
                        or self._corner_transition.phase not in {"TURN", "SETTLE", "EXIT"}):
                    return
                motor = self._motor
                if motor is None and not self._motor_active:
                    return
            try:
                if motor is None:
                    raise RuntimeError("Motor driver unavailable; physical stop is not confirmed")
                motor.stop()
            except Exception as exc:
                with self._lock:
                    self._armed = False
                    self._command_generation += 1
                    self._tracking_active = False
                    self._near_lost_frames = 0
                    self._tracking_seen_since_arm = False
                    self._motor_active = True
                    self._status.update(
                        state="MOTOR_STOP_ERROR",
                        armed=False,
                        motor_stop_error=str(exc),
                        message="Motor stop command failed; physical stop is not confirmed",
                    )
                log.exception("Motor stop command failed; physical stop is not confirmed")
                raise
            else:
                with self._lock:
                    self._motor_active = False
                    self._navigation_dispatched_at_s = None
                    self._status.update(
                        motor_stop_error=None,
                        imu_feedback_enabled=False,
                        target_yaw_rate_dps=None,
                        yaw_rate_error_dps=None,
                        imu_turn_correction=None,
                        forward_duty_command=0.0,
                        turn_duty_command=0.0,
                        wheel_duty_commands={"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0},
                    )

    def _close_devices(self) -> None:
        with self._lock:
            self._raw_snapshot = None
        sampler_stopped = self._line_sampler is None or self._line_sampler.stop()
        if sampler_stopped:
            self._line_sampler = None
        else:
            log.error("Line-sensor read still in progress; keeping its I2C handle open")
        self._test_runs.stop()
        self._imu.stop()
        self._vision_capture.stop()
        line_sensor = self._line_sensor if sampler_stopped else None
        for device in (line_sensor, self._distance_sensor, self._camera, self._motor):
            if device is None:
                continue
            try:
                device.close()
            except Exception:
                log.exception("Failed to close device %s", type(device).__name__)
        self._distance_sensor = None
        if sampler_stopped:
            self._line_sensor = None
        self._camera = None
        self._motor = None
