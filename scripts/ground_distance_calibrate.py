from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from line_follow_demo.camera import Camera, CameraConfig
from line_follow_demo.config import load_config
from line_follow_demo.ground_calibration import GroundPlaneMapper


def _camera_from_config(config: dict) -> CameraConfig:
    camera = config.get("camera", {})
    device = camera.get("device", camera.get("index", 0))
    if not isinstance(device, str) or device.isdigit():
        device = int(device)
    return CameraConfig(
        device=device,
        width=None if camera.get("width") is None else int(camera.get("width", 640)),
        height=None if camera.get("height") is None else int(camera.get("height", 480)),
        fps=None if camera.get("fps") is None else int(camera.get("fps", 30)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect measured ground-plane correspondences; this tool does not move the gimbal or motors."
    )
    parser.add_argument("--config", default="config/demo.json")
    parser.add_argument("--output", default="config/ground_distance_calibration.json")
    parser.add_argument("--pan-pwm", type=int, required=True, help="Current pan PWM command in microseconds")
    parser.add_argument("--tilt-pwm", type=int, required=True, help="Current tilt PWM command in microseconds")
    args = parser.parse_args()

    config = load_config(args.config)
    camera = Camera(_camera_from_config({"camera": config.camera}))
    image_points: list[list[float]] = []
    ground_points: list[list[float]] = []
    pending: list[tuple[int, int] | None] = [None]

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending[0] = (x, y)

    output_path = Path(args.output)
    camera.open()
    try:
        cv2.namedWindow("ground_distance_calibration")
        cv2.setMouseCallback("ground_distance_calibration", on_mouse)
        print("Camera is preview-only. Keep the vehicle and camera fixed; do not start the motors.")
        print("Place measured floor marks relative to the front-edge centre. Click each mark, press 'a', then enter x_m,y_m.")
        print("Use at least four non-collinear marks. +X is chassis-right; +Y is forward. Press 'u' to undo, 's' to validate/save, 'q' to quit.")
        frame_size: tuple[int, int] | None = None
        while True:
            frame = camera.read()
            if frame is None:
                continue
            height, width = frame.shape[:2]
            frame_size = (width, height)
            debug = frame.copy()
            for (x_px, y_px), (x_m, y_m) in zip(image_points, ground_points):
                cv2.circle(debug, (round(x_px), round(y_px)), 5, (0, 255, 0), 2)
                cv2.putText(debug, f"({x_m:+.3f},{y_m:.3f})m", (round(x_px) + 6, round(y_px) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            if pending[0] is not None:
                cv2.drawMarker(debug, pending[0], (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(debug, f"points={len(image_points)}  a:add  u:undo  s:save  q:quit",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("ground_distance_calibration", debug)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("a") and pending[0] is not None:
                x_px, y_px = pending[0]
                raw = input(f"Image ({x_px},{y_px}); enter measured chassis x_m,y_m from front centre: ").strip()
                try:
                    x_m, y_m = (float(value.strip()) for value in raw.split(",", maxsplit=1))
                except (ValueError, TypeError):
                    print("Expected two measured numbers separated by a comma; point was not added.")
                    continue
                image_points.append([float(x_px), float(y_px)])
                ground_points.append([x_m, y_m])
                pending[0] = None
                print(f"Added measured correspondence #{len(image_points)}.")
            elif key == ord("u") and image_points:
                image_points.pop()
                ground_points.pop()
                print(f"Removed last point; {len(image_points)} remain.")
            elif key == ord("s"):
                if frame_size is None or len(image_points) < 4:
                    print("At least four measured correspondences are required.")
                    continue
                values = {
                    "_comment": "Measured floor correspondences only. Coordinates are relative to the front-edge centre; +X right, +Y forward.",
                    "schema_version": 1,
                    "enabled": True,
                    "reference_resolution": {"width": frame_size[0], "height": frame_size[1]},
                    "camera_pose_pwm_us": {"pan": args.pan_pwm, "tilt": args.tilt_pwm},
                    "image_points_px": image_points,
                    "ground_points_m": ground_points,
                    "row_distance_samples": sorted(
                        [
                            {"row_px": image_point[1], "distance_m": ground_point[1]}
                            for image_point, ground_point in zip(image_points, ground_points)
                            if ground_point[0] == 0.0
                        ],
                        key=lambda sample: sample["row_px"],
                    ),
                    "min_path_confidence": 0.65,
                    "max_fit_rms_m": None,
                    "max_extrapolation_m": 0.0,
                    "collected_utc": datetime.now(timezone.utc).isoformat(),
                }
                mapper = GroundPlaneMapper(values)
                if not mapper.available:
                    print(f"Calibration rejected: {mapper.reason}")
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"Saved measured calibration to {output_path}. Prediction remains disabled until fit and extrapolation limits are set.")
                break
            elif key == ord("q") or key == 27:
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
