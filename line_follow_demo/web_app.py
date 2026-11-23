from __future__ import annotations

import argparse
import json
import logging
import re
import tarfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .runtime import TrackingRuntime


log = logging.getLogger("line_follow_demo.web")
MAX_REQUEST_BYTES = 32 * 1024
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


class DebugHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def create_handler(runtime: TrackingRuntime):
    class DebugHandler(BaseHTTPRequestHandler):
        server_version = "LineFollowDebug/1.0"

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in {"/", "/index.html", "/config", "/config.html"}:
                try:
                    page = (WEB_ROOT / ("config.html" if path in {"/config", "/config.html"} else "index.html")).read_bytes()
                except OSError as exc:
                    self._send_json(500, {"error": f"Unable to load debug page: {exc}"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
            elif path == "/i18n.js":
                try:
                    script = (WEB_ROOT / "i18n.js").read_bytes()
                except OSError as exc:
                    self._send_json(500, {"error": f"Unable to load UI language script: {exc}"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(script)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(script)
            elif path == "/api/status":
                self._send_json(200, runtime.get_status())
            elif path == "/api/camera/raw.png":
                snapshot = runtime.raw_frame_png()
                if snapshot is None:
                    self._send_json(503, {"error": "Fresh static raw frame unavailable"})
                    return
                info, png = snapshot
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Id", str(info["frame_id"]))
                self.send_header("X-Frame-Monotonic-S", str(info["timestamp_monotonic_s"]))
                for axis in ("pan", "tilt"):
                    value = info[f"gimbal_{axis}_pulse_us"]
                    if value is not None:
                        self.send_header(f"X-Gimbal-{axis.title()}-Pulse-Us", str(value))
                self.end_headers()
                self.wfile.write(png)
            elif path == "/api/config":
                self._send_json(200, runtime.get_debug_config())
            elif path.startswith("/api/test-runs/") and path.endswith(".tar"):
                run_id = path.removeprefix("/api/test-runs/").removesuffix(".tar")
                if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z_[0-9a-f]{12}", run_id):
                    self._send_json(404, {"error": "Unknown test run"})
                    return
                status = runtime.get_status()
                if status["armed"] or status["motor_active"]:
                    self._send_json(409, {"error": "Stop the car before downloading evidence"})
                    return
                folder = runtime._test_runs.directory / run_id
                if not (folder / "summary.json").is_file():
                    self._send_json(404, {"error": "Run is not finalized"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-tar")
                self.send_header("Content-Disposition", f'attachment; filename="{run_id}.tar"')
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    with tarfile.open(fileobj=self.wfile, mode="w|") as archive:
                        archive.add(folder, arcname=run_id)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            elif path == "/stream.mjpg":
                self._stream_frames()
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path not in {
                "/api/config",
                "/api/start",
                "/api/stop",
                "/api/gimbal/home",
                "/api/gimbal/move",
            }:
                self._send_json(404, {"error": "Not found"})
                return
            origin = self.headers.get("Origin")
            host = self.headers.get("Host")
            origin_parts = urlsplit(origin) if origin else None
            if origin_parts is None or origin_parts.scheme != "http" or origin_parts.netloc != host:
                self._send_json(403, {"error": "Cross-origin requests are not allowed"})
                return
            if path == "/api/start":
                try:
                    self._send_json(200, runtime.arm())
                except RuntimeError as exc:
                    self._send_json(409, {"error": str(exc)})
                return
            if path == "/api/stop":
                try:
                    self._send_json(200, runtime.disarm())
                except Exception as exc:
                    log.exception("Motor stop request failed")
                    self._send_json(
                        503,
                        {**runtime.get_status(), "error": f"Motor stop command failed: {exc}"},
                    )
                return
            if path == "/api/gimbal/home":
                try:
                    self._send_json(200, runtime.home_gimbal())
                except RuntimeError as exc:
                    self._send_json(409, {"error": str(exc)})
                except Exception as exc:
                    log.exception("Gimbal home request failed")
                    self._send_json(500, {"error": f"Unable to home gimbal: {exc}"})
                return
            if path == "/api/gimbal/move":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > MAX_REQUEST_BYTES:
                        raise ValueError(
                            f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes"
                        )
                    payload = json.loads(self.rfile.read(length))
                    self._send_json(200, runtime.move_gimbal(payload))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._send_json(400, {"error": str(exc)})
                except RuntimeError as exc:
                    self._send_json(409, {"error": str(exc)})
                except Exception as exc:
                    log.exception("Gimbal move request failed")
                    self._send_json(500, {"error": f"Unable to move gimbal: {exc}"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError(f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                self._send_json(200, runtime.update_config(payload))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
            except OSError as exc:
                log.exception("Failed to persist debug settings")
                self._send_json(500, {"error": f"Unable to save settings: {exc}"})

        def _send_json(self, status: int, data: dict) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream_frames(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Connection", "close")
            self.end_headers()
            frame_id = 0
            try:
                while runtime.get_status().get("running"):
                    next_id, jpeg = runtime.next_frame(frame_id, timeout=1.0)
                    if jpeg is None or next_id == frame_id:
                        continue
                    frame_id = next_id
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def log_message(self, format_string: str, *args) -> None:
            log.info("%s - %s", self.client_address[0], format_string % args)

    return DebugHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Web debug interface for visual line following")
    parser.add_argument("--config", default="config/demo.json", help="Path to the demo JSON configuration")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8081, help="HTTP port")
    parser.add_argument("--auto-start", action="store_true", help="Arm tracking after startup (for systemd only)")
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    runtime = TrackingRuntime(args.config)
    server = DebugHTTPServer((args.host, args.port), create_handler(runtime))
    try:
        runtime.start(auto_start=args.auto_start)
        log.info("Debug UI listening on http://%s:%d/; auto_start=%s", args.host, args.port, args.auto_start)
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping web debug service")
    finally:
        server.server_close()
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
