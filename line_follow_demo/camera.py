from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraConfig:
    device: int | str = 0
    width: int | None = 640
    height: int | None = 480
    fps: int | None = 30
    reconnect_interval_s: float = 2.0


class Camera:
    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.close()
        cap = cv2.VideoCapture(self._cfg.device)
        try:
            if self._cfg.width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._cfg.width))
            if self._cfg.height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._cfg.height))
            if self._cfg.fps is not None:
                cap.set(cv2.CAP_PROP_FPS, int(self._cfg.fps))
            buffers_accepted = cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logging.getLogger(__name__).info(
                "Camera capture buffers requested=1 accepted=%s actual=%s",
                buffers_accepted, cap.get(cv2.CAP_PROP_BUFFERSIZE),
            )

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera device={self._cfg.device}")

            time.sleep(0.2)
            self._cap = cap
        except Exception:
            cap.release()
            raise

    def reopen(self) -> None:
        self.close()
        self.open()

    def read(self):
        if self._cap is None:
            raise RuntimeError("Camera not opened")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
