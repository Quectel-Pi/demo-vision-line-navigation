from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DemoConfig:
    raw: dict[str, Any]

    @property
    def camera(self) -> dict[str, Any]:
        return dict(self.raw.get("camera", {}))

    @property
    def vision(self) -> dict[str, Any]:
        return dict(self.raw.get("vision", {}))

    @property
    def control(self) -> dict[str, Any]:
        return dict(self.raw.get("control", {}))

    @property
    def motor(self) -> dict[str, Any]:
        return dict(self.raw.get("motor", {}))

    @property
    def obstacle(self) -> dict[str, Any]:
        return dict(self.raw.get("obstacle", {}))


def load_config(path: str | Path) -> DemoConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {config_path}")
    return DemoConfig(raw=data)
