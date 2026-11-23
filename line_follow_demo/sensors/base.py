from __future__ import annotations


class DistanceSensor:
    def open(self) -> None:
        raise NotImplementedError

    def read_distance_mm(self) -> int | None:
        raise NotImplementedError

    def close(self) -> None:
        pass
