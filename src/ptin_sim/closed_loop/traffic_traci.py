from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class TrafficStepResult:
    status: str
    edge_travel_time_s: dict[str, float]
    edge_speed_mps: dict[str, float]
    mean_travel_time_s: float
    blockers: tuple[str, ...] = field(default_factory=tuple)


class ClosedLoopTrafficAdapter:
    def __init__(
        self,
        *,
        overlay_provider: Callable[[float], dict[str, Any]] | None = None,
        closeable: Any | None = None,
    ) -> None:
        self.overlay_provider = overlay_provider
        self._closeable = closeable

    @classmethod
    def from_online_traci_run_config(
        cls,
        *,
        root: str | Path,
        run_config: dict[str, Any],
    ) -> "ClosedLoopTrafficAdapter":
        from ptin_sim.sumo_online_traci_adapter import SumoOnlineTraciAdapter

        adapter = SumoOnlineTraciAdapter.from_run_config(root=Path(root), run_config=run_config)
        return cls(overlay_provider=adapter.overlay_for_time, closeable=adapter)

    @classmethod
    def from_online_traci_adapter(cls, adapter: Any) -> "ClosedLoopTrafficAdapter":
        return cls(overlay_provider=adapter.overlay_for_time, closeable=adapter)

    def close(self) -> None:
        close = getattr(self._closeable, "close", None)
        if callable(close):
            close()

    def step(self, *, time_min: float) -> TrafficStepResult:
        if self.overlay_provider is None:
            return TrafficStepResult(
                status="blocked_no_traci_provider",
                edge_travel_time_s={},
                edge_speed_mps={},
                mean_travel_time_s=0.0,
                blockers=("online_traci_provider_missing",),
            )
        try:
            try:
                overlay = self.overlay_provider(time_min=time_min)
            except TypeError:
                overlay = self.overlay_provider(time_min)
        except Exception as exc:
            return TrafficStepResult(
                status="blocked_traci_error",
                edge_travel_time_s={},
                edge_speed_mps={},
                mean_travel_time_s=0.0,
                blockers=(f"overlay_provider_failed:{type(exc).__name__}",),
            )
        if overlay.get("status") != "ok":
            return TrafficStepResult(
                status=str(overlay.get("status") or "blocked_traci_overlay"),
                edge_travel_time_s={},
                edge_speed_mps={},
                mean_travel_time_s=0.0,
                blockers=tuple(str(item) for item in overlay.get("blockers", ())),
            )
        edge_metrics = overlay.get("edge_metrics") or {}
        travel_times = {
            str(edge_id): float(metric.get("travel_time_s", 0.0))
            for edge_id, metric in edge_metrics.items()
            if isinstance(metric, dict)
        }
        speeds = {
            str(edge_id): float(metric.get("speed_mps", 0.0))
            for edge_id, metric in edge_metrics.items()
            if isinstance(metric, dict)
        }
        mean = sum(travel_times.values()) / len(travel_times) if travel_times else 0.0
        return TrafficStepResult(
            status="ok",
            edge_travel_time_s=travel_times,
            edge_speed_mps=speeds,
            mean_travel_time_s=mean,
            blockers=(),
        )
