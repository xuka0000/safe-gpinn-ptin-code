"""Offline SUMO replay adapter for PTIN traffic overlays.

This module reads SUMO outputs that were already produced outside the live
viewer/session loop. It deliberately does not couple to TraCI or any online
SUMO process.
"""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUMO_REPLAY_SCHEMA_VERSION = "ptin_traffic_simulation_overlay_v0_35"
SUMO_REPLAY_TRUTH_BOUNDARY = "sumo_replay_offline_external_output_not_online_traci_coupling"

DEFAULT_EDGE_MAP_PATH = Path("outputs/python_backend/sumo_microscopic_traffic_validation_20260608/sumo_edge_map.csv")
DEFAULT_EDGE_DATA_PATH = Path("outputs/python_backend/sumo_microscopic_traffic_execution_20260608/ptin_utn_edge_data.csv")
DEFAULT_TRIPINFO_PATH = Path("outputs/python_backend/sumo_microscopic_traffic_execution_20260608/ptin_utn_tripinfo.xml")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _resolve_path(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _empty_trip_summary() -> dict[str, float | int]:
    return {
        "trip_count": 0,
        "mean_duration_s": 0.0,
        "mean_time_loss_s": 0.0,
        "mean_waiting_time_s": 0.0,
    }


def blocked_sumo_replay_overlay(
    *,
    time_s: float,
    blockers: list[str],
    status: str = "blocked_no_sumo_data",
) -> dict[str, Any]:
    return {
        "schema_version": SUMO_REPLAY_SCHEMA_VERSION,
        "source": "blocked_no_sumo_data",
        "status": status,
        "time_s": round(float(time_s), 4),
        "mapped_edge_count": 0,
        "unmapped_edge_count": 0,
        "mean_speed_mps": 0.0,
        "max_waiting_time_s": 0.0,
        "edge_metrics": {},
        "trip_summary": _empty_trip_summary(),
        "blockers": list(blockers),
        "truth_boundary": SUMO_REPLAY_TRUTH_BOUNDARY,
    }


@dataclass(frozen=True)
class _EdgeRecord:
    sumo_edge_id: str
    speed_mps: float
    occupancy: float
    travel_time_s: float
    waiting_time_s: float


class SumoReplayAdapter:
    def __init__(
        self,
        *,
        root: Path | str,
        edge_data_path: Path | str | None = None,
        edge_map_path: Path | str | None = None,
        tripinfo_path: Path | str | None = None,
    ) -> None:
        self.root = Path(root)
        self.edge_data_path = _resolve_path(self.root, edge_data_path) if edge_data_path else self.root / DEFAULT_EDGE_DATA_PATH
        self.edge_map_path = _resolve_path(self.root, edge_map_path) if edge_map_path else self.root / DEFAULT_EDGE_MAP_PATH
        self.tripinfo_path = _resolve_path(self.root, tripinfo_path) if tripinfo_path else self.root / DEFAULT_TRIPINFO_PATH

    @classmethod
    def from_run_config(cls, root: Path | str, run_config: dict[str, Any]) -> "SumoReplayAdapter":
        root_path = Path(root)
        edge_data = (
            run_config.get("sumo_edge_data_path")
            or run_config.get("sumo_edge_data_xml_path")
            or run_config.get("sumo_edge_data_csv_path")
        )
        return cls(
            root=root_path,
            edge_data_path=_resolve_path(root_path, edge_data),
            edge_map_path=_resolve_path(root_path, run_config.get("sumo_edge_map_path")) if run_config.get("sumo_edge_map_path") else root_path / DEFAULT_EDGE_MAP_PATH,
            tripinfo_path=_resolve_path(root_path, run_config.get("sumo_tripinfo_path")) if run_config.get("sumo_tripinfo_path") else root_path / DEFAULT_TRIPINFO_PATH,
        )

    def overlay_for_time(self, *, time_min: float) -> dict[str, Any]:
        time_s = float(time_min) * 60.0
        if self.edge_data_path is None or not self.edge_data_path.exists():
            return blocked_sumo_replay_overlay(time_s=time_s, blockers=["missing_sumo_edge_data"])

        try:
            records = self._records_for_time(time_s)
            if not records:
                return blocked_sumo_replay_overlay(
                    time_s=time_s,
                    blockers=["no_matching_interval"],
                    status="no_matching_interval",
                )
            edge_map = self._load_edge_map()
            return self._build_overlay(time_s, records, edge_map)
        except Exception as exc:  # fail closed on malformed external data
            return blocked_sumo_replay_overlay(
                time_s=time_s,
                blockers=[f"invalid_sumo_data:{type(exc).__name__}"],
                status="blocked_invalid_sumo_data",
            )

    def _records_for_time(self, time_s: float) -> list[_EdgeRecord]:
        suffix = self.edge_data_path.suffix.lower() if self.edge_data_path else ""
        if suffix == ".csv":
            return self._csv_records_for_time(time_s)
        return self._xml_records_for_time(time_s)

    def _xml_records_for_time(self, time_s: float) -> list[_EdgeRecord]:
        assert self.edge_data_path is not None
        root = ET.parse(self.edge_data_path).getroot()
        intervals = list(root.iter("interval"))
        if not intervals:
            return self._records_from_xml_edges(root.iter("edge"))

        chosen: ET.Element | None = None
        for interval in intervals:
            begin = _finite_float(interval.get("begin"))
            end = _finite_float(interval.get("end"))
            if begin <= time_s <= end:
                chosen = interval
                break
        if chosen is None:
            return []
        return self._records_from_xml_edges(chosen.iter("edge"))

    def _records_from_xml_edges(self, edges: Any) -> list[_EdgeRecord]:
        records: list[_EdgeRecord] = []
        for edge in edges:
            edge_id = str(edge.get("id") or "").strip()
            if not edge_id:
                continue
            records.append(
                _EdgeRecord(
                    sumo_edge_id=edge_id,
                    speed_mps=_finite_float(edge.get("speed")),
                    occupancy=_finite_float(edge.get("occupancy")),
                    travel_time_s=_finite_float(edge.get("traveltime") or edge.get("travelTime")),
                    waiting_time_s=_finite_float(edge.get("waitingTime")),
                )
            )
        return records

    def _csv_records_for_time(self, time_s: float) -> list[_EdgeRecord]:
        assert self.edge_data_path is not None
        rows: list[dict[str, str]] = []
        with self.edge_data_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
        if not rows:
            return []
        times = sorted({_finite_float(_first_value(row, "time_s", "time", "begin")) for row in rows})
        closest = min(times, key=lambda value: abs(value - time_s))
        records: list[_EdgeRecord] = []
        for row in rows:
            if _finite_float(_first_value(row, "time_s", "time", "begin")) != closest:
                continue
            edge_id = str(_first_value(row, "sumo_edge_id", "edge_id", "id") or "").strip()
            if not edge_id:
                continue
            records.append(
                _EdgeRecord(
                    sumo_edge_id=edge_id,
                    speed_mps=_finite_float(_first_value(row, "speed_mps", "speed", "speed_m_s")),
                    occupancy=_finite_float(row.get("occupancy")),
                    travel_time_s=_finite_float(_first_value(row, "travel_time_s", "travelTime", "traveltime")),
                    waiting_time_s=_finite_float(row.get("waiting_time_s")),
                )
            )
        return records

    def _load_edge_map(self) -> dict[str, str]:
        if self.edge_map_path is None or not self.edge_map_path.exists():
            return {}
        mapping: dict[str, str] = {}
        with self.edge_map_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sumo_id = str(_first_value(row, "sumo_edge_id", "edge_id", "id") or "").strip()
                source_id = str(_first_value(row, "source_edge_id", "utn_edge_id") or "").strip()
                if sumo_id and source_id:
                    mapping[sumo_id] = source_id
        return mapping

    def _map_edge_id(self, sumo_edge_id: str, edge_map: dict[str, str]) -> str | None:
        if sumo_edge_id in edge_map:
            return edge_map[sumo_edge_id]
        if sumo_edge_id.endswith("_fwd") or sumo_edge_id.endswith("_rev"):
            base = sumo_edge_id.rsplit("_", 1)[0]
            if base.startswith("UTN_"):
                return base
        if sumo_edge_id.startswith("UTN_"):
            return sumo_edge_id
        return None

    def _build_overlay(
        self,
        time_s: float,
        records: list[_EdgeRecord],
        edge_map: dict[str, str],
    ) -> dict[str, Any]:
        grouped: dict[str, list[_EdgeRecord]] = {}
        blockers: list[str] = []
        unmapped_count = 0
        for record in records:
            source_id = self._map_edge_id(record.sumo_edge_id, edge_map)
            if source_id is None:
                unmapped_count += 1
                blockers.append(f"unmapped_sumo_edge:{record.sumo_edge_id}")
                continue
            grouped.setdefault(source_id, []).append(record)

        if not grouped:
            return blocked_sumo_replay_overlay(
                time_s=time_s,
                blockers=blockers or ["no_mappable_sumo_edges"],
                status="blocked_invalid_sumo_data",
            )

        edge_metrics: dict[str, dict[str, float | str]] = {}
        for source_id, items in sorted(grouped.items()):
            speed = sum(item.speed_mps for item in items) / len(items)
            occupancy = sum(item.occupancy for item in items) / len(items)
            travel_time = sum(item.travel_time_s for item in items) / len(items)
            waiting_time = max(item.waiting_time_s for item in items)
            edge_metrics[source_id] = {
                "speed_mps": speed,
                "occupancy": occupancy,
                "travel_time_s": travel_time,
                "waiting_time_s": waiting_time,
                "status": _traffic_status(speed, occupancy, waiting_time),
            }

        speeds = [float(metric["speed_mps"]) for metric in edge_metrics.values()]
        waits = [float(metric["waiting_time_s"]) for metric in edge_metrics.values()]
        return {
            "schema_version": SUMO_REPLAY_SCHEMA_VERSION,
            "source": "sumo_replay",
            "status": "ok",
            "time_s": float(time_s),
            "mapped_edge_count": len(edge_metrics),
            "unmapped_edge_count": unmapped_count,
            "mean_speed_mps": sum(speeds) / len(speeds) if speeds else 0.0,
            "max_waiting_time_s": max(waits) if waits else 0.0,
            "edge_metrics": edge_metrics,
            "trip_summary": self._load_trip_summary(),
            "blockers": blockers,
            "truth_boundary": SUMO_REPLAY_TRUTH_BOUNDARY,
        }

    def _load_trip_summary(self) -> dict[str, float | int]:
        if self.tripinfo_path is None or not self.tripinfo_path.exists():
            return _empty_trip_summary()
        root = ET.parse(self.tripinfo_path).getroot()
        durations: list[float] = []
        waits: list[float] = []
        losses: list[float] = []
        for trip in root.iter("tripinfo"):
            durations.append(_finite_float(trip.get("duration")))
            waits.append(_finite_float(trip.get("waitingTime")))
            losses.append(_finite_float(trip.get("timeLoss")))
        count = len(durations)
        if count == 0:
            return _empty_trip_summary()
        return {
            "trip_count": count,
            "mean_duration_s": sum(durations) / count,
            "mean_time_loss_s": sum(losses) / count,
            "mean_waiting_time_s": sum(waits) / count,
        }


def _traffic_status(speed_mps: float, occupancy: float, waiting_time_s: float) -> str:
    if speed_mps <= 0.1 and occupancy >= 0.9:
        return "blocked"
    if occupancy >= 0.45 or speed_mps < 6.0 or waiting_time_s > 0.0:
        return "congested"
    return "free_flow"


def apply_sumo_replay_overlay(state: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    state["traffic_simulation_overlay"] = overlay
    dashboard = state.setdefault("dashboard", {})
    ok = overlay.get("status") == "ok"
    overlay_source = str(overlay.get("source") or "sumo_replay")
    dashboard.update(
        {
            "traffic_simulation_source": overlay.get("source", "blocked_no_sumo_data"),
            "traffic_simulation_status": overlay.get("status"),
            "traffic_simulation_mapped_edges": overlay.get("mapped_edge_count", 0),
            "traffic_simulation_unmapped_edges": overlay.get("unmapped_edge_count", 0),
            "traffic_simulation_mean_speed_mps": overlay.get("mean_speed_mps", 0.0),
            "traffic_simulation_max_waiting_time_s": overlay.get("max_waiting_time_s", 0.0),
        }
    )

    edge_metrics = overlay.get("edge_metrics") or {}
    for edge in state.get("utn_edges", []):
        edge_id = str(edge.get("id") or "")
        metric = edge_metrics.get(edge_id) if ok else None
        if not metric:
            edge.setdefault("traffic_simulation_source", "calibrated_route_time")
            continue
        speed_mps = _finite_float(metric.get("speed_mps"))
        occupancy = _finite_float(metric.get("occupancy"))
        travel_time_s = _finite_float(metric.get("travel_time_s"))
        waiting_time_s = _finite_float(metric.get("waiting_time_s"))
        status = str(metric.get("status") or _traffic_status(speed_mps, occupancy, waiting_time_s))
        edge.update(
            {
                "traffic_simulation_source": overlay_source,
                "sumo_speed_mps": speed_mps,
                "sumo_occupancy": occupancy,
                "sumo_travel_time_s": travel_time_s,
                "sumo_waiting_time_s": waiting_time_s,
                "speed_km_h": speed_mps * 3.6,
                "travel_time_min": travel_time_s / 60.0,
                "congestion_level": max(_finite_float(edge.get("congestion_level")), occupancy),
            }
        )
        if status in {"congested", "blocked"}:
            edge["state_tag"] = status
        if status == "blocked":
            edge["blocked"] = True
    return state
