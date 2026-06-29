"""Online SUMO TraCI adapter for PTIN traffic overlays.

The adapter is fail-closed by default. It only reports online traffic metrics
when a TraCI worker returns edge-level data for the requested live time.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


SUMO_ONLINE_TRACI_SCHEMA_VERSION = "ptin_traffic_simulation_overlay_v0_40"
SUMO_ONLINE_TRACI_TRUTH_BOUNDARY = (
    "online_traci_sumo_wsl_step_coupling_over_reconstructed_utn;"
    "not_osm_calibrated;not_closed_loop_policy_advantage_evidence"
)

DEFAULT_EDGE_MAP_PATH = Path("outputs/python_backend/sumo_microscopic_traffic_validation_20260608/sumo_edge_map.csv")
DEFAULT_NET_PATH = Path("outputs/python_backend/sumo_microscopic_traffic_execution_20260608/ptin_utn.net.xml")
DEFAULT_ROUTE_PATH = Path("outputs/python_backend/sumo_microscopic_traffic_execution_20260608/ptin_utn.rou.xml")
DEFAULT_WORKER_SCRIPT_PATH = Path("src/ptin_sim/sumo_online_traci_worker.py")
DEFAULT_WSL_DISTRO = "Ubuntu-22.04"
DEFAULT_TRACI_STEP_LENGTH_S = 1.0
TRACI_JSON_PREFIX = "PTIN_TRACI_JSON "


class TraCIWorker(Protocol):
    def request(self, payload: dict[str, object]) -> dict[str, object]:
        ...

    def close(self) -> None:
        ...


def _empty_trip_summary() -> dict[str, float | int]:
    return {
        "trip_count": 0,
        "mean_duration_s": 0.0,
        "mean_time_loss_s": 0.0,
        "mean_waiting_time_s": 0.0,
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _resolve_path(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def windows_path_to_wsl_path(path: Path | str) -> str:
    text = str(path)
    normalized = text.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        drive = normalized[0].lower()
        rest = normalized[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return normalized


def build_wsl_traci_worker_command(
    *,
    worker_script_path: Path | str,
    distro: str = DEFAULT_WSL_DISTRO,
) -> list[str]:
    return [
        "wsl.exe",
        "-d",
        distro,
        "-u",
        "root",
        "--exec",
        "env",
        "PYTHONPATH=/usr/share/sumo/tools",
        "python3",
        "-u",
        windows_path_to_wsl_path(worker_script_path),
    ]


def _default_process_factory(command: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


class WslTraciJsonWorker:
    def __init__(
        self,
        *,
        net_path: Path | str,
        route_path: Path | str,
        worker_script_path: Path | str,
        process_factory: Any = _default_process_factory,
        distro: str = DEFAULT_WSL_DISTRO,
        step_length_s: float = DEFAULT_TRACI_STEP_LENGTH_S,
    ) -> None:
        self.net_path = Path(net_path)
        self.route_path = Path(route_path)
        self.worker_script_path = Path(worker_script_path)
        self.process_factory = process_factory
        self.distro = distro
        self.step_length_s = float(step_length_s)
        self._process: Any | None = None
        self._closed = False

    def request(self, payload: dict[str, object]) -> dict[str, object]:
        self._ensure_process()
        return self._send(payload)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                self._send({"command": "close"})
        except Exception:
            pass
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        command = build_wsl_traci_worker_command(
            worker_script_path=self.worker_script_path,
            distro=self.distro,
        )
        self._process = self.process_factory(command)
        init_response = self._send(
            {
                "command": "init",
                "net_path": windows_path_to_wsl_path(self.net_path),
                "route_path": windows_path_to_wsl_path(self.route_path),
                "step_length_s": self.step_length_s,
            }
        )
        if init_response.get("status") != "ready":
            error = init_response.get("error") or init_response.get("message") or "unknown"
            raise RuntimeError(f"TraCI worker init failed: {error}")

    def _send(self, payload: dict[str, object]) -> dict[str, object]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("TraCI worker process is not available")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("TraCI worker closed stdout without response")
            payload_text = line
            if line.startswith(TRACI_JSON_PREFIX):
                payload_text = line[len(TRACI_JSON_PREFIX) :]
            else:
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(response, dict):
                    continue
                return response
            response = json.loads(payload_text)
            if not isinstance(response, dict):
                raise RuntimeError("TraCI worker returned non-object JSON")
            return response


def blocked_sumo_online_traci_overlay(
    *,
    time_s: float,
    blockers: list[str],
    status: str = "blocked_online_traci",
) -> dict[str, Any]:
    return {
        "schema_version": SUMO_ONLINE_TRACI_SCHEMA_VERSION,
        "source": "blocked_online_traci",
        "status": status,
        "time_s": round(float(time_s), 4),
        "mapped_edge_count": 0,
        "unmapped_edge_count": 0,
        "mean_speed_mps": 0.0,
        "max_waiting_time_s": 0.0,
        "edge_metrics": {},
        "trip_summary": _empty_trip_summary(),
        "sumo_min_expected_number": 0,
        "blockers": list(blockers),
        "truth_boundary": SUMO_ONLINE_TRACI_TRUTH_BOUNDARY,
    }


@dataclass(frozen=True)
class _EdgeRecord:
    sumo_edge_id: str
    speed_mps: float
    occupancy: float
    travel_time_s: float
    waiting_time_s: float


class SumoOnlineTraciAdapter:
    def __init__(
        self,
        *,
        root: Path | str,
        edge_map_path: Path | str | None = None,
        worker: TraCIWorker | None = None,
        enabled: bool = True,
        startup_blockers: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.edge_map_path = (
            _resolve_path(self.root, edge_map_path)
            if edge_map_path
            else self.root / DEFAULT_EDGE_MAP_PATH
        )
        self._worker = worker
        self.enabled = enabled
        self._startup_blockers = list(startup_blockers or [])

    @classmethod
    def from_run_config(cls, root: Path | str, run_config: dict[str, Any]) -> "SumoOnlineTraciAdapter":
        root_path = Path(root)
        enabled = online_traci_enabled(run_config)
        edge_map_path = (
            _resolve_path(root_path, run_config.get("sumo_edge_map_path"))
            if run_config.get("sumo_edge_map_path")
            else root_path / DEFAULT_EDGE_MAP_PATH
        )
        net_path = (
            _resolve_path(root_path, run_config.get("sumo_online_net_path"))
            if run_config.get("sumo_online_net_path")
            else root_path / DEFAULT_NET_PATH
        )
        route_path = (
            _resolve_path(root_path, run_config.get("sumo_online_route_path"))
            if run_config.get("sumo_online_route_path")
            else root_path / DEFAULT_ROUTE_PATH
        )
        worker_script_path = (
            _resolve_path(root_path, run_config.get("sumo_online_worker_script_path"))
            if run_config.get("sumo_online_worker_script_path")
            else root_path / DEFAULT_WORKER_SCRIPT_PATH
        )
        startup_blockers: list[str] = []
        worker: TraCIWorker | None = None
        if enabled:
            if net_path is None or not net_path.exists():
                startup_blockers.append("missing_sumo_online_net_path")
            if route_path is None or not route_path.exists():
                startup_blockers.append("missing_sumo_online_route_path")
            if worker_script_path is None or not worker_script_path.exists():
                startup_blockers.append("missing_sumo_online_worker_script")
            if not startup_blockers:
                worker = WslTraciJsonWorker(
                    net_path=net_path,
                    route_path=route_path,
                    worker_script_path=worker_script_path,
                    distro=str(run_config.get("sumo_wsl_distro") or DEFAULT_WSL_DISTRO),
                    step_length_s=_finite_float(
                        run_config.get("sumo_online_step_length_s"),
                        DEFAULT_TRACI_STEP_LENGTH_S,
                    ),
                )
        return cls(
            root=root_path,
            edge_map_path=edge_map_path,
            worker=worker,
            enabled=enabled,
            startup_blockers=startup_blockers,
        )

    def overlay_for_time(self, *, time_min: float) -> dict[str, Any]:
        time_s = float(time_min) * 60.0
        if not self.enabled:
            return blocked_sumo_online_traci_overlay(
                time_s=time_s,
                blockers=["sumo_online_traci_disabled"],
            )
        if self._startup_blockers:
            return blocked_sumo_online_traci_overlay(
                time_s=time_s,
                blockers=self._startup_blockers,
                status="blocked_online_traci_dependency",
            )
        if self._worker is None:
            return blocked_sumo_online_traci_overlay(
                time_s=time_s,
                blockers=["sumo_online_traci_worker_unavailable"],
            )
        try:
            response = self._worker.request({"command": "step_to", "target_s": time_s})
        except Exception as exc:
            return blocked_sumo_online_traci_overlay(
                time_s=time_s,
                blockers=[f"worker_request_failed:{type(exc).__name__}"],
                status="blocked_online_traci_error",
            )
        if response.get("status") != "ok":
            error = str(response.get("error") or "unknown")
            return blocked_sumo_online_traci_overlay(
                time_s=time_s,
                blockers=[f"worker_error:{error}"],
                status="blocked_online_traci_error",
            )
        records = self._records_from_response(response)
        if not records:
            return blocked_sumo_online_traci_overlay(
                time_s=_finite_float(response.get("time_s"), time_s),
                blockers=["no_online_traci_edge_metrics"],
                status="blocked_online_traci_empty",
            )
        return self._build_overlay(
            time_s=_finite_float(response.get("time_s"), time_s),
            records=records,
            edge_map=self._load_edge_map(),
            min_expected_number=int(_finite_float(response.get("min_expected_number"), 0.0)),
        )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()

    def _records_from_response(self, response: dict[str, Any]) -> list[_EdgeRecord]:
        records: list[_EdgeRecord] = []
        for row in response.get("edge_metrics") or []:
            if not isinstance(row, dict):
                continue
            edge_id = str(_first_value(row, "sumo_edge_id", "edge_id", "id") or "").strip()
            if not edge_id:
                continue
            occupancy = _finite_float(row.get("occupancy"))
            if occupancy > 1.0:
                occupancy /= 100.0
            records.append(
                _EdgeRecord(
                    sumo_edge_id=edge_id,
                    speed_mps=_finite_float(_first_value(row, "speed_mps", "speed", "speed_m_s")),
                    occupancy=occupancy,
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
        *,
        time_s: float,
        records: list[_EdgeRecord],
        edge_map: dict[str, str],
        min_expected_number: int,
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
            return blocked_sumo_online_traci_overlay(
                time_s=time_s,
                blockers=blockers or ["no_mappable_online_traci_edges"],
                status="blocked_online_traci_invalid_data",
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
            "schema_version": SUMO_ONLINE_TRACI_SCHEMA_VERSION,
            "source": "online_traci",
            "status": "ok",
            "time_s": float(time_s),
            "mapped_edge_count": len(edge_metrics),
            "unmapped_edge_count": unmapped_count,
            "mean_speed_mps": sum(speeds) / len(speeds) if speeds else 0.0,
            "max_waiting_time_s": max(waits) if waits else 0.0,
            "edge_metrics": edge_metrics,
            "trip_summary": _empty_trip_summary(),
            "sumo_min_expected_number": min_expected_number,
            "blockers": blockers,
            "truth_boundary": SUMO_ONLINE_TRACI_TRUTH_BOUNDARY,
        }


def online_traci_enabled(run_config: dict[str, Any]) -> bool:
    mode = str(
        run_config.get("traffic_simulation_mode")
        or run_config.get("sumo_traffic_mode")
        or run_config.get("sumo_mode")
        or ""
    ).strip().lower()
    return mode == "online_traci" or bool(run_config.get("sumo_online_traci_enabled"))


def _traffic_status(speed_mps: float, occupancy: float, waiting_time_s: float) -> str:
    if speed_mps <= 0.1 and occupancy >= 0.9:
        return "blocked"
    if occupancy >= 0.45 or speed_mps < 6.0 or waiting_time_s > 0.0:
        return "congested"
    return "free_flow"
