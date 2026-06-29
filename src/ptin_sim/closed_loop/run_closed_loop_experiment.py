from __future__ import annotations

import argparse
import csv
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .communication_ns3 import ClosedLoopCommunicationAdapter, CommunicationStepResult
from .env import ClosedLoopPTINEnv
from .power_ac_opf import ACPowerResult, ClosedLoopACPowerAdapter
from .scenario import load_closed_loop_scenario
from .traffic_traci import ClosedLoopTrafficAdapter, TrafficStepResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "closed_loop_runs"
TRUTH_BOUNDARY_FAKE = "closed_loop_smoke_fake_adapters_not_final_physical_run"
TRUTH_BOUNDARY_REAL_POWER = (
    "closed_loop_real_pandapower_ac_opf_with_fake_traffic_comm_not_final_joint_run"
)
TRUTH_BOUNDARY_REAL_POWER_TRACI = (
    "closed_loop_real_pandapower_ac_opf_and_online_traci_with_fake_comm_not_final_joint_run"
)
TRUTH_BOUNDARY_REAL_POWER_TRACI_NS3_REPLAY = (
    "closed_loop_real_pandapower_ac_opf_online_traci_and_ns3_packet_csv_replay"
)
TRUTH_BOUNDARY_REAL_POWER_TRACI_ONLINE_NS3 = (
    "closed_loop_real_pandapower_ac_opf_online_traci_and_online_ns3_step_feedback"
)
DEFAULT_NS3_PACKET_RESULTS_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "python_backend"
    / "ns3_rf_packet_execution_20260608"
    / "ns3_packet_results.csv"
)


@dataclass(frozen=True)
class ClosedLoopRunConfig:
    scenario_dir: Path = PROJECT_ROOT / "data" / "scenario_reconstruction_official_v1"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_name: str = "closed_loop_ptin_v1"
    adapter_mode: str = "fake"
    seed: int = 7
    episodes: int = 2
    max_steps: int = 4
    step_minutes: float = 5.0
    communication_min_delivery_rate: float = 0.7
    communication_max_mean_delay_ms: float = 250.0
    ns3_replay_results_csv: Path = DEFAULT_NS3_PACKET_RESULTS_CSV
    ns3_online_command: str = ""
    ns3_online_workdir: Path = PROJECT_ROOT
    ns3_online_timeout_s: float = 30.0


class _FakePowerAdapter:
    def run_ac_opf_or_pf(
        self,
        scenario,
        *,
        closed_pdn_edges,
        restored_load_fraction,
        mess_support_kw=0.0,
        v2g_support_kw=0.0,
    ):
        requested_load_kw = round(sum(node.active_power_kw for node in scenario.pdn_nodes) * restored_load_fraction, 6)
        mobile_support_capacity_kw = mess_support_kw + v2g_support_kw
        return ACPowerResult(
            status="fake_opf_ok",
            solver_available=True,
            opf_attempted=True,
            opf_converged=True,
            power_flow_converged=None,
            min_voltage_pu=round(0.96 + restored_load_fraction * 0.02, 6),
            max_voltage_pu=1.02,
            max_line_loading_pct=round(92.0 - restored_load_fraction * 8.0, 6),
            requested_load_kw=requested_load_kw,
            served_load_kw=round(requested_load_kw * 0.9, 6),
            shed_load_kw=round(requested_load_kw * 0.1, 6),
            mobile_support_capacity_kw=round(mobile_support_capacity_kw, 6),
            mobile_dispatch_kw=round(mobile_support_capacity_kw * 0.5, 6),
            blockers=(),
        )


class _FakeTrafficAdapter:
    def step(self, *, time_min):
        return TrafficStepResult(
            status="ok",
            edge_travel_time_s={"UTN_FAKE": 35.0 + time_min},
            edge_speed_mps={"UTN_FAKE": 8.0},
            mean_travel_time_s=35.0 + time_min,
            blockers=(),
        )


class _FakeCommunicationAdapter:
    def step(
        self,
        *,
        time_min,
        target_id="",
        communication_mode="direct",
        target_robust_travel_time_s=0.0,
    ):
        del target_id
        relay_bonus = 0.0
        relay_delay = 0.0
        if communication_mode == "uav_relay":
            relay_bonus = 0.1
            relay_delay = 20.0 + 0.02 * max(0.0, float(target_robust_travel_time_s))
        elif communication_mode == "dual_channel":
            relay_bonus = 0.1
            relay_delay = 12.0 + 0.006 * max(0.0, float(target_robust_travel_time_s))
        return CommunicationStepResult(
            status="ok",
            packet_count=4,
            delivery_rate=min(1.0, 0.9 + relay_bonus),
            mean_delay_ms=25.0 + time_min + relay_delay,
            control_available=True,
            blockers=(),
        )


def run_closed_loop_experiment(
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    adapter_mode: str | None = None,
    episodes: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    config = load_run_config(config_path)
    if adapter_mode is not None:
        config = _replace_config(config, adapter_mode=adapter_mode)
    if episodes is not None:
        config = _replace_config(config, episodes=episodes)
    if max_steps is not None:
        config = _replace_config(config, max_steps=max_steps)
    run_dir = Path(output_dir) if output_dir is not None else config.output_root / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    scenario = load_closed_loop_scenario(config.scenario_dir)
    all_trace_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for episode in range(config.episodes):
        env = _build_env(config, scenario)
        observation = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        while not terminated and not truncated:
            actions = env.available_actions()
            if not actions:
                break
            observation, reward, terminated, truncated, _info = env.step(actions[0])
            total_reward += reward
        for row in env.trace_rows():
            all_trace_rows.append({"episode": episode, **row})
        episode_rows.append(
            {
                "episode": episode,
                "total_reward": round(total_reward, 6),
                "steps": observation["step_index"],
                "restored_failed_edge_count": observation["restored_failed_edge_count"],
                "remaining_failed_edge_count": observation["remaining_failed_edge_count"],
                "terminated": int(terminated),
                "truncated": int(truncated),
            }
        )

    _write_csv(run_dir / "step_trace.csv", all_trace_rows)
    _write_csv(run_dir / "episode_metrics.csv", episode_rows)
    resolved = _config_to_json(config)
    (run_dir / "config_resolved.json").write_text(
        json.dumps(resolved, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "run_name": config.run_name,
        "scenario_id": scenario.scenario_id,
        "adapter_mode": config.adapter_mode,
        "episode_count": config.episodes,
        "max_steps": config.max_steps,
        "step_trace_csv": "step_trace.csv",
        "episode_metrics_csv": "episode_metrics.csv",
        "config_resolved_json": "config_resolved.json",
        "truth_boundary": _truth_boundary_for_mode(config.adapter_mode),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def _build_adapters(config: ClosedLoopRunConfig):
    if config.adapter_mode == "fake":
        power_adapter = _FakePowerAdapter()
        traffic_adapter = _FakeTrafficAdapter()
        communication_adapter = _FakeCommunicationAdapter()
    elif config.adapter_mode == "real_power_fake_traffic_comm":
        power_adapter = ClosedLoopACPowerAdapter()
        traffic_adapter = _FakeTrafficAdapter()
        communication_adapter = _FakeCommunicationAdapter()
    elif config.adapter_mode == "real_power_online_traci_fake_comm":
        power_adapter = ClosedLoopACPowerAdapter()
        traffic_adapter = ClosedLoopTrafficAdapter.from_online_traci_run_config(
            root=PROJECT_ROOT,
            run_config={
                "traffic_simulation_mode": "online_traci",
            },
        )
        communication_adapter = _FakeCommunicationAdapter()
    elif config.adapter_mode in {
        "real_power_online_traci_ns3",
        "real_power_online_traci_ns3_replay",
    }:
        power_adapter = ClosedLoopACPowerAdapter()
        traffic_adapter = ClosedLoopTrafficAdapter.from_online_traci_run_config(
            root=PROJECT_ROOT,
            run_config={
                "traffic_simulation_mode": "online_traci",
            },
        )
        communication_adapter = ClosedLoopCommunicationAdapter.from_ns3_results_csv(
            config.ns3_replay_results_csv,
            min_delivery_rate=config.communication_min_delivery_rate,
            max_mean_delay_ms=config.communication_max_mean_delay_ms,
        )
    elif config.adapter_mode == "real_power_online_traci_online_ns3":
        if not config.ns3_online_command.strip():
            raise ValueError(
                "ns3_online_command is required for "
                "adapter_mode='real_power_online_traci_online_ns3'."
            )
        power_adapter = ClosedLoopACPowerAdapter()
        traffic_adapter = ClosedLoopTrafficAdapter.from_online_traci_run_config(
            root=PROJECT_ROOT,
            run_config={
                "traffic_simulation_mode": "online_traci",
            },
        )
        communication_adapter = ClosedLoopCommunicationAdapter.from_online_ns3_command(
            shlex.split(config.ns3_online_command),
            cwd=config.ns3_online_workdir,
            timeout_s=config.ns3_online_timeout_s,
            min_delivery_rate=config.communication_min_delivery_rate,
            max_mean_delay_ms=config.communication_max_mean_delay_ms,
        )
    else:
        raise ValueError(
            f"Unsupported adapter_mode={config.adapter_mode!r}. "
            "Use fake, real_power_fake_traffic_comm, "
            "real_power_online_traci_fake_comm, "
            "real_power_online_traci_ns3_replay, or "
            "real_power_online_traci_online_ns3."
        )
    return power_adapter, traffic_adapter, communication_adapter


def _build_env(config: ClosedLoopRunConfig, scenario, adapters=None):
    if adapters is None:
        adapters = _build_adapters(config)
    power_adapter, traffic_adapter, communication_adapter = adapters
    return ClosedLoopPTINEnv(
        scenario,
        power_adapter=power_adapter,
        traffic_adapter=traffic_adapter,
        communication_adapter=communication_adapter,
        max_steps=config.max_steps,
        step_minutes=config.step_minutes,
    )


def _truth_boundary_for_mode(adapter_mode: str) -> str:
    if adapter_mode == "fake":
        return TRUTH_BOUNDARY_FAKE
    if adapter_mode == "real_power_fake_traffic_comm":
        return TRUTH_BOUNDARY_REAL_POWER
    if adapter_mode == "real_power_online_traci_fake_comm":
        return TRUTH_BOUNDARY_REAL_POWER_TRACI
    if adapter_mode in {"real_power_online_traci_ns3", "real_power_online_traci_ns3_replay"}:
        return TRUTH_BOUNDARY_REAL_POWER_TRACI_NS3_REPLAY
    if adapter_mode == "real_power_online_traci_online_ns3":
        return TRUTH_BOUNDARY_REAL_POWER_TRACI_ONLINE_NS3
    return "closed_loop_real_adapters_require_dependency_validation"


def load_run_config(path: str | Path | None) -> ClosedLoopRunConfig:
    base = ClosedLoopRunConfig()
    if path is None:
        return base
    raw = _parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    values = _config_to_json(base)
    values.update(raw)
    values["scenario_dir"] = _resolve_path(values["scenario_dir"])
    values["output_root"] = _resolve_path(values["output_root"])
    values["seed"] = int(values["seed"])
    values["episodes"] = int(values["episodes"])
    values["max_steps"] = int(values["max_steps"])
    values["step_minutes"] = float(values["step_minutes"])
    values["communication_min_delivery_rate"] = float(values["communication_min_delivery_rate"])
    values["communication_max_mean_delay_ms"] = float(values["communication_max_mean_delay_ms"])
    values["ns3_replay_results_csv"] = _resolve_path(values["ns3_replay_results_csv"])
    values["ns3_online_workdir"] = _resolve_path(values["ns3_online_workdir"])
    values["ns3_online_timeout_s"] = float(values["ns3_online_timeout_s"])
    return ClosedLoopRunConfig(**values)


def _replace_config(config: ClosedLoopRunConfig, **updates: Any) -> ClosedLoopRunConfig:
    values = _config_to_json(config)
    values.update(updates)
    values["scenario_dir"] = _resolve_path(values["scenario_dir"])
    values["output_root"] = _resolve_path(values["output_root"])
    values["ns3_replay_results_csv"] = _resolve_path(values["ns3_replay_results_csv"])
    values["ns3_online_workdir"] = _resolve_path(values["ns3_online_workdir"])
    return ClosedLoopRunConfig(**values)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip().strip("'\"")
        if value.lower() in {"true", "false"}:
            data[key.strip()] = value.lower() == "true"
        else:
            try:
                data[key.strip()] = int(value)
            except ValueError:
                try:
                    data[key.strip()] = float(value)
                except ValueError:
                    data[key.strip()] = value
    return data


def _resolve_path(value: str | Path) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _config_to_json(config: ClosedLoopRunConfig) -> dict[str, Any]:
    values = asdict(config)
    values["scenario_dir"] = str(config.scenario_dir)
    values["output_root"] = str(config.output_root)
    values["ns3_replay_results_csv"] = str(config.ns3_replay_results_csv)
    values["ns3_online_workdir"] = str(config.ns3_online_workdir)
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adapter-mode", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    manifest = run_closed_loop_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        adapter_mode=args.adapter_mode,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    print("Closed-loop PTIN experiment")
    print(f"- run_name: {manifest['run_name']}")
    print(f"- adapter_mode: {manifest['adapter_mode']}")
    print(f"- episode_count: {manifest['episode_count']}")
    print(f"- truth_boundary: {manifest['truth_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
