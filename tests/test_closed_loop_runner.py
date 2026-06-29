import csv
import json
from pathlib import Path

from ptin_sim.closed_loop.run_closed_loop_experiment import (
    _truth_boundary_for_mode,
    load_run_config,
    run_closed_loop_experiment,
)


def test_closed_loop_runner_writes_reproducible_smoke_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "closed_loop_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "scenario_dir: data/scenario_reconstruction_official_v1",
                "run_name: closed_loop_smoke_test",
                "adapter_mode: fake",
                "seed: 11",
                "episodes: 2",
                "max_steps: 3",
                "step_minutes: 5",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    summary = run_closed_loop_experiment(
        config_path=config_path,
        output_dir=output_dir,
        adapter_mode="fake",
        episodes=2,
        max_steps=3,
    )

    assert summary["episode_count"] == 2
    assert summary["adapter_mode"] == "fake"
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "step_trace.csv").exists()
    assert (output_dir / "episode_metrics.csv").exists()
    assert (output_dir / "config_resolved.json").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["truth_boundary"] == "closed_loop_smoke_fake_adapters_not_final_physical_run"
    with (output_dir / "step_trace.csv").open("r", encoding="utf-8", newline="") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert trace_rows
    assert {"episode", "step_index", "target_id", "reward"}.issubset(trace_rows[0])


def test_closed_loop_runner_can_execute_real_power_adapter_boundary(tmp_path: Path) -> None:
    config_path = tmp_path / "closed_loop_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "scenario_dir: data/scenario_reconstruction_official_v1",
                "run_name: closed_loop_real_power_test",
                "adapter_mode: real_power_fake_traffic_comm",
                "seed: 11",
                "episodes: 1",
                "max_steps: 1",
                "step_minutes: 5",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    summary = run_closed_loop_experiment(
        config_path=config_path,
        output_dir=output_dir,
        episodes=1,
        max_steps=1,
    )

    assert summary["adapter_mode"] == "real_power_fake_traffic_comm"
    assert (
        summary["truth_boundary"]
        == "closed_loop_real_pandapower_ac_opf_with_fake_traffic_comm_not_final_joint_run"
    )
    with (output_dir / "step_trace.csv").open("r", encoding="utf-8", newline="") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert trace_rows[0]["ac_status"] == "opf_ok"
    assert float(trace_rows[0]["served_load_kw"]) > 0.0


def test_closed_loop_runner_config_parses_online_ns3_command(tmp_path: Path) -> None:
    config_path = tmp_path / "closed_loop_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "scenario_dir: data/scenario_reconstruction_official_v1",
                "run_name: closed_loop_online_ns3_test",
                "adapter_mode: real_power_online_traci_online_ns3",
                "ns3_online_command: python scripts/ns3_step.py --time-min {time_min}",
                "ns3_online_workdir: .",
                "ns3_online_timeout_s: 4.5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.adapter_mode == "real_power_online_traci_online_ns3"
    assert config.ns3_online_command == "python scripts/ns3_step.py --time-min {time_min}"
    assert config.ns3_online_timeout_s == 4.5
    assert config.ns3_online_workdir.is_absolute()


def test_closed_loop_runner_labels_online_ns3_as_step_feedback() -> None:
    assert (
        _truth_boundary_for_mode("real_power_online_traci_online_ns3")
        == "closed_loop_real_pandapower_ac_opf_online_traci_and_online_ns3_step_feedback"
    )
    assert "replay" in _truth_boundary_for_mode("real_power_online_traci_ns3_replay")
