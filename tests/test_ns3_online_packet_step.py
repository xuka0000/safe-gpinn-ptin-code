import argparse
import subprocess
from pathlib import Path

from ptin_sim.ns3_online_packet_step import run_online_step


def test_ns3_online_packet_step_filters_schedule_and_returns_json(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "ns3_nodes.csv").write_text(
        "\n".join(
            [
                "ns3_node_id,cn_node_id,role,x_m,y_m,coverage_radius_m,demand_kw,power_supply_bus",
                "0,CN_001,center,0,0,3000,5,712",
                "1,CN_002,base_station,100,0,3000,5,708",
            ]
        ),
        encoding="utf-8",
    )
    (input_dir / "ns3_packet_schedule.csv").write_text(
        "\n".join(
            [
                "packet_id,source_trace_id,time_s,src_cn_node,dst_cn_node,payload_bytes,deadline_ms,action_family,scenario_id,truth_boundary",
                "p0,s0,0,CN_001,CN_002,512,250,a,D0,input",
                "p1,s1,300,CN_001,CN_002,512,250,a,D0,input",
                "p1b,s1b,300,CN_001,CN_002,512,250,a,D0,input",
                "p2,s2,600,CN_001,CN_002,512,250,a,D0,input",
            ]
        ),
        encoding="utf-8",
    )
    cpp_source = tmp_path / "ns3_ptin_packet_replay.cc"
    cpp_source.write_text("// fake", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    temp_root = tmp_path / "temp"
    calls = []

    def runner(command: str, *, log_path: Path, run_dir: Path, timeout_s: float):
        calls.append(command)
        if " ./ns3_ptin_packet_replay " in command:
            (run_dir / "ns3_packet_results.csv").write_text(
                "\n".join(
                    [
                        "packet_id,src_cn_node,dst_cn_node,time_s,distance_m,rx_power_dbm,path_loss_db,delay_ms,queue_delay_ms,contention_count,effective_data_rate_mbps,multipath_fading_db,sinr_db,delivered,drop_reason,scenario_id,truth_boundary",
                        "p1,CN_001,CN_002,300,100,-70,110,21,3,4,1.5,0.6,18,1,delivered,D0,online",
                    ]
                ),
                encoding="utf-8",
            )
        return {"returncode": 0, "stdout": "", "stderr": "", "log_path": str(log_path)}

    summary = run_online_step(
        argparse.Namespace(
            input_dir=input_dir,
            output_dir=output_dir,
            cpp_source=cpp_source,
            temp_root=temp_root,
            time_min=5.0,
            time_window_s=1.0,
            max_packets_per_domain=1,
            timeout_s=5.0,
        ),
        command_runner=runner,
    )

    assert len(calls) == 2
    step_schedule = (temp_root / "ptin_ns3_online_step_300" / "ns3_packet_schedule.csv").read_text(
        encoding="utf-8"
    )
    assert "p1" in step_schedule
    assert "p1b" not in step_schedule
    assert "p0" not in step_schedule
    assert "p2" not in step_schedule
    assert summary["status"] == "ns3_online_step_complete"
    assert summary["packets"] == [
        {
            "packet_id": "p1",
            "delivered": True,
            "delay_ms": 21.0,
            "queue_delay_ms": 3.0,
            "contention_count": 4,
            "effective_data_rate_mbps": 1.5,
            "multipath_fading_db": 0.6,
            "sinr_db": 18.0,
        }
    ]


def test_ns3_online_packet_step_reports_timeout_as_blocker(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "ns3_nodes.csv").write_text(
        "ns3_node_id,cn_node_id,role,x_m,y_m,coverage_radius_m,demand_kw,power_supply_bus\n"
        "0,CN_001,center,0,0,3000,5,712\n",
        encoding="utf-8",
    )
    (input_dir / "ns3_packet_schedule.csv").write_text(
        "packet_id,source_trace_id,time_s,src_cn_node,dst_cn_node,payload_bytes,deadline_ms,action_family,scenario_id,truth_boundary\n"
        "p1,s1,300,CN_001,CN_001,512,250,a,D0,input\n",
        encoding="utf-8",
    )
    cpp_source = tmp_path / "ns3_ptin_packet_replay.cc"
    cpp_source.write_text("// fake", encoding="utf-8")

    def runner(command: str, *, log_path: Path, run_dir: Path, timeout_s: float):
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_s)

    summary = run_online_step(
        argparse.Namespace(
            input_dir=input_dir,
            output_dir=tmp_path / "outputs",
            cpp_source=cpp_source,
            temp_root=tmp_path / "temp",
            time_min=5.0,
            time_window_s=1.0,
            max_packets_per_domain=0,
            timeout_s=5.0,
        ),
        command_runner=runner,
    )

    assert summary["status"] == "blocked"
    assert "ns3_online_compile_timeout" in summary["blockers"]
    assert summary["packets"] == []


def test_ns3_online_packet_step_uses_nearest_schedule_when_window_empty(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "ns3_nodes.csv").write_text(
        "\n".join(
            [
                "ns3_node_id,cn_node_id,role,x_m,y_m,coverage_radius_m,demand_kw,power_supply_bus",
                "0,CN_001,center,0,0,3000,5,712",
                "1,CN_002,base_station,100,0,3000,5,708",
            ]
        ),
        encoding="utf-8",
    )
    (input_dir / "ns3_packet_schedule.csv").write_text(
        "\n".join(
            [
                "packet_id,source_trace_id,time_s,src_cn_node,dst_cn_node,payload_bytes,deadline_ms,action_family,scenario_id,truth_boundary",
                "p0,s0,0,CN_001,CN_002,512,250,a,D0,input",
                "p1,s1,300,CN_001,CN_002,512,250,a,D0,input",
            ]
        ),
        encoding="utf-8",
    )
    cpp_source = tmp_path / "ns3_ptin_packet_replay.cc"
    cpp_source.write_text("// fake", encoding="utf-8")

    def runner(command: str, *, log_path: Path, run_dir: Path, timeout_s: float):
        if " ./ns3_ptin_packet_replay " in command:
            (run_dir / "ns3_packet_results.csv").write_text(
                "\n".join(
                    [
                        "packet_id,src_cn_node,dst_cn_node,time_s,distance_m,rx_power_dbm,path_loss_db,delay_ms,queue_delay_ms,contention_count,effective_data_rate_mbps,multipath_fading_db,sinr_db,delivered,drop_reason,scenario_id,truth_boundary",
                        "p1,CN_001,CN_002,300,100,-70,110,21,3,4,1.5,0.6,18,1,delivered,D0,online",
                    ]
                ),
                encoding="utf-8",
            )
        return {"returncode": 0, "stdout": "", "stderr": "", "log_path": str(log_path)}

    summary = run_online_step(
        argparse.Namespace(
            input_dir=input_dir,
            output_dir=tmp_path / "outputs",
            cpp_source=cpp_source,
            temp_root=tmp_path / "temp",
            time_min=20.0,
            time_window_s=1.0,
            max_packets_per_domain=0,
            timeout_s=5.0,
        ),
        command_runner=runner,
    )

    step_schedule = (tmp_path / "temp" / "ptin_ns3_online_step_1200" / "ns3_packet_schedule.csv").read_text(
        encoding="utf-8"
    )
    assert "p1" in step_schedule
    assert "nearest_schedule_time_s" in summary
    assert summary["nearest_schedule_time_s"] == 300.0
    assert summary["status"] == "ns3_online_step_complete"
