from pathlib import Path
import json
import subprocess

from ptin_sim.closed_loop.communication_ns3 import (
    ClosedLoopCommunicationAdapter,
    PacketEvidence,
)
from ptin_sim.closed_loop.power_ac_opf import ClosedLoopACPowerAdapter
from ptin_sim.closed_loop.scenario import load_closed_loop_scenario
from ptin_sim.closed_loop.traffic_traci import ClosedLoopTrafficAdapter


SCENARIO_DIR = Path("data/scenario_reconstruction_official_v1")


class _FakeTable:
    def __init__(self, values):
        self._values = values

    def min(self):
        return min(self._values)

    def max(self):
        return max(self._values)


class _FakeResultTable:
    def __init__(self, column, values):
        setattr(self, column, _FakeTable(values))

    def __len__(self):
        return 1


class _FakePP:
    def __init__(self, *, fail_opf: bool = False):
        self.fail_opf = fail_opf
        self.calls: list[str] = []
        self.loads: list[dict] = []
        self.sgens: list[dict] = []
        self.costs: list[dict] = []

    def create_empty_network(self, **_kwargs):
        net = type("FakeNet", (), {})()
        net.converged = True
        net.res_bus = _FakeResultTable("vm_pu", [0.967, 1.013])
        net.res_line = _FakeResultTable("loading_percent", [71.5, 88.2])
        net.buses = []
        net.res_load = {"p_mw": _FakeTable([0.1, 0.2, 0.3])}
        net.res_sgen = {"p_mw": _FakeTable([0.05, 0.15])}
        return net

    def create_bus(self, net, **_kwargs):
        net.buses.append(len(net.buses))
        return len(net.buses) - 1

    def create_ext_grid(self, *_args, **_kwargs):
        return 0

    def create_line_from_parameters(self, *_args, **_kwargs):
        return 0

    def create_load(self, *_args, **_kwargs):
        self.loads.append(dict(_kwargs))
        return len(self.loads) - 1

    def create_sgen(self, *_args, **_kwargs):
        self.sgens.append(dict(_kwargs))
        return len(self.sgens) - 1

    def create_poly_cost(self, *_args, **_kwargs):
        self.costs.append(dict(_kwargs))
        return len(self.costs) - 1

    def runopp(self, net, **_kwargs):
        self.calls.append("runopp")
        if self.fail_opf:
            raise RuntimeError("opf infeasible")
        net.OPF_converged = True

    def runpp(self, net, **_kwargs):
        self.calls.append("runpp")
        net.converged = True


def test_ac_power_adapter_attempts_opf_before_power_flow() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    fake_pp = _FakePP()
    adapter = ClosedLoopACPowerAdapter(module_loader=lambda _name: fake_pp)

    result = adapter.run_ac_opf_or_pf(
        scenario,
        closed_pdn_edges={edge.edge_id for edge in scenario.pdn_edges[:10]},
        restored_load_fraction=0.4,
    )

    assert result.opf_attempted is True
    assert result.opf_converged is True
    assert fake_pp.calls == ["runopp"]
    assert result.min_voltage_pu == 0.967
    assert result.max_line_loading_pct == 88.2
    assert result.served_load_kw == 600.0
    assert result.requested_load_kw is not None
    assert result.shed_load_kw is not None


def test_ac_power_adapter_builds_opf_controls_for_load_and_mobile_support() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    fake_pp = _FakePP()
    adapter = ClosedLoopACPowerAdapter(module_loader=lambda _name: fake_pp)

    result = adapter.run_ac_opf_or_pf(
        scenario,
        closed_pdn_edges={edge.edge_id for edge in scenario.pdn_edges[:12]},
        restored_load_fraction=0.5,
        mess_support_kw=800.0,
        v2g_support_kw=250.0,
    )

    assert result.opf_converged is True
    assert any(load["controllable"] is True for load in fake_pp.loads)
    assert fake_pp.sgens
    assert any(cost["et"] == "load" for cost in fake_pp.costs)
    assert any(cost["et"] == "sgen" for cost in fake_pp.costs)
    assert result.mobile_support_capacity_kw == 1050.0


def test_ac_power_adapter_records_opf_fallback_to_power_flow() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    fake_pp = _FakePP(fail_opf=True)
    adapter = ClosedLoopACPowerAdapter(module_loader=lambda _name: fake_pp)

    result = adapter.run_ac_opf_or_pf(
        scenario,
        closed_pdn_edges={edge.edge_id for edge in scenario.pdn_edges[:10]},
        restored_load_fraction=0.4,
    )

    assert fake_pp.calls == ["runopp", "runpp"]
    assert result.opf_attempted is True
    assert result.opf_converged is False
    assert result.power_flow_converged is True
    assert "opf_failed:RuntimeError" in result.blockers


def test_traffic_adapter_exposes_travel_time_feedback() -> None:
    adapter = ClosedLoopTrafficAdapter(
        overlay_provider=lambda time_min: {
            "status": "ok",
            "edge_metrics": {
                "UTN_001": {"travel_time_s": 31.5, "speed_mps": 8.0},
                "UTN_002": {"travel_time_s": 47.0, "speed_mps": 5.0},
            },
        }
    )

    result = adapter.step(time_min=5.0)

    assert result.status == "ok"
    assert result.edge_travel_time_s["UTN_001"] == 31.5
    assert result.mean_travel_time_s == 39.25


def test_traffic_adapter_wraps_online_traci_overlay_adapter() -> None:
    class FakeOnlineTraCI:
        def overlay_for_time(self, *, time_min):
            return {
                "status": "ok",
                "edge_metrics": {
                    "UTN_010": {"travel_time_s": 55.0, "speed_mps": 9.0}
                },
            }

    adapter = ClosedLoopTrafficAdapter.from_online_traci_adapter(FakeOnlineTraCI())

    result = adapter.step(time_min=10.0)

    assert result.status == "ok"
    assert result.edge_speed_mps["UTN_010"] == 9.0


def test_communication_adapter_summarizes_packet_evidence() -> None:
    adapter = ClosedLoopCommunicationAdapter(
        evidence_provider=lambda _time_min: [
            PacketEvidence(packet_id="p1", delivered=True, delay_ms=10.0),
            PacketEvidence(packet_id="p2", delivered=False, delay_ms=300.0),
            PacketEvidence(packet_id="p3", delivered=True, delay_ms=20.0),
        ]
    )

    result = adapter.step(time_min=5.0)

    assert result.packet_count == 3
    assert result.delivery_rate == 2 / 3
    assert result.mean_delay_ms == 110.0
    assert result.control_available is False


def test_communication_adapter_boosts_uav_relay_monitoring_without_target() -> None:
    adapter = ClosedLoopCommunicationAdapter(
        evidence_provider=lambda _time_min: [
            PacketEvidence(packet_id="p1", delivered=True, delay_ms=10.0),
            PacketEvidence(packet_id="p2", delivered=False, delay_ms=300.0),
            PacketEvidence(packet_id="p3", delivered=True, delay_ms=20.0),
        ],
        min_delivery_rate=0.8,
    )

    direct = adapter.step(time_min=5.0, communication_mode="direct")
    relay = adapter.step(time_min=5.0, communication_mode="uav_relay")

    assert direct.control_available is False
    assert relay.delivery_rate > direct.delivery_rate
    assert relay.mean_delay_ms > direct.mean_delay_ms
    assert relay.control_available is True


def test_communication_adapter_dual_channel_uses_parallel_reliability_with_direct_timing() -> None:
    adapter = ClosedLoopCommunicationAdapter(
        evidence_provider=lambda _time_min: [
            PacketEvidence(packet_id="p1", delivered=True, delay_ms=10.0),
            PacketEvidence(packet_id="p2", delivered=False, delay_ms=300.0),
            PacketEvidence(packet_id="p3", delivered=True, delay_ms=20.0),
        ],
        min_delivery_rate=0.8,
    )

    direct = adapter.step(
        time_min=5.0,
        target_id="PDN_036",
        communication_mode="direct",
        target_robust_travel_time_s=900.0,
    )
    relay = adapter.step(
        time_min=5.0,
        target_id="PDN_036",
        communication_mode="uav_relay",
        target_robust_travel_time_s=900.0,
    )
    dual = adapter.step(
        time_min=5.0,
        target_id="PDN_036",
        communication_mode="dual_channel",
        target_robust_travel_time_s=900.0,
    )

    assert dual.delivery_rate >= relay.delivery_rate
    assert dual.delivery_rate > direct.delivery_rate
    assert direct.mean_delay_ms < dual.mean_delay_ms < relay.mean_delay_ms
    assert dual.control_available is True


def test_communication_adapter_reads_ns3_results_by_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "ns3_packet_results.csv"
    csv_path.write_text(
        "\n".join(
            [
                "packet_id,time_s,delivered,delay_ms",
                "p0,0,1,10",
                "p1,300,1,20",
                "p2,300,0,80",
                "p3,600,1,30",
            ]
        ),
        encoding="utf-8",
    )
    adapter = ClosedLoopCommunicationAdapter.from_ns3_results_csv(
        csv_path,
        min_delivery_rate=0.5,
    )

    result = adapter.step(time_min=5.0)

    assert result.packet_count == 2
    assert result.delivery_rate == 0.5
    assert result.mean_delay_ms == 50.0
    assert result.control_available is True


def test_communication_adapter_calls_online_ns3_command_per_step(tmp_path: Path) -> None:
    calls = []

    def runner(command, *, timeout_s, cwd):
        calls.append({"command": command, "timeout_s": timeout_s, "cwd": cwd})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(
                {
                    "packets": [
                        {"packet_id": "online_1", "delivered": True, "delay_ms": 18.0},
                        {"packet_id": "online_2", "delivered": True, "delay_ms": 22.0},
                    ]
                }
            ),
            stderr="",
        )

    adapter = ClosedLoopCommunicationAdapter.from_online_ns3_command(
        ["python", "ns3_step.py", "--time-min", "{time_min}"],
        command_runner=runner,
        cwd=tmp_path,
        timeout_s=3.0,
        min_delivery_rate=0.9,
    )

    result = adapter.step(time_min=5.0)

    assert result.status == "ok"
    assert result.packet_count == 2
    assert result.delivery_rate == 1.0
    assert result.mean_delay_ms == 20.0
    assert result.control_available is True
    assert calls == [
        {
            "command": ["python", "ns3_step.py", "--time-min", "5.000000"],
            "timeout_s": 3.0,
            "cwd": tmp_path,
        }
    ]


def test_communication_adapter_fails_closed_when_online_ns3_command_fails(
    tmp_path: Path,
) -> None:
    def runner(command, *, timeout_s, cwd):
        return subprocess.CompletedProcess(
            args=command,
            returncode=2,
            stdout="",
            stderr="ns3 binary missing",
        )

    adapter = ClosedLoopCommunicationAdapter.from_online_ns3_command(
        ["ns3-step", "--time-min", "{time_min}"],
        command_runner=runner,
        cwd=tmp_path,
    )

    result = adapter.step(time_min=10.0)

    assert result.status == "blocked_ns3_online_command_failed"
    assert result.packet_count == 0
    assert result.control_available is False
    assert "ns3_online_returncode:2" in result.blockers


def test_communication_adapter_reuses_online_ns3_step_cache(tmp_path: Path) -> None:
    calls = []

    def runner(command, *, timeout_s, cwd):
        calls.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(
                {
                    "packets": [
                        {"packet_id": "cached_1", "delivered": True, "delay_ms": 11.0}
                    ]
                }
            ),
            stderr="",
        )

    command = ["python", "unique_ns3_step.py", "--time-min", "{time_min}"]
    first = ClosedLoopCommunicationAdapter.from_online_ns3_command(
        command,
        command_runner=runner,
        cwd=tmp_path,
    )
    second = ClosedLoopCommunicationAdapter.from_online_ns3_command(
        command,
        command_runner=runner,
        cwd=tmp_path,
    )

    assert first.step(time_min=5.0).packet_count == 1
    assert second.step(time_min=5.0).packet_count == 1
    assert len(calls) == 1
