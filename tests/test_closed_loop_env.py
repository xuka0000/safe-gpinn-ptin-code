from pathlib import Path
from dataclasses import replace

from ptin_sim.closed_loop.communication_ns3 import CommunicationStepResult
from ptin_sim.closed_loop.env import ClosedLoopPTINEnv
from ptin_sim.closed_loop.power_ac_opf import ACPowerResult
from ptin_sim.closed_loop.scenario import load_closed_loop_scenario
from ptin_sim.closed_loop.traffic_traci import TrafficStepResult
from ptin_sim.closed_loop.types import ClosedLoopAction, EVAvailabilityPoint


SCENARIO_DIR = Path("data/scenario_reconstruction_official_v1")


class _PowerStub:
    def __init__(self) -> None:
        self.calls = 0
        self.mess_support_kw: list[float] = []
        self.v2g_support_kw: list[float] = []

    def run_ac_opf_or_pf(
        self,
        scenario,
        *,
        closed_pdn_edges,
        restored_load_fraction,
        mess_support_kw=0.0,
        v2g_support_kw=0.0,
    ):
        self.calls += 1
        self.mess_support_kw.append(mess_support_kw)
        self.v2g_support_kw.append(v2g_support_kw)
        return ACPowerResult(
            status="opf_ok",
            solver_available=True,
            opf_attempted=True,
            opf_converged=True,
            power_flow_converged=None,
            min_voltage_pu=0.97 + 0.01 * restored_load_fraction,
            max_voltage_pu=1.02,
            max_line_loading_pct=80.0 - self.calls,
            requested_load_kw=1000.0,
            served_load_kw=900.0 + mess_support_kw * 0.01,
            shed_load_kw=100.0,
            mobile_support_capacity_kw=mess_support_kw + v2g_support_kw,
            mobile_dispatch_kw=mess_support_kw * 0.5 + v2g_support_kw * 0.25,
            blockers=(),
        )


class _TrafficStub:
    def step(self, *, time_min):
        return TrafficStepResult(
            status="ok",
            edge_travel_time_s={"UTN_001": 30.0 + time_min},
            edge_speed_mps={"UTN_001": 10.0},
            mean_travel_time_s=30.0 + time_min,
            blockers=(),
        )


class _TargetTrafficStub:
    def step(self, *, time_min):
        return TrafficStepResult(
            status="ok",
            edge_travel_time_s={
                "UTN_001": 20.0,
                "UTN_002": 80.0,
                "UTN_013": 140.0,
            },
            edge_speed_mps={
                "UTN_001": 12.0,
                "UTN_002": 6.0,
                "UTN_013": 4.0,
            },
            mean_travel_time_s=80.0,
            blockers=(),
        )


class _CommunicationStub:
    def step(self, *, time_min):
        return CommunicationStepResult(
            status="ok",
            packet_count=2,
            delivery_rate=1.0,
            mean_delay_ms=20.0,
            control_available=True,
            blockers=(),
        )


def test_reward_uses_duration_energy_reliability_and_terminal_terms() -> None:
    ac = ACPowerResult(
        status="opf_ok",
        solver_available=True,
        opf_attempted=True,
        opf_converged=True,
        power_flow_converged=None,
        min_voltage_pu=0.98,
        max_voltage_pu=1.02,
        max_line_loading_pct=85.0,
        requested_load_kw=1000.0,
        served_load_kw=400.0,
        shed_load_kw=600.0,
        mobile_support_capacity_kw=0.0,
        mobile_dispatch_kw=0.0,
        blockers=(),
    )
    traffic = TrafficStepResult(
        status="ok",
        edge_travel_time_s={},
        edge_speed_mps={},
        mean_travel_time_s=40.0,
        blockers=(),
    )
    reliable_comm = CommunicationStepResult(
        status="ok",
        packet_count=4,
        delivery_rate=1.0,
        mean_delay_ms=30.0,
        control_available=True,
        blockers=(),
    )
    lossy_comm = CommunicationStepResult(
        status="degraded",
        packet_count=4,
        delivery_rate=0.5,
        mean_delay_ms=30.0,
        control_available=True,
        blockers=(),
    )

    fast_reliable_reward = ClosedLoopPTINEnv._reward(
        restored_load_gain_kw=1000.0,
        ac=ac,
        traffic=traffic,
        communication=reliable_comm,
        applied=True,
        weighted_restoration_value_kw=1000.0,
        action_duration_min=5.0,
        terminal_restoration=False,
    )
    slow_lossy_reward = ClosedLoopPTINEnv._reward(
        restored_load_gain_kw=1000.0,
        ac=ac,
        traffic=traffic,
        communication=lossy_comm,
        applied=True,
        weighted_restoration_value_kw=1000.0,
        action_duration_min=15.0,
        terminal_restoration=False,
    )
    terminal_reward = ClosedLoopPTINEnv._reward(
        restored_load_gain_kw=1000.0,
        ac=ac,
        traffic=traffic,
        communication=reliable_comm,
        applied=True,
        weighted_restoration_value_kw=1000.0,
        action_duration_min=5.0,
        terminal_restoration=True,
    )

    assert slow_lossy_reward < fast_reliable_reward - 100.0
    assert terminal_reward > fast_reliable_reward + 100.0


class _ModeAwareCommunicationStub:
    def step(
        self,
        *,
        time_min,
        target_id="",
        communication_mode="direct",
        target_robust_travel_time_s=0.0,
    ):
        del time_min, target_id, target_robust_travel_time_s
        if communication_mode == "uav_relay":
            return CommunicationStepResult(
                status="ok",
                packet_count=4,
                delivery_rate=1.0,
                mean_delay_ms=40.0,
                control_available=True,
                blockers=(),
            )
        return CommunicationStepResult(
            status="blocked_direct_control_gap",
            packet_count=1,
            delivery_rate=0.0,
            mean_delay_ms=500.0,
            control_available=False,
            blockers=("communication_control_unavailable",),
        )


def test_wait_action_uses_uav_relay_when_available() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    assert scenario.fleet.uav_count > 0
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TrafficStub(),
        communication_adapter=_ModeAwareCommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    observation = env.reset()
    wait_action = next(
        action for action in env.available_actions() if action.action_type == "wait"
    )
    _next_observation, _reward, _terminated, _truncated, info = env.step(wait_action)

    assert wait_action.metadata["communication_mode"] == "uav_relay"
    assert info["communication_mode"] == "uav_relay"
    assert info["packet_delivery_rate"] == 1.0
    assert info["control_available"] is True


def test_closed_loop_step_updates_state_and_next_observation() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    observation = env.reset()
    first_failed = observation["failed_pdn_edges"][0]
    assert observation["restored_failed_edge_count"] == 0

    next_observation, reward, terminated, truncated, info = env.step(
        ClosedLoopAction(
            action_id="restore_first",
            action_type="restore_pdn_edge",
            target_id=first_failed,
            resource_id="MESS_1",
        )
    )

    assert reward > 0
    assert terminated is False
    assert truncated is False
    assert info["applied"] is True
    assert first_failed in next_observation["restored_pdn_edges"]
    assert first_failed not in next_observation["failed_pdn_edges"]
    assert next_observation["restored_failed_edge_count"] == 1
    assert next_observation["last_ac_status"] == "opf_ok"
    assert next_observation["last_packet_delivery_rate"] == 1.0


def test_available_actions_include_direct_relay_and_wait_options() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    env.reset()
    actions = env.available_actions()
    first_target = actions[0].target_id

    assert any(
        action.target_id == first_target
        and action.metadata.get("communication_mode") == "direct"
        for action in actions
    )
    assert any(
        action.target_id == first_target
        and action.metadata.get("communication_mode") == "uav_relay"
        for action in actions
    )
    assert any(
        action.target_id == first_target
        and action.metadata.get("communication_mode") == "dual_channel"
        for action in actions
    )
    assert any(action.action_type == "wait" for action in actions)


def test_relay_action_changes_communication_gate_outcome() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    direct_env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TrafficStub(),
        communication_adapter=_ModeAwareCommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )
    relay_env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TrafficStub(),
        communication_adapter=_ModeAwareCommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    direct_env.reset()
    target = direct_env.available_actions()[0].target_id
    _obs, _reward, _terminated, _truncated, direct_info = direct_env.step(
        ClosedLoopAction(
            action_id=f"restore_{target}__direct",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "direct"},
        )
    )
    relay_env.reset()
    _obs, _reward, _terminated, _truncated, relay_info = relay_env.step(
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
    )

    assert direct_info["applied"] is False
    assert direct_info["block_reason"] == "communication_control_unavailable"
    assert relay_info["applied"] is True
    assert relay_info["communication_mode"] == "uav_relay"
    assert relay_env.trace_rows()[-1]["packet_delivery_rate"] > direct_env.trace_rows()[-1]["packet_delivery_rate"]


class _DualChannelCommunicationStub:
    def step(
        self,
        *,
        time_min,
        target_id="",
        communication_mode="direct",
        target_robust_travel_time_s=0.0,
    ):
        del time_min, target_id, target_robust_travel_time_s
        if communication_mode == "dual_channel":
            return CommunicationStepResult(
                status="ok",
                packet_count=4,
                delivery_rate=0.98,
                mean_delay_ms=32.0,
                control_available=True,
                blockers=(),
            )
        if communication_mode == "uav_relay":
            return CommunicationStepResult(
                status="ok",
                packet_count=4,
                delivery_rate=1.0,
                mean_delay_ms=44.0,
                control_available=True,
                blockers=(),
            )
        return CommunicationStepResult(
            status="ok",
            packet_count=4,
            delivery_rate=0.86,
            mean_delay_ms=18.0,
            control_available=True,
            blockers=(),
        )


def test_dual_channel_restore_keeps_direct_duration_with_redundant_reliability() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    direct_env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TargetTrafficStub(),
        communication_adapter=_DualChannelCommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )
    dual_env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TargetTrafficStub(),
        communication_adapter=_DualChannelCommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )
    relay_env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TargetTrafficStub(),
        communication_adapter=_DualChannelCommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    direct_env.reset()
    dual_env.reset()
    relay_env.reset()
    target = "PDN_003"
    _obs, _reward, _terminated, _truncated, direct_info = direct_env.step(
        ClosedLoopAction(
            action_id=f"restore_{target}__direct",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "direct"},
        )
    )
    _obs, _reward, _terminated, _truncated, dual_info = dual_env.step(
        ClosedLoopAction(
            action_id=f"restore_{target}__dual_channel",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "dual_channel"},
        )
    )
    _obs, _reward, _terminated, _truncated, relay_info = relay_env.step(
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
    )

    assert dual_info["communication_mode"] == "dual_channel"
    assert dual_info["packet_delivery_rate"] > direct_info["packet_delivery_rate"]
    assert dual_info["action_duration_min"] > direct_info["action_duration_min"]
    assert dual_info["action_duration_min"] < relay_info["action_duration_min"]


def test_restore_action_duration_depends_on_target_route_time() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TargetTrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    env.reset()
    _obs, _reward, _terminated, _truncated, info = env.step(
        ClosedLoopAction(
            action_id="restore_pdn_009__direct",
            action_type="restore_pdn_edge",
            target_id="PDN_009",
            resource_id="MESS_1",
            metadata={"communication_mode": "direct"},
        )
    )

    trace = env.trace_rows()[-1]
    assert info["action_duration_min"] > 5.0
    assert trace["time_min"] == info["time_min"]
    assert trace["target_robust_travel_time_s"] > 0.0


def test_closed_loop_environment_is_not_one_step_candidate_scorer() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    env.reset()
    first = next(
        action
        for action in env.available_actions()
        if action.metadata.get("communication_mode") == "uav_relay"
    )
    obs_after_first, _reward, terminated, _truncated, _info = env.step(first)
    second = next(
        action
        for action in env.available_actions()
        if action.metadata.get("communication_mode") == "uav_relay"
    )
    obs_after_second, _reward2, terminated2, _truncated2, _info2 = env.step(second)

    assert terminated is False
    assert terminated2 is False
    assert obs_after_second["step_index"] == obs_after_first["step_index"] + 1
    assert obs_after_second["restored_failed_edge_count"] == 2


def test_closed_loop_step_passes_mobile_support_to_ac_opf() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    power = _PowerStub()
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=power,
        traffic_adapter=_TrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    env.reset()
    env.step(env.available_actions()[0])

    assert power.mess_support_kw[-1] == scenario.fleet.mess_unit_power_kw
    assert power.v2g_support_kw[-1] == scenario.fleet.ev_unit_power_kw
    trace = env.trace_rows()[-1]
    assert trace["mobile_support_capacity_kw"] == 550.0
    assert trace["served_load_kw"] == 905.0
    assert trace["packet_delivery_rate"] == 1.0
    assert trace["mean_delay_ms"] == 20.0
    assert trace["mean_travel_time_s"] == 35.0
    assert trace["v2g_dispatchable_ev_count"] == 390
    assert trace["v2g_energy_required_kwh"] > 0.0
    assert trace["v2g_energy_remaining_kwh"] < (
        scenario.fleet.ev_count
        * scenario.fleet.v2g_willingness
        * scenario.fleet.ev_unit_energy_kwh
    )


def test_closed_loop_blocks_action_when_v2g_energy_is_insufficient() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    low_v2g_fleet = replace(
        scenario.fleet,
        ev_count=1,
        v2g_willingness=1.0,
        ev_unit_power_kw=50.0,
        ev_unit_energy_kwh=0.01,
    )
    low_v2g_scenario = replace(scenario, fleet=low_v2g_fleet)
    power = _PowerStub()
    env = ClosedLoopPTINEnv(
        low_v2g_scenario,
        power_adapter=power,
        traffic_adapter=_TrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    observation = env.reset()
    assert observation["v2g_dispatchable_ev_count"] == 1
    assert observation["v2g_energy_remaining_kwh"] == 0.01

    _obs, _reward, _terminated, _truncated, info = env.step(env.available_actions()[0])

    trace = env.trace_rows()[-1]
    assert info["applied"] is False
    assert info["block_reason"] == "v2g_energy_unavailable"
    assert power.v2g_support_kw[-1] == 0.0
    assert trace["resource_energy_feasible"] == 0
    assert trace["v2g_energy_feasible"] == 0
    assert trace["v2g_support_kw"] == 0.0
    assert trace["v2g_energy_required_kwh"] > trace["v2g_energy_remaining_kwh"]


def test_closed_loop_updates_v2g_capacity_from_ev_availability_profile() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    time_varying_fleet = replace(
        scenario.fleet,
        ev_count=10,
        v2g_willingness=1.0,
        ev_unit_power_kw=20.0,
        ev_unit_energy_kwh=10.0,
        ev_availability_profile=(
            EVAvailabilityPoint(time_min=0.0, availability_factor=0.2),
            EVAvailabilityPoint(time_min=5.0, availability_factor=0.8),
            EVAvailabilityPoint(time_min=10.0, availability_factor=0.0),
        ),
    )
    scenario = replace(scenario, fleet=time_varying_fleet)
    power = _PowerStub()
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=power,
        traffic_adapter=_TrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    observation = env.reset()

    assert observation["v2g_availability_factor"] == 0.2
    assert observation["v2g_dispatchable_ev_count"] == 2
    assert observation["v2g_available_energy_capacity_kwh"] == 20.0

    env.step(env.available_actions()[0])
    first_trace = env.trace_rows()[-1]

    assert first_trace["v2g_availability_factor"] == 0.8
    assert first_trace["v2g_dispatchable_ev_count"] == 8
    assert first_trace["v2g_available_energy_capacity_kwh"] == 80.0
    assert first_trace["v2g_energy_remaining_kwh"] == (
        first_trace["v2g_available_energy_capacity_kwh"]
        - first_trace["v2g_energy_required_kwh"]
    )
    assert power.v2g_support_kw[-1] == 20.0

    env.step(env.available_actions()[0])
    second_trace = env.trace_rows()[-1]

    assert second_trace["v2g_availability_factor"] == 0.0
    assert second_trace["v2g_dispatchable_ev_count"] == 0
    assert second_trace["v2g_available_energy_capacity_kwh"] == 0.0
    assert second_trace["v2g_energy_remaining_kwh"] == 0.0
    assert power.v2g_support_kw[-1] == 0.0


def test_closed_loop_trace_records_cross_layer_feasibility_terms() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    env.reset()
    env.step(
        ClosedLoopAction(
            action_id="restore_pdn_004",
            action_type="restore_pdn_edge",
            target_id="PDN_004",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
    )

    trace = env.trace_rows()[-1]
    assert trace["switch_controller_count"] == 1
    assert trace["powered_switch_controller_count"] == 0
    assert trace["uav_relay_used"] == 1
    assert trace["traffic_feasible"] == 1
    assert trace["resource_energy_feasible"] == 1
    assert float(trace["resource_energy_remaining_kwh"]) < (
        scenario.fleet.mess_count * scenario.fleet.mess_unit_energy_kwh
    )
    assert trace["traffic_uncertainty_mode"] == "interval_robust"
    assert trace["robust_mean_travel_time_s"] > trace["mean_travel_time_s"]
    assert trace["predeployment_quality_index"] > 0.0
    assert trace["weighted_restoration_value_kw"] >= trace["restored_load_gain_kw"]
    assert "topology_vulnerability_score" in trace


def test_closed_loop_trace_records_target_level_traffic_audit() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TargetTrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    env.reset()
    _obs, _reward, _terminated, _truncated, info = env.step(
        ClosedLoopAction(
            action_id="restore_pdn_009",
            action_type="restore_pdn_edge",
            target_id="PDN_009",
            resource_id="MESS_1",
        )
    )

    trace = env.trace_rows()[-1]
    assert info["target_traffic_edge_ids"] == "UTN_013"
    assert trace["target_traffic_edge_ids"] == "UTN_013"
    assert trace["target_travel_time_s"] == 140.0
    assert trace["target_robust_travel_time_s"] == (
        140.0 * scenario.disaster.robust_travel_time_multiplier
    )
    assert trace["target_traffic_edge_count"] == 1


def test_closed_loop_blocks_action_when_mobile_energy_is_insufficient() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    low_energy_fleet = replace(scenario.fleet, mess_count=1, mess_unit_energy_kwh=0.01)
    low_energy_scenario = replace(scenario, fleet=low_energy_fleet)
    env = ClosedLoopPTINEnv(
        low_energy_scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    env.reset()
    _obs, _reward, _terminated, _truncated, info = env.step(
        ClosedLoopAction(
            action_id="restore_pdn_003",
            action_type="restore_pdn_edge",
            target_id="PDN_003",
            resource_id="MESS_1",
        )
    )

    trace = env.trace_rows()[-1]
    assert info["applied"] is False
    assert info["block_reason"] == "resource_energy_unavailable"
    assert trace["resource_energy_feasible"] == 0
    assert trace["resource_energy_required_kwh"] > trace["resource_energy_remaining_kwh"]


def test_progressive_disaster_reveals_failed_edges_over_time() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)
    first, second, *rest = scenario.failed_pdn_edges
    progressive = replace(
        scenario.disaster,
        disaster_category="progressive",
        control_paradigm="mpc_rolling_horizon",
        failure_release_step_by_edge={first: 0, second: 2, **{edge: 5 for edge in rest}},
    )
    scenario = replace(scenario, disaster=progressive, failed_pdn_edges=(first, second))
    env = ClosedLoopPTINEnv(
        scenario,
        power_adapter=_PowerStub(),
        traffic_adapter=_TrafficStub(),
        communication_adapter=_CommunicationStub(),
        max_steps=4,
        step_minutes=5.0,
    )

    observation = env.reset()

    assert observation["failed_pdn_edges"] == [first]
    assert observation["latent_failed_edge_count"] == 1
    env.step(
        ClosedLoopAction(
            action_id=f"restore_{first}",
            action_type="restore_pdn_edge",
            target_id=first,
            resource_id="MESS_1",
        )
    )
    observation_after_first = env.observe()
    assert observation_after_first["failed_pdn_edges"] == []
    assert observation_after_first["latent_failed_edge_count"] == 1

    observation_after_wait, _reward, _terminated, _truncated, info = env.step(
        ClosedLoopAction(
            action_id="wait_for_progressive_update",
            action_type="wait",
            target_id="",
        )
    )

    assert info["applied"] is True
    assert second in observation_after_wait["failed_pdn_edges"]
    assert observation_after_wait["latent_failed_edge_count"] == 0
