from pathlib import Path

from ptin_sim.closed_loop.scenario import _fleet_spec, load_closed_loop_scenario
from ptin_sim.closed_loop.types import PTINScenario


SCENARIO_DIR = Path("data/scenario_reconstruction_official_v1")
PROGRESSIVE_SCENARIO_DIR = Path("data/scenario_reconstruction_progressive_v1")
PROGRESSIVE_STRESS_SCENARIO_DIR = Path(
    "data/scenario_reconstruction_progressive_stress_v2"
)
PROGRESSIVE_DYNAMIC_EV_SCENARIO_DIR = Path(
    "data/scenario_reconstruction_progressive_stress_v2_dynamic_ev"
)
PROGRESSIVE_DYNAMIC_EV_LOW_PREDEPLOY_SCENARIO_DIR = Path(
    "data/scenario_reconstruction_progressive_stress_v2_dynamic_ev_low_predeploy"
)


def test_loads_official_ptin_scenario_as_typed_bundle() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)

    assert isinstance(scenario, PTINScenario)
    assert len(scenario.pdn_nodes) >= 37
    assert len(scenario.pdn_edges) >= 36
    assert len(scenario.cn_nodes) >= 43
    assert len(scenario.utn_nodes) >= 24
    assert len(scenario.dependencies) >= 100
    assert scenario.failed_pdn_edges
    assert scenario.fleet.mess_count > 0
    assert scenario.fleet.uav_count > 0
    assert scenario.fleet.ev_count == 1300
    assert scenario.fleet.ev_ratio == 0.5
    assert scenario.fleet.v2g_willingness == 0.3


def test_scenario_exposes_cross_layer_dependency_maps() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)

    assert scenario.dependencies_by_target("CN")
    assert scenario.dependencies_by_target("UTN")
    assert scenario.dependencies_by_target("PDN_EDGE")
    assert any(edge.edge_id in scenario.failed_pdn_edges for edge in scenario.pdn_edges)


def test_scenario_exposes_framework_disaster_and_uncertainty_profile() -> None:
    scenario = load_closed_loop_scenario(SCENARIO_DIR)

    assert scenario.disaster.disaster_category == "instantaneous"
    assert scenario.disaster.control_paradigm == "two_stage_deterministic"
    assert scenario.disaster.traffic_uncertainty_mode == "interval_robust"
    assert scenario.disaster.robust_travel_time_multiplier > 1.0
    assert scenario.disaster.predeployment_quality_index > 0.0
    assert scenario.disaster.topology_vulnerability_weight > 0.0
    assert set(scenario.disaster.failure_release_step_by_edge) == set(
        scenario.failed_pdn_edges
    )


def test_fleet_spec_loads_time_varying_ev_availability_profile(tmp_path) -> None:
    fleet_file = tmp_path / "resource_fleet.yaml"
    fleet_file.write_text(
        """
resources:
  mess_ugv:
    count: 1
    unit_power_kw: 100
    unit_energy_kwh: 200
  uav:
    count: 1
    communication_radius_km: 1
    flight_speed_km_per_h: 120
    flight_endurance_km: 40
  ev_unit:
    unit_power_kw: 20
    unit_energy_kwh: 40
    ev_ratio: 0.5
    v2g_willingness: 0.4
  traffic_vehicle_population:
    stable_operating_vehicle_count: 100
  ev_availability_profile:
    - time_min: 10
      availability_factor: 0.2
    - time_min: 0
      availability_factor: 0.6
""",
        encoding="utf-8",
    )

    fleet = _fleet_spec(fleet_file)

    assert fleet.ev_count == 50
    assert [point.time_min for point in fleet.ev_availability_profile] == [0.0, 10.0]
    assert [point.availability_factor for point in fleet.ev_availability_profile] == [
        0.6,
        0.2,
    ]


def test_progressive_overlay_reuses_official_topology_with_staged_failures() -> None:
    scenario = load_closed_loop_scenario(PROGRESSIVE_SCENARIO_DIR)
    official = load_closed_loop_scenario(SCENARIO_DIR)

    assert scenario.scenario_id == "applied_energy_progressive_v1_switch_failure"
    assert len(scenario.pdn_nodes) == len(official.pdn_nodes)
    assert len(scenario.cn_nodes) == len(official.cn_nodes)
    assert len(scenario.utn_nodes) == len(official.utn_nodes)
    assert scenario.failed_pdn_edges == official.failed_pdn_edges
    assert scenario.disaster.disaster_category == "progressive"
    assert scenario.disaster.control_paradigm == "mpc_rolling_horizon"
    assert scenario.disaster.robust_travel_time_multiplier > official.disaster.robust_travel_time_multiplier
    assert sorted(scenario.disaster.failure_release_step_by_edge.values()) == [0, 0, 3, 5]


def test_progressive_stress_overlay_extends_failure_set_without_copying_topology() -> None:
    scenario = load_closed_loop_scenario(PROGRESSIVE_STRESS_SCENARIO_DIR)
    official = load_closed_loop_scenario(SCENARIO_DIR)

    assert scenario.scenario_id == "applied_energy_progressive_stress_v2_switch_failure"
    assert len(scenario.pdn_edges) == len(official.pdn_edges)
    assert len(scenario.cn_nodes) == len(official.cn_nodes)
    assert len(scenario.utn_edges) == len(official.utn_edges)
    assert len(scenario.failed_pdn_edges) == 8
    assert set(official.failed_pdn_edges).issubset(set(scenario.failed_pdn_edges))
    assert scenario.disaster.disaster_category == "progressive"
    assert max(scenario.disaster.failure_release_step_by_edge.values()) == 6


def test_progressive_dynamic_ev_overlay_loads_time_varying_v2g_profile() -> None:
    scenario = load_closed_loop_scenario(PROGRESSIVE_DYNAMIC_EV_SCENARIO_DIR)
    official = load_closed_loop_scenario(SCENARIO_DIR)

    assert scenario.scenario_id == "applied_energy_progressive_stress_v2_dynamic_ev"
    assert len(scenario.pdn_edges) == len(official.pdn_edges)
    assert len(scenario.failed_pdn_edges) == 8
    assert scenario.fleet.ev_count == official.fleet.ev_count
    assert scenario.fleet.v2g_willingness < official.fleet.v2g_willingness
    assert [point.time_min for point in scenario.fleet.ev_availability_profile] == [
        0.0,
        10.0,
        25.0,
        40.0,
        55.0,
    ]


def test_progressive_dynamic_ev_low_predeployment_overlay_loads_weakened_resources() -> None:
    scenario = load_closed_loop_scenario(PROGRESSIVE_DYNAMIC_EV_LOW_PREDEPLOY_SCENARIO_DIR)
    reference = load_closed_loop_scenario(PROGRESSIVE_DYNAMIC_EV_SCENARIO_DIR)

    assert scenario.scenario_id == "applied_energy_progressive_stress_v2_dynamic_ev_low_predeploy"
    assert scenario.disaster.predeployment_quality_index < reference.disaster.predeployment_quality_index
    assert scenario.fleet.mess_unit_energy_kwh < reference.fleet.mess_unit_energy_kwh
    assert scenario.fleet.ev_ratio < reference.fleet.ev_ratio
    assert scenario.fleet.v2g_willingness < reference.fleet.v2g_willingness
    assert [point.availability_factor for point in scenario.fleet.ev_availability_profile] == [
        0.10,
        0.30,
        0.18,
        0.05,
        0.22,
    ]
