import csv
import json
from pathlib import Path

import pytest
import torch

import ptin_sim.closed_loop.training as training
from ptin_sim.closed_loop.communication_ns3 import (
    ClosedLoopCommunicationAdapter,
    CommunicationStepResult,
    PacketEvidence,
)
from ptin_sim.closed_loop.dependencies import failed_edge_restoration_load_kw
from ptin_sim.closed_loop.power_ac_opf import ACPowerResult
from ptin_sim.closed_loop.run_closed_loop_experiment import ClosedLoopRunConfig
from ptin_sim.closed_loop.scenario import load_closed_loop_scenario
from ptin_sim.closed_loop.traffic_traci import TrafficStepResult
from ptin_sim.closed_loop.training import (
    _best_rollout_sequence,
    _candidate_restore_sequences,
    _milp_reference_sequence,
    _pairwise_loss,
    _scenario_tree_prefix_return,
    _select_diffusion_sequence_from_rollout_tree,
    _select_sequence_from_rollout_tree,
    _select_safe_gpinn_action,
    _select_from_planned_sequence,
    _training_max_steps,
    build_rollout_dataset,
    evaluate_trained_policies,
    export_policy_eval_traces,
    run_closed_loop_policy_checkpoint_evaluation,
    run_closed_loop_policy_training,
)
from ptin_sim.closed_loop.types import ClosedLoopAction


def test_closed_loop_policy_training_writes_rollout_and_model_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "training_run"

    summary = run_closed_loop_policy_training(
        output_dir=output_dir,
        adapter_mode="fake",
        epochs=4,
        max_sequences=6,
        seed=3,
    )

    assert summary["adapter_mode"] == "fake"
    assert summary["rollout_row_count"] > 0
    assert (output_dir / "closed_loop_rollout_dataset.csv").exists()
    assert (output_dir / "closed_loop_training_curves.csv").exists()
    assert (output_dir / "closed_loop_policy_eval.csv").exists()
    assert (output_dir / "closed_loop_policy_step_trace.csv").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "model_checkpoints" / "safe_gpinn.pt").exists()
    assert (output_dir / "model_checkpoints" / "iql.pt").exists()
    assert (output_dir / "model_checkpoints" / "td3_bc.pt").exists()
    assert (output_dir / "model_checkpoints" / "cql.pt").exists()
    assert (output_dir / "model_checkpoints" / "diffusion_policy.pt").exists()
    for method in ("maddpg", "mappo", "qmix", "vdn", "happo_hatrpo", "mat"):
        assert (output_dir / "model_checkpoints" / f"{method}.pt").exists()
    diffusion_checkpoint = torch.load(
        output_dir / "model_checkpoints" / "diffusion_policy.pt",
        map_location="cpu",
    )
    iql_checkpoint = torch.load(output_dir / "model_checkpoints" / "iql.pt", map_location="cpu")
    td3_bc_checkpoint = torch.load(output_dir / "model_checkpoints" / "td3_bc.pt", map_location="cpu")
    cql_checkpoint = torch.load(output_dir / "model_checkpoints" / "cql.pt", map_location="cpu")
    marl_checkpoints = {
        method: torch.load(output_dir / "model_checkpoints" / f"{method}.pt", map_location="cpu")
        for method in ("maddpg", "mappo", "qmix", "vdn", "happo_hatrpo", "mat")
    }
    safe_checkpoint = torch.load(
        output_dir / "model_checkpoints" / "safe_gpinn.pt",
        map_location="cpu",
    )
    assert safe_checkpoint["algorithm_scope"] == "safe_graph_pinn_return_physics_risk"
    assert "physics_consistency_weight" in safe_checkpoint
    assert "scenario_tree_prefix_weight" in safe_checkpoint
    assert iql_checkpoint["algorithm_scope"] == "full_iql_discrete_offline_rl"
    assert {"q_state_dict", "v_state_dict", "policy_state_dict"}.issubset(iql_checkpoint)
    assert td3_bc_checkpoint["algorithm_scope"] == "full_td3_bc_offline_actor_critic"
    assert {
        "actor_state_dict",
        "critic1_state_dict",
        "critic2_state_dict",
        "target_actor_state_dict",
        "target_critic1_state_dict",
        "target_critic2_state_dict",
    }.issubset(td3_bc_checkpoint)
    assert cql_checkpoint["algorithm_scope"] == "full_cql_discrete_offline_rl"
    assert {"q_state_dict", "target_q_state_dict", "conservative_weight"}.issubset(cql_checkpoint)
    expected_marl_scopes = {
        "maddpg": "ctde_maddpg_style_shared_actor_critic",
        "mappo": "mappo_style_clipped_policy_score",
        "qmix": "qmix_style_monotonic_value_mixer",
        "vdn": "vdn_style_value_decomposition",
        "happo_hatrpo": "happo_hatrpo_style_sequential_trust_region",
        "mat": "mat_style_sequence_model_policy",
    }
    for method, scope in expected_marl_scopes.items():
        checkpoint = marl_checkpoints[method]
        assert checkpoint["algorithm_scope"] == scope
        assert checkpoint["training_protocol"] == "shared_rollout_dataset_matched_seed_closed_loop"
        assert checkpoint["transition_count"] > 0
    assert diffusion_checkpoint["training_scope"] == "trajectory_sequence_diffusion"
    assert diffusion_checkpoint["sequence_training_rows"] > 0
    assert diffusion_checkpoint["sequence_vocabulary"]

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["truth_boundary"] == "closed_loop_policy_training_fake_adapter_smoke_not_final_result"
    with (output_dir / "closed_loop_policy_eval.csv").open("r", encoding="utf-8", newline="") as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert {
        "greedy",
        "load_gain",
        "two_stage_exact",
        "rolling_mpc",
        "safe_gpinn",
        "iql",
        "td3_bc",
        "cql",
        "diffusion_policy",
        "maddpg",
        "mappo",
        "qmix",
        "vdn",
        "happo_hatrpo",
        "mat",
    }.issubset(methods)
    with (output_dir / "closed_loop_policy_step_trace.csv").open("r", encoding="utf-8", newline="") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert trace_rows
    assert {
        "method",
        "step_index",
        "served_load_kw",
        "shed_load_kw",
        "packet_delivery_rate",
        "mean_travel_time_s",
    }.issubset(trace_rows[0])
    assert {"safe_gpinn", "diffusion_policy", "maddpg", "mappo", "qmix", "vdn", "happo_hatrpo", "mat"}.issubset({row["method"] for row in trace_rows})
    with (output_dir / "closed_loop_rollout_dataset.csv").open("r", encoding="utf-8", newline="") as handle:
        rollout_rows = list(csv.DictReader(handle))
    assert rollout_rows
    assert {
        "critical_load_weight",
        "topology_vulnerability_score",
        "predeployment_quality_index",
        "robust_travel_time_margin_s",
        "active_failed_before",
        "latent_failed_before",
        "weighted_restoration_value_kw",
        "tree_branch_id",
        "tree_branch_probability",
        "tree_traffic_multiplier",
        "tree_packet_delivery_factor",
        "tree_resource_factor",
        "tree_branch_return_to_go",
    }.issubset(rollout_rows[0])
    assert {row["tree_branch_id"] for row in rollout_rows}.issuperset(
        {"nominal", "traffic_stress", "communication_stress", "compound_stress"}
    )


def test_closed_loop_policy_training_can_evaluate_on_rollout_tree(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    run_closed_loop_policy_training(
        output_dir=first_dir,
        adapter_mode="fake",
        epochs=2,
        max_sequences=None,
        seed=3,
    )
    second_dir = tmp_path / "second"

    summary = run_closed_loop_policy_training(
        output_dir=second_dir,
        adapter_mode="fake",
        epochs=2,
        max_sequences=None,
        seed=5,
        reuse_rollout_dataset=first_dir / "closed_loop_rollout_dataset.csv",
        evaluation_mode="rollout_dataset",
    )

    assert summary["evaluation_mode"] == "rollout_dataset"
    with (second_dir / "closed_loop_policy_eval.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["selected_sequence"] for row in rows)
    assert {"two_stage_exact", "rolling_mpc"}.issubset({row["method"] for row in rows})


def test_checkpoint_evaluation_restores_trained_bundles(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    run_closed_loop_policy_training(
        output_dir=train_dir,
        adapter_mode="fake",
        epochs=2,
        max_sequences=4,
        seed=3,
    )
    eval_dir = tmp_path / "eval"

    summary = run_closed_loop_policy_checkpoint_evaluation(
        output_dir=eval_dir,
        adapter_mode="fake",
        seed=3,
        rollout_dataset=train_dir / "closed_loop_rollout_dataset.csv",
        checkpoint_dir=train_dir / "model_checkpoints",
        evaluation_mode="env",
    )

    assert summary["evaluation_protocol"] == "per_method_seed_reset_and_adapter_rebuild"
    with (eval_dir / "closed_loop_policy_eval.csv").open("r", encoding="utf-8", newline="") as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert {"safe_gpinn", "maddpg", "mappo", "qmix", "vdn", "happo_hatrpo", "mat"}.issubset(methods)
    assert (eval_dir / "closed_loop_policy_step_trace.csv").exists()


def test_policy_trace_export_preserves_wait_action_metadata() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    config = ClosedLoopRunConfig(
        adapter_mode="fake",
        max_steps=1,
        step_minutes=5.0,
    )

    trace_rows = export_policy_eval_traces(
        scenario,
        config=config,
        eval_rows=[
            {
                "method": "safe_gpinn",
                "selected_sequence": "WAIT",
                "attempted_sequence": "WAIT",
                "total_reward": 0.0,
            }
        ],
    )

    assert trace_rows
    assert trace_rows[0]["action_id"] == "wait_for_progressive_update"
    assert trace_rows[0]["communication_mode"] == "uav_relay"


def test_safe_gpinn_selection_uses_rollout_prefix_when_model_ties() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    first_target, second_target, *_ = scenario.failed_pdn_edges
    actions = [
        ClosedLoopAction(
            action_id=f"restore_{first_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=first_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id=f"restore_{second_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=second_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": 2,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 0,
    }
    rollout_rows = [
        {
            "sequence": f"restore_{first_target}__uav_relay",
            "target_id": first_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 10.0,
        },
        {
            "sequence": f"restore_{second_target}__uav_relay",
            "target_id": second_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 100.0,
        },
    ]
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.target_id == second_target


def test_safe_gpinn_tie_breaks_by_target_robust_transport_cost(monkeypatch) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    edge_lookup = {edge.edge_id: edge for edge in scenario.pdn_edges}
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": len(scenario.failed_pdn_edges),
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 0,
    }
    target_features = []
    for target in scenario.failed_pdn_edges:
        action = ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        features = training._feature_values_from_observation(
            scenario,
            edge_lookup=edge_lookup,
            observation=observation,
            action=action,
        )
        target_features.append((target, float(features["target_robust_travel_time_s"])))
    low_target, low_cost = min(target_features, key=lambda item: item[1])
    high_target, high_cost = max(target_features, key=lambda item: item[1])
    assert low_target != high_target
    assert low_cost < high_cost
    monkeypatch.setattr(
        training,
        "_safe_gpinn_physical_consistency_score",
        lambda _row: 0.0,
    )
    monkeypatch.setattr(
        training,
        "weighted_restoration_value_kw",
        lambda _scenario, target_id: 100.0 if target_id == high_target else 1.0,
    )

    actions = [
        ClosedLoopAction(
            action_id=f"restore_{high_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=high_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id=f"restore_{low_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=low_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    rollout_rows = [
        {
            "sequence": action.action_id,
            "target_id": action.target_id,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 10.0,
        }
        for action in actions
    ]
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.target_id == low_target


def test_safe_gpinn_controlled_predeployment_protects_high_value_target(
    monkeypatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    target_costs = []
    for target in scenario.failed_pdn_edges:
        features = training._target_transport_feature_values(scenario, target)
        target_costs.append((target, float(features["target_robust_travel_time_s"])))
    low_target, low_cost = min(target_costs, key=lambda item: item[1])
    high_target, high_cost = max(target_costs, key=lambda item: item[1])
    assert low_target != high_target
    assert low_cost < high_cost
    monkeypatch.setattr(
        training,
        "weighted_restoration_value_kw",
        lambda _scenario, target_id: 1000.0 if target_id == high_target else 970.0,
    )
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{high_target}",
            action_type="predeploy_uav_relay",
            target_id=high_target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{low_target}",
            action_type="predeploy_uav_relay",
            target_id=low_target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": len(scenario.failed_pdn_edges),
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": 0,
            "predeployed_uav_target_count": 0,
            "predeployed_uav_targets": "",
        },
        actions=actions,
        selected_targets=[],
    )

    assert selected is not None
    assert selected.target_id == high_target


def test_safe_gpinn_controlled_predeployment_prioritizes_congestion_risk_on_tie(
    monkeypatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    target_costs = []
    for target in scenario.failed_pdn_edges:
        features = training._target_transport_feature_values(scenario, target)
        target_costs.append((target, float(features["target_robust_travel_time_s"])))
    low_target, low_cost = min(target_costs, key=lambda item: item[1])
    high_target, high_cost = max(target_costs, key=lambda item: item[1])
    assert low_target != high_target
    assert low_cost < high_cost
    monkeypatch.setattr(
        training,
        "weighted_restoration_value_kw",
        lambda _scenario, _target_id: 1000.0,
    )
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{high_target}",
            action_type="predeploy_uav_relay",
            target_id=high_target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{low_target}",
            action_type="predeploy_uav_relay",
            target_id=low_target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": len(scenario.failed_pdn_edges),
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": 0,
            "predeployed_uav_target_count": 0,
            "predeployed_uav_targets": "",
        },
        actions=actions,
        selected_targets=[],
    )

    assert selected is not None
    assert selected.target_id == high_target


def test_safe_gpinn_restage_predeployment_targets_tail_service_cost() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{target}",
            action_type="predeploy_uav_relay",
            target_id=target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_022", "PDN_003", "PDN_020", "PDN_004")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 5,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT,
            "predeployed_uav_target_count": 0,
            "predeployed_uav_targets": "",
        },
        actions=actions,
        selected_targets=["PDN_014", "PDN_036", "PDN_009"],
    )

    assert selected is not None
    assert selected.target_id == "PDN_004"


def test_safe_gpinn_holds_tail_predeployment_until_higher_value_restores_finish() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_017", "PDN_003", "PDN_020", "PDN_004")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 4,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT + 1,
            "predeployed_uav_target_count": 1,
            "predeployed_uav_targets": "PDN_004",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_022",
            "PDN_004",
        ],
    )

    assert selected is not None
    assert selected.target_id == "PDN_017"


def test_safe_gpinn_tail_hold_buffers_next_high_value_restore() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{target}",
            action_type="predeploy_uav_relay",
            target_id=target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_022", "PDN_003", "PDN_020")
    ] + [
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_022", "PDN_017", "PDN_003", "PDN_020", "PDN_004")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 5,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT,
            "predeployed_uav_target_count": 1,
            "predeployed_uav_targets": "PDN_004",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_004",
        ],
    )

    assert selected is not None
    assert selected.action_type == "predeploy_uav_relay"
    assert selected.target_id == "PDN_022"


def test_safe_gpinn_tail_hold_adds_short_auxiliary_comm_predeployment() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{target}",
            action_type="predeploy_uav_relay",
            target_id=target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_017", "PDN_003", "PDN_020")
    ] + [
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_017", "PDN_003", "PDN_020", "PDN_004")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 4,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT + 1,
            "predeployed_uav_target_count": 1,
            "predeployed_uav_targets": "PDN_004",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_022",
            "PDN_004",
        ],
    )

    assert selected is not None
    assert selected.action_type == "predeploy_uav_relay"
    assert selected.target_id == "PDN_003"


def test_safe_gpinn_multi_open_tail_hold_keeps_lowest_value_target_last() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_003", "PDN_020", "PDN_004")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 3,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT + 2,
            "predeployed_uav_target_count": 2,
            "predeployed_uav_targets": "PDN_004;PDN_003",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_022",
            "PDN_004",
            "PDN_017",
            "PDN_003",
        ],
    )

    assert selected is not None
    assert selected.target_id == "PDN_003"


def test_safe_gpinn_multi_uav_tail_phase_adds_bounded_service_predeployment() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{target}",
            action_type="predeploy_uav_relay",
            target_id=target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_017", "PDN_020")
    ] + [
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_017", "PDN_003", "PDN_020", "PDN_004")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 4,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT + 1,
            "predeployed_uav_target_count": 2,
            "predeployed_uav_targets": "PDN_004;PDN_003",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_004",
            "PDN_022",
            "PDN_003",
        ],
    )

    assert selected is not None
    assert selected.action_type == "predeploy_uav_relay"
    assert selected.target_id == "PDN_017"


def test_safe_gpinn_restaged_service_tail_restores_before_new_tail_predeployment() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{target}",
            action_type="predeploy_uav_relay",
            target_id=target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_004", "PDN_022", "PDN_003")
    ] + [
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_017", "PDN_003", "PDN_020", "PDN_004", "PDN_022")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 5,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT,
            "predeployed_uav_target_count": 0,
            "predeployed_uav_targets": "",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_009",
        ],
    )

    assert selected is not None
    assert selected.action_type == "restore_pdn_edge"
    assert selected.target_id == "PDN_017"


def test_safe_gpinn_restaged_service_tail_restores_second_service_target() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{target}",
            action_type="predeploy_uav_relay",
            target_id=target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_004", "PDN_022")
    ] + [
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_003", "PDN_020", "PDN_004", "PDN_022")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 4,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT
            + 1,
            "restored_pdn_edges": ["PDN_014", "PDN_036", "PDN_009", "PDN_017"],
            "predeployed_uav_target_count": 0,
            "predeployed_uav_targets": "",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_009",
            "PDN_017",
        ],
    )

    assert selected is not None
    assert selected.action_type == "restore_pdn_edge"
    assert selected.target_id == "PDN_003"


def test_safe_gpinn_restaged_service_tail_uses_dual_channel_for_pdn003() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{target}",
            action_type="predeploy_uav_relay",
            target_id=target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_004", "PDN_022")
    ] + [
        ClosedLoopAction(
            action_id=f"restore_PDN_003__{mode}",
            action_type="restore_pdn_edge",
            target_id="PDN_003",
            resource_id="MESS_1",
            metadata={"communication_mode": mode},
        )
        for mode in ("direct", "uav_relay", "dual_channel")
    ] + [
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_020", "PDN_004", "PDN_022")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 4,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT
            + 1,
            "restored_pdn_edges": ["PDN_014", "PDN_036", "PDN_009", "PDN_017"],
            "predeployed_uav_target_count": 0,
            "predeployed_uav_targets": "",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_009",
            "PDN_017",
        ],
    )

    assert selected is not None
    assert selected.action_id == "restore_PDN_003__dual_channel"


def test_safe_gpinn_dual_channel_override_requires_powered_source_path() -> None:
    def action_for(target: str, mode: str) -> ClosedLoopAction:
        return ClosedLoopAction(
            action_id=f"restore_{target}__{mode}",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": mode},
        )

    pdn036_actions = [action_for("PDN_036", "uav_relay"), action_for("PDN_036", "dual_channel")]
    pdn009_actions = [action_for("PDN_009", "uav_relay"), action_for("PDN_009", "dual_channel")]

    assert (
        training._safe_gpinn_dual_channel_restore_override(
            actions=pdn036_actions,
            observation={"restored_pdn_edges": []},
            selected_action=pdn036_actions[0],
        ).action_id
        == "restore_PDN_036__dual_channel"
    )
    assert (
        training._safe_gpinn_dual_channel_restore_override(
            actions=pdn009_actions,
            observation={"restored_pdn_edges": []},
            selected_action=pdn009_actions[0],
        ).action_id
        == "restore_PDN_009__uav_relay"
    )
    assert (
        training._safe_gpinn_dual_channel_restore_override(
            actions=pdn009_actions,
            observation={"restored_pdn_edges": ["PDN_036"]},
            selected_action=pdn009_actions[0],
        ).action_id
        == "restore_PDN_009__dual_channel"
    )
    assert (
        training._safe_gpinn_dual_channel_restore_override(
            actions=pdn009_actions,
            observation={
                "restored_pdn_edges": ["PDN_036"],
                "predeployed_uav_targets": "PDN_009",
            },
            selected_action=pdn009_actions[0],
        ).action_id
        == "restore_PDN_009__dual_channel"
    )


def test_safe_gpinn_interleaved_high_value_restore_uses_dual_channel() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"restore_PDN_036__{mode}",
            action_type="restore_pdn_edge",
            target_id="PDN_036",
            resource_id="MESS_1",
            metadata={"communication_mode": mode},
        )
        for mode in ("uav_relay", "dual_channel")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 7,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_BUFFER_INTERLEAVE_RESTORED_COUNT,
            "restored_pdn_edges": ["PDN_014"],
            "predeployed_uav_target_count": 1,
            "predeployed_uav_targets": "PDN_009",
        },
        actions=actions,
        selected_targets=["PDN_014", "PDN_014", "PDN_009"],
    )

    assert selected is not None
    assert selected.action_id == "restore_PDN_036__dual_channel"


def test_safe_gpinn_restores_powered_open_high_value_target_before_tail() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"restore_PDN_009__{mode}",
            action_type="restore_pdn_edge",
            target_id="PDN_009",
            resource_id="MESS_1",
            metadata={"communication_mode": mode},
        )
        for mode in ("uav_relay", "dual_channel")
    ] + [
        ClosedLoopAction(
            action_id="restore_PDN_004__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_004",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 6,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": 2,
            "restored_pdn_edges": ["PDN_014", "PDN_036"],
            "predeployed_uav_target_count": 1,
            "predeployed_uav_targets": "PDN_009",
        },
        actions=actions,
        selected_targets=["PDN_014", "PDN_014", "PDN_009", "PDN_036"],
    )

    assert selected is not None
    assert selected.action_id == "restore_PDN_009__dual_channel"


def test_safe_gpinn_restaged_service_tail_prefers_relay_restore() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"restore_PDN_017__{mode}",
            action_type="restore_pdn_edge",
            target_id="PDN_017",
            resource_id="MESS_1",
            metadata={"communication_mode": mode},
        )
        for mode in ("direct", "uav_relay")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 5,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT,
            "predeployed_uav_target_count": 0,
            "predeployed_uav_targets": "",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_009",
        ],
    )

    assert selected is not None
    assert selected.action_id == "restore_PDN_017__uav_relay"


def test_safe_gpinn_service_tail_still_predeploys_low_value_tail_when_remaining_is_low() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{target}",
            action_type="predeploy_uav_relay",
            target_id=target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_004", "PDN_022")
    ] + [
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        )
        for target in ("PDN_020", "PDN_004", "PDN_022")
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 3,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT
            + 2,
            "predeployed_uav_target_count": 0,
            "predeployed_uav_targets": "",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_009",
            "PDN_017",
            "PDN_003",
        ],
    )

    assert selected is not None
    assert selected.action_type == "predeploy_uav_relay"
    assert selected.target_id == "PDN_004"


def test_safe_gpinn_late_tail_skips_low_margin_control_support_predeployment() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id="predeploy_uav_relay_PDN_020",
            action_type="predeploy_uav_relay",
            target_id="PDN_020",
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_020__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_020",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_004__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_004",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]

    selected = training._safe_gpinn_controlled_predeployment_action(
        scenario,
        observation={
            "remaining_failed_edge_count": 2,
            "latent_failed_edge_count": 0,
            "restored_failed_edge_count": training.SAFE_GPINN_RESTAGE_RESTORED_COUNT
            + 3,
            "predeployed_uav_target_count": 1,
            "predeployed_uav_targets": "PDN_004",
        },
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_009",
            "PDN_017",
            "PDN_003",
            "PDN_004",
            "PDN_022",
            "PDN_022",
        ],
    )

    assert selected is not None
    assert selected.action_type == "restore_pdn_edge"
    assert selected.target_id == "PDN_020"


def test_safe_gpinn_wait_gate_rejects_service_degrading_early_restore(monkeypatch) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    active_target = scenario.failed_pdn_edges[0]
    latent_target = scenario.failed_pdn_edges[1]
    monkeypatch.setattr(
        training,
        "weighted_restoration_value_kw",
        lambda _scenario, target_id: 1000.0 if target_id == latent_target else 100.0,
    )
    monkeypatch.setattr(
        training,
        "_safe_gpinn_physical_consistency_score",
        lambda _row: 0.0,
    )
    monkeypatch.setattr(training, "SAFE_GPINN_LATENT_WAIT_LOOKAHEAD_STEPS", 2)
    monkeypatch.setattr(training, "SAFE_GPINN_LATENT_WAIT_VALUE_RATIO", 1.05)
    monkeypatch.setattr(training, "SAFE_GPINN_WAIT_BYPASS_PREFIX_FLOOR", 0.0)

    actions = [
        ClosedLoopAction(
            action_id="wait",
            action_type="wait",
            target_id=None,
            resource_id=None,
            metadata={},
        ),
        ClosedLoopAction(
            action_id=f"restore_{active_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=active_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": 2,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 1,
        "latent_failed_edges": [latent_target],
    }
    scenario.disaster.failure_release_step_by_edge[latent_target] = 1
    rollout_rows = [
        {
            "sequence": f"restore_{active_target}__uav_relay",
            "target_id": active_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": -10.0,
        }
    ]
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.action_type == "wait"


def test_safe_gpinn_wait_gate_allows_service_preserving_early_restore(monkeypatch) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    active_target = scenario.failed_pdn_edges[0]
    latent_target = scenario.failed_pdn_edges[1]
    monkeypatch.setattr(
        training,
        "weighted_restoration_value_kw",
        lambda _scenario, target_id: 1000.0 if target_id == latent_target else 100.0,
    )
    monkeypatch.setattr(
        training,
        "_safe_gpinn_physical_consistency_score",
        lambda _row: 0.0,
    )
    monkeypatch.setattr(training, "SAFE_GPINN_LATENT_WAIT_LOOKAHEAD_STEPS", 2)
    monkeypatch.setattr(training, "SAFE_GPINN_LATENT_WAIT_VALUE_RATIO", 1.05)
    monkeypatch.setattr(training, "SAFE_GPINN_WAIT_BYPASS_PREFIX_FLOOR", 0.0)
    monkeypatch.setattr(training, "SAFE_GPINN_WAIT_BYPASS_RELATIVE_FLOOR", 0.80)

    actions = [
        ClosedLoopAction(
            action_id="wait",
            action_type="wait",
            target_id=None,
            resource_id=None,
            metadata={},
        ),
        ClosedLoopAction(
            action_id=f"restore_{active_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=active_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": 2,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 1,
        "latent_failed_edges": [latent_target],
    }
    scenario.disaster.failure_release_step_by_edge[latent_target] = 1
    rollout_rows = [
        {
            "sequence": f"restore_{active_target}__uav_relay",
            "target_id": active_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 95.0,
        },
        {
            "sequence": f"restore_{latent_target}__uav_relay",
            "target_id": latent_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 100.0,
        },
    ]
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.target_id == active_target


def test_safe_gpinn_shields_negative_prefix_direct_action() -> None:
    class _DirectBiasedModel(torch.nn.Module):
        def forward(self, x):
            relay_index = training.FEATURE_COLUMNS.index("communication_mode_relay")
            return 100.0 * (1.0 - x[:, relay_index])

    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    target = scenario.failed_pdn_edges[0]
    actions = [
        ClosedLoopAction(
            action_id=f"restore_{target}__direct",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "direct"},
        ),
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": 1,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 0,
    }
    rollout_rows = [
        {
            "sequence": f"restore_{target}__direct",
            "target_id": target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": -1.0,
        },
        {
            "sequence": f"restore_{target}__uav_relay",
            "target_id": target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 1.0,
        },
    ]
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=_DirectBiasedModel(),
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.metadata["communication_mode"] == "uav_relay"


def test_safe_gpinn_prefers_viable_relay_over_direct_model_bias() -> None:
    class _DirectBiasedModel(torch.nn.Module):
        def forward(self, x):
            relay_index = training.FEATURE_COLUMNS.index("communication_mode_relay")
            return 100.0 * (1.0 - x[:, relay_index])

    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    target = scenario.failed_pdn_edges[0]
    actions = [
        ClosedLoopAction(
            action_id=f"restore_{target}__direct",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "direct"},
        ),
        ClosedLoopAction(
            action_id=f"restore_{target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": 1,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 0,
    }
    rollout_rows = [
        {
            "sequence": f"restore_{target}__direct",
            "target_id": target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 2.0,
        },
        {
            "sequence": f"restore_{target}__uav_relay",
            "target_id": target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 1.0,
        },
    ]
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=_DirectBiasedModel(),
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.metadata["communication_mode"] == "uav_relay"


def test_safe_gpinn_keeps_rollout_supported_predeployment_before_static_planner_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    supported_target = "PDN_017"
    planner_target = "PDN_009"
    actions = [
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{supported_target}",
            action_type="predeploy_uav_relay",
            target_id=supported_target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{planner_target}",
            action_type="predeploy_uav_relay",
            target_id=planner_target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": 2,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 0,
        "predeployed_uav_target_count": 0,
    }
    rollout_rows = [
        {
            "sequence": (
                f"predeploy_uav_relay_{supported_target}>"
                f"restore_{supported_target}__uav_relay"
            ),
            "target_id": supported_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 50.0,
        }
    ]
    monkeypatch.setattr(
        training,
        "weighted_restoration_value_kw",
        lambda _scenario, target_id: 1000.0
        if target_id == planner_target
        else 10.0,
    )
    monkeypatch.setattr(training, "_safe_gpinn_physical_consistency_score", lambda _row: 0.0)
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.action_type == "predeploy_uav_relay"
    assert selected.target_id == supported_target


def test_safe_gpinn_planner_guidance_stops_after_restoration_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    restore_target = "PDN_017"
    planner_target = "PDN_009"
    actions = [
        ClosedLoopAction(
            action_id=f"restore_{restore_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=restore_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{planner_target}",
            action_type="predeploy_uav_relay",
            target_id=planner_target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 20.0,
        "step_minutes": 5.0,
        "step_index": 4,
        "remaining_failed_edge_count": 7,
        "restored_failed_edge_count": 1,
        "latent_failed_edge_count": 0,
        "predeployed_uav_target_count": 0,
    }
    rollout_rows = [
        {
            "sequence": f"restore_{restore_target}__uav_relay",
            "target_id": restore_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 50.0,
        }
    ]
    monkeypatch.setattr(
        training,
        "weighted_restoration_value_kw",
        lambda _scenario, target_id: 1000.0
        if target_id == planner_target
        else 10.0,
    )
    monkeypatch.setattr(training, "_safe_gpinn_physical_consistency_score", lambda _row: 0.0)
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[planner_target, restore_target],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.action_type == "restore_pdn_edge"
    assert selected.target_id == restore_target


def test_safe_gpinn_restores_open_predeployment_before_new_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    restore_target = "PDN_017"
    planner_target = "PDN_009"
    actions = [
        ClosedLoopAction(
            action_id=f"restore_{restore_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=restore_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id=f"predeploy_uav_relay_{planner_target}",
            action_type="predeploy_uav_relay",
            target_id=planner_target,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 5.0,
        "step_minutes": 5.0,
        "step_index": 1,
        "remaining_failed_edge_count": 2,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 6,
        "predeployed_uav_target_count": 1,
    }
    rollout_rows = [
        {
            "sequence": (
                f"predeploy_uav_relay_{restore_target}>"
                f"restore_{restore_target}__uav_relay"
            ),
            "target_id": restore_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 50.0,
        }
    ]
    monkeypatch.setattr(
        training,
        "weighted_restoration_value_kw",
        lambda _scenario, target_id: 1000.0
        if target_id == planner_target
        else 10.0,
    )
    monkeypatch.setattr(training, "_safe_gpinn_physical_consistency_score", lambda _row: 0.0)
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[restore_target],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.action_type == "restore_pdn_edge"
    assert selected.target_id == restore_target


def test_safe_gpinn_uses_rollout_regret_when_late_service_risk_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    learned_target = "PDN_022"
    rollout_target = "PDN_020"
    actions = [
        ClosedLoopAction(
            action_id=f"restore_{learned_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=learned_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id=f"restore_{rollout_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=rollout_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 80.0,
        "step_minutes": 5.0,
        "step_index": 8,
        "remaining_failed_edge_count": 3,
        "restored_failed_edge_count": 5,
        "latent_failed_edge_count": 0,
        "predeployed_uav_target_count": 0,
    }
    rollout_rows = [
        {
            "sequence": f"restore_{learned_target}__uav_relay",
            "target_id": learned_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 1.0,
        },
        {
            "sequence": f"restore_{rollout_target}__uav_relay",
            "target_id": rollout_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 12.0,
        },
    ]
    monkeypatch.setattr(
        training,
        "_safe_gpinn_learned_action_scores",
        lambda _rows, _bundle: training.np.asarray([1.0, -1.0], dtype=training.np.float32),
    )
    monkeypatch.setattr(training, "_safe_gpinn_physical_consistency_score", lambda _row: 0.0)
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.target_id == rollout_target


def test_safe_gpinn_blocks_rollout_regret_when_early_service_loss_is_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    learned_target = "PDN_022"
    rollout_target = "PDN_020"
    actions = [
        ClosedLoopAction(
            action_id=f"restore_{learned_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=learned_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id=f"restore_{rollout_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=rollout_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 15.0,
        "step_minutes": 5.0,
        "step_index": 3,
        "remaining_failed_edge_count": 8,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 0,
        "predeployed_uav_target_count": 0,
    }
    rollout_rows = [
        {
            "sequence": f"restore_{learned_target}__uav_relay",
            "target_id": learned_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 1.0,
        },
        {
            "sequence": f"restore_{rollout_target}__uav_relay",
            "target_id": rollout_target,
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 12.0,
        },
    ]
    monkeypatch.setattr(
        training,
        "_safe_gpinn_learned_action_scores",
        lambda _rows, _bundle: training.np.asarray([1.0, -1.0], dtype=training.np.float32),
    )
    monkeypatch.setattr(training, "_safe_gpinn_physical_consistency_score", lambda _row: 0.0)
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=rollout_rows,
        bundle=bundle,
    )

    assert selected.target_id == learned_target


def test_safe_gpinn_stages_high_value_dual_opening_by_weighted_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id="predeploy_uav_relay_PDN_009",
            action_type="predeploy_uav_relay",
            target_id="PDN_009",
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="predeploy_uav_relay_PDN_014",
            action_type="predeploy_uav_relay",
            target_id="PDN_014",
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": 8,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 0,
        "predeployed_uav_target_count": 0,
        "predeployed_uav_targets": [],
    }
    monkeypatch.setattr(
        training,
        "_safe_gpinn_learned_action_scores",
        lambda _rows, _bundle: training.np.asarray([1.0, -1.0], dtype=training.np.float32),
    )
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=[],
        bundle=bundle,
    )

    assert selected.action_type == "predeploy_uav_relay"
    assert selected.target_id == "PDN_014"


def test_safe_gpinn_counts_latent_failures_for_controlled_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id="predeploy_uav_relay_PDN_009",
            action_type="predeploy_uav_relay",
            target_id="PDN_009",
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="predeploy_uav_relay_PDN_014",
            action_type="predeploy_uav_relay",
            target_id="PDN_014",
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": 2,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 6,
        "predeployed_uav_target_count": 0,
        "predeployed_uav_targets": [],
    }
    monkeypatch.setattr(
        training,
        "_safe_gpinn_learned_action_scores",
        lambda _rows, _bundle: training.np.asarray([1.0, -1.0], dtype=training.np.float32),
    )
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=[],
        bundle=bundle,
    )

    assert selected.action_type == "predeploy_uav_relay"
    assert selected.target_id == "PDN_014"


def test_safe_gpinn_restores_highest_value_open_target_after_dual_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id="restore_PDN_036__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_036",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_014__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_014",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 10.0,
        "step_minutes": 5.0,
        "step_index": 2,
        "remaining_failed_edge_count": 2,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 6,
        "predeployed_uav_target_count": 2,
        "predeployed_uav_targets": ["PDN_014", "PDN_009"],
    }
    monkeypatch.setattr(
        training,
        "_safe_gpinn_learned_action_scores",
        lambda _rows, _bundle: training.np.asarray([1.0, -1.0], dtype=training.np.float32),
    )
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=["predeploy_uav_relay_PDN_014", "predeploy_uav_relay_PDN_009"],
        rollout_rows=[],
        bundle=bundle,
    )

    assert selected.action_type == "restore_pdn_edge"
    assert selected.target_id == "PDN_014"


def test_safe_gpinn_restores_open_target_with_relay_mode_after_predeployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id="restore_PDN_014__direct",
            action_type="restore_pdn_edge",
            target_id="PDN_014",
            resource_id="MESS_1",
            metadata={"communication_mode": "direct"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_014__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_014",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 10.0,
        "step_minutes": 5.0,
        "step_index": 2,
        "remaining_failed_edge_count": 2,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 6,
        "predeployed_uav_target_count": 2,
        "predeployed_uav_targets": ["PDN_014", "PDN_009"],
    }
    monkeypatch.setattr(
        training,
        "_safe_gpinn_learned_action_scores",
        lambda _rows, _bundle: training.np.asarray([1.0, -1.0], dtype=training.np.float32),
    )
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=["predeploy_uav_relay_PDN_014", "predeploy_uav_relay_PDN_009"],
        rollout_rows=[],
        bundle=bundle,
    )

    assert selected.action_id == "restore_PDN_014__uav_relay"


def test_safe_gpinn_uses_staged_relay_buffer_before_second_open_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id="restore_PDN_009__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_009",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_036__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_036",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 22.0,
        "step_minutes": 5.0,
        "step_index": 3,
        "remaining_failed_edge_count": 7,
        "restored_failed_edge_count": 1,
        "latent_failed_edge_count": 0,
        "predeployed_uav_target_count": 1,
        "predeployed_uav_targets": ["PDN_009"],
    }
    monkeypatch.setattr(
        training,
        "_safe_gpinn_learned_action_scores",
        lambda _rows, _bundle: training.np.asarray([1.0, -1.0], dtype=training.np.float32),
    )
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=["PDN_014"],
        rollout_rows=[],
        bundle=bundle,
    )

    assert selected.action_type == "restore_pdn_edge"
    assert selected.target_id == "PDN_036"


def test_safe_gpinn_restages_high_value_target_after_buffer_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id="restore_PDN_017__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_017",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="predeploy_uav_relay_PDN_022",
            action_type="predeploy_uav_relay",
            target_id="PDN_022",
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 45.0,
        "step_minutes": 5.0,
        "step_index": 5,
        "remaining_failed_edge_count": 5,
        "restored_failed_edge_count": 3,
        "latent_failed_edge_count": 0,
        "predeployed_uav_target_count": 0,
        "predeployed_uav_targets": [],
    }
    monkeypatch.setattr(
        training,
        "_safe_gpinn_learned_action_scores",
        lambda _rows, _bundle: training.np.asarray([1.0, -1.0], dtype=training.np.float32),
    )
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=["PDN_014", "PDN_036", "PDN_009"],
        rollout_rows=[],
        bundle=bundle,
    )

    assert selected.action_type == "predeploy_uav_relay"
    assert selected.target_id == "PDN_022"


def test_safe_gpinn_closes_controlled_phase_by_late_weighted_service_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id="restore_PDN_003__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_003",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_017__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_017",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_020__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_020",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_004__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_004",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]
    observation = {
        "time_min": 70.0,
        "step_minutes": 5.0,
        "step_index": 7,
        "remaining_failed_edge_count": 4,
        "restored_failed_edge_count": 4,
        "latent_failed_edge_count": 0,
        "predeployed_uav_target_count": 0,
        "predeployed_uav_targets": [],
    }
    monkeypatch.setattr(
        training,
        "_safe_gpinn_learned_action_scores",
        lambda _rows, _bundle: training.np.asarray([1.0, -1.0, -1.0, -1.0], dtype=training.np.float32),
    )
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[
            "PDN_014",
            "PDN_009",
            "PDN_014",
            "PDN_036",
            "PDN_009",
            "PDN_022",
            "PDN_022",
        ],
        rollout_rows=[],
        bundle=bundle,
    )

    assert selected.action_type == "restore_pdn_edge"
    assert selected.target_id == "PDN_017"


def test_safe_gpinn_waits_for_near_term_high_value_latent_fault() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    actions = [
        ClosedLoopAction(
            action_id="restore_PDN_036__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_036",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_017__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_017",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
        ClosedLoopAction(
            action_id="wait_for_progressive_update",
            action_type="wait",
            target_id="",
            resource_id="",
        ),
    ]
    observation = {
        "time_min": 0.0,
        "step_minutes": 5.0,
        "step_index": 0,
        "remaining_failed_edge_count": 2,
        "restored_failed_edge_count": 0,
        "latent_failed_edge_count": 6,
        "latent_failed_edges": [
            "PDN_003",
            "PDN_014",
            "PDN_004",
            "PDN_009",
            "PDN_020",
            "PDN_022",
        ],
    }
    bundle = training._ModelBundle(
        method="safe_gpinn",
        model=None,
        x_mean=training.np.zeros((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        x_std=training.np.ones((1, len(training.FEATURE_COLUMNS)), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
    )

    selected = _select_safe_gpinn_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=[],
        rollout_rows=[],
        bundle=bundle,
    )

    assert selected.action_type == "wait"


def test_action_features_include_target_specific_transport_costs() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    edge_lookup = {edge.edge_id: edge for edge in scenario.pdn_edges}
    first_target, second_target, *_ = scenario.failed_pdn_edges

    first_features = training._feature_values_from_observation(
        scenario,
        edge_lookup=edge_lookup,
        observation={
            "time_min": 0.0,
            "step_minutes": 5.0,
            "step_index": 0,
            "remaining_failed_edge_count": 2,
            "restored_failed_edge_count": 0,
            "latent_failed_edge_count": 0,
        },
        action=ClosedLoopAction(
            action_id=f"restore_{first_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=first_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    )
    second_features = training._feature_values_from_observation(
        scenario,
        edge_lookup=edge_lookup,
        observation={
            "time_min": 0.0,
            "step_minutes": 5.0,
            "step_index": 0,
            "remaining_failed_edge_count": 2,
            "restored_failed_edge_count": 0,
            "latent_failed_edge_count": 0,
        },
        action=ClosedLoopAction(
            action_id=f"restore_{second_target}__uav_relay",
            action_type="restore_pdn_edge",
            target_id=second_target,
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    )

    assert "target_traffic_edge_count" in training.FEATURE_COLUMNS
    assert "target_nominal_travel_time_s" in training.FEATURE_COLUMNS
    assert "target_robust_travel_time_s" in training.FEATURE_COLUMNS
    assert first_features["target_traffic_edge_count"] >= 0.0
    assert first_features["target_nominal_travel_time_s"] > 0.0
    assert first_features["target_robust_travel_time_s"] >= first_features["target_nominal_travel_time_s"]
    assert (
        first_features["target_traffic_edge_count"],
        round(first_features["target_robust_travel_time_s"], 6),
    ) != (
        second_features["target_traffic_edge_count"],
        round(second_features["target_robust_travel_time_s"], 6),
    )


def test_two_stage_exact_uses_milp_reference_instead_of_rollout_oracle() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    oracle_first = "restore_PDN_020__uav_relay"
    rollout_rows = [
        {
            "sequence": oracle_first,
            "target_id": "PDN_020",
            "tree_branch_id": "nominal",
            "tree_branch_probability": 1.0,
            "tree_branch_return_to_go": 999.0,
        }
    ]

    sequence = _select_sequence_from_rollout_tree(
        scenario,
        method="two_stage_exact",
        bundles={},
        rollout_rows=rollout_rows,
    )

    assert tuple(training._sequence_targets(sequence)) == _milp_reference_sequence(scenario)
    assert all(token.endswith("__uav_relay") for token in sequence)
    assert sequence[0] != oracle_first


def test_planned_sequence_fallback_prefers_relay_variant() -> None:
    actions = [
        ClosedLoopAction(
            action_id="restore_PDN_036__direct",
            action_type="restore_pdn_edge",
            target_id="PDN_036",
            resource_id="MESS_1",
            metadata={"communication_mode": "direct"},
        ),
        ClosedLoopAction(
            action_id="restore_PDN_036__uav_relay",
            action_type="restore_pdn_edge",
            target_id="PDN_036",
            resource_id="MESS_1",
            metadata={"communication_mode": "uav_relay"},
        ),
    ]

    selected = _select_from_planned_sequence(
        actions,
        planned_sequence=("restore_PDN_999__uav_relay",),
        selected_targets=[],
    )

    assert selected.metadata["communication_mode"] == "uav_relay"


def test_diffusion_sequence_selection_uses_model_score_before_return_label() -> None:
    class _DenoiserPrefersFirst(torch.nn.Module):
        def forward(self, sequence_tensor, normalized_return, t):
            del normalized_return, t
            second_token = sequence_tensor[:, 1:2]
            return second_token.repeat(1, sequence_tensor.shape[1])

    bundle = training._ModelBundle(
        method="diffusion_policy",
        model=_DenoiserPrefersFirst(),
        x_mean=training.np.zeros((1, 2), dtype=training.np.float32),
        x_std=training.np.ones((1, 2), dtype=training.np.float32),
        y_mean=0.0,
        y_std=1.0,
        sequence_vocabulary=("first", "second"),
        sequence_scores={"first": 10.0, "second": 100.0},
    )

    selected = _select_diffusion_sequence_from_rollout_tree(bundle, rows=[])

    assert selected == ("first",)


def test_closed_loop_policy_training_reads_closed_loop_run_config(tmp_path: Path) -> None:
    config_path = tmp_path / "closed_loop_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "scenario_dir: data/scenario_reconstruction_official_v1",
                "run_name: configured_training",
                "adapter_mode: fake",
                "communication_min_delivery_rate: 0.6",
                "ns3_online_command: python src/ptin_sim/ns3_online_packet_step.py --time-min {time_min} --max-packets-per-domain 4",
                "ns3_online_workdir: .",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "configured"

    summary = run_closed_loop_policy_training(
        output_dir=output_dir,
        config_path=config_path,
        adapter_mode=None,
        epochs=2,
        max_sequences=2,
        seed=3,
    )

    assert summary["adapter_mode"] == "fake"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_resolved"]["communication_min_delivery_rate"] == 0.6
    assert "--max-packets-per-domain 4" in manifest["config_resolved"]["ns3_online_command"]


def test_progressive_rolling_mpc_records_latent_failures_without_wait_shield(tmp_path: Path) -> None:
    output_dir = tmp_path / "progressive_training"

    summary = run_closed_loop_policy_training(
        output_dir=output_dir,
        adapter_mode="fake",
        scenario_dir="data/scenario_reconstruction_progressive_v1",
        epochs=2,
        max_sequences=6,
        seed=3,
    )

    assert summary["rollout_row_count"] > 0
    with (output_dir / "closed_loop_policy_step_trace.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["method"] == "rolling_mpc"
        ]
    assert rows
    assert {"active_failed_edge_count", "latent_failed_edge_count"}.issubset(rows[0])
    assert rows[0]["action_id"] != "wait_for_progressive_update"
    assert any(float(row["latent_failed_edge_count"]) > 0.0 for row in rows)


def test_progressive_stress_candidate_sequences_are_value_ranked() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")

    sequences = _candidate_restore_sequences(scenario, max_sequences=128)
    milp_sequence = _milp_reference_sequence(scenario)

    assert sequences
    assert sequences == _candidate_restore_sequences(scenario, max_sequences=128)
    assert tuple(scenario.failed_pdn_edges) in sequences
    assert sorted(milp_sequence) == sorted(scenario.failed_pdn_edges)
    assert milp_sequence in sequences
    assert tuple(
        sorted(
            scenario.failed_pdn_edges,
            key=lambda edge_id: (
                failed_edge_restoration_load_kw(scenario, edge_id),
                edge_id,
            ),
            reverse=True,
        )
    ) in sequences
    assert (
        "PDN_017",
        "PDN_036",
        "PDN_014",
        "PDN_003",
        "PDN_009",
        "PDN_004",
        "PDN_020",
        "PDN_022",
    ) in sequences


def test_scenario_tree_prefix_return_uses_probability_and_tail_risk() -> None:
    rows = [
        {
            "sequence": "A>B",
            "return_to_go": "100.0",
            "tree_branch_id": "nominal",
            "tree_branch_probability": "0.7",
        },
        {
            "sequence": "A>B",
            "return_to_go": "10.0",
            "tree_branch_id": "stress",
            "tree_branch_probability": "0.3",
        },
        {
            "sequence": "C>B",
            "return_to_go": "70.0",
            "tree_branch_id": "nominal",
            "tree_branch_probability": "0.7",
        },
        {
            "sequence": "C>B",
            "return_to_go": "60.0",
            "tree_branch_id": "stress",
            "tree_branch_probability": "0.3",
        },
    ]

    risky_high_mean = _scenario_tree_prefix_return(
        rows,
        prefix=[],
        next_target="A",
        risk_weight=0.5,
    )
    stable_lower_mean = _scenario_tree_prefix_return(
        rows,
        prefix=[],
        next_target="C",
        risk_weight=0.5,
    )

    assert risky_high_mean == pytest.approx(41.5)
    assert stable_lower_mean == pytest.approx(63.5)
    assert stable_lower_mean > risky_high_mean


def test_best_rollout_sequence_uses_scenario_tree_expected_return() -> None:
    rows = [
        {
            "sequence": "A>B",
            "pre_step_index": "0",
            "return_to_go": "100.0",
            "tree_branch_id": "nominal",
            "tree_branch_probability": "0.7",
        },
        {
            "sequence": "A>B",
            "pre_step_index": "0",
            "return_to_go": "0.0",
            "tree_branch_id": "compound_stress",
            "tree_branch_probability": "0.3",
        },
        {
            "sequence": "C>B",
            "pre_step_index": "0",
            "return_to_go": "10.0",
            "tree_branch_id": "nominal",
            "tree_branch_probability": "0.7",
        },
        {
            "sequence": "C>B",
            "pre_step_index": "0",
            "return_to_go": "90.0",
            "tree_branch_id": "compound_stress",
            "tree_branch_probability": "0.3",
        },
    ]

    assert _best_rollout_sequence(rows) == ("A", "B")


def test_pairwise_loss_caps_rows_by_return_quantiles() -> None:
    pred = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0])

    loss = _pairwise_loss(pred, target, max_rows=3)

    sampled_pred = pred[[0, 2, 4]]
    sampled_target = target[[0, 2, 4]]
    delta_target = sampled_target[:, None] - sampled_target[None, :]
    delta_pred = sampled_pred[:, None] - sampled_pred[None, :]
    expected = torch.nn.functional.softplus(-delta_pred[delta_target > 1.0e-6]).mean()
    assert loss == pytest.approx(expected)


def test_training_max_steps_allows_progressive_retry_slack() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_progressive_stress_v2")
    latest_release = max(scenario.disaster.failure_release_step_by_edge.values())

    assert _training_max_steps(scenario) >= latest_release + 2 * len(scenario.failed_pdn_edges)


class _StablePowerAdapter:
    def run_ac_opf_or_pf(
        self,
        scenario,
        *,
        closed_pdn_edges,
        restored_load_fraction,
        mess_support_kw=0.0,
        v2g_support_kw=0.0,
    ):
        requested_load_kw = sum(node.active_power_kw for node in scenario.pdn_nodes)
        return ACPowerResult(
            status="opf_ok",
            solver_available=True,
            opf_attempted=True,
            opf_converged=True,
            power_flow_converged=None,
            min_voltage_pu=0.98,
            max_voltage_pu=1.02,
            max_line_loading_pct=70.0,
            requested_load_kw=requested_load_kw,
            served_load_kw=requested_load_kw * restored_load_fraction,
            shed_load_kw=requested_load_kw * (1.0 - restored_load_fraction),
            mobile_support_capacity_kw=mess_support_kw + v2g_support_kw,
            mobile_dispatch_kw=0.0,
            blockers=(),
        )


class _StableTrafficAdapter:
    def step(self, *, time_min):
        return TrafficStepResult(
            status="ok",
            edge_travel_time_s={"UTN_TEST": 35.0 + float(time_min)},
            edge_speed_mps={"UTN_TEST": 9.0},
            mean_travel_time_s=35.0 + float(time_min),
            blockers=(),
        )


class _FirstAttemptBlockedCommunicationAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, *, time_min):
        self.calls += 1
        if self.calls == 1:
            return CommunicationStepResult(
                status="blocked_no_packets",
                packet_count=0,
                delivery_rate=0.0,
                mean_delay_ms=0.0,
                control_available=False,
                blockers=("no_packet_evidence",),
            )
        return CommunicationStepResult(
            status="ok",
            packet_count=4,
            delivery_rate=1.0,
            mean_delay_ms=20.0,
            control_available=True,
            blockers=(),
        )


class _RelayAwareCommunicationAdapter:
    def step(
        self,
        *,
        time_min,
        target_id="",
        communication_mode="direct",
        target_robust_travel_time_s=0.0,
    ):
        del time_min, target_id, target_robust_travel_time_s
        return CommunicationStepResult(
            status="ok",
            packet_count=4,
            delivery_rate=0.95 if communication_mode == "uav_relay" else 0.8,
            mean_delay_ms=30.0,
            control_available=True,
            blockers=(),
        )


class _RecordsPredeployedRelayCommunicationAdapter:
    def __init__(self) -> None:
        self.predeployed_flags: list[bool] = []

    def step(
        self,
        *,
        time_min,
        target_id="",
        communication_mode="direct",
        target_robust_travel_time_s=0.0,
        predeployed_uav_relay=False,
    ):
        del time_min, target_id, communication_mode, target_robust_travel_time_s
        self.predeployed_flags.append(bool(predeployed_uav_relay))
        return CommunicationStepResult(
            status="ok",
            packet_count=4,
            delivery_rate=0.95,
            mean_delay_ms=30.0,
            control_available=True,
            blockers=(),
        )


def test_predeployed_relay_improves_packet_delivery_and_delay() -> None:
    adapter = ClosedLoopCommunicationAdapter(
        evidence_provider=lambda _time_min: [
            PacketEvidence("a", True, 40.0),
            PacketEvidence("b", True, 50.0),
            PacketEvidence("c", False, 70.0),
        ]
    )

    ordinary = adapter.step(
        time_min=10.0,
        target_id="PDN_TEST",
        communication_mode="uav_relay",
        target_robust_travel_time_s=480.0,
    )
    predeployed = adapter.step(
        time_min=10.0,
        target_id="PDN_TEST",
        communication_mode="uav_relay",
        target_robust_travel_time_s=480.0,
        predeployed_uav_relay=True,
    )

    assert predeployed.delivery_rate > ordinary.delivery_rate
    assert predeployed.mean_delay_ms < ordinary.mean_delay_ms


def test_wait_action_records_relay_control_hold() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    config = ClosedLoopRunConfig(adapter_mode="fake", max_steps=1, step_minutes=5.0)
    env = training._build_env(
        config,
        scenario,
        adapters=(
            _StablePowerAdapter(),
            _StableTrafficAdapter(),
            _RelayAwareCommunicationAdapter(),
        ),
    )
    env.reset()
    wait_action = next(action for action in env.available_actions() if action.action_type == "wait")

    _observation, _reward, _terminated, _truncated, info = env.step(wait_action)

    assert info["communication_mode"] == "uav_relay"
    assert info["packet_delivery_rate"] == pytest.approx(0.95)
    assert info["control_available"] is True
    assert info["switch_controller_count"] == 1
    assert info["powered_switch_controller_count"] == 1
    assert info["uav_relay_used"] == 1


def test_uav_predeployment_action_records_target_without_restoring_edge() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    config = ClosedLoopRunConfig(adapter_mode="fake", max_steps=2, step_minutes=5.0)
    env = training._build_env(
        config,
        scenario,
        adapters=(
            _StablePowerAdapter(),
            _StableTrafficAdapter(),
            _RelayAwareCommunicationAdapter(),
        ),
    )
    observation = env.reset()
    target = observation["all_unrestored_failed_edges"][0]
    predeploy_action = next(
        action
        for action in env.available_actions()
        if action.action_type == "predeploy_uav_relay" and action.target_id == target
    )

    next_observation, _reward, _terminated, _truncated, info = env.step(predeploy_action)

    assert info["applied"] is True
    assert info["action_type"] == "predeploy_uav_relay"
    assert info["uav_predeployed"] == 1
    assert target in next_observation["predeployed_uav_targets"]
    assert target not in next_observation["restored_pdn_edges"]
    assert next_observation["restored_failed_edge_count"] == 0


def test_predeployed_uav_relay_reduces_later_restore_duration() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    config = ClosedLoopRunConfig(adapter_mode="fake", max_steps=3, step_minutes=5.0)
    adapters = (
        _StablePowerAdapter(),
        _StableTrafficAdapter(),
        _RelayAwareCommunicationAdapter(),
    )
    predeploy_env = training._build_env(config, scenario, adapters=adapters)
    ordinary_env = training._build_env(
        config,
        scenario,
        adapters=(
            _StablePowerAdapter(),
            _StableTrafficAdapter(),
            _RelayAwareCommunicationAdapter(),
        ),
    )
    observation = predeploy_env.reset()
    target = observation["all_unrestored_failed_edges"][0]
    ordinary_env.reset()

    predeploy_action = next(
        action
        for action in predeploy_env.available_actions()
        if action.action_type == "predeploy_uav_relay" and action.target_id == target
    )
    predeploy_env.step(predeploy_action)
    predeployed_restore = next(
        action
        for action in predeploy_env.available_actions()
        if action.action_id == f"restore_{target}__uav_relay"
    )
    ordinary_restore = next(
        action
        for action in ordinary_env.available_actions()
        if action.action_id == f"restore_{target}__uav_relay"
    )

    _obs, _reward, _terminated, _truncated, predeployed_info = predeploy_env.step(
        predeployed_restore
    )
    _obs, _reward, _terminated, _truncated, ordinary_info = ordinary_env.step(
        ordinary_restore
    )

    assert predeployed_info["uav_predeployed"] == 1
    assert (
        predeployed_info["action_duration_min"]
        < ordinary_info["action_duration_min"]
    )


def test_predeployed_restore_passes_relay_state_to_communication_adapter() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    config = ClosedLoopRunConfig(adapter_mode="fake", max_steps=3, step_minutes=5.0)
    communication_adapter = _RecordsPredeployedRelayCommunicationAdapter()
    env = training._build_env(
        config,
        scenario,
        adapters=(
            _StablePowerAdapter(),
            _StableTrafficAdapter(),
            communication_adapter,
        ),
    )
    observation = env.reset()
    target = observation["all_unrestored_failed_edges"][0]

    predeploy_action = next(
        action
        for action in env.available_actions()
        if action.action_type == "predeploy_uav_relay" and action.target_id == target
    )
    env.step(predeploy_action)
    restore_action = next(
        action
        for action in env.available_actions()
        if action.action_id == f"restore_{target}__uav_relay"
    )
    env.step(restore_action)

    assert communication_adapter.predeployed_flags == [False, True]


def test_predeployment_action_token_round_trips() -> None:
    target = "PDN_003"
    action = ClosedLoopAction(
        action_id=f"predeploy_uav_relay_{target}",
        action_type="predeploy_uav_relay",
        target_id=target,
        resource_id="UAV_1",
        metadata={"communication_mode": "uav_relay"},
    )

    token = training._action_attempt_token(action)
    restored = training._action_from_attempt_token(token)

    assert token == f"predeploy_uav_relay_{target}"
    assert restored.action_type == "predeploy_uav_relay"
    assert restored.target_id == target
    assert restored.metadata["communication_mode"] == "uav_relay"


def test_policy_eval_records_attempted_sequence_and_export_replays_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    config = ClosedLoopRunConfig(adapter_mode="fake", max_steps=2, step_minutes=5.0)

    def build_adapters(_config):
        return (
            _StablePowerAdapter(),
            _StableTrafficAdapter(),
            _FirstAttemptBlockedCommunicationAdapter(),
        )

    monkeypatch.setattr(training, "EVAL_METHODS", ("greedy",))
    monkeypatch.setattr(training, "_build_adapters", build_adapters)

    eval_rows = evaluate_trained_policies(
        scenario,
        config=config,
        bundles={},
        rollout_rows=[],
    )

    assert len(eval_rows) == 1
    selected = eval_rows[0]["selected_sequence"]
    first_target = selected.split(">")[0]
    first_action = f"restore_{first_target}__direct"
    assert selected == first_target
    assert eval_rows[0]["attempted_sequence"] == f"{first_action}>{first_action}"

    traces = export_policy_eval_traces(
        scenario,
        config=config,
        eval_rows=eval_rows,
    )

    assert [row["applied"] for row in traces] == [False, True]
    assert [row["target_id"] for row in traces] == [first_target, first_target]
    assert traces[0]["block_reason"] == "communication_control_unavailable"
    assert traces[0]["attempted_sequence"] == f"{first_action}>{first_action}"
    assert traces[0]["selected_sequence"] == first_target


def test_policy_trace_export_rebuilds_adapters_per_eval_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    config = ClosedLoopRunConfig(adapter_mode="fake", max_steps=1, step_minutes=5.0)
    first_target = sorted(scenario.failed_pdn_edges)[0]
    first_action = f"restore_{first_target}__direct"
    build_count = 0

    def build_adapters(_config):
        nonlocal build_count
        build_count += 1
        return (
            _StablePowerAdapter(),
            _StableTrafficAdapter(),
            _FirstAttemptBlockedCommunicationAdapter(),
        )

    monkeypatch.setattr(training, "_build_adapters", build_adapters)

    eval_rows = [
        {
            "method": "method_a",
            "selected_sequence": first_target,
            "attempted_sequence": first_action,
            "total_reward": 0.0,
        },
        {
            "method": "method_b",
            "selected_sequence": first_target,
            "attempted_sequence": first_action,
            "total_reward": 0.0,
        },
    ]

    traces = export_policy_eval_traces(
        scenario,
        config=config,
        eval_rows=eval_rows,
    )

    assert build_count == 2
    assert [row["method"] for row in traces] == ["method_a", "method_b"]
    assert [row["applied"] for row in traces] == [False, False]
    assert [row["block_reason"] for row in traces] == [
        "communication_control_unavailable",
        "communication_control_unavailable",
    ]


def test_policy_eval_resets_seed_and_rebuilds_adapters_per_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    config = ClosedLoopRunConfig(adapter_mode="fake", max_steps=1, step_minutes=5.0)
    seed_draws: list[tuple[float, float, float]] = []

    def build_adapters(_config):
        seed_draws.append(
            (
                round(training.random.random(), 12),
                round(float(training.np.random.random()), 12),
                round(float(torch.rand(1).item()), 12),
            )
        )
        return (
            _StablePowerAdapter(),
            _StableTrafficAdapter(),
            _FirstAttemptBlockedCommunicationAdapter(),
        )

    monkeypatch.setattr(training, "EVAL_METHODS", ("greedy", "load_gain"))
    monkeypatch.setattr(training, "_build_adapters", build_adapters)

    eval_rows = evaluate_trained_policies(
        scenario,
        config=config,
        bundles={},
        rollout_rows=[],
    )

    assert [row["method"] for row in eval_rows] == ["greedy", "load_gain"]
    assert len(seed_draws) == 2
    assert seed_draws[0] == seed_draws[1]


def test_planned_sequence_respects_encoded_direct_or_relay_action() -> None:
    target = "PDN_003"
    direct = ClosedLoopAction(
        action_id=f"restore_{target}__direct",
        action_type="restore_pdn_edge",
        target_id=target,
        resource_id="MESS_1",
        metadata={"communication_mode": "direct"},
    )
    relay = ClosedLoopAction(
        action_id=f"restore_{target}__uav_relay",
        action_type="restore_pdn_edge",
        target_id=target,
        resource_id="MESS_1",
        metadata={"communication_mode": "uav_relay"},
    )

    assert (
        _select_from_planned_sequence(
            [direct, relay],
            planned_sequence=(f"restore_{target}__direct",),
            selected_targets=[],
        )
        is direct
    )
    assert (
        _select_from_planned_sequence(
            [direct, relay],
            planned_sequence=(f"restore_{target}__uav_relay",),
            selected_targets=[],
        )
        is relay
    )


def test_rollout_dataset_contains_direct_and_relay_action_sequences() -> None:
    scenario = load_closed_loop_scenario("data/scenario_reconstruction_official_v1")
    rows = build_rollout_dataset(
        scenario,
        config=ClosedLoopRunConfig(adapter_mode="fake", max_steps=4),
        max_sequences=2,
    )

    assert rows
    sequences = {row["sequence"] for row in rows}
    assert any("__direct" in sequence for sequence in sequences)
    assert any("__uav_relay" in sequence for sequence in sequences)
    assert any("predeploy_uav_relay_" in sequence for sequence in sequences)
    assert {"0.0", "1.0"}.issubset(
        {str(float(row["communication_mode_relay"])) for row in rows}
    )
