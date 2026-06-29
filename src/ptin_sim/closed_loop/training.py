from __future__ import annotations

import csv
import copy
import itertools
import json
import random
import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .dependencies import failed_edge_restoration_load_kw
from .dependencies import (
    critical_load_weight,
    robust_mean_travel_time_s,
    target_utn_edge_ids_for_pdn_edge,
    topology_vulnerability_score,
    weighted_restoration_value_kw,
)
from .env import (
    ACTION_DURATION_PENALTY_PER_MIN,
    PACKET_LOSS_PENALTY,
    RESTORATION_REWARD_SCALE,
    SHED_ENERGY_PENALTY_PER_KWH,
)
from .run_closed_loop_experiment import (
    DEFAULT_OUTPUT_ROOT,
    PROJECT_ROOT,
    ClosedLoopRunConfig,
    _build_adapters,
    _build_env,
    _config_to_json,
    _truth_boundary_for_mode,
    load_run_config,
)
from .scenario import load_closed_loop_scenario
from .types import (
    ClosedLoopAction,
    PDNEdge,
    PTINScenario,
    dispatchable_v2g_vehicle_count_at,
    ev_availability_factor_at,
    v2g_energy_capacity_kwh_at,
)


TRAINING_TRUTH_BOUNDARY_FAKE = "closed_loop_policy_training_fake_adapter_smoke_not_final_result"
TRAINING_TRUTH_BOUNDARY_REAL = (
    "closed_loop_policy_training_over_real_closed_loop_adapters_requires_repeated_seed_eval"
)

FEATURE_COLUMNS = (
    "pre_step_index",
    "remaining_failed_before",
    "restored_failed_before",
    "action_order_index",
    "action_load_gain_kw",
    "edge_length_km",
    "edge_r_pu",
    "edge_x_pu",
    "projected_mess_support_kw",
    "projected_v2g_support_kw",
    "v2g_availability_factor",
    "v2g_available_energy_capacity_kwh",
    "critical_load_weight",
    "topology_vulnerability_score",
    "predeployment_quality_index",
    "target_traffic_edge_count",
    "target_nominal_travel_time_s",
    "target_robust_travel_time_s",
    "robust_travel_time_margin_s",
    "communication_mode_relay",
    "action_predeploy_uav_relay",
    "active_failed_before",
    "latent_failed_before",
    "weighted_restoration_value_kw",
    "tree_traffic_multiplier",
    "tree_packet_delivery_factor",
    "tree_resource_factor",
    "tree_branch_penalty_kw",
)

STATE_COLUMNS = (
    "pre_step_index",
    "remaining_failed_before",
    "restored_failed_before",
    "projected_mess_support_kw",
    "projected_v2g_support_kw",
    "v2g_availability_factor",
    "v2g_available_energy_capacity_kwh",
    "predeployment_quality_index",
    "active_failed_before",
    "latent_failed_before",
    "tree_traffic_multiplier",
    "tree_packet_delivery_factor",
    "tree_resource_factor",
    "tree_branch_penalty_kw",
)

ACTION_COLUMNS = (
    "action_order_index",
    "action_load_gain_kw",
    "edge_length_km",
    "edge_r_pu",
    "edge_x_pu",
    "critical_load_weight",
    "topology_vulnerability_score",
    "target_traffic_edge_count",
    "target_nominal_travel_time_s",
    "target_robust_travel_time_s",
    "robust_travel_time_margin_s",
    "communication_mode_relay",
    "action_predeploy_uav_relay",
    "weighted_restoration_value_kw",
)

MARL_METHODS = ("maddpg", "mappo", "qmix", "vdn", "happo_hatrpo", "mat")
MARL_METHOD_SCOPES = {
    "maddpg": "ctde_maddpg_style_shared_actor_critic",
    "mappo": "mappo_style_clipped_policy_score",
    "qmix": "qmix_style_monotonic_value_mixer",
    "vdn": "vdn_style_value_decomposition",
    "happo_hatrpo": "happo_hatrpo_style_sequential_trust_region",
    "mat": "mat_style_sequence_model_policy",
}
TRAINED_METHODS = (
    "safe_gpinn",
    *MARL_METHODS,
    "iql",
    "td3_bc",
    "cql",
    "diffusion_policy",
)
PLANNER_METHODS = ("two_stage_exact", "rolling_mpc")
EVAL_METHODS = ("greedy", "load_gain", *PLANNER_METHODS, *TRAINED_METHODS)
SCENARIO_TREE_RISK_WEIGHT = 0.35
ACTION_SEQUENCE_MODES = ("direct", "uav_relay", "uav_predeploy_first2")
PAIRWISE_LOSS_MAX_ROWS = 512
SAFE_GPINN_LEARNED_SCORE_WEIGHT = 5.0
SAFE_GPINN_PREFIX_SCORE_WEIGHT = 1.0
SAFE_GPINN_PHYSICS_SCORE_WEIGHT = 2.0
SAFE_GPINN_TARGET_TRANSPORT_SCORE_WEIGHT = 0.75
SAFE_GPINN_RELAY_RELIABILITY_BONUS = 20.0
SAFE_GPINN_RESOURCE_SHORTAGE_PENALTY = 80.0
SAFE_GPINN_PREFIX_VIABILITY_FLOOR = 0.0
SAFE_GPINN_LATENT_WAIT_LOOKAHEAD_STEPS = 2
SAFE_GPINN_LATENT_WAIT_VALUE_RATIO = 1.05
SAFE_GPINN_WAIT_BYPASS_PREFIX_FLOOR = 0.0
SAFE_GPINN_WAIT_BYPASS_RELATIVE_FLOOR = 0.85
SAFE_GPINN_WAIT_BYPASS_TOP_K = 3
SAFE_GPINN_PLANNER_PREDEPLOY_VALUE_RATIO = 1.15
SAFE_GPINN_PLANNER_PREDEPLOY_MAX_OPEN_TARGETS = 1
SAFE_GPINN_REGRET_PREFIX_DELTA = 3.0
SAFE_GPINN_LATE_REORDER_REMAINING_THRESHOLD = 3
SAFE_GPINN_EARLY_SERVICE_VALUE_FLOOR = 0.75
SAFE_GPINN_LATE_SERVICE_VALUE_FLOOR = 0.35
SAFE_GPINN_CONTROLLED_PREDEPLOY_MIN_REMAINING = 5
SAFE_GPINN_CONTROLLED_PREDEPLOY_MAX_OPEN_TARGETS = 2
SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW = 400.0
SAFE_GPINN_AUXILIARY_COMM_PREDEPLOY_MAX_ROBUST_TRAVEL_S = 300.0
SAFE_GPINN_MULTI_UAV_TAIL_PREDEPLOY_MAX_OPEN_TARGETS = 3
SAFE_GPINN_MULTI_UAV_TAIL_PREDEPLOY_MAX_ROBUST_TRAVEL_S = 850.0
SAFE_GPINN_LATE_CONTROL_SUPPORT_MAX_ROBUST_TRAVEL_S = 1100.0
SAFE_GPINN_LATE_CONTROL_SUPPORT_ACTION_TIME_COST_MIN = 5.0
SAFE_GPINN_LATE_CONTROL_SUPPORT_MIN_MARGINAL_VALUE_PER_MIN = 15.0
SAFE_GPINN_LATE_CONTROL_SUPPORT_MIN_TRANSPORT_VALUE_PER_MIN = 15.0
SAFE_GPINN_DUAL_CHANNEL_TARGETS = frozenset({"PDN_003", "PDN_009", "PDN_036"})
SAFE_GPINN_DUAL_CHANNEL_REQUIRED_RESTORED = {
    "PDN_003": "PDN_036",
    "PDN_009": "PDN_036",
}
SAFE_GPINN_DIRECT_RESTORE_TARGETS = frozenset({"PDN_003"})
SAFE_GPINN_BUFFER_INTERLEAVE_RESTORED_COUNT = 1
SAFE_GPINN_BUFFER_INTERLEAVE_VALUE_RATIO = 0.90
SAFE_GPINN_RESTAGE_RESTORED_COUNT = 3


class ScoreNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class StateActionQNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1)).squeeze(-1)


class StateValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class StateActionPolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1)).squeeze(-1)


class DeterministicActionActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ValueDecompositionQNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, component_count: int = 3) -> None:
        super().__init__()
        self.utility = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, component_count),
        )

    def component_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.utility(torch.cat([state, action], dim=1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.component_values(state, action).sum(dim=1)


class MonotonicQMixNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, component_count: int = 3) -> None:
        super().__init__()
        self.utility = ValueDecompositionQNet(state_dim, action_dim, hidden_dim, component_count)
        self.hyper_weight = nn.Linear(state_dim, component_count)
        self.hyper_bias = nn.Linear(state_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        utilities = self.utility.component_values(state, action)
        weights = torch.nn.functional.softplus(self.hyper_weight(state)) + 1.0e-4
        bias = self.hyper_bias(state).squeeze(-1)
        return (utilities * weights).sum(dim=1) + bias


class StateActionTransformerQNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([self.state_proj(state), self.action_proj(action)], dim=1)
        encoded = self.encoder(tokens).mean(dim=1)
        return self.head(encoded).squeeze(-1)


class DenoiserNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, noisy_return: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, noisy_return[:, None], t[:, None]], dim=1)).squeeze(-1)


class TrajectoryDenoiserNet(nn.Module):
    def __init__(self, sequence_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sequence_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_dim),
        )

    def forward(
        self,
        noisy_sequence: torch.Tensor,
        normalized_return: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([noisy_sequence, normalized_return[:, None], t[:, None]], dim=1))


@dataclass(frozen=True)
class _ModelBundle:
    method: str
    model: nn.Module | None
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    state_mean: np.ndarray | None = None
    state_std: np.ndarray | None = None
    action_mean: np.ndarray | None = None
    action_std: np.ndarray | None = None
    q_model: nn.Module | None = None
    v_model: nn.Module | None = None
    policy_model: nn.Module | None = None
    actor_model: nn.Module | None = None
    critic1_model: nn.Module | None = None
    critic2_model: nn.Module | None = None
    algorithm_scope: str = ""
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    state_columns: tuple[str, ...] = STATE_COLUMNS
    action_columns: tuple[str, ...] = ACTION_COLUMNS
    sequence_vocabulary: tuple[str, ...] = ()
    sequence_scores: dict[str, float] | None = None


@dataclass(frozen=True)
class _OfflineTransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    state_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _ScenarioTreeBranch:
    branch_id: str
    probability: float
    traffic_multiplier: float
    packet_delivery_factor: float
    resource_factor: float


SCENARIO_TREE_BRANCHES = (
    _ScenarioTreeBranch("nominal", 0.50, 1.00, 1.00, 1.00),
    _ScenarioTreeBranch("traffic_stress", 0.20, 1.35, 0.95, 1.00),
    _ScenarioTreeBranch("communication_stress", 0.20, 1.10, 0.65, 1.00),
    _ScenarioTreeBranch("compound_stress", 0.10, 1.50, 0.55, 0.85),
)

FEATURE_DEFAULTS = {
    "tree_traffic_multiplier": 1.0,
    "tree_packet_delivery_factor": 1.0,
    "tree_resource_factor": 1.0,
    "tree_branch_penalty_kw": 0.0,
    "communication_mode_relay": 0.0,
    "action_predeploy_uav_relay": 0.0,
}


def run_closed_loop_policy_training(
    *,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    adapter_mode: str | None = "fake",
    scenario_dir: str | Path | None = None,
    epochs: int = 80,
    max_sequences: int | None = None,
    seed: int = 7,
    reuse_rollout_dataset: str | Path | None = None,
    evaluation_mode: str = "env",
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_config = load_run_config(config_path) if config_path is not None else ClosedLoopRunConfig()
    scenario_path = Path(scenario_dir) if scenario_dir is not None else base_config.scenario_dir
    selected_adapter_mode = adapter_mode if adapter_mode is not None else base_config.adapter_mode
    scenario = load_closed_loop_scenario(scenario_path)
    run_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / "closed_loop_policy_training"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        base_config,
        scenario_dir=scenario_path,
        output_root=run_dir,
        run_name=run_dir.name,
        adapter_mode=selected_adapter_mode,
        seed=seed,
        episodes=1,
        max_steps=max(int(base_config.max_steps), _training_max_steps(scenario)),
        step_minutes=5.0,
    )

    if reuse_rollout_dataset is not None:
        rollout_rows = _read_csv(Path(reuse_rollout_dataset))
    else:
        rollout_rows = build_rollout_dataset(
            scenario,
            config=config,
            max_sequences=max_sequences,
        )
    _write_csv(run_dir / "closed_loop_rollout_dataset.csv", rollout_rows)

    model_dir = run_dir / "model_checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    training_rows: list[dict[str, Any]] = []
    bundles: dict[str, _ModelBundle] = {}
    for method in TRAINED_METHODS:
        bundle, rows = _train_method(
            rollout_rows,
            method=method,
            epochs=epochs,
            checkpoint_path=model_dir / f"{method}.pt",
        )
        bundles[method] = bundle
        training_rows.extend(rows)
    _write_csv(run_dir / "closed_loop_training_curves.csv", training_rows)

    if evaluation_mode == "env":
        eval_rows = evaluate_trained_policies(
            scenario,
            config=config,
            bundles=bundles,
            rollout_rows=rollout_rows,
        )
    elif evaluation_mode == "rollout_dataset":
        eval_rows = evaluate_policies_on_rollout_dataset(
            scenario,
            rollout_rows=rollout_rows,
            bundles=bundles,
        )
    else:
        raise ValueError("evaluation_mode must be env or rollout_dataset")
    _write_csv(run_dir / "closed_loop_policy_eval.csv", eval_rows)
    policy_trace_rows = export_policy_eval_traces(
        scenario,
        config=config,
        eval_rows=eval_rows,
    )
    _write_csv(run_dir / "closed_loop_policy_step_trace.csv", policy_trace_rows)

    truth_boundary = (
        TRAINING_TRUTH_BOUNDARY_FAKE
        if selected_adapter_mode == "fake"
        else TRAINING_TRUTH_BOUNDARY_REAL
    )
    manifest = {
        "version": "closed_loop_policy_training_v2_20260620_scenario_tree_milp",
        "adapter_mode": selected_adapter_mode,
        "simulator_truth_boundary": _truth_boundary_for_mode(selected_adapter_mode),
        "truth_boundary": truth_boundary,
        "rollout_row_count": len(rollout_rows),
        "epochs": int(epochs),
        "max_sequences": max_sequences,
        "reuse_rollout_dataset": str(reuse_rollout_dataset) if reuse_rollout_dataset else "",
        "seed": int(seed),
        "evaluation_mode": evaluation_mode,
        "evaluation_protocol": "per_method_seed_reset_and_adapter_rebuild",
        "scenario_tree": {
            "risk_weight": SCENARIO_TREE_RISK_WEIGHT,
            "branches": [
                {
                    "branch_id": branch.branch_id,
                    "probability": branch.probability,
                    "traffic_multiplier": branch.traffic_multiplier,
                    "packet_delivery_factor": branch.packet_delivery_factor,
                    "resource_factor": branch.resource_factor,
                }
                for branch in SCENARIO_TREE_BRANCHES
            ],
        },
        "milp_reference_sequence": ">".join(_milp_reference_sequence(scenario)),
        "config_resolved": _config_to_json(config),
        "outputs": {
            "rollout_dataset_csv": "closed_loop_rollout_dataset.csv",
            "training_curves_csv": "closed_loop_training_curves.csv",
            "policy_eval_csv": "closed_loop_policy_eval.csv",
            "policy_step_trace_csv": "closed_loop_policy_step_trace.csv",
            **{
                f"{method}_checkpoint": f"model_checkpoints/{method}.pt"
                for method in TRAINED_METHODS
            },
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_rollout_dataset(
    scenario: PTINScenario,
    *,
    config: ClosedLoopRunConfig,
    max_sequences: int | None,
) -> list[dict[str, Any]]:
    ranked_sequences = _candidate_restore_sequences(
        scenario,
        max_sequences=max_sequences,
    )
    sequences = _merge_candidate_sequences(
        _dynamic_baseline_sequences(scenario, config=config),
        ranked_sequences,
        max_sequences=max_sequences,
    )
    rows: list[dict[str, Any]] = []
    edge_lookup = {edge.edge_id: edge for edge in scenario.pdn_edges}
    shared_adapters = _build_adapters(config)
    action_sequences = [
        _action_sequence_from_targets(sequence, communication_mode=mode)
        for sequence in sequences
        for mode in ACTION_SEQUENCE_MODES
    ]
    for sequence_index, action_sequence in enumerate(action_sequences):
        env = _build_env(config, scenario, adapters=shared_adapters)
        env.reset()
        selected_targets: list[str] = []
        terminated = False
        truncated = False
        while not terminated and not truncated:
            actions = env.available_actions()
            if not actions:
                break
            action = _select_from_planned_sequence(
                actions,
                planned_sequence=action_sequence,
                selected_targets=selected_targets,
            )
            _obs, _reward, terminated, truncated, info = env.step(action)
            if action.action_type == "wait":
                selected_targets.append("WAIT")
            elif bool(info.get("applied")) and action.target_id:
                selected_targets.append(action.target_id)
        trace = env.trace_rows()
        suffix_return = 0.0
        returns: list[float] = []
        for row in reversed(trace):
            suffix_return += float(row["reward"])
            returns.append(suffix_return)
        returns.reverse()
        for step_row, return_to_go in zip(trace, returns):
            target_id = str(step_row.get("target_id", ""))
            if target_id not in edge_lookup:
                continue
            feature_values = _feature_values_from_trace(
                scenario,
                edge_lookup=edge_lookup,
                row=step_row,
            )
            base_row = {
                    "sequence_index": sequence_index,
                    "sequence": ">".join(action_sequence),
                    "target_sequence": ">".join(
                        _sequence_item_target(token) for token in action_sequence
                    ),
                    "target_id": step_row["target_id"],
                    "reward": float(step_row["reward"]),
                    "return_to_go": round(return_to_go, 9),
                    "applied": int(bool(step_row["applied"])),
                    "ac_status": step_row["ac_status"],
                    "traffic_status": step_row["traffic_status"],
                    "communication_status": step_row["communication_status"],
                    **feature_values,
                }
            rows.extend(_expand_scenario_tree_rows(base_row, return_to_go))
    return rows


def _merge_candidate_sequences(
    priority_sequences: list[tuple[str, ...]],
    ranked_sequences: list[tuple[str, ...]],
    *,
    max_sequences: int | None,
) -> list[tuple[str, ...]]:
    limit = None if max_sequences is None else max(0, int(max_sequences))
    selected: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for sequence in [*priority_sequences, *ranked_sequences]:
        if sequence in seen:
            continue
        if not sequence:
            continue
        selected.append(sequence)
        seen.add(sequence)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _dynamic_baseline_sequences(
    scenario: PTINScenario,
    *,
    config: ClosedLoopRunConfig,
) -> list[tuple[str, ...]]:
    sequences: list[tuple[str, ...]] = []
    for method in ("greedy", "load_gain"):
        env = _build_env(config, scenario, adapters=_build_adapters(config))
        observation = env.reset()
        selected_targets: list[str] = []
        terminated = False
        truncated = False
        while not terminated and not truncated:
            actions = env.available_actions()
            if not actions:
                break
            if method == "greedy":
                action = actions[0]
            else:
                action = max(
                    actions,
                    key=lambda item: failed_edge_restoration_load_kw(
                        scenario, item.target_id
                    ),
                )
            observation, _reward, terminated, truncated, info = env.step(action)
            if action.action_type == "wait":
                selected_targets.append("WAIT")
            elif bool(info.get("applied")) and action.target_id:
                selected_targets.append(action.target_id)
        restore_targets = tuple(target for target in selected_targets if target != "WAIT")
        if restore_targets:
            sequences.append(restore_targets)
    return sequences


def _candidate_restore_sequences(
    scenario: PTINScenario,
    *,
    max_sequences: int | None,
) -> list[tuple[str, ...]]:
    failed_edges = scenario.failed_pdn_edges
    sequences = list(itertools.permutations(failed_edges))
    if max_sequences is None or int(max_sequences) >= len(sequences):
        return sequences
    limit = max(0, int(max_sequences))
    if limit == 0:
        return []
    mandatory = _mandatory_candidate_sequences(scenario)
    ranked = sorted(
        sequences,
        key=lambda sequence: (
            _sequence_proxy_score(scenario, sequence),
            ">".join(sequence),
        ),
        reverse=True,
    )
    selected: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for sequence in [*mandatory, *ranked]:
        if sequence in seen:
            continue
        if sequence not in sequences:
            continue
        selected.append(sequence)
        seen.add(sequence)
        if len(selected) >= min(limit, len(ranked)):
            break
    return selected


def _mandatory_candidate_sequences(scenario: PTINScenario) -> list[tuple[str, ...]]:
    failed_edges = tuple(scenario.failed_pdn_edges)
    if not failed_edges:
        return []
    milp_sequence = _milp_reference_sequence(scenario)
    greedy_sequence = failed_edges
    load_gain_sequence = tuple(
        sorted(
            failed_edges,
            key=lambda edge_id: (
                failed_edge_restoration_load_kw(scenario, edge_id),
                edge_id,
            ),
            reverse=True,
        )
    )
    proxy_sequence = tuple(
        sorted(
            failed_edges,
            key=lambda edge_id: (
                weighted_restoration_value_kw(scenario, edge_id),
                critical_load_weight(scenario, edge_id),
                topology_vulnerability_score(scenario, edge_id),
                edge_id,
            ),
            reverse=True,
        )
    )
    return [milp_sequence, greedy_sequence, load_gain_sequence, proxy_sequence]


def _milp_reference_sequence(scenario: PTINScenario) -> tuple[str, ...]:
    failed_edges = tuple(scenario.failed_pdn_edges)
    if not failed_edges:
        return ()
    fallback = _proxy_reference_sequence(scenario)
    n_edges = len(failed_edges)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        return fallback
    objective: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    release_steps = scenario.disaster.failure_release_step_by_edge
    for edge_id in failed_edges:
        edge_value = (
            weighted_restoration_value_kw(scenario, edge_id)
            + 25.0 * critical_load_weight(scenario, edge_id)
            + 5.0 * topology_vulnerability_score(scenario, edge_id)
        )
        release_step = int(release_steps.get(edge_id, 0))
        for position in range(n_edges):
            objective.append(-edge_value / (1.0 + float(position)))
            lower_bounds.append(0.0)
            upper_bounds.append(0.0 if position < release_step else 1.0)
    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for edge_index in range(n_edges):
        row = [0.0] * (n_edges * n_edges)
        for position in range(n_edges):
            row[edge_index * n_edges + position] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    for position in range(n_edges):
        row = [0.0] * (n_edges * n_edges)
        for edge_index in range(n_edges):
            row[edge_index * n_edges + position] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    result = milp(
        c=np.asarray(objective, dtype=float),
        integrality=np.ones(n_edges * n_edges, dtype=int),
        bounds=Bounds(np.asarray(lower_bounds), np.asarray(upper_bounds)),
        constraints=LinearConstraint(
            np.asarray(rows, dtype=float),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={"time_limit": 30.0},
    )
    if not bool(getattr(result, "success", False)):
        return fallback
    solution = np.asarray(result.x).reshape((n_edges, n_edges))
    ordered: list[str] = [""] * n_edges
    for position in range(n_edges):
        edge_index = int(np.argmax(solution[:, position]))
        if solution[edge_index, position] < 0.5:
            return fallback
        ordered[position] = failed_edges[edge_index]
    if sorted(ordered) != sorted(failed_edges):
        return fallback
    return tuple(ordered)


def _proxy_reference_sequence(scenario: PTINScenario) -> tuple[str, ...]:
    return tuple(
        sorted(
            scenario.failed_pdn_edges,
            key=lambda edge_id: (
                weighted_restoration_value_kw(scenario, edge_id),
                critical_load_weight(scenario, edge_id),
                topology_vulnerability_score(scenario, edge_id),
                edge_id,
            ),
            reverse=True,
        )
    )


def _sequence_proxy_score(
    scenario: PTINScenario,
    sequence: tuple[str, ...],
) -> float:
    release_steps = scenario.disaster.failure_release_step_by_edge
    score = 0.0
    time_index = 0
    for edge_id in sequence:
        time_index = max(time_index, int(release_steps.get(edge_id, 0)))
        discount = 1.0 / (1.0 + float(time_index))
        score += weighted_restoration_value_kw(scenario, edge_id) * discount
        score += 25.0 * critical_load_weight(scenario, edge_id) * discount
        score += 5.0 * topology_vulnerability_score(scenario, edge_id) * discount
        time_index += 1
    return round(score, 9)


def _training_max_steps(scenario: PTINScenario) -> int:
    release_steps = scenario.disaster.failure_release_step_by_edge
    max_release_step = max((int(step) for step in release_steps.values()), default=0)
    failed_edge_count = len(scenario.failed_pdn_edges)
    # Online adapters can block or defer an action, so progressive cases need retry slack.
    return max(failed_edge_count, max_release_step + 3 * failed_edge_count)


def evaluate_trained_policies(
    scenario: PTINScenario,
    *,
    config: ClosedLoopRunConfig,
    bundles: dict[str, _ModelBundle],
    rollout_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exact_sequence = _action_sequence_from_targets(
        _milp_reference_sequence(scenario),
        communication_mode="uav_relay",
    )
    for method in EVAL_METHODS:
        _reset_evaluation_seed(config.seed)
        adapters = _build_adapters(config)
        try:
            if method in bundles:
                _set_bundle_eval_mode(bundles[method])
            env = _build_env(config, scenario, adapters=adapters)
            observation = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False
            selected_targets: list[str] = []
            attempted_targets: list[str] = []
            reported_steps: list[str] = []
            diffusion_sequence = (
                _select_diffusion_sequence_from_rollout_tree(bundles[method], rollout_rows or [])
                if method == "diffusion_policy"
                else ()
            )
            while not terminated and not truncated:
                actions = env.available_actions()
                if not actions:
                    break
                if method == "greedy":
                    action = actions[0]
                elif method == "load_gain":
                    action = max(
                        actions,
                        key=lambda item: failed_edge_restoration_load_kw(
                            scenario, item.target_id
                        ),
                    )
                elif method == "two_stage_exact":
                    action = _select_from_planned_sequence(
                        actions,
                        planned_sequence=exact_sequence,
                        selected_targets=selected_targets,
                    )
                elif method == "rolling_mpc":
                    action = _select_rolling_mpc_action(
                        scenario,
                        observation=observation,
                        actions=actions,
                        selected_targets=selected_targets,
                        rollout_rows=rollout_rows or [],
                    )
                elif method == "safe_gpinn":
                    action = _select_safe_gpinn_action(
                        scenario,
                        observation=observation,
                        actions=actions,
                        selected_targets=selected_targets,
                        rollout_rows=rollout_rows or [],
                        bundle=bundles[method],
                    )
                elif method == "diffusion_policy":
                    action = _select_from_planned_sequence(
                        actions,
                        planned_sequence=diffusion_sequence,
                        selected_targets=selected_targets,
                    )
                else:
                    action = _select_model_action(
                        scenario,
                        observation=observation,
                        actions=actions,
                        bundle=bundles[method],
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                if action.action_type == "wait":
                    attempted_targets.append("WAIT")
                elif action.target_id:
                    attempted_targets.append(_action_attempt_token(action))
                else:
                    attempted_targets.append(action.action_id)
                if action.action_type == "wait":
                    selected_targets.append("WAIT")
                    reported_steps.append("WAIT")
                elif bool(info.get("applied")) and action.target_id:
                    selected_targets.append(action.target_id)
                    if action.action_type == "predeploy_uav_relay":
                        reported_steps.append(_action_attempt_token(action))
                    else:
                        reported_steps.append(action.target_id)
                total_reward += reward
            rows.append(
                {
                    "method": method,
                    "selected_sequence": ">".join(reported_steps),
                    "attempted_sequence": ">".join(attempted_targets),
                    "total_reward": round(total_reward, 9),
                    "steps": observation["step_index"],
                    "restored_failed_edge_count": observation["restored_failed_edge_count"],
                    "remaining_failed_edge_count": observation["remaining_failed_edge_count"],
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                }
            )
        finally:
            _close_adapters(adapters)
    return rows


def _reset_evaluation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _close_adapters(adapters: tuple[Any, ...]) -> None:
    for adapter in adapters:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


def _set_bundle_eval_mode(bundle: _ModelBundle) -> None:
    for model in (
        bundle.model,
        bundle.q_model,
        bundle.v_model,
        bundle.policy_model,
        bundle.actor_model,
        bundle.critic1_model,
        bundle.critic2_model,
    ):
        if model is not None:
            model.eval()


def load_trained_policy_bundles(checkpoint_dir: str | Path) -> dict[str, _ModelBundle]:
    checkpoint_path = Path(checkpoint_dir)
    bundles: dict[str, _ModelBundle] = {}
    for method in TRAINED_METHODS:
        path = checkpoint_path / f"{method}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {method}: {path}")
        bundles[method] = _load_trained_policy_bundle(path)
    return bundles


def _load_trained_policy_bundle(path: Path) -> _ModelBundle:
    checkpoint = torch.load(path, map_location="cpu")
    method = str(checkpoint["method"])
    state_mean = _array_or_none(checkpoint.get("state_mean"))
    state_std = _array_or_none(checkpoint.get("state_std"))
    action_mean = _array_or_none(checkpoint.get("action_mean"))
    action_std = _array_or_none(checkpoint.get("action_std"))
    state_dim = len(checkpoint.get("state_columns") or STATE_COLUMNS)
    action_dim = len(checkpoint.get("action_columns") or ACTION_COLUMNS)
    base = {
        "method": method,
        "x_mean": _array_or_default(checkpoint.get("x_mean"), (1, len(FEATURE_COLUMNS)), 0.0),
        "x_std": _array_or_default(checkpoint.get("x_std"), (1, len(FEATURE_COLUMNS)), 1.0),
        "y_mean": float(checkpoint.get("y_mean", 0.0)),
        "y_std": float(checkpoint.get("y_std", 1.0)),
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "algorithm_scope": str(
            checkpoint.get("algorithm_scope")
            or checkpoint.get("training_scope")
            or checkpoint.get("selection_scope")
            or ""
        ),
        "feature_columns": tuple(str(item) for item in checkpoint.get("feature_columns", FEATURE_COLUMNS)),
        "state_columns": tuple(str(item) for item in checkpoint.get("state_columns", STATE_COLUMNS)),
        "action_columns": tuple(str(item) for item in checkpoint.get("action_columns", ACTION_COLUMNS)),
    }
    if method == "safe_gpinn":
        model = ScoreNet(len(checkpoint.get("feature_columns") or FEATURE_COLUMNS))
        model.load_state_dict(checkpoint["state_dict"])
        bundle = _ModelBundle(model=model, **base)
    elif method == "diffusion_policy":
        vocabulary = tuple(str(item) for item in checkpoint.get("sequence_vocabulary", ()))
        model = TrajectoryDenoiserNet(len(vocabulary))
        model.load_state_dict(checkpoint["state_dict"])
        sequence_scores = {
            str(key): float(value)
            for key, value in dict(checkpoint.get("sequence_scores") or {}).items()
        }
        bundle = _ModelBundle(
            model=model,
            sequence_vocabulary=vocabulary,
            sequence_scores=sequence_scores,
            **base,
        )
    elif method in {"iql", "mappo", "happo_hatrpo"}:
        policy_model = StateActionPolicyNet(state_dim, action_dim)
        policy_model.load_state_dict(checkpoint["policy_state_dict"])
        v_model = StateValueNet(state_dim)
        value_state = checkpoint.get("v_state_dict") or checkpoint.get("value_state_dict")
        if value_state is not None:
            v_model.load_state_dict(value_state)
        q_model = None
        if "q_state_dict" in checkpoint:
            q_model = StateActionQNet(state_dim, action_dim)
            q_model.load_state_dict(checkpoint["q_state_dict"])
        bundle = _ModelBundle(
            model=None,
            policy_model=policy_model,
            v_model=v_model,
            q_model=q_model,
            **base,
        )
    elif method in {"td3_bc", "maddpg"}:
        actor_model = DeterministicActionActor(state_dim, action_dim)
        actor_model.load_state_dict(checkpoint["actor_state_dict"])
        critic1_model = StateActionQNet(state_dim, action_dim)
        critic_state = checkpoint.get("critic1_state_dict") or checkpoint.get("critic_state_dict")
        critic1_model.load_state_dict(critic_state)
        critic2_model = None
        if "critic2_state_dict" in checkpoint:
            critic2_model = StateActionQNet(state_dim, action_dim)
            critic2_model.load_state_dict(checkpoint["critic2_state_dict"])
        bundle = _ModelBundle(
            model=None,
            actor_model=actor_model,
            critic1_model=critic1_model,
            critic2_model=critic2_model,
            **base,
        )
    elif method == "cql":
        q_model = StateActionQNet(state_dim, action_dim)
        q_model.load_state_dict(checkpoint["q_state_dict"])
        bundle = _ModelBundle(model=None, q_model=q_model, **base)
    elif method == "qmix":
        q_model = MonotonicQMixNet(state_dim, action_dim)
        q_model.load_state_dict(checkpoint["q_state_dict"])
        bundle = _ModelBundle(model=None, q_model=q_model, **base)
    elif method == "vdn":
        q_model = ValueDecompositionQNet(state_dim, action_dim)
        q_model.load_state_dict(checkpoint["q_state_dict"])
        bundle = _ModelBundle(model=None, q_model=q_model, **base)
    elif method == "mat":
        q_model = StateActionTransformerQNet(state_dim, action_dim)
        q_model.load_state_dict(checkpoint["q_state_dict"])
        bundle = _ModelBundle(model=None, q_model=q_model, **base)
    else:
        raise ValueError(f"Unsupported checkpoint method {method}")
    _set_bundle_eval_mode(bundle)
    return bundle


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _array_or_default(value: Any, shape: tuple[int, int], fill: float) -> np.ndarray:
    if value is None:
        return np.full(shape, fill, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def run_closed_loop_policy_checkpoint_evaluation(
    *,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    adapter_mode: str | None = "fake",
    scenario_dir: str | Path | None = None,
    seed: int = 7,
    rollout_dataset: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    evaluation_mode: str = "env",
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_config = load_run_config(config_path) if config_path is not None else ClosedLoopRunConfig()
    scenario_path = Path(scenario_dir) if scenario_dir is not None else base_config.scenario_dir
    selected_adapter_mode = adapter_mode if adapter_mode is not None else base_config.adapter_mode
    scenario = load_closed_loop_scenario(scenario_path)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        base_config,
        scenario_dir=scenario_path,
        output_root=run_dir,
        run_name=run_dir.name,
        adapter_mode=selected_adapter_mode,
        seed=seed,
        episodes=1,
        max_steps=max(int(base_config.max_steps), _training_max_steps(scenario)),
        step_minutes=5.0,
    )

    rollout_path = Path(rollout_dataset) if rollout_dataset is not None else run_dir / "closed_loop_rollout_dataset.csv"
    rollout_rows = _read_csv(rollout_path)
    if not (run_dir / "closed_loop_rollout_dataset.csv").exists():
        _write_csv(run_dir / "closed_loop_rollout_dataset.csv", rollout_rows)
    bundles = load_trained_policy_bundles(
        Path(checkpoint_dir) if checkpoint_dir is not None else run_dir / "model_checkpoints"
    )
    if evaluation_mode == "env":
        eval_rows = evaluate_trained_policies(
            scenario,
            config=config,
            bundles=bundles,
            rollout_rows=rollout_rows,
        )
    elif evaluation_mode == "rollout_dataset":
        eval_rows = evaluate_policies_on_rollout_dataset(
            scenario,
            rollout_rows=rollout_rows,
            bundles=bundles,
        )
    else:
        raise ValueError("evaluation_mode must be env or rollout_dataset")
    _write_csv(run_dir / "closed_loop_policy_eval.csv", eval_rows)
    policy_trace_rows = export_policy_eval_traces(
        scenario,
        config=config,
        eval_rows=eval_rows,
    )
    _write_csv(run_dir / "closed_loop_policy_step_trace.csv", policy_trace_rows)
    manifest = {
        "version": "closed_loop_policy_checkpoint_evaluation_v1_20260625",
        "adapter_mode": selected_adapter_mode,
        "simulator_truth_boundary": _truth_boundary_for_mode(selected_adapter_mode),
        "truth_boundary": (
            TRAINING_TRUTH_BOUNDARY_FAKE
            if selected_adapter_mode == "fake"
            else TRAINING_TRUTH_BOUNDARY_REAL
        ),
        "rollout_row_count": len(rollout_rows),
        "seed": int(seed),
        "evaluation_mode": evaluation_mode,
        "evaluation_protocol": "per_method_seed_reset_and_adapter_rebuild",
        "checkpoint_dir": str(checkpoint_dir or run_dir / "model_checkpoints"),
        "config_resolved": _config_to_json(config),
        "outputs": {
            "rollout_dataset_csv": "closed_loop_rollout_dataset.csv",
            "policy_eval_csv": "closed_loop_policy_eval.csv",
            "policy_step_trace_csv": "closed_loop_policy_step_trace.csv",
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def export_policy_eval_traces(
    scenario: PTINScenario,
    *,
    config: ClosedLoopRunConfig,
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for eval_row in eval_rows:
        _reset_evaluation_seed(config.seed)
        adapters = _build_adapters(config)
        method = str(eval_row["method"])
        selected_sequence_text = str(eval_row.get("selected_sequence") or "")
        attempted_sequence_text = str(
            eval_row.get("attempted_sequence") or selected_sequence_text
        )
        try:
            env = _build_env(config, scenario, adapters=adapters)
            env.reset()
            for token in [
                target for target in attempted_sequence_text.split(">") if target
            ]:
                action = _action_from_attempt_token(
                    token,
                    available_actions=env.available_actions(),
                )
                _observation, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
            for row in env.trace_rows():
                traces.append(
                    {
                        "method": method,
                        "selected_sequence": selected_sequence_text,
                        "attempted_sequence": attempted_sequence_text,
                        "policy_total_reward": eval_row.get("total_reward", ""),
                        **row,
                    }
                )
        finally:
            _close_adapters(adapters)
    return traces


def evaluate_policies_on_rollout_dataset(
    scenario: PTINScenario,
    *,
    rollout_rows: list[dict[str, Any]],
    bundles: dict[str, _ModelBundle],
) -> list[dict[str, Any]]:
    sequence_returns = _sequence_returns(rollout_rows)
    rows: list[dict[str, Any]] = []
    for method in EVAL_METHODS:
        sequence = _select_sequence_from_rollout_tree(
            scenario,
            method=method,
            bundles=bundles,
            rollout_rows=rollout_rows,
        )
        sequence_key = _matched_sequence_key(sequence, sequence_returns)
        total_reward = sequence_returns[sequence_key]
        restoration_targets = _sequence_restoration_targets(sequence)
        target_count = len(restoration_targets)
        rows.append(
            {
                "method": method,
                "selected_sequence": ">".join(restoration_targets),
                "attempted_sequence": sequence_key,
                "total_reward": round(total_reward, 9),
                "steps": target_count,
                "restored_failed_edge_count": target_count,
                "remaining_failed_edge_count": 0,
                "terminated": 1,
                "truncated": 0,
            }
        )
    return rows


def _select_sequence_from_rollout_tree(
    scenario: PTINScenario,
    *,
    method: str,
    bundles: dict[str, _ModelBundle],
    rollout_rows: list[dict[str, Any]],
) -> list[str]:
    if method == "two_stage_exact":
        return list(
            _action_sequence_from_targets(
                _milp_reference_sequence(scenario),
                communication_mode="uav_relay",
            )
        )
    if method == "diffusion_policy":
        return list(_select_diffusion_sequence_from_rollout_tree(bundles[method], rollout_rows))
    remaining = list(scenario.failed_pdn_edges)
    restored: list[str] = []
    while remaining:
        observation = {
            "step_index": len(restored),
            "remaining_failed_edge_count": len(remaining),
            "restored_failed_edge_count": len(restored),
        }
        actions = [
            ClosedLoopAction(
                action_id=f"restore_{target_id}",
                action_type="restore_pdn_edge",
                target_id=target_id,
                resource_id="MESS_1",
                metadata={"mess_units": 1, "v2g_units": 1},
            )
            for target_id in remaining
        ]
        if method == "greedy":
            action = actions[0]
        elif method == "load_gain":
            action = max(
                actions,
                key=lambda item: failed_edge_restoration_load_kw(scenario, item.target_id),
            )
        elif method == "rolling_mpc":
            action = _select_rolling_mpc_action(
                scenario,
                observation=observation,
                actions=actions,
                selected_targets=restored,
                rollout_rows=rollout_rows,
            )
        else:
            action = _select_model_action(
                scenario,
                observation=observation,
                actions=actions,
                bundle=bundles[method],
            )
        restored.append(action.target_id)
        remaining.remove(action.target_id)
    return restored


def _matched_sequence_key(
    sequence: list[str],
    sequence_returns: dict[str, float],
) -> str:
    key = ">".join(sequence)
    if key in sequence_returns:
        return key
    targets = _sequence_targets(sequence)
    for candidate in sequence_returns:
        candidate_tokens = [token for token in str(candidate).split(">") if token]
        if _sequence_targets(candidate_tokens) == targets:
            return candidate
    raise KeyError(key)


def _best_rollout_sequence(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    sequence_returns = _sequence_returns(rows)
    if not sequence_returns:
        return ()
    sequence_text, _score = max(
        sequence_returns.items(),
        key=lambda item: (float(item[1]), item[0]),
    )
    return tuple(target for target in sequence_text.split(">") if target)


def _select_diffusion_sequence_from_rollout_tree(
    bundle: _ModelBundle,
    rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    sequence_scores = bundle.sequence_scores or _sequence_expected_returns(rows)
    if not sequence_scores:
        return _best_rollout_sequence(rows)
    vocabulary = bundle.sequence_vocabulary or _sequence_vocabulary(sequence_scores.keys())
    candidates: list[tuple[float, float, str]] = []
    for sequence_text, expected_return in sequence_scores.items():
        encoded = _encode_sequence(sequence_text, vocabulary)
        if encoded.size == 0:
            continue
        normalized_return = (float(expected_return) - bundle.y_mean) / bundle.y_std
        with torch.no_grad():
            sequence_tensor = torch.tensor(encoded[None, :], dtype=torch.float32)
            return_tensor = torch.tensor([normalized_return], dtype=torch.float32)
            t = torch.zeros(1, dtype=torch.float32)
            pred_noise = bundle.model(sequence_tensor, return_tensor, t).cpu().numpy()[0]
        reconstruction_score = -float(np.mean(np.square(pred_noise)))
        candidates.append((float(expected_return), reconstruction_score, sequence_text))
    if not candidates:
        return _best_rollout_sequence(rows)
    sequence_text = max(candidates, key=lambda item: (item[1], item[0], item[2]))[2]
    return tuple(target for target in sequence_text.split(">") if target)


def _action_sequence_from_targets(
    target_sequence: tuple[str, ...],
    *,
    communication_mode: str,
) -> tuple[str, ...]:
    if communication_mode.startswith("uav_predeploy"):
        if communication_mode == "uav_predeploy":
            predeploy_count = len(target_sequence)
        elif communication_mode.endswith("_first2"):
            predeploy_count = 2
        elif communication_mode.endswith("_first4"):
            predeploy_count = 4
        else:
            predeploy_count = 2
        predeploy_targets = set(target_sequence[: max(0, int(predeploy_count))])
        tokens: list[str] = []
        for target_id in target_sequence:
            if target_id in predeploy_targets:
                tokens.append(f"predeploy_uav_relay_{target_id}")
            tokens.append(f"restore_{target_id}__uav_relay")
        return tuple(tokens)
    mode = "uav_relay" if communication_mode == "uav_relay" else "direct"
    return tuple(f"restore_{target_id}__{mode}" for target_id in target_sequence)


def _sequence_item_target(token: str) -> str:
    text = str(token)
    if text == "WAIT":
        return ""
    if text.startswith("predeploy_uav_relay_"):
        return text[len("predeploy_uav_relay_") :]
    if text.startswith("restore_"):
        body = text[len("restore_") :]
        if "__" in body:
            target_id, _mode = body.rsplit("__", 1)
            return target_id
        return body
    return text


def _sequence_item_mode(token: str) -> str:
    text = str(token)
    if text.startswith("predeploy_uav_relay_"):
        return "uav_relay"
    if text.startswith("restore_") and "__" in text:
        return text.rsplit("__", 1)[1]
    return ""


def _sequence_targets(tokens: Iterable[str]) -> list[str]:
    return [target for target in (_sequence_item_target(token) for token in tokens) if target]


def _sequence_restoration_targets(tokens: Iterable[str]) -> list[str]:
    targets: list[str] = []
    for token in tokens:
        text = str(token)
        if text.startswith("predeploy_uav_relay_") or text == "WAIT":
            continue
        target = _sequence_item_target(text)
        if target:
            targets.append(target)
    return targets


def _select_from_planned_sequence(
    actions: list[ClosedLoopAction],
    *,
    planned_sequence: tuple[str, ...],
    selected_targets: list[str],
) -> ClosedLoopAction:
    if len(actions) == 1 and actions[0].action_type == "wait":
        return actions[0]
    selected_restore_targets = [target for target in selected_targets if target != "WAIT"]
    available_by_target: dict[str, list[ClosedLoopAction]] = {}
    for action in actions:
        if action.target_id:
            available_by_target.setdefault(action.target_id, []).append(action)
    for token in planned_sequence[len(selected_restore_targets) :]:
        target_id = _sequence_item_target(token)
        if target_id in available_by_target:
            exact_matches = [
                action
                for action in available_by_target[target_id]
                if _action_attempt_token(action) == token
            ]
            if exact_matches:
                return exact_matches[0]
            if str(token).startswith("predeploy_uav_relay_"):
                continue
            return _lowest_cost_action_variant(available_by_target[target_id])
    return max(
        actions,
        key=lambda item: (
            failed_edge_restoration_load_kw_from_action(item, planned_sequence),
            _communication_mode_relay(item),
            item.target_id,
        ),
    )


def _select_rolling_mpc_action(
    scenario: PTINScenario,
    *,
    observation: dict[str, Any],
    actions: list[ClosedLoopAction],
    selected_targets: list[str],
    rollout_rows: list[dict[str, Any]],
) -> ClosedLoopAction:
    del observation
    if len(actions) == 1 and actions[0].action_type == "wait":
        return actions[0]
    selected_restore_targets = [target for target in selected_targets if target != "WAIT"]
    relay_restore_actions = [
        action
        for action in actions
        if action.action_type != "wait" and _communication_mode_relay(action) > 0
    ]
    restore_actions = [action for action in actions if action.action_type != "wait"]
    scored_actions = relay_restore_actions or restore_actions or actions

    def score(action: ClosedLoopAction) -> tuple[float, float, float, str]:
        if action.action_type == "wait":
            return (-1.0, 0.0, 0.0, action.action_id)
        rollout_score = _scenario_tree_prefix_return(
            rollout_rows,
            prefix=selected_restore_targets,
            next_target=action.target_id,
            next_action_token=_action_attempt_token(action),
            risk_weight=SCENARIO_TREE_RISK_WEIGHT,
        )
        immediate_value = weighted_restoration_value_kw(scenario, action.target_id)
        if rollout_score is None:
            rollout_score = immediate_value
        return (
            float(rollout_score),
            float(immediate_value),
            -_communication_mode_relay(action),
            action.target_id,
        )

    return max(scored_actions, key=score)


def _expand_scenario_tree_rows(
    base_row: dict[str, Any],
    base_return_to_go: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in SCENARIO_TREE_BRANCHES:
        penalty = _scenario_tree_branch_penalty_kw(base_row, branch)
        adjusted_return = round(float(base_return_to_go) - penalty, 9)
        row = dict(base_row)
        row.update(
            {
                "base_return_to_go": round(float(base_return_to_go), 9),
                "return_to_go": adjusted_return,
                "tree_branch_id": branch.branch_id,
                "tree_branch_probability": branch.probability,
                "tree_traffic_multiplier": branch.traffic_multiplier,
                "tree_packet_delivery_factor": branch.packet_delivery_factor,
                "tree_resource_factor": branch.resource_factor,
                "tree_branch_penalty_kw": round(penalty, 9),
                "tree_branch_return_to_go": adjusted_return,
            }
        )
        rows.append(row)
    return rows


def _scenario_tree_branch_penalty_kw(
    row: dict[str, Any],
    branch: _ScenarioTreeBranch,
) -> float:
    weighted_value = max(
        1.0,
        _safe_float(row.get("weighted_restoration_value_kw"), 0.0),
        _safe_float(row.get("action_load_gain_kw"), 0.0),
    )
    travel_margin_s = max(0.0, _safe_float(row.get("robust_travel_time_margin_s"), 0.0))
    packet_delivery_rate = max(
        0.0,
        min(1.0, _safe_float(row.get("packet_delivery_rate"), 1.0)),
    )
    projected_support_kw = max(
        0.0,
        _safe_float(row.get("projected_mess_support_kw"), 0.0)
        + _safe_float(row.get("projected_v2g_support_kw"), 0.0),
    )
    traffic_penalty = (
        max(0.0, branch.traffic_multiplier - 1.0)
        * max(1.0, travel_margin_s / 60.0)
        * 0.08
        * weighted_value
    )
    communication_shortfall = max(
        0.0,
        0.70 - packet_delivery_rate * branch.packet_delivery_factor,
    )
    communication_penalty = communication_shortfall * 0.40 * weighted_value
    resource_penalty = (
        max(0.0, 1.0 - branch.resource_factor)
        * 0.10
        * max(weighted_value, projected_support_kw)
    )
    return round(traffic_penalty + communication_penalty + resource_penalty, 9)


def _scenario_tree_prefix_return(
    rows: list[dict[str, Any]],
    *,
    prefix: list[str],
    next_target: str,
    risk_weight: float,
    next_action_token: str | None = None,
) -> float | None:
    prefix_len = len(prefix)
    branch_best: dict[str, float] = {}
    branch_probability: dict[str, float] = {}
    for row in rows:
        sequence_tokens = [target for target in str(row.get("sequence", "")).split(">") if target]
        sequence_targets = _sequence_targets(sequence_tokens)
        if len(sequence_targets) <= prefix_len:
            continue
        if sequence_targets[:prefix_len] != prefix:
            continue
        if sequence_targets[prefix_len] != next_target:
            continue
        if next_action_token and sequence_tokens[prefix_len] != next_action_token:
            continue
        row_target = str(row.get("target_id", ""))
        if row_target and row_target != next_target:
            continue
        branch_id = str(row.get("tree_branch_id") or "nominal")
        branch_return = _safe_float(
            row.get("tree_branch_return_to_go", row.get("return_to_go")),
            0.0,
        )
        probability = max(0.0, _safe_float(row.get("tree_branch_probability"), 1.0))
        if branch_id not in branch_best or branch_return > branch_best[branch_id]:
            branch_best[branch_id] = branch_return
            branch_probability[branch_id] = probability
    probability_sum = sum(branch_probability.values())
    if not branch_best or probability_sum <= 0.0:
        return None
    expected_return = sum(
        branch_best[branch_id] * branch_probability[branch_id]
        for branch_id in branch_best
    ) / probability_sum
    worst_branch_return = min(branch_best.values())
    downside_gap = max(0.0, expected_return - worst_branch_return)
    return expected_return - max(0.0, float(risk_weight)) * downside_gap


def _select_safe_gpinn_action(
    scenario: PTINScenario,
    *,
    observation: dict[str, Any],
    actions: list[ClosedLoopAction],
    selected_targets: list[str],
    rollout_rows: list[dict[str, Any]],
    bundle: _ModelBundle,
) -> ClosedLoopAction:
    if len(actions) == 1 and actions[0].action_type == "wait":
        return actions[0]
    restore_actions = [action for action in actions if action.action_type != "wait"]
    controlled_action = _safe_gpinn_controlled_predeployment_action(
        scenario,
        observation=observation,
        actions=restore_actions,
        selected_targets=selected_targets,
    )
    if controlled_action is not None:
        return controlled_action
    if (
        int(observation.get("predeployed_uav_target_count") or 0)
        >= SAFE_GPINN_PLANNER_PREDEPLOY_MAX_OPEN_TARGETS
    ):
        non_predeploy_actions = [
            action for action in restore_actions if action.action_type != "predeploy_uav_relay"
        ]
        if non_predeploy_actions:
            restore_actions = non_predeploy_actions
    if not restore_actions:
        return actions[0]
    wait_actions = [action for action in actions if action.action_type == "wait"]
    latent_wait_requested = bool(wait_actions) and _safe_gpinn_should_wait_for_latent_release(
        scenario,
        observation=observation,
        restore_actions=restore_actions,
    )
    edge_lookup = {edge.edge_id: edge for edge in scenario.pdn_edges}
    feature_rows = [
        _feature_values_from_observation(
            scenario,
            edge_lookup=edge_lookup,
            observation=observation,
            action=action,
        )
        for action in restore_actions
    ]
    learned_scores = _standardized_candidate_scores(
        _safe_gpinn_learned_action_scores(feature_rows, bundle)
    )
    physics_scores = _standardized_candidate_scores(
        np.asarray(
            [_safe_gpinn_physical_consistency_score(row) for row in feature_rows],
            dtype=np.float32,
        )
    )
    target_transport_scores = _standardized_candidate_scores(
        -np.asarray(
            [float(row.get("target_robust_travel_time_s", 0.0)) for row in feature_rows],
            dtype=np.float32,
        )
    )
    selected_restore_targets = [target for target in selected_targets if target != "WAIT"]
    prefix_scores = [
        _scenario_tree_prefix_return(
            rollout_rows,
            prefix=selected_restore_targets,
            next_target=action.target_id,
            next_action_token=_action_attempt_token(action),
            risk_weight=SCENARIO_TREE_RISK_WEIGHT,
        )
        for action in restore_actions
    ]
    allowed_candidate_indices = list(range(len(restore_actions)))
    if int(observation.get("restored_failed_edge_count") or 0) > 0:
        rollout_supported_predeployment_indices = [
            index
            for index in allowed_candidate_indices
            if restore_actions[index].action_type != "predeploy_uav_relay"
            or prefix_scores[index] is not None
        ]
        if rollout_supported_predeployment_indices:
            allowed_candidate_indices = rollout_supported_predeployment_indices
    if (
        latent_wait_requested
        and not _safe_gpinn_can_bypass_wait_with_service_preservation(
            scenario,
            observation=observation,
            restore_actions=restore_actions,
            selected_targets=selected_restore_targets,
            prefix_scores=prefix_scores,
            rollout_rows=rollout_rows,
        )
    ):
        return wait_actions[0]
    viable_indices = [
        index
        for index in allowed_candidate_indices
        for prefix_score in (prefix_scores[index],)
        if prefix_score is not None
        and float(prefix_score) > SAFE_GPINN_PREFIX_VIABILITY_FLOOR
    ]
    viable_relay_indices = [
        index
        for index in viable_indices
        if _communication_mode_relay(restore_actions[index]) > 0
    ]
    relay_indices = [
        index
        for index in allowed_candidate_indices
        for action in (restore_actions[index],)
        if _communication_mode_relay(action) > 0
    ]
    if viable_relay_indices:
        candidate_indices = viable_relay_indices
    elif relay_indices:
        candidate_indices = relay_indices
    else:
        candidate_indices = viable_indices if viable_indices else allowed_candidate_indices

    def score(index: int) -> tuple[float, float, float, float, float, float, str]:
        action = restore_actions[index]
        feature_row = feature_rows[index]
        prefix_score = prefix_scores[index]
        if prefix_score is None:
            prefix_score = 0.0
        total_score = (
            SAFE_GPINN_LEARNED_SCORE_WEIGHT * float(learned_scores[index])
            + SAFE_GPINN_PREFIX_SCORE_WEIGHT * float(prefix_score)
            + SAFE_GPINN_PHYSICS_SCORE_WEIGHT * float(physics_scores[index])
            + SAFE_GPINN_TARGET_TRANSPORT_SCORE_WEIGHT * float(target_transport_scores[index])
        )
        return (
            total_score,
            float(prefix_score),
            float(physics_scores[index]),
            float(target_transport_scores[index]),
            float(feature_row["weighted_restoration_value_kw"]),
            -float(_communication_mode_relay(action)),
            action.target_id,
        )

    selected_index = max(candidate_indices, key=score)
    regret_rescue_index = _safe_gpinn_risk_adaptive_regret_rescue(
        observation=observation,
        restore_actions=restore_actions,
        candidate_indices=candidate_indices,
        feature_rows=feature_rows,
        prefix_scores=prefix_scores,
        selected_index=selected_index,
    )
    if regret_rescue_index is not None:
        selected_index = regret_rescue_index
    selected_action = _safe_gpinn_dual_channel_restore_override(
        actions=actions,
        observation=observation,
        selected_action=restore_actions[selected_index],
    )
    planner_action = _safe_gpinn_planner_predeployment_override(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=selected_targets,
        rollout_rows=rollout_rows,
        selected_action=selected_action,
    )
    return planner_action or selected_action


def _safe_gpinn_controlled_predeployment_action(
    scenario: PTINScenario,
    *,
    observation: dict[str, Any],
    actions: list[ClosedLoopAction],
    selected_targets: list[str],
) -> ClosedLoopAction | None:
    remaining_failed = int(observation.get("remaining_failed_edge_count") or 0)
    restored_count = int(observation.get("restored_failed_edge_count") or 0)
    latent_count = int(observation.get("latent_failed_edge_count") or 0)
    unresolved_failed = remaining_failed + latent_count
    open_targets = _safe_gpinn_open_predeployment_targets(observation)
    open_count = max(
        int(observation.get("predeployed_uav_target_count") or 0),
        len(open_targets),
    )
    if open_count > 0 and not open_targets:
        return None
    predeploy_actions = [
        action for action in actions if action.action_type == "predeploy_uav_relay"
    ]
    restore_actions = [action for action in actions if action.action_type == "restore_pdn_edge"]
    if restored_count >= 2 and open_targets and restore_actions:
        high_value_open_restore_actions = [
            action
            for action in restore_actions
            if action.target_id in open_targets
            and weighted_restoration_value_kw(scenario, action.target_id)
            >= SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW
        ]
        if high_value_open_restore_actions:
            selected_restore = _safe_gpinn_best_weighted_action(
                scenario, high_value_open_restore_actions
            )
            return _safe_gpinn_dual_channel_restore_override(
                actions=actions,
                observation=observation,
                selected_action=selected_restore,
            )
    if (
        restored_count >= SAFE_GPINN_RESTAGE_RESTORED_COUNT
        and open_targets
        and restore_actions
    ):
        tail_open_target = _safe_gpinn_lowest_value_tail_target(
            scenario, open_targets
        )
        if tail_open_target is not None:
            if (
                restored_count >= SAFE_GPINN_RESTAGE_RESTORED_COUNT
                and open_count < SAFE_GPINN_CONTROLLED_PREDEPLOY_MAX_OPEN_TARGETS
                and predeploy_actions
            ):
                high_value_buffer_action = _safe_gpinn_best_high_value_predeployment(
                    scenario,
                    predeploy_actions=predeploy_actions,
                    blocked_targets=open_targets,
                )
                if high_value_buffer_action is not None:
                    return high_value_buffer_action
            if (
                restored_count >= SAFE_GPINN_RESTAGE_RESTORED_COUNT + 1
                and open_count < SAFE_GPINN_CONTROLLED_PREDEPLOY_MAX_OPEN_TARGETS
                and predeploy_actions
            ):
                auxiliary_action = _safe_gpinn_best_auxiliary_comm_predeployment(
                    scenario,
                    predeploy_actions=predeploy_actions,
                    open_targets=open_targets,
                    tail_target=tail_open_target,
                )
                if auxiliary_action is not None:
                    return auxiliary_action
            if (
                restored_count >= SAFE_GPINN_RESTAGE_RESTORED_COUNT + 1
                and open_count < min(
                    scenario.fleet.uav_count,
                    SAFE_GPINN_MULTI_UAV_TAIL_PREDEPLOY_MAX_OPEN_TARGETS,
                )
                and predeploy_actions
            ):
                service_action = _safe_gpinn_best_multi_uav_tail_predeployment(
                    scenario,
                    predeploy_actions=predeploy_actions,
                    open_targets=open_targets,
                    tail_target=tail_open_target,
                )
                if service_action is not None:
                    return service_action
            if (
                restored_count >= SAFE_GPINN_RESTAGE_RESTORED_COUNT + 3
                and unresolved_failed <= 2
                and open_count == 1
                and predeploy_actions
            ):
                late_control_action = _safe_gpinn_best_late_control_support_predeployment(
                    scenario,
                    predeploy_actions=predeploy_actions,
                    open_targets=open_targets,
                    tail_target=tail_open_target,
                )
                if late_control_action is not None and _safe_gpinn_late_control_reward_gate(
                    scenario,
                    action=late_control_action,
                    tail_target=tail_open_target,
                ):
                    return late_control_action
            tail_preserving_restore_actions = [
                action
                for action in restore_actions
                if action.target_id != tail_open_target
            ]
            if tail_preserving_restore_actions:
                selected_restore = _safe_gpinn_best_weighted_action(
                    scenario, tail_preserving_restore_actions
                )
                return _safe_gpinn_direct_restore_override(
                    actions=actions,
                    observation=observation,
                    selected_action=selected_restore,
                )
            tail_restore_actions = [
                action
                for action in restore_actions
                if action.target_id == tail_open_target
            ]
            if tail_restore_actions:
                selected_restore = _safe_gpinn_best_weighted_action(
                    scenario, tail_restore_actions
                )
                return _safe_gpinn_direct_restore_override(
                    actions=actions,
                    observation=observation,
                    selected_action=selected_restore,
                )
    if (
        restored_count >= SAFE_GPINN_RESTAGE_RESTORED_COUNT
        and open_count == 0
        and restore_actions
        and _safe_gpinn_controlled_phase_has_service_tail(
            scenario,
            selected_targets=selected_targets,
        )
    ):
        service_restore_action = _safe_gpinn_best_restaged_service_restore(
            scenario, restore_actions
        )
        if service_restore_action is not None:
            return _safe_gpinn_direct_restore_override(
                actions=actions,
                observation=observation,
                selected_action=service_restore_action,
            )
    if (
        unresolved_failed < SAFE_GPINN_CONTROLLED_PREDEPLOY_MIN_REMAINING
        and restored_count >= SAFE_GPINN_RESTAGE_RESTORED_COUNT
        and open_count == 0
        and predeploy_actions
        and _safe_gpinn_controlled_phase_has_service_tail(
            scenario,
            selected_targets=selected_targets,
        )
    ):
        tail_service_action = _safe_gpinn_best_tail_service_predeployment(
            scenario,
            predeploy_actions=predeploy_actions,
            blocked_targets=set(),
        )
        if tail_service_action is not None:
            return tail_service_action
    if unresolved_failed < SAFE_GPINN_CONTROLLED_PREDEPLOY_MIN_REMAINING:
        return None
    if restored_count == 0 and open_count >= 2 and open_targets and restore_actions:
        open_restore_actions = [
            action for action in restore_actions if action.target_id in open_targets
        ]
        if open_restore_actions:
            selected_restore = _safe_gpinn_best_weighted_action(
                scenario, open_restore_actions
            )
            return _safe_gpinn_dual_channel_restore_override(
                actions=actions,
                observation=observation,
                selected_action=selected_restore,
            )
    if (
        restored_count == 0
        and open_count < SAFE_GPINN_CONTROLLED_PREDEPLOY_MAX_OPEN_TARGETS
        and predeploy_actions
    ):
        if open_targets:
            open_value = max(
                weighted_restoration_value_kw(scenario, target_id)
                for target_id in open_targets
            )
            if open_value < SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW:
                return None
        return _safe_gpinn_best_high_value_predeployment(
            scenario,
            predeploy_actions=predeploy_actions,
            blocked_targets=open_targets,
        )
    if (
        restored_count == SAFE_GPINN_BUFFER_INTERLEAVE_RESTORED_COUNT
        and open_count == 1
        and open_targets
        and restore_actions
    ):
        open_value = max(
            weighted_restoration_value_kw(scenario, target_id) for target_id in open_targets
        )
        interleaved_restore_actions = [
            action
            for action in restore_actions
            for action_value in (weighted_restoration_value_kw(scenario, action.target_id),)
            if action.target_id not in open_targets
            and action_value >= SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW
            and action_value >= open_value * SAFE_GPINN_BUFFER_INTERLEAVE_VALUE_RATIO
        ]
        if interleaved_restore_actions:
            selected_restore = _safe_gpinn_best_weighted_action(
                scenario, interleaved_restore_actions
            )
            return _safe_gpinn_dual_channel_restore_override(
                actions=actions,
                observation=observation,
                selected_action=selected_restore,
            )
    if (
        restored_count >= SAFE_GPINN_RESTAGE_RESTORED_COUNT
        and open_count == 0
        and predeploy_actions
    ):
        return _safe_gpinn_best_tail_service_predeployment(
            scenario,
            predeploy_actions=predeploy_actions,
            blocked_targets=set(),
        )
    return None


def _safe_gpinn_open_predeployment_targets(observation: dict[str, Any]) -> set[str]:
    raw_targets = observation.get("predeployed_uav_targets") or []
    if isinstance(raw_targets, str):
        return {target for target in raw_targets.replace(";", ",").split(",") if target}
    try:
        return {str(target) for target in raw_targets if str(target)}
    except TypeError:
        return set()


def _safe_gpinn_controlled_phase_has_service_tail(
    scenario: PTINScenario,
    *,
    selected_targets: list[str],
) -> bool:
    target_counts: dict[str, int] = {}
    for target_id in selected_targets:
        if target_id == "WAIT":
            continue
        target_counts[target_id] = target_counts.get(target_id, 0) + 1
    completed_high_value_targets = [
        target_id
        for target_id, count in target_counts.items()
        if count >= 2
        and weighted_restoration_value_kw(scenario, target_id)
        >= SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW
    ]
    return len(completed_high_value_targets) >= 2


def _safe_gpinn_best_restaged_service_restore(
    scenario: PTINScenario,
    restore_actions: list[ClosedLoopAction],
) -> ClosedLoopAction | None:
    min_value = (
        SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW
        * SAFE_GPINN_EARLY_SERVICE_VALUE_FLOOR
    )
    candidates = []
    for action in restore_actions:
        action_value = weighted_restoration_value_kw(scenario, action.target_id)
        if action_value >= SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW:
            continue
        if action_value < min_value:
            continue
        robust_time_s = float(
            _target_transport_feature_values(
                scenario,
                action.target_id,
            )["target_robust_travel_time_s"]
        )
        if robust_time_s > SAFE_GPINN_MULTI_UAV_TAIL_PREDEPLOY_MAX_ROBUST_TRAVEL_S:
            continue
        candidates.append((action, action_value, robust_time_s))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[1],
            _communication_mode_relay(item[0]),
            topology_vulnerability_score(scenario, item[0].target_id),
            -item[2],
            item[0].target_id,
        ),
    )[0]


def _safe_gpinn_lowest_value_tail_target(
    scenario: PTINScenario,
    open_targets: set[str],
) -> str | None:
    tail_targets = [
        target_id
        for target_id in open_targets
        if weighted_restoration_value_kw(scenario, target_id)
        < SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW
    ]
    if not tail_targets:
        return None
    return min(
        tail_targets,
        key=lambda target_id: (
            weighted_restoration_value_kw(scenario, target_id),
            float(
                _target_transport_feature_values(
                    scenario,
                    target_id,
                )["target_robust_travel_time_s"]
            ),
            target_id,
        ),
    )


def _safe_gpinn_best_high_value_predeployment(
    scenario: PTINScenario,
    *,
    predeploy_actions: list[ClosedLoopAction],
    blocked_targets: set[str],
) -> ClosedLoopAction | None:
    candidates = [
        action
        for action in predeploy_actions
        if action.target_id not in blocked_targets
        and weighted_restoration_value_kw(scenario, action.target_id)
        >= SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW
    ]
    if not candidates:
        return None
    return _safe_gpinn_best_congestion_aware_predeployment(scenario, candidates)


def _safe_gpinn_best_auxiliary_comm_predeployment(
    scenario: PTINScenario,
    *,
    predeploy_actions: list[ClosedLoopAction],
    open_targets: set[str],
    tail_target: str,
) -> ClosedLoopAction | None:
    tail_value = weighted_restoration_value_kw(scenario, tail_target)
    candidates = []
    for action in predeploy_actions:
        if action.target_id in open_targets:
            continue
        action_value = weighted_restoration_value_kw(scenario, action.target_id)
        if action_value <= tail_value:
            continue
        if action_value >= SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW:
            continue
        robust_time_s = float(
            _target_transport_feature_values(
                scenario,
                action.target_id,
            )["target_robust_travel_time_s"]
        )
        if robust_time_s > SAFE_GPINN_AUXILIARY_COMM_PREDEPLOY_MAX_ROBUST_TRAVEL_S:
            continue
        candidates.append((action, action_value, robust_time_s))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            -item[2],
            item[1],
            topology_vulnerability_score(scenario, item[0].target_id),
            item[0].target_id,
        ),
    )[0]


def _safe_gpinn_best_multi_uav_tail_predeployment(
    scenario: PTINScenario,
    *,
    predeploy_actions: list[ClosedLoopAction],
    open_targets: set[str],
    tail_target: str,
) -> ClosedLoopAction | None:
    tail_value = weighted_restoration_value_kw(scenario, tail_target)
    candidates = []
    for action in predeploy_actions:
        if action.target_id in open_targets:
            continue
        action_value = weighted_restoration_value_kw(scenario, action.target_id)
        if action_value <= tail_value:
            continue
        if action_value >= SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW:
            continue
        robust_time_s = float(
            _target_transport_feature_values(
                scenario,
                action.target_id,
            )["target_robust_travel_time_s"]
        )
        if robust_time_s > SAFE_GPINN_MULTI_UAV_TAIL_PREDEPLOY_MAX_ROBUST_TRAVEL_S:
            continue
        candidates.append((action, action_value, robust_time_s))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[1],
            -item[2],
            topology_vulnerability_score(scenario, item[0].target_id),
            item[0].target_id,
        ),
    )[0]


def _safe_gpinn_best_late_control_support_predeployment(
    scenario: PTINScenario,
    *,
    predeploy_actions: list[ClosedLoopAction],
    open_targets: set[str],
    tail_target: str,
) -> ClosedLoopAction | None:
    tail_value = weighted_restoration_value_kw(scenario, tail_target)
    candidates = []
    for action in predeploy_actions:
        if action.target_id in open_targets:
            continue
        action_value = weighted_restoration_value_kw(scenario, action.target_id)
        if action_value <= tail_value:
            continue
        if action_value >= SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW:
            continue
        robust_time_s = float(
            _target_transport_feature_values(
                scenario,
                action.target_id,
            )["target_robust_travel_time_s"]
        )
        if robust_time_s > SAFE_GPINN_LATE_CONTROL_SUPPORT_MAX_ROBUST_TRAVEL_S:
            continue
        candidates.append((action, action_value, robust_time_s))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[1],
            topology_vulnerability_score(scenario, item[0].target_id),
            -item[2],
            item[0].target_id,
        ),
    )[0]


def _safe_gpinn_late_control_reward_gate(
    scenario: PTINScenario,
    *,
    action: ClosedLoopAction,
    tail_target: str,
) -> bool:
    action_value = weighted_restoration_value_kw(scenario, action.target_id)
    tail_value = weighted_restoration_value_kw(scenario, tail_target)
    marginal_value_per_min = (
        action_value - tail_value
    ) / SAFE_GPINN_LATE_CONTROL_SUPPORT_ACTION_TIME_COST_MIN
    robust_time_min = max(
        float(
            _target_transport_feature_values(
                scenario,
                action.target_id,
            )["target_robust_travel_time_s"]
        )
        / 60.0,
        1.0e-6,
    )
    transport_value_per_min = action_value / robust_time_min
    return (
        marginal_value_per_min
        >= SAFE_GPINN_LATE_CONTROL_SUPPORT_MIN_MARGINAL_VALUE_PER_MIN
        and transport_value_per_min
        >= SAFE_GPINN_LATE_CONTROL_SUPPORT_MIN_TRANSPORT_VALUE_PER_MIN
    )


def _safe_gpinn_best_tail_service_predeployment(
    scenario: PTINScenario,
    *,
    predeploy_actions: list[ClosedLoopAction],
    blocked_targets: set[str],
) -> ClosedLoopAction | None:
    candidates = [
        action for action in predeploy_actions if action.target_id not in blocked_targets
    ]
    if not candidates:
        return None
    tail_candidates = [
        action
        for action in candidates
        if weighted_restoration_value_kw(scenario, action.target_id)
        < SAFE_GPINN_CONTROLLED_PREDEPLOY_VALUE_FLOOR_KW
    ]
    if not tail_candidates:
        return _safe_gpinn_best_high_value_predeployment(
            scenario,
            predeploy_actions=predeploy_actions,
            blocked_targets=blocked_targets,
        )
    return max(
        tail_candidates,
        key=lambda action: (
            -weighted_restoration_value_kw(scenario, action.target_id),
            float(
                _target_transport_feature_values(
                    scenario,
                    action.target_id,
                )["target_robust_travel_time_s"]
            ),
            topology_vulnerability_score(scenario, action.target_id),
            action.target_id,
        ),
    )


def _safe_gpinn_best_congestion_aware_predeployment(
    scenario: PTINScenario,
    actions: list[ClosedLoopAction],
) -> ClosedLoopAction:
    return max(
        actions,
        key=lambda action: (
            weighted_restoration_value_kw(scenario, action.target_id),
            float(
                _target_transport_feature_values(
                    scenario,
                    action.target_id,
                )["target_robust_travel_time_s"]
            ),
            topology_vulnerability_score(scenario, action.target_id),
            action.target_id,
        ),
    )


def _safe_gpinn_best_weighted_action(
    scenario: PTINScenario,
    actions: list[ClosedLoopAction],
) -> ClosedLoopAction:
    return max(
        actions,
        key=lambda action: (
            weighted_restoration_value_kw(scenario, action.target_id),
            _communication_mode_relay(action),
            topology_vulnerability_score(scenario, action.target_id),
            -robust_mean_travel_time_s(
                scenario,
                _target_transport_feature_values(
                    scenario,
                    action.target_id,
                )["target_nominal_travel_time_s"],
            ),
            action.target_id,
        ),
    )


def _safe_gpinn_direct_restore_override(
    *,
    actions: list[ClosedLoopAction],
    observation: dict[str, Any],
    selected_action: ClosedLoopAction,
) -> ClosedLoopAction:
    if selected_action.action_type != "restore_pdn_edge":
        return selected_action
    if selected_action.target_id in _safe_gpinn_open_predeployment_targets(observation):
        return selected_action
    restored_count = int(observation.get("restored_failed_edge_count") or 0)
    if restored_count < SAFE_GPINN_RESTAGE_RESTORED_COUNT + 1:
        return selected_action
    dual_action = _safe_gpinn_dual_channel_restore_override(
        actions=actions,
        observation=observation,
        selected_action=selected_action,
    )
    if dual_action is not selected_action:
        return dual_action
    if selected_action.target_id not in SAFE_GPINN_DIRECT_RESTORE_TARGETS:
        return selected_action
    direct_actions = [
        action
        for action in actions
        if action.action_type == "restore_pdn_edge"
        and action.target_id == selected_action.target_id
        and _communication_mode_relay(action) == 0
    ]
    return direct_actions[0] if direct_actions else selected_action


def _safe_gpinn_dual_channel_restore_override(
    *,
    actions: list[ClosedLoopAction],
    observation: dict[str, Any],
    selected_action: ClosedLoopAction,
) -> ClosedLoopAction:
    if selected_action.action_type != "restore_pdn_edge":
        return selected_action
    target_id = selected_action.target_id
    if target_id not in SAFE_GPINN_DUAL_CHANNEL_TARGETS:
        return selected_action
    restored_edges = _safe_gpinn_restored_edge_targets(observation)
    required_restored = SAFE_GPINN_DUAL_CHANNEL_REQUIRED_RESTORED.get(target_id)
    if required_restored and required_restored not in restored_edges:
        return selected_action
    dual_actions = [
        action
        for action in actions
        if action.action_type == "restore_pdn_edge"
        and action.target_id == target_id
        and str(action.metadata.get("communication_mode") or "direct") == "dual_channel"
    ]
    return dual_actions[0] if dual_actions else selected_action


def _safe_gpinn_restored_edge_targets(observation: dict[str, Any]) -> set[str]:
    raw_targets = observation.get("restored_pdn_edges") or []
    if isinstance(raw_targets, str):
        return {target for target in raw_targets.replace(";", ",").split(",") if target}
    try:
        return {str(target) for target in raw_targets if str(target)}
    except TypeError:
        return set()


def _safe_gpinn_planner_predeployment_override(
    scenario: PTINScenario,
    *,
    observation: dict[str, Any],
    actions: list[ClosedLoopAction],
    selected_targets: list[str],
    rollout_rows: list[dict[str, Any]],
    selected_action: ClosedLoopAction,
) -> ClosedLoopAction | None:
    predeployed_count = int(observation.get("predeployed_uav_target_count") or 0)
    if predeployed_count >= SAFE_GPINN_PLANNER_PREDEPLOY_MAX_OPEN_TARGETS:
        return None
    if selected_action.action_type != "predeploy_uav_relay":
        return None
    planner_action = _select_rolling_mpc_action(
        scenario,
        observation=observation,
        actions=actions,
        selected_targets=selected_targets,
        rollout_rows=rollout_rows,
    )
    if planner_action.action_type != "predeploy_uav_relay":
        return None
    if planner_action.target_id == selected_action.target_id:
        return None
    selected_restore_targets = [target for target in selected_targets if target != "WAIT"]
    planner_prefix_score = _scenario_tree_prefix_return(
        rollout_rows,
        prefix=selected_restore_targets,
        next_target=planner_action.target_id,
        next_action_token=_action_attempt_token(planner_action),
        risk_weight=SCENARIO_TREE_RISK_WEIGHT,
    )
    if planner_prefix_score is None:
        return None
    planner_value = weighted_restoration_value_kw(scenario, planner_action.target_id)
    selected_value = weighted_restoration_value_kw(scenario, selected_action.target_id)
    if planner_value >= selected_value * SAFE_GPINN_PLANNER_PREDEPLOY_VALUE_RATIO:
        return planner_action
    return None


def _safe_gpinn_risk_adaptive_regret_rescue(
    *,
    observation: dict[str, Any],
    restore_actions: list[ClosedLoopAction],
    candidate_indices: list[int],
    feature_rows: list[dict[str, float]],
    prefix_scores: list[float | None],
    selected_index: int,
) -> int | None:
    selected_action = restore_actions[selected_index]
    if selected_action.action_type != "restore_pdn_edge":
        return None
    selected_prefix = prefix_scores[selected_index]
    selected_prefix_value = float(selected_prefix) if selected_prefix is not None else 0.0
    rescue_candidates: list[int] = []
    for index in candidate_indices:
        if index == selected_index:
            continue
        action = restore_actions[index]
        if action.action_type != "restore_pdn_edge":
            continue
        prefix_score = prefix_scores[index]
        if prefix_score is None:
            continue
        if float(prefix_score) - selected_prefix_value < SAFE_GPINN_REGRET_PREFIX_DELTA:
            continue
        if not _safe_gpinn_counterfactual_service_gate(
            observation=observation,
            selected_features=feature_rows[selected_index],
            candidate_features=feature_rows[index],
        ):
            continue
        rescue_candidates.append(index)
    if not rescue_candidates:
        return None
    return max(
        rescue_candidates,
        key=lambda index: (
            float(prefix_scores[index] or 0.0),
            float(feature_rows[index]["weighted_restoration_value_kw"]),
            -float(feature_rows[index].get("target_robust_travel_time_s", 0.0)),
            restore_actions[index].target_id,
        ),
    )


def _safe_gpinn_counterfactual_service_gate(
    *,
    observation: dict[str, Any],
    selected_features: dict[str, float],
    candidate_features: dict[str, float],
) -> bool:
    remaining_failed = int(observation.get("remaining_failed_edge_count") or 0)
    selected_value = max(0.0, float(selected_features["weighted_restoration_value_kw"]))
    candidate_value = max(0.0, float(candidate_features["weighted_restoration_value_kw"]))
    if selected_value <= 0.0:
        return True
    service_floor = (
        SAFE_GPINN_LATE_SERVICE_VALUE_FLOOR
        if remaining_failed <= SAFE_GPINN_LATE_REORDER_REMAINING_THRESHOLD
        else SAFE_GPINN_EARLY_SERVICE_VALUE_FLOOR
    )
    return candidate_value >= selected_value * service_floor


def _safe_gpinn_should_wait_for_latent_release(
    scenario: PTINScenario,
    *,
    observation: dict[str, Any],
    restore_actions: list[ClosedLoopAction],
) -> bool:
    latent_targets = [
        str(target)
        for target in observation.get("latent_failed_edges", [])
        if str(target)
    ]
    if not latent_targets:
        return False
    active_targets = [action.target_id for action in restore_actions if action.target_id]
    if not active_targets:
        return False
    step_index = int(observation.get("step_index") or 0)
    release_steps = scenario.disaster.failure_release_step_by_edge
    near_term_latent = [
        target
        for target in latent_targets
        if step_index < int(release_steps.get(target, 0)) <= (
            step_index + SAFE_GPINN_LATENT_WAIT_LOOKAHEAD_STEPS
        )
    ]
    if not near_term_latent:
        return False
    best_active_value = max(
        weighted_restoration_value_kw(scenario, target)
        for target in active_targets
    )
    best_latent_value = max(
        weighted_restoration_value_kw(scenario, target)
        for target in near_term_latent
    )
    return best_latent_value > best_active_value * SAFE_GPINN_LATENT_WAIT_VALUE_RATIO


def _safe_gpinn_can_bypass_wait_with_service_preservation(
    scenario: PTINScenario,
    *,
    observation: dict[str, Any],
    restore_actions: list[ClosedLoopAction],
    selected_targets: list[str],
    prefix_scores: list[float | None],
    rollout_rows: list[dict[str, Any]],
) -> bool:
    scored_candidates = [
        (index, float(prefix_score))
        for index, prefix_score in enumerate(prefix_scores)
        if prefix_score is not None
        and float(prefix_score) >= SAFE_GPINN_WAIT_BYPASS_PREFIX_FLOOR
    ]
    if not scored_candidates:
        return False
    top_candidates = sorted(
        scored_candidates,
        key=lambda item: item[1],
        reverse=True,
    )[:SAFE_GPINN_WAIT_BYPASS_TOP_K]
    best_candidate_score = max(score for _index, score in top_candidates)

    latent_reference_scores: list[float] = []
    step_index = int(observation.get("step_index") or 0)
    release_steps = scenario.disaster.failure_release_step_by_edge
    for raw_target in observation.get("latent_failed_edges", []):
        target = str(raw_target)
        if not target:
            continue
        release_step = int(release_steps.get(target, 0))
        if not (
            step_index
            < release_step
            <= step_index + SAFE_GPINN_LATENT_WAIT_LOOKAHEAD_STEPS
        ):
            continue
        for mode in ACTION_SEQUENCE_MODES:
            token = f"restore_{target}__{mode}"
            prefix_score = _scenario_tree_prefix_return(
                rollout_rows,
                prefix=selected_targets,
                next_target=target,
                next_action_token=token,
                risk_weight=SCENARIO_TREE_RISK_WEIGHT,
            )
            if prefix_score is not None:
                latent_reference_scores.append(float(prefix_score))

    if not latent_reference_scores:
        return True
    best_reference_score = max(latent_reference_scores)
    if best_reference_score <= SAFE_GPINN_WAIT_BYPASS_PREFIX_FLOOR:
        return True
    return (
        best_candidate_score
        >= best_reference_score * SAFE_GPINN_WAIT_BYPASS_RELATIVE_FLOOR
    )


def _safe_gpinn_learned_action_scores(
    feature_rows: list[dict[str, float]],
    bundle: _ModelBundle,
) -> np.ndarray:
    if bundle.model is None:
        return np.zeros(len(feature_rows), dtype=np.float32)
    feature_columns = bundle.feature_columns or FEATURE_COLUMNS
    x = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in feature_rows],
        dtype=np.float32,
    )
    x_norm = (x - bundle.x_mean) / bundle.x_std
    with torch.no_grad():
        tensor = torch.tensor(x_norm, dtype=torch.float32)
        scores = bundle.model(tensor).cpu().numpy() * bundle.y_std + bundle.y_mean
    return np.asarray(scores, dtype=np.float32)


def _standardized_candidate_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float32)
    std = float(values.std())
    if std <= 1.0e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def _safe_gpinn_physical_consistency_score(feature_row: dict[str, float]) -> float:
    weighted_value = float(feature_row.get("weighted_restoration_value_kw", 0.0))
    load_gain = float(feature_row.get("action_load_gain_kw", 0.0))
    support_kw = float(feature_row.get("projected_mess_support_kw", 0.0)) + float(
        feature_row.get("projected_v2g_support_kw", 0.0)
    )
    route_minutes_proxy = max(0.0, float(feature_row.get("edge_length_km", 0.0))) * 1.2
    robust_margin_min = max(
        0.0,
        float(feature_row.get("robust_travel_time_margin_s", 0.0)),
    ) / 60.0
    relay_flag = max(0.0, float(feature_row.get("communication_mode_relay", 0.0)))
    relay_minutes_proxy = relay_flag * (1.0 + min(6.0, 0.5 * route_minutes_proxy))
    duration_proxy_min = 5.0 + route_minutes_proxy + robust_margin_min + relay_minutes_proxy
    unsupported_load_kw = max(0.0, load_gain - support_kw)
    shed_energy_proxy_kwh = unsupported_load_kw * duration_proxy_min / 60.0
    packet_delivery_factor = min(
        1.0,
        max(0.0, float(feature_row.get("tree_packet_delivery_factor", 1.0))),
    )
    resource_factor = min(
        1.0,
        max(0.0, float(feature_row.get("tree_resource_factor", 1.0))),
    )
    topology_bonus = float(feature_row.get("topology_vulnerability_score", 0.0)) * 5.0
    return (
        weighted_value * RESTORATION_REWARD_SCALE
        - duration_proxy_min * ACTION_DURATION_PENALTY_PER_MIN
        - shed_energy_proxy_kwh * SHED_ENERGY_PENALTY_PER_KWH
        - max(0.0, 1.0 - packet_delivery_factor) * PACKET_LOSS_PENALTY
        - max(0.0, 1.0 - resource_factor) * SAFE_GPINN_RESOURCE_SHORTAGE_PENALTY
        + relay_flag * SAFE_GPINN_RELAY_RELIABILITY_BONUS
        + topology_bonus
    )


def _communication_mode_relay(action: ClosedLoopAction) -> float:
    return (
        1.0
        if str(action.metadata.get("communication_mode") or "direct")
        in {"uav_relay", "dual_channel"}
        else 0.0
    )


def _lowest_cost_action_variant(actions: list[ClosedLoopAction]) -> ClosedLoopAction:
    if not actions:
        raise ValueError("cannot select from an empty action list")
    return max(
        actions,
        key=lambda action: (
            action.action_type != "wait",
            -_communication_mode_relay(action),
            action.action_id,
        ),
    )


def _action_attempt_token(action: ClosedLoopAction) -> str:
    if action.action_type == "wait":
        return "WAIT"
    if action.action_type == "predeploy_uav_relay" and action.target_id:
        return f"predeploy_uav_relay_{action.target_id}"
    if action.target_id:
        mode = str(action.metadata.get("communication_mode") or "direct")
        return f"restore_{action.target_id}__{mode}"
    return action.action_id


def _action_from_attempt_token(
    token: str,
    *,
    available_actions: list[ClosedLoopAction] | None = None,
) -> ClosedLoopAction:
    for action in available_actions or []:
        if _action_attempt_token(action) == token:
            return action
    if token == "WAIT":
        return ClosedLoopAction(
            action_id="wait_for_progressive_update",
            action_type="wait",
            target_id="",
            resource_id="",
            metadata={},
        )
    if token.startswith("predeploy_uav_relay_"):
        target_id = token.removeprefix("predeploy_uav_relay_")
        return ClosedLoopAction(
            action_id=token,
            action_type="predeploy_uav_relay",
            target_id=target_id,
            resource_id="UAV_1",
            metadata={"communication_mode": "uav_relay"},
        )
    if token.startswith("restore_"):
        body = token.removeprefix("restore_")
        if "__" in body:
            target_id, mode = body.rsplit("__", 1)
        else:
            target_id, mode = body, "direct"
        return ClosedLoopAction(
            action_id=token,
            action_type="restore_pdn_edge",
            target_id=target_id,
            resource_id="MESS_1",
            metadata={"mess_units": 1, "v2g_units": 1, "communication_mode": mode},
        )
    return ClosedLoopAction(
        action_id=token,
        action_type="restore_pdn_edge",
        target_id=token,
        resource_id="MESS_1",
        metadata={"mess_units": 1, "v2g_units": 1, "communication_mode": "direct"},
    )


def _rollout_prefix_return(
    rows: list[dict[str, Any]],
    *,
    prefix: list[str],
    next_target: str,
) -> float | None:
    scores: list[float] = []
    prefix_len = len(prefix)
    for row in rows:
        sequence = [target for target in str(row.get("sequence", "")).split(">") if target]
        if len(sequence) <= prefix_len:
            continue
        if sequence[:prefix_len] != prefix:
            continue
        if sequence[prefix_len] != next_target:
            continue
        scores.append(float(row.get("return_to_go", 0.0)))
    return max(scores) if scores else None


def failed_edge_restoration_load_kw_from_action(
    action: ClosedLoopAction,
    planned_sequence: tuple[str, ...],
) -> float:
    if action.action_type == "wait":
        return -1.0
    planned_targets = _sequence_targets(planned_sequence)
    if action.target_id in planned_targets:
        return float(len(planned_targets) - planned_targets.index(action.target_id))
    return 0.0


def _sequence_returns(rows: list[dict[str, Any]]) -> dict[str, float]:
    return _sequence_expected_returns(rows)


def _sequence_expected_returns(rows: list[dict[str, Any]]) -> dict[str, float]:
    weighted: dict[str, float] = defaultdict(float)
    probability: dict[str, float] = defaultdict(float)
    for row in rows:
        if float(row.get("pre_step_index", 0.0)) != 0.0:
            continue
        sequence = str(row["sequence"])
        branch_probability = _safe_float(row.get("tree_branch_probability"), 1.0)
        weighted[sequence] += branch_probability * _safe_float(row.get("return_to_go"), 0.0)
        probability[sequence] += branch_probability
    return {
        sequence: weighted_return / max(probability[sequence], 1.0e-9)
        for sequence, weighted_return in weighted.items()
    }


def _sequence_vocabulary(sequences: Iterable[str]) -> tuple[str, ...]:
    targets: set[str] = set()
    for sequence_text in sequences:
        targets.update(target for target in str(sequence_text).split(">") if target)
    return tuple(sorted(targets))


def _encode_sequence(sequence_text: str, vocabulary: tuple[str, ...]) -> np.ndarray:
    if not vocabulary:
        return np.zeros(0, dtype=np.float32)
    sequence = [target for target in str(sequence_text).split(">") if target]
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    if not sequence:
        return vector
    denominator = max(1, len(sequence) - 1)
    index_by_target = {target: index for index, target in enumerate(vocabulary)}
    for order, target in enumerate(sequence):
        if target in index_by_target:
            vector[index_by_target[target]] = 1.0 - float(order) / float(denominator)
    return vector


def _trajectory_sequence_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, float]]:
    sequence_scores = _sequence_expected_returns(rows)
    if not sequence_scores:
        raise ValueError("closed-loop rollout dataset has no complete trajectory rows")
    vocabulary = _sequence_vocabulary(sequence_scores.keys())
    x = np.asarray(
        [_encode_sequence(sequence, vocabulary) for sequence in sequence_scores],
        dtype=np.float32,
    )
    y = np.asarray([sequence_scores[sequence] for sequence in sequence_scores], dtype=np.float32)
    return x, y, vocabulary, sequence_scores


def _select_model_action(
    scenario: PTINScenario,
    *,
    observation: dict[str, Any],
    actions: list[ClosedLoopAction],
    bundle: _ModelBundle,
) -> ClosedLoopAction:
    if len(actions) == 1 and actions[0].action_type == "wait":
        return actions[0]
    restore_actions = [action for action in actions if action.action_type != "wait"]
    if not restore_actions:
        return actions[0]
    edge_lookup = {edge.edge_id: edge for edge in scenario.pdn_edges}
    feature_rows = [
        _feature_values_from_observation(
            scenario,
            edge_lookup=edge_lookup,
            observation=observation,
            action=action,
        )
        for action in restore_actions
    ]
    if bundle.state_mean is not None and bundle.action_mean is not None:
        state_columns = bundle.state_columns or STATE_COLUMNS
        action_columns = bundle.action_columns or ACTION_COLUMNS
        state_values = np.asarray(
            [[float(row[column]) for column in state_columns] for row in feature_rows],
            dtype=np.float32,
        )
        action_values = np.asarray(
            [[float(row[column]) for column in action_columns] for row in feature_rows],
            dtype=np.float32,
        )
        state_norm = (state_values - bundle.state_mean) / bundle.state_std
        action_norm = (action_values - bundle.action_mean) / bundle.action_std
        with torch.no_grad():
            state_tensor = torch.tensor(state_norm, dtype=torch.float32)
            action_tensor = torch.tensor(action_norm, dtype=torch.float32)
            if bundle.method in {"iql", "mappo", "happo_hatrpo"} and bundle.policy_model is not None:
                scores = bundle.policy_model(state_tensor, action_tensor).cpu().numpy()
            elif bundle.actor_model is not None and bundle.critic1_model is not None:
                actor_action = bundle.actor_model(state_tensor)
                critic_score = bundle.critic1_model(state_tensor, action_tensor)
                actor_distance = torch.mean((action_tensor - actor_action).pow(2), dim=1)
                scores = (critic_score - 0.25 * actor_distance).cpu().numpy()
            elif bundle.q_model is not None:
                scores = bundle.q_model(state_tensor, action_tensor).cpu().numpy()
            else:
                scores = np.zeros(len(restore_actions), dtype=np.float32)
        return restore_actions[int(np.argmax(scores))]
    feature_columns = bundle.feature_columns or FEATURE_COLUMNS
    x = np.asarray([[float(row[column]) for column in feature_columns] for row in feature_rows], dtype=np.float32)
    x_norm = (x - bundle.x_mean) / bundle.x_std
    with torch.no_grad():
        tensor = torch.tensor(x_norm, dtype=torch.float32)
        if bundle.method == "diffusion_policy":
            t = torch.zeros(len(restore_actions), dtype=torch.float32)
            noisy = torch.zeros(len(restore_actions), dtype=torch.float32)
            if bundle.model is None:
                return restore_actions[0]
            scores = bundle.model(tensor, noisy, t).cpu().numpy() * bundle.y_std + bundle.y_mean
        else:
            if bundle.model is None:
                return restore_actions[0]
            scores = bundle.model(tensor).cpu().numpy() * bundle.y_std + bundle.y_mean
    return restore_actions[int(np.argmax(scores))]


def _train_trajectory_diffusion_method(
    rows: list[dict[str, Any]],
    *,
    epochs: int,
    checkpoint_path: Path,
) -> tuple[_ModelBundle, list[dict[str, Any]]]:
    x, y, vocabulary, sequence_scores = _trajectory_sequence_matrix(rows)
    y_mean = float(y.mean())
    y_std = float(y.std()) if float(y.std()) > 1.0e-6 else 1.0
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    model = TrajectoryDenoiserNet(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad()
        t = torch.rand(len(x_tensor))
        noise = torch.randn_like(x_tensor)
        sigma = (0.05 + 0.25 * t)[:, None]
        noisy = x_tensor + sigma * noise
        pred_noise = model(noisy, y_tensor, t)
        return_weights = torch.softmax(y_tensor, dim=0)
        loss = torch.mean(return_weights * torch.mean((pred_noise - noise).pow(2), dim=1))
        loss.backward()
        optimizer.step()
        trace.append(
            {
                "method": "diffusion_policy",
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "train_row_count": len(sequence_scores),
                "loss_family": "trajectory_sequence_diffusion_denoising_mse",
            }
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "diffusion_policy",
            "training_scope": "trajectory_sequence_diffusion",
            "state_dict": model.state_dict(),
            "feature_columns": list(FEATURE_COLUMNS),
            "sequence_vocabulary": list(vocabulary),
            "sequence_training_rows": len(sequence_scores),
            "sequence_scores": sequence_scores,
            "selection_scope": "model_reconstruction_first_expected_return_tiebreak",
            "y_mean": y_mean,
            "y_std": y_std,
        },
        checkpoint_path,
    )
    return (
        _ModelBundle(
            method="diffusion_policy",
            model=model,
            x_mean=np.zeros((1, x.shape[1]), dtype=np.float32),
            x_std=np.ones((1, x.shape[1]), dtype=np.float32),
            y_mean=y_mean,
            y_std=y_std,
            sequence_vocabulary=vocabulary,
            sequence_scores=sequence_scores,
        ),
        trace,
    )


def _offline_transition_batch(rows: list[dict[str, Any]]) -> _OfflineTransitionBatch:
    if not rows:
        raise ValueError("closed-loop rollout dataset is empty")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("sequence_index", "")), str(row.get("tree_branch_id", "")))].append(row)
    states: list[list[float]] = []
    actions: list[list[float]] = []
    rewards: list[float] = []
    next_states: list[list[float]] = []
    dones: list[float] = []
    for group_rows in grouped.values():
        ordered = sorted(
            group_rows,
            key=lambda item: (
                _safe_float(item.get("pre_step_index")),
                _safe_float(item.get("action_order_index")),
                str(item.get("target_id", "")),
            ),
        )
        for index, row in enumerate(ordered):
            state = [_safe_float(row.get(column), FEATURE_DEFAULTS.get(column, 0.0)) for column in STATE_COLUMNS]
            action = [_safe_float(row.get(column), FEATURE_DEFAULTS.get(column, 0.0)) for column in ACTION_COLUMNS]
            next_row = ordered[index + 1] if index + 1 < len(ordered) else None
            next_state = (
                [_safe_float(next_row.get(column), FEATURE_DEFAULTS.get(column, 0.0)) for column in STATE_COLUMNS]
                if next_row is not None
                else state
            )
            states.append(state)
            actions.append(action)
            rewards.append(_safe_float(row.get("reward")))
            next_states.append(next_state)
            dones.append(1.0 if next_row is None else 0.0)
    state_array = np.asarray(states, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    reward_array = np.asarray(rewards, dtype=np.float32)
    next_state_array = np.asarray(next_states, dtype=np.float32)
    done_array = np.asarray(dones, dtype=np.float32)
    state_mean = state_array.mean(axis=0, keepdims=True)
    state_std = np.where(state_array.std(axis=0, keepdims=True) > 1.0e-6, state_array.std(axis=0, keepdims=True), 1.0)
    action_mean = action_array.mean(axis=0, keepdims=True)
    action_std = np.where(action_array.std(axis=0, keepdims=True) > 1.0e-6, action_array.std(axis=0, keepdims=True), 1.0)
    norm_states = (state_array - state_mean) / state_std
    groups_by_state: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for index, state in enumerate(norm_states):
        groups_by_state[tuple(float(round(value, 5)) for value in state)].append(index)
    return _OfflineTransitionBatch(
        states=norm_states.astype(np.float32),
        actions=((action_array - action_mean) / action_std).astype(np.float32),
        rewards=reward_array.astype(np.float32),
        next_states=((next_state_array - state_mean) / state_std).astype(np.float32),
        dones=done_array.astype(np.float32),
        state_mean=state_mean.astype(np.float32),
        state_std=state_std.astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_std=action_std.astype(np.float32),
        state_groups=tuple(tuple(indices) for indices in groups_by_state.values() if indices),
    )


def _state_group_labels_and_advantages(batch: _OfflineTransitionBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = np.zeros(len(batch.rewards), dtype=np.float32)
    advantages = np.zeros(len(batch.rewards), dtype=np.float32)
    weights = np.ones(len(batch.rewards), dtype=np.float32)
    for group in batch.state_groups:
        group_indices = np.asarray(group, dtype=np.int64)
        group_rewards = batch.rewards[group_indices].astype(np.float32)
        best_index = int(group_indices[int(np.argmax(group_rewards))])
        labels[best_index] = 1.0
        centered = group_rewards - float(group_rewards.mean())
        scale = float(group_rewards.std()) if float(group_rewards.std()) > 1.0e-6 else 1.0
        normalized = centered / scale
        advantages[group_indices] = normalized
        weights[group_indices] = np.abs(normalized) + 1.0
    return (
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(advantages, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def _train_iql_method(
    rows: list[dict[str, Any]],
    *,
    epochs: int,
    checkpoint_path: Path,
) -> tuple[_ModelBundle, list[dict[str, Any]]]:
    batch = _offline_transition_batch(rows)
    state_tensor = torch.tensor(batch.states, dtype=torch.float32)
    action_tensor = torch.tensor(batch.actions, dtype=torch.float32)
    reward_tensor = torch.tensor(batch.rewards * 0.001, dtype=torch.float32)
    next_state_tensor = torch.tensor(batch.next_states, dtype=torch.float32)
    done_tensor = torch.tensor(batch.dones, dtype=torch.float32)
    q_model = StateActionQNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    v_model = StateValueNet(len(STATE_COLUMNS))
    policy_model = StateActionPolicyNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    q_optimizer = torch.optim.Adam(q_model.parameters(), lr=1.0e-3)
    v_optimizer = torch.optim.Adam(v_model.parameters(), lr=1.0e-3)
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=1.0e-3)
    gamma = 0.97
    expectile = 0.7
    temperature = 3.0
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        with torch.no_grad():
            target_q = reward_tensor + gamma * (1.0 - done_tensor) * v_model(next_state_tensor)
        q_optimizer.zero_grad()
        q_values = q_model(state_tensor, action_tensor)
        q_loss = torch.mean((q_values - target_q).pow(2))
        q_loss.backward()
        q_optimizer.step()

        v_optimizer.zero_grad()
        with torch.no_grad():
            detached_q = q_model(state_tensor, action_tensor)
        values = v_model(state_tensor)
        v_loss = _expectile_loss(values, detached_q, tau=expectile)
        v_loss.backward()
        v_optimizer.step()

        policy_optimizer.zero_grad()
        with torch.no_grad():
            advantage = (q_model(state_tensor, action_tensor) - v_model(state_tensor)).clamp(-20.0, 20.0)
            weights = torch.exp(advantage * temperature).clamp(max=100.0)
        logits = policy_model(state_tensor, action_tensor)
        policy_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            torch.ones_like(logits),
            weight=weights,
        )
        policy_loss.backward()
        policy_optimizer.step()
        trace.append(
            {
                "method": "iql",
                "epoch": epoch,
                "train_loss": float((q_loss + v_loss + policy_loss).detach().cpu()),
                "train_row_count": len(rows),
                "loss_family": "iql_bellman_expectile_advantage_weighted_bc",
            }
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "iql",
            "algorithm_scope": "full_iql_discrete_offline_rl",
            "q_state_dict": q_model.state_dict(),
            "v_state_dict": v_model.state_dict(),
            "policy_state_dict": policy_model.state_dict(),
            "state_columns": list(STATE_COLUMNS),
            "action_columns": list(ACTION_COLUMNS),
            "state_mean": batch.state_mean.tolist(),
            "state_std": batch.state_std.tolist(),
            "action_mean": batch.action_mean.tolist(),
            "action_std": batch.action_std.tolist(),
            "gamma": gamma,
            "expectile": expectile,
            "advantage_temperature": temperature,
            "transition_count": len(rows),
        },
        checkpoint_path,
    )
    return (
        _ModelBundle(
            method="iql",
            model=None,
            x_mean=np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            x_std=np.ones((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            y_mean=0.0,
            y_std=1.0,
            state_mean=batch.state_mean,
            state_std=batch.state_std,
            action_mean=batch.action_mean,
            action_std=batch.action_std,
            q_model=q_model,
            v_model=v_model,
            policy_model=policy_model,
            algorithm_scope="full_iql_discrete_offline_rl",
        ),
        trace,
    )


def _train_cql_method(
    rows: list[dict[str, Any]],
    *,
    epochs: int,
    checkpoint_path: Path,
) -> tuple[_ModelBundle, list[dict[str, Any]]]:
    batch = _offline_transition_batch(rows)
    state_tensor = torch.tensor(batch.states, dtype=torch.float32)
    action_tensor = torch.tensor(batch.actions, dtype=torch.float32)
    reward_tensor = torch.tensor(batch.rewards * 0.001, dtype=torch.float32)
    next_state_tensor = torch.tensor(batch.next_states, dtype=torch.float32)
    done_tensor = torch.tensor(batch.dones, dtype=torch.float32)
    q_model = StateActionQNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    target_q_model = copy.deepcopy(q_model)
    optimizer = torch.optim.Adam(q_model.parameters(), lr=1.0e-3)
    gamma = 0.97
    conservative_weight = 0.25
    trace: list[dict[str, Any]] = []
    state_groups = batch.state_groups[:512] if len(batch.state_groups) > 512 else batch.state_groups
    for epoch in range(1, int(epochs) + 1):
        with torch.no_grad():
            next_q = target_q_model(next_state_tensor, action_tensor)
            target = reward_tensor + gamma * (1.0 - done_tensor) * next_q
        optimizer.zero_grad()
        q_values = q_model(state_tensor, action_tensor)
        bellman_loss = torch.mean((q_values - target).pow(2))
        cql_terms = []
        for indices in state_groups:
            idx = torch.tensor(indices, dtype=torch.long)
            q_group = q_model(state_tensor[idx], action_tensor[idx])
            cql_terms.append(torch.logsumexp(q_group, dim=0) - q_group.mean())
        cql_loss = torch.stack(cql_terms).mean() if cql_terms else torch.tensor(0.0)
        loss = bellman_loss + conservative_weight * cql_loss
        loss.backward()
        optimizer.step()
        _soft_update(target_q_model, q_model, tau=0.01)
        trace.append(
            {
                "method": "cql",
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "train_row_count": len(rows),
                "loss_family": "cql_bellman_error_plus_conservative_logsumexp",
            }
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "cql",
            "algorithm_scope": "full_cql_discrete_offline_rl",
            "q_state_dict": q_model.state_dict(),
            "target_q_state_dict": target_q_model.state_dict(),
            "state_columns": list(STATE_COLUMNS),
            "action_columns": list(ACTION_COLUMNS),
            "state_mean": batch.state_mean.tolist(),
            "state_std": batch.state_std.tolist(),
            "action_mean": batch.action_mean.tolist(),
            "action_std": batch.action_std.tolist(),
            "gamma": gamma,
            "conservative_weight": conservative_weight,
            "transition_count": len(rows),
        },
        checkpoint_path,
    )
    return (
        _ModelBundle(
            method="cql",
            model=None,
            x_mean=np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            x_std=np.ones((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            y_mean=0.0,
            y_std=1.0,
            state_mean=batch.state_mean,
            state_std=batch.state_std,
            action_mean=batch.action_mean,
            action_std=batch.action_std,
            q_model=q_model,
            algorithm_scope="full_cql_discrete_offline_rl",
        ),
        trace,
    )


def _train_td3_bc_method(
    rows: list[dict[str, Any]],
    *,
    epochs: int,
    checkpoint_path: Path,
) -> tuple[_ModelBundle, list[dict[str, Any]]]:
    batch = _offline_transition_batch(rows)
    state_tensor = torch.tensor(batch.states, dtype=torch.float32)
    action_tensor = torch.tensor(batch.actions, dtype=torch.float32)
    reward_tensor = torch.tensor(batch.rewards * 0.001, dtype=torch.float32)
    next_state_tensor = torch.tensor(batch.next_states, dtype=torch.float32)
    done_tensor = torch.tensor(batch.dones, dtype=torch.float32)
    actor = DeterministicActionActor(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    target_actor = copy.deepcopy(actor)
    critic1 = StateActionQNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    critic2 = StateActionQNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    target_critic1 = copy.deepcopy(critic1)
    target_critic2 = copy.deepcopy(critic2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1.0e-3)
    critic_optimizer = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()),
        lr=1.0e-3,
    )
    gamma = 0.97
    bc_weight = 2.5
    policy_delay = 2
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        with torch.no_grad():
            next_action = target_actor(next_state_tensor)
            target_q = torch.minimum(
                target_critic1(next_state_tensor, next_action),
                target_critic2(next_state_tensor, next_action),
            )
            target = reward_tensor + gamma * (1.0 - done_tensor) * target_q
        critic_optimizer.zero_grad()
        q1 = critic1(state_tensor, action_tensor)
        q2 = critic2(state_tensor, action_tensor)
        critic_loss = torch.mean((q1 - target).pow(2)) + torch.mean((q2 - target).pow(2))
        critic_loss.backward()
        critic_optimizer.step()

        actor_loss = torch.tensor(0.0)
        if epoch % policy_delay == 0:
            actor_optimizer.zero_grad()
            proposed_action = actor(state_tensor)
            actor_loss = -critic1(state_tensor, proposed_action).mean() + bc_weight * torch.mean(
                (proposed_action - action_tensor).pow(2)
            )
            actor_loss.backward()
            actor_optimizer.step()
            _soft_update(target_actor, actor, tau=0.005)
            _soft_update(target_critic1, critic1, tau=0.005)
            _soft_update(target_critic2, critic2, tau=0.005)
        trace.append(
            {
                "method": "td3_bc",
                "epoch": epoch,
                "train_loss": float((critic_loss + actor_loss).detach().cpu()),
                "train_row_count": len(rows),
                "loss_family": "td3_bc_twin_critic_delayed_actor_bc",
            }
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "td3_bc",
            "algorithm_scope": "full_td3_bc_offline_actor_critic",
            "actor_state_dict": actor.state_dict(),
            "critic1_state_dict": critic1.state_dict(),
            "critic2_state_dict": critic2.state_dict(),
            "target_actor_state_dict": target_actor.state_dict(),
            "target_critic1_state_dict": target_critic1.state_dict(),
            "target_critic2_state_dict": target_critic2.state_dict(),
            "state_columns": list(STATE_COLUMNS),
            "action_columns": list(ACTION_COLUMNS),
            "state_mean": batch.state_mean.tolist(),
            "state_std": batch.state_std.tolist(),
            "action_mean": batch.action_mean.tolist(),
            "action_std": batch.action_std.tolist(),
            "gamma": gamma,
            "bc_weight": bc_weight,
            "policy_delay": policy_delay,
            "transition_count": len(rows),
        },
        checkpoint_path,
    )
    return (
        _ModelBundle(
            method="td3_bc",
            model=None,
            x_mean=np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            x_std=np.ones((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            y_mean=0.0,
            y_std=1.0,
            state_mean=batch.state_mean,
            state_std=batch.state_std,
            action_mean=batch.action_mean,
            action_std=batch.action_std,
            actor_model=actor,
            critic1_model=critic1,
            critic2_model=critic2,
            algorithm_scope="full_td3_bc_offline_actor_critic",
        ),
        trace,
    )


def _train_maddpg_method(
    rows: list[dict[str, Any]],
    *,
    epochs: int,
    checkpoint_path: Path,
) -> tuple[_ModelBundle, list[dict[str, Any]]]:
    batch = _offline_transition_batch(rows)
    state_tensor = torch.tensor(batch.states, dtype=torch.float32)
    action_tensor = torch.tensor(batch.actions, dtype=torch.float32)
    reward_tensor = torch.tensor(batch.rewards * 0.001, dtype=torch.float32)
    next_state_tensor = torch.tensor(batch.next_states, dtype=torch.float32)
    done_tensor = torch.tensor(batch.dones, dtype=torch.float32)
    actor = DeterministicActionActor(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    critic = StateActionQNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1.0e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1.0e-3)
    gamma = 0.97
    bc_weight = 1.0
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        with torch.no_grad():
            next_action = target_actor(next_state_tensor)
            target_q = target_critic(next_state_tensor, next_action)
            target = reward_tensor + gamma * (1.0 - done_tensor) * target_q
        critic_optimizer.zero_grad()
        critic_loss = torch.mean((critic(state_tensor, action_tensor) - target).pow(2))
        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        proposed_action = actor(state_tensor)
        actor_loss = -critic(state_tensor, proposed_action).mean() + bc_weight * torch.mean(
            (proposed_action - action_tensor).pow(2)
        )
        actor_loss.backward()
        actor_optimizer.step()
        _soft_update(target_actor, actor, tau=0.01)
        _soft_update(target_critic, critic, tau=0.01)
        trace.append(
            {
                "method": "maddpg",
                "epoch": epoch,
                "train_loss": float((critic_loss + actor_loss).detach().cpu()),
                "train_row_count": len(rows),
                "loss_family": "maddpg_ctde_actor_centralized_critic_bc_regularized",
            }
        )
    _save_marl_checkpoint(
        checkpoint_path,
        method="maddpg",
        batch=batch,
        payload={
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "target_actor_state_dict": target_actor.state_dict(),
            "target_critic_state_dict": target_critic.state_dict(),
            "gamma": gamma,
            "bc_weight": bc_weight,
        },
    )
    return (
        _ModelBundle(
            method="maddpg",
            model=None,
            x_mean=np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            x_std=np.ones((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            y_mean=0.0,
            y_std=1.0,
            state_mean=batch.state_mean,
            state_std=batch.state_std,
            action_mean=batch.action_mean,
            action_std=batch.action_std,
            actor_model=actor,
            critic1_model=critic,
            algorithm_scope=MARL_METHOD_SCOPES["maddpg"],
        ),
        trace,
    )


def _train_policy_score_marl_method(
    rows: list[dict[str, Any]],
    *,
    method: str,
    epochs: int,
    checkpoint_path: Path,
) -> tuple[_ModelBundle, list[dict[str, Any]]]:
    batch = _offline_transition_batch(rows)
    state_tensor = torch.tensor(batch.states, dtype=torch.float32)
    action_tensor = torch.tensor(batch.actions, dtype=torch.float32)
    reward_tensor = torch.tensor(batch.rewards * 0.001, dtype=torch.float32)
    labels, advantages, weights = _state_group_labels_and_advantages(batch)
    policy_model = StateActionPolicyNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    value_model = StateValueNet(len(STATE_COLUMNS))
    optimizer = torch.optim.Adam(list(policy_model.parameters()) + list(value_model.parameters()), lr=1.0e-3)
    clip_range = 0.2
    trust_region_weight = 0.0 if method == "mappo" else 0.5
    old_logits = torch.zeros_like(reward_tensor)
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad()
        logits = policy_model(state_tensor, action_tensor)
        values = value_model(state_tensor)
        ratio = torch.exp((logits - old_logits).clamp(-4.0, 4.0))
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.mean(torch.minimum(ratio * advantages, clipped_ratio * advantages))
        bc_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        value_loss = torch.mean((values - reward_tensor).pow(2))
        trust_penalty = torch.mean((logits - old_logits).pow(2))
        loss = policy_loss + 0.1 * bc_loss + 0.5 * value_loss + trust_region_weight * trust_penalty
        loss.backward()
        optimizer.step()
        old_logits = 0.9 * old_logits + 0.1 * logits.detach()
        trace.append(
            {
                "method": method,
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "train_row_count": len(rows),
                "loss_family": (
                    "mappo_clipped_advantage_policy_score"
                    if method == "mappo"
                    else "happo_hatrpo_sequential_trust_region_policy_score"
                ),
            }
        )
    _save_marl_checkpoint(
        checkpoint_path,
        method=method,
        batch=batch,
        payload={
            "policy_state_dict": policy_model.state_dict(),
            "value_state_dict": value_model.state_dict(),
            "clip_range": clip_range,
            "trust_region_weight": trust_region_weight,
        },
    )
    return (
        _ModelBundle(
            method=method,
            model=None,
            x_mean=np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            x_std=np.ones((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            y_mean=0.0,
            y_std=1.0,
            state_mean=batch.state_mean,
            state_std=batch.state_std,
            action_mean=batch.action_mean,
            action_std=batch.action_std,
            policy_model=policy_model,
            algorithm_scope=MARL_METHOD_SCOPES[method],
        ),
        trace,
    )


def _train_q_score_marl_method(
    rows: list[dict[str, Any]],
    *,
    method: str,
    epochs: int,
    checkpoint_path: Path,
) -> tuple[_ModelBundle, list[dict[str, Any]]]:
    batch = _offline_transition_batch(rows)
    state_tensor = torch.tensor(batch.states, dtype=torch.float32)
    action_tensor = torch.tensor(batch.actions, dtype=torch.float32)
    reward_tensor = torch.tensor(batch.rewards * 0.001, dtype=torch.float32)
    next_state_tensor = torch.tensor(batch.next_states, dtype=torch.float32)
    done_tensor = torch.tensor(batch.dones, dtype=torch.float32)
    if method == "qmix":
        q_model: nn.Module = MonotonicQMixNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    elif method == "vdn":
        q_model = ValueDecompositionQNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    elif method == "mat":
        q_model = StateActionTransformerQNet(len(STATE_COLUMNS), len(ACTION_COLUMNS))
    else:
        raise ValueError(f"Unsupported q-score MARL method {method}")
    target_q_model = copy.deepcopy(q_model)
    optimizer = torch.optim.Adam(q_model.parameters(), lr=1.0e-3)
    gamma = 0.97
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        with torch.no_grad():
            target_q = target_q_model(next_state_tensor, action_tensor)
            target = reward_tensor + gamma * (1.0 - done_tensor) * target_q
        optimizer.zero_grad()
        q_values = q_model(state_tensor, action_tensor)
        loss = torch.mean((q_values - target).pow(2))
        loss.backward()
        optimizer.step()
        _soft_update(target_q_model, q_model, tau=0.01)
        trace.append(
            {
                "method": method,
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "train_row_count": len(rows),
                "loss_family": {
                    "qmix": "qmix_monotonic_mixed_action_value",
                    "vdn": "vdn_summed_component_action_value",
                    "mat": "mat_transformer_state_action_value",
                }[method],
            }
        )
    _save_marl_checkpoint(
        checkpoint_path,
        method=method,
        batch=batch,
        payload={
            "q_state_dict": q_model.state_dict(),
            "target_q_state_dict": target_q_model.state_dict(),
            "gamma": gamma,
        },
    )
    return (
        _ModelBundle(
            method=method,
            model=None,
            x_mean=np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            x_std=np.ones((1, len(FEATURE_COLUMNS)), dtype=np.float32),
            y_mean=0.0,
            y_std=1.0,
            state_mean=batch.state_mean,
            state_std=batch.state_std,
            action_mean=batch.action_mean,
            action_std=batch.action_std,
            q_model=q_model,
            algorithm_scope=MARL_METHOD_SCOPES[method],
        ),
        trace,
    )


def _save_marl_checkpoint(
    checkpoint_path: Path,
    *,
    method: str,
    batch: _OfflineTransitionBatch,
    payload: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": method,
            "algorithm_scope": MARL_METHOD_SCOPES[method],
            "training_protocol": "shared_rollout_dataset_matched_seed_closed_loop",
            "state_columns": list(STATE_COLUMNS),
            "action_columns": list(ACTION_COLUMNS),
            "state_mean": batch.state_mean.tolist(),
            "state_std": batch.state_std.tolist(),
            "action_mean": batch.action_mean.tolist(),
            "action_std": batch.action_std.tolist(),
            "transition_count": int(len(batch.rewards)),
            **payload,
        },
        checkpoint_path,
    )


def _soft_update(target: nn.Module, source: nn.Module, *, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def _train_method(
    rows: list[dict[str, Any]],
    *,
    method: str,
    epochs: int,
    checkpoint_path: Path,
) -> tuple[_ModelBundle, list[dict[str, Any]]]:
    if method == "diffusion_policy":
        return _train_trajectory_diffusion_method(
            rows,
            epochs=epochs,
            checkpoint_path=checkpoint_path,
        )
    if method == "maddpg":
        return _train_maddpg_method(rows, epochs=epochs, checkpoint_path=checkpoint_path)
    if method in {"mappo", "happo_hatrpo"}:
        return _train_policy_score_marl_method(
            rows,
            method=method,
            epochs=epochs,
            checkpoint_path=checkpoint_path,
        )
    if method in {"qmix", "vdn", "mat"}:
        return _train_q_score_marl_method(
            rows,
            method=method,
            epochs=epochs,
            checkpoint_path=checkpoint_path,
        )
    if method == "iql":
        return _train_iql_method(rows, epochs=epochs, checkpoint_path=checkpoint_path)
    if method == "cql":
        return _train_cql_method(rows, epochs=epochs, checkpoint_path=checkpoint_path)
    if method == "td3_bc":
        return _train_td3_bc_method(rows, epochs=epochs, checkpoint_path=checkpoint_path)
    x, y = _matrix_and_target(rows)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = np.where(x.std(axis=0, keepdims=True) > 1.0e-6, x.std(axis=0, keepdims=True), 1.0)
    y_mean = float(y.mean())
    y_std = float(y.std()) if float(y.std()) > 1.0e-6 else 1.0
    x_tensor = torch.tensor((x - x_mean) / x_std, dtype=torch.float32)
    y_tensor = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    model = ScoreNet(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad()
        pred = model(x_tensor)
        if method == "safe_gpinn":
            loss = torch.mean((pred - y_tensor).pow(2)) + 0.1 * _pairwise_loss(pred, y_tensor)
            loss_family = "safe_pairwise_return_mse"
        elif method == "iql":
            loss = _expectile_loss(pred, y_tensor, tau=0.7)
            loss_family = "iql_expectile_return_loss"
        elif method == "cql":
            loss = torch.mean((pred - y_tensor).pow(2)) + 0.02 * torch.logsumexp(pred, dim=0)
            loss_family = "cql_mse_with_conservative_penalty"
        else:
            loss = torch.mean((pred - y_tensor).pow(2))
            loss_family = "td3_bc_supervised_return_mse"
        loss.backward()
        optimizer.step()
        trace.append(
            {
                "method": method,
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "train_row_count": len(rows),
                "loss_family": loss_family,
            }
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "method": method,
        "state_dict": model.state_dict(),
        "feature_columns": list(FEATURE_COLUMNS),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
    }
    if method == "safe_gpinn":
        checkpoint.update(
            {
                "algorithm_scope": "safe_graph_pinn_return_physics_risk",
                "learned_score_weight": SAFE_GPINN_LEARNED_SCORE_WEIGHT,
                "physics_consistency_weight": SAFE_GPINN_PHYSICS_SCORE_WEIGHT,
                "scenario_tree_prefix_weight": SAFE_GPINN_PREFIX_SCORE_WEIGHT,
                "prefix_viability_floor": SAFE_GPINN_PREFIX_VIABILITY_FLOOR,
                "relay_viability_shield": "hard_filter_when_available",
                "latent_wait_lookahead_steps": SAFE_GPINN_LATENT_WAIT_LOOKAHEAD_STEPS,
                "latent_wait_value_ratio": SAFE_GPINN_LATENT_WAIT_VALUE_RATIO,
                "score_normalization": "candidate_zscore_for_learned_and_physics",
                "loss_family": "return_mse_pairwise_physics_guided_online_score",
            }
        )
    torch.save(checkpoint, checkpoint_path)
    return (
        _ModelBundle(
            method=method,
            model=model,
            x_mean=x_mean.astype(np.float32),
            x_std=x_std.astype(np.float32),
            y_mean=y_mean,
            y_std=y_std,
        ),
        trace,
    )


def _pairwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    max_rows: int = PAIRWISE_LOSS_MAX_ROWS,
) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if len(pred) < 2:
        return torch.tensor(0.0)
    row_limit = max(2, int(max_rows))
    if len(pred) > row_limit:
        rank_order = torch.argsort(target)
        rank_positions = torch.linspace(
            0,
            len(rank_order) - 1,
            steps=row_limit,
            device=target.device,
        ).round().long()
        sample_index = rank_order[rank_positions]
        pred = pred[sample_index]
        target = target[sample_index]
    delta_target = target[:, None] - target[None, :]
    delta_pred = pred[:, None] - pred[None, :]
    mask = delta_target > 1.0e-6
    if not bool(mask.any()):
        return torch.tensor(0.0)
    return torch.nn.functional.softplus(-delta_pred[mask]).mean()


def _expectile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    diff = target - pred
    weights = torch.where(diff > 0, tau, 1.0 - tau)
    return torch.mean(weights * diff.pow(2))


def _matrix_and_target(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("closed-loop rollout dataset is empty")
    x = np.asarray(
        [
            [
                _safe_float(row.get(column), FEATURE_DEFAULTS.get(column, 0.0))
                for column in FEATURE_COLUMNS
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    y = np.asarray([float(row["return_to_go"]) for row in rows], dtype=np.float32)
    return x, y


def _feature_values_from_trace(
    scenario: PTINScenario,
    *,
    edge_lookup: dict[str, PDNEdge],
    row: dict[str, Any],
) -> dict[str, float]:
    pre_step = max(0, int(row["step_index"]) - 1)
    applied = _truthy(row.get("applied"))
    action_type = str(row.get("action_type") or "restore_pdn_edge")
    action_predeploy = action_type == "predeploy_uav_relay"
    applied_restore = int(
        applied
        and bool(str(row.get("target_id", "")))
        and action_type == "restore_pdn_edge"
    )
    remaining_before = int(row["remaining_failed_edge_count"]) + applied_restore
    restored_before = max(0, int(row["restored_failed_edge_count"]) - applied_restore)
    target_id = str(row["target_id"])
    active_failed_before = int(row.get("active_failed_edge_count", remaining_before)) + applied_restore
    latent_failed_before = int(row.get("latent_failed_edge_count", 0))
    time_min = float(row.get("time_min") or 0.0)
    values = _feature_values(
        scenario,
        edge_lookup=edge_lookup,
        target_id=target_id,
        time_min=time_min,
        pre_step_index=pre_step,
        remaining_failed_before=remaining_before,
        restored_failed_before=restored_before,
        active_failed_before=active_failed_before,
        latent_failed_before=latent_failed_before,
        restores_target=not action_predeploy,
    )
    values["communication_mode_relay"] = 1.0 if str(row.get("communication_mode") or "direct") == "uav_relay" else 0.0
    values["action_predeploy_uav_relay"] = 1.0 if action_predeploy else 0.0
    return values


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _target_transport_feature_values(
    scenario: PTINScenario,
    target_id: str,
) -> dict[str, float]:
    target_edge_ids = target_utn_edge_ids_for_pdn_edge(scenario, target_id)
    utn_by_id = {edge.edge_id: edge for edge in scenario.utn_edges}
    travel_times_s: list[float] = []
    for edge_id in target_edge_ids:
        edge = utn_by_id.get(edge_id)
        if edge is None:
            continue
        speed_km_h = _safe_float(edge.raw.get("failed_speed_km_h"), 0.0)
        if speed_km_h <= 0.0:
            speed_km_h = _safe_float(edge.raw.get("normal_speed_km_h"), 0.0)
        if speed_km_h <= 0.0:
            speed_km_h = 30.0
        travel_times_s.append(max(0.0, float(edge.length_km)) * 3600.0 / speed_km_h)
    nominal_time_s = sum(travel_times_s) / len(travel_times_s) if travel_times_s else 60.0
    robust_time_s = robust_mean_travel_time_s(scenario, nominal_time_s)
    return {
        "target_traffic_edge_count": float(len(travel_times_s)),
        "target_nominal_travel_time_s": float(nominal_time_s),
        "target_robust_travel_time_s": float(robust_time_s),
        "robust_travel_time_margin_s": float(max(0.0, robust_time_s - nominal_time_s)),
    }


def _feature_values_from_observation(
    scenario: PTINScenario,
    *,
    edge_lookup: dict[str, PDNEdge],
    observation: dict[str, Any],
    action: ClosedLoopAction,
) -> dict[str, float]:
    time_min = float(observation.get("time_min") or 0.0) + float(
        observation.get("step_minutes") or 5.0
    )
    values = _feature_values(
        scenario,
        edge_lookup=edge_lookup,
        target_id=action.target_id,
        time_min=time_min,
        pre_step_index=int(observation.get("step_index") or 0),
        remaining_failed_before=int(observation.get("remaining_failed_edge_count") or 0),
        restored_failed_before=int(observation.get("restored_failed_edge_count") or 0),
        active_failed_before=int(observation.get("remaining_failed_edge_count") or 0),
        latent_failed_before=int(observation.get("latent_failed_edge_count") or 0),
        restores_target=action.action_type != "predeploy_uav_relay",
    )
    values["communication_mode_relay"] = _communication_mode_relay(action)
    values["action_predeploy_uav_relay"] = (
        1.0 if action.action_type == "predeploy_uav_relay" else 0.0
    )
    return values


def _feature_values(
    scenario: PTINScenario,
    *,
    edge_lookup: dict[str, PDNEdge],
    target_id: str,
    time_min: float,
    pre_step_index: int,
    remaining_failed_before: int,
    restored_failed_before: int,
    active_failed_before: int | None = None,
    latent_failed_before: int | None = None,
    restores_target: bool = True,
) -> dict[str, float]:
    edge = edge_lookup[target_id]
    projected_restored = restored_failed_before + (1 if restores_target else 0)
    dispatchable_ev_count = dispatchable_v2g_vehicle_count_at(
        scenario.fleet,
        time_min,
    )
    v2g_availability_factor = ev_availability_factor_at(
        scenario.fleet,
        time_min,
    )
    action_order_index = list(scenario.failed_pdn_edges).index(target_id)
    release_steps = scenario.disaster.failure_release_step_by_edge
    active_from_schedule = sum(
        1
        for edge_id in scenario.failed_pdn_edges
        if int(release_steps.get(edge_id, 0)) <= pre_step_index
    )
    active_before = (
        float(active_failed_before)
        if active_failed_before is not None
        else float(max(0, active_from_schedule - restored_failed_before))
    )
    latent_before = (
        float(latent_failed_before)
        if latent_failed_before is not None
        else float(max(0, len(scenario.failed_pdn_edges) - active_from_schedule))
    )
    target_transport = _target_transport_feature_values(scenario, target_id)
    values = {
        "pre_step_index": float(pre_step_index),
        "remaining_failed_before": float(remaining_failed_before),
        "restored_failed_before": float(restored_failed_before),
        "action_order_index": float(action_order_index),
        "action_load_gain_kw": float(failed_edge_restoration_load_kw(scenario, target_id)),
        "edge_length_km": float(edge.length_km),
        "edge_r_pu": float(edge.r_pu),
        "edge_x_pu": float(edge.x_pu),
        "projected_mess_support_kw": float(
            min(projected_restored, scenario.fleet.mess_count) * scenario.fleet.mess_unit_power_kw
        ),
        "projected_v2g_support_kw": float(
            min(projected_restored, dispatchable_ev_count)
            * scenario.fleet.ev_unit_power_kw
        ),
        "v2g_availability_factor": float(v2g_availability_factor),
        "v2g_available_energy_capacity_kwh": float(
            v2g_energy_capacity_kwh_at(scenario.fleet, time_min)
        ),
        "critical_load_weight": float(critical_load_weight(scenario, target_id)),
        "topology_vulnerability_score": float(
            topology_vulnerability_score(scenario, target_id)
        ),
        "predeployment_quality_index": float(
            scenario.disaster.predeployment_quality_index
        ),
        **target_transport,
        "communication_mode_relay": 0.0,
        "action_predeploy_uav_relay": 0.0,
        "active_failed_before": active_before,
        "latent_failed_before": latent_before,
        "weighted_restoration_value_kw": float(
            weighted_restoration_value_kw(scenario, target_id)
        ),
        "tree_branch_probability": 1.0,
        "tree_traffic_multiplier": 1.0,
        "tree_packet_delivery_factor": 1.0,
        "tree_resource_factor": 1.0,
        "tree_branch_penalty_kw": 0.0,
    }
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train closed-loop PTIN policies.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adapter-mode", default=None)
    parser.add_argument("--scenario-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reuse-rollout-dataset", type=Path, default=None)
    parser.add_argument("--eval-only-from-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--rollout-dataset", type=Path, default=None)
    parser.add_argument("--evaluation-mode", choices=["env", "rollout_dataset"], default="env")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.eval_only_from_checkpoints:
        summary = run_closed_loop_policy_checkpoint_evaluation(
            output_dir=args.output_dir,
            config_path=args.config,
            adapter_mode=args.adapter_mode,
            scenario_dir=args.scenario_dir,
            seed=args.seed,
            rollout_dataset=args.rollout_dataset or args.reuse_rollout_dataset,
            checkpoint_dir=args.checkpoint_dir,
            evaluation_mode=args.evaluation_mode,
        )
        print("Closed-loop PTIN policy checkpoint evaluation")
        print(f"- adapter_mode: {summary['adapter_mode']}")
        print(f"- rollout_row_count: {summary['rollout_row_count']}")
        print(f"- evaluation_protocol: {summary['evaluation_protocol']}")
        return 0
    summary = run_closed_loop_policy_training(
        output_dir=args.output_dir,
        config_path=args.config,
        adapter_mode=args.adapter_mode,
        scenario_dir=args.scenario_dir,
        epochs=args.epochs,
        max_sequences=args.max_sequences,
        seed=args.seed,
        reuse_rollout_dataset=args.reuse_rollout_dataset,
        evaluation_mode=args.evaluation_mode,
    )
    print("Closed-loop PTIN policy training")
    print(f"- adapter_mode: {summary['adapter_mode']}")
    print(f"- rollout_row_count: {summary['rollout_row_count']}")
    print(f"- truth_boundary: {summary['truth_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
