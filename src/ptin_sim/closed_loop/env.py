from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from .communication_ns3 import ClosedLoopCommunicationAdapter, CommunicationStepResult
from .dependencies import (
    CrossLayerFeasibility,
    critical_load_weight,
    evaluate_cross_layer_feasibility,
    failed_edge_restoration_load_kw,
    robust_mean_travel_time_s,
    target_travel_time_s,
    topology_vulnerability_score,
    weighted_restoration_value_kw,
)
from .power_ac_opf import ACPowerResult, ClosedLoopACPowerAdapter
from .traffic_traci import ClosedLoopTrafficAdapter, TrafficStepResult
from .types import (
    ClosedLoopAction,
    PTINScenario,
    dispatchable_v2g_vehicle_count_at,
    ev_availability_factor_at,
    v2g_energy_capacity_kwh_at,
)


RESTORATION_REWARD_SCALE = 0.25
SHED_ENERGY_PENALTY_PER_KWH = 0.8
ACTION_DURATION_PENALTY_PER_MIN = 1.5
PACKET_LOSS_PENALTY = 120.0
TRAVEL_TIME_PENALTY_PER_S = 0.1
COMM_DELAY_PENALTY_PER_MS = 0.2
CONTROL_UNAVAILABLE_PENALTY = 80.0
BLOCKED_ACTION_PENALTY = 180.0
VOLTAGE_VIOLATION_PENALTY = 1200.0
LINE_LOADING_VIOLATION_PENALTY = 12.0
TERMINAL_RESTORATION_BONUS = 500.0
DUAL_CHANNEL_COORDINATION_MIN = 0.5


@dataclass
class _RuntimeState:
    step_index: int = 0
    time_min: float = 0.0
    restored_pdn_edges: set[str] = field(default_factory=set)
    predeployed_uav_targets: set[str] = field(default_factory=set)
    resource_energy_remaining_kwh: float = 0.0
    v2g_energy_remaining_kwh: float = 0.0
    v2g_available_energy_capacity_kwh: float = 0.0
    last_ac: ACPowerResult | None = None
    last_traffic: TrafficStepResult | None = None
    last_comm: CommunicationStepResult | None = None
    trace_rows: list[dict[str, Any]] = field(default_factory=list)


class ClosedLoopPTINEnv:
    def __init__(
        self,
        scenario: PTINScenario,
        *,
        power_adapter: Any | None = None,
        traffic_adapter: Any | None = None,
        communication_adapter: Any | None = None,
        max_steps: int = 12,
        step_minutes: float = 5.0,
    ) -> None:
        self.scenario = scenario
        self.power_adapter = power_adapter or ClosedLoopACPowerAdapter()
        self.traffic_adapter = traffic_adapter or ClosedLoopTrafficAdapter()
        self.communication_adapter = communication_adapter or ClosedLoopCommunicationAdapter()
        self.max_steps = max_steps
        self.step_minutes = step_minutes
        self._state = _RuntimeState()

    def reset(self) -> dict[str, Any]:
        v2g_capacity = self._v2g_energy_capacity_kwh(time_min=0.0)
        self._state = _RuntimeState(
            resource_energy_remaining_kwh=round(
                max(0, self.scenario.fleet.mess_count)
                * max(0.0, self.scenario.fleet.mess_unit_energy_kwh),
                6,
            ),
            v2g_energy_remaining_kwh=v2g_capacity,
            v2g_available_energy_capacity_kwh=v2g_capacity,
        )
        return self.observe()

    def observe(self) -> dict[str, Any]:
        remaining = self._remaining_failed_edges()
        last_ac = self._state.last_ac
        last_traffic = self._state.last_traffic
        last_comm = self._state.last_comm
        return {
            "scenario_id": self.scenario.scenario_id,
            "disaster_category": self.scenario.disaster.disaster_category,
            "control_paradigm": self.scenario.disaster.control_paradigm,
            "step_index": self._state.step_index,
            "time_min": self._state.time_min,
            "step_minutes": self.step_minutes,
            "failed_pdn_edges": sorted(remaining),
            "latent_failed_edges": sorted(self._latent_failed_edges()),
            "all_unrestored_failed_edges": sorted(self._unrestored_failed_edges()),
            "restored_pdn_edges": sorted(self._state.restored_pdn_edges),
            "predeployed_uav_targets": sorted(self._state.predeployed_uav_targets),
            "predeployed_uav_target_count": len(self._state.predeployed_uav_targets),
            "restored_failed_edge_count": len(self._state.restored_pdn_edges),
            "remaining_failed_edge_count": len(remaining),
            "latent_failed_edge_count": len(self._latent_failed_edges()),
            "available_actions": [action.__dict__ for action in self.available_actions()],
            "last_ac_status": last_ac.status if last_ac else "",
            "last_min_voltage_pu": last_ac.min_voltage_pu if last_ac else None,
            "last_max_line_loading_pct": last_ac.max_line_loading_pct if last_ac else None,
            "last_requested_load_kw": last_ac.requested_load_kw if last_ac else None,
            "last_served_load_kw": last_ac.served_load_kw if last_ac else None,
            "last_shed_load_kw": last_ac.shed_load_kw if last_ac else None,
            "last_mobile_dispatch_kw": last_ac.mobile_dispatch_kw if last_ac else None,
            "last_traffic_status": last_traffic.status if last_traffic else "",
            "last_mean_travel_time_s": last_traffic.mean_travel_time_s if last_traffic else 0.0,
            "last_communication_status": last_comm.status if last_comm else "",
            "last_packet_delivery_rate": last_comm.delivery_rate if last_comm else 0.0,
            "resource_energy_remaining_kwh": self._state.resource_energy_remaining_kwh,
            "v2g_availability_factor": self._v2g_availability_factor(),
            "v2g_dispatchable_ev_count": self._dispatchable_v2g_vehicle_count(),
            "v2g_available_energy_capacity_kwh": self._state.v2g_available_energy_capacity_kwh,
            "v2g_energy_remaining_kwh": self._state.v2g_energy_remaining_kwh,
        }

    def available_actions(self) -> list[ClosedLoopAction]:
        actions: list[ClosedLoopAction] = []
        for edge_id in sorted(self._remaining_failed_edges()):
            actions.append(
                ClosedLoopAction(
                    action_id=f"restore_{edge_id}__direct",
                    action_type="restore_pdn_edge",
                    target_id=edge_id,
                    resource_id="MESS_1",
                    metadata={
                        "mess_units": 1,
                        "v2g_units": 1,
                        "communication_mode": "direct",
                    },
                )
            )
            if self.scenario.fleet.uav_count > 0:
                actions.append(
                    ClosedLoopAction(
                        action_id=f"restore_{edge_id}__uav_relay",
                        action_type="restore_pdn_edge",
                        target_id=edge_id,
                        resource_id="MESS_1",
                        metadata={
                            "mess_units": 1,
                            "v2g_units": 1,
                            "communication_mode": "uav_relay",
                        },
                    )
                )
                actions.append(
                    ClosedLoopAction(
                        action_id=f"restore_{edge_id}__dual_channel",
                        action_type="restore_pdn_edge",
                        target_id=edge_id,
                        resource_id="MESS_1",
                        metadata={
                            "mess_units": 1,
                            "v2g_units": 1,
                            "communication_mode": "dual_channel",
                        },
                    )
                )
        if self.scenario.fleet.uav_count > 0:
            for edge_id in sorted(self._unrestored_failed_edges()):
                if edge_id in self._state.predeployed_uav_targets:
                    continue
                actions.append(
                    ClosedLoopAction(
                        action_id=f"predeploy_uav_relay_{edge_id}",
                        action_type="predeploy_uav_relay",
                        target_id=edge_id,
                        resource_id="UAV_1",
                        metadata={"communication_mode": "uav_relay"},
                    )
                )
        if self._state.step_index < self.max_steps:
            wait_metadata = {}
            if self.scenario.fleet.uav_count > 0:
                wait_metadata["communication_mode"] = "uav_relay"
            actions.append(
                ClosedLoopAction(
                    action_id="wait_for_progressive_update",
                    action_type="wait",
                    target_id="",
                    resource_id="",
                    metadata=wait_metadata,
                )
            )
        return actions

    def step(
        self, action: ClosedLoopAction
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._state.step_index >= self.max_steps:
            raise RuntimeError("Episode is already truncated.")

        communication_mode = self._communication_mode(action)
        predeployed_for_action = bool(
            action.target_id
            and communication_mode in {"uav_relay", "dual_channel"}
            and action.target_id in self._state.predeployed_uav_targets
        )
        nominal_time_min = self._state.time_min + self.step_minutes
        traffic = self.traffic_adapter.step(time_min=nominal_time_min)
        target_time_s, target_traffic_edges = target_travel_time_s(
            self.scenario,
            action.target_id,
            traffic,
        )
        target_robust_time_s = robust_mean_travel_time_s(self.scenario, target_time_s)
        communication = self._communication_step(
            time_min=nominal_time_min,
            target_id=action.target_id,
            communication_mode=communication_mode,
            target_robust_travel_time_s=target_robust_time_s,
            predeployed_uav_relay=predeployed_for_action,
        )
        action_duration_min = self._action_duration_min(
            action=action,
            communication=communication,
            communication_mode=communication_mode,
            target_robust_travel_time_s=target_robust_time_s,
        )
        self._state.time_min = round(self._state.time_min + action_duration_min, 6)
        self._sync_v2g_availability()

        remaining_before = self._remaining_failed_edges()
        applied = False
        block_reason = ""
        restored_load_gain_kw = 0.0
        weighted_value_kw = 0.0
        critical_weight = 1.0
        vulnerability_score = 0.0
        feasibility = (
            self._wait_feasibility(
                traffic,
                communication=communication,
                communication_mode=communication_mode,
            )
            if action.action_type in {"wait", "predeploy_uav_relay"}
            else self._cross_layer_feasibility(
                target_id=action.target_id,
                communication=communication,
                traffic=traffic,
                communication_mode=communication_mode,
                action_duration_min=action_duration_min,
            )
        )
        if action.action_type == "wait":
            applied = True
        elif action.action_type == "predeploy_uav_relay":
            if self.scenario.fleet.uav_count <= 0:
                block_reason = "uav_relay_unavailable"
            elif action.target_id not in self._unrestored_failed_edges():
                block_reason = "target_not_failed_or_already_restored"
            elif action.target_id in self._state.predeployed_uav_targets:
                block_reason = "uav_relay_already_predeployed"
            elif not feasibility.feasible:
                block_reason = feasibility.blockers[0] if feasibility.blockers else "cross_layer_unavailable"
            else:
                self._state.predeployed_uav_targets.add(action.target_id)
                applied = True
        elif action.action_type != "restore_pdn_edge":
            block_reason = "unsupported_action_type"
        elif action.target_id not in remaining_before:
            block_reason = "target_not_failed_or_already_restored"
        elif not feasibility.feasible:
            block_reason = feasibility.blockers[0] if feasibility.blockers else "cross_layer_unavailable"
        else:
            self._state.restored_pdn_edges.add(action.target_id)
            self._state.resource_energy_remaining_kwh = round(
                max(
                    0.0,
                    self._state.resource_energy_remaining_kwh
                    - feasibility.resource_energy_required_kwh,
                ),
                6,
            )
            applied = True
            restored_load_gain_kw = failed_edge_restoration_load_kw(
                self.scenario, action.target_id
            )
            weighted_value_kw = weighted_restoration_value_kw(self.scenario, action.target_id)
            critical_weight = critical_load_weight(self.scenario, action.target_id)
            vulnerability_score = topology_vulnerability_score(self.scenario, action.target_id)
            self._state.predeployed_uav_targets.discard(action.target_id)

        closed_edges = self._closed_pdn_edges()
        restored_fraction = self._restored_load_fraction()
        mess_support_kw = self._active_mess_support_kw()
        v2g_support_kw = self._active_v2g_support_kw()
        ac = self.power_adapter.run_ac_opf_or_pf(
            self.scenario,
            closed_pdn_edges=closed_edges,
            restored_load_fraction=restored_fraction,
            mess_support_kw=mess_support_kw,
            v2g_support_kw=v2g_support_kw,
        )
        v2g_energy_required_kwh = feasibility.v2g_energy_required_kwh
        if applied and action.action_type == "restore_pdn_edge":
            self._state.v2g_energy_remaining_kwh = round(
                max(0.0, self._state.v2g_energy_remaining_kwh - v2g_energy_required_kwh),
                6,
            )
        terminal_restoration = not self._unrestored_failed_edges()
        shed_energy_kwh = self._shed_energy_kwh(
            ac=ac,
            action_duration_min=action_duration_min,
        )
        reward = self._reward(
            restored_load_gain_kw=restored_load_gain_kw,
            ac=ac,
            traffic=traffic,
            communication=communication,
            applied=applied,
            weighted_restoration_value_kw=weighted_value_kw,
            action_duration_min=action_duration_min,
            terminal_restoration=terminal_restoration,
        )
        self._state.step_index += 1
        self._state.last_ac = ac
        self._state.last_traffic = traffic
        self._state.last_comm = communication
        info = {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "target_id": action.target_id,
            "applied": applied,
            "block_reason": block_reason,
            "restored_load_gain_kw": restored_load_gain_kw,
            "weighted_restoration_value_kw": weighted_value_kw,
            "critical_load_weight": critical_weight,
            "topology_vulnerability_score": vulnerability_score,
            "ac_status": ac.status,
            "traffic_status": traffic.status,
            "communication_status": communication.status,
            "reward": reward,
            "time_min": self._state.time_min,
            "action_duration_min": round(action_duration_min, 6),
            "communication_mode": communication_mode,
            "requested_load_kw": ac.requested_load_kw,
            "served_load_kw": ac.served_load_kw,
            "shed_load_kw": ac.shed_load_kw,
            "shed_energy_kwh_step": shed_energy_kwh,
            "mobile_support_capacity_kw": ac.mobile_support_capacity_kw,
            "mobile_dispatch_kw": ac.mobile_dispatch_kw,
            "packet_count": communication.packet_count,
            "packet_delivery_rate": communication.delivery_rate,
            "mean_delay_ms": communication.mean_delay_ms,
            "control_available": communication.control_available,
            "mean_travel_time_s": traffic.mean_travel_time_s,
            "robust_mean_travel_time_s": robust_mean_travel_time_s(
                self.scenario, traffic.mean_travel_time_s
            ),
            "target_traffic_edge_ids": ";".join(target_traffic_edges),
            "target_traffic_edge_count": len(target_traffic_edges),
            "target_travel_time_s": target_time_s,
            "target_robust_travel_time_s": target_robust_time_s,
            "traffic_uncertainty_mode": self.scenario.disaster.traffic_uncertainty_mode,
            "traffic_edge_count": len(traffic.edge_travel_time_s),
            "switch_controller_count": feasibility.switch_controller_count,
            "powered_switch_controller_count": feasibility.powered_switch_controller_count,
            "uav_relay_used": int(feasibility.uav_relay_used),
            "uav_predeployed": int(
                predeployed_for_action
                or (
                    applied
                    and action.action_type == "predeploy_uav_relay"
                    and communication_mode == "uav_relay"
                )
            ),
            "predeployed_uav_target_count": len(self._state.predeployed_uav_targets),
            "predeployed_uav_targets": ";".join(
                sorted(self._state.predeployed_uav_targets)
            ),
            "traffic_feasible": int(feasibility.traffic_feasible),
            "resource_energy_feasible": int(feasibility.resource_energy_feasible),
            "resource_energy_required_kwh": feasibility.resource_energy_required_kwh,
            "resource_energy_remaining_kwh": self._state.resource_energy_remaining_kwh,
            "v2g_availability_factor": self._v2g_availability_factor(),
            "v2g_support_kw": v2g_support_kw,
            "v2g_dispatchable_ev_count": self._dispatchable_v2g_vehicle_count(),
            "v2g_available_energy_capacity_kwh": self._state.v2g_available_energy_capacity_kwh,
            "v2g_energy_feasible": int(feasibility.v2g_energy_feasible),
            "v2g_energy_required_kwh": v2g_energy_required_kwh,
            "v2g_energy_remaining_kwh": self._state.v2g_energy_remaining_kwh,
            "predeployment_quality_index": self.scenario.disaster.predeployment_quality_index,
            "disaster_category": self.scenario.disaster.disaster_category,
            "control_paradigm": self.scenario.disaster.control_paradigm,
            "terminal_restoration_bonus": (
                TERMINAL_RESTORATION_BONUS if terminal_restoration else 0.0
            ),
        }
        self._state.trace_rows.append(
            {
                "step_index": self._state.step_index,
                "time_min": self._state.time_min,
                **info,
                "remaining_failed_edge_count": len(self._remaining_failed_edges()),
                "active_failed_edge_count": len(self._remaining_failed_edges()),
                "latent_failed_edge_count": len(self._latent_failed_edges()),
                "restored_failed_edge_count": len(self._state.restored_pdn_edges),
            }
        )
        terminated = terminal_restoration
        truncated = self._state.step_index >= self.max_steps and not terminated
        return self.observe(), reward, terminated, truncated, info

    def trace_rows(self) -> list[dict[str, Any]]:
        return list(self._state.trace_rows)

    def _remaining_failed_edges(self) -> set[str]:
        return self._active_failed_edges()

    def _unrestored_failed_edges(self) -> set[str]:
        return set(self.scenario.failed_pdn_edges) - self._state.restored_pdn_edges

    def _active_failed_edges(self) -> set[str]:
        release_steps = self.scenario.disaster.failure_release_step_by_edge
        return {
            edge_id
            for edge_id in self._unrestored_failed_edges()
            if int(release_steps.get(edge_id, 0)) <= self._state.step_index
        }

    def _latent_failed_edges(self) -> set[str]:
        return self._unrestored_failed_edges() - self._active_failed_edges()

    def _closed_pdn_edges(self) -> set[str]:
        remaining_failed = self._remaining_failed_edges()
        closed = {
            edge.edge_id
            for edge in self.scenario.pdn_edges
            if edge.normal_switch_state == "closed" and edge.edge_id not in remaining_failed
        }
        closed.update(self._state.restored_pdn_edges)
        return closed

    def _restored_load_fraction(self) -> float:
        total = sum(max(0.0, node.active_power_kw) for node in self.scenario.pdn_nodes)
        if total <= 0.0:
            return 0.0
        restored = total - sum(
            failed_edge_restoration_load_kw(self.scenario, edge_id)
            for edge_id in self._remaining_failed_edges()
        )
        return max(0.0, min(1.0, restored / total))

    def _active_mess_support_kw(self) -> float:
        active_units = min(
            len(self._state.restored_pdn_edges), max(0, self.scenario.fleet.mess_count)
        )
        return round(active_units * self.scenario.fleet.mess_unit_power_kw, 6)

    def _active_v2g_support_kw(self) -> float:
        dispatchable_units = self._dispatchable_v2g_vehicle_count()
        active_units = min(len(self._state.restored_pdn_edges), dispatchable_units)
        power_cap_kw = active_units * max(0.0, self.scenario.fleet.ev_unit_power_kw)
        if self.step_minutes <= 0.0:
            return round(power_cap_kw, 6)
        energy_cap_kw = self._state.v2g_energy_remaining_kwh * 60.0 / self.step_minutes
        return round(min(power_cap_kw, max(0.0, energy_cap_kw)), 6)

    def _dispatchable_v2g_vehicle_count(self) -> int:
        return dispatchable_v2g_vehicle_count_at(
            self.scenario.fleet,
            self._state.time_min,
        )

    def _v2g_availability_factor(self) -> float:
        return ev_availability_factor_at(
            self.scenario.fleet,
            self._state.time_min,
        )

    def _v2g_energy_capacity_kwh(self, *, time_min: float | None = None) -> float:
        return v2g_energy_capacity_kwh_at(
            self.scenario.fleet,
            self._state.time_min if time_min is None else time_min,
        )

    def _sync_v2g_availability(self) -> None:
        previous_capacity = max(0.0, self._state.v2g_available_energy_capacity_kwh)
        capacity = self._v2g_energy_capacity_kwh()
        remaining = max(0.0, self._state.v2g_energy_remaining_kwh)
        if capacity > previous_capacity:
            remaining += capacity - previous_capacity
        else:
            remaining = min(remaining, capacity)
        self._state.v2g_available_energy_capacity_kwh = capacity
        self._state.v2g_energy_remaining_kwh = round(
            min(remaining, capacity),
            6,
        )

    def _cross_layer_feasibility(
        self,
        *,
        target_id: str,
        communication: CommunicationStepResult,
        traffic: TrafficStepResult,
        communication_mode: str,
        action_duration_min: float,
    ) -> CrossLayerFeasibility:
        projected_restored_count = len(self._state.restored_pdn_edges)
        if target_id in self._remaining_failed_edges():
            projected_restored_count += 1
        projected_mess_support_kw = (
            min(projected_restored_count, max(0, self.scenario.fleet.mess_count))
            * self.scenario.fleet.mess_unit_power_kw
        )
        projected_v2g_support_kw = (
            min(projected_restored_count, self._dispatchable_v2g_vehicle_count())
            * max(0.0, self.scenario.fleet.ev_unit_power_kw)
        )
        return evaluate_cross_layer_feasibility(
            self.scenario,
            target_pdn_edge_id=target_id,
            closed_pdn_edges=self._closed_pdn_edges(),
            communication_control_available=communication.control_available,
            traffic_result=traffic,
            resource_energy_remaining_kwh=self._state.resource_energy_remaining_kwh,
            projected_mess_support_kw=projected_mess_support_kw,
            v2g_energy_remaining_kwh=self._state.v2g_energy_remaining_kwh,
            projected_v2g_support_kw=projected_v2g_support_kw,
            step_minutes=action_duration_min,
            requested_uav_relay=communication_mode == "uav_relay",
        )

    def _wait_feasibility(
        self,
        traffic: TrafficStepResult,
        *,
        communication: CommunicationStepResult,
        communication_mode: str,
    ) -> CrossLayerFeasibility:
        traffic_ok = str(traffic.status) == "ok"
        communication_ok = bool(communication.control_available)
        blockers: list[str] = []
        if not traffic_ok:
            blockers.append("traffic_support_unavailable")
        if not communication_ok:
            blockers.extend(communication.blockers or ("communication_control_unavailable",))
        relay_requested = str(communication_mode) == "uav_relay"
        return CrossLayerFeasibility(
            feasible=traffic_ok and communication_ok,
            blockers=tuple(blockers),
            switch_controller_count=1 if relay_requested else 0,
            powered_switch_controller_count=1 if relay_requested and communication_ok else 0,
            uav_relay_used=relay_requested and communication_ok,
            traffic_feasible=traffic_ok,
            resource_energy_feasible=True,
            resource_energy_required_kwh=0.0,
            resource_energy_remaining_kwh=self._state.resource_energy_remaining_kwh,
            v2g_energy_feasible=True,
            v2g_energy_required_kwh=0.0,
            v2g_energy_remaining_kwh=self._state.v2g_energy_remaining_kwh,
        )

    @staticmethod
    def _reward(
        *,
        restored_load_gain_kw: float,
        ac: ACPowerResult,
        traffic: TrafficStepResult,
        communication: CommunicationStepResult,
        applied: bool,
        weighted_restoration_value_kw: float,
        action_duration_min: float,
        terminal_restoration: bool,
    ) -> float:
        restoration_value_kw = (
            weighted_restoration_value_kw
            if weighted_restoration_value_kw > 0.0
            else restored_load_gain_kw
        )
        reward = restoration_value_kw * RESTORATION_REWARD_SCALE
        reward -= BLOCKED_ACTION_PENALTY if not applied else 0.0
        if ac.min_voltage_pu is not None and ac.min_voltage_pu < 0.95:
            reward -= (0.95 - ac.min_voltage_pu) * VOLTAGE_VIOLATION_PENALTY
        if ac.max_line_loading_pct is not None and ac.max_line_loading_pct > 100.0:
            reward -= (
                ac.max_line_loading_pct - 100.0
            ) * LINE_LOADING_VIOLATION_PENALTY
        reward -= ClosedLoopPTINEnv._shed_energy_kwh(
            ac=ac,
            action_duration_min=action_duration_min,
        ) * SHED_ENERGY_PENALTY_PER_KWH
        reward -= max(0.0, action_duration_min) * ACTION_DURATION_PENALTY_PER_MIN
        reward -= (
            max(0.0, 1.0 - communication.delivery_rate) * PACKET_LOSS_PENALTY
        )
        reward -= (
            max(0.0, traffic.mean_travel_time_s - 60.0)
            * TRAVEL_TIME_PENALTY_PER_S
        )
        reward -= (
            max(0.0, communication.mean_delay_ms - 250.0)
            * COMM_DELAY_PENALTY_PER_MS
        )
        reward -= CONTROL_UNAVAILABLE_PENALTY if not communication.control_available else 0.0
        reward += TERMINAL_RESTORATION_BONUS if terminal_restoration else 0.0
        return reward

    @staticmethod
    def _shed_energy_kwh(
        *,
        ac: ACPowerResult,
        action_duration_min: float,
    ) -> float:
        shed_load_kw = ac.shed_load_kw if ac.shed_load_kw is not None else 0.0
        return max(0.0, shed_load_kw) * max(0.0, action_duration_min) / 60.0

    @staticmethod
    def _communication_mode(action: ClosedLoopAction) -> str:
        mode = str(action.metadata.get("communication_mode") or "direct")
        return mode if mode in {"uav_relay", "dual_channel"} else "direct"

    def _communication_step(
        self,
        *,
        time_min: float,
        target_id: str,
        communication_mode: str,
        target_robust_travel_time_s: float,
        predeployed_uav_relay: bool = False,
    ) -> CommunicationStepResult:
        kwargs = {
            "time_min": time_min,
            "target_id": target_id,
            "communication_mode": communication_mode,
            "target_robust_travel_time_s": target_robust_travel_time_s,
            "predeployed_uav_relay": predeployed_uav_relay,
        }
        signature = inspect.signature(self.communication_adapter.step)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return self.communication_adapter.step(**kwargs)
        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        return self.communication_adapter.step(**accepted)

    def _action_duration_min(
        self,
        *,
        action: ClosedLoopAction,
        communication: CommunicationStepResult,
        communication_mode: str,
        target_robust_travel_time_s: float,
    ) -> float:
        if action.action_type in {"wait", "predeploy_uav_relay"}:
            return round(max(0.0, self.step_minutes), 6)
        route_minutes = max(0.0, target_robust_travel_time_s) / 60.0
        relay_minutes = 0.0
        if communication_mode == "uav_relay":
            if action.target_id in self._state.predeployed_uav_targets:
                relay_minutes = 0.25 + min(2.0, route_minutes * 0.15)
            else:
                relay_minutes = 1.0 + min(6.0, route_minutes * 0.5)
        elif communication_mode == "dual_channel":
            relay_minutes = (
                0.25
                if action.target_id in self._state.predeployed_uav_targets
                else DUAL_CHANNEL_COORDINATION_MIN
            )
        retry_minutes = 0.0 if communication.control_available else 0.5 * self.step_minutes
        return round(max(0.0, self.step_minutes) + route_minutes + relay_minutes + retry_minutes, 6)
