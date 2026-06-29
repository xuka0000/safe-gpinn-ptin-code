from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import PTINScenario


@dataclass(frozen=True)
class ACPowerResult:
    status: str
    solver_available: bool
    opf_attempted: bool
    opf_converged: bool | None
    power_flow_converged: bool | None
    min_voltage_pu: float | None
    max_voltage_pu: float | None
    max_line_loading_pct: float | None
    requested_load_kw: float | None = None
    served_load_kw: float | None = None
    shed_load_kw: float | None = None
    mobile_support_capacity_kw: float = 0.0
    mobile_dispatch_kw: float | None = None
    blockers: tuple[str, ...] = field(default_factory=tuple)


class ClosedLoopACPowerAdapter:
    def __init__(
        self,
        *,
        module_loader: Callable[[str], Any] | None = None,
        source_node_id: str = "799",
    ) -> None:
        self.module_loader = module_loader or importlib.import_module
        self.source_node_id = source_node_id
        self._result_cache: dict[tuple[Any, ...], ACPowerResult] = {}

    def run_ac_opf_or_pf(
        self,
        scenario: PTINScenario,
        *,
        closed_pdn_edges: set[str],
        restored_load_fraction: float,
        mess_support_kw: float = 0.0,
        v2g_support_kw: float = 0.0,
    ) -> ACPowerResult:
        cache_key = (
            scenario.scenario_id,
            tuple(sorted(closed_pdn_edges)),
            round(float(restored_load_fraction), 8),
            round(float(mess_support_kw), 6),
            round(float(v2g_support_kw), 6),
        )
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached
        result: ACPowerResult
        try:
            pp = self.module_loader("pandapower")
        except Exception as exc:
            result = ACPowerResult(
                status="blocked_dependency_missing",
                solver_available=False,
                opf_attempted=False,
                opf_converged=None,
                power_flow_converged=None,
                min_voltage_pu=None,
                max_voltage_pu=None,
                max_line_loading_pct=None,
                blockers=(f"pandapower_import_failed:{type(exc).__name__}",),
            )
            self._result_cache[cache_key] = result
            return result

        try:
            net, build_summary = self._build_network(
                pp,
                scenario,
                closed_pdn_edges=closed_pdn_edges,
                restored_load_fraction=restored_load_fraction,
                mess_support_kw=mess_support_kw,
                v2g_support_kw=v2g_support_kw,
            )
        except Exception as exc:
            result = ACPowerResult(
                status="failed_case_build",
                solver_available=True,
                opf_attempted=False,
                opf_converged=None,
                power_flow_converged=None,
                min_voltage_pu=None,
                max_voltage_pu=None,
                max_line_loading_pct=None,
                blockers=(f"case_build_failed:{type(exc).__name__}",),
            )
            self._result_cache[cache_key] = result
            return result

        blockers: list[str] = []
        try:
            pp.runopp(net, calculate_voltage_angles=True, init="flat", numba=False)
            result = ACPowerResult(
                status="opf_ok",
                solver_available=True,
                opf_attempted=True,
                opf_converged=True,
                power_flow_converged=None,
                min_voltage_pu=_column_min(getattr(net, "res_bus", None), "vm_pu"),
                max_voltage_pu=_column_max(getattr(net, "res_bus", None), "vm_pu"),
                max_line_loading_pct=_column_max(
                    getattr(net, "res_line", None), "loading_percent"
                ),
                requested_load_kw=build_summary["requested_load_kw"],
                served_load_kw=_served_load_kw(net),
                shed_load_kw=_shed_load_kw(build_summary["requested_load_kw"], net),
                mobile_support_capacity_kw=build_summary["mobile_support_capacity_kw"],
                mobile_dispatch_kw=_mobile_dispatch_kw(net),
                blockers=(),
            )
            self._result_cache[cache_key] = result
            return result
        except Exception as exc:
            blockers.append(f"opf_failed:{type(exc).__name__}")

        try:
            pp.runpp(net, algorithm="nr", init="flat", numba=False)
            converged = bool(getattr(net, "converged", False))
            result = ACPowerResult(
                status="opf_fallback_power_flow_ok" if converged else "opf_fallback_power_flow_failed",
                solver_available=True,
                opf_attempted=True,
                opf_converged=False,
                power_flow_converged=converged,
                min_voltage_pu=_column_min(getattr(net, "res_bus", None), "vm_pu"),
                max_voltage_pu=_column_max(getattr(net, "res_bus", None), "vm_pu"),
                max_line_loading_pct=_column_max(
                    getattr(net, "res_line", None), "loading_percent"
                ),
                requested_load_kw=build_summary["requested_load_kw"],
                served_load_kw=_served_load_kw(net),
                shed_load_kw=_shed_load_kw(build_summary["requested_load_kw"], net),
                mobile_support_capacity_kw=build_summary["mobile_support_capacity_kw"],
                mobile_dispatch_kw=_mobile_dispatch_kw(net),
                blockers=tuple(blockers),
            )
            self._result_cache[cache_key] = result
            return result
        except Exception as exc:
            blockers.append(f"power_flow_failed:{type(exc).__name__}")
            result = ACPowerResult(
                status="opf_and_power_flow_failed",
                solver_available=True,
                opf_attempted=True,
                opf_converged=False,
                power_flow_converged=False,
                min_voltage_pu=None,
                max_voltage_pu=None,
                max_line_loading_pct=None,
                blockers=tuple(blockers),
            )
            self._result_cache[cache_key] = result
            return result

    def _build_network(
        self,
        pp: Any,
        scenario: PTINScenario,
        *,
        closed_pdn_edges: set[str],
        restored_load_fraction: float,
        mess_support_kw: float,
        v2g_support_kw: float,
    ) -> tuple[Any, dict[str, float]]:
        net = pp.create_empty_network(sn_mva=10.0)
        bus_lookup = {
            node.node_id: pp.create_bus(
                net,
                vn_kv=node.base_voltage_kv,
                name=node.node_id,
                min_vm_pu=0.95,
                max_vm_pu=1.05,
            )
            for node in scenario.pdn_nodes
        }
        source = self.source_node_id if self.source_node_id in bus_lookup else scenario.pdn_nodes[0].node_id
        energized_nodes = _energized_pdn_node_ids(
            scenario, source_node_id=source, closed_pdn_edges=closed_pdn_edges
        )
        pp.create_ext_grid(net, bus=bus_lookup[source], vm_pu=1.0, name="source")
        _try_create_poly_cost(
            pp,
            net,
            element=0,
            et="ext_grid",
            cp1_eur_per_mw=20.0,
        )

        for edge in scenario.pdn_edges:
            if edge.edge_id not in closed_pdn_edges:
                continue
            if edge.from_node not in energized_nodes or edge.to_node not in energized_nodes:
                continue
            if edge.from_node not in bus_lookup or edge.to_node not in bus_lookup:
                continue
            line_idx = pp.create_line_from_parameters(
                net,
                from_bus=bus_lookup[edge.from_node],
                to_bus=bus_lookup[edge.to_node],
                length_km=max(edge.length_km, 1.0e-6),
                r_ohm_per_km=max(edge.r_pu * 2.304, 1.0e-6),
                x_ohm_per_km=max(edge.x_pu * 2.304, 1.0e-6),
                c_nf_per_km=0.0,
                max_i_ka=0.2,
                name=edge.edge_id,
            )
            _set_table_value(
                getattr(net, "line", None),
                line_idx,
                "max_loading_percent",
                100.0,
            )

        load_fraction = max(0.0, min(1.0, restored_load_fraction))
        requested_load_kw = 0.0
        for node in scenario.pdn_nodes:
            if node.node_id not in energized_nodes:
                continue
            p_kw = max(0.0, node.active_power_kw * load_fraction)
            q_kvar = max(0.0, node.reactive_power_kvar * load_fraction)
            if p_kw <= 0.0 and q_kvar <= 0.0:
                continue
            requested_load_kw += p_kw
            load_idx = pp.create_load(
                net,
                bus=bus_lookup[node.node_id],
                p_mw=p_kw / 1000.0,
                q_mvar=q_kvar / 1000.0,
                name=f"load_{node.node_id}",
                controllable=True,
                min_p_mw=0.0,
                max_p_mw=p_kw / 1000.0,
                min_q_mvar=0.0,
                max_q_mvar=q_kvar / 1000.0,
            )
            utility = -1200.0 if node.criticality == "critical" else -800.0
            _try_create_poly_cost(
                pp,
                net,
                element=load_idx,
                et="load",
                cp1_eur_per_mw=utility,
            )

        mobile_support_capacity_kw = max(0.0, mess_support_kw) + max(0.0, v2g_support_kw)
        self._add_mobile_support(
            pp,
            net,
            scenario,
            bus_lookup,
            energized_nodes=energized_nodes,
            support_kw=max(0.0, mess_support_kw),
            label="MESS",
            cost_per_mw=12.0,
        )
        self._add_mobile_support(
            pp,
            net,
            scenario,
            bus_lookup,
            energized_nodes=energized_nodes,
            support_kw=max(0.0, v2g_support_kw),
            label="V2G",
            cost_per_mw=8.0,
        )
        return net, {
            "requested_load_kw": round(requested_load_kw, 6),
            "mobile_support_capacity_kw": round(mobile_support_capacity_kw, 6),
        }

    def _add_mobile_support(
        self,
        pp: Any,
        net: Any,
        scenario: PTINScenario,
        bus_lookup: dict[str, Any],
        *,
        energized_nodes: set[str],
        support_kw: float,
        label: str,
        cost_per_mw: float,
    ) -> None:
        if support_kw <= 0.0:
            return
        candidate_nodes = [
            node.node_id
            for node in scenario.pdn_nodes
            if node.node_id in energized_nodes and node.criticality == "critical"
        ]
        if not candidate_nodes:
            candidate_nodes = [node_id for node_id in energized_nodes if node_id in bus_lookup]
        if not candidate_nodes:
            return
        per_node_kw = support_kw / len(candidate_nodes)
        for node_id in candidate_nodes:
            sgen_idx = pp.create_sgen(
                net,
                bus=bus_lookup[node_id],
                p_mw=0.0,
                q_mvar=0.0,
                name=f"{label}_{node_id}",
                controllable=True,
                min_p_mw=0.0,
                max_p_mw=per_node_kw / 1000.0,
                min_q_mvar=-0.2 * per_node_kw / 1000.0,
                max_q_mvar=0.2 * per_node_kw / 1000.0,
            )
            _try_create_poly_cost(
                pp,
                net,
                element=sgen_idx,
                et="sgen",
                cp1_eur_per_mw=cost_per_mw,
            )


def _column_min(table: Any, column: str) -> float | None:
    values = _table_column(table, column)
    if values is None:
        return None
    return round(float(values.min() if hasattr(values, "min") else min(values)), 6)


def _column_max(table: Any, column: str) -> float | None:
    values = _table_column(table, column)
    if values is None:
        return None
    return round(float(values.max() if hasattr(values, "max") else max(values)), 6)


def _column_sum(table: Any, column: str) -> float | None:
    values = _table_column(table, column)
    if values is None:
        return None
    if hasattr(values, "sum"):
        return round(float(values.sum()), 6)
    if hasattr(values, "_values"):
        return round(float(sum(values._values)), 6)
    return round(float(sum(values)), 6)


def _table_column(table: Any, column: str) -> Any:
    if table is None:
        return None
    try:
        return table[column]
    except (TypeError, KeyError, AttributeError):
        return getattr(table, column, None)


def _served_load_kw(net: Any) -> float | None:
    served_mw = _column_sum(getattr(net, "res_load", None), "p_mw")
    if served_mw is None:
        return None
    return round(served_mw * 1000.0, 6)


def _shed_load_kw(requested_load_kw: float | None, net: Any) -> float | None:
    served = _served_load_kw(net)
    if requested_load_kw is None or served is None:
        return None
    return round(max(0.0, requested_load_kw - served), 6)


def _mobile_dispatch_kw(net: Any) -> float | None:
    dispatch_mw = _column_sum(getattr(net, "res_sgen", None), "p_mw")
    if dispatch_mw is None:
        return None
    return round(dispatch_mw * 1000.0, 6)


def _energized_pdn_node_ids(
    scenario: PTINScenario, *, source_node_id: str, closed_pdn_edges: set[str]
) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for edge in scenario.pdn_edges:
        if edge.edge_id not in closed_pdn_edges:
            continue
        adjacency.setdefault(edge.from_node, set()).add(edge.to_node)
        adjacency.setdefault(edge.to_node, set()).add(edge.from_node)

    visited: set[str] = set()
    stack = [source_node_id]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(sorted(adjacency.get(node_id, set()) - visited))
    return visited


def _try_create_poly_cost(pp: Any, net: Any, **kwargs: Any) -> None:
    create_poly_cost = getattr(pp, "create_poly_cost", None)
    if create_poly_cost is None:
        return
    create_poly_cost(net, **kwargs)


def _set_table_value(table: Any, index: Any, column: str, value: Any) -> None:
    if table is None:
        return
    try:
        table.at[index, column] = value
    except Exception:
        return
