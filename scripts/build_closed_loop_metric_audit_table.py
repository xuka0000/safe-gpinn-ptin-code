from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ptin_sim.closed_loop.dependencies import (
    failed_edge_restoration_load_kw,
    weighted_restoration_value_kw,
)
from ptin_sim.closed_loop.scenario import load_closed_loop_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build restoration-conditioned service metric audit table."
    )
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--scenario-dir", default="data/scenario_reconstruction_progressive_stress_v2")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = summarize_metric_audit(
        _read_csv(args.trace_csv),
        _read_csv(args.eval_csv),
        scenario_dir=args.scenario_dir,
    )
    _write_csv(args.output_csv, rows)
    print(f"csv={args.output_csv}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _served_ratio(row: dict[str, Any]) -> float:
    requested = _float(row, "requested_load_kw")
    served = _float(row, "served_load_kw")
    shed = _float(row, "shed_load_kw")
    denominator = requested if requested > 0.0 else served + shed
    return served / denominator if denominator > 0.0 else 0.0


def summarize_metric_audit(
    trace_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    *,
    scenario_dir: str | Path,
) -> list[dict[str, Any]]:
    scenario = load_closed_loop_scenario(scenario_dir)
    failed_edges = tuple(sorted(scenario.failed_pdn_edges))
    release_steps = scenario.disaster.failure_release_step_by_edge
    raw_load_kw = {
        edge_id: failed_edge_restoration_load_kw(scenario, edge_id)
        for edge_id in failed_edges
    }
    weighted_load_kw = {
        edge_id: weighted_restoration_value_kw(scenario, edge_id)
        for edge_id in failed_edges
    }
    total_failed_load_kw = sum(raw_load_kw.values())
    total_failed_weighted_kw = sum(weighted_load_kw.values())

    trace_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        trace_by_key[
            (str(row.get("shard", "")), str(row.get("seed", "")), row["method"])
        ].append(row)
    eval_by_key = {
        (str(row.get("shard", "")), str(row.get("seed", "")), row["method"]): row
        for row in eval_rows
    }

    per_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    for key, rows in sorted(trace_by_key.items()):
        _shard, _seed, method = key
        rows = sorted(rows, key=lambda row: _float(row, "step_index"))
        eval_row = eval_by_key.get(key, {})
        restored: set[str] = set()
        duration_total = 0.0
        served_ratio_time = 0.0
        restoration_weighted_served_time = 0.0
        active_load_time = 0.0
        active_unrestored_load_time = 0.0
        active_weighted_time = 0.0
        active_unrestored_weighted_time = 0.0
        shed_energy_kwh = 0.0

        for row in rows:
            if (
                _truthy(row.get("applied"))
                and row.get("action_type") == "restore_pdn_edge"
                and row.get("target_id") in raw_load_kw
            ):
                restored.add(str(row["target_id"]))

            step_index = int(_float(row, "step_index"))
            active_edges = {
                edge_id
                for edge_id in failed_edges
                if int(release_steps.get(edge_id, 0)) <= step_index
            }
            active_load = sum(raw_load_kw[edge_id] for edge_id in active_edges)
            active_weighted = sum(weighted_load_kw[edge_id] for edge_id in active_edges)
            unrestored_active = active_edges - restored
            active_unrestored_load = sum(
                raw_load_kw[edge_id] for edge_id in unrestored_active
            )
            active_unrestored_weighted = sum(
                weighted_load_kw[edge_id] for edge_id in unrestored_active
            )
            restored_active_load = max(active_load - active_unrestored_load, 0.0)
            active_restored_fraction = (
                restored_active_load / active_load if active_load > 0.0 else 0.0
            )
            duration = max(_float(row, "action_duration_min", 5.0), 0.0)
            ratio = _served_ratio(row)

            duration_total += duration
            served_ratio_time += ratio * duration
            restoration_weighted_served_time += (
                ratio * active_restored_fraction * duration
            )
            active_load_time += active_load * duration / 60.0
            active_unrestored_load_time += active_unrestored_load * duration / 60.0
            active_weighted_time += active_weighted * duration / 60.0
            active_unrestored_weighted_time += (
                active_unrestored_weighted * duration / 60.0
            )
            shed_energy_kwh += _float(row, "shed_load_kw") * duration / 60.0

        restored_load = sum(raw_load_kw[edge_id] for edge_id in restored)
        restored_weighted = sum(weighted_load_kw[edge_id] for edge_id in restored)
        active_backlog_reduction = (
            1.0 - active_unrestored_load_time / active_load_time
            if active_load_time > 0.0
            else 0.0
        )
        weighted_backlog_reduction = (
            1.0 - active_unrestored_weighted_time / active_weighted_time
            if active_weighted_time > 0.0
            else 0.0
        )
        per_method[method].append(
            {
                "total_reward": _float(eval_row, "total_reward"),
                "terminal_time_min": _float(rows[-1], "time_min") if rows else 0.0,
                "mean_served_ratio": (
                    served_ratio_time / duration_total if duration_total > 0.0 else 0.0
                ),
                "restoration_weighted_served_ratio": (
                    restoration_weighted_served_time / duration_total
                    if duration_total > 0.0
                    else 0.0
                ),
                "final_restored_load_fraction": (
                    restored_load / total_failed_load_kw
                    if total_failed_load_kw > 0.0
                    else 0.0
                ),
                "final_weighted_restored_fraction": (
                    restored_weighted / total_failed_weighted_kw
                    if total_failed_weighted_kw > 0.0
                    else 0.0
                ),
                "active_unrestored_load_backlog_kwh": active_unrestored_load_time,
                "active_unrestored_weighted_backlog_kwh": active_unrestored_weighted_time,
                "active_backlog_reduction_ratio": active_backlog_reduction,
                "weighted_backlog_reduction_ratio": weighted_backlog_reduction,
                "shed_energy_kwh": shed_energy_kwh,
                "terminated": 1.0 if _truthy(eval_row.get("terminated")) else 0.0,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for method, method_rows in sorted(per_method.items()):
        out: dict[str, Any] = {"method": method, "seed_count": len(method_rows)}
        for metric in (
            "total_reward",
            "terminal_time_min",
            "mean_served_ratio",
            "restoration_weighted_served_ratio",
            "final_restored_load_fraction",
            "final_weighted_restored_fraction",
            "active_unrestored_load_backlog_kwh",
            "active_unrestored_weighted_backlog_kwh",
            "active_backlog_reduction_ratio",
            "weighted_backlog_reduction_ratio",
            "shed_energy_kwh",
            "terminated",
        ):
            values = np.asarray([row[metric] for row in method_rows], dtype=float)
            out[metric] = float(values.mean())
            out[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(out)
    return summary_rows


if __name__ == "__main__":
    main()
