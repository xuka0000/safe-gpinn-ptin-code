from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHOD_LABELS = {
    "safe_gpinn": "SAFE-GPINN",
    "maddpg": "MADDPG~\\cite{lowe2017maddpg}",
    "mappo": "MAPPO~\\cite{yu2022mappo}",
    "qmix": "QMIX~\\cite{rashid2018qmix}",
    "vdn": "VDN~\\cite{sunehag2018vdn}",
    "happo_hatrpo": "HAPPO/HATRPO~\\cite{kuba2022trustregionmarl}",
    "mat": "MAT~\\cite{wen2022mat}",
    "rolling_mpc": "RH-MPC~\\cite{rawlings2017mpc}",
    "two_stage_exact": "Two-stage MILP~\\cite{zhong2025ptin}",
    "diffusion_policy": "Diffusion planner~\\cite{janner2022diffuser}",
    "cql": "CQL~\\cite{kumar2020cql}",
    "greedy": "Greedy",
    "iql": "IQL~\\cite{kostrikov2022iql}",
    "td3_bc": "TD3~\\cite{fujimoto2021td3bc}",
    "load_gain": "Load-gain",
}

METHOD_CLASSES = {
    "safe_gpinn": "MARL",
    "maddpg": "MARL",
    "mappo": "MARL",
    "qmix": "MARL",
    "vdn": "MARL",
    "happo_hatrpo": "MARL",
    "mat": "MARL",
    "rolling_mpc": "Planning",
    "two_stage_exact": "Planning",
    "diffusion_policy": "Planning",
    "cql": "Offline RL",
    "td3_bc": "Offline RL",
    "iql": "Offline RL",
    "greedy": "Heuristic",
    "load_gain": "Heuristic",
}

METHOD_ORDER = (
    "safe_gpinn",
    "maddpg",
    "mappo",
    "qmix",
    "vdn",
    "happo_hatrpo",
    "mat",
    "cql",
    "td3_bc",
    "iql",
    "diffusion_policy",
    "two_stage_exact",
    "rolling_mpc",
    "greedy",
    "load_gain",
)

METRIC_SLOT_COUNT = 12
METHOD_COL_WIDTH = "0.17\\textwidth"
CLASS_COL_WIDTH = "0.10\\textwidth"
MetricSpec = tuple[str, int, float, str | None, str | None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="closed_loop_physical_main_policy_table")
    args = parser.parse_args()

    trace_rows = _read_csv(args.trace_csv)
    eval_rows = _read_csv(args.eval_csv)
    summary_rows = summarize_physical_policy_metrics(trace_rows, eval_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.label}.csv"
    tex_path = args.output_dir / f"{args.label}.tex"
    _write_csv(csv_path, summary_rows)
    tex_path.write_text(_latex_table(summary_rows), encoding="utf-8")
    print(f"csv={csv_path}")
    print(f"tex={tex_path}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def summarize_physical_policy_metrics(
    trace_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    trace_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        trace_by_key[(str(row.get("shard", "")), str(row.get("seed", "")), row["method"])].append(row)
    eval_by_key = {
        (str(row.get("shard", "")), str(row.get("seed", "")), row["method"]): row
        for row in eval_rows
    }

    per_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    for key, rows in sorted(trace_by_key.items()):
        _shard, _seed, method = key
        rows = sorted(rows, key=lambda row: _float(row, "step_index"))
        eval_row = eval_by_key.get(key, {})
        final = rows[-1]
        durations = [
            max(_float(row, "action_duration_min", 5.0), 0.0)
            for row in rows
        ]
        served_ratios = []
        for row in rows:
            requested = _float(row, "requested_load_kw")
            served = _float(row, "served_load_kw")
            shed = _float(row, "shed_load_kw")
            denominator = requested if requested > 0.0 else served + shed
            if denominator > 0.0:
                served_ratios.append(served / denominator)
        full_restoration = 1.0 if _truthy(eval_row.get("terminated")) and _float(final, "remaining_failed_edge_count") <= 0 else 0.0
        per_method[method].append(
            {
                "terminal_time_min": _float(final, "time_min"),
                "total_reward": _float(eval_row, "total_reward"),
                "full_restoration_rate": full_restoration,
                "restored_edges": _float(final, "restored_failed_edge_count"),
                "remaining_failed_edges": _float(final, "remaining_failed_edge_count"),
                "mean_served_ratio": float(np.mean(served_ratios)) if served_ratios else 0.0,
                "shed_energy_kwh": sum(
                    _float(row, "shed_load_kw") * duration / 60.0
                    for row, duration in zip(rows, durations)
                ),
                "mean_packet_delivery_rate": float(np.mean([_float(row, "packet_delivery_rate") for row in rows])),
                "mean_delay_ms": float(np.mean([_float(row, "mean_delay_ms") for row in rows])),
                "control_available_rate": float(np.mean([1.0 if _truthy(row.get("control_available")) else 0.0 for row in rows])),
                "blocked_attempts": float(sum(1 for row in rows if not _truthy(row.get("applied")))),
                "relay_action_rate": float(np.mean([1.0 if str(row.get("communication_mode")) == "uav_relay" else 0.0 for row in rows])),
                "powered_controller_rate": float(
                    np.mean(
                        [
                            _float(row, "powered_switch_controller_count")
                            / max(_float(row, "switch_controller_count"), 1.0)
                            for row in rows
                        ]
                    )
                ),
                "mean_target_traffic_edge_count": float(np.mean([_float(row, "target_traffic_edge_count") for row in rows])),
                "mean_target_travel_time_s": float(np.mean([_float(row, "target_travel_time_s") for row in rows])),
                "mean_target_robust_travel_time_s": float(np.mean([_float(row, "target_robust_travel_time_s") for row in rows])),
                "mean_action_duration_min": float(np.mean(durations)),
                "traffic_feasible_rate": float(np.mean([1.0 if _truthy(row.get("traffic_feasible")) else 0.0 for row in rows])),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = per_method.get(method, [])
        if not method_rows:
            continue
        out: dict[str, Any] = {
            "method": METHOD_LABELS.get(method, method),
            "category": METHOD_CLASSES.get(method, "Other"),
        }
        out["seed_count"] = len(method_rows)
        for key in (
            "total_reward",
            "terminal_time_min",
            "full_restoration_rate",
            "restored_edges",
            "remaining_failed_edges",
            "mean_served_ratio",
            "shed_energy_kwh",
            "mean_packet_delivery_rate",
            "mean_delay_ms",
            "control_available_rate",
            "blocked_attempts",
            "relay_action_rate",
            "powered_controller_rate",
            "mean_target_traffic_edge_count",
            "mean_target_travel_time_s",
            "mean_target_robust_travel_time_s",
            "mean_action_duration_min",
            "traffic_feasible_rate",
        ):
            values = np.asarray([row[key] for row in method_rows], dtype=float)
            out[key] = float(values.mean())
            out[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


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


def _seed_count_text(rows: list[dict[str, Any]]) -> str:
    counts = sorted({int(row.get("seed_count", 0)) for row in rows if row.get("seed_count")})
    if len(counts) == 1:
        return f"{counts[0]}"
    return "matched"


def _pm(
    row: dict[str, Any],
    key: str,
    *,
    digits: int = 1,
    scale: float = 1.0,
    style: str | None = None,
) -> str:
    value = float(row[key]) * scale
    std = float(row.get(f"{key}_std", 0.0)) * scale
    mean = f"{value:.{digits}f}"
    spread = f"{{\\scriptsize $\\pm$ {std:.{digits}f}}}"
    if style == "best":
        mean = f"\\textbf{{{mean}}}"
    elif style == "second":
        mean = f"\\underline{{{mean}}}"
    return f"{mean} {spread}"


def _passes_rank_gate(row: dict[str, Any], gate: str | None) -> bool:
    if gate == "full_restoration":
        return math.isclose(float(row.get("full_restoration_rate", 0.0)), 1.0, abs_tol=1e-9)
    return True


def _rank_styles(
    rows: list[dict[str, Any]],
    values: list[MetricSpec],
) -> dict[tuple[int, str], str]:
    styles: dict[tuple[int, str], str] = {}
    for key, digits, scale, direction, gate in values:
        if direction is None:
            continue
        candidates = [
            (index, round(float(row[key]) * scale, digits))
            for index, row in enumerate(rows)
            if _passes_rank_gate(row, gate)
        ]
        ranked_values = sorted(
            {value for _index, value in candidates},
            reverse=direction == "up",
        )
        if len(ranked_values) <= 1:
            continue
        best = ranked_values[0]
        second = ranked_values[1]
        for index, value in candidates:
            if math.isclose(value, best, abs_tol=10 ** (-(digits + 1))):
                styles[(index, key)] = "best"
            elif math.isclose(value, second, abs_tol=10 ** (-(digits + 1))):
                styles[(index, key)] = "second"
    return styles


def _class_cells(rows: list[dict[str, Any]]) -> list[str]:
    cells = [""] * len(rows)
    start = 0
    while start < len(rows):
        category = str(rows[start]["category"])
        end = start + 1
        while end < len(rows) and str(rows[end]["category"]) == category:
            end += 1
        span = end - start
        if span == 1:
            cells[start] = f"\\multicolumn{{1}}{{c}}{{{category}}}"
        else:
            cells[start] = f"\\multirow{{{span}}}{{{CLASS_COL_WIDTH}}}{{\\centering {category}}}"
        start = end
    return cells


def _metric_spans(metric_count: int) -> list[int]:
    if METRIC_SLOT_COUNT % metric_count:
        raise ValueError(f"{metric_count} metrics cannot evenly fill {METRIC_SLOT_COUNT} slots")
    return [METRIC_SLOT_COUNT // metric_count] * metric_count


def _append_panel_section(
    lines: list[str],
    rows: list[dict[str, Any]],
    *,
    title: str,
    headers: list[str],
    values: list[MetricSpec],
    first: bool = False,
) -> None:
    class_cells = _class_cells(rows)
    metric_spans = _metric_spans(len(headers))
    rank_styles = _rank_styles(rows, values)
    header_cells = [
        f"\\multicolumn{{{span}}}{{c}}{{{header}}}"
        for header, span in zip(headers, metric_spans)
    ]
    if not first:
        lines.append("\\midrule[0.8pt]")
    panel_title = (
        f"Method & Class & \\multicolumn{{{METRIC_SLOT_COUNT}}}{{c}}{{{title}}}\\\\"
        if first
        else f"& & \\multicolumn{{{METRIC_SLOT_COUNT}}}{{c}}{{{title}}}\\\\"
    )
    lines.extend(
        [
            panel_title,
            f"\\cmidrule(lr){{3-{METRIC_SLOT_COUNT + 2}}}",
            f"& & {' & '.join(header_cells)}\\\\",
            "\\midrule" if first else f"\\cmidrule(lr){{1-{METRIC_SLOT_COUNT + 2}}}",
        ]
    )
    for index, row in enumerate(rows):
        metric_cells = [
            f"\\multicolumn{{{span}}}{{c}}{{{_pm(row, key, digits=digits, scale=scale, style=rank_styles.get((index, key)))}}}"
            for (key, digits, scale, _direction, _gate), span in zip(values, metric_spans)
        ]
        lines.append(
            f"{row['method']} & {class_cells[index]} & "
            f"{' & '.join(metric_cells)} \\\\"
        )
        if index < len(rows) - 1 and str(row["category"]) != str(rows[index + 1]["category"]):
            lines.append(f"\\cmidrule(lr){{1-{METRIC_SLOT_COUNT + 2}}}")


def _latex_table(rows: list[dict[str, Any]]) -> str:
    seed_count = _seed_count_text(rows)
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        f"\\caption{{Closed loop PTIN policy comparison with action dependent recovery indicators over {seed_count} seeds.}}",
        "\\label{tab:closed-loop-policy-comparison}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.45pt}",
        "\\renewcommand{\\arraystretch}{0.86}",
        f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}p{{{METHOD_COL_WIDTH}}}p{{{CLASS_COL_WIDTH}}}*{{{METRIC_SLOT_COUNT}}}{{r}}@{{}}}}",
        "\\toprule",
    ]

    _append_panel_section(
        lines,
        rows,
        title="Power grid",
        headers=["$R\\uparrow$", "FR $\\uparrow$", "Failed $\\downarrow$", "Served (\\%) $\\uparrow$", "$E^{shed}$ (kWh) $\\downarrow$", "Edges $\\uparrow$"],
        values=[
            ("total_reward", 1, 1.0, "up", None),
            ("full_restoration_rate", 2, 1.0, "up", None),
            ("remaining_failed_edges", 1, 1.0, "down", None),
            ("mean_served_ratio", 2, 100.0, "up", "full_restoration"),
            ("shed_energy_kwh", 1, 1.0, "down", "full_restoration"),
            ("restored_edges", 1, 1.0, "up", None),
        ],
        first=True,
    )
    _append_panel_section(
        lines,
        rows,
        title="Transportation",
        headers=["$T^{end}$ (min) $\\downarrow$", "Roads", "Travel (s) $\\downarrow$", "Robust (s) $\\downarrow$", "Action (min) $\\downarrow$", "Feas. (\\%) $\\uparrow$"],
        values=[
            ("terminal_time_min", 1, 1.0, None, None),
            ("mean_target_traffic_edge_count", 1, 1.0, None, None),
            ("mean_target_travel_time_s", 1, 1.0, None, None),
            ("mean_target_robust_travel_time_s", 1, 1.0, None, None),
            ("mean_action_duration_min", 2, 1.0, None, None),
            ("traffic_feasible_rate", 1, 100.0, None, None),
        ],
    )
    _append_panel_section(
        lines,
        rows,
        title="Communication",
        headers=["PDR (\\%) $\\uparrow$", "Delay (ms) $\\downarrow$", "Ctrl. (\\%) $\\uparrow$", "Blocked $\\downarrow$", "Relay (\\%)", "Powered ctrl. (\\%) $\\uparrow$"],
        values=[
            ("mean_packet_delivery_rate", 2, 100.0, None, None),
            ("mean_delay_ms", 1, 1.0, None, None),
            ("control_available_rate", 2, 100.0, None, None),
            ("blocked_attempts", 1, 1.0, None, None),
            ("relay_action_rate", 1, 100.0, None, None),
            ("powered_controller_rate", 1, 100.0, None, None),
        ],
    )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\vspace{1pt}",
            "\\begin{flushleft}",
            "\\scriptsize Values are mean $\\pm$ standard deviation over matched seeds. Bold and underlined means mark the best and second-best valid values only in the power-grid outcome block. Served load and shedding energy are ranked only among methods with FR equal to one. Transportation and communication metrics report operating cost and support conditions rather than independent dominance. FR denotes full restoration rate, $E^{shed}$ load shedding energy, $R$ total reward, $T^{end}$ terminal recovery time, and PDR packet delivery rate. Roads, Travel, Robust, Ctrl., Powered ctrl. and Relay denote target traffic edges, target travel time, target robust travel time, control availability, powered switch-controller availability and UAV-relay use.",
            "\\end{flushleft}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
