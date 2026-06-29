from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ptin_sim.closed_loop.training import (
    SCENARIO_TREE_BRANCHES,
    _ScenarioTreeBranch,
    _read_csv,
    _scenario_tree_branch_penalty_kw,
    _write_csv,
)


def scaled_scenario_tree_branches(
    uncertainty_scale: float,
) -> tuple[_ScenarioTreeBranch, ...]:
    scale = max(0.0, float(uncertainty_scale))
    branches: list[_ScenarioTreeBranch] = []
    for branch in SCENARIO_TREE_BRANCHES:
        traffic_multiplier = 1.0 + scale * (float(branch.traffic_multiplier) - 1.0)
        packet_delivery_factor = 1.0 - scale * (1.0 - float(branch.packet_delivery_factor))
        resource_factor = 1.0 - scale * (1.0 - float(branch.resource_factor))
        branches.append(
            _ScenarioTreeBranch(
                branch_id=branch.branch_id,
                probability=branch.probability,
                traffic_multiplier=round(max(1.0, traffic_multiplier), 9),
                packet_delivery_factor=round(min(1.0, max(0.0, packet_delivery_factor)), 9),
                resource_factor=round(min(1.0, max(0.0, resource_factor)), 9),
            )
        )
    return tuple(branches)


def scale_scenario_tree_rows(
    rows: Iterable[dict[str, Any]],
    *,
    uncertainty_scale: float,
) -> list[dict[str, Any]]:
    branches = {
        branch.branch_id: branch
        for branch in scaled_scenario_tree_branches(uncertainty_scale)
    }
    scaled_rows: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        branch = branches.get(str(row.get("tree_branch_id", "")))
        if branch is None:
            scaled_rows.append(row)
            continue
        base_return = _float_value(
            row.get("base_return_to_go"),
            _float_value(row.get("return_to_go"), 0.0),
        )
        penalty = _scenario_tree_branch_penalty_kw(row, branch)
        adjusted_return = round(float(base_return) - float(penalty), 9)
        row.update(
            {
                "base_return_to_go": round(float(base_return), 9),
                "return_to_go": adjusted_return,
                "tree_branch_probability": branch.probability,
                "tree_traffic_multiplier": branch.traffic_multiplier,
                "tree_packet_delivery_factor": branch.packet_delivery_factor,
                "tree_resource_factor": branch.resource_factor,
                "tree_branch_penalty_kw": round(float(penalty), 9),
                "tree_branch_return_to_go": adjusted_return,
                "uncertainty_scale": round(float(uncertainty_scale), 9),
            }
        )
        scaled_rows.append(row)
    return scaled_rows


def build_scaled_rollout_datasets(
    *,
    batch_root: Path,
    output_root: Path,
    scales: Iterable[float],
) -> list[dict[str, Any]]:
    shard_rows = _read_shard_processes(batch_root / "shard_processes.csv")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for scale in scales:
        scale_dir = output_root / _scale_dir_name(scale)
        scale_dir.mkdir(parents=True, exist_ok=True)
        for shard_row in shard_rows:
            shard = str(shard_row["shard"])
            source_dir = Path(str(shard_row["output_dir"]))
            source_dataset = source_dir / "closed_loop_rollout_dataset.csv"
            scaled_rows = scale_scenario_tree_rows(
                _read_csv(source_dataset),
                uncertainty_scale=float(scale),
            )
            shard_dir = scale_dir / shard
            shard_dir.mkdir(parents=True, exist_ok=True)
            output_dataset = shard_dir / "scaled_rollout_dataset.csv"
            _write_csv(output_dataset, scaled_rows)
            manifest_rows.append(
                {
                    "scale": float(scale),
                    "scale_dir": str(scale_dir),
                    "shard": shard,
                    "seed": shard_row.get("seed", ""),
                    "source_dataset": str(source_dataset),
                    "output_dataset": str(output_dataset),
                    "row_count": len(scaled_rows),
                }
            )
    manifest = {
        "version": "closed_loop_uncertainty_sensitivity_scaled_rollout_v1",
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "scales": [float(scale) for scale in scales],
        "branches_by_scale": {
            str(float(scale)): [
                {
                    "branch_id": branch.branch_id,
                    "probability": branch.probability,
                    "traffic_multiplier": branch.traffic_multiplier,
                    "packet_delivery_factor": branch.packet_delivery_factor,
                    "resource_factor": branch.resource_factor,
                }
                for branch in scaled_scenario_tree_branches(float(scale))
            ]
            for scale in scales
        },
        "datasets": manifest_rows,
    }
    (output_root / "scaled_rollout_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_root / "scaled_rollout_manifest.csv", manifest_rows)
    return manifest_rows


def _read_shard_processes(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _scale_dir_name(scale: float) -> str:
    return f"scale_{float(scale):.2f}".replace(".", "p")


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_scales(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build uncertainty-scaled closed-loop rollout datasets."
    )
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scales", default="1.0,1.5,2.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_scaled_rollout_datasets(
        batch_root=args.batch_root,
        output_root=args.output_root,
        scales=_parse_scales(args.scales),
    )
    print(json.dumps({"status": "ok", "dataset_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
