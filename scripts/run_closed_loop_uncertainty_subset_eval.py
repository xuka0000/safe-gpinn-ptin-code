from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from pathlib import Path
from typing import Any, Iterable

from ptin_sim.closed_loop.run_closed_loop_experiment import load_run_config
from scripts.build_closed_loop_physical_policy_table import (
    _latex_table,
    _write_csv,
    summarize_physical_policy_metrics,
)
from scripts.closed_loop_uncertainty_sensitivity import _scale_dir_name
from scripts.run_closed_loop_checkpoint_subset_eval import (
    parse_method_subset,
    run_subset_checkpoint_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = ROOT / (
    "outputs/closed_loop_runs/"
    "formal_cross_layer_online_ns3_seed_sweep_20260625_batch31_transport_aware_wait_relay_reuse_rollout"
)
DEFAULT_SCALED_ROOT = ROOT / (
    "outputs/analysis/algorithm_repair_goal_20260627/"
    "robustness_scaled_rollouts_late_reward_gate"
)
DEFAULT_OUTPUT_ROOT = ROOT / (
    "outputs/analysis/algorithm_repair_goal_20260627/"
    "robustness_uncertainty_subset_eval_late_reward_gate"
)
DEFAULT_TABLE_ROOT = ROOT / (
    "outputs/analysis/algorithm_repair_goal_20260627/"
    "robustness_uncertainty_subset_table_late_reward_gate"
)
DEFAULT_METHODS = "safe_gpinn,rolling_mpc"
DEFAULT_SCALES = "0.75,1.00,1.25,1.50"
DEFAULT_SHARDS = "shard01,shard02,shard03,shard04,shard05"


def parse_scales(text: str) -> tuple[float, ...]:
    scales = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not scales:
        raise ValueError("at least one uncertainty scale is required")
    return scales


def parse_shards(text: str) -> tuple[str, ...]:
    shards = tuple(item.strip() for item in text.split(",") if item.strip())
    if not shards:
        raise ValueError("at least one shard is required")
    return shards


def scale_dir_name(scale: float) -> str:
    return _scale_dir_name(scale)


def scale_text(scale: float) -> str:
    return f"{float(scale):g}"


def run_key_shard(scale: float, shard: str) -> str:
    return f"{scale_dir_name(scale)}__{shard}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def add_run_fields(
    rows: Iterable[dict[str, str]],
    *,
    scale: float,
    shard: str,
    seed: int,
) -> list[dict[str, str]]:
    tagged_rows: list[dict[str, str]] = []
    for row in rows:
        tagged = dict(row)
        tagged["source_shard"] = shard
        tagged["shard"] = run_key_shard(scale, shard)
        tagged["seed"] = str(seed)
        tagged["uncertainty_scale"] = scale_text(scale)
        tagged_rows.append(tagged)
    return tagged_rows


def summarize_by_scale(
    trace_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    scales = sorted(
        {scale_text(float(row["uncertainty_scale"])) for row in trace_rows},
        key=float,
    )
    summary_rows: list[dict[str, Any]] = []
    for scale in scales:
        scale_trace = [
            row
            for row in trace_rows
            if scale_text(float(row["uncertainty_scale"])) == scale
        ]
        scale_eval = [
            row
            for row in eval_rows
            if scale_text(float(row["uncertainty_scale"])) == scale
        ]
        for row in summarize_physical_policy_metrics(scale_trace, scale_eval):
            summary_rows.append({"uncertainty_scale": float(scale), **row})
    return summary_rows


def write_progress(output_root: Path, payload: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "progress.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def evaluate_uncertainty_subset(
    *,
    batch_root: Path,
    scaled_root: Path,
    output_root: Path,
    table_root: Path,
    methods: tuple[str, ...],
    scales: tuple[float, ...],
    shards: tuple[str, ...],
    table_label: str,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    merged_eval: list[dict[str, str]] = []
    merged_trace: list[dict[str, str]] = []
    completed: list[dict[str, Any]] = []

    total_runs = len(scales) * len(shards)
    run_index = 0
    for scale in scales:
        for shard in shards:
            run_index += 1
            shard_dir = batch_root / shard
            config_path = shard_dir / "run_config.yaml"
            config = load_run_config(config_path)
            seed = int(config.seed)
            scaled_dataset = (
                scaled_root
                / scale_dir_name(scale)
                / shard
                / "scaled_rollout_dataset.csv"
            )
            if not scaled_dataset.exists():
                raise FileNotFoundError(
                    f"missing scaled rollout dataset: {scaled_dataset}"
                )
            output_dir = output_root / scale_dir_name(scale) / shard
            write_progress(
                output_root,
                {
                    "status": "running",
                    "current_scale": float(scale),
                    "current_shard": shard,
                    "run_index": run_index,
                    "total_runs": total_runs,
                    "completed": completed,
                    "methods": list(methods),
                },
            )
            run_subset_checkpoint_evaluation(
                output_dir=output_dir,
                config_path=config_path,
                seed=seed,
                rollout_dataset=scaled_dataset,
                checkpoint_dir=shard_dir / "model_checkpoints",
                methods=methods,
                uncertainty_scale=float(scale),
            )
            merged_eval.extend(
                add_run_fields(
                    read_csv(output_dir / "closed_loop_policy_eval.csv"),
                    scale=scale,
                    shard=shard,
                    seed=seed,
                )
            )
            merged_trace.extend(
                add_run_fields(
                    read_csv(output_dir / "closed_loop_policy_step_trace.csv"),
                    scale=scale,
                    shard=shard,
                    seed=seed,
                )
            )
            completed.append({"scale": float(scale), "shard": shard, "seed": seed})

    eval_path = output_root / "merged_policy_eval.csv"
    trace_path = output_root / "merged_policy_step_trace.csv"
    summary_csv_path = table_root / f"{table_label}.csv"
    write_csv(eval_path, merged_eval)
    write_csv(trace_path, merged_trace)
    summary_rows = summarize_by_scale(merged_trace, merged_eval)
    _write_csv(summary_csv_path, summary_rows)

    per_scale_tables: list[dict[str, str]] = []
    for scale in scales:
        scale_rows = [
            row for row in summary_rows if float(row["uncertainty_scale"]) == float(scale)
        ]
        scale_label = f"{table_label}_{scale_dir_name(scale)}"
        scale_csv = table_root / f"{scale_label}.csv"
        scale_tex = table_root / f"{scale_label}.tex"
        _write_csv(scale_csv, scale_rows)
        scale_tex.write_text(_latex_table(scale_rows), encoding="utf-8")
        per_scale_tables.append(
            {
                "scale": str(float(scale)),
                "csv": str(scale_csv),
                "tex": str(scale_tex),
            }
        )

    manifest = {
        "version": "closed_loop_uncertainty_subset_eval_v1_20260628",
        "truth_boundary": (
            "five_shard_uncertainty_scaled_rollout_eval_for_safe_gpinn_and_selected_baselines"
        ),
        "batch_root": str(batch_root),
        "scaled_root": str(scaled_root),
        "methods": list(methods),
        "scales": [float(scale) for scale in scales],
        "shards": list(shards),
        "outputs": {
            "merged_eval": str(eval_path),
            "merged_trace": str(trace_path),
            "summary_csv": str(summary_csv_path),
            "per_scale_tables": per_scale_tables,
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_progress(
        output_root,
        {
            "status": "done",
            "completed": completed,
            "methods": list(methods),
            "outputs": manifest["outputs"],
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run uncertainty-scaled closed-loop evaluation for a method subset."
    )
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--scaled-root", type=Path, default=DEFAULT_SCALED_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get(
        "PTIN_UNCERTAINTY_SUBSET_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)
    )))
    parser.add_argument("--table-root", type=Path, default=Path(os.environ.get(
        "PTIN_UNCERTAINTY_SUBSET_TABLE_ROOT", str(DEFAULT_TABLE_ROOT)
    )))
    parser.add_argument(
        "--table-label",
        default=os.environ.get(
            "PTIN_UNCERTAINTY_SUBSET_TABLE_LABEL",
            "robustness_uncertainty_subset_physical_table",
        ),
    )
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--shards", default=DEFAULT_SHARDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = evaluate_uncertainty_subset(
            batch_root=args.batch_root,
            scaled_root=args.scaled_root,
            output_root=args.output_root,
            table_root=args.table_root,
            methods=parse_method_subset(args.methods),
            scales=parse_scales(args.scales),
            shards=parse_shards(args.shards),
            table_label=args.table_label,
        )
        print(json.dumps({"status": "ok", "outputs": manifest["outputs"]}, indent=2))
        return 0
    except Exception as exc:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "error.txt").write_text(
            "".join(traceback.format_exception(exc)),
            encoding="utf-8",
        )
        write_progress(
            args.output_root,
            {
                "status": "error",
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
