from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from pathlib import Path
from typing import Any, Iterable

import ptin_sim.closed_loop.training as training
from ptin_sim.closed_loop.run_closed_loop_experiment import load_run_config
from scripts.build_closed_loop_physical_policy_table import (
    _latex_table,
    _write_csv,
    summarize_physical_policy_metrics,
)
from scripts.run_closed_loop_checkpoint_subset_eval import parse_method_subset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = ROOT / (
    "outputs/closed_loop_runs/"
    "formal_cross_layer_online_ns3_seed_sweep_20260625_batch31_transport_aware_wait_relay_reuse_rollout"
)
DEFAULT_BASE_SCENARIO_DIR = ROOT / "data/scenario_reconstruction_progressive_stress_v2"
DEFAULT_VARIANT_ROOT = ROOT / (
    "outputs/analysis/algorithm_repair_goal_20260627/"
    "scenario_robustness_variants_late_reward_gate"
)
DEFAULT_OUTPUT_ROOT = ROOT / (
    "outputs/analysis/algorithm_repair_goal_20260627/"
    "scenario_robustness_subset_eval_late_reward_gate"
)
DEFAULT_TABLE_ROOT = ROOT / (
    "outputs/analysis/algorithm_repair_goal_20260627/"
    "scenario_robustness_subset_table_late_reward_gate"
)
DEFAULT_METHODS = "safe_gpinn,rolling_mpc"
DEFAULT_MULTIPLIERS = "1.35,1.60,1.85"
DEFAULT_SHARDS = "shard01,shard02,shard03,shard04,shard05"


def parse_multipliers(text: str) -> tuple[float, ...]:
    multipliers = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not multipliers:
        raise ValueError("at least one robust travel-time multiplier is required")
    return multipliers


def parse_shards(text: str) -> tuple[str, ...]:
    shards = tuple(item.strip() for item in text.split(",") if item.strip())
    if not shards:
        raise ValueError("at least one shard is required")
    return shards


def multiplier_text(multiplier: float) -> str:
    return f"{float(multiplier):g}"


def case_dir_name(multiplier: float) -> str:
    return f"traffic_x{float(multiplier):.2f}".replace(".", "p")


def run_key_shard(multiplier: float, shard: str) -> str:
    return f"{case_dir_name(multiplier)}__{shard}"


def _set_section_key(text: str, section: str, key: str, value: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_section = False
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped == f"{section}:":
            in_section = True
            out.append(line)
            continue
        if in_section and stripped and not line.startswith((" ", "\t")):
            if not replaced:
                out.append(f"  {key}: {value}")
                replaced = True
            in_section = False
        if in_section and stripped.startswith(f"{key}:"):
            indent = line[: len(line) - len(line.lstrip())] or "  "
            out.append(f"{indent}{key}: {value}")
            replaced = True
        else:
            out.append(line)
    if in_section and not replaced:
        out.append(f"  {key}: {value}")
        replaced = True
    if not replaced:
        out.extend(["", f"{section}:", f"  {key}: {value}"])
    return "\n".join(out) + "\n"


def patch_robust_travel_time_multiplier(text: str, multiplier: float) -> str:
    return _set_section_key(
        text,
        "framework_alignment",
        "robust_travel_time_multiplier",
        multiplier_text(multiplier),
    )


def _patch_base_scenario_dir(text: str, base_data_root: Path) -> str:
    return _set_section_key(
        text,
        "source_data",
        "base_scenario_dir",
        str(base_data_root.resolve()),
    )


def _resolve_source_data_root(base_scenario_dir: Path) -> Path:
    disaster_path = base_scenario_dir / "disaster_scenario.yaml"
    text = disaster_path.read_text(encoding="utf-8")
    in_source_data = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "source_data:":
            in_source_data = True
            continue
        if in_source_data and stripped and not line.startswith((" ", "\t")):
            break
        if in_source_data and stripped.startswith("base_scenario_dir:"):
            raw = stripped.split(":", 1)[1].strip().strip("'\"")
            path = Path(raw)
            return path if path.is_absolute() else (base_scenario_dir / path).resolve()
    return base_scenario_dir.resolve()


def build_scenario_variants(
    *,
    base_scenario_dir: Path,
    variant_root: Path,
    multipliers: Iterable[float],
) -> dict[float, Path]:
    base_scenario_dir = base_scenario_dir.resolve()
    base_text = (base_scenario_dir / "disaster_scenario.yaml").read_text(
        encoding="utf-8"
    )
    base_data_root = _resolve_source_data_root(base_scenario_dir)
    variant_root.mkdir(parents=True, exist_ok=True)
    variants: dict[float, Path] = {}
    manifest_rows: list[dict[str, Any]] = []
    for multiplier in multipliers:
        case_dir = variant_root / case_dir_name(multiplier)
        case_dir.mkdir(parents=True, exist_ok=True)
        patched = patch_robust_travel_time_multiplier(base_text, multiplier)
        patched = _patch_base_scenario_dir(patched, base_data_root)
        (case_dir / "disaster_scenario.yaml").write_text(patched, encoding="utf-8")
        variants[float(multiplier)] = case_dir
        manifest_rows.append(
            {
                "scenario_case": case_dir_name(multiplier),
                "robust_travel_time_multiplier": float(multiplier),
                "scenario_dir": str(case_dir),
                "base_scenario_dir": str(base_scenario_dir),
                "source_data_root": str(base_data_root),
            }
        )
    (variant_root / "scenario_variant_manifest.json").write_text(
        json.dumps(
            {
                "version": "closed_loop_scenario_robustness_variants_v1_20260628",
                "base_scenario_dir": str(base_scenario_dir),
                "source_data_root": str(base_data_root),
                "variants": manifest_rows,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    _write_csv(variant_root / "scenario_variant_manifest.csv", manifest_rows)
    return variants


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


def add_case_fields(
    rows: Iterable[dict[str, str]],
    *,
    multiplier: float,
    shard: str,
    seed: int,
) -> list[dict[str, str]]:
    tagged_rows: list[dict[str, str]] = []
    for row in rows:
        tagged = dict(row)
        tagged["source_shard"] = shard
        tagged["shard"] = run_key_shard(multiplier, shard)
        tagged["seed"] = str(seed)
        tagged["scenario_case"] = case_dir_name(multiplier)
        tagged["robust_travel_time_multiplier"] = multiplier_text(multiplier)
        tagged_rows.append(tagged)
    return tagged_rows


def summarize_by_case(
    trace_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    multipliers = sorted(
        {
            multiplier_text(float(row["robust_travel_time_multiplier"]))
            for row in trace_rows
        },
        key=float,
    )
    summary_rows: list[dict[str, Any]] = []
    for multiplier in multipliers:
        case_trace = [
            row
            for row in trace_rows
            if multiplier_text(float(row["robust_travel_time_multiplier"]))
            == multiplier
        ]
        case_eval = [
            row
            for row in eval_rows
            if multiplier_text(float(row["robust_travel_time_multiplier"]))
            == multiplier
        ]
        for row in summarize_physical_policy_metrics(case_trace, case_eval):
            summary_rows.append(
                {
                    "scenario_case": case_dir_name(float(multiplier)),
                    "robust_travel_time_multiplier": float(multiplier),
                    **row,
                }
            )
    return summary_rows


def write_progress(output_root: Path, payload: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "progress.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def run_subset_checkpoint_evaluation_with_scenario(
    *,
    output_dir: Path,
    config_path: Path,
    scenario_dir: Path,
    seed: int,
    rollout_dataset: Path,
    checkpoint_dir: Path,
    methods: tuple[str, ...],
    robust_travel_time_multiplier: float,
) -> dict[str, Any]:
    previous_methods = training.EVAL_METHODS
    try:
        training.EVAL_METHODS = methods
        manifest = training.run_closed_loop_policy_checkpoint_evaluation(
            output_dir=output_dir,
            config_path=config_path,
            adapter_mode=None,
            scenario_dir=scenario_dir,
            seed=seed,
            rollout_dataset=rollout_dataset,
            checkpoint_dir=checkpoint_dir,
            evaluation_mode="env",
        )
    finally:
        training.EVAL_METHODS = previous_methods
    manifest = dict(manifest)
    manifest["method_subset"] = list(methods)
    manifest["scenario_dir"] = str(scenario_dir)
    manifest["robust_travel_time_multiplier"] = float(robust_travel_time_multiplier)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def evaluate_scenario_robustness_subset(
    *,
    batch_root: Path,
    base_scenario_dir: Path,
    variant_root: Path,
    output_root: Path,
    table_root: Path,
    methods: tuple[str, ...],
    multipliers: tuple[float, ...],
    shards: tuple[str, ...],
    table_label: str,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    variants = build_scenario_variants(
        base_scenario_dir=base_scenario_dir,
        variant_root=variant_root,
        multipliers=multipliers,
    )
    merged_eval: list[dict[str, str]] = []
    merged_trace: list[dict[str, str]] = []
    completed: list[dict[str, Any]] = []

    total_runs = len(multipliers) * len(shards)
    run_index = 0
    for multiplier in multipliers:
        for shard in shards:
            run_index += 1
            shard_dir = batch_root / shard
            config_path = shard_dir / "run_config.yaml"
            config = load_run_config(config_path)
            seed = int(config.seed)
            output_dir = output_root / case_dir_name(multiplier) / shard
            write_progress(
                output_root,
                {
                    "status": "running",
                    "current_multiplier": float(multiplier),
                    "current_shard": shard,
                    "run_index": run_index,
                    "total_runs": total_runs,
                    "completed": completed,
                    "methods": list(methods),
                },
            )
            run_subset_checkpoint_evaluation_with_scenario(
                output_dir=output_dir,
                config_path=config_path,
                scenario_dir=variants[float(multiplier)],
                seed=seed,
                rollout_dataset=shard_dir / "enriched_rollout_dataset.csv",
                checkpoint_dir=shard_dir / "model_checkpoints",
                methods=methods,
                robust_travel_time_multiplier=float(multiplier),
            )
            merged_eval.extend(
                add_case_fields(
                    read_csv(output_dir / "closed_loop_policy_eval.csv"),
                    multiplier=multiplier,
                    shard=shard,
                    seed=seed,
                )
            )
            merged_trace.extend(
                add_case_fields(
                    read_csv(output_dir / "closed_loop_policy_step_trace.csv"),
                    multiplier=multiplier,
                    shard=shard,
                    seed=seed,
                )
            )
            completed.append(
                {
                    "robust_travel_time_multiplier": float(multiplier),
                    "shard": shard,
                    "seed": seed,
                }
            )

    eval_path = output_root / "merged_policy_eval.csv"
    trace_path = output_root / "merged_policy_step_trace.csv"
    summary_csv_path = table_root / f"{table_label}.csv"
    write_csv(eval_path, merged_eval)
    write_csv(trace_path, merged_trace)
    summary_rows = summarize_by_case(merged_trace, merged_eval)
    _write_csv(summary_csv_path, summary_rows)

    per_case_tables: list[dict[str, str]] = []
    for multiplier in multipliers:
        case_rows = [
            row
            for row in summary_rows
            if float(row["robust_travel_time_multiplier"]) == float(multiplier)
        ]
        case_label = f"{table_label}_{case_dir_name(multiplier)}"
        case_csv = table_root / f"{case_label}.csv"
        case_tex = table_root / f"{case_label}.tex"
        _write_csv(case_csv, case_rows)
        case_tex.write_text(_latex_table(case_rows), encoding="utf-8")
        per_case_tables.append(
            {
                "robust_travel_time_multiplier": str(float(multiplier)),
                "csv": str(case_csv),
                "tex": str(case_tex),
            }
        )

    manifest = {
        "version": "closed_loop_scenario_robustness_subset_eval_v1_20260628",
        "truth_boundary": (
            "five_shard_same_checkpoint_evaluation_under_robust_travel_time_scenario_perturbations"
        ),
        "batch_root": str(batch_root),
        "base_scenario_dir": str(base_scenario_dir),
        "variant_root": str(variant_root),
        "methods": list(methods),
        "robust_travel_time_multipliers": [float(value) for value in multipliers],
        "shards": list(shards),
        "outputs": {
            "merged_eval": str(eval_path),
            "merged_trace": str(trace_path),
            "summary_csv": str(summary_csv_path),
            "per_case_tables": per_case_tables,
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
        description="Run closed-loop evaluation under scenario travel-time perturbations."
    )
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--base-scenario-dir", type=Path, default=DEFAULT_BASE_SCENARIO_DIR)
    parser.add_argument("--variant-root", type=Path, default=Path(os.environ.get(
        "PTIN_SCENARIO_ROBUSTNESS_VARIANT_ROOT", str(DEFAULT_VARIANT_ROOT)
    )))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get(
        "PTIN_SCENARIO_ROBUSTNESS_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)
    )))
    parser.add_argument("--table-root", type=Path, default=Path(os.environ.get(
        "PTIN_SCENARIO_ROBUSTNESS_TABLE_ROOT", str(DEFAULT_TABLE_ROOT)
    )))
    parser.add_argument(
        "--table-label",
        default=os.environ.get(
            "PTIN_SCENARIO_ROBUSTNESS_TABLE_LABEL",
            "scenario_robustness_subset_physical_table",
        ),
    )
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--multipliers", default=DEFAULT_MULTIPLIERS)
    parser.add_argument("--shards", default=DEFAULT_SHARDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = evaluate_scenario_robustness_subset(
            batch_root=args.batch_root,
            base_scenario_dir=args.base_scenario_dir,
            variant_root=args.variant_root,
            output_root=args.output_root,
            table_root=args.table_root,
            methods=parse_method_subset(args.methods),
            multipliers=parse_multipliers(args.multipliers),
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
