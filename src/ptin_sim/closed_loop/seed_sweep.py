from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def aggregate_seed_sweep(
    *,
    seed_root: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 1000,
    reference_method: str = "safe_gpinn",
) -> dict[str, Any]:
    seed_root = Path(seed_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = _read_seed_eval_rows(seed_root)
    _write_csv(output_dir / "seed_sweep_eval.csv", eval_rows)
    method_summary = _method_summary(eval_rows)
    _write_csv(output_dir / "seed_sweep_method_summary.csv", method_summary)
    pairwise = _pairwise_ci(
        eval_rows,
        reference_method=reference_method,
        bootstrap_samples=bootstrap_samples,
    )
    _write_csv(output_dir / "seed_sweep_pairwise_ci.csv", pairwise)
    manifest = {
        "version": "closed_loop_seed_sweep_summary_v1_20260617",
        "seed_root": str(seed_root),
        "seed_count": len({row["seed"] for row in eval_rows}),
        "row_count": len(eval_rows),
        "reference_method": reference_method,
        "bootstrap_samples": int(bootstrap_samples),
        "outputs": {
            "seed_sweep_eval_csv": "seed_sweep_eval.csv",
            "seed_sweep_method_summary_csv": "seed_sweep_method_summary.csv",
            "seed_sweep_pairwise_ci_csv": "seed_sweep_pairwise_ci.csv",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _read_seed_eval_rows(seed_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir, seed in _iter_seed_run_dirs(seed_root):
        eval_csv = run_dir / "closed_loop_policy_eval.csv"
        if not eval_csv.exists():
            continue
        with eval_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "seed": int(seed),
                        "method": row["method"],
                        "selected_sequence": row.get("selected_sequence", ""),
                        "total_reward": float(row["total_reward"]),
                        "steps": int(float(row.get("steps", 0) or 0)),
                        "terminated": int(float(row.get("terminated", 0) or 0)),
                    }
                )
    if not rows:
        raise ValueError(f"No seed eval rows found under {seed_root}")
    return rows


def _iter_seed_run_dirs(seed_root: Path) -> list[tuple[Path, int]]:
    process_seed_by_dir = _read_shard_seed_map(seed_root)
    runs: list[tuple[Path, int]] = []
    for run_dir in sorted(seed_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("seed_"):
            seed_text = run_dir.name.split("seed_", 1)[-1]
            if seed_text.isdigit():
                runs.append((run_dir, int(seed_text)))
        elif run_dir.name.startswith("shard"):
            seed = process_seed_by_dir.get(run_dir.resolve())
            if seed is not None:
                runs.append((run_dir, seed))
    return runs


def _read_shard_seed_map(seed_root: Path) -> dict[Path, int]:
    process_csv = seed_root / "shard_processes.csv"
    if not process_csv.exists():
        return {}
    seed_by_dir: dict[Path, int] = {}
    with process_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                output_dir = Path(str(row.get("output_dir", ""))).resolve()
                seed = int(str(row.get("seed", "")).strip())
            except (OSError, ValueError):
                continue
            seed_by_dir[output_dir] = seed
    return seed_by_dir


def _method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[float]] = {}
    sequence_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        method = str(row["method"])
        by_method.setdefault(method, []).append(float(row["total_reward"]))
        sequence = str(row.get("selected_sequence", ""))
        sequence_counts.setdefault(method, {})[sequence] = sequence_counts.setdefault(method, {}).get(sequence, 0) + 1
    summary = []
    for method, values in sorted(by_method.items()):
        arr = np.asarray(values, dtype=float)
        sequences = sequence_counts.get(method, {})
        modal_sequence = max(sequences.items(), key=lambda item: item[1])[0] if sequences else ""
        summary.append(
            {
                "method": method,
                "seed_count": len(values),
                "mean_total_reward": round(float(arr.mean()), 9),
                "std_total_reward": round(float(arr.std(ddof=1)) if len(values) > 1 else 0.0, 9),
                "min_total_reward": round(float(arr.min()), 9),
                "max_total_reward": round(float(arr.max()), 9),
                "modal_sequence": modal_sequence,
            }
        )
    return summary


def _pairwise_ci(
    rows: list[dict[str, Any]],
    *,
    reference_method: str,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    by_seed_method: dict[int, dict[str, float]] = {}
    for row in rows:
        by_seed_method.setdefault(int(row["seed"]), {})[str(row["method"])] = float(row["total_reward"])
    seeds = sorted(seed for seed, values in by_seed_method.items() if reference_method in values)
    methods = sorted({str(row["method"]) for row in rows if str(row["method"]) != reference_method})
    rng = np.random.default_rng(7)
    out: list[dict[str, Any]] = []
    for baseline in methods:
        paired = [
            by_seed_method[seed][reference_method] - by_seed_method[seed][baseline]
            for seed in seeds
            if baseline in by_seed_method[seed]
        ]
        if not paired:
            continue
        deltas = np.asarray(paired, dtype=float)
        boot = []
        for _ in range(int(bootstrap_samples)):
            sample = rng.choice(deltas, size=len(deltas), replace=True)
            boot.append(float(sample.mean()))
        lower, upper = np.percentile(boot, [2.5, 97.5])
        out.append(
            {
                "reference": reference_method,
                "baseline": baseline,
                "paired_seed_count": len(deltas),
                "mean_delta": round(float(deltas.mean()), 9),
                "ci95_low": round(float(lower), 9),
                "ci95_high": round(float(upper), 9),
                "supported_positive": int(lower > 0.0),
            }
        )
    return out


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate closed-loop PTIN seed sweep results.")
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--reference-method", default="safe_gpinn")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = aggregate_seed_sweep(
        seed_root=args.seed_root,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        reference_method=args.reference_method,
    )
    print("Closed-loop PTIN seed sweep summary")
    print(f"- seed_count: {manifest['seed_count']}")
    print(f"- row_count: {manifest['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
