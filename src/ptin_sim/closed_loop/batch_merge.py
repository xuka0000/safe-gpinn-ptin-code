from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MERGE_FILES = {
    "closed_loop_policy_eval.csv": "merged_policy_eval.csv",
    "closed_loop_policy_step_trace.csv": "merged_policy_step_trace.csv",
    "closed_loop_training_curves.csv": "merged_training_curves.csv",
    "closed_loop_rollout_dataset.csv": "merged_rollout_dataset.csv",
}


def merge_closed_loop_shards(
    *,
    batch_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    batch_path = Path(batch_root)
    output_path = Path(output_dir) if output_dir is not None else batch_path / "merged"
    output_path.mkdir(parents=True, exist_ok=True)
    shard_rows = _read_shard_processes(batch_path / "shard_processes.csv")
    outputs: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for source_name, target_name in MERGE_FILES.items():
        rows = _read_shard_csv_rows(shard_rows, source_name)
        target = output_path / target_name
        _write_csv(target, rows)
        outputs[target_name] = str(target)
        row_counts[target_name] = len(rows)
    manifest = {
        "version": "closed_loop_shard_merge_v1_20260620",
        "batch_root": str(batch_path),
        "shard_count": len(shard_rows),
        "row_counts": row_counts,
        "outputs": outputs,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _read_shard_processes(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing shard process table: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No shard rows in {path}")
    return rows


def _read_shard_csv_rows(
    shard_rows: list[dict[str, str]],
    source_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shard in shard_rows:
        source = Path(shard["output_dir"]) / source_name
        if not source.exists():
            continue
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                out.append(
                    {
                        "shard": shard.get("shard", ""),
                        "seed": shard.get("seed", ""),
                        **row,
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
    parser = argparse.ArgumentParser(description="Merge closed-loop shard outputs.")
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = merge_closed_loop_shards(
        batch_root=args.batch_root,
        output_dir=args.output_dir,
    )
    print("Closed-loop shard merge")
    print(f"- shard_count: {manifest['shard_count']}")
    for name, count in manifest["row_counts"].items():
        print(f"- {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
