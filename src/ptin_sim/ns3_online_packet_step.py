"""Run one ns-3 packet step for closed-loop PTIN communication feedback."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "python_backend" / "ns3_rf_packet_validation_20260608"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "python_backend" / "ns3_online_packet_step_20260617"
DEFAULT_CPP_SOURCE = ROOT / "src" / "ptin_sim" / "ns3_ptin_packet_replay.cc"
DEFAULT_TEMP_ROOT = Path("C:/Temp")
VERSION = "ns3_online_packet_step_20260617_v1"
TRUTH_BOUNDARY = (
    "online_step_ns3_log_distance_contention_queueing_fading_packet_feedback;"
    "not_full_wifi_mac_or_hardware_rf_validation"
)


CommandRunner = Callable[[str], dict[str, Any]]


def run_online_step(
    args: argparse.Namespace,
    *,
    command_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cpp_source = Path(args.cpp_source)
    temp_root = Path(args.temp_root)
    target_time_s = float(args.time_min) * 60.0
    step_tag = str(int(round(target_time_s)))
    run_dir = temp_root / f"ptin_ns3_online_step_{step_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    nodes_csv = input_dir / "ns3_nodes.csv"
    schedule_csv = input_dir / "ns3_packet_schedule.csv"
    result_csv = run_dir / "ns3_packet_results.csv"
    output_result_csv = output_dir / f"ns3_packet_results_{step_tag}.csv"
    summary_json = output_dir / f"ns3_online_packet_step_{step_tag}.json"
    blockers = _missing_inputs(nodes_csv, schedule_csv, cpp_source)
    commands: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    nearest_schedule_time_s = None

    if not blockers:
        shutil.copy2(nodes_csv, run_dir / "ns3_nodes.csv")
        shutil.copy2(cpp_source, run_dir / "ns3_ptin_packet_replay.cc")
        selected_rows = _filter_schedule_rows(
            schedule_csv,
            target_time_s=target_time_s,
            window_s=float(args.time_window_s),
            max_packets_per_domain=int(getattr(args, "max_packets_per_domain", 0)),
        )
        if not selected_rows:
            selected_rows, nearest_schedule_time_s = _nearest_schedule_rows(
                schedule_csv,
                target_time_s=target_time_s,
                max_packets_per_domain=int(getattr(args, "max_packets_per_domain", 0)),
            )
        _write_csv(run_dir / "ns3_packet_schedule.csv", selected_rows)
        if not selected_rows:
            blockers.append("no_packets_for_step_time")
        else:
            runner = command_runner or _run_wsl_command
            compile_cmd = _compile_command(run_dir)
            commands.append(
                _call_runner(
                    runner,
                    compile_cmd,
                    log_path=output_dir / f"ns3_online_compile_{step_tag}.log",
                    run_dir=run_dir,
                    timeout_s=float(args.timeout_s),
                    timeout_blocker="ns3_online_compile_timeout",
                    error_blocker_prefix="ns3_online_compile_error",
                )
            )
            if commands[-1].get("returncode") != 0:
                blockers.append(str(commands[-1].get("blocker") or "ns3_online_compile_failed"))
            else:
                run_cmd = _run_command(run_dir)
                commands.append(
                    _call_runner(
                        runner,
                        run_cmd,
                        log_path=output_dir / f"ns3_online_run_{step_tag}.log",
                        run_dir=run_dir,
                        timeout_s=float(args.timeout_s),
                        timeout_blocker="ns3_online_run_timeout",
                        error_blocker_prefix="ns3_online_run_error",
                    )
                )
                if commands[-1].get("returncode") != 0:
                    blockers.append(str(commands[-1].get("blocker") or "ns3_online_run_failed"))
                elif not result_csv.exists():
                    blockers.append("ns3_online_result_missing")
                else:
                    shutil.copy2(result_csv, output_result_csv)
                    packets = _packet_json_rows(result_csv)

    status = "ns3_online_step_complete" if not blockers else "blocked"
    summary = {
        "version": VERSION,
        "status": status,
        "time_min": float(args.time_min),
        "time_s": target_time_s,
        "nearest_schedule_time_s": nearest_schedule_time_s,
        "schedule_time_mode": "nearest_replay" if nearest_schedule_time_s is not None else "window_exact",
        "packet_count": len(packets),
        "packets": packets,
        "commands": commands,
        "blockers": blockers,
        "truth_boundary": TRUTH_BOUNDARY,
        "outputs": {
            "step_result_csv": str(output_result_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _missing_inputs(*paths: Path) -> list[str]:
    return [f"missing_{path.name}" for path in paths if not path.exists()]


def _filter_schedule_rows(
    schedule_csv: Path,
    *,
    target_time_s: float,
    window_s: float,
    max_packets_per_domain: int = 0,
) -> list[dict[str, str]]:
    with schedule_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    half_window = max(0.0, window_s) / 2.0
    selected: list[dict[str, str]] = []
    domain_counts: dict[str, int] = {}
    for row in rows:
        try:
            time_s = float(row.get("time_s") or 0.0)
        except ValueError:
            continue
        if abs(time_s - target_time_s) <= half_window:
            domain_key = f"{time_s:.3f}|{row.get('dst_cn_node') or ''}"
            if max_packets_per_domain > 0 and domain_counts.get(domain_key, 0) >= max_packets_per_domain:
                continue
            domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
            selected.append(row)
    return selected


def _nearest_schedule_rows(
    schedule_csv: Path,
    *,
    target_time_s: float,
    max_packets_per_domain: int = 0,
) -> tuple[list[dict[str, str]], float | None]:
    with schedule_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows_by_time: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        try:
            time_s = float(row.get("time_s") or 0.0)
        except ValueError:
            continue
        rows_by_time.setdefault(time_s, []).append(row)
    if not rows_by_time:
        return [], None
    nearest_time = min(rows_by_time, key=lambda value: abs(value - target_time_s))
    selected: list[dict[str, str]] = []
    domain_counts: dict[str, int] = {}
    for row in rows_by_time[nearest_time]:
        domain_key = f"{nearest_time:.3f}|{row.get('dst_cn_node') or ''}"
        if max_packets_per_domain > 0 and domain_counts.get(domain_key, 0) >= max_packets_per_domain:
            continue
        domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
        selected.append(row)
    return selected, nearest_time


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "packet_id",
        "source_trace_id",
        "time_s",
        "src_cn_node",
        "dst_cn_node",
        "payload_bytes",
        "deadline_ms",
        "action_family",
        "scenario_id",
        "truth_boundary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _packet_json_rows(result_csv: Path) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    with result_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            delivered = str(row.get("delivered") or "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "delivered",
            }
            try:
                delay_ms = float(row.get("delay_ms") or 0.0)
            except ValueError:
                delay_ms = 0.0
            packets.append(
                {
                    "packet_id": str(row.get("packet_id") or f"packet_{len(packets)}"),
                    "delivered": delivered,
                    "delay_ms": delay_ms,
                    **_optional_numeric_packet_fields(row),
                }
            )
    return packets


def _optional_numeric_packet_fields(row: dict[str, str]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in [
        "queue_delay_ms",
        "effective_data_rate_mbps",
        "multipath_fading_db",
        "sinr_db",
    ]:
        if key in row and row.get(key) not in {None, ""}:
            fields[key] = _safe_float(row.get(key))
    if row.get("contention_count") not in {None, ""}:
        try:
            fields["contention_count"] = int(float(str(row.get("contention_count"))))
        except ValueError:
            fields["contention_count"] = 0
    return fields


def _safe_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _run_wsl_command(
    command: str,
    *,
    log_path: Path,
    run_dir: Path,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["wsl", "-d", "Ubuntu-22.04", "-u", "root", "--exec", "bash", "-lc", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "\n".join(
                [
                    f"COMMAND: {command}",
                    f"RETURN_CODE: -9",
                    f"TIMEOUT_S: {timeout_s}",
                    "",
                    "STDOUT:",
                    str(exc.stdout or ""),
                    "",
                    "STDERR:",
                    str(exc.stderr or ""),
                ]
            ),
            encoding="utf-8",
        )
        return {
            "command": command,
            "returncode": -9,
            "stdout": str(exc.stdout or "")[:1000],
            "stderr": str(exc.stderr or "")[:1000],
            "log_path": str(log_path),
            "run_dir": str(run_dir),
            "blocker": "ns3_online_timeout",
        }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"COMMAND: {command}",
                f"RETURN_CODE: {completed.returncode}",
                "",
                "STDOUT:",
                completed.stdout,
                "",
                "STDERR:",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[:1000],
        "stderr": completed.stderr.strip()[:1000],
        "log_path": str(log_path),
        "run_dir": str(run_dir),
    }


def _call_runner(
    runner: Callable[..., dict[str, Any]],
    command: str,
    *,
    log_path: Path,
    run_dir: Path,
    timeout_s: float,
    timeout_blocker: str,
    error_blocker_prefix: str,
) -> dict[str, Any]:
    try:
        result = runner(
            command,
            log_path=log_path,
            run_dir=run_dir,
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": -9,
            "stdout": str(exc.stdout or "")[:1000],
            "stderr": str(exc.stderr or "")[:1000],
            "log_path": str(log_path),
            "run_dir": str(run_dir),
            "blocker": timeout_blocker,
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc)[:1000],
            "log_path": str(log_path),
            "run_dir": str(run_dir),
            "blocker": f"{error_blocker_prefix}:{type(exc).__name__}",
        }
    if result.get("blocker") == "ns3_online_timeout":
        result = dict(result)
        result["blocker"] = timeout_blocker
    elif "blocker" not in result and result.get("returncode") != 0:
        result = dict(result)
        result["blocker"] = f"{error_blocker_prefix}:returncode_{result.get('returncode')}"
    return result


def _compile_command(run_dir: Path) -> str:
    return (
        f"cd {_bash_quote(_wsl_path(run_dir))} && "
        "g++ -std=c++17 ns3_ptin_packet_replay.cc -o ns3_ptin_packet_replay "
        "$(pkg-config --cflags --libs libns3.35-core libns3.35-network "
        "libns3.35-mobility libns3.35-propagation)"
    )


def _run_command(run_dir: Path) -> str:
    return (
        f"cd {_bash_quote(_wsl_path(run_dir))} && "
        "./ns3_ptin_packet_replay --nodes=ns3_nodes.csv "
        "--packets=ns3_packet_schedule.csv --output=ns3_packet_results.csv"
    )


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = str(resolved)[3:].replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def _bash_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cpp-source", type=Path, default=DEFAULT_CPP_SOURCE)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--time-min", type=float, required=True)
    parser.add_argument("--time-window-s", type=float, default=1.0)
    parser.add_argument("--max-packets-per-domain", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_online_step(parse_args()), ensure_ascii=True))


if __name__ == "__main__":
    main()
