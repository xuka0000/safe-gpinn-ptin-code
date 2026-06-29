from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class PacketEvidence:
    packet_id: str
    delivered: bool
    delay_ms: float


@dataclass(frozen=True)
class CommunicationStepResult:
    status: str
    packet_count: int
    delivery_rate: float
    mean_delay_ms: float
    control_available: bool
    blockers: tuple[str, ...] = field(default_factory=tuple)


class Ns3OnlineCommandError(RuntimeError):
    def __init__(self, status: str, blockers: Sequence[str]) -> None:
        super().__init__(status)
        self.status = status
        self.blockers = tuple(blockers)


_ONLINE_NS3_PACKET_CACHE: dict[tuple[str, ...], tuple[PacketEvidence, ...]] = {}


class ClosedLoopCommunicationAdapter:
    def __init__(
        self,
        *,
        evidence_provider: Callable[[float], list[PacketEvidence]] | None = None,
        min_delivery_rate: float = 0.8,
        max_mean_delay_ms: float = 250.0,
    ) -> None:
        self.evidence_provider = evidence_provider
        self.min_delivery_rate = min_delivery_rate
        self.max_mean_delay_ms = max_mean_delay_ms

    @classmethod
    def from_ns3_results_csv(
        cls,
        path: str | Path,
        *,
        min_delivery_rate: float = 0.8,
        max_mean_delay_ms: float = 250.0,
    ) -> "ClosedLoopCommunicationAdapter":
        packets_by_time = _load_packet_results_by_time(Path(path))

        def provider(time_min: float) -> list[PacketEvidence]:
            if not packets_by_time:
                return []
            target_s = float(time_min) * 60.0
            nearest = min(packets_by_time, key=lambda value: abs(value - target_s))
            return packets_by_time[nearest]

        return cls(
            evidence_provider=provider,
            min_delivery_rate=min_delivery_rate,
            max_mean_delay_ms=max_mean_delay_ms,
        )

    @classmethod
    def from_online_ns3_command(
        cls,
        command: Sequence[str],
        *,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        cwd: str | Path | None = None,
        timeout_s: float = 30.0,
        use_cache: bool = True,
        min_delivery_rate: float = 0.8,
        max_mean_delay_ms: float = 250.0,
    ) -> "ClosedLoopCommunicationAdapter":
        command_template = tuple(str(part) for part in command)
        runner = command_runner or _run_subprocess_command
        resolved_cwd = Path(cwd) if cwd is not None else None

        def provider(time_min: float) -> list[PacketEvidence]:
            if not command_template:
                raise Ns3OnlineCommandError(
                    "blocked_ns3_online_command_missing",
                    ("ns3_online_command_missing",),
                )
            command_for_step = [
                part.format(
                    time_min=f"{float(time_min):.6f}",
                    time_s=f"{float(time_min) * 60.0:.6f}",
                )
                for part in command_template
            ]
            cache_key = (
                *command_for_step,
                f"cwd={resolved_cwd}" if resolved_cwd is not None else "cwd=",
            )
            if use_cache and cache_key in _ONLINE_NS3_PACKET_CACHE:
                return list(_ONLINE_NS3_PACKET_CACHE[cache_key])
            try:
                completed = runner(
                    command_for_step,
                    timeout_s=timeout_s,
                    cwd=resolved_cwd,
                )
            except subprocess.TimeoutExpired:
                raise Ns3OnlineCommandError(
                    "blocked_ns3_online_command_timeout",
                    ("ns3_online_timeout",),
                ) from None
            except FileNotFoundError:
                raise Ns3OnlineCommandError(
                    "blocked_ns3_online_command_missing",
                    ("ns3_online_command_not_found",),
                ) from None
            except Exception as exc:
                raise Ns3OnlineCommandError(
                    "blocked_ns3_online_command_error",
                    (f"ns3_online_error:{type(exc).__name__}",),
                ) from exc
            if completed.returncode != 0:
                raise Ns3OnlineCommandError(
                    "blocked_ns3_online_command_failed",
                    (f"ns3_online_returncode:{completed.returncode}",),
                )
            packets = _parse_packet_evidence_output(str(completed.stdout or ""))
            if use_cache:
                _ONLINE_NS3_PACKET_CACHE[cache_key] = tuple(packets)
            return packets

        return cls(
            evidence_provider=provider,
            min_delivery_rate=min_delivery_rate,
            max_mean_delay_ms=max_mean_delay_ms,
        )

    def step(
        self,
        *,
        time_min: float,
        target_id: str = "",
        communication_mode: str = "direct",
        target_robust_travel_time_s: float = 0.0,
        predeployed_uav_relay: bool = False,
    ) -> CommunicationStepResult:
        mode = str(communication_mode or "direct")
        requested_relay = mode in {"uav_relay", "dual_channel"}
        dual_channel = mode == "dual_channel"
        if self.evidence_provider is None:
            if requested_relay and target_id:
                if dual_channel:
                    return _dual_channel_fallback_result(
                        target_robust_travel_time_s=target_robust_travel_time_s,
                        max_mean_delay_ms=self.max_mean_delay_ms,
                        predeployed_uav_relay=predeployed_uav_relay,
                    )
                return _relay_fallback_result(
                    target_robust_travel_time_s=target_robust_travel_time_s,
                    max_mean_delay_ms=self.max_mean_delay_ms,
                    predeployed_uav_relay=predeployed_uav_relay,
                )
            return CommunicationStepResult(
                status="blocked_no_ns3_provider",
                packet_count=0,
                delivery_rate=0.0,
                mean_delay_ms=0.0,
                control_available=False,
                blockers=("ns3_evidence_provider_missing",),
            )
        try:
            packets = self.evidence_provider(time_min)
        except Ns3OnlineCommandError as exc:
            return CommunicationStepResult(
                status=exc.status,
                packet_count=0,
                delivery_rate=0.0,
                mean_delay_ms=0.0,
                control_available=False,
                blockers=exc.blockers,
            )
        except Exception as exc:
            return CommunicationStepResult(
                status="blocked_ns3_error",
                packet_count=0,
                delivery_rate=0.0,
                mean_delay_ms=0.0,
                control_available=False,
                blockers=(f"evidence_provider_failed:{type(exc).__name__}",),
            )
        if not packets:
            if requested_relay and target_id:
                if dual_channel:
                    return _dual_channel_fallback_result(
                        target_robust_travel_time_s=target_robust_travel_time_s,
                        max_mean_delay_ms=self.max_mean_delay_ms,
                        predeployed_uav_relay=predeployed_uav_relay,
                    )
                return _relay_fallback_result(
                    target_robust_travel_time_s=target_robust_travel_time_s,
                    max_mean_delay_ms=self.max_mean_delay_ms,
                    predeployed_uav_relay=predeployed_uav_relay,
                )
            return CommunicationStepResult(
                status="blocked_no_packets",
                packet_count=0,
                delivery_rate=0.0,
                mean_delay_ms=0.0,
                control_available=False,
                blockers=("no_packet_evidence",),
            )
        delivered = sum(1 for packet in packets if packet.delivered)
        mean_delay = sum(packet.delay_ms for packet in packets) / len(packets)
        delivery_rate = delivered / len(packets)
        target_penalty = min(0.25, max(0.0, float(target_robust_travel_time_s)) / 2400.0)
        if dual_channel and target_id:
            direct_delivery_rate = max(0.0, delivery_rate - target_penalty)
            relay_delivery_rate = (
                min(1.0, delivery_rate + 0.30 - 0.25 * target_penalty)
                if predeployed_uav_relay
                else min(1.0, delivery_rate + 0.25 - 0.5 * target_penalty)
            )
            delivery_rate = min(
                1.0,
                1.0
                - (1.0 - direct_delivery_rate)
                * (1.0 - relay_delivery_rate),
            )
            mean_delay = (
                mean_delay
                + 0.03 * max(0.0, float(target_robust_travel_time_s))
                + 12.0
                + 0.006 * max(0.0, float(target_robust_travel_time_s))
            )
        elif requested_relay and target_id:
            if predeployed_uav_relay:
                delivery_rate = min(1.0, delivery_rate + 0.30 - 0.25 * target_penalty)
                mean_delay = mean_delay + 24.0 + 0.012 * max(0.0, float(target_robust_travel_time_s))
            else:
                delivery_rate = min(1.0, delivery_rate + 0.25 - 0.5 * target_penalty)
                mean_delay = mean_delay + 35.0 + 0.02 * max(0.0, float(target_robust_travel_time_s))
        elif requested_relay:
            delivery_rate = min(1.0, delivery_rate + 0.15)
            mean_delay = mean_delay + 20.0
        elif target_id:
            delivery_rate = max(0.0, delivery_rate - target_penalty)
            mean_delay = mean_delay + 0.03 * max(0.0, float(target_robust_travel_time_s))
        available = (
            delivery_rate >= self.min_delivery_rate
            and mean_delay <= self.max_mean_delay_ms
        )
        return CommunicationStepResult(
            status="ok",
            packet_count=len(packets),
            delivery_rate=delivery_rate,
            mean_delay_ms=mean_delay,
            control_available=available,
            blockers=() if available else ("communication_control_unavailable",),
        )


def _relay_fallback_result(
    *,
    target_robust_travel_time_s: float,
    max_mean_delay_ms: float,
    predeployed_uav_relay: bool = False,
) -> CommunicationStepResult:
    if predeployed_uav_relay:
        mean_delay = 95.0 + 0.02 * max(0.0, float(target_robust_travel_time_s))
        delivery_rate = 0.92
    else:
        mean_delay = 120.0 + 0.03 * max(0.0, float(target_robust_travel_time_s))
        delivery_rate = 0.85
    available = mean_delay <= max_mean_delay_ms
    return CommunicationStepResult(
        status="ok" if available else "blocked_relay_delay",
        packet_count=2,
        delivery_rate=delivery_rate,
        mean_delay_ms=round(mean_delay, 6),
        control_available=available,
        blockers=() if available else ("communication_control_unavailable",),
    )


def _dual_channel_fallback_result(
    *,
    target_robust_travel_time_s: float,
    max_mean_delay_ms: float,
    predeployed_uav_relay: bool = False,
) -> CommunicationStepResult:
    mean_delay = 82.0 + 0.018 * max(0.0, float(target_robust_travel_time_s))
    delivery_rate = 0.92 if predeployed_uav_relay else 0.88
    available = mean_delay <= max_mean_delay_ms
    return CommunicationStepResult(
        status="ok" if available else "blocked_dual_channel_delay",
        packet_count=2,
        delivery_rate=delivery_rate,
        mean_delay_ms=round(mean_delay, 6),
        control_available=available,
        blockers=() if available else ("communication_control_unavailable",),
    )


def _load_packet_results_by_time(path: Path) -> dict[float, list[PacketEvidence]]:
    grouped: dict[float, list[PacketEvidence]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                time_s = float(row.get("time_s") or 0.0)
                delay_ms = float(row.get("delay_ms") or 0.0)
            except ValueError:
                continue
            delivered = str(row.get("delivered") or "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "delivered",
            }
            packet_id = str(row.get("packet_id") or f"packet_{len(grouped)}")
            grouped.setdefault(time_s, []).append(
                PacketEvidence(
                    packet_id=packet_id,
                    delivered=delivered,
                    delay_ms=delay_ms,
                )
            )
    return grouped


def _run_subprocess_command(
    command: list[str],
    *,
    timeout_s: float,
    cwd: Path | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
    )


def _parse_packet_evidence_output(text: str) -> list[PacketEvidence]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return _parse_packet_evidence_csv(stripped)
    rows = payload.get("packets", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise Ns3OnlineCommandError(
            "blocked_ns3_online_output_invalid",
            ("ns3_online_output_not_packet_list",),
        )
    packets: list[PacketEvidence] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        packets.append(_packet_from_mapping(row, index=index))
    return packets


def _parse_packet_evidence_csv(text: str) -> list[PacketEvidence]:
    packets: list[PacketEvidence] = []
    for index, row in enumerate(csv.DictReader(text.splitlines())):
        packets.append(_packet_from_mapping(row, index=index))
    return packets


def _packet_from_mapping(row: dict[str, Any], *, index: int) -> PacketEvidence:
    raw_delivered = row.get("delivered", False)
    delivered = raw_delivered
    if isinstance(raw_delivered, str):
        delivered = raw_delivered.strip().lower() in {
            "1",
            "true",
            "yes",
            "delivered",
        }
    try:
        delay_ms = float(row.get("delay_ms") or 0.0)
    except (TypeError, ValueError):
        delay_ms = 0.0
    return PacketEvidence(
        packet_id=str(row.get("packet_id") or f"online_packet_{index}"),
        delivered=bool(delivered),
        delay_ms=delay_ms,
    )
