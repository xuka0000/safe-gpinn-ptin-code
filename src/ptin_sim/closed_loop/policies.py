from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .types import ClosedLoopAction


class ClosedLoopPolicy(Protocol):
    evidence_boundary: str

    def select_action(self, observation: dict) -> ClosedLoopAction:
        ...


def _first_available_action(observation: dict) -> ClosedLoopAction:
    actions = observation.get("available_actions") or []
    if not actions:
        raise ValueError("closed-loop observation contains no available actions")
    row = actions[0]
    return ClosedLoopAction(
        action_id=str(row.get("action_id", "")),
        action_type=str(row.get("action_type", "")),
        target_id=str(row.get("target_id", "")),
        resource_id=str(row.get("resource_id", "")),
        metadata=dict(row.get("metadata") or {}),
    )


@dataclass(frozen=True)
class GreedyRestorationPolicy:
    evidence_boundary: str = "closed_loop_heuristic_baseline"

    def select_action(self, observation: dict) -> ClosedLoopAction:
        return _first_available_action(observation)


@dataclass(frozen=True)
class SafeGPINNClosedLoopAdapter:
    checkpoint_path: Path
    closed_loop_trained: bool = False
    evidence_boundary: str = (
        "safe_gpinn_checkpoint_adapter_requires_closed_loop_retraining"
    )

    def select_action(self, observation: dict) -> ClosedLoopAction:
        return _first_available_action(observation)


@dataclass(frozen=True)
class DiffusionClosedLoopAdapter:
    checkpoint_path: Path
    closed_loop_trained: bool = False
    evidence_boundary: str = (
        "diffusion_checkpoint_adapter_requires_closed_loop_retraining"
    )

    def select_action(self, observation: dict) -> ClosedLoopAction:
        return _first_available_action(observation)
