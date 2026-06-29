from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from .run_closed_loop_experiment import (
    ClosedLoopRunConfig,
    _build_env,
    _config_to_json,
    _replace_config,
    load_run_config,
)
from .scenario import load_closed_loop_scenario
from .types import ClosedLoopAction


ONLINE_RL_TRUTH_BOUNDARY = "online_rl_training_interacts_with_closed_loop_env"


def run_online_q_learning(
    *,
    config: ClosedLoopRunConfig | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    adapter_mode: str | None = None,
    episodes: int = 80,
    seed: int = 7,
    alpha: float = 0.4,
    gamma: float = 0.95,
    epsilon_start: float = 0.8,
    epsilon_end: float = 0.05,
) -> dict[str, Any]:
    run_config = config if config is not None else load_run_config(config_path)
    if adapter_mode is not None:
        run_config = _replace_config(run_config, adapter_mode=adapter_mode)
    scenario = load_closed_loop_scenario(run_config.scenario_dir)
    run_dir = Path(output_dir) if output_dir is not None else run_config.output_root / "online_q_learning"
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    q_table: dict[str, dict[str, float]] = {}
    training_rows: list[dict[str, Any]] = []
    step_trace_rows: list[dict[str, Any]] = []

    for episode in range(1, int(episodes) + 1):
        epsilon = _linear_epsilon(
            episode=episode,
            episodes=max(1, int(episodes)),
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
        )
        env = _build_env(run_config, scenario)
        observation = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        while not terminated and not truncated:
            actions = env.available_actions()
            if not actions:
                break
            state_key = _state_key(observation)
            action = _epsilon_greedy_action(
                q_table,
                state_key=state_key,
                actions=actions,
                epsilon=epsilon,
                rng=rng,
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_actions = env.available_actions()
            _q_update(
                q_table,
                state_key=state_key,
                action_id=_action_key(action),
                reward=reward,
                next_state_key=_state_key(next_observation),
                next_actions=[_action_key(item) for item in next_actions],
                alpha=alpha,
                gamma=gamma,
                terminal=terminated or truncated,
            )
            total_reward += reward
            observation = next_observation
            step_trace_rows.append(
                {
                    "phase": "train",
                    "episode": episode,
                    "epsilon": round(epsilon, 6),
                    **env.trace_rows()[-1],
                    "info_block_reason": info.get("block_reason", ""),
                }
            )
        training_rows.append(
            {
                "episode": episode,
                "epsilon": round(epsilon, 6),
                "total_reward": round(total_reward, 9),
                "steps": observation["step_index"],
                "remaining_failed_edge_count": observation["remaining_failed_edge_count"],
                "restored_failed_edge_count": observation["restored_failed_edge_count"],
            }
        )

    eval_row, eval_trace = _evaluate_q_table(run_config, scenario, q_table)
    for row in eval_trace:
        step_trace_rows.append({"phase": "eval", "episode": "eval", **row})

    _write_csv(run_dir / "online_rl_training_curve.csv", training_rows)
    _write_csv(run_dir / "online_rl_eval.csv", [eval_row])
    _write_csv(run_dir / "online_rl_step_trace.csv", step_trace_rows)
    (run_dir / "q_table.json").write_text(
        json.dumps(q_table, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "version": "closed_loop_online_q_learning_v1_20260618",
        "truth_boundary": ONLINE_RL_TRUTH_BOUNDARY,
        "adapter_mode": run_config.adapter_mode,
        "episodes": int(episodes),
        "seed": int(seed),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "epsilon_start": float(epsilon_start),
        "epsilon_end": float(epsilon_end),
        "config_resolved": _config_to_json(run_config),
        "eval": eval_row,
        "outputs": {
            "training_curve_csv": "online_rl_training_curve.csv",
            "eval_csv": "online_rl_eval.csv",
            "step_trace_csv": "online_rl_step_trace.csv",
            "q_table_json": "q_table.json",
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def _evaluate_q_table(
    config: ClosedLoopRunConfig,
    scenario: Any,
    q_table: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = _build_env(config, scenario)
    observation = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    selected: list[str] = []
    while not terminated and not truncated:
        actions = env.available_actions()
        if not actions:
            break
        action = _greedy_action(q_table, _state_key(observation), actions)
        selected.append(_action_key(action))
        observation, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
    return (
        {
            "method": "online_q_learning",
            "selected_sequence": ">".join(selected),
            "total_reward": round(total_reward, 9),
            "steps": observation["step_index"],
            "restored_failed_edge_count": observation["restored_failed_edge_count"],
            "remaining_failed_edge_count": observation["remaining_failed_edge_count"],
            "terminated": int(terminated),
            "truncated": int(truncated),
        },
        env.trace_rows(),
    )


def _epsilon_greedy_action(
    q_table: dict[str, dict[str, float]],
    *,
    state_key: str,
    actions: list[ClosedLoopAction],
    epsilon: float,
    rng: random.Random,
) -> ClosedLoopAction:
    if rng.random() < epsilon:
        return rng.choice(actions)
    return _greedy_action(q_table, state_key, actions)


def _greedy_action(
    q_table: dict[str, dict[str, float]],
    state_key: str,
    actions: list[ClosedLoopAction],
) -> ClosedLoopAction:
    values = q_table.get(state_key, {})
    return max(
        actions,
        key=lambda action: (
            values.get(_action_key(action), 0.0),
            1 if action.target_id else 0,
            1 if str(action.metadata.get("communication_mode") or "direct") == "uav_relay" else 0,
            _action_key(action),
        ),
    )


def _action_key(action: ClosedLoopAction) -> str:
    if action.action_id:
        return str(action.action_id)
    if action.target_id:
        return str(action.target_id)
    return str(action.action_type)


def _q_update(
    q_table: dict[str, dict[str, float]],
    *,
    state_key: str,
    action_id: str,
    reward: float,
    next_state_key: str,
    next_actions: list[str],
    alpha: float,
    gamma: float,
    terminal: bool,
) -> None:
    state_values = q_table.setdefault(state_key, {})
    current = float(state_values.get(action_id, 0.0))
    next_values = q_table.get(next_state_key, {})
    bootstrap = 0.0 if terminal or not next_actions else max(
        float(next_values.get(action_id, 0.0)) for action_id in next_actions
    )
    state_values[action_id] = current + alpha * (reward + gamma * bootstrap - current)


def _state_key(observation: dict[str, Any]) -> str:
    restored = observation.get("restored_pdn_edges") or []
    return ">".join(str(item) for item in restored) if restored else "__start__"


def _linear_epsilon(
    *,
    episode: int,
    episodes: int,
    epsilon_start: float,
    epsilon_end: float,
) -> float:
    if episodes <= 1:
        return float(epsilon_end)
    ratio = (episode - 1) / (episodes - 1)
    return float(epsilon_start + ratio * (epsilon_end - epsilon_start))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adapter-mode", default=None)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=0.8)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_online_q_learning(
        config_path=args.config,
        output_dir=args.output_dir,
        adapter_mode=args.adapter_mode,
        episodes=args.episodes,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
    )
    print("Closed-loop online Q-learning")
    print(f"- adapter_mode: {manifest['adapter_mode']}")
    print(f"- episodes: {manifest['episodes']}")
    print(f"- truth_boundary: {manifest['truth_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
