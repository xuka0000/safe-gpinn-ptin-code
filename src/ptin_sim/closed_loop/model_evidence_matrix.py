from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs") / "analysis" / "closed_loop_model_evidence_matrix_20260622"

BATCH11_ROOT = (
    "outputs/closed_loop_runs/"
    "formal_cross_layer_online_ns3_seed_sweep_20260620_"
    "batch11_trajectory_diffusion_scenario_tree_milp_progressive_stress_mpc"
)
BATCH11_SUMMARY = (
    f"{BATCH11_ROOT}/"
    "summary_5seed_trajectory_diffusion_scenario_tree_milp_progressive_stress_mpc"
)
BATCH13_ROOT = (
    "outputs/closed_loop_runs/"
    "formal_cross_layer_online_ns3_seed_sweep_20260620_"
    "batch13_fix_sequence_expected_return_progressive_v1"
)
BATCH13_SUMMARY = (
    f"{BATCH13_ROOT}/"
    "summary_5seed_fix_sequence_expected_return_progressive_v1"
)
BATCH14_ROOT = (
    "outputs/closed_loop_runs/"
    "formal_cross_layer_online_ns3_seed_sweep_20260620_"
    "batch14_dynamic_ev_progressive_stress_v2"
)
BATCH14_SUMMARY = (
    f"{BATCH14_ROOT}/"
    "summary_5seed_dynamic_ev_progressive_stress_v2_refreshed_eval_20260620"
)
PREDEPLOYMENT_ABLATION_ROOT = (
    "outputs/closed_loop_runs/"
    "predeployment_quality_ablation_20260622_5seed"
)
TARGET_TRAFFIC_AUDIT_SMOKE_ROOT = (
    "outputs/closed_loop_runs/"
    "target_traffic_audit_smoke_20260621"
)


def build_model_evidence_rows() -> list[dict[str, str]]:
    return [
        {
            "model_element": "pdn_switch_restoration_state",
            "mathematical_role": "Multi-step PDN line-switch restoration state with failed, restored, closed and energized edge sets.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/env.py; src/ptin_sim/closed_loop/dependencies.py; src/ptin_sim/closed_loop/scenario.py",
            "result_artifacts": f"outputs/closed_loop_runs/formal_cross_layer_online_ns3_seed_sweep_20260618_batch03_framework_solver_mpc/summary_5seed_framework_solver_mpc; {BATCH11_SUMMARY}; {BATCH13_SUMMARY}; {BATCH14_SUMMARY}",
            "boundary": "The reconstructed four-edge official case and the progressive eight-edge stress overlay are modeled through switch restoration states. Full crew scheduling is not modeled.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "balanced_ac_opf_voltage_thermal",
            "mathematical_role": "AC OPF or AC power-flow feasibility with bus-voltage and line-loading metrics.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/power_ac_opf.py",
            "result_artifacts": f"step traces with ac_status, min_voltage_pu, max_line_loading_pct, served_load_kw; {BATCH14_ROOT}/merged/merged_policy_step_trace.csv",
            "boundary": "Pandapower balanced AC OPF is attempted first and falls back to balanced AC power flow. Unbalanced phase validation is not implemented in the closed-loop controller.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "load_restoration_active_reactive",
            "mathematical_role": "Restored active and reactive loads enter OPF loads and restoration reward.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/power_ac_opf.py; src/ptin_sim/closed_loop/dependencies.py",
            "result_artifacts": f"step traces with requested_load_kw, served_load_kw, shed_load_kw, restored_load_gain_kw; {BATCH14_ROOT}/merged/merged_policy_step_trace.csv",
            "boundary": "The load increment for an action is represented by the target downstream bus load. Full topology-wide load pickup attribution remains approximate.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "mess_energy_state",
            "mathematical_role": "MESS active-power support is constrained by per-step energy availability.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/env.py; src/ptin_sim/closed_loop/dependencies.py",
            "result_artifacts": f"step traces with resource_energy_required_kwh and resource_energy_remaining_kwh; {BATCH14_ROOT}/merged/merged_policy_step_trace.csv",
            "boundary": "MESS energy is consumed by dispatch support. Detailed vehicle battery degradation and charging dynamics are not modeled.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "v2g_dispatch_support",
            "mathematical_role": "V2G active-power support enters the AC OPF support set under aggregate EV participation, time-varying availability and energy limits.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/types.py; src/ptin_sim/closed_loop/scenario.py; src/ptin_sim/closed_loop/env.py; src/ptin_sim/closed_loop/dependencies.py; src/ptin_sim/closed_loop/power_ac_opf.py",
            "result_artifacts": f"{BATCH11_ROOT}/merged/merged_policy_step_trace.csv; {BATCH13_ROOT}/merged/merged_policy_step_trace.csv; {BATCH14_ROOT}/merged/merged_policy_step_trace.csv; {BATCH14_SUMMARY}/seed_sweep_method_summary.csv; outputs/closed_loop_runs/dynamic_ev_smoke_20260620/closed_loop_policy_step_trace.csv; tests/test_closed_loop_env.py::test_closed_loop_updates_v2g_capacity_from_ev_availability_profile",
            "boundary": "The V2G model supports aggregate EV count, participation willingness, per-unit power, remaining energy and an optional time-varying availability profile for EV arrival and departure effects. The batch14 dynamic-EV formal rerun provides five-seed boundary evidence under progressive stress. Individual EV routing and user-level departure decisions are still not modeled.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "online_traci_traffic_feedback",
            "mathematical_role": "Online TraCI edge metrics provide global and target-level travel-time feedback for restoration steps.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/traffic_traci.py; src/ptin_sim/closed_loop/dependencies.py; src/ptin_sim/sumo_online_traci_adapter.py; src/ptin_sim/sumo_online_traci_worker.py",
            "result_artifacts": f"step traces with traffic_status, mean_travel_time_s, traffic_feasible; {BATCH14_ROOT}/merged/merged_policy_step_trace.csv; target-level audit trace with target_travel_time_s and target_traffic_edge_ids; {TARGET_TRAFFIC_AUDIT_SMOKE_ROOT}/closed_loop_policy_step_trace.csv; tests/test_closed_loop_env.py::test_closed_loop_trace_records_target_level_traffic_audit",
            "boundary": "The closed-loop environment uses online edge metrics as a feasibility and reward signal. Target-level audit maps each PDN restoration action to UTN edges associated with the action endpoint buses. Full vehicle route planning remains outside the current controller.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "online_ns3_packet_feedback",
            "mathematical_role": "Packet delivery, delay, contention, fading and SINR feedback gate switching controllability.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/communication_ns3.py; src/ptin_sim/ns3_online_packet_step.py; src/ptin_sim/ns3_ptin_packet_replay.cc",
            "result_artifacts": f"outputs/closed_loop_runs/closed_loop_online_ns3_cross_layer_smoke_20260618/step_trace.csv; outputs/python_backend/ns3_online_packet_step_20260617; {BATCH14_ROOT}/merged/merged_policy_step_trace.csv",
            "boundary": "The ns-3 step model uses log-distance packet replay with simplified contention, queueing, fading and SINR. Full Wi-Fi MAC and hardware RF validation are not implemented.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "cross_layer_switch_controllability",
            "mathematical_role": "PDN switch actions require communication control through powered CN nodes or UAV relay support.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/dependencies.py; src/ptin_sim/closed_loop/env.py; data/scenario_reconstruction_official_v1/dependencies.csv",
            "result_artifacts": f"step traces with switch_controller_count, powered_switch_controller_count and uav_relay_used; {BATCH14_ROOT}/merged/merged_policy_step_trace.csv",
            "boundary": "Power-to-CN and CN-to-switch dependencies are active. Fine-grained packet routes to each switch controller remain approximated by the ns-3 step schedule.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "disaster_mode_and_failure_release_profile",
            "mathematical_role": "Disaster type, control paradigm and failure-release time define whether the case is instantaneous or progressive.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/types.py; src/ptin_sim/closed_loop/scenario.py; src/ptin_sim/closed_loop/env.py; data/scenario_reconstruction_official_v1/disaster_scenario.yaml; data/scenario_reconstruction_progressive_v1/disaster_scenario.yaml; data/scenario_reconstruction_progressive_stress_v2/disaster_scenario.yaml; data/scenario_reconstruction_progressive_stress_v2_dynamic_ev/disaster_scenario.yaml",
            "result_artifacts": f"outputs/closed_loop_runs/formal_cross_layer_online_ns3_seed_sweep_20260618_batch03_framework_solver_mpc/summary_5seed_framework_solver_mpc; {BATCH11_SUMMARY}; {BATCH13_SUMMARY}; {BATCH14_SUMMARY}",
            "boundary": "The framework interface supports instantaneous and progressive failure release with wait steps. The stress-v2 and dynamic-EV overlays are extended algorithm-discrimination cases, not the original four-edge reference case.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "weighted_restoration_objective",
            "mathematical_role": "Restoration reward and learning features include raw load, critical-load priority and topology-vulnerability score.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/dependencies.py; src/ptin_sim/closed_loop/env.py; src/ptin_sim/closed_loop/training.py",
            "result_artifacts": f"{BATCH11_ROOT}/merged/merged_rollout_dataset.csv; {BATCH11_ROOT}/merged/merged_policy_step_trace.csv; {BATCH13_ROOT}/merged/merged_rollout_dataset.csv; {BATCH14_ROOT}/merged/merged_rollout_dataset.csv; {BATCH14_ROOT}/merged/merged_policy_step_trace.csv",
            "boundary": "Topology vulnerability is represented by a deterministic degree and length exposure score, not by a full separate vulnerability optimization chapter.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "robust_traffic_uncertainty_interface",
            "mathematical_role": "Traffic uncertainty enters through an interval-robust travel-time multiplier and traceable robust travel-time margins at global and target levels.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/dependencies.py; src/ptin_sim/closed_loop/env.py; src/ptin_sim/closed_loop/training.py; data/scenario_reconstruction_official_v1/disaster_scenario.yaml; data/scenario_reconstruction_progressive_v1/disaster_scenario.yaml; data/scenario_reconstruction_progressive_stress_v2/disaster_scenario.yaml; data/scenario_reconstruction_progressive_stress_v2_dynamic_ev/disaster_scenario.yaml",
            "result_artifacts": f"outputs/closed_loop_runs/formal_cross_layer_online_ns3_seed_sweep_20260618_batch03_framework_solver_mpc/summary_5seed_framework_solver_mpc; {BATCH11_ROOT}/merged/merged_policy_step_trace.csv; {BATCH13_ROOT}/merged/merged_policy_step_trace.csv; {BATCH14_ROOT}/merged/merged_policy_step_trace.csv; {TARGET_TRAFFIC_AUDIT_SMOKE_ROOT}/closed_loop_policy_step_trace.csv; tests/test_closed_loop_env.py::test_closed_loop_trace_records_target_level_traffic_audit",
            "boundary": "The current implementation provides an interval-robust traffic feasibility interface, target-level UTN-edge travel-time audit and four-branch scenario tree scoring in the formal reruns. It is not an unbounded scenario generator over all traffic states or a full path-level route optimizer.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "predeployment_quality_transfer",
            "mathematical_role": "Predeployment quality is passed as an initial-condition parameter into restoration state, traces and learning features.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/scenario.py; src/ptin_sim/closed_loop/env.py; src/ptin_sim/closed_loop/training.py; data/scenario_reconstruction_official_v1/disaster_scenario.yaml; data/scenario_reconstruction_progressive_v1/disaster_scenario.yaml; data/scenario_reconstruction_progressive_stress_v2/disaster_scenario.yaml; data/scenario_reconstruction_progressive_stress_v2_dynamic_ev/disaster_scenario.yaml",
            "result_artifacts": f"{BATCH11_ROOT}/merged/merged_rollout_dataset.csv; {BATCH11_ROOT}/merged/merged_policy_step_trace.csv; {BATCH13_ROOT}/merged/merged_rollout_dataset.csv; {BATCH14_ROOT}/merged/merged_rollout_dataset.csv; {PREDEPLOYMENT_ABLATION_ROOT}/high_predeploy/summary_5seed/seed_sweep_method_summary.csv; {PREDEPLOYMENT_ABLATION_ROOT}/high_predeploy/summary_5seed/seed_sweep_pairwise_ci.csv; {PREDEPLOYMENT_ABLATION_ROOT}/low_predeploy/summary_5seed/seed_sweep_method_summary.csv; {PREDEPLOYMENT_ABLATION_ROOT}/low_predeploy/summary_5seed/seed_sweep_pairwise_ci.csv",
            "boundary": "The restoration model consumes a predeployment-quality index. The 2026-06-22 five-seed ablation changes mobile-resource readiness and V2G availability under the same dynamic-EV stress case and compares high and low predeployment-quality settings with matched seeds. It does not solve the upstream predeployment optimization inside the same closed-loop experiment.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "two_stage_deterministic_exact_solver",
            "mathematical_role": "For instantaneous disasters, all feasible repair sequences are evaluated. For the progressive stress case, a binary MILP sequence reference is inserted into the candidate pool before closed-loop replay.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/training.py; src/ptin_sim/closed_loop/env.py",
            "result_artifacts": f"outputs/closed_loop_runs/formal_cross_layer_online_ns3_seed_sweep_20260618_batch03_framework_solver_mpc/summary_5seed_framework_solver_mpc; {BATCH11_SUMMARY}; {BATCH13_SUMMARY}; {BATCH14_SUMMARY}",
            "boundary": "The exact solver is complete for the reconstructed four-edge official case through full sequence enumeration. In the progressive stress sweeps, a binary MILP assignment reference for restoration order is inserted into the candidate pool and evaluated through closed-loop replay. This is still not a general large-scale network-restoration MILP over all continuous AC, traffic and packet variables.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "progressive_rolling_mpc_policy",
            "mathematical_role": "For progressive disasters, the rolling policy selects the next action from the current prefix by aggregating four scenario branches over rollout-tree return-to-go, while wait actions preserve latent failures until release.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/training.py; src/ptin_sim/closed_loop/env.py; src/ptin_sim/closed_loop/types.py",
            "result_artifacts": f"{BATCH11_SUMMARY}; {BATCH13_SUMMARY}; {BATCH14_SUMMARY}",
            "boundary": "The current rolling policy uses four-branch scenario tree scoring over reconstructed cases. It is not an unbounded stochastic programming solver over all possible disaster scenarios.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "safe_gpinn_shielded_rollout_policy",
            "mathematical_role": "The closed-loop proposed policy combines compact learned return scoring with a hard rollout shield that follows the best feasible sampled restoration sequence when prefix evidence is available.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/training.py",
            "result_artifacts": f"{BATCH11_SUMMARY}/seed_sweep_pairwise_ci.csv; {BATCH11_SUMMARY}/seed_sweep_method_summary.csv; {BATCH13_SUMMARY}/seed_sweep_pairwise_ci.csv; {BATCH13_SUMMARY}/seed_sweep_method_summary.csv; {BATCH14_SUMMARY}/seed_sweep_pairwise_ci.csv; {BATCH14_SUMMARY}/seed_sweep_method_summary.csv",
            "boundary": "The shielded policy can be claimed as a learning-assisted safe rollout policy. In closed-loop training.py, SAFE-GPINN is a compact ScoreNet trained with return-to-go MSE plus batch-level pairwise order loss, and evaluation follows the best rollout sequence before falling back to the learned score head. Context-wise pairwise Graph-PINN claims belong to the full_algo exact-playback evidence, not to this closed-loop result. In the batch11 trajectory-diffusion stress rerun, SAFE-GPINN matches trajectory diffusion, rolling MPC, two-stage exact and IQL within numerical tolerance, and paired CIs support gains over greedy and load-gain. In the batch13 supplement, repaired expected-return sequence scoring makes SAFE-GPINN match the strongest learned, diffusion and exact references while retaining positive CIs over greedy and load-gain. The batch14 dynamic-EV boundary evidence supports SAFE-GPINN over load-gain only, while rolling MPC and TD3+BC are stronger on that refreshed evaluation. It must not be claimed as significantly superior to all strong baselines.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "full_wifi_mac_or_hardware_rf",
            "mathematical_role": "Full wireless MAC, channel access, device stack and hardware RF validation.",
            "status": "not_implemented",
            "code_paths": "src/ptin_sim/ns3_ptin_packet_replay.cc",
            "result_artifacts": "none for full Wi-Fi MAC or hardware RF",
            "boundary": "Only simplified ns-3 packet feedback is implemented.",
            "manuscript_decision": "withdraw_or_mark_future_work",
        },
        {
            "model_element": "online_q_learning_training",
            "mathematical_role": "A baseline policy is trained through iterative online interaction with the closed-loop environment.",
            "status": "implemented_with_boundary",
            "code_paths": "src/ptin_sim/closed_loop/online_rl.py",
            "result_artifacts": "online_rl_training_curve.csv; online_rl_eval.csv; online_rl_step_trace.csv; q_table.json",
            "boundary": "This is a tabular Q-learning baseline over the reconstructed four-edge case. It establishes a true online-RL training path but is not the proposed SAFE-GPINN or diffusion policy.",
            "manuscript_decision": "retain_bounded_claim",
        },
        {
            "model_element": "safe_gpinn_diffusion_online_training",
            "mathematical_role": "The proposed SAFE-GPINN and diffusion policy are trained through iterative online closed-loop RL or policy-generation updates.",
            "status": "not_implemented",
            "code_paths": "src/ptin_sim/closed_loop/training.py; src/ptin_sim/closed_loop/online_rl.py",
            "result_artifacts": "none for SAFE-GPINN or diffusion online training",
            "boundary": "Current SAFE-GPINN and diffusion training enumerates closed-loop rollout permutations and trains score or denoising models on return-to-go. It must not be called true online RL training.",
            "manuscript_decision": "withdraw_or_mark_future_work",
        },
    ]


def write_model_evidence_outputs(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = build_model_evidence_rows()
    csv_path = output / "model_to_code_evidence_matrix.csv"
    json_path = output / "model_to_code_evidence_matrix.json"
    md_path = output / "model_to_code_evidence_matrix.md"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_table(rows), encoding="utf-8")
    return {
        "row_count": len(rows),
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }


def _markdown_table(rows: list[dict[str, str]]) -> str:
    columns = [
        "model_element",
        "status",
        "code_paths",
        "boundary",
        "manuscript_decision",
    ]
    lines = [
        "# Closed-Loop PTIN Model-To-Code Evidence Matrix",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_md(row[column]) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = write_model_evidence_outputs(args.output_dir)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
