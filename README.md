# SAFE-GPINN PTIN Research Code

This repository is a compact research-code release for the SAFE-GPINN closed-loop PTIN restoration experiments.

Project page: https://xuka0000.github.io/safe-gpinn-ptin-project-page/

## What Can Run Online

The project page runs as a static browser site. It can show the restoration replay GIF, load result CSV files, and present the paper tables. It cannot run the full Python, SUMO, ns-3, pandapower, or PyTorch simulation backend inside GitHub Pages.

The full software is intended to run locally or in a containerized research environment. GitHub Pages should therefore be described as an online replay and result viewer, not as a live backend simulator.

## Repository Contents

- `src/ptin_sim/closed_loop` contains the closed-loop PTIN environment, adapters, policies, training utilities, seed sweep and result aggregation code.
- `src/ptin_sim/sumo_online_traci_adapter.py` connects online traffic feedback from SUMO through TraCI.
- `src/ptin_sim/ns3_online_packet_step.py` and `src/ptin_sim/ns3_ptin_packet_replay.cc` provide the packet-level communication evidence path.
- `data/scenario_reconstruction_official_v1` contains the reconstructed PTIN case tables. Progressive, stress and dynamic-EV overlays are included for the released tests and configurations.
- `configs` contains runnable closed-loop experiment configurations.
- `examples/ns3_packet_replay` contains packet-result evidence used by replay-mode runs.
- `tests` contains focused tests for the released closed-loop code.

## Quick Smoke Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m ptin_sim.scenario_loader --scenario-dir data/scenario_reconstruction_official_v1
python -m ptin_sim.closed_loop.run_closed_loop_experiment `
  --config configs/closed_loop_ptin_v1_20260617.yaml `
  --adapter-mode fake `
  --episodes 1 `
  --max-steps 2 `
  --output-dir outputs/smoke_fake
```

This smoke run checks the software interface only. It does not represent the physical results reported in the paper.

## Physical Adapter Modes

| Mode | Meaning | Extra dependencies |
|---|---|---|
| `fake` | Interface smoke test with deterministic fake adapters | none |
| `real_power_fake_traffic_comm` | pandapower AC check with fake traffic and communication | `pandapower` |
| `real_power_online_traci_fake_comm` | pandapower plus SUMO/TraCI traffic feedback | SUMO and TraCI |
| `real_power_online_traci_ns3_replay` | pandapower, SUMO/TraCI and packet-result replay | SUMO, TraCI, ns-3 replay CSV |
| `real_power_online_traci_online_ns3` | step-level ns-3 command feedback | SUMO, TraCI, ns-3 |

For the optional power adapter:

```powershell
pip install -r requirements-optional.txt
```

SUMO and ns-3 are external simulator installations. They are not Python packages and must be installed separately.

## Data and Simulator Design

The case data are stored as network tables rather than as a single opaque simulator file. The power layer uses PDN bus and line tables, active and reactive load fields, switch states, failure-release steps and mobile support variables. The traffic layer uses UTN nodes and edges as the route substrate. SUMO or SUMO-derived edge feedback supplies travel time, speed, waiting time and occupancy. These quantities enter the closed-loop environment through the TraCI adapter and determine action duration and traffic feasibility. The communication layer uses CN node and link tables, packet-delivery evidence and delay thresholds. ns-3 replay or online command feedback is mapped to packet delivery rate, mean delay and controller availability.

The reconstructed case follows the PTIN-interdependency idea used by recent mobile energy-storage microgrid restoration studies, but the released code keeps the data boundary explicit. Figure-derived entries, standard benchmark entries and synthetic fills are labeled in the scenario files where available. The current public release does not claim hardware RF validation, full city-scale online cosimulation or unbalanced field-feeder validation.

See `docs/data_and_simulation.md` for a layer-by-layer data note.

## Main Entry Points

```powershell
$env:PYTHONPATH = "src"
python -m ptin_sim.closed_loop.run_closed_loop_experiment --help
python -m pytest tests/test_closed_loop_runner.py tests/test_closed_loop_scenario.py tests/test_ns3_online_packet_step.py
```

## Citation Boundary

If this code is used in a manuscript or presentation, report which adapter mode was used. Results from `fake` mode are software smoke checks only. Paper-facing claims should cite the result CSV files and adapter mode used to generate them.
