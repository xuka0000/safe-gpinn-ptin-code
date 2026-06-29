# Scenario Reconstruction Official V1

This folder will replace the synthetic V0 benchmark with an
official-figure-derived benchmark reconstructed from the Applied Energy final
article:

- DOI: https://doi.org/10.1016/j.apenergy.2025.125716
- PII: `S0306261925004465`
- Figure assets:
  `references/applied_energy_2025_official_figures/`

## Evidence Labels

Every row should include a `source_evidence` field:

| Label | Meaning |
|---|---|
| `official_final_figure` | Node/link/status visible in Applied Energy final-version figures. |
| `standard_benchmark` | Value imported from standard IEEE 37 or Sioux-Falls benchmark data. |
| `derived_from_figure` | Numeric coordinate or mapping digitized from a final-version figure. |
| `synthetic_fill` | Assumption added only because no public machine-readable data was available. |

## Generated Files

```text
pdn_nodes.csv
pdn_edges.csv
utn_nodes.csv
utn_edges.csv
cn_nodes.csv
cn_edges.csv
cn_edges_distance_rule_candidates.csv
dependencies.csv
figure_coordinate_map.csv
official_v1_audit.json
```

The following generated files now exist:

```text
pdn_nodes.csv
pdn_edges.csv
utn_nodes.csv
utn_edges.csv
cn_nodes.csv
cn_edges.csv
dependencies.csv
resource_fleet.yaml
disaster_scenario.yaml
ptin_case_config.yaml
mobility_config.yaml
```

Build command:

```powershell
python -X utf8 -m ptin_sim.build_official_v1_scenario
```

Validation command:

```powershell
python -X utf8 -m ptin_sim.scenario_loader --scenario-dir .\data\scenario_reconstruction_official_v1
```

Validated counts:

| Item | Count |
|---|---:|
| PDN nodes | 37 |
| PDN edges | 36 |
| CN nodes | 43 |
| CN edges | 42 |
| UTN nodes | 24 |
| UTN edges | 38 |
| dependencies | 147 |
| MESS/UGV units | 5 |
| UAV units | 4 |

Warnings:

- Appendix seed CSVs are manually transcribed and need second-pass proofreading.
- `cn_edges.csv` is a first-pass manual visual read from Fig. 6(b) and still
  needs second-pass proofreading.
- `cn_edges_distance_rule_candidates.csv` preserves the previous 93 coordinate
  distance-rule links for audit and fallback.
- V2G bus assignments are provisional visual readings from Fig. 5.

## Reconstruction Priority

1. Use `gr6` for separated PDN, CN, and UTN topology.
2. Use `gr7` for initial disaster/failure state.
3. Use `gr8` for restoration sequence validation.
4. Use `gr5` for integrated spatial placement and UE5 scene layout.

The old `data/scenario_reconstruction/` folder remains synthetic V0 and should
not be used for final data-support claims.
