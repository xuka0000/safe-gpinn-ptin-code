# Progressive PTIN Scenario Overlay

This directory defines the progressive-disaster branch for the closed-loop PTIN
experiments. It reuses the official reconstructed PTIN data in
`../scenario_reconstruction_official_v1` and changes only the disaster timing
profile.

The four failed PDN lines are released over steps 0, 0, 3 and 5. This keeps the
network, dependency and resource data comparable with the instantaneous batch03
experiment while providing a formal test case for the rolling-horizon MPC claim.
