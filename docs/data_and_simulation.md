# Data and Simulation Notes

SAFE-GPINN uses a table-first PTIN case description so that each layer can be audited before a coupled run.

## Power Layer

The power layer is built from `pdn_nodes.csv` and `pdn_edges.csv`. Each bus records load, voltage base, coordinates and criticality. Each edge records switch state, impedance and length. The pandapower adapter constructs a balanced AC network from these tables, opens failed or unrepaired switches, and computes served load, shed load, voltage and line loading.

## Traffic Layer

The transportation layer is built from `utn_nodes.csv` and `utn_edges.csv`. These tables define the road graph used by the mobile resource and relay actions. The SUMO path is handled through `sumo_online_traci_adapter.py` and `closed_loop/traffic_traci.py`. SUMO or replayed SUMO edge feedback provides edge travel time, speed, waiting time and occupancy. The environment converts these quantities into robust route time, traffic feasibility and action duration.

This design separates route-dependent execution time from power-grid feasibility. A restoration action can therefore be power-feasible but still delayed or blocked by the transportation layer.

## Communication Layer

The communication layer is built from `cn_nodes.csv`, `cn_edges.csv` and packet evidence. ns-3 replay data or an online ns-3 command returns packet delivery and delay. The environment converts this evidence into packet delivery rate, mean delay and controller availability. UAV relay and dual-channel modes can improve delivery but also add coordination delay.

## Coupling Tables

`dependencies.csv` links the three networks. The closed-loop environment uses these mappings to test whether a restoration candidate has enough power support, route feasibility and communication control authority before it is treated as executable.

## Boundary

The public repository contains the code needed to inspect the interface and replay evidence. A full physical run needs the external simulator stack and should report its adapter mode, configuration file, seed count and generated result path.

