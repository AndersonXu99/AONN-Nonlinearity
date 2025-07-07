The `code/` directory contains three standalone implementations of electromagnetically induced transparency (EIT) models, each targeting a specific regime or physical effect:

1. **Weak-Probe Analytical Model**  
   This script derives and evaluates the closed-form solution for a three-level Λ-system in the weak-probe approximation. Under the assumption that the probe Rabi frequency is vanishingly small compared to the control field, it computes analytical expressions for the relevant density-matrix elements and the resulting linear susceptibility.  

2. **Steady-State Numerical Simulation**  
   When probe and control fields have comparable strengths, no simple analytical result exists. This code formulates the steady-state Liouville equations as a linear system, performs a matrix inversion to obtain all density-matrix components (ρ_ij), and then calculates the spatial propagation of both optical fields through the atomic cell.  

3. **Doppler-Broadened EIT Solver**  
   To include thermal motion in a warm vapor, this module integrates the steady-state solution over a Maxwell–Boltzmann velocity distribution. By sampling discrete velocity classes weighted by the Boltzmann factor, it produces a Doppler-broadened susceptibility profile and simulates its impact on the transparency window.  
