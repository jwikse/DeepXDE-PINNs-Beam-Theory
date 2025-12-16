**Physics-Informed Neural Networks for Euler–Bernoulli Beam Problems**

This repository contains an implementation of Physics-Informed Neural Networks (PINNs) using the DeepXDE library to solve classical Euler–Bernoulli beam problems. The project demonstrates how PINNs can be applied to both forward and inverse mechanics problems across several beam configurations and loading conditions.

**Forward PINN Models**

PINNs are trained to compute beam deflection for multiple structural configurations:

1. Cantilever beam

2. Simply supported beam

2. Propped cantilever beam

Each configuration is evaluated under:

1. Uniform distributed load

2. Point load

The models enforce the Euler–Bernoulli beam PDE and corresponding boundary conditions directly through DeepXDE's automatic differentiation and PDE residual formulation.

**Inverse PINN Models**

Inverse problems are implemented where PINNs estimate the Young’s modulus E from displacement data:

1. Using the full displacement field, or

2. Using only the maximum deflection

Both inverse PINN setups show high accuracy, demonstrating the effectiveness of DeepXDE for parameter identification in structural mechanics.
