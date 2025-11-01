# Matrix Product State (MPS) SWAP Network QUBO Solver

This repository contains the code used to produce the results of the paper "Enhancing Quantum-Inspired Tensor Network Optimization using SWAP Networks and Problem-Aware Qubit Layout".

## Contents
The repository is organized as follows:

- 'src/': Contains the source code for the MPS-TEBD QUBO solver.
- 'benchmarks/': Contains benchmark results and scripts to generate them.
- 'generate_instances/': Scripts to generate problem instances in QUBO format.
- 'figures/': Contains figures and plots used in the paper, and scripts to generate them.

The 'src' directory includes the main implementation of the proposed algorithm, along with utilities for handling QUBO problems, SWAP networks, and qubit layouts. The main implementation of the rectangular and triangular SWAP networks can be found in 'src/tebd.jl'.

## Requirements
The code is written in Julia and was tested with Julia version 1.8.1. Tensor networks simulations were performed using the ITensors.jl 0.6.19 and ITensorMPS 0.2.5 packages.

## Contact
For any questions or issues, please contact the authors of the paper.
