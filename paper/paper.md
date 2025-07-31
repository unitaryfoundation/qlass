---
title: 'qlass: A Python Package for Variational Quantum Algorithms on Photonic Devices'
tags:
  - Python
  - quantum computing
  - photonics
  - variational quantum eigensolver
  - quantum chemistry
  - linear optics
authors:
  - name: Farrokh Labib
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Nathan Shammah
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Unitary Foundation
   index: 1
date: 30 July 2025
bibliography: paper.bib
---

# Summary

`qlass` is a Python package designed to enable the execution of variational quantum algorithms on photonic quantum computing devices. As part of the Quantum Glass-based Photonic Integrated Circuits (QLASS) project funded by the European Union, this package bridges the gap between quantum algorithm development in popular frameworks like Qiskit and their implementation on photonic hardware using Perceval. The package provides tools for circuit compilation, resource estimation, and specialized implementations of the Variational Quantum Eigensolver (VQE) algorithm optimized for linear optical quantum computing platforms.

# Statement of need

Photonic quantum computing represents a promising approach to building scalable quantum computers, offering advantages such as room-temperature operation, high-fidelity gates, and natural connectivity to quantum communication networks [@kok2007linear; @obrien2009photonic]. However, the translation of quantum algorithms from the abstract circuit model to photonic implementations presents unique challenges. Existing quantum software frameworks like Qiskit [@aleksandrowicz2019qiskit] and Cirq [@cirq2021] primarily target gate-based quantum computers with qubit architectures, while photonic platforms operate on fundamentally different principles using linear optical elements, photons, and modes.

`qlass` addresses this gap by providing:

1. **Seamless compilation** from Qiskit quantum circuits to Perceval photonic processors, enabling researchers to leverage existing quantum algorithm implementations.
2. **Resource-aware compilation** that analyzes circuits against realistic hardware constraints including photon loss, detector efficiency, and fusion gate success rates.
3. **Optimized VQE implementations** specifically designed for the constraints and capabilities of photonic quantum computers.
4. **Quantum chemistry tools** integrated with established packages (OpenFermion, PySCF) for molecular simulation applications.

The package is particularly valuable for researchers working at the intersection of quantum algorithms and photonic quantum computing, as it provides the necessary tools to evaluate algorithm performance under realistic hardware conditions and optimize implementations for photonic architectures.

# Package Architecture and Features

## Circuit Compilation

The main functionality of `qlass` centers on the `compile()` function, which translates Qiskit `QuantumCircuit` objects into Perceval `Processor` objects. This translation handles the mapping between the abstract gate model and the physical implementation using beam splitters, phase shifters, and photon detectors:

```python
from qiskit import QuantumCircuit
from qlass import compile

# Create a Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Compile to photonic processor
processor = compile(qc)
```

The compilation process supports multiple backend strategies ("Naive", "SLOS") and includes options for post-selection and custom input states, providing flexibility for different photonic architectures and encoding schemes.

## Resource-Aware Analysis

A distinguishing feature of `qlass` is its resource-aware compiler, which provides detailed analysis of quantum circuits in the context of specific hardware configurations. The `ResourceAwareCompiler` class models realistic photonic chip parameters:

```python
from qlass.compiler import ResourceAwareCompiler, HardwareConfig

config = HardwareConfig(
    photon_loss_component_db=0.05,
    fusion_success_prob=0.11,
    hom_visibility=0.95
)

compiler = ResourceAwareCompiler(config=config)
processor = compiler.compile(qc)
```

The analysis includes component counts, estimated photon loss, and overall success probability calculations that account for source efficiency, detector efficiency, and gate fidelities. This enables researchers to make informed decisions about algorithm design and hardware requirements.

## Variational Quantum Eigensolver Implementation

The package provides a complete VQE framework optimized for photonic quantum computing. The implementation includes:

- **Linear Entangled Ansatz**: A hardware-efficient ansatz specifically designed for photonic implementations
- **Custom Unitary Ansatz**: Support for arbitrary unitary transformations
- **Automatic Pauli Grouping**: Optimization of measurement strategies by grouping commuting Pauli terms
- **Flexible Executor Interface**: Support for different simulation backends and measurement formats

```python
from qlass.vqe import VQE, le_ansatz
from qlass.quantum_chemistry import LiH_hamiltonian
from perceval.algorithm import Sampler

# Generate molecular Hamiltonian
hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)

# Define executor for photonic simulation
def executor(params, pauli_string):
    processor = le_ansatz(params, pauli_string)
    sampler = Sampler(processor)
    return sampler.samples(10_000)

# Run VQE optimization
vqe = VQE(hamiltonian=hamiltonian, executor=executor, num_params=4)
energy = vqe.run(max_iterations=10)
```

## Quantum Chemistry Integration

`qlass` integrates with established quantum chemistry packages to provide molecular Hamiltonians suitable for VQE calculations. The package uses OpenFermion [@mcclean2020openfermion] for fermionic operator transformations and PySCF [@sun2018pyscf] for electronic structure calculations. Key features include:

- Active space reduction for efficient qubit utilization.
- Multiple transformation schemes (Jordan-Wigner, Bravyi-Kitaev).
- Automatic handling of symmetries and conservation laws.
- Support for various molecular systems with customizable parameters.

# Implementation Details

The package is structured into four main modules:

1. **`compiler`**: Handles circuit translation and resource analysis.
2. **`quantum_chemistry`**: Provides Hamiltonian generation and classical solutions.
3. **`vqe`**: Implements variational algorithms and ansätze.
4. **`utils`**: Contains utility functions for measurement processing and expectation value calculations.

The implementation leverages several design patterns to ensure extensibility:

- **Abstract executor interface**: Allows integration with different quantum backends.
- **Modular ansatz design**: Enables easy addition of new variational forms.
- **Flexible measurement handling**: Supports multiple output formats from different simulators.

# Performance and Validation

The package includes comprehensive test coverage validating:

- Correct compilation of standard quantum gates to photonic implementations.
- Accuracy of resource estimation.
- VQE pipeline tests for small molecular systems.
- Compatibility with both Perceval and Qiskit simulation backends.

Performance optimizations include:
- Efficient Pauli string grouping algorithms to minimize measurement overhead
- Transpilation optimization for reduced circuit depth
- Numba-accelerated classical computations for large Hamiltonians

# Conclusions and Future Work

`qlass` provides essential infrastructure for implementing variational quantum algorithms on photonic quantum computers. By bridging the gap between abstract quantum circuits and photonic hardware implementations, it enables researchers to explore the potential of photonic platforms for quantum chemistry and optimization problems.

Future development priorities include:
- Integration with additional photonic simulation platforms.
- Optimization algorithms specifically designed for photonic architectures.
- Extended support for error mitigation techniques in the photonic context.

The package is actively maintained by the Unitary Foundation as part of the QLASS project, with contributions welcome from the quantum computing community.

# Acknowledgements

We acknowledge support from the European Union through the QLASS project (grant agreement 101135876). We thank the contributors to the open-source quantum computing ecosystem, particularly the developers of Qiskit, Perceval, OpenFermion, and PySCF.

# References