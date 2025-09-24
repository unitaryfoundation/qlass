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
    affiliation: "1, 2"
  - name: Nathan Shammah
    orcid: 0000-0002-8775-3667
    affiliation: "1, 2"
affiliations:
 - name: Unitary Fund France, 1 Impasse du Palais, 37000 Tours, France
   index: 1
 - name: Unitary Fund, 505 Montgomery St, 94111 San Francisco, USA
   index: 2

date: 24 September 2025
bibliography: paper.bib
---

# Summary

`qlass` is a Python package designed to enable the execution of variational quantum algorithms on photonic quantum computing devices. As part of the Quantum Glass-based Photonic Integrated Circuits (QLASS) project funded by the European Union, this package bridges the gap between quantum algorithm development in popular frameworks like Qiskit and their implementation on photonic hardware using Perceval. The package provides tools for circuit compilation, resource estimation, and specialized implementations of the Variational Quantum Eigensolver (VQE) algorithm optimized for linear optical quantum computing platforms.

# Statement of need

Photonic quantum computing represents a powerful approach to building scalable quantum computers, offering advantages such as avoiding the need for dilution fridges, reconfigurable chips, and natural connectivity to quantum communication networks [@kok2007linear; @obrien2009photonic]. However, the translation of quantum algorithms from the abstract circuit model to photonic implementations presents unique challenges. Existing quantum software frameworks like Qiskit [@aleksandrowicz2019qiskit] and Cirq [@cirq2021] primarily target gate-based quantum computers with qubit architectures, while photonic platforms operate on fundamentally different principles using linear optical elements, photons, and modes. 

There are only a few open source software projects that focus on the simulation or compilation of photonic quantum computers. These include Perceval [@heurtel2023perceval], Strawberry  Fields [@killoran2019strawberry], Piquasso [@kolarovszki2025piquasso] and Graphix [@sunami2022graphix], all Python packages. These tools are primarily focussed on photonic circuit compilation and simulation. We are however interested in specific applications of photonic quantum computing. In particular in applications related to quantum chemistry. As far as we know, there is no software platform that can run VQE simulations on photonic quantum computing platforms starting with an ansatz defined in the qubit architecture all the way to running the VQE on a photonic quantum computer to obtain ground state energies for molecular hamiltonians.

`qlass` addresses this gap by providing:

1. **Quantum chemistry integration** with established packages (OpenFermion, PySCF) for molecular problem definition.
2. **Optimized VQE implementations** specifically designed for the constraints and capabilities of photonic quantum computers.
3. **Hardware-specific resource estimation and compilation** from quantum circuits (Qiskit) to linear optics layout of photonic processors (Perceval), including the analysis of circuits against realistic hardware constraints such as photon loss, detector efficiency, and fusion gate success rates.

The package is particularly valuable for researchers working at the intersection of quantum algorithms and photonic quantum computing, as it provides the necessary tools to evaluate algorithm performance under realistic hardware conditions and optimize implementations for photonic architectures.

# Package Architecture and Features

## Complete Quantum Chemistry Pipeline

The core functionality of `qlass` provides an end-to-end pipeline for quantum chemistry simulations on photonic quantum computers. Rather than focusing on individual components, `qlass` integrates existing tools (Qiskit, Perceval, OpenFermion, PySCF) into a cohesive workflow that takes molecular systems as input and produces ground state energies using photonic hardware simulations.

The complete pipeline encompasses:

1. **Molecular hamiltonian generation**: Using OpenFermion and PySCF for electronic structure calculations
2. **Circuit compilation**: Translation from Qiskit circuits to Perceval photonic processors (leveraging existing tools).
3. **Variational algorithm execution**: Complete VQE implementation focussed on running on photonic architectures.
4. **Measurement processing**: Handling of photonic measurement results and expectation value computation.

```python
from qlass.quantum_chemistry import LiH_hamiltonian
from qlass.vqe import VQE, le_ansatz
from perceval.algorithm import Sampler

# Complete pipeline: molecule → photonic VQE → ground state energy
hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)

def executor(params, pauli_string):
    processor = le_ansatz(params, pauli_string)  # Includes compilation step
    sampler = Sampler(processor)
    return sampler.samples(10_000)

vqe = VQE(hamiltonian=hamiltonian, executor=executor, num_params=4)
energy = vqe.run(max_iterations=10)
```

While `qlass` includes a `compile()` function that wraps Perceval's QiskitConverter for convenience, the primary contribution is the integrated workflow that seamlessly connects quantum chemistry problem formulation to photonic quantum simulation for enhanced user experience.

As part of this pipeline, our package provides a complete VQE framework optimized for photonic quantum computing. The implementation includes:

- **Linear entangled ansatz**: A simple ansatz with low resource requirements.
- **Custom unitary ansatz**: Support for arbitrary unitary transformations.
- **Automatic pauli grouping**: Optimization of measurement strategies by grouping commuting Pauli terms.
- **Flexible executor interface**: Support for different simulation backends and measurement formats.

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

# Implementation Details

The package is structured into four main modules:

1. **`compiler`**: Handles circuit translation and resource analysis.
2. **`quantum_chemistry`**: Provides Hamiltonian generation and classical solutions.
3. **`vqe`**: Implements variational algorithms and ansätze.
4. **`utils`**: Contains utility functions for measurement processing and expectation value calculations.

The structure can be seen below in the diagram, highlighting the relevant folders and files:

qlass/
* docs/
* examples/
* notebooks/
*  qlass/
    * compiler/
      * compiler.py
      * hardware_config.py
    * quantum_chemistry/
      * classical_solution.py
      * hamiltonians.py
    * tests/
    * utils/
      * utils.py
    * vqe/
      * ansatz.py
      * vqe.py

The implementation leverages several design patterns to ensure extensibility. It allows integration with different quantum backends and enables easy addition of new variational forms. We also support multiple output formats from different simulator backends.

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

We acknowledge support from the European Union through the QLASS project (EU Horizon Europe grant agreement 101135876). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them. We thank the contributors to the open-source quantum computing ecosystem, particularly the developers of Qiskit, Perceval, OpenFermion, and PySCF. We would like to thank Bruno Senjean, Jean-Sébastien Filhol and Francesco Malaspina for their valuable insight and the Github users @Kitsunp and @Qubit1718 for their code contributions.

# References
