Introduction
============

Photonic quantum computing represents a powerful approach to building scalable quantum computers, offering advantages such as avoiding the need for dilution fridges, providing natural connectivity to quantum communication networks :cite:p:`kok2007linear, o2009photonic` and, as in the case of QLASS devices, allowing for reconfigurable chips. However, the translation of quantum algorithms from the abstract circuit model to photonic implementations presents unique challenges. Existing quantum software frameworks like Qiskit :cite:p:`aleksandrowicz2019qiskit` and Cirq :cite:p:`cirq2021` primarily target gate-based quantum computers with qubit architectures, while photonic platforms operate on fundamentally different principles using linear optical elements, photons, and modes. 

There are only a few open source software projects that focus on the simulation or compilation of photonic quantum computers. These include Perceval :cite:p:`heurtel2023perceval`, Strawberry  Fields :cite:p:`killoran2019strawberry`, Piquasso :cite:p:`kolarovszki2025piquasso` and Graphix :cite:p:`sunami2022graphix`, all Python packages. These tools are primarily focussed on photonic circuit compilation and simulation. We are however interested in specific applications of photonic quantum computing. In particular in applications related to quantum chemistry. As far as we know, there is no software platform that can run VQE simulations on photonic quantum computing platforms starting with an ansatz defined in the qubit architecture all the way to running the VQE on a photonic quantum computer to obtain ground state energies for molecular Hamiltonians.

What is qlass?
-------------

``qlass`` addresses this gap by providing:

1. **Quantum chemistry integration** with established packages (OpenFermion, PySCF) for molecular problem definition.
2. **Optimized VQE implementations** specifically designed for the constraints and capabilities of photonic quantum computers.
3. **Hardware-specific resource estimation and compilation** from quantum circuits (Qiskit) to linear optics layout of photonic processors (Perceval), including the analysis of circuits against realistic hardware constraints such as photon loss, detector efficiency, and fusion gate success rates.

The package is particularly valuable for researchers working at the intersection of quantum algorithms and photonic quantum computing, as it provides the necessary tools to evaluate algorithm performance under realistic hardware conditions and optimize implementations for photonic architectures.

Key Features
-----------

- **Circuit Compilation**: Convert Qiskit quantum circuits to Perceval photonic processors
- **Multiple VQE Modes**: 
  - Standard VQE for ground state optimization
  - Ensemble-VQE (e-VQE) for computing multiple excited states
  - Support for sampling, qubit unitary, and photonic unitary executors
- **Quantum Chemistry Tools**: 
  - Generate molecular Hamiltonians (LiH)
  - Kohn-Sham Hamiltonians for DFT-based calculations
  - Hamiltonian manipulation utilities
- **Advanced Ansätze**: Hartree-Fock ansatz with WFT and DFT support
- **Utility Functions**: Process measurement results and compute expectation values

Use Cases
--------

``qlass`` is designed for researchers and developers who want to:

1. Explore quantum algorithms on photonic hardware
2. Study molecular systems using variational quantum algorithms
3. Develop hybrid quantum-classical algorithms for chemistry simulations
4. Benchmark the performance of photonic quantum computers

Project Background
----------------

The ``qlass`` package is developed by Unitary Foundation as part of the Quantum Glass-based Photonic Integrated Circuits (QLASS) project, 
which aims to develop quantum technologies based on photonic integrated circuits. The project is funded by the European Union.

.. bibliography:: refs.bib
   :filter: docname in docnames
