Introduction
============

What is qlass?
-------------

``qlass`` is a Python package designed to facilitate quantum computing on photonic devices. It is part of the Quantum Glass-based Photonic Integrated Circuits (QLASS) project funded by the European Union.

The package provides tools for compiling quantum algorithms written in Qiskit to run on photonic quantum computers, with a particular focus on variational quantum algorithms like the Variational Quantum Eigensolver (VQE).

Key Features
-----------

- **Circuit Compilation**: Convert Qiskit quantum circuits to Perceval photonic processors
- **VQE Implementation**: Run Variational Quantum Eigensolver algorithms on photonic hardware
- **Quantum Chemistry Tools**: Generate and manipulate molecular Hamiltonians
- **Utility Functions**: Process measurement results and compute expectation values

Architecture
-----------

The ``qlass`` package is organized into several modules:

- **compiler**: Tools for converting quantum circuits to photonic processors
- **quantum_chemistry**: Functions for generating and analyzing molecular Hamiltonians
- **vqe**: Implementations of variational quantum ansatzes for photonic computing
- **utils**: Utility functions for algorithm execution and result processing

Use Cases
--------

``qlass`` is designed for researchers and developers who want to:

1. Explore quantum algorithms on photonic hardware
2. Study molecular systems using variational quantum algorithms
3. Develop hybrid quantum-classical algorithms for chemistry simulations
4. Benchmark the performance of photonic quantum computers

Project Background
----------------

The ``qlass`` package is developed by the Unitary Foundation as part of the QLASS project, which aims to develop quantum technologies based on photonic integrated circuits. The project is funded by the European Union.