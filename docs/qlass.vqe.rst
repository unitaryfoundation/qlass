qlass.vqe package
==============

The ``vqe`` module provides implementations of Variational Quantum Eigensolver algorithms tailored for photonic quantum computing. The VQE algorithm is a hybrid quantum-classical approach originally proposed for photonic processors :cite:p:`peruzzo2014variational`.

Key Features
------------

**Executor Types**

The VQE class supports three executor types to handle different abstraction levels of the photonic hardware:

- ``sampling``: Uses sampling from quantum processors (default).
- ``qubit_unitary``: Uses unitary matrices directly for qubit states (ideal simulation).
- ``photonic_unitary``: Uses photonic unitaries with dual-rail encoding and post-selection. This explicitly models the mapping of qubits to optical modes described in LOQC architectures :cite:p:`kok2007linear`.

**Optimization Algorithms**

- **Standard VQE**: Minimizes ground state energy using ``cost="VQE"``.
- **Ensemble-VQE (e-VQE)**: Computes multiple states simultaneously using ``cost="e-VQE"``. This implements subspace-search variational quantum eigensolvers :cite:p:`nakanishi2019subspace` to find excited states.
  
  - ``weighted``: Linearly decreasing weights
  - ``equi``: Equal weights for all states
  - ``ground_state_only``: Only ground state contributes

**Circuit Inspection**

- ``VQE.draw_ansatz``: Builds a circuit for a chosen parameter vector and sends it to the
  shared ``qlass.utils.draw_circuit`` rendering utility.

**Ansätze**

- ``le_ansatz``: Linear entangled ansatz (parametrized rotations with CNOT entanglers)
- ``custom_unitary_ansatz``: Compiles an arbitrary qubit unitary into a photonic processor
- ``CSF_initial_states``: Hartree-Fock initial state for wave-function-theory Hamiltonians,
  with optional singlet excitations
- ``Bitstring_initial_states``: Computational-basis reference states for density-functional-theory
  Hamiltonians :cite:p:`kohn1965self`; compatible with both VQE and e-VQE costs
- ``kerr_ansatz``: Kerr-nonlinearity-based photonic ansatz

VQE Class
---------

.. autoclass:: qlass.vqe.VQE
   :members:
   :show-inheritance:

Ansatz Module
-------------

.. automodule:: qlass.vqe.ansatz
   :members:
   :show-inheritance:
   :undoc-members:

.. bibliography:: refs.bib
   :filter: docname in docnames
