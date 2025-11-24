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

**Ansätze**

- ``hf_ansatz``: Hartree-Fock based ansatz supporting:
  
  - ``method="WFT"``: Wave function theory
  - ``method="DFT"``: Density functional theory :cite:p:`kohn1965self`
  - Compatible with both VQE and e-VQE costs

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