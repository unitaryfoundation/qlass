qlass.vqe package
==============

The ``vqe`` module provides implementations of Variational Quantum Eigensolver algorithms tailored for photonic quantum computing.

Key Features
------------

**Executor Types**

The VQE class supports three executor types:

- ``sampling``: Uses sampling from quantum processors (default)
- ``qubit_unitary``: Uses unitary matrices directly for qubit states
- ``photonic_unitary``: Uses photonic unitaries with dual-rail encoding and post-selection

**Optimization Algorithms**

- **Standard VQE**: Minimizes ground state energy using ``cost="VQE"``
- **Ensemble-VQE (e-VQE)**: Computes multiple states simultaneously using ``cost="e-VQE"`` with configurable weighting schemes:
  
  - ``weighted``: Linearly decreasing weights
  - ``equi``: Equal weights for all states
  - ``ground_state_only``: Only ground state contributes

**Ansätze**

- ``hf_ansatz``: Hartree-Fock based ansatz supporting:
  
  - ``method="WFT"``: Wave function theory
  - ``method="DFT"``: Density functional theory
  - Compatible with both VQE and e-VQE costs

VQE Class
---------

.. autoclass:: qlass.vqe.VQE
   :members:
   :show-inheritance:

   **Parameters:**
   
   - ``hamiltonian`` (Dict[str, float]): Hamiltonian as Pauli strings with coefficients
   - ``executor`` (Callable): Function that executes the quantum circuit/unitary
   - ``num_params`` (int): Number of variational parameters
   - ``optimizer`` (str): Optimization method (default: "COBYLA")
   - ``executor_type`` (str): Type of executor - "sampling", "qubit_unitary", or "photonic_unitary"
   - ``initial_state`` (np.ndarray): Initial quantum state (for photonic_unitary)
   - ``ancillary_modes`` (List[int]): Ancillary mode indices for post-selection (photonic_unitary)

   **run() method parameters:**
   
   - ``initial_params``: Starting parameters (default: random)
   - ``max_iterations``: Maximum optimization iterations (default: 100)
   - ``verbose``: Print progress information (default: True)
   - ``weight_option``: Weighting scheme for e-VQE ("weighted", "equi", "ground_state_only")
   - ``cost``: Cost function type ("VQE" or "e-VQE")

Ansatz Module
-------------

.. automodule:: qlass.vqe.ansatz
   :members:
   :show-inheritance:
   :undoc-members:

Submodules
----------

.. automodule:: qlass.vqe
   :members:
   :show-inheritance:
   :undoc-members: