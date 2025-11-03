qlass.quantum\_chemistry package
============================

The ``quantum_chemistry`` module provides tools for generating and manipulating quantum chemistry Hamiltonians for VQE algorithms to simulate molecular systems.

Key Features
------------

**Hamiltonian Generation**

- ``LiH_hamiltonian``: Generates Hamiltonians for lithium hydride molecules
- ``LiH_hamiltonian_tapered``: Tapered version with reduced qubit count
- ``Hchain_KS_hamiltonian``: Generates Kohn-Sham Hamiltonians for hydrogen chains using DFT

  - Returns tuple: ``(hamiltonian, scf_mo_energy, n_orbs)``
  - Parameters: ``num_atoms`` (int), ``bond_length`` (float)

**Hamiltonian Utilities**

- ``hamiltonian_matrix``: Converts Pauli string Hamiltonian to matrix form
- ``brute_force_minimize``: Computes exact ground state energy via diagonalization

Submodules
----------

.. automodule:: qlass.quantum_chemistry
   :members:
   :show-inheritance:
   :undoc-members: