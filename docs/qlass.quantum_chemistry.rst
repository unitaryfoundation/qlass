qlass.quantum\_chemistry package
============================

The ``quantum_chemistry`` module provides tools for generating and manipulating quantum chemistry Hamiltonians for VQE algorithms to simulate molecular systems.

Key Features
------------

**Hamiltonian Generation**

- ``LiH_hamiltonian``: Generates Hamiltonians for lithium hydride molecules, a standard benchmark in quantum chemistry.
- ``LiH_hamiltonian_tapered``: Tapered version with reduced qubit count utilizing symmetry reduction.
- ``Hchain_KS_hamiltonian``: Generates Kohn-Sham Hamiltonians for hydrogen chains. This utilizes Density Functional Theory (DFT), specifically solving the Kohn-Sham equations :cite:p:`kohn1965self` to map the electronic structure problem.

  - Returns tuple: ``(hamiltonian, scf_mo_energy, n_orbs)``
  - Parameters: ``num_atoms`` (int), ``bond_length`` (float)

**Hamiltonian Utilities**

- ``hamiltonian_matrix``: Converts Pauli string Hamiltonian to matrix form
- ``brute_force_minimize``: Computes exact ground state energy via diagonalization (Full CI) for benchmarking.

Submodules
----------

.. automodule:: qlass.quantum_chemistry
   :members:
   :show-inheritance:
   :undoc-members:

.. bibliography:: refs.bib
   :filter: docname in docnames