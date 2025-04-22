
qlass documentation
===================

**qlass**: A package to compile quantum algorithms on photonic devices, 
part of the Quantum Glass-based Photonic Integrated Circuits (QLASS) project.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   modules
   examples

Introduction
-----------

``qlass`` is a package developed to facilitate quantum computation on photonic devices.
It provides tools for compiling quantum circuits, implementing VQE algorithms,
and working with quantum chemistry Hamiltonians.

Key features include:

* Compiling Qiskit circuits to Perceval photonic processors
* Variational Quantum Eigensolver (VQE) implementation for photonic devices
* Quantum chemistry tools for molecular Hamiltonians
* Utility functions for quantum algorithm execution

Installation
-----------

The development install of ``qlass`` requirements can be done by setting the working directory to the top level of the repository and running:

.. code-block:: bash

   pip install -e .

Quick Start
----------

Compiling a Quantum Circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qiskit import QuantumCircuit
   from qlass import compile

   # Create a Qiskit circuit
   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   # Compile to Perceval processor
   processor = compile(qc)

Running a VQE Algorithm
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qlass.vqe import le_ansatz
   from qlass.utils import loss_function
   from qlass.quantum_chemistry.hamiltonians import LiH_hamiltonian
   
   # Generate a Hamiltonian for the LiH molecule
   hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)
   
   # Define an executor function that uses the linear entangled ansatz
   def executor(params, pauli_string):
       processor = le_ansatz(params, pauli_string)
       sampler = Sampler(processor)
       samples = sampler.samples(10_000)
       return samples
   
   # Run VQE optimization
   from scipy.optimize import minimize
   import numpy as np

   initial_params = np.random.rand(4)
   
   result = minimize(
       loss_function,
       initial_params,
       args=(hamiltonian, executor),
       method='COBYLA',
       options={'maxiter': 30}
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`