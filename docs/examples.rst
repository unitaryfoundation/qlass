Examples
========

This section provides examples of how to use the qlass package for various quantum computing tasks on photonic devices.

Basic Circuit Compilation
-------------------------

Compiling a quantum circuit from Qiskit to a Perceval processor:

.. code-block:: python

   from qiskit import QuantumCircuit
   from qlass import compile
   
   # Create a Qiskit circuit
   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)
   
   # Compile to Perceval processor
   processor = compile(qc)
   
   # Run the processor using Perceval's Sampler
   from perceval.algorithm import Sampler
   sampler = Sampler(processor)
   results = sampler.samples(1000)

Variational Quantum Eigensolver (VQE)
-------------------------------------

Implementing VQE to find the ground state energy of a molecule:

.. code-block:: python

   import warnings
   warnings.simplefilter('ignore')
   warnings.filterwarnings('ignore')

   import numpy as np
   from perceval.algorithm import Sampler
   
   from qlass.vqe import VQE, le_ansatz
   from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize
   
   # Create a molecular Hamiltonian
   hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)
   
   # Define an executor function that runs the quantum circuit
   def executor(params, pauli_string):
       processor = le_ansatz(params, pauli_string)
       sampler = Sampler(processor)
       samples = sampler.samples(10000)
       return samples
   
    # Initialize the VQE solver with the custom executor
    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=executor,
        num_params=4, # Number of parameters in the linear entangled ansatz
    )
    
    # Run the VQE optimization
    vqe_energy = vqe.run(
        max_iterations=10,
        verbose=True
    )
   
   # Print the results
   print(f"VQE Energy: {vqe_energy:.6f}")
   
   # Compare with the exact solution
   exact_energy = brute_force_minimize(hamiltonian)
   print(f"Exact Energy: {exact_energy:.6f}")
   print(f"Energy Difference: {abs(vqe_energy - exact_energy):.6f}")

Working with Molecular Hamiltonians
-----------------------------------

Generating and analyzing molecular Hamiltonians:

.. code-block:: python

   from qlass.quantum_chemistry import LiH_hamiltonian
   from qlass.quantum_chemistry import hamiltonian_matrix, brute_force_minimize
   
   # Generate a Hamiltonian for LiH with different parameters
   hamiltonian = LiH_hamiltonian(
       R=1.5,  # Bond length in Angstroms
       charge=0,
       spin=0,
       num_electrons=2,
       num_orbitals=1
   )
   
   # Print the Hamiltonian terms
   print("Hamiltonian terms:")
   for pauli_string, coefficient in hamiltonian.items():
       print(f"  {pauli_string}: {coefficient:.6f}")
   
   # Convert to matrix form
   H_matrix = hamiltonian_matrix(hamiltonian)
   print(f"Hamiltonian matrix shape: {H_matrix.shape}")
   
   # Calculate the ground state energy
   energy = brute_force_minimize(hamiltonian)
   print(f"Ground state energy: {energy:.6f}")