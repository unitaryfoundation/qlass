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

Resource-Aware Compilation
--------------------------

Analyzing quantum circuits against hardware configurations to estimate real-world performance:

.. code-block:: python

   from qiskit import QuantumCircuit
   from qlass.compiler import ResourceAwareCompiler, HardwareConfig, generate_report

   # Create a quantum circuit
   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   # Define hardware configuration for a photonic chip
   chip_config = HardwareConfig(
       photon_loss_component_db=0.05,
       fusion_success_prob=0.11,
       hom_visibility=0.95
   )

   # Compile with resource analysis
   compiler = ResourceAwareCompiler(config=chip_config)
   processor = compiler.compile(qc)

   # Access and display the analysis report
   report = processor.analysis_report
   generate_report(report)

The report provides insights into component counts, estimated photon loss, and overall success probability for running the circuit on the specified hardware.

Variational Quantum Eigensolver (VQE)
-------------------------------------

VQE with Sampling Executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard VQE using sampling-based executor with Hartree-Fock ansatz:

.. code-block:: python

   import numpy as np
   from qlass.vqe import VQE
   from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize
   from qlass.vqe.ansatz import hf_ansatz
   from perceval.algorithm import Sampler
   import warnings
   warnings.filterwarnings('ignore')
   
   # Create molecular Hamiltonian
   ham = LiH_hamiltonian(num_electrons=2, num_orbitals=1)
   
   # Define executor using Hartree-Fock ansatz
   def executor(params, pauli_string):
       processors = hf_ansatz(1, 1, params, pauli_string, method="WFT", cost="VQE")
       samplers = Sampler(processors)
       samples = samplers.samples(10_000)
       return samples
   
   # Initialize VQE solver
   vqe = VQE(
       hamiltonian=ham,
       executor=executor,
       num_params=4,
   )
   
   # Run optimization
   vqe_energy = vqe.run(max_iterations=50, verbose=True)
   
   # Compare with exact solution
   exact_energy = brute_force_minimize(ham)
   print(f"VQE Energy: {vqe_energy:.6f}")
   print(f"Exact Energy: {exact_energy:.6f}")

VQE with Qubit Unitary Executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using unitary matrices directly for VQE:

.. code-block:: python

   import numpy as np
   from qlass.vqe import VQE
   from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize
   from scipy.linalg import expm
   
   # Define unitary executor using parameterized generators
   def unitary_executor(params):
       num_qubits = 2
       dim = 2**num_qubits
       
       # Define Pauli matrices
       I = np.eye(2)
       X = np.array([[0, 1], [1, 0]])
       Y = np.array([[0, -1j], [1j, 0]])
       Z = np.array([[1, 0], [0, -1]])
       
       # Create 2-qubit generators
       generators = [
           np.kron(Y, I),
           np.kron(I, Y),
           np.kron(X, X),
           np.kron(Z, Z),
       ]
       
       # Build Hamiltonian for time evolution
       H = sum(params[i] * generators[i] for i in range(len(params)))
       
       # Compute unitary: U = exp(-iH)
       return expm(-1j * H)
   
   hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)
   
   vqe = VQE(
       hamiltonian=hamiltonian,
       executor=unitary_executor,
       num_params=4,
       executor_type="qubit_unitary"
   )
   
   vqe_energy = vqe.run(max_iterations=50, verbose=True)

VQE with Photonic Unitary Executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using photonic unitaries with dual-rail encoding and post-selection:

.. code-block:: python

   import numpy as np
   from qlass.vqe import VQE
   from qlass.quantum_chemistry import LiH_hamiltonian_tapered, brute_force_minimize
   from qlass.utils import linear_circuit_to_unitary
   from perceval.converters import QiskitConverter
   from qiskit.circuit.library import n_local
   
   qiskit_converter = QiskitConverter(backend_name="Naive", noise_model=None)
   
   def executor(params):
       num_qubits = 4
       # Create variational ansatz circuit
       ansatz = n_local(num_qubits, 'ry', 'cx', reps=1, entanglement='linear')
       ansatz_assigned = ansatz.assign_parameters(params)
       
       # Convert to Perceval processor
       processor = qiskit_converter.convert(ansatz_assigned, use_postselection=True)
       linear = processor.linear_circuit()
       unitary = linear_circuit_to_unitary(linear)
       
       return unitary
   
   hamiltonian = LiH_hamiltonian_tapered(R=0.1)
   
   vqe = VQE(
       hamiltonian=hamiltonian,
       executor=executor,
       num_params=8,
       executor_type="photonic_unitary",
       ancillary_modes=list(range(8, 14)),  # 6 ancillary modes for 3 CNOTs
   )
   
   vqe_energy = vqe.run(max_iterations=100, verbose=True)

Ensemble-VQE (e-VQE)
~~~~~~~~~~~~~~~~~~~~

Computing multiple excited states simultaneously using ensemble-VQE:

.. code-block:: python

   import numpy as np
   from qlass.vqe import VQE
   from qlass.quantum_chemistry import Hchain_KS_hamiltonian, hamiltonian_matrix
   from qlass.vqe.ansatz import hf_ansatz
   from perceval.algorithm import Sampler
   import matplotlib.pyplot as plt
   
   # Generate Kohn-Sham Hamiltonian for H4 chain
   ham, scf_mo_energy, n_orbs = Hchain_KS_hamiltonian(4, 1.2)
   
   def executor(params, pauli_string):
       processors = hf_ansatz(1, n_orbs, params, pauli_string, method="DFT", cost="e-VQE")
       samplers = [Sampler(p) for p in processors]
       samples = [sampler.samples(10_000) for sampler in samplers]
       return samples
   
   vqe = VQE(
       hamiltonian=ham,
       executor=executor,
       num_params=4,
   )
   
   # Run e-VQE with weighted ensemble
   vqe_energy = vqe.run(
       max_iterations=50,
       verbose=True,
       weight_option="weighted",  # Options: "weighted", "equi", "ground_state_only"
       cost="e-VQE"
   )

VQE with Kohn-Sham Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using DFT-based Kohn-Sham Hamiltonians:

.. code-block:: python

   from qlass.vqe import VQE
   from qlass.quantum_chemistry import Hchain_KS_hamiltonian, brute_force_minimize
   from qlass.vqe.ansatz import hf_ansatz
   from perceval.algorithm import Sampler
   
   # Generate Kohn-Sham Hamiltonian for H4 chain (4 atoms, 1.2 Å spacing)
   ham, scf_mo_energy, n_orbs = Hchain_KS_hamiltonian(4, 1.2)
   
   def executor(params, pauli_string):
       processors = hf_ansatz(1, n_orbs, params, pauli_string, method="DFT", cost="VQE")
       samplers = Sampler(processors)
       samples = samplers.samples(10_000)
       return samples
   
   vqe = VQE(
       hamiltonian=ham,
       executor=executor,
       num_params=4,
   )
   
   vqe_energy = vqe.run(max_iterations=50, verbose=True)

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