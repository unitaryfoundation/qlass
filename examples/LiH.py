import numpy as np
from qlass.vqe import VQE
from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize, Hchain_geometry
from perceval.algorithm import Sampler
import warnings
import matplotlib.pyplot as plt
from qlass.vqe.ansatz import hf_ansatz

warnings.filterwarnings('ignore')

ham, n_orbs = LiH_hamiltonian(num_electrons=2, num_orbitals=1)


# Define an executor function that uses the linear entangled ansatz
def executor(params, pauli_string):

    processors = hf_ansatz(1, n_orbs, params, pauli_string, method="WFT")
    samplers = [Sampler(p) for p in processors]
    samples = [sampler.samples(10_000) for sampler in samplers]

    return samples


# Initialize the VQE solver
vqe = VQE(
    hamiltonian=ham,
    executor=executor,
    num_params=4,  # Number of parameters in the linear entangled ansatz
)

# Run the VQE optimization
vqe_energy = vqe.run(
    max_iterations=50,
    verbose=True,
    weight_option="weighted"
)

# Calculate the exact ground state energy for comparison
exact_energy = brute_force_minimize(ham)

plt.figure(figsize=(10, 6))
iterations = range(len(vqe.loss_history))
plt.plot(vqe.energy_collector.loss_data, label='cost')
plt.plot(vqe.energy_collector.energy_data[0], ls=":", color="blue", label='E_A')
plt.plot(vqe.energy_collector.energy_data[1], ls=":", color="green", label='E_B')
plt.axhline(y=exact_energy[0], color='b', linestyle='--', label='E_0')
plt.axhline(y=exact_energy[1], color='g', linestyle='--', label='E_1')
plt.xlabel('Iteration')
plt.ylabel('Energy (Hartee)')
plt.title('VQE Convergence')
plt.legend()
plt.tight_layout()
plt.savefig('vqe_convergence_WFT_LiH.png')
plt.show()

