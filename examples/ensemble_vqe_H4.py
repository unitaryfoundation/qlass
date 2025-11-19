import warnings

import matplotlib.pyplot as plt
import numpy as np
from perceval.algorithm import Sampler

from qlass.quantum_chemistry import (
    Hchain_KS_hamiltonian,
    hamiltonian_matrix,
)
from qlass.vqe import VQE
from qlass.vqe.ansatz import hf_ansatz

warnings.filterwarnings('ignore')

ham, scf_mo_energy, n_orbs = Hchain_KS_hamiltonian(4, 1.2)


def executor(params, pauli_string):

    processors = hf_ansatz(1, n_orbs, params, pauli_string, method="DFT", cost="e-VQE")
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
    weight_option="weighted",
    cost="e-VQE"
)

# Calculate the exact energies for comparison
H_matrix = hamiltonian_matrix(ham)
exact_energy = np.sort(np.linalg.eigvals(H_matrix))
plt.figure(figsize=(10, 6))

plt.plot(vqe.energy_collector.loss_data, label='Cost')
plt.plot(vqe.energy_collector.energy_data[0], ls=":", color="blue", label='E_A')
plt.plot(vqe.energy_collector.energy_data[1], ls=":", color="green", label='E_B')
plt.axhline(y=exact_energy[0], color='b', linestyle='--', label='E_0')
plt.axhline(y=exact_energy[1], color='g', linestyle='--', label='E_1')
plt.xlabel('Iteration')
plt.ylabel('Energy (Hartree)')
plt.title('ensemble VQE Convergence')
plt.legend()
plt.tight_layout()
plt.savefig('vqe_convergence_DFT.png')
plt.show()
