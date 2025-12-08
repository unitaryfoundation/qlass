import warnings

import matplotlib.pyplot as plt
import numpy as np
from perceval.algorithm import Sampler

from qlass.quantum_chemistry import hamiltonian_matrix
from qlass.quantum_chemistry.hamiltonians import Hchain_hamiltonian_WFT
from qlass.vqe import VQE
from qlass.vqe.ansatz import CSF_initial_states

warnings.filterwarnings("ignore")

ham = Hchain_hamiltonian_WFT(2, 0.741, tampering=False)
print("Printing Hamiltonian", ham)


def executor(params, pauli_string):
    # for VQE
    processor = CSF_initial_states(1, (1, 1), params, pauli_string)
    samplers = Sampler(processor)
    samples = samplers.samples(10_000)

    return samples


# Initialize the VQE solver
vqe = VQE(
    hamiltonian=ham,
    executor=executor,
    num_params=8,  # Number of parameters in the linear entangled ansatz
)

# Run the VQE optimization
vqe_energy = vqe.run(max_iterations=50, verbose=True, cost="VQE")

H_matrix = hamiltonian_matrix(ham)
exact_energy = np.sort(np.linalg.eigvals(H_matrix))
print(f"energy from exact diag: {exact_energy[0]}")
plt.figure(figsize=(10, 6))
plt.plot(vqe.energy_history, label="Ground state from VQE")
plt.axhline(y=exact_energy[0], color="b", linestyle="--", label="Ground state from exact")
plt.xlabel("Iteration")
plt.ylabel("Energy (Hartee)")
plt.title("VQE Convergence")
plt.legend()
plt.tight_layout()
plt.show()
