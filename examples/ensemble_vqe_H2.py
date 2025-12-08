import numpy as np
from qlass.quantum_chemistry.hamiltonians import Hchain_hamiltonian_WFT
from qlass.vqe import VQE
from qlass.quantum_chemistry import hamiltonian_matrix
from perceval.algorithm import Sampler
import warnings
import matplotlib.pyplot as plt
from qlass.vqe.ansatz import Bitstring_initial_states


warnings.filterwarnings("ignore")

ham = Hchain_hamiltonian_WFT(2, 0.741, tampering=True)
print("Printing Hamiltonian", ham)


def executor(params, pauli_string):
    # for e-VQE
    processor = Bitstring_initial_states(1, 2, params, pauli_string, cost="e-VQE")
    samplers = [Sampler(p) for p in processor]
    samples = [sampler.samples(10_000) for sampler in samplers]

    return samples


# Initialize the VQE solver
vqe = VQE(
    hamiltonian=ham,
    executor=executor,
    num_params=4,  # Number of parameters in the linear entangled ansatz
)

# Run the VQE optimization
vqe_energy = vqe.run(max_iterations=40, verbose=True, weight_option="weighted", cost="e-VQE")

# Calculate the exact energies for comparison
H_matrix = hamiltonian_matrix(ham)
exact_energy = np.sort(np.linalg.eigvals(H_matrix))
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
print(f"Exact ground state: {exact_energy[0]}, first excited state: {exact_energy[1]}")
VQE_T = np.add(vqe.energy_collector.energy_data[0], vqe.energy_collector.energy_data[1])
EX_T = exact_energy[0] + exact_energy[1]
error_T = VQE_T - EX_T
error_EA = vqe.energy_collector.energy_data[0] - exact_energy[0]
error_EB = vqe.energy_collector.energy_data[1] - exact_energy[1]
ax[0].plot(vqe.energy_collector.loss_data, color="red", label="cost")
ax[0].plot(vqe.energy_collector.energy_data[0], ls=":", color="blue", label="$E_A$")
ax[0].plot(vqe.energy_collector.energy_data[1], ls=":", color="green", label="$E_B$")
ax[0].axhline(y=exact_energy[0], color="b", linestyle="--", label="$E_0$")
ax[0].axhline(y=exact_energy[1], color="g", linestyle="--", label="$E_1$")
ax[1].plot(error_T, color="black", ls="-", label="$E_T$")
ax[1].plot(error_EA, color="b", ls=":", label="E_A")
ax[1].plot(error_EB, color="g", ls=":", label="E_B")
ax[1].set_yscale("log")
ax[0].set_ylabel("Energy (hartree)", fontsize=16)
ax[1].set_ylabel("Error (hartree", fontsize=16)
ax[1].set_xlabel("Iteration", fontsize=16)
# plt.ylabel('Energy (Hartee)')
ax[0].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
ax[1].tick_params(axis="both", labelsize=16)
ax[0].tick_params(axis="y", labelsize=16)
ax[1].tick_params(axis="y", labelsize=16)
ax[0].set_title("ensemble VQE Convergence", fontsize=16)
ax[0].legend(fontsize=16, ncol=4, labelspacing=0.02, columnspacing=0.1)
ax[1].legend(fontsize=16)
plt.show()
