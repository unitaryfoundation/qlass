import warnings

from perceval.algorithm import Sampler

from qlass.quantum_chemistry import Hchain_KS_hamiltonian, brute_force_minimize
from qlass.vqe import VQE
from qlass.vqe.ansatz import hf_ansatz

warnings.filterwarnings("ignore")

ham, scf_mo_energy, n_orbs = Hchain_KS_hamiltonian(4, 1.2)


# Define an executor function that uses the linear entangled ansatz
def executor(params, pauli_string):
    processors = hf_ansatz(1, n_orbs, params, pauli_string, method="DFT", cost="VQE")
    samplers = Sampler(processors)
    samples = samplers.samples(10_000)

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
)

# Calculate the exact energies for comparison
exact_energy = brute_force_minimize(ham)
print(f"energy from exact diag: {exact_energy}")
