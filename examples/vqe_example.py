"""
Example demonstrating the use of the VQE class.
"""

import warnings

from perceval.algorithm import Sampler

from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize
from qlass.vqe import VQE, le_ansatz

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


# Define an executor function that uses the linear entangled ansatz
def executor(params, pauli_string):
    """
    Executor function that creates a processor with the le_ansatz.
    """
    processor = le_ansatz(params, pauli_string)
    sampler = Sampler(processor)
    samples = sampler.samples(10_000)
    return samples


def main():
    # Number of qubits
    num_qubits = 2

    # Generate a 2-qubit Hamiltonian for LiH
    hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)

    # Print the Hamiltonian
    print("LiH Hamiltonian:")
    for pauli_string, coefficient in hamiltonian.items():
        print(f"  {pauli_string}: {coefficient:.4f}")

    # Calculate exact ground state energy for comparison
    exact_energy = brute_force_minimize(hamiltonian)
    print(f"\nExact ground state energy: {exact_energy:.6f}")

    # Initialize the VQE solver with the custom executor
    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=executor,
        num_params=2 * num_qubits,  # Number of parameters in the linear entangled ansatz
    )

    # Run the VQE optimization
    print("\nRunning VQE optimization...")
    vqe_energy = vqe.run(max_iterations=10, verbose=True)

    # Get the optimal parameters
    optimal_params = vqe.get_optimal_parameters()

    # Compare with exact solution
    comparison = vqe.compare_with_exact(exact_energy)

    # Print the results
    print("\nOptimization complete!")
    print(f"Final energy: {vqe_energy:.6f}")
    print(f"Optimal parameters: {optimal_params}")
    print(f"Number of iterations: {vqe.optimization_result.nfev}")
    print(f"Exact ground state energy: {exact_energy:.6f}")
    print(f"Energy difference: {comparison['absolute_error']:.6f}")

    # Uncomment the following block to show a plot of the result
    # print("\nPlotting convergence history...")
    # plt.figure(figsize=(10, 6))
    # iterations = range(len(vqe.energy_history))
    # plt.plot(iterations, vqe.energy_history, "o-", label="VQE Energy")
    # plt.axhline(y=exact_energy, color="r", linestyle="--", label="Exact Energy")
    # plt.xlabel("Iteration")
    # plt.ylabel("Energy")
    # plt.title("VQE Convergence")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.savefig("vqe_convergence.png")
    # print("Convergence plot saved as 'vqe_convergence.png'")


if __name__ == "__main__":
    main()
