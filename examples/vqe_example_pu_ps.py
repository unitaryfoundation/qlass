"""
Example demonstrating the use of the VQE class.
"""

import warnings

import matplotlib.pyplot as plt
from perceval.converters import QiskitConverter
from qiskit.circuit.library import n_local

from qlass.quantum_chemistry import LiH_hamiltonian_tapered, brute_force_minimize
from qlass.utils import linear_circuit_to_unitary
from qlass.vqe import VQE

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Initialize the Qiskit converter
qiskit_converter = QiskitConverter(backend_name="Naive", noise_model=None)


def executor(params):
    """
    Executor function that creates a processor with the le_ansatz.
    """
    num_qubits = 4
    # 1. Create the variational ansatz circuit
    ansatz = n_local(num_qubits, "ry", "cx", reps=1, entanglement="linear")
    ansatz_assigned = ansatz.assign_parameters(params)

    # Convert the circuit to a Perceval processor
    processor = qiskit_converter.convert(ansatz_assigned, use_postselection=True)
    linear = processor.linear_circuit()
    unitary = linear_circuit_to_unitary(linear)

    return unitary


def main():
    # Number of qubits
    num_qubits = 4

    # Generate a 4-qubit Hamiltonian for LiH
    hamiltonian = LiH_hamiltonian_tapered(R=0.1)

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
        executor_type="photonic_unitary",
        ancillary_modes=list(
            range(8, 8 + 6)
        ),  # modes 0,1,..., 7 are for the qubits. need 6 more modes for 3 CNOTS.
    )

    # Run the VQE optimization
    print("\nRunning VQE optimization...")
    vqe_energy = vqe.run(max_iterations=100, verbose=True)

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

    # Plot the convergence
    print("\nPlotting convergence history...")
    plt.figure(figsize=(10, 6))
    iterations = range(len(vqe.energy_history))
    plt.plot(iterations, vqe.energy_history, "o-", label="VQE Energy")
    plt.axhline(y=exact_energy, color="r", linestyle="--", label="Exact Energy")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("VQE Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("vqe_convergence.png")
    print("Convergence plot saved as 'vqe_convergence.png'")


if __name__ == "__main__":
    main()
