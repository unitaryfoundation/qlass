"""
VQE simulations of larger Hamiltonians using the VQE class.
"""

import warnings

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import TwoLocal
from qiskit_aer import AerSimulator

from qlass.quantum_chemistry import LiH_hamiltonian_tapered, brute_force_minimize
from qlass.utils import rotate_qubits
from qlass.vqe import VQE

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def qiskit_executor(params: np.ndarray, pauli_string: str, shots: int = 4096) -> dict[str, int]:
    """
    An executor that uses Qiskit to build, run, and sample a circuit.

    Args:
        params (np.ndarray): Parameters for the variational circuit.
        pauli_string (str): The Pauli term to be measured.
        shots (int): The number of times to run the simulation.

    Returns:
        Dict[str, int]: A dictionary of measurement counts, e.g., {'01': 2048, '10': 2048}.
    """
    num_qubits = len(pauli_string)

    # 1. Create the variational ansatz circuit
    ansatz = TwoLocal(num_qubits, "ry", "cx", reps=1, entanglement="linear")
    ansatz_assigned = ansatz.assign_parameters(params)
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=["u3", "cx"], optimization_level=3)

    # 2. Apply basis-change rotations for measuring the Pauli string
    circuit = rotate_qubits(pauli_string, ansatz_transpiled.copy())

    # 3. Add measurement gates
    circuit.measure_all()

    # 4. Run the simulation and get counts
    simulator = AerSimulator()
    result = simulator.run(circuit, shots=shots).result()
    counts = result.get_counts(0)

    return {"counts": counts}


# --- Main Execution ---


def main():
    # Define a simple 2-qubit Hamiltonian (e.g., for H2 molecule)
    # hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=2)
    hamiltonian = LiH_hamiltonian_tapered(R=0.1)
    num_qubits = 4

    # Calculate exact ground state energy for comparison
    exact_energy = brute_force_minimize(hamiltonian)

    print("Hamiltonian:")
    for pauli_string, coeff in hamiltonian.items():
        print(f"  {pauli_string}: {coeff:.4f}")
    print(f"\nExact ground state energy: {exact_energy:.6f}")

    # Initialize the VQE solver with the Qiskit executor
    # For a TwoLocal with 'ry' and 'cx' and reps=1, params = (reps+1)*num_qubits
    num_params = (1 + 1) * num_qubits
    vqe = VQE(hamiltonian=hamiltonian, executor=qiskit_executor, num_params=num_params)

    # Run the VQE optimization
    vqe_energy = vqe.run(max_iterations=25, verbose=True)

    # Get and print results
    comparison = vqe.compare_with_exact(exact_energy)

    print("\n--- Qiskit VQE Results ---")
    print(f"Final VQE energy: {vqe_energy:.6f}")
    print(f"Energy difference: {comparison['absolute_error']:.6f}")

    # Uncomment the following block to show a plot of the result
    # plt.figure(figsize=(10, 6))
    # plt.plot(
    #     range(len(vqe.energy_history)), vqe.energy_history, "o-", label="VQE Energy (Qiskit Sim)"
    # )
    # plt.axhline(y=exact_energy, color="r", linestyle="--", label="Exact Energy")
    # plt.xlabel("Iteration")
    # plt.ylabel("Energy")
    # plt.title("VQE Convergence using Qiskit Executor")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
