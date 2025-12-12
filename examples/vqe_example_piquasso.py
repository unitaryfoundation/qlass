"""
Example demonstrating the use of the VQE class using Piquasso for simulation.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import piquasso as pq
from piquasso.dual_rail_encoding import (
    dual_rail_encode_from_qiskit,
    get_bosonic_qubit_samples,
)
from qiskit import transpile
from qiskit.circuit.library import n_local

from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize
from qlass.utils import rotate_qubits
from qlass.vqe import VQE

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")



def le_ansatz_piquasso(lp: np.ndarray, pauli_string: str) -> pq.Program:
    """
    Creates Piquasso quantum program for the Linear Entangled Ansatz.
    This ansatz consists of a layer of parametrized rotations, followed by
    a layer of CNOT gates, and finally another layer of parametrized rotations.

    Args:
        lp (np.ndarray): Array of parameter values
        pauli_string (str): Pauli string

    Returns:
        Program: The quantum circuit as a Piquasso program
    """
    num_qubits = len(pauli_string)
    ansatz = n_local(num_qubits, "ry", "cx", reps=1, entanglement="linear")

    ansatz_assigned = ansatz.assign_parameters(lp)
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=["u3", "cx"], optimization_level=3)

    ansatz_rot = rotate_qubits(pauli_string, ansatz_transpiled.copy())
    program = dual_rail_encode_from_qiskit(ansatz_rot)

    return program


def _sample_from_prob_dist(prob_dist: dict, shots: int) -> list[list[int]]:
    """
    Helper function to sample outcomes based on their probabilities.

    Args:
        prob_dist (dict): A dictionary mapping outcomes to their probabilities.
        shots (int): Number of samples to draw.

    Returns:
        list[list[int]]: A list of samples.
    """

    states = np.asarray(list(prob_dist.keys()), dtype=int)
    probs = np.fromiter(prob_dist.values(), dtype=float)

    cdf = np.cumsum(probs)
    r = np.random.rand(shots)
    idx = np.searchsorted(cdf, r)

    return states[idx].tolist()


# Define an executor function that uses the linear entangled ansatz
def executor(params, pauli_string):
    """
    Executor function that creates a program with `le_ansatz_piquasso`.
    """
    program = le_ansatz_piquasso(params, pauli_string)
    simulator = pq.SamplingSimulator(config=pq.Config(cutoff=5))
    state = simulator.execute(program).state
    state.normalize()
    fock_probs = state.fock_probabilities_map
    samples = _sample_from_prob_dist(fock_probs, shots=10_000)
    qubit_samples = get_bosonic_qubit_samples(samples)
    return qubit_samples


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
    start_time = time.time()
    vqe_energy = vqe.run(max_iterations=10, verbose=True)
    end_time = time.time()
    print(f"VQE optimization took {end_time - start_time:.2f} seconds.")

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
