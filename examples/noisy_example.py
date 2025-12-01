"""
This script performs a noisy VQE simulation for the tapered LiH Hamiltonian
at various internuclear distances using the qlass library. It compares the
simulated energy curve with the theoretical one.
"""

import warnings

import numpy as np
from perceval import LogicalState
from perceval.algorithm import Sampler
from qiskit import transpile
from qiskit.circuit.library import TwoLocal
from tqdm import tqdm

from qlass.compiler import HardwareConfig, ResourceAwareCompiler

# Imports from the qlass library
from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize
from qlass.utils import rotate_qubits
from qlass.vqe import VQE

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


# Define an executor function that uses the TwoLocal ansatz
def executor(params, pauli_string):
    """
    Executor function that creates a TwoLocal ansatz and compiles it
    using the ResourceAwareCompiler for resource analysis.
    """
    # Create the TwoLocal ansatz circuit
    num_qubits = len(pauli_string)
    ansatz = TwoLocal(num_qubits, "ry", "cx", reps=1)

    # Assign parameters to the ansatz
    ansatz_assigned = ansatz.assign_parameters(params)
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=["u3", "cx"], optimization_level=3)

    # Apply rotation for Pauli measurement
    ansatz_rot = rotate_qubits(pauli_string, ansatz_transpiled.copy())

    # Define hardware configuration for the example photonic chip
    chip_config = HardwareConfig(
        brightness=0.09,
        indistinguishability=0.9,
        g2=0.01,
        g2_distinguishable=True,
        transmittance=0.4,
        phase_imprecision=0.02,
        phase_error=0.02,
    )

    # Compile with ResourceAwareCompiler
    compiler = ResourceAwareCompiler(config=chip_config)
    processor = compiler.compile(ansatz_rot)

    # Set the input state
    processor.with_input(LogicalState([0] * num_qubits))

    # Run the sampler
    sampler = Sampler(processor)
    samples = sampler.samples(10_000)
    return samples


def main():
    """
    Main function to run the VQE simulation and plot the results.
    """

    # 1. Define the range of bond radii to simulate
    radii = np.linspace(0.5, 2.5, 10)
    exact_energies = []
    vqe_energies = []

    # The LiH Hamiltonian (with num_orbitals=1) has 2 qubits
    num_qubits = 2
    # The le_ansatz uses a TwoLocal circuit with reps=1, so num_params = (1+1)*num_qubits
    num_params = 2 * num_qubits

    # 2. Loop through each radius, run VQE, and store the results
    print("Running VQE simulations for different bond radii...")
    for r in tqdm(radii, desc="Simulating Radii"):
        # Generate the tapered Hamiltonian for the current radius
        hamiltonian = LiH_hamiltonian(R=r, num_electrons=2, num_orbitals=1)

        # Calculate the exact ground state energy for the theoretical curve
        exact_energy = brute_force_minimize(hamiltonian)
        exact_energies.append(exact_energy)

        # Initialize the VQE solver
        vqe = VQE(hamiltonian=hamiltonian, executor=executor, num_params=num_params)

        # Run the VQE optimization
        vqe_energy = vqe.run(
            max_iterations=25,  # More iterations for better convergence
            verbose=False,  # Turn off verbose output for the loop
        )
        vqe_energies.append(vqe_energy)

    # Uncomment the following block to show a plot of the result
    # 3. Plot the results
    # import matplotlib.pyplot as plt
    # plt.style.use("seaborn-v0_8-whitegrid")
    # plt.figure(figsize=(12, 7))

    # # Plot theoretical (exact) energy curve
    # plt.plot(radii, exact_energies, "bo", label="Exact Theoretical Energy", linewidth=2)

    # # Plot noisy VQE simulation results
    # plt.plot(radii, vqe_energies, "ro", label="Noisy VQE Simulation", markersize=8)

    # plt.xlabel("Internuclear Distance (Å)", fontsize=14)
    # plt.ylabel("Energy (Hartree)", fontsize=14)
    # plt.title("Noisy VQE Simulation of LiH Molecule", fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.savefig("Lih_tapered_4q.png")
    # print("\nSimulation complete. Displaying plot.")
    # plt.show()


if __name__ == "__main__":
    main()
