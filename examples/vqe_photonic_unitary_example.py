"""
Example demonstrating VQE with photonic unitary executor and post-selection.

This example shows how to use a photonic unitary (acting on 2m optical modes)
to simulate m qubits via dual-rail encoding and post-selection.
"""

import warnings

import numpy as np

from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize
from qlass.vqe import VQE

warnings.simplefilter("ignore")


def photonic_unitary_executor(params):
    """
    Example photonic unitary executor: Creates a 2m × 2m unitary for m qubits.

    For a 2-qubit system, this creates a 4×4 photonic unitary using
    parameterized beam splitters and phase shifters.

    In practice, this would come from compiling a quantum circuit to
    photonic hardware or from a photonic circuit simulator.
    """
    num_qubits = 2
    num_modes = 2 * num_qubits  # 4 modes for 2 qubits

    # Create a simple parameterized photonic unitary
    # This is a simplified model - real photonic unitaries would be more complex

    # Start with identity
    U = np.eye(num_modes, dtype=complex)

    # Apply parameterized rotations (representing beam splitters + phase shifters)
    for i in range(num_modes - 1):
        theta = params[i % len(params)]

        # Create a 2×2 beam splitter-like matrix
        c = np.cos(theta)
        s = np.sin(theta)
        bs = np.array([[c, -s], [s, c]], dtype=complex)

        # Embed it in the full space
        U_local = np.eye(num_modes, dtype=complex)
        U_local[i : i + 2, i : i + 2] = bs

        U = U_local @ U

    # Add some phase shifts
    phases = np.exp(1j * params[:num_modes])
    U = np.diag(phases) @ U

    return U


def main():
    # Generate Hamiltonian for 2-qubit LiH
    num_qubits = 2
    hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)
    exact_energy = brute_force_minimize(hamiltonian)

    print("=== VQE with Photonic Unitary Executor ===")
    print(f"System: LiH ({num_qubits} qubits)")
    print(f"Exact ground state energy: {exact_energy:.6f}\n")

    # Initialize VQE with photonic unitary executor
    # For 2 qubits, we need a 4×4 photonic unitary (2m modes)
    num_params = 6  # Number of parameters for the photonic circuit

    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=photonic_unitary_executor,
        num_params=num_params,
        executor_type="photonic_unitary",
    )

    # Run optimization
    print("Running VQE optimization with photonic post-selection...")
    vqe_energy = vqe.run(max_iterations=50, verbose=True)

    # Compare results
    comparison = vqe.compare_with_exact(exact_energy)
    print("\n=== Results ===")
    print(f"VQE Energy: {vqe_energy:.6f}")
    print(f"Exact Energy: {exact_energy:.6f}")
    print(f"Absolute Error: {comparison['absolute_error']:.6f}")
    print(f"Relative Error: {comparison['relative_error']:.2%}")

    # Uncomment the following block to show a plot of the result
    # print("\nPlotting convergence...")
    # vqe.plot_convergence(exact_energy=exact_energy)

    # Additional information about post-selection
    print("\n=== Photonic Implementation Notes ===")
    print(f"Number of optical modes: {2 * num_qubits}")
    print("Encoding: Dual-rail (|0⟩ → mode 2k, |1⟩ → mode 2k+1)")
    print("Post-selection: Only states with exactly one photon per qubit pair")
    print("Note: The effective unitary U' is non-unitary due to post-selection")


if __name__ == "__main__":
    main()
