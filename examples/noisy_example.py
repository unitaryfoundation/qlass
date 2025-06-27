"""
Example demonstrating the use of the VQE class with a noisy ansatz.
"""

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from perceval.algorithm import Sampler
from perceval.utils import NoiseModel

from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize
from qlass.vqe import VQE, le_ansatz

def executor_with_noise(params, pauli_string, noise_model):
    """
    Executor function that creates a processor with the le_ansatz,
    applies the specified noise model, and runs the simulation.
    """
    processor = le_ansatz(params, pauli_string, noise_model=noise_model)
    sampler = Sampler(processor)
    samples = sampler.samples(10_000)
    return samples

def main():
    # --- 1. Define the Problem ---
    num_qubits = 2
    hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)
    
    print("LiH Hamiltonian:")
    for pauli_string, coefficient in hamiltonian.items():
        print(f"  {pauli_string}: {coefficient:.4f}")
    
    exact_energy = brute_force_minimize(hamiltonian)
    print(f"\nExact ground state energy: {exact_energy:.6f}")
    
    # --- 2. Define the Noise Model ---
    # We define a NoiseModel object. Here, we'll model system-wide photon loss
    # and phase imprecision, which are common sources of error in photonic
    # quantum computers.
    noise_model = NoiseModel(
        transmittance=0.9,      # System-wide transmittance, e.g., 0.9 means 10% photon loss
        phase_imprecision=0.01  # Small random phase shifts (in radians)
    )
    print("\nDefined a noise model with system-wide transmittance and phase imprecision.")

    # --- 3. Set up and Run VQE ---
    
    # We use a lambda function to pass the noise_model to our executor.
    # This is a convenient way to adapt the executor's signature for the VQE class,
    # which expects an executor of the form `executor(params, pauli_string)`.
    noisy_executor = lambda params, pauli_string: executor_with_noise(params, pauli_string, noise_model)

    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=noisy_executor,
        num_params=2*num_qubits,
    )
    
    print("\nRunning VQE optimization with the noise model...")
    vqe_energy = vqe.run(
        max_iterations=20,
        verbose=True
    )
    
    # --- 4. Analyze Results ---
    optimal_params = vqe.get_optimal_parameters()
    comparison = vqe.compare_with_exact(exact_energy)
    
    print(f"\n--- Results with Noise ---")
    print(f"Final noisy energy: {vqe_energy:.6f}")
    print(f"Exact energy:       {exact_energy:.6f}")
    print(f"Energy difference:  {comparison['absolute_error']:.6f}")
    print(f"Optimal parameters found: {np.round(optimal_params, 4)}")
    
    # --- 5. Plot Convergence ---
    print("\nPlotting convergence history...")
    plt.figure(figsize=(10, 6))
    iterations = range(len(vqe.energy_history))
    plt.plot(iterations, vqe.energy_history, 'o-', label='Noisy VQE Energy')
    plt.axhline(y=exact_energy, color='r', linestyle='--', label='Exact Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('VQE Convergence with Photon Loss Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('vqe_convergence_with_noise.png')
    print("Convergence plot saved as 'vqe_convergence_with_noise.png'")
    plt.show()

if __name__ == "__main__":
    main()