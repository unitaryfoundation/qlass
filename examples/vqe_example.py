"""
Example demonstrating the use of the VQE class.
"""

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from perceval.algorithm import Sampler
import perceval as pcvl
from qiskit import transpile
from qiskit.circuit.library import TwoLocal

from qlass.quantum_chemistry import LiH_hamiltonian_tapered, brute_force_minimize
from qlass.vqe import VQE
from qlass.compiler import ResourceAwareCompiler, HardwareConfig
from qlass.utils import rotate_qubits

# Define an executor function that uses the TwoLocal ansatz
def executor(params, pauli_string):
    """
    Executor function that creates a TwoLocal ansatz and compiles it
    using the ResourceAwareCompiler for resource analysis.
    """
    # Create the TwoLocal ansatz circuit
    num_qubits = len(pauli_string)
    ansatz = TwoLocal(num_qubits, 'ry', 'cx', reps=1)
    
    # Assign parameters to the ansatz
    ansatz_assigned = ansatz.assign_parameters(params)
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=['u3', 'cx'], optimization_level=3)
    
    # Apply rotation for Pauli measurement
    ansatz_rot = rotate_qubits(pauli_string, ansatz_transpiled.copy())
    
    # Define hardware configuration for the example photonic chip
    chip_config = HardwareConfig(
        photon_loss_component_db=0.05,
        fusion_success_prob=0.11,
        hom_visibility=0.95,
        source_efficiency=0.9,
        detector_efficiency=0.95
    )
    
    # Compile with ResourceAwareCompiler
    compiler = ResourceAwareCompiler(config=chip_config)
    processor = compiler.compile(ansatz_rot)
    
    # Access the analysis report (optional - could log this if needed)
    # report = processor.analysis_report
    # print(f"Success probability: {report['probability_estimation']['overall_success_prob']:.4%}")
    
    # Set the input state
    processor.with_input(pcvl.LogicalState([0]*num_qubits))
    
    # Run the sampler
    sampler = Sampler(processor)
    samples = sampler.samples(10_000)
    return samples

def main():
    # Number of qubits
    num_qubits = 4
    
    # Generate a 2-qubit Hamiltonian for LiH
    hamiltonian = LiH_hamiltonian_tapered(R=1.0)
    
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
        num_params=2*num_qubits, # Number of parameters in the linear entangled ansatz
    )
    
    # Run the VQE optimization
    print("\nRunning VQE optimization...")
    vqe_energy = vqe.run(
        max_iterations=100,
        verbose=True
    )
    
    # Get the optimal parameters
    optimal_params = vqe.get_optimal_parameters()
    
    # Compare with exact solution
    comparison = vqe.compare_with_exact(exact_energy)
    
    # Print the results
    print(f"\nOptimization complete!")
    print(f"Final energy: {vqe_energy:.6f}")
    print(f"Optimal parameters: {optimal_params}")
    print(f"Number of iterations: {vqe.optimization_result.nfev}")
    print(f"Exact ground state energy: {exact_energy:.6f}")
    print(f"Energy difference: {comparison['absolute_error']:.6f}")
    
    # Plot the convergence
    print("\nPlotting convergence history...")
    plt.figure(figsize=(10, 6))
    iterations = range(len(vqe.energy_history))
    plt.plot(iterations, vqe.energy_history, 'o-', label='VQE Energy')
    plt.axhline(y=exact_energy, color='r', linestyle='--', label='Exact Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('VQE Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('vqe_convergence.png')
    print("Convergence plot saved as 'vqe_convergence.png'")

if __name__ == "__main__":
    main()