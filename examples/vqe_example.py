"""
Example demonstrating the use of the qlass.compile function in a VQE algorithm.
"""

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import TwoLocal
from perceval.algorithm import Sampler
from scipy.optimize import minimize

from qlass import compile
from qlass.helper_functions import (
    get_probabilities, 
    qubit_state_marginal, 
    compute_energy, 
    pauli_string_bin,
)
from qlass.hamiltonians import generate_random_hamiltonian

def executor(params, pauli_string, num_qubits=2):
    """
    Execute a quantum circuit with given parameters and measure in the basis specified by the Pauli string.
    
    Args:
        params (np.ndarray): Parameters for the variational circuit
        pauli_string (str): String representation of Pauli operators (e.g., "IXYZ")
        num_qubits (int): Number of qubits in the circuit
    
    Returns:
        dict: Sampling results
    """
    # Create a parameterized quantum circuit (ansatz)
    ansatz = TwoLocal(num_qubits, 'ry', 'cx', reps=1)
    
    # Assign parameters
    bound_circuit = ansatz.assign_parameters(params)
    
    # Apply measurement basis rotations based on the Pauli string
    rotated_circuit = QuantumCircuit(num_qubits)
    rotated_circuit.compose(bound_circuit, inplace=True)
    
    for i, pauli in enumerate(pauli_string):
        if pauli == 'X':
            rotated_circuit.h(i)
        elif pauli == 'Y':
            rotated_circuit.rx(np.pi/2, i)
    
    transpiled_circuit = transpile(rotated_circuit, basis_gates=['u3', 'cx'], optimization_level=3)
    # Convert to Perceval processor using our compile function
    processor = compile(transpiled_circuit)
    
    # Sample from the processor
    sampler = Sampler(processor)
    samples = sampler.samples(10_000)
    
    return samples

def energy_expectation(params, hamiltonian, executor, num_qubits=2):
    """
    Calculate the energy expectation value for a given Hamiltonian and circuit parameters.
    
    Args:
        params (np.ndarray): Parameters for the variational circuit
        hamiltonian (dict): Dictionary mapping Pauli strings to coefficients
        num_qubits (int): Number of qubits in the circuit
    
    Returns:
        float: Energy expectation value
    """
    energy = 0.0
    
    for pauli_string, coefficient in hamiltonian.items():
        # Execute the circuit and get samples
        samples = executor(params, pauli_string, num_qubits)
        
        # Process the results
        prob_dist = get_probabilities(samples['results'])
        pauli_bin = pauli_string_bin(pauli_string)
        
        # Calculate the qubit state marginal probabilities
        qubit_state_marg = qubit_state_marginal(prob_dist)
        
        # Compute the expectation value for this Pauli term
        expectation = compute_energy(pauli_bin, qubit_state_marg)
        
        # Add contribution to total energy
        energy += coefficient * expectation
    
    return energy

def main():
    # Number of qubits
    num_qubits = 2
    
    # Generate a random 2-qubit Hamiltonian
    hamiltonian = generate_random_hamiltonian(num_qubits)
    
    # Print the Hamiltonian
    print("Random Hamiltonian:")
    for pauli_string, coefficient in hamiltonian.items():
        print(f"  {pauli_string}: {coefficient:.4f}")
    
    # Initial random parameters for the variational circuit
    initial_params = np.random.rand(2 * num_qubits)
    
    # Run the VQE optimization
    print("\nRunning VQE optimization...")
    result = minimize(
        energy_expectation,
        initial_params,
        args=(hamiltonian, executor, num_qubits),
        method='COBYLA',
        options={'maxiter': 10}
    )
    
    # Print the results
    print(f"\nOptimization complete!")
    print(f"Final energy: {result.fun:.6f}")
    print(f"Optimal parameters: {result.x}")
    print(f"Number of iterations: {result.nfev}")
    
    # Calculate exact ground state energy for comparison
    from qlass.classical_solution import hamiltonian_matrix, brute_force_minimize
    exact_energy = brute_force_minimize(hamiltonian)
    print(f"Exact ground state energy: {exact_energy:.6f}")
    print(f"Energy difference: {abs(result.fun - exact_energy):.6f}")

if __name__ == "__main__":
    main()