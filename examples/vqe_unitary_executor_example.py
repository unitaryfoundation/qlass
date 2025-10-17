
import numpy as np
import warnings
warnings.simplefilter('ignore')

from qlass.vqe import VQE
from qlass.quantum_chemistry import LiH_hamiltonian, brute_force_minimize
from scipy.linalg import expm



def unitary_executor(params):
    """
    Example unitary executor: Creates a unitary using parameterized generators.
    
    For a 2-qubit system, this creates U = exp(-i * sum(params[i] * generator[i]))
    """
    num_qubits = 2
    dim = 2**num_qubits
    
    # Define generators (Pauli matrices)
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    # Create 2-qubit generators
    generators = [
        np.kron(Y, I),  # Y ⊗ I
        np.kron(I, Y),  # I ⊗ Y  
        np.kron(X, X),  # X ⊗ X
        np.kron(Z, Z),  # Z ⊗ Z
    ]
    
    # Build Hamiltonian for time evolution
    H = sum(params[i] * generators[i] for i in range(len(params)))
    
    # Compute unitary: U = exp(-iH)
    unitary = expm(-1j * H)
    
    return unitary

def main():
    # Generate Hamiltonian
    hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)
    exact_energy = brute_force_minimize(hamiltonian)
    
    print("LiH Hamiltonian (2 qubits)")
    print(f"Exact ground state energy: {exact_energy:.6f}\n")
    
    # Initialize VQE with unitary executor
    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=unitary_executor,
        num_params=4,
        executor_type="qubit_unitary"  # Explicitly specify type
    )
    
    # Run optimization
    vqe_energy = vqe.run(max_iterations=50, verbose=True)
    
    # Compare results
    comparison = vqe.compare_with_exact(exact_energy)
    print(f"\n--- Results ---")
    print(f"VQE Energy: {vqe_energy:.6f}")
    print(f"Exact Energy: {exact_energy:.6f}")
    print(f"Absolute Error: {comparison['absolute_error']:.6f}")
    print(f"Relative Error: {comparison['relative_error']:.2%}")
    
    # Plot convergence
    vqe.plot_convergence(exact_energy=exact_energy)

if __name__ == "__main__":
    main()