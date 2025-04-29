from typing import Dict, Callable
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from qlass.utils import loss_function

class VQE:
    """
    Variational Quantum Eigensolver for photonic quantum computing.
    
    This class provides a high-level interface for running VQE experiments
    on photonic simulators, finding ground state energies of quantum systems.
    """
    
    def __init__(
        self, 
        hamiltonian: Dict[str, float],
        executor: Callable,
        num_params: int,
        optimizer: str = "COBYLA",
    ):
        """
        Initialize the VQE solver.
        
        Args:
            hamiltonian (Dict[str, float]): Hamiltonian dictionary with Pauli string keys 
                                           and coefficient values
            optimizer (str): Optimization method to use. Any method supported by scipy.optimize.minimize
            executor (Callable): Custom executor function, if None, a default one will be created
        """
        self.hamiltonian = hamiltonian
        self.executor = executor
        self.num_params = num_params
        self.optimizer = optimizer
        
        # Extract number of qubits from the Hamiltonian
        self.num_qubits = len(next(iter(hamiltonian.keys())))
        
        # Results storage
        self.optimization_result = None
        self.energy_history = []
        self.parameter_history = []
    
    def _callback(self, params):
        """Callback function to record optimization progress."""
        energy = loss_function(params, self.hamiltonian, self.executor)
        self.energy_history.append(energy)
        self.parameter_history.append(params.copy())
        
    def run(self, initial_params=None, max_iterations=100, verbose=True):
        """
        Run the VQE optimization to find the ground state energy.
        
        Args:
            initial_params (np.ndarray): Initial parameters for the variational circuit.
                                        If None, random parameters will be used.
            max_iterations (int): Maximum number of iterations for the optimization
            verbose (bool): Whether to print progress updates
            
        Returns:
            float: The minimum energy found
        """
        # Initialize parameters if not provided
        if initial_params is None:
            initial_params = np.random.rand(self.num_params)
        
        # Reset history
        self.energy_history = []
        self.parameter_history = []
        
        if verbose:
            print(f"Starting VQE optimization using {self.optimizer} optimizer")
            print(f"Number of qubits: {self.num_qubits}")
            print(f"Number of parameters: {len(initial_params)}")
            
        # Run the optimization
        self.optimization_result = minimize(
            loss_function,
            initial_params,
            args=(self.hamiltonian, self.executor),
            method=self.optimizer,
            callback=self._callback,
            options={'maxiter': max_iterations}
        )
        
        if verbose:
            print(f"Optimization complete!")
            print(f"Final energy: {self.optimization_result.fun:.6f}")
            print(f"Number of iterations: {self.optimization_result.nfev}")
            
        return self.optimization_result.fun
    
    def get_optimal_parameters(self):
        """Get the optimal parameters found during optimization."""
        if self.optimization_result is None:
            raise ValueError("VQE optimization has not been run yet.")
        return self.optimization_result.x
    
    def plot_convergence(self, exact_energy=None):
        """
        Plot the energy convergence during the optimization.
        
        Args:
            exact_energy (float): Exact ground state energy for comparison, if available
        """
        if not self.energy_history:
            raise ValueError("No optimization history available. Run VQE first.")
            
        plt.figure(figsize=(10, 6))
        iterations = range(len(self.energy_history))
        plt.plot(iterations, self.energy_history, 'o-', label='VQE Energy')
        
        if exact_energy is not None:
            plt.axhline(y=exact_energy, color='r', linestyle='--', label='Exact Energy')
            
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('VQE Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def compare_with_exact(self, exact_energy=None):
        """
        Compare the VQE result with the exact ground state energy.
        
        Args:
            exact_energy (float): Exact ground state energy
            
        Returns:
            dict: Comparison metrics
        """
        if self.optimization_result is None:
            raise ValueError("VQE optimization has not been run yet.")
            
        if exact_energy is None:
            from qlass.quantum_chemistry.classical_solution import brute_force_minimize
            exact_energy = brute_force_minimize(self.hamiltonian)
            
        vqe_energy = self.optimization_result.fun
        absolute_error = abs(vqe_energy - exact_energy)
        relative_error = absolute_error / abs(exact_energy) if exact_energy != 0 else float('inf')
        
        comparison = {
            'vqe_energy': vqe_energy,
            'exact_energy': exact_energy,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }
        
        return comparison