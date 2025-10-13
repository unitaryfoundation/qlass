import sys
from typing import Dict, Callable
import numpy as np
import qlass
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from qlass.utils import loss_function
from qlass.utils.utils import DataCollector
from qlass.utils.utils import _extract_samples_from_executor_result, normalize_samples, qubit_state_marginal, compute_energy, get_probabilities, pauli_string_bin

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
        self.loss_history = []
        self.parameter_history = []
        self.energies_final = None
        self.energies_exact = None
        self.energy_collector = DataCollector()

    
    def _callback(self, params):
        """Callback function to record optimization progress."""
        cost = loss_function(params, self.hamiltonian, self.executor,self.energy_collector)
        self.loss_history.append(cost)
        self.parameter_history.append(params.copy())
        
    def run(self, initial_params=None, max_iterations=100, verbose=True, weight_option: str = "weighted"):
        """
        Run the VQE optimization to find the ground state energy.
        
        Args:
            initial_params (np.ndarray): Initial parameters for the variational circuit.
                                        If None, random parameters will be used.
            max_iterations (int): Maximum number of iterations for the optimization
            verbose (bool): Whether to print progress updates
            weight_option (str): "weighted" (w_i < w_j) , "equi" (w_i = w_j), "ground_state_only" (1., 0., ..., 0.)
            
        Returns:
            float: The minimum cost found
        """
        # Initialize parameters if not provided
        if initial_params is None:
            initial_params = np.random.rand(self.num_params)
        
        if weight_option == "weighted":
            print(f"ensemble vqe is running with the weights: w_i < w_j")
        elif weight_option == "equi":
            print(f"ensemble vqe is running with the weights: w_i = w_j")
        elif weight_option == "ground_state_only":
            print("Ensemble energy calculates ground state with maximum weight and excited states may not be correct")
        else:
            sys.exit(" Available weight options for ensemble VQE is: weighted or equi")
        
        # Reset history
        self.loss_history = []
        self.parameter_history = []
        
        if verbose:
            print(f"Starting VQE optimization using {self.optimizer} optimizer")
            print(f"Number of qubits: {self.num_qubits}")
            print(f"Number of parameters: {len(initial_params)}")


        # Run the optimization
        self.optimization_result = minimize(
            loss_function,
            initial_params,
            args=(self.hamiltonian, self.executor, self.energy_collector, weight_option),
            method=self.optimizer,
            callback=self._callback,
            options={'maxiter': max_iterations}
        )
        from qlass.quantum_chemistry.hamiltonians import group_commuting_pauli_terms
        final_param_values = self.optimization_result.x
        grouped_hamiltonians = group_commuting_pauli_terms(self.hamiltonian)


        for group in grouped_hamiltonians:
            # Each group contains mutually commuting terms
            # In the future, this could be optimized to measure entire groups simultaneously
            # For now, we process each term individually but with the grouping organization
            for pauli_string, coefficient in group.items():
                samples = self.executor(final_param_values, pauli_string)

                # Handle different executor return formats
                sample_lists = [_extract_samples_from_executor_result(s) for s in samples]

                # Normalize samples to consistent format
                normalized_samples = [normalize_samples(sample_list) for sample_list in sample_lists]

                prob_dist = [get_probabilities(normalized_sample) for normalized_sample in normalized_samples]
                pauli_bin = pauli_string_bin(pauli_string)

                qubit_state_marg = [qubit_state_marginal(pd) for pd in prob_dist]
                expectation = [compute_energy(pauli_bin, qsm) for qsm in qubit_state_marg]
                energies = [coefficient * expect for expect in expectation]
                # Initialize accumulator on first iteration
                if self.energies_final is None:
                    self.energies_final = [0.0] * len(energies)

                # Accumulate energies dynamically
                for i, energy in enumerate(energies):
                    self.energies_final[i] += energy
        
        self.energies_exact = qlass.brute_force_minimize(self.hamiltonian)

        if verbose:
            print(f"Optimization complete!")
            print(f"Final energies from exact Diag: {self.energies_exact}")
            print(f"Final energies from ensemble VQE: {self.energies_final} | sum: {np.sum(self.energies_final)}")
            print(f"Final cost: {self.optimization_result.fun:.6f}")
            print(f"Optimized parameter values: {final_param_values}")
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
        if not self.loss_history:
            raise ValueError("No optimization history available. Run VQE first.")
            
        plt.figure(figsize=(10, 6))
        iterations = range(len(self.loss_history))
        plt.plot(iterations, self.loss_history, 'o-', label='loss (Hartree)')
        
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
