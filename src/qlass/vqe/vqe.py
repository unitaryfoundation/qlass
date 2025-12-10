from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from qlass.utils import e_vqe_loss_function, loss_function
from qlass.utils.utils import DataCollector


class VQE:
    """
    Variational Quantum Eigensolver for photonic quantum computing.

    This class provides a high-level interface for running VQE experiments
    on photonic simulators, finding ground state energies of quantum systems.
    """

    def __init__(
        self,
        hamiltonian: dict[str, float],
        executor: Callable,
        num_params: int,
        optimizer: str = "COBYLA",
        executor_type: str = "sampling",
        initial_state: np.ndarray | None = None,  # For now only relevant for photonic_unitary,
        ancillary_modes: list[int] | None = None,
    ):
        """
        Initialize the VQE solver.

        Args:
            hamiltonian (Dict[str, float]): Hamiltonian dictionary with Pauli string keys
                                           and coefficient values
            executor (Callable): Custom executor function, if None, a default one will be created
            num_params (int): number of parameters that the executor accepts
            optimizer (str): Optimization method to use. Any method supported by scipy.optimize.minimize
            executor_type (str): Type of executor.
                - "sampling": Uses measurement-based sampling.
                - "qubit_unitary": For qubit-based simulations (e.g., standard circuit-based quantum computing).
                - "photonic_unitary": For photonic quantum computing,
                where the initial state and ancillary modes may be relevant for post-selection.
                Use this when simulating photonic systems or when post-selection on ancillary modes is required.
            ancillary_modes (List[int], optional): List of ancillary mode indices
                for post-selection when using 'photonic_unitary' executor.
        """
        self.hamiltonian = hamiltonian
        self.executor = executor
        self.num_params = num_params
        self.optimizer = optimizer
        self.initial_state = initial_state
        self.ancillary_modes = ancillary_modes

        # Extract number of qubits from the Hamiltonian
        self.num_qubits = len(next(iter(hamiltonian.keys())))

        # Executor type for loss function computation
        if executor_type in ["sampling", "qubit_unitary", "photonic_unitary"]:
            self.executor_type = executor_type
        else:
            raise ValueError(
                f"Invalid executor_type: {executor_type}. Must be either sampling, qubit_unitary or photonic_unitary."
            )

        # Results storage
        self.optimization_result: OptimizeResult | None = None
        self.energy_history: list[float] = []
        self.parameter_history: list[np.ndarray] = []
        self.loss_history: list[float] = []
        self.energy_collector = DataCollector()

    def _callback(
        self, params: np.ndarray, cost_type: str = "VQE", weight_option: str = "weighted"
    ) -> None:
        """Callback function to record optimization progress."""
        if cost_type == "e-VQE":
            # Ensemble-VQE mode
            cost = e_vqe_loss_function(
                params,
                self.hamiltonian,
                self.executor,
                self.energy_collector,
                weight_option=weight_option,
            )
            self.loss_history.append(cost)
        else:
            # Standard VQE mode
            if self.executor_type == "qubit_unitary":
                from qlass.utils import loss_function_matrix

                energy = loss_function_matrix(params, self.hamiltonian, self.executor)
            elif self.executor_type == "photonic_unitary":
                from qlass.utils import loss_function_photonic_unitary

                energy = loss_function_photonic_unitary(
                    params,
                    self.hamiltonian,
                    self.executor,
                    self.initial_state,
                    self.ancillary_modes,
                )
            else:
                energy = loss_function(params, self.hamiltonian, self.executor)

            self.energy_history.append(energy)

            # Always record parameters
        self.parameter_history.append(params.copy())

    def run(
        self,
        initial_params: np.ndarray | None = None,
        max_iterations: int = 100,
        verbose: bool = True,
        weight_option: str = "weighted",
        cost: str = "VQE",
    ) -> float:
        """
        Run a Variational Quantum Eigensolver (VQE) or ensemble-VQE optimization to find
        the ground state energy of a given Hamiltonian.

        This method executes the classical optimization loop using SciPy's ``minimize``
        function, updating the variational parameters of a quantum circuit. It supports
        both standard VQE and ensemble-VQE (e-VQE) algorithms, with customizable weighting
        schemes for the ensemble. Progress can be logged at each step if ``verbose=True``.

        Parameters
        ----------
        initial_params : np.ndarray, optional
            Initial parameters for the variational quantum circuit. If ``None``, random
            parameters will be generated uniformly in [0,1). Default is ``None``.
        max_iterations : int, optional
            Maximum number of iterations for the optimizer. Default is 100.
        verbose : bool, optional
            If ``True``, prints progress information including number of qubits, parameters,
            and final energies. Default is ``True``.
        weight_option : {'weighted', 'equi', 'ground_state_only'}, optional
            Weighting scheme for ensemble-VQE:
            - ``'weighted'`` : linearly decreasing weights (w_i < w_j for i > j)
            - ``'equi'`` : equal weights for all occupied orbitals (w_i = w_j).
            - ``'ground_state_only'`` : only the ground state contributes (w_0 = 1)
            Default is ``'weighted'``.
        cost : {'VQE', 'e-VQE'}, optional
            Choice of optimization algorithm:
            - ``'VQE'`` : standard single-state VQE optimization.
            - ``'e-VQE'`` : ensemble-VQE using multiple states and the specified weights.
            Default is ``'VQE'``.

        Returns
        -------
        float
            Minimum cost (energy) found by the optimizer.

        Notes
        -----
        - The method resets the loss and parameter history at the start of each run.
        - For ensemble-VQE, the ``weight_option`` determines how individual state energies
          are combined into the total loss.
        - Exact energies for the Hamiltonian are computed at the end using a brute-force
          diagonalization routine for reference.
        - Verbose mode prints information about optimizer progress, final energies, and
          number of function evaluations.
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
            print(f"Executor type: {self.executor_type}")

        if cost == "VQE":
            if self.executor_type == "qubit_unitary":
                from qlass.utils import loss_function_matrix

                self.optimization_result = minimize(
                    loss_function_matrix,
                    initial_params,
                    args=(self.hamiltonian, self.executor),
                    method=self.optimizer,
                    callback=lambda p: self._callback(p, cost_type="VQE"),
                    options={"maxiter": max_iterations},
                )
            elif self.executor_type == "photonic_unitary":
                from qlass.utils import loss_function_photonic_unitary

                self.optimization_result = minimize(
                    loss_function_photonic_unitary,
                    initial_params,
                    args=(
                        self.hamiltonian,
                        self.executor,
                        self.initial_state,
                        self.ancillary_modes,
                    ),
                    method=self.optimizer,
                    callback=lambda p: self._callback(p, cost_type="VQE"),
                    options={"maxiter": max_iterations},
                )

            elif self.executor_type == "sampling":
                self.optimization_result = minimize(
                    loss_function,
                    initial_params,
                    args=(self.hamiltonian, self.executor),
                    method=self.optimizer,
                    callback=lambda p: self._callback(p, cost_type="VQE"),
                    options={"maxiter": max_iterations},
                )

        elif cost == "e-VQE":
            if self.executor_type == "sampling":
                self.optimization_result = minimize(
                    e_vqe_loss_function,
                    initial_params,
                    args=(self.hamiltonian, self.executor, self.energy_collector, weight_option),
                    method=self.optimizer,
                    callback=lambda p: self._callback(
                        p, cost_type="e-VQE", weight_option=weight_option
                    ),
                    options={"maxiter": max_iterations},
                )
            else:
                raise ValueError("option: e-VQE takes only executor_type: sampling")
        else:
            raise ValueError("Invalid cost option. Use 'VQE' or 'e-VQE'.")

        if verbose:
            assert self.optimization_result is not None
            print("Optimization complete!")
            if cost == "VQE":
                print(f"Final energy: {self.optimization_result.fun:.6f}")
            else:
                print(f"Final cost: {self.optimization_result.fun:.6f}")
                last_entries = [
                    float(f"{vals[-1]:.6f}") for vals in self.energy_collector.energy_data.values()
                ]
                print(f"Final energies: {last_entries}")
            print(f"Number of iterations: {self.optimization_result.nfev}")

        assert self.optimization_result is not None
        return float(self.optimization_result.fun)

    def get_optimal_parameters(self) -> np.ndarray:
        """Get the optimal parameters found during optimization."""
        if self.optimization_result is None:
            raise ValueError("VQE optimization has not been run yet.")
        return np.asarray(self.optimization_result.x)

    def plot_convergence(self, exact_energy: float | None = None) -> None:
        """
        Plot the energy convergence during the optimization.

        Args:
            exact_energy (float): Exact ground state energy for comparison, if available
        """
        if not self.energy_history:
            raise ValueError("No optimization history available. Run VQE first.")

        plt.figure(figsize=(10, 6))
        iterations = range(len(self.energy_history))
        plt.plot(iterations, self.energy_history, "o-", label="VQE Energy")

        if exact_energy is not None:
            plt.axhline(y=exact_energy, color="r", linestyle="--", label="Exact Energy")

        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("VQE Convergence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def compare_with_exact(self, exact_energy: float | None = None) -> dict[str, float]:
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
        relative_error = absolute_error / abs(exact_energy) if exact_energy != 0 else float("inf")

        comparison = {
            "vqe_energy": vqe_energy,
            "exact_energy": exact_energy,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
        }

        return comparison
