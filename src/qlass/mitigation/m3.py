import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


class PhotonicErrorModel:
    """
    A simple error model for a multi-mode photonic system.

    Assumes uncorrelated errors between modes.
    Errors are defined by a probability matrix for each mode:
    P[i, j] = probability of measuring 'i' photons when 'j' were present.
    """

    def __init__(self, num_modes: int, max_photons_per_mode: int = 2):
        self.num_modes = num_modes
        self.max_photons = max_photons_per_mode
        # calibration_data will store a list of probability matrices, one for each mode.
        self.calibration_data: list[np.ndarray] = [
            np.identity(self.max_photons + 1) for _ in range(self.num_modes)
        ]

    def set_mode_calibration(self, mode_index: int, prob_matrix: np.ndarray) -> None:
        """
        Sets the calibration data (error matrix) for a specific mode.

        Args:
            mode_index (int): The index of the mode.
            prob_matrix (np.ndarray): A (max_photons+1) x (max_photons+1) matrix
                                      where P[i,j] is P(measure=i | ideal=j).
        """

        if prob_matrix.shape != (self.max_photons + 1, self.max_photons + 1):
            raise ValueError("Probability matrix has incorrect dimensions.")
        # Ensure columns sum to 1
        if not np.allclose(np.sum(prob_matrix, axis=0), 1.0):
            raise ValueError(
                f"Columns of probability matrix for mode {mode_index} do not sum to 1."
            )

        self.calibration_data[mode_index] = prob_matrix

    def get_error_prob(self, mode_index: int, measured_photons: int, ideal_photons: int) -> float:
        """Returns P(measure | ideal) for a single mode."""
        if ideal_photons > self.max_photons or measured_photons > self.max_photons:
            # If the state is outside our calibrated model, assume zero probability.
            # A more sophisticated model might handle this differently.
            return 0.0
        return float(self.calibration_data[mode_index][measured_photons, ideal_photons])


class M3Mitigator:
    """
    Implements the Matrix-Free Measurement Mitigation (M3) method for
    photonic experiments measuring Fock states.
    """

    def __init__(self, error_model: PhotonicErrorModel):
        """
        Initializes the mitigator.

        Args:
            error_model (PhotonicErrorModel): The calibrated error model for the device.
        """
        self.model = error_model

    def _calculate_A_element(self, row_state: tuple[int, ...], col_state: tuple[int, ...]) -> float:
        """
        Calculates a single element A_row,col of the full assignment matrix.
        A_row,col = P(measure=row_state | ideal=col_state).
        """
        prob = 1.0
        for i in range(self.model.num_modes):
            # Safe access in case state length differs from num_modes slightly or logic differs
            # But typically len(row_state) == num_modes
            r = row_state[i] if i < len(row_state) else 0
            c = col_state[i] if i < len(col_state) else 0
            prob *= self.model.get_error_prob(i, r, c)
        return prob

    def _calculate_column_normalizations(
        self, subspace_states: list[tuple[int, ...]]
    ) -> dict[tuple[int, ...], float]:
        """
        Pre-calculates the sum over the rows in the subspace for each column state.
        This is needed to renormalize A_tilde.
        """
        norms = {}
        for col_state in subspace_states:
            norm = 0.0
            for row_state in subspace_states:
                norm += self._calculate_A_element(row_state, col_state)
            norms[col_state] = norm if norm > 1e-9 else 1.0  # Avoid division by zero
        return norms

    def mitigate(
        self, noisy_counts: dict[tuple[int, ...], int], tol: float = 1e-5
    ) -> dict[tuple[int, ...], float]:
        """
        Performs the mitigation using a preconditioned GMRES iterative solver.

        Args:
            noisy_counts (dict): A dictionary mapping measured Fock state tuples to counts.
                                 e.g., {(1, 0, 1): 750, (1, 0, 0): 150}
            tol (float): Tolerance for the GMRES solver.

        Returns:
            dict: A dictionary mapping Fock state tuples to their mitigated probabilities.
        """
        # The subspace is the set of unique Fock states observed in the measurement.
        # We use a sorted list to ensure a consistent ordering.
        # Use tuple conversion for sorting key to handle objects like exqalibur.FockState
        subspace_states = sorted(noisy_counts.keys(), key=lambda x: tuple(x))
        subspace_dim = len(subspace_states)

        # Pre-calculate the column normalizations for the reduced matrix A_tilde
        column_normalizations = self._calculate_column_normalizations(subspace_states)

        def matvec(x: np.ndarray) -> np.ndarray:
            """
            The core matrix-free method: computes A_tilde * x.
            Defined as a closure to capture local `subspace_states` and `column_normalizations`.
            """
            y = np.zeros_like(x)
            # x is a vector where x_j corresponds to the j-th state in subspace_states
            # y_i = sum_j (A_tilde_ij * x_j)
            for i, row_state in enumerate(subspace_states):
                for j, col_state in enumerate(subspace_states):
                    # Calculate the renormalized matrix element on the fly
                    a_ij = self._calculate_A_element(row_state, col_state)
                    a_tilde_ij = a_ij / column_normalizations[col_state]
                    y[i] += a_tilde_ij * x[j]
            return y

        # 1. Define the LinearOperator for SciPy's solver
        A_tilde_op = LinearOperator((subspace_dim, subspace_dim), matvec=matvec)

        # 2. Get the noisy probability vector within the subspace
        total_shots = sum(noisy_counts.values())
        p_noisy = np.array([noisy_counts[state] / total_shots for state in subspace_states])

        # 3. Create a Jacobi (diagonal) preconditioner for faster convergence
        diag_A_tilde = np.zeros(subspace_dim)
        for i, state in enumerate(subspace_states):
            a_ii = self._calculate_A_element(state, state)
            diag_A_tilde[i] = a_ii / column_normalizations[state]

        # The preconditioner M should approximate the inverse of A_tilde.
        # A simple choice is the inverse of its diagonal.
        with np.errstate(divide="ignore"):
            M_inv_diag = 1.0 / diag_A_tilde
            M_inv_diag[np.isinf(M_inv_diag)] = 1.0  # Handle division by zero safely if any

        M_inv = np.diag(M_inv_diag)
        preconditioner = LinearOperator((subspace_dim, subspace_dim), matvec=lambda x: M_inv @ x)

        # 4. Solve the linear system A_tilde * p_ideal = p_noisy
        p_ideal_subspace, exit_code = gmres(A_tilde_op, p_noisy, atol=tol, M=preconditioner)

        if exit_code != 0:
            warnings.warn(
                f"GMRES solver did not converge. Exit code: {exit_code}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Normalize the result to ensure it's a valid probability distribution
        # Some values might be slightly negative due to numerics, clip them to 0??
        # Usually M3 can produce negative quasi-probs, which is expected for mitigation sometimes.
        # But for VQE we typically want to sum them up. We'll leave them as is or normalize.
        # Normalizing to sum to 1 is safe.
        p_sum = np.sum(p_ideal_subspace)
        if abs(p_sum) > 1e-9:
            p_ideal_subspace /= p_sum

        # 5. Map the mitigated probability vector back to the Fock state representation
        mitigated_distribution = dict(zip(subspace_states, p_ideal_subspace, strict=True))

        return mitigated_distribution
