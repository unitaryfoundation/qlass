import os
from collections.abc import Callable

import numpy as np
from numba import njit


def pauli_string_to_matrix(pauli_string: str) -> np.ndarray:
    """
    Convert a Pauli string to a matrix representation.

    Args:
        pauli_string (str): A string of Pauli operators (I, X, Y, Z)

    Returns:
        np.ndarray: Matrix representation of the Pauli string

    """

    res: np.ndarray = np.array([[1.0]])

    # Define the Pauli matrices.
    I_pauli = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1.0j], [1.0j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    for c in pauli_string:
        if c == "X":
            res = np.kron(res, X)
        elif c == "Y":
            res = np.kron(res, Y)
        elif c == "Z":
            res = np.kron(res, Z)
        else:
            res = np.kron(res, I_pauli)

    return res


def hamiltonian_matrix(H: dict[str, float]) -> np.ndarray:
    """
    Convert a Hamiltonian dictionary to a matrix representation.

    Args:
        H (Dict[str, float]): Hamiltonian dictionary

    Returns:
        np.ndarray: Matrix representation of the Hamiltonian

    """

    coeffs = list(H.values())
    matrices = [
        coeff * pauli_string_to_matrix(pauli_string)
        for pauli_string, coeff in zip(H.keys(), coeffs, strict=False)
    ]

    result: np.ndarray = np.sum(matrices, axis=0)
    return result


def brute_force_minimize(H: dict[str, float]) -> float:
    """
    Compute the minimum eigenvalue of a Hamiltonian using brute force.

    Args:
        H (Dict[str, float]): Hamiltonian dictionary

    Returns:
        float: Minimum eigenvalue of the Hamiltonian

    """

    H_matrix = hamiltonian_matrix(H)
    eigenvalues = np.linalg.eigvalsh(H_matrix)

    return float(eigenvalues[0])


def _lanczos_impl(A: np.ndarray, v_init: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Core Lanczos algorithm implementation."""
    n = len(v_init)
    if m > n:
        m = n

    V = np.zeros((m, n), dtype=np.complex128)
    T = np.zeros((m, m), dtype=np.complex128)

    # Normalize the initial vector
    v = v_init / np.linalg.norm(v_init)
    V[0, :] = v

    # First step
    w = A @ v
    alpha = np.dot(w.conj(), v)
    w = w - alpha * v
    T[0, 0] = alpha

    # Main iteration loop
    for j in range(1, m):
        beta = np.linalg.norm(w)

        # If beta is very small, the process has converged or the Krylov space is exhausted
        if beta < 1e-12:
            break

        # Normalize the next vector
        v = w / beta

        # --- Start of Re-orthogonalization ---
        # Explicitly make the new vector orthogonal to all previous vectors
        # This is the key to fixing numerical stability issues.
        for i in range(j):
            v -= np.dot(V[i, :].conj(), v) * V[i, :]
        v /= np.linalg.norm(v)  # Re-normalize after correction
        # --- End of Re-orthogonalization ---

        V[j, :] = v

        # Perform the next step of the Lanczos iteration
        w = A @ v
        alpha = np.dot(w.conj(), v)
        T[j, j] = alpha

        # Update w using the previous vector
        w = w - alpha * v - beta * V[j - 1, :]

        # Store beta in the off-diagonal of T
        T[j, j - 1] = beta
        T[j - 1, j] = beta

    return T, V


# Setting the environment variable QLASS_DISABLE_JIT=1 *before importing qlass*
# skips numba JIT compilation of the Lanczos kernel. This is mainly useful for
# accurate coverage measurement (CI sets it) and for debugging; it is read once
# at import time, so changing it afterwards has no effect.
DISABLE_JIT = os.environ.get("QLASS_DISABLE_JIT", "0") == "1"

# `lanczos` is the (optionally JIT-compiled) low-level kernel returning the
# tridiagonal matrix T and the Lanczos basis V; most users want
# eig_decomp_lanczos instead.
lanczos: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]]
lanczos = njit(_lanczos_impl) if not DISABLE_JIT else _lanczos_impl


def eig_decomp_lanczos(R: np.ndarray, n: int | None = None, m: int = 100) -> np.ndarray:
    """
    Compute the eigenvalues of a matrix using the Lanczos algorithm.

    Args:
        R (np.ndarray): Matrix to compute the eigenvalues of
        n (Optional[int]): Number of smallest eigenvalues to return.
            If None (default), all Ritz values are returned.
        m (int): Number of iterations

    Returns:
        np.ndarray: Eigenvalues of the matrix, in ascending order

    """

    v0 = np.array(np.random.rand(np.shape(R)[0]), dtype=np.complex128)
    v0 /= np.sqrt(np.abs(np.dot(v0, np.conjugate(v0))))

    T, V = lanczos(R, v0, m)
    esT, _ = np.linalg.eigh(T)

    return esT if n is None else esT[:n]
