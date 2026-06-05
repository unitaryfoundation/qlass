"""
Demo: sampling-based Bose-Hubbard energy vs exact matrix multiplication.

Trial state: psi = (|1,0> + |0,1>) / sqrt(2) on a 2-site dimer.
Hamiltonian: H = -t(a†_0 a_1 + h.c.) - mu*(n_0+n_1) + U/2*sum n_i(n_i-1)

Run:
    python examples/bose_hubbard_dimer_example.py
"""

import numpy as np
from openfermion.linalg import boson_operator_sparse
from openfermion.ops import BosonOperator

from qlass.utils import loss_function_bose_hubbard

# Truncation level: Fock space per mode goes up to n_max = TRUNC-1 photons
TRUNC = 3
NUM_SAMPLES = 10_000

# 2-site Bose-Hubbard dimer parameters
t = 1.0
mu = 0.5
U = 4.0

# Build the Hamiltonian as an OpenFermion BosonOperator
H = (
    BosonOperator("0^ 1", -t)
    + BosonOperator("1^ 0", -t)
    + BosonOperator("0^ 0", -mu)
    + BosonOperator("1^ 1", -mu)
    + BosonOperator("0^ 0^ 0 0", U / 2)
    + BosonOperator("1^ 1^ 1 1", U / 2)
)

# Trial state psi = (|1,0> + |0,1>) / sqrt(2) in the truncated Fock basis
# Basis ordering: state index = n0 * TRUNC + n1
psi = np.zeros(TRUNC * TRUNC, dtype=complex)
psi[1 * TRUNC + 0] = 1.0 / np.sqrt(2)  # |1,0>
psi[0 * TRUNC + 1] = 1.0 / np.sqrt(2)  # |0,1>


def sample_from_state(state, num_samples):
    """Draw Fock-state samples from the probability distribution |state|^2."""
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()
    indices = np.random.choice(len(probs), size=num_samples, p=probs)
    # Convert flat index to (n0, n1) occupation tuple
    return [(idx // TRUNC, idx % TRUNC) for idx in indices]


def apply_bs(state):
    """Apply a 50/50 beam splitter on modes (0,1) within the 1-photon sector."""
    # For the trial state living in span{|1,0>, |0,1>} the BS acts as the Hadamard:
    #   |1,0> -> (|1,0> + |0,1>) / sqrt(2)
    #   |0,1> -> (|1,0> - |0,1>) / sqrt(2)
    rotated = state.copy()
    amp_10 = state[1 * TRUNC + 0]
    amp_01 = state[0 * TRUNC + 1]
    rotated[1 * TRUNC + 0] = (amp_10 + amp_01) / np.sqrt(2)
    rotated[0 * TRUNC + 1] = (amp_10 - amp_01) / np.sqrt(2)
    return rotated


def executor(params, measurement_type, *modes):
    """
    Executor for loss_function_bose_hubbard.

    Samples Fock occupations from the fixed trial state using numpy.
    In a real experiment this would drive a photonic processor.
    """
    if measurement_type == "identity":
        # Measure in computational basis — no rotation needed
        samples = sample_from_state(psi, NUM_SAMPLES)
    elif measurement_type == "hop":
        # Apply beam-splitter rotation on modes (p, q) before sampling
        rotated = apply_bs(psi)
        samples = sample_from_state(rotated, NUM_SAMPLES)
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")
    return {"results": samples}


def main():
    print("=== Bose-Hubbard Dimer: Sampling vs Exact ===\n")
    print(f"Parameters: t={t}, mu={mu}, U={U}")
    print("Trial state: (|1,0> + |0,1>) / sqrt(2)")
    print(f"Samples per measurement: {NUM_SAMPLES}\n")

    # Compute exact energy via dense matrix multiplication for reference
    h_matrix = boson_operator_sparse(H, TRUNC).toarray()
    exact_energy = float(np.real(psi.conj() @ h_matrix @ psi))
    analytic_energy = -t - mu  # closed-form for this trial state

    # Compute sampling-based energy using loss_function_bose_hubbard
    sampled_energy = loss_function_bose_hubbard(np.array([0.0]), H, executor)

    print(f"Exact matrix energy:    {exact_energy:.6f}")
    print(f"Analytic (1-photon):    {analytic_energy:.6f}")
    print(f"Sampled energy:         {sampled_energy:.6f}")
    print(f"|sampled - exact|:      {abs(sampled_energy - exact_energy):.6f}\n")

    print("Term breakdown (analytic for this trial state):")
    print(f"  hopping  -t        = {-t:.6f}")
    print(f"  chem pot -mu*(1+1) = {-mu:.6f}  (mean photon number per site = 0.5)")
    print("  on-site  U/2*0     =  0.000000  (n_i in {0,1} -> n*(n-1)=0)")


if __name__ == "__main__":
    main()
