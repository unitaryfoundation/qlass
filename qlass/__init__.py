"""
qlass - Quantum Linear-optical Algorithms and Simulations for the QLASS project.
"""

from .compiler import compile
from .ansatz import le_ansatz
from .helper_functions import (
    is_qubit_state,
    qubit_state_marginal,
    get_probabilities,
    compute_energy,
    pauli_string_bin,
    rotate_qubits,
    loss_function,
    LE_ansatz,
    linear_circuit_to_unitary
)
from .hamiltonians import (
    sparsepauliop_dictionary,
    LiH_hamiltonian,
    generate_random_hamiltonian,
    LiH_hamiltonian_tapered
)
from .classical_solution import (
    pauli_string_to_matrix,
    hamiltonian_matrix,
    brute_force_minimize,
    Lanczos,
    eig_decomp_lanczos
)

# Version information
__version__ = "0.1.1"