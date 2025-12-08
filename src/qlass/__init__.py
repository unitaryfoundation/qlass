# export from compiler module
from .compiler.compiler import ResourceAwareCompiler, compile, generate_report
from .compiler.hardware_config import HardwareConfig

# export from problems module
from .quantum_chemistry.classical_solution import brute_force_minimize, hamiltonian_matrix
from .quantum_chemistry.hamiltonians import (
    Hchain_KS_hamiltonian,
    LiH_hamiltonian,
    group_commuting_pauli_terms,
    pauli_commute,
    sparsepauliop_dictionary,
    transformation_Hmatrix_Hqubit,
)

# export from utils module
from .utils.utils import e_vqe_loss_function, loss_function, rotate_qubits

# export from vqe module
from .vqe.ansatz import custom_unitary_ansatz, le_ansatz

# Define the public API exposed directly under 'qlass'
__all__ = [
    "compile",
    "hamiltonian_matrix",
    "brute_force_minimize",
    "LiH_hamiltonian",
    "Hchain_KS_hamiltonian",
    "transformation_Hmatrix_Hqubit",
    "pauli_commute",
    "group_commuting_pauli_terms",
    "sparsepauliop_dictionary",
    "le_ansatz",
    "custom_unitary_ansatz",
    "loss_function",
    "e_vqe_loss_function",
    "rotate_qubits",
    "ResourceAwareCompiler",
    "HardwareConfig",
    "generate_report",
]

# Version information
from importlib.metadata import version

__version__ = version("qlass")
