# export from compiler module
from .compiler.compiler import ResourceAwareCompiler, compile, compile_circuit, generate_report
from .compiler.hardware_config import HardwareConfig

# export from mitigation module
from .mitigation import (
    M3Mitigator,
    PhotonicErrorModel,
    ZNEMitigator,
    fold_global_interferometer,
    scale_loss_config,
)

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
from .utils.utils import draw_circuit, e_vqe_loss_function, loss_function, rotate_qubits

# export from vqe module
from .vqe.ansatz import custom_unitary_ansatz, kerr_ansatz, le_ansatz
from .vqe.vqe import VQE

# Define the public API exposed directly under 'qlass'
__all__ = [
    "compile",
    "compile_circuit",
    "hamiltonian_matrix",
    "brute_force_minimize",
    "LiH_hamiltonian",
    "Hchain_KS_hamiltonian",
    "transformation_Hmatrix_Hqubit",
    "pauli_commute",
    "group_commuting_pauli_terms",
    "sparsepauliop_dictionary",
    "VQE",
    "le_ansatz",
    "custom_unitary_ansatz",
    "kerr_ansatz",
    "draw_circuit",
    "loss_function",
    "e_vqe_loss_function",
    "rotate_qubits",
    "ResourceAwareCompiler",
    "HardwareConfig",
    "generate_report",
    "M3Mitigator",
    "PhotonicErrorModel",
    "ZNEMitigator",
    "fold_global_interferometer",
    "scale_loss_config",
]

# Version information
from importlib.metadata import version

__version__ = version("qlass")
