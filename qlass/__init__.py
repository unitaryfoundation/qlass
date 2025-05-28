# qlass/__init__.py

# export from compiler module
from .compiler.compiler import compile

# export from problems module
from .quantum_chemistry.classical_solution import hamiltonian_matrix, brute_force_minimize
from .quantum_chemistry.hamiltonians import LiH_hamiltonian

# export from vqe module
from .vqe.ansatz import le_ansatz, custom_unitary_ansatz

# export from utils module 
from .utils.utils import loss_function, rotate_qubits

# Define the public API exposed directly under 'qlass'
__all__ = [
    "compile",
    "hamiltonian_matrix",
    "brute_force_minimize",
    "LiH_hamiltonian",
    "le_ansatz",
    "custom_unitary_ansatz",
    "loss_function",
    "rotate_qubits",
]

# Version information
__version__ = "0.1.1"
