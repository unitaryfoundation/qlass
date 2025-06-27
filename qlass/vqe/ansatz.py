from perceval.converters import QiskitConverter
from perceval.utils import NoiseModel
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import TwoLocal

import perceval as pcvl
import numpy as np
from perceval.components import Processor
from qlass.utils import rotate_qubits
from qlass.compiler import compile

def le_ansatz(lp: np.ndarray, pauli_string: str, noise_model: NoiseModel = None) -> Processor:
    """
    Creates Perceval quantum processor for the Linear Entangled Ansatz.
    This ansatz consists of a layer of parametrized rotations, followed by 
    a layer of CNOT gates, and finally another layer of parametrized rotations.

    Args:
        lp (np.ndarray): Array of parameter values
        pauli_string (str): Pauli string
        noise_model (NoiseModel): A perceval NoiseModel object representing the noise model

    Returns:
        Processor: The quantum circuit as a Perceval processor
    """
    num_qubits = len(pauli_string)
    ansatz = TwoLocal(num_qubits, 'ry', 'cx', reps=1)

    ansatz_assigned = ansatz.assign_parameters(lp)
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=['u3', 'cx'], optimization_level=3)

    ansatz_rot = rotate_qubits(pauli_string, ansatz_transpiled.copy())
    processor = compile(ansatz_rot, input_state=pcvl.LogicalState([0]*num_qubits), noise_model=noise_model)  

    return processor

def custom_unitary_ansatz(lp: np.ndarray, 
                          pauli_string: str, 
                          U: np.ndarray, 
                          noise_model: NoiseModel = None) -> Processor:
    """
    Creates Perceval quantum processor that directly implements a given unitary matrix.
    This function serves as a custom ansatz that bypasses circuit construction and instead 
    loads a full unitary matrix representing the quantum operation.

    The unitary is embedded in a Qiskit QuantumCircuit, converted to Perceval format,
    and returned as a Processor object with post-selection enabled.

    Args:
        lp (np.ndarray): Placeholder array of parameter values (unused, for compatibility).
        pauli_string (str): Pauli string used to determine the number of qubits.
        U (np.ndarray): A unitary matrix of shape (2^n, 2^n) where n = len(pauli_string).
        noise_model (NoiseModel): A perceval NoiseModel object representing the noise model.

    Returns:
        Processor: The quantum circuit as a Perceval processor implementing unitary U.
    """
    num_qubits = len(pauli_string)
    dim = 2 ** num_qubits

    if U.shape != (dim, dim):
        raise ValueError(f"Expected unitary of shape ({dim}, {dim}), got {U.shape}")
    if not np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-10):
        raise ValueError("Matrix U is not unitary")

    qc = QuantumCircuit(num_qubits)
    qc.unitary(U, range(num_qubits))

    qc_rotated = rotate_qubits(pauli_string, qc.copy())
    processor = compile(qc_rotated, input_state=pcvl.LogicalState([0]*num_qubits), noise_model=noise_model)
    processor.with_input(pcvl.LogicalState([0]*num_qubits))

    return processor
