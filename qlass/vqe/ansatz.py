from perceval.converters import QiskitConverter
from qiskit import transpile
from qiskit.circuit.library import TwoLocal

import perceval as pcvl
import numpy as np
from perceval.components import Processor
from qlass.utils.utils import rotate_qubits

qiskit_converter = QiskitConverter(backend_name="Naive") #or SLOS

def le_ansatz(lp: np.ndarray, pauli_string: str) -> Processor:
    """
    Creates Perceval quantum processor for the Linear Entangled Ansatz.
    This ansatz consists of a layer of parametrized rotations, followed by 
    a layer of CNOT gates, and finally another layer of parametrized rotations.

    Args:
        lp (np.ndarray): Array of parameter values
        pauli_string (str): Pauli string

    Returns:
        Processor: The quantum circuit as a Perceval processor
    """
    num_qubits = len(pauli_string)
    ansatz = TwoLocal(num_qubits, 'ry', 'cx', reps=1)

    ansatz_assigned = ansatz.assign_parameters(lp)
    ansatz_transpiled = transpile(ansatz_assigned, basis_gates=['u3', 'cx'], optimization_level=3)

    ansatz_rot = rotate_qubits(pauli_string, ansatz_transpiled.copy())
    processor = qiskit_converter.convert(ansatz_rot, use_postselection=True)  
    processor.with_input(pcvl.LogicalState([0]*num_qubits))

    return processor