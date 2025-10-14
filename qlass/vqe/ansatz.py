from perceval.converters import QiskitConverter
from perceval.utils import NoiseModel
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
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
    ansatz = n_local(num_qubits, 'ry', 'cx', reps=1, entanglement='linear')

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


def list_of_ones(computational_basis_state: int, n_qubits):
    """
    Indices of ones in the binary expansion of an integer in big endian
    order. e.g. 010110 -> [1, 3, 4] (which is the reverse of the qubit ordering...)
    """

    bitstring = format(computational_basis_state, 'b').zfill(n_qubits)

    return [abs(j - n_qubits + 1) for j in range(len(bitstring)) if bitstring[j] == '1']



def hf_ansatz(layers: int, n_orbs: int, lp: np.ndarray, pauli_string: str, method: str, noise_model: NoiseModel = None) -> Processor:
    """
    Args:
        layers (int): number of circuit layers
        n_orbs (int): number of orbitals
        lp (np.ndarray): array of parameter values
        pauli_string (str): Pauli string
        method (str): 'WFT' n_occ spin orbitals mapped into n_aubits, 'DFT' spatial orbitals mapped into log2(N) qubits by considering only spin-alpha or spin-beta blocks
        see article https://scipost.org/SciPostPhys.14.3.055 for DFT mapping.
        noise_model (NoiseModel): A perceval NoiseModel object representing the noise model
    Return:
        Processors list[processor]: The list of quantum circuits as a Perceval processors
    """

    '''Circuit implementation'''
    from qiskit.visualization import circuit_drawer
    from itertools import combinations
    num_qubits = len(pauli_string)
    if method == "WFT":
        n_occ = n_orbs * 2 # number of spin orbitals
    elif method == "DFT":
        n_occ = n_orbs // 2 # number of spatial orbitals
    initial_circuits = []

    for i in range(n_occ): initial_circuits += [QuantumCircuit(num_qubits)]



    '''Intial states'''
    for state in range(n_occ):  # binarystring representation of the integer
        for i in list_of_ones(state, num_qubits):
            initial_circuits[state].x(i)


    circuits = [
        TwoLocal(num_qubits, 'ry', 'cx', 'linear', reps=layers,
                 initial_state=initial_circuits[state]) for state in range(n_occ)]

    circuits = [c.assign_parameters(lp) for c in circuits]




    intial_states = []
    for c in range(len(initial_circuits)):
        sv = Statevector.from_instruction(initial_circuits[c])
        index = list(sv.data).index(1 + 0j)  # position of the "1"
        bitstring = format(index, f'0{sv.num_qubits}b')
        bits_list = [int(b) for b in bitstring]
        intial_states.append(pcvl.LogicalState(bits_list))

    # print('Printing circuit')
    # print(circuits[0].decompose())


    ansatz_transpiled = [transpile(c, basis_gates=['u3', 'cx'], optimization_level=3) for c in circuits]

    ansatz_rot = [rotate_qubits(pauli_string, at.copy()) for at in ansatz_transpiled]

    processors = [compile(ar, input_state=st, noise_model=noise_model) for ar, st in zip(ansatz_rot, intial_states)]


    return processors
