from perceval.converters import QiskitConverter
from perceval.utils import NoiseModel
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import n_local
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
    Return the indices of ones in the binary expansion of an integer, in big-endian order.

    Parameters
    ----------
    computational_basis_state : int
        Integer representing the computational basis state.
    n_qubits : int
        Total number of qubits (length of the binary string).

    Returns
    -------
    list of int
        List of indices where the binary representation has ones.
        Indices are in big-endian order. For example, for `n_qubits=6` and
        `computational_basis_state=22` (binary `010110`), the result is `[1, 3, 4]`.

    Notes
    -----
    The output index ordering corresponds to the big-endian representation of the qubits.
    """

    bitstring = format(computational_basis_state, 'b').zfill(n_qubits)

    return [abs(j - n_qubits + 1) for j in range(len(bitstring)) if bitstring[j] == '1']



def hf_ansatz(layers: int, n_orbs: int, lp: np.ndarray, pauli_string: str, method: str, cost = "VQE", noise_model: NoiseModel = None) -> Processor:
    """
        Build a Hartree–Fock-based variational ansatz using Qiskit's ``n_local`` circuit,
        combined with initial reference states and compiled into Perceval processors.

        Parameters
        ----------
        layers : int
            Number of circuit layers (repetitions) in the ansatz.
        n_orbs : int
            Number of orbitals.
        lp : np.ndarray
            Array of parameter values for the ansatz circuit.
        pauli_string : str
            Pauli operator string defining the measurement basis.
        method : str
            Mapping method. One of:
                - ``"WFT"`` : Wavefunction theory mapping (spin orbitals → qubits)
                - ``"DFT"`` : Density functional mapping (spatial orbitals → qubits)
        cost : str, optional
            Type of cost function to prepare. One of:
                - ``"VQE"`` : Return only the ground-state processor (default)
                - ``"e-VQE"`` : Return a list of processors for excited states
        noise_model : NoiseModel, optional
            A Perceval ``NoiseModel`` object representing the noise model to include in compilation.

        Returns
        -------
        Processor or list of Processor
            If ``cost="VQE"``, returns a single Perceval ``Processor`` instance.
            If ``cost="e-VQE"``, returns a list of processors based on method. One of: "WFT", "DFT".

        Raises
        ------
        ValueError
            If ``method`` or ``cost`` arguments are invalid.

        Notes
        -----
        - The ansatz is constructed by composing the initial Hartree–Fock reference circuit
          with a parameterized ``n_local`` circuit (Ry–CX entangling pattern).
        - See `https://scipost.org/SciPostPhys.14.3.055` for details on the DFT mapping.
        """

    '''Circuit implementation'''
    num_qubits = len(pauli_string)
    if method == "WFT":
        n_occ = n_orbs * 2 # number of spin orbitals
    elif method == "DFT":
        n_occ = n_orbs // 2 # number of spatial orbitals
    else:
        raise ValueError("Invalid method. Use 'WFT' or 'DFT'.")

    initial_circuits = []

    for i in range(n_occ): initial_circuits += [QuantumCircuit(num_qubits)]
    '''Intial states'''
    for state in range(n_occ):  # binarystring representation of the integer
        for i in list_of_ones(state, num_qubits):
            initial_circuits[state].x(i)

    circuits = []
    for state in range(n_occ):
        ansatz = n_local(
            num_qubits=num_qubits,
            rotation_blocks='ry',
            entanglement_blocks='cx',
            entanglement='linear',
            reps=layers,
            insert_barriers=False
        )
        # Prepend the initial state
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(initial_circuits[state], inplace=True)
        circuit.compose(ansatz, inplace=True)
        circuits.append(circuit)

    '''Assign parameters'''
    circuits = [c.assign_parameters(lp) for c in circuits]

    intial_states = []
    for c in range(len(initial_circuits)):
        sv = Statevector.from_instruction(initial_circuits[c])
        index = list(sv.data).index(1 + 0j)  # position of the "1"
        bitstring = format(index, f'0{sv.num_qubits}b')
        bits_list = [int(b) for b in bitstring]
        intial_states.append(pcvl.LogicalState(bits_list))

    ansatz_transpiled = [transpile(c, basis_gates=['u3', 'cx'], optimization_level=3) for c in circuits]

    ansatz_rot = [rotate_qubits(pauli_string, at.copy()) for at in ansatz_transpiled]

    processors = [compile(ar, input_state=st, noise_model=noise_model) for ar, st in zip(ansatz_rot, intial_states)]
    if cost == "VQE":
        return processors[0]
    elif cost == "e-VQE":
        return processors
    else:
        raise ValueError("Invalid cost option. Use 'VQE' or 'e-VQE'.")
