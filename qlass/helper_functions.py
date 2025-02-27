
import numpy as np
import perceval as pcvl
import qiskit
import exqalibur
from typing import List, Tuple, Dict, Union

from qiskit.circuit.library import TwoLocal

H_matrix = (1/np.sqrt(2)) * pcvl.Matrix([[1.0, 1.0], [1.0, -1.0]])
M_matrix = (1/np.sqrt(2)) * pcvl.Matrix([[1.0, 1.0], [1.0j, -1.0j]])
mzi = (pcvl.BS() // (0, pcvl.PS(pcvl.Parameter("phi1"))) // 
        pcvl.BS() // (0, pcvl.PS(pcvl.Parameter("phi2"))))
H_circ = pcvl.Circuit.decomposition(H_matrix, mzi, shape=pcvl.InterferometerShape.TRIANGLE)
M_circ = pcvl.Circuit.decomposition(M_matrix, mzi, shape=pcvl.InterferometerShape.TRIANGLE)


def is_qubit_state(state: exqalibur.FockState) -> Union[Tuple[int, ...], bool]:
    """
    Check if a given state is a valid qubit state.

    Args:
    state (exqalibur.FockState): The state to check

    Returns:
    Union[Tuple[int, ...], bool]: The corresponding qubit state if valid, False otherwise
    """
    q_state = []
    for i in range(state.m // 2):
        q = state[2*i : 2*i+2]
        if (q[0] == 0 and q[1] == 1):
            q_state.append(1)
        elif (q[0] == 1 and q[1] == 0):
            q_state.append(0)
        else:
            return False
        
    return tuple(q_state)

def qubit_state_marginal(prob_dist: pcvl.BSDistribution) -> Dict[Tuple[int, ...], float]:
    """
    Calculate the frequencies of measured qubit states from frequencies of 
    sampled Fock states.

    Args:
    res (List[exqalibur.FockState]): Sampled Fock states

    Returns:
    Dict[Tuple[int, ...], float]: Frequencies of measured qubit states
    """
    q_state_frequency = {}
    total_prob_mass = 0
    for state in prob_dist:
        q_state = is_qubit_state(state)
        if q_state is not False:
            total_prob_mass += prob_dist[state]
            q_state_frequency[q_state] = prob_dist[state]
    
    for key in q_state_frequency.keys():
        q_state_frequency[key] /= total_prob_mass
    
    return q_state_frequency

def get_probabilities(samples: List[exqalibur.FockState]) -> Dict[exqalibur.FockState, float]:
    """
    Get the probabilities of sampled Fock states.

    Args:
    samples (List[exqalibur.FockState]): Sampled Fock states

    Returns:
    Dict[exqalibur.FockState, float]: Probabilities of sampled Fock states
    """
    prob_dist = {}
    for state in samples:
        if state in prob_dist:
            prob_dist[state] += 1
        else:
            prob_dist[state] = 1
    
    total_samples = len(samples)
    for key in prob_dist.keys():
        prob_dist[key] /= total_samples
    
    return prob_dist

def compute_energy(pauli_bin: Tuple[int, ...], res: Dict[Tuple[int, ...], float]) -> float:
    """
    Compute the expectation value for a given Pauli string and measurement results.

    Args:
    pauli_bin (Tuple[int, ...]): A tuple of 0's and 1's (0's are identities, 1's are non-identities (X or Z))
    res (Dict[Tuple[int, ...], float]): Frequencies of measured qubit bitstrings

    Returns:
    float: The corresponding expectation value
    """
    for key in res.keys():
        inner = np.dot(key, pauli_bin)
        sign = (-1)**inner
        res[key] *= sign
    
    energy = float(np.sum(np.fromiter(res.values(), dtype=float), where=np.isfinite))
    return energy

def pauli_string_bin(pauli_string: str) -> Tuple[int, ...]:
    """
    Convert a Pauli string to a binary representation.

    Args:
    pauli_string (str): A string representation of Pauli operators (e.g., "IXZI")

    Returns:
    Tuple[int, ...]: Binary representation of the Pauli string
    """
    return tuple(0 if c == "I" else 1 for c in pauli_string)

def rotate_qubits(pauli_string: str, vqe_circuit: pcvl.Circuit | qiskit.QuantumCircuit) -> pcvl.Circuit:
    """
    Apply the correct rotations on corresponding qubits for expectation value computation.

    Args:
    pauli_string (str): A string representation of Pauli operators
    vqe_circuit (pcvl.Circuit): The VQE circuit to modify

    Returns:
    pcvl.Circuit: The modified VQE circuit with applied rotations
    """
    if isinstance(vqe_circuit, qiskit.QuantumCircuit):
        for i, pauli in enumerate(pauli_string):
            if pauli == "X":
                vqe_circuit.h(i)
            elif pauli == "Y":
                vqe_circuit.ry(np.pi/2, i)

    else: 
        for i, pauli in enumerate(pauli_string):
            qubit = (2*i, 2*i+1)
            if pauli == "X":
                vqe_circuit.add(qubit, H_circ)
            elif pauli == "Y":
                vqe_circuit.add(qubit, M_circ)
    
    return vqe_circuit


def loss_function(lp: np.ndarray, H: Dict[str, float], executor) -> float:
    """
    Compute the loss function for the VQE algorithm.

    Args:
    lp (np.ndarray): Array of parameter values
    H (Dict[str, float]): Hamiltonian dictionary
    executor: A callable function that executes the quantum circuit

    Returns:
    float: The computed loss value
    """
    loss = 0.0
    for pauli_string, coefficient in H.items():
        samples = executor(lp, pauli_string)

        prob_dist = get_probabilities(samples['results'])
        pauli_bin = pauli_string_bin(pauli_string)

        qubit_state_marg = qubit_state_marginal(prob_dist)
        expectation = compute_energy(pauli_bin, qubit_state_marg)
        loss += coefficient * expectation
    
    return loss

def LE_ansatz(num_qubits):
    return TwoLocal(num_qubits, 'ry', 'cz', reps=1)

def linear_circuit_to_unitary(circuit: pcvl.Circuit) -> np.ndarray:
    unitary_matrix = np.array(circuit.compute_unitary())
    return unitary_matrix