import time

import numpy as np
import perceval as pcvl
import exqalibur
from typing import List, Tuple, Dict, Union


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
    for i in range((state.m - 1) // 2):
        q = state[2*i+1 : 2*i+3]
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
        # print(state)
        q_state = is_qubit_state(state)
        if q_state is not False:
            total_prob_mass += prob_dist[state]
            q_state_frequency[q_state] = prob_dist[state]
    
    for key in q_state_frequency.keys():
        q_state_frequency[key] /= total_prob_mass
    
    return q_state_frequency

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
    
    return np.sum(np.fromiter(res.values(), dtype=float), where=np.isfinite)

def pauli_string_bin(pauli_string: str) -> Tuple[int, ...]:
    """
    Convert a Pauli string to a binary representation.

    Args:
    pauli_string (str): A string representation of Pauli operators (e.g., "IXZI")

    Returns:
    Tuple[int, ...]: Binary representation of the Pauli string
    """
    return tuple(0 if c == "I" else 1 for c in pauli_string)

def rotate_qubits(pauli_string: str, vqe_circuit: pcvl.Circuit) -> pcvl.Circuit:
    """
    Apply the correct rotations on corresponding qubits for expectation value computation.

    Args:
    pauli_string (str): A string representation of Pauli operators
    vqe_circuit (pcvl.Circuit): The VQE circuit to modify

    Returns:
    pcvl.Circuit: The modified VQE circuit with applied rotations
    """
    
    for i, pauli in enumerate(pauli_string):
        qubit = (2*i+1, 2*i+2)
        if pauli == "X":
            vqe_circuit.add(qubit, H_circ)
        elif pauli == "Y":
            vqe_circuit.add(qubit, M_circ)
    
    return vqe_circuit

def hamiltonian_dictionary(h: np.ndarray) -> Dict[str, float]:
    """
    Convert a 2-qubit Hamiltonian from array form to a dictionary.

    Args:
    h (np.ndarray): 2-qubit Hamiltonian in array form

    Returns:
    Dict[str, float]: Dictionary with Pauli string keys and coefficient values
    """
    pauli_strings = ["II", "IX", "IZ", "XI", "XX", "XZ", "ZI", "ZX", "ZZ"]
    return dict(zip(pauli_strings, h))

def loss_function(lp: np.ndarray, List_Parameters: List[pcvl.Parameter], 
                  ansatz: pcvl.Circuit, H: Dict[str, float]) -> float:
    """
    Compute the loss function for the VQE algorithm.

    Args:
    lp (np.ndarray): Array of parameter values
    List_Parameters (List[pcvl.Parameter]): List of Perceval parameters
    ansatz (pcvl.Circuit): The ansatz circuit
    H (Dict[str, float]): Hamiltonian dictionary

    Returns:
    float: The computed loss value
    """
    num_qubits = len(list(H.keys())[0])

    for p, value in zip(List_Parameters, lp):
        p.set_value(value)

    loss = 0.0
    for pauli_string, coefficient in H.items():
        start = time.time()
        ansatz_rot = rotate_qubits(pauli_string, ansatz.copy())
        # start = time.time()
        end = time.time()
        processor = pcvl.Processor(pcvl.NaiveBackend(), ansatz_rot)
        processor.with_input(pcvl.BasicState([0] + [1,0]*num_qubits + [0]))
        
        # print(f'rotation time: {end-start}')
        
        start = time.time()
        sampler = pcvl.algorithm.Sampler(processor)
        samples = sampler.samples(100_000)
        end = time.time()
        # print(f'sample time: {end-start}')
        
        prob_dist = sampler.probs()['results']
        pauli_bin = pauli_string_bin(pauli_string)
        start = time.time()
        expectation = compute_energy(pauli_bin, qubit_state_marginal(prob_dist))
        loss += coefficient * expectation
        end = time.time()
        # print(f'loss compute time: {end-start}')
    
    return loss

def ansatz(num_modes=6) -> Tuple[pcvl.Circuit, List[pcvl.Parameter]]:
    """
    Create the ansatz circuit for the VQE algorithm.

    Returns:
    Tuple[pcvl.Circuit, List[pcvl.Parameter]]: The ansatz circuit and list of parameters
    """
    List_Parameters = []
    VQE = pcvl.Circuit(num_modes)

    # First layer
    for i in range(1, num_modes-1, 2):
        VQE.add((i,i+1), pcvl.BS())

    for i in range(2, num_modes, 2):
        param = pcvl.Parameter(f"φ{i}")
        VQE.add((i,), pcvl.PS(phi=param))
        List_Parameters.append(param)

    for i in range(1, num_modes-1, 2):
        VQE.add((i,i+1), pcvl.BS())

    for i in range(2, num_modes, 2):
        param = pcvl.Parameter(f"φ{num_modes+i}")
        VQE.add((i,), pcvl.PS(phi=param))
        List_Parameters.append(param)

    # CNOT (Post-selected with a success probability of 1/9)
    VQE.add(list(range(num_modes)), pcvl.PERM(list(range(num_modes))))

    for i in range(3, num_modes-2, 2):
        VQE.add((i, i+1), pcvl.BS())

    VQE.add(list(range(num_modes)), pcvl.PERM(list(range(num_modes))))

    for i in range(0, num_modes-1, 2):
        VQE.add((i, i+1), pcvl.BS(pcvl.BS.r_to_theta(1/3)))

    VQE.add(list(range(num_modes)), pcvl.PERM(list(range(num_modes))))

    for i in range(3, num_modes-2, 2):
        VQE.add((i, i+1), pcvl.BS())

    VQE.add(list(range(num_modes)), pcvl.PERM(list(range(num_modes))))
    
    # Second layer
    for i in range(2, num_modes, 2):
        param = pcvl.Parameter(f"φ{2*num_modes+i}")
        VQE.add((i,), pcvl.PS(phi=param))
        List_Parameters.append(param)

    for i in range(1, num_modes-1, 2):
        VQE.add((i,i+1), pcvl.BS())

    for i in range(2, num_modes, 2):
        param = pcvl.Parameter(f"φ{3*num_modes+i}")
        VQE.add((i,), pcvl.PS(phi=param))
        List_Parameters.append(param)

    for i in range(1, num_modes-1, 2):
        VQE.add((i,i+1), pcvl.BS())

    return VQE, List_Parameters

def linear_circuit_to_unitary(circuit: pcvl.Circuit) -> np.ndarray:
    unitary_matrix = np.array(circuit.compute_unitary())
    return unitary_matrix