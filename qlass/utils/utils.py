import numpy as np
import perceval as pcvl
import qiskit
import exqalibur
from typing import List, Tuple, Dict, Union, Callable


H_matrix = (1/np.sqrt(2)) * pcvl.Matrix([[1.0, 1.0], [1.0, -1.0]])
M_matrix = (1/np.sqrt(2)) * pcvl.Matrix([[1.0, 1.0], [1.0j, -1.0j]])
mzi = (pcvl.BS() // (0, pcvl.PS(pcvl.Parameter("phi1"))) // 
        pcvl.BS() // (0, pcvl.PS(pcvl.Parameter("phi2"))))
H_circ = pcvl.Circuit.decomposition(H_matrix, mzi, shape=pcvl.InterferometerShape.TRIANGLE)
M_circ = pcvl.Circuit.decomposition(M_matrix, mzi, shape=pcvl.InterferometerShape.TRIANGLE)


def is_qubit_state(state: exqalibur.FockState) -> Union[Tuple[int, ...], bool]:
    """
    Check if a given Fock state is a valid qubit state.

    Args:
        state (exqalibur.FockState): The Fock state to check

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

def qubit_state_marginal(prob_dist: Dict[Union[exqalibur.FockState, Tuple[int, ...]], float]) -> Dict[Tuple[int, ...], float]:
    """
    Calculate the frequencies of measured qubit states from a probability distribution.
    
    This function now handles both Fock states and bitstring inputs:
    - If input contains Fock states, converts them to qubit states using is_qubit_state
    - If input already contains bitstrings (tuples), passes them through directly

    Args:
        prob_dist (Dict[Union[exqalibur.FockState, Tuple[int, ...]], float]): 
                  Probability distribution of either Fock states or bitstrings

    Returns:
        Dict[Tuple[int, ...], float]: Frequencies of measured qubit states
    """
    q_state_frequency = {}
    total_prob_mass = 0
    
    # Check if we're dealing with Fock states or bitstrings
    if not prob_dist:
        return q_state_frequency
        
    first_key = next(iter(prob_dist.keys()))
    
    if isinstance(first_key, tuple):
        # Input is already bitstrings, normalize probabilities and return
        total_prob_mass = sum(prob_dist.values())
        for bitstring, prob in prob_dist.items():
            q_state_frequency[bitstring] = prob / total_prob_mass
    else:
        # Input is Fock states, convert to qubit states
        for state in prob_dist:
            q_state = is_qubit_state(state)
            if q_state is not False:
                total_prob_mass += prob_dist[state]
                if q_state in q_state_frequency:
                    q_state_frequency[q_state] += prob_dist[state]
                else:
                    q_state_frequency[q_state] = prob_dist[state]
        
        # Normalize probabilities
        if total_prob_mass > 0:
            for key in q_state_frequency.keys():
                q_state_frequency[key] /= total_prob_mass
    
    return q_state_frequency

def get_probabilities(samples: List[Union[exqalibur.FockState, Tuple[int, ...], str]]) -> Dict[Union[exqalibur.FockState, Tuple[int, ...]], float]:
    """
    Get the probabilities of sampled states.
    
    This function now handles:
    - Fock states: List[exqalibur.FockState] -> Dict[exqalibur.FockState, float]
    - Bitstring tuples: List[Tuple[int, ...]] -> Dict[Tuple[int, ...], float]  
    - Bitstring strings (from Qiskit): List[str] -> Dict[Tuple[int, ...], float]

    Args:
        samples (List[Union[exqalibur.FockState, Tuple[int, ...], str]]): 
                Sampled states (Fock states, bitstring tuples, or bitstring strings)

    Returns:
        Dict[Union[exqalibur.FockState, Tuple[int, ...]], float]: 
        Probabilities of sampled states
    """
    if not samples:
        return {}
        
    prob_dist = {}
    for state in samples:
        # Convert string bitstrings (from Qiskit) to tuples
        if isinstance(state, str):
            state = tuple(int(bit) for bit in state)
            
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
    if not res:
        return 0.0
        
    # Create a copy to avoid modifying the original dictionary
    res_copy = res.copy()
    
    for key in res_copy.keys():
        inner = np.dot(key, pauli_bin)
        sign = (-1)**inner
        res_copy[key] *= sign
    
    energy = float(np.sum(np.fromiter(res_copy.values(), dtype=float), where=np.isfinite))
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


def normalize_samples(samples) -> List[Union[exqalibur.FockState, Tuple[int, ...]]]:
    """
    Normalize samples from different executor formats to a consistent format.
    
    Handles:
    - Qiskit bitstring format: ['00', '01', '11'] -> [(0,0), (0,1), (1,1)]
    - Already normalized tuples: [(0,0), (0,1)] -> [(0,0), (0,1)]
    - ExQalibur FockStates: [FockState, ...] -> [FockState, ...]
    
    Args:
        samples: Raw samples in various formats
        
    Returns:
        List[Union[exqalibur.FockState, Tuple[int, ...]]]: Normalized samples
    """
    if not samples:
        return []
        
    normalized = []
    for sample in samples:
        if isinstance(sample, str):
            # Convert Qiskit bitstring format '00' -> (0,0)
            normalized.append(tuple(int(bit) for bit in sample))
        elif isinstance(sample, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in sample):
            # Convert list/tuple of ints to tuple
            normalized.append(tuple(sample))
        else:
            # Assume it's already in correct format (e.g., exqalibur.FockState)
            normalized.append(sample)
    
    return normalized


def loss_function(lp: np.ndarray, H: Dict[str, float], executor) -> float:
    """
    Compute the loss function for the VQE algorithm with automatic Pauli grouping.
    
    This function automatically groups commuting Pauli terms to reduce the number 
    of measurements required. The grouping is applied transparently without 
    changing the function interface, providing automatic optimization for VQE 
    algorithms.
    
    The function works with executors that return either:
    1. Fock states (from linear optical circuits) - exqalibur.FockState objects
    2. Bitstring tuples (from regular qubit-based circuits)  
    3. Bitstring strings (from Qiskit Sampler)
    
    The executor should return samples in one of these formats:
    - Dict with 'results' key: {'results': [samples]}
    - Direct list of samples: [samples]
    - Qiskit-style format with bitstrings or counts

    Args:
        lp (np.ndarray): Array of parameter values
        H (Dict[str, float]): Hamiltonian dictionary
        executor: A callable function that executes the quantum circuit.

    Returns:
        float: The computed loss value
    """
    # Import here to avoid circular imports
    try:
        from qlass.quantum_chemistry.hamiltonians import group_commuting_pauli_terms
        use_grouping = True
    except ImportError:
        # Fallback to individual processing if grouping not available
        use_grouping = False
    
    loss = 0.0
    
    if use_grouping:
        # Use automatic grouping for optimized measurements
        grouped_hamiltonians = group_commuting_pauli_terms(H)
        
        for group in grouped_hamiltonians:
            # Each group contains mutually commuting terms
            # In the future, this could be optimized to measure entire groups simultaneously
            # For now, we process each term individually but with the grouping organization
            for pauli_string, coefficient in group.items():
                samples = executor(lp, pauli_string)
                
                # Handle different executor return formats
                sample_list = _extract_samples_from_executor_result(samples)
                
                # Normalize samples to consistent format
                normalized_samples = normalize_samples(sample_list)
                
                prob_dist = get_probabilities(normalized_samples)
                pauli_bin = pauli_string_bin(pauli_string)

                qubit_state_marg = qubit_state_marginal(prob_dist)
                expectation = compute_energy(pauli_bin, qubit_state_marg)
                loss += coefficient * expectation
    else:
        # Fallback to original implementation without grouping
        for pauli_string, coefficient in H.items():
            samples = executor(lp, pauli_string)
            
            # Handle different executor return formats
            sample_list = _extract_samples_from_executor_result(samples)
            
            # Normalize samples to consistent format
            normalized_samples = normalize_samples(sample_list)
            
            prob_dist = get_probabilities(normalized_samples)
            pauli_bin = pauli_string_bin(pauli_string)

            qubit_state_marg = qubit_state_marginal(prob_dist)
            expectation = compute_energy(pauli_bin, qubit_state_marg)
            loss += coefficient * expectation
    
    return loss.real


def _extract_samples_from_executor_result(samples):
    """
    Helper function to extract sample list from different executor return formats.
    
    Args:
        samples: Raw samples from executor in various formats
        
    Returns:
        List: Extracted sample list
        
    Raises:
        ValueError: If samples format is not recognized
    """
    sample_list = None
    
    if isinstance(samples, dict):
        if 'results' in samples:
            sample_list = samples['results']
        elif 'counts' in samples:
            # Handle Qiskit counts format: {'00': 500, '11': 500}
            sample_list = []
            for bitstring, count in samples['counts'].items():
                sample_list.extend([bitstring] * count)
        else:
            # Try to find any list-like values in the dict
            for key, value in samples.items():
                if isinstance(value, (list, tuple)):
                    sample_list = value
                    break
                    
    elif isinstance(samples, (list, tuple)):
        # Direct list of samples
        sample_list = samples
    else:
        raise ValueError(f"Executor returned unexpected format: {type(samples)}. "
                       "Expected dict with 'results' key, dict with 'counts' key, or list of samples.")

    if sample_list is None:
        raise ValueError("Could not extract sample list from executor return value.")
        
    return sample_list


def linear_circuit_to_unitary(circuit: pcvl.Circuit) -> np.ndarray:
    """
    Convert a linear optical circuit to a unitary matrix.

    Args:
        circuit (pcvl.Circuit): Linear optical circuit

    Returns:    
        np.ndarray: Unitary matrix representation of the circuit
    """

    unitary_matrix = np.array(circuit.compute_unitary())

    return unitary_matrix

def compute_expectation_value_from_unitary(
    unitary: np.ndarray, 
    pauli_matrix: np.ndarray,
    initial_state: np.ndarray = None
) -> float:
    """
    Compute expectation value <ψ|H|ψ> where |ψ> = U|0>.
    
    Args:
        unitary (np.ndarray): Unitary matrix representing the circuit
        pauli_matrix (np.ndarray): Matrix representation of Pauli operator
        initial_state (np.ndarray): Initial state vector (default: |0...0>)
    
    Returns:
        float: Expectation value
    """
    n_qubits = int(np.log2(unitary.shape[0]))
    
    if initial_state is None:
        # Default to |0...0> state
        initial_state = np.zeros(2**n_qubits, dtype=complex)
        initial_state[0] = 1.0
    
    # Compute |ψ> = U|0>
    state = unitary @ initial_state
    
    # Compute <ψ|H|ψ>
    expectation = np.real(state.conj() @ pauli_matrix @ state)
    
    return float(expectation.real)

def loss_function_matrix(
    params: np.ndarray,
    H: Dict[str, float],
    unitary_executor: Callable
) -> float:
    """
    Compute loss function using unitary matrices directly.
    
    Args:
        params (np.ndarray): Variational parameters
        H (Dict[str, float]): Hamiltonian dictionary
        unitary_executor: Function that returns unitary matrix given params
        
    Returns:
        float: Energy expectation value
    """
    from qlass.quantum_chemistry import pauli_string_to_matrix

    # Get the unitary from the executor
    unitary = unitary_executor(params)
    
    loss = 0.0
    for pauli_string, coefficient in H.items():
        # Convert Pauli string to matrix
        pauli_matrix = pauli_string_to_matrix(pauli_string)
        
        # Compute expectation value
        expectation = compute_expectation_value_from_unitary(
            unitary, 
            pauli_matrix
        )
        
        loss += coefficient * expectation
    
    return float(np.real(loss))

def permanent(matrix: np.ndarray) -> complex:
    """
    Calculate the permanent of a matrix.
    
    The permanent is like a determinant but without alternating signs:
    Perm(A) = Σ_{σ∈Sₘ} Π_{i=1 to m} A_{i,σ(i)}
    
    Parameters:
    -----------
    matrix : np.ndarray
        Square matrix to calculate permanent of
        
    Returns:
    --------
    complex
        The permanent of the matrix
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Permanent is only defined for square matrices.")
    if n == 0:
        return 1
    # Ryser's formula for the permanent
    rows = np.arange(n)
    result = 0
    for subset in range(1, 1 << n):
        S = [(subset >> j) & 1 for j in range(n)]
        parity = (-1) ** (n - sum(S))
        prod = 1
        for i in rows:
            s = 0
            for j in range(n):
                if S[j]:
                    s += matrix[i, j]
            prod *= s
        result += parity * prod
    return result


def logical_state_to_modes(logical_state: int, m: int) -> List[int]:
    """
    Convert a logical qubit state to the set of occupied photon modes.
    
    Parameters:
    -----------
    logical_state : int
        Integer representing the logical state (0 to 2^m - 1)
    m : int
        Number of qubits
        
    Returns:
    --------
    List[int]
        List of occupied mode indices (0-indexed)
    """
    # Convert integer to bit list
    bits = [(logical_state >> (m - 1 - k)) & 1 for k in range(m)]
    
    # For qubit k (0-indexed), the modes are 2k and 2k+1
    # |0⟩ₖ → mode 2k, |1⟩ₖ → mode 2k+1
    modes = []
    for k in range(m):
        mode = 2 * k + bits[k]
        modes.append(mode)
    
    return modes


def photon_to_qubit_unitary(U_photon: np.ndarray) -> np.ndarray:
    """
    Convert a photon unitary to the effective qubit unitary via post-selection.
    
    Parameters:
    -----------
    U_photon : np.ndarray
        The 2m × 2m unitary matrix acting on photon modes
        
    Returns:
    --------
    np.ndarray
        The 2^m × 2^m effective qubit unitary matrix
    """
    # Determine number of qubits
    modes_2m = U_photon.shape[0]
    if modes_2m % 2 != 0:
        raise ValueError("Photon unitary must have even dimension (2m × 2m)")
    
    m = modes_2m // 2
    num_logical_states = 2 ** m
    
    # Initialize the qubit unitary
    U_qubit = np.zeros((num_logical_states, num_logical_states), dtype=complex)
    
    # For each pair of input and output logical states
    for r in range(num_logical_states):
        for s in range(num_logical_states):
            # Get the mode sets for input state |r⟩ and output state |s⟩
            R = logical_state_to_modes(r, m)
            S = logical_state_to_modes(s, m)
            
            # Extract the submatrix U(S,R)
            # Rows indexed by S, columns indexed by R
            submatrix = U_photon[np.ix_(S, R)]
            
            # The matrix element is the permanent of this submatrix
            U_qubit[s, r] = permanent(submatrix)
    
    return U_qubit


def loss_function_photonic_unitary(
    params: np.ndarray,
    H: Dict[str, float],
    photonic_unitary_executor: Callable,
    initial_state: np.ndarray = None
) -> float:
    """
    Computes the loss function for a photonic VQE using the efficient, matrix-free
    state vector approach for post-selection.

    This version accepts a full state vector for the initial state, allowing for
    superposition states.

    Args:
        params: Variational parameters for the ansatz.
        H: Hamiltonian dictionary.
        photonic_unitary_executor: A function that takes `params` and returns the
                                   2m x 2m photonic unitary `U`.
        initial_state: The initial qubit state as a 2^m dimensional numpy vector.
                       If None, defaults to the |00...0> state.

    Returns:
        The computed energy expectation value.
    """
    from qlass.quantum_chemistry import pauli_string_to_matrix

    # Step 1: Get the physical unitary from the executor
    U_photon = photonic_unitary_executor(params)
    m = U_photon.shape[0] // 2
    dim_logical = 2**m

    # Handle the default initial state if None is provided
    if initial_state is None:
        initial_state = np.zeros(dim_logical, dtype=complex)
        initial_state[0] = 1.0  # Default to |0...0>

    # Step 2: Compute the unnormalized post-selected state vector U'|ψ_in>
    # Since |ψ_in> = Σ_r c_r |r>, the output is U'|ψ_in> = Σ_r c_r (U'|r>)
    psi_out_unnormalized = np.zeros(dim_logical, dtype=complex)

    # Iterate through each basis state |r> in the initial superposition
    for r_idx, c_r in enumerate(initial_state):
        # Skip basis states with negligible amplitude
        if abs(c_r) < 1e-15:
            continue

        # Get the input modes for the current basis state |r>
        R_modes = logical_state_to_modes(r_idx, m)
        
        # Calculate the contribution vector v_r = U'|r> and add its effect
        # The j-th component of v_r is <j|U'|r> = permanent(U(S_j, R))
        for s_idx in range(dim_logical):
            S_modes = logical_state_to_modes(s_idx, m)
            submatrix = U_photon[np.ix_(S_modes, R_modes)]
            permanent_val = permanent(submatrix)
            
            # Add the weighted contribution to the final state vector
            psi_out_unnormalized[s_idx] += c_r * permanent_val

    # Step 3: Calculate success probability
    success_prob = np.vdot(psi_out_unnormalized, psi_out_unnormalized).real

    if success_prob < 1e-15:
        # Return a large penalty value if the state is unreachable
        # to guide the optimizer away from this parameter region.
        return 1e6

    # Step 4: Compute the energy expectation <v|H|v> / <v|v>
    # where |v> = psi_out_unnormalized
    
    # Calculate the numerator term <v|H|v>
    numerator_energy = 0.0
    for pauli_string, coeff in H.items():
        if abs(coeff) < 1e-15:
            continue
        # Get the matrix for the current Pauli term
        pauli_matrix = pauli_string_to_matrix(pauli_string)
        
        # Calculate <v|P|v> efficiently
        term_expectation = np.vdot(psi_out_unnormalized, pauli_matrix @ psi_out_unnormalized)
        numerator_energy += coeff * term_expectation

    # Final energy is numerator / denominator (success_prob)
    final_energy = numerator_energy / success_prob

    return float(np.real(final_energy))