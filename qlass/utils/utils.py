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

def e_vqe_loss_function(lp: np.ndarray, H: Dict[str, float], executor, energy_collector, weight_option: str= "weighted") -> float:
    """
    Compute the loss function for the ensemble Variational Quantum Eigensolver (VQE)
    with automatic Pauli grouping for measurement optimization.

    This function automatically groups commuting Pauli terms to reduce the number
    of measurements required. The grouping is applied transparently without
    changing the function interface, providing automatic optimization for ensemble VQE
    algorithms.

    The function works with executors that return either:
    1. Fock states (from linear optical circuits) - exqalibur.FockState objects
    2. Bitstring tuples (from regular qubit-based circuits)
    3. Bitstring strings (from Qiskit Sampler)

    The executor should return samples in one of these formats:
    - Dict with 'results' key: {'results': [samples]}
    - Direct list of samples: [samples]
    - Qiskit-style format with bitstrings or counts

    Parameters
    ----------
    lp : np.ndarray
        Array of variational circuit parameters. Typically optimized to minimize
        the expectation value of the Hamiltonian.
    H : dict of {str: float}
        Hamiltonian represented as a dictionary mapping Pauli strings
        (e.g., ``'X0 Z1'``) to their coefficients.
    executor : callable
        Function or callable object that executes the quantum circuit and returns
        measurement samples. Must accept arguments ``(lp, pauli_string)`` and return
        samples in one of the accepted formats.
    energy_collector : object
        Object responsible for tracking or logging the energy convergence history.
        Must implement a method ``energies_convergence(energies, n_ensembles, total_loss)``.
    weight_option : {'weighted', 'uniform'}
        Scheme for assigning ensemble weights:
        - ``'weighted'`` (default): Linearly decreasing weights with index, i.e., w_i < w_j for i > j.
        - ``'equi'`` : Equal weights for all occupied orbitals, w_i = w_j.
        - ``'ground_state_only'`` : Only the ground state contributes, w_0 = 1, others 0.

    Returns
    -------
    loss : float
        The computed ensemble VQE loss value, equal to the weighted sum of ensemble
        energies.
    """
    # Import here to avoid circular imports
    try:
        from qlass.quantum_chemistry.hamiltonians import group_commuting_pauli_terms
        use_grouping = True
    except ImportError:
        # Fallback to individual processing if grouping not available
        use_grouping = False

    loss = 0.0
    lst_energies = None

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
                sample_lists = [_extract_samples_from_executor_result(s) for s in samples]

                # Normalize samples to consistent format
                normalized_samples = [normalize_samples(sample_list) for sample_list in sample_lists]

                prob_dist = [get_probabilities(normalized_sample) for normalized_sample in normalized_samples]
                pauli_bin = pauli_string_bin(pauli_string)

                qubit_state_marg = [qubit_state_marginal(pd) for pd in prob_dist]
                expectation = [compute_energy(pauli_bin, qsm) for qsm in qubit_state_marg]
                energies = [coefficient * expect for expect in expectation]
                # Initialize accumulator on first iteration
                if lst_energies is None:
                    lst_energies = [0.0] * len(energies)

                # Accumulate energies dynamically
                for i, energy in enumerate(energies):
                    lst_energies[i] += energy
    else:
        # Fallback to original implementation without grouping
        for pauli_string, coefficient in H.items():
            samples = executor(lp, pauli_string)

            # Handle different executor return formats
            sample_lists = [_extract_samples_from_executor_result(s) for s in samples]

            # Normalize samples to consistent format
            normalized_samples = [normalize_samples(sample_list) for sample_list in sample_lists]

            prob_dist = [get_probabilities(normalized_sample) for normalized_sample in normalized_samples]
            pauli_bin = pauli_string_bin(pauli_string)

            qubit_state_marg = [qubit_state_marginal(pd) for pd in prob_dist]
            expectation = [compute_energy(pauli_bin, qsm) for qsm in qubit_state_marg]
            energies = [coefficient * expect for expect in expectation]
            # Initialize accumulator on first iteration
            if lst_energies is None:
                lst_energies = [0.0] * len(energies)

            # Accumulate energies dynamically
            for i, energy in enumerate(energies):
                lst_energies[i] += energy

    weights = ensemble_weights(weight_option, len(energies))
    for i in range(len(lst_energies)): loss += lst_energies[i] * weights[i]
    energy_collector.enegies_convergence(lst_energies, len(lst_energies), loss)

    return loss

def ensemble_weights(weights_choice, n_occ):
    """
     Generate ensemble weights for the ensemble Variational Quantum Eigensolver (VQE).

     Ensemble VQE uses weighted contributions from multiple eigenstates when
     computing the loss function. This function provides a choice of weighting
     schemes for the occupied orbitals.

     Parameters
     ----------
     weights_choice : {'weighted', 'equi', 'ground_state_only'}
         Scheme for assigning ensemble weights:
         - ``'weighted'`` : Linearly decreasing weights with index, i.e., w_i < w_j for i > j.
         - ``'equi'`` : Equal weights for all occupied orbitals, w_i = w_j.
         - ``'ground_state_only'`` : Only the ground state contributes, w_0 = 1, others 0.
     n_occ : int
         Number of occupied orbitals in the system. Determines the length of the weight vector.

     Returns
     -------
     weights : np.ndarray
         Array of length ``n_occ`` containing the ensemble weights corresponding
         to the chosen weighting scheme.

     Notes
     -----
     - See the original article for the derivation of ensemble weights:
       `arXiv:2509.17982 <https://doi.org/10.48550/arXiv.2509.17982>`_.

     Examples
     --------
     >>> ensemble_weights("equi", 4)
     array([0.25, 0.25, 0.25, 0.25])
     >>> ensemble_weights("weighted", 4)
     array([0.375, 0.25 , 0.125, 0.   ])
     >>> ensemble_weights("ground_state_only", 4)
     array([1., 0., 0., 0.])
     """
    if weights_choice == "equi":
        weights = [1. / n_occ for i in range(n_occ)]  # should be n_occ of them
    elif weights_choice == "weighted":
        weights = []
        for i in range(n_occ):
            weights.append(1/(n_occ**2) * (2 * n_occ - 1 - 2 * i))
    elif weights_choice == "ground_state_only":
        weights = [1.] + [0.0]*(n_occ - 1)

    return weights


class DataCollector:
    """
    Collects and stores energies and loss values during ensemble VQE optimization.

    This class is intended to track the evolution of the loss function and
    corresponding energies each time the loss function is evaluated. Unlike
    the default ``loss_history`` from SciPy's ``minimize``, the number of
    iterations recorded here may differ, because it logs every call to the
    loss function, not just accepted steps in the optimizer.

    Attributes
    ----------
    energy_data : dict of {int: list of float}
        Dictionary mapping the index of an orbital/eigenstate to a list of its
        observed energies over successive loss function evaluations.
    loss_data : list of float
        List of loss values recorded at each evaluation of the loss function.
    """
    def __init__(self):
        """
        Initialize the DataCollector with empty data structures.
        """
        self.energy_data = {}
        self.loss_data = []

    def enegies_convergence(self, energy_values, eign_index, loss_values):
        """
               Record energies and loss values for the current evaluation.

               Each entry in ``energy_values`` is appended to the corresponding orbital
               index in ``energy_data``. The total loss for this evaluation is appended
               to ``loss_data``.

               Parameters
               ----------
               energy_values : list of float
                   List of energies for each occupied orbital or eigenstate at the current
                   loss function evaluation.
               eign_index : int
                   Number of energies in ``energy_values`` to record (typically equal to the
                   number of occupied orbitals in the ensemble).
               loss_values : float
                   Scalar value of the loss function at the current evaluation.

               """
        for i in range(eign_index):
            if i not in self.energy_data:
                self.energy_data[i] = []  # initialize a list if key is new
            self.energy_data[i].append(energy_values[i])  # append instead of replacing

        self.loss_data.append(loss_values)
