#TODO: add some tests for the helper functions and hamiltonian functions

from qlass.helper_functions import compute_energy, get_probabilities, qubit_state_marginal, is_qubit_state
import perceval as pcvl

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import TwoLocal
from perceval.algorithm import Sampler
from scipy.optimize import minimize

from qlass import compile
from qlass.helper_functions import (
    loss_function,
)
from qlass.hamiltonians import LiH_hamiltonian

def test_compute_energy():
    # test case 1
    pauli_bin = (0, 0, 0)
    res = {(0, 0, 0): 0.5, (0, 0, 1): 0.3, (0, 1, 0): 0.1, (1, 0, 0): 0.1}
    assert compute_energy(pauli_bin, res) == float(1.0)

    # test case 2
    pauli_bin = (0, 0, 1)
    res = {(0, 0, 0): 0.5, (0, 0, 1): 0.3, (0, 1, 0): 0.1, (1, 0, 0): 0.1}
    assert compute_energy(pauli_bin, res) == float(0.4)

    # test case 3
    pauli_bin = (0, 1, 0)
    res = {(0, 0, 0): 0.5, (0, 0, 1): 0.3, (0, 1, 0): 0.1, (1, 0, 0): 0.1}
    assert compute_energy(pauli_bin, res) == float(0.8)

    # test case 4
    pauli_bin = (1, 0, 0)
    res = {(0, 0, 0): 0.45, (0, 0, 1): 0.23, (0, 1, 0): 0.1, (1, 0, 0): 0.32}
    assert compute_energy(pauli_bin, res) == float(0.46)

def test_get_probabilities():
    # test case 1
    samples = [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert get_probabilities(samples) == {(0, 0, 0): 0.4, (0, 0, 1): 0.4, (0, 1, 0): 0.2}

    # test case 2
    samples = [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
    assert get_probabilities(samples) == {(0, 0, 0): 0.5, (0, 0, 1): 0.3333333333333333, (0, 1, 0): 0.16666666666666666}

    # test case 3
    samples = [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0)]
    assert get_probabilities(samples) == {(0, 0, 0): 0.5714285714285714, (0, 0, 1): 0.2857142857142857, (0, 1, 0): 0.14285714285714285}

def test_qubit_state_marginal():
    # test case 1
    prob_dist = {pcvl.BasicState([0,0,0,0]): 0.4, pcvl.BasicState([0,1,0,1]): 0.3, pcvl.BasicState([1,0,0,1]): 0.3}
    assert qubit_state_marginal(prob_dist) == {(1, 1): 0.5, (0, 1): 0.5}

    # test case 2
    prob_dist = {pcvl.BasicState([0,1,0,1]): 0.4, pcvl.BasicState([0,1,1,0]): 0.3, pcvl.BasicState([1,0,0,1]): 0.3}
    assert qubit_state_marginal(prob_dist) == {(1, 1): 0.4, (1, 0): 0.3, (0, 1): 0.3}

    # test case 3
    prob_dist = {pcvl.BasicState([0,1,0,1,0,0]): 0.5, pcvl.BasicState([0,1,1,0,1,0]): 0.4, pcvl.BasicState([1,0,0,1,0,1]): 0.1}
    assert qubit_state_marginal(prob_dist) == {(1, 0, 0): 0.8, (0, 1, 1): 0.2}

def test_is_qubit_state():
    # test case 1
    state = pcvl.BasicState([0,1,0,1])
    assert is_qubit_state(state) == (1, 1)

    # test case 2
    state = pcvl.BasicState([1,0,0,1])
    assert is_qubit_state(state) == (0, 1)

    # test case 3
    state = pcvl.BasicState([1,0,1,0])
    assert is_qubit_state(state) == (0, 0)

    # test case 4
    state = pcvl.BasicState([0,1,1,0])
    assert is_qubit_state(state) == (1, 0)

    # test case 5
    state = pcvl.BasicState([1,1,0,1])
    assert is_qubit_state(state) == False

    # test case 6
    state = pcvl.BasicState([0,1,1,1])
    assert is_qubit_state(state) == False

    # test case 7
    state = pcvl.BasicState([1,1,1,1])
    assert is_qubit_state(state) == False

    # test case 8
    state = pcvl.BasicState([0,0,0,1])
    assert is_qubit_state(state) == False

def test_vqe_pipeline():

    # Define an executor function
    def executor(params, pauli_string, num_qubits=2):
        """
        Execute a quantum circuit with given parameters and measure in the basis specified by the Pauli string.
        
        Args:
            params (np.ndarray): Parameters for the variational circuit
            pauli_string (str): String representation of Pauli operators (e.g., "IXYZ")
            num_qubits (int): Number of qubits in the circuit
        
        Returns:
            dict: Sampling results
        """
        # Create a parameterized quantum circuit (ansatz)
        ansatz = TwoLocal(num_qubits, 'ry', 'cx', reps=1)
        
        # Assign parameters
        bound_circuit = ansatz.assign_parameters(params)
        
        # Apply measurement basis rotations based on the Pauli string
        rotated_circuit = QuantumCircuit(num_qubits)
        rotated_circuit.compose(bound_circuit, inplace=True)
        
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                rotated_circuit.h(i)
            elif pauli == 'Y':
                rotated_circuit.rx(np.pi/2, i)
        
        transpiled_circuit = transpile(rotated_circuit, basis_gates=['u3', 'cx'], optimization_level=3)
        # Convert to Perceval processor using our compile function
        processor = compile(transpiled_circuit)
        
        # Sample from the processor
        sampler = Sampler(processor)
        samples = sampler.samples(10_000)
        
        return samples
    
    # Number of qubits
    num_qubits = 2
    
    # Generate a 2-qubit Hamiltonian
    hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)

    # Initial random parameters for the variational circuit
    initial_params = np.random.rand(2 * num_qubits)
    
    # Run the VQE optimization
    result = minimize(
        loss_function,
        initial_params,
        args=(hamiltonian, executor),
        method='COBYLA',
        options={'maxiter': 3}
    )

    if not isinstance(result.fun, float):
        raise ValueError("Optimization result is not a valid float")
    