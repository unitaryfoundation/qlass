from qlass.utils.utils import (
    compute_energy, 
    get_probabilities, 
    qubit_state_marginal, 
    is_qubit_state, 
    loss_function,
)
import perceval as pcvl
import numpy as np

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

def test_loss_function_automatic_grouping():
    """
    Test that loss_function automatically uses Pauli grouping when available.
    This test verifies that the function can import and use the grouping functionality.
    """
    # Define a simple mock executor for testing
    def mock_executor(params, pauli_string):
        # Return a simple mock result that's consistent
        return {'results': [pcvl.BasicState([1,0,0,1]), pcvl.BasicState([0,1,1,0])]}
    
    # test case 1: simple 2-qubit hamiltonian with commuting terms
    simple_ham = {"II": 1.0, "ZI": 0.5, "IZ": -0.3, "ZZ": 0.2}
    test_params = np.array([0.1, 0.2])
    
    # Function should work with automatic grouping
    result = loss_function(test_params, simple_ham, mock_executor)
    assert isinstance(result, float), "loss_function should return a float"
    
    # test case 2: empty hamiltonian should work
    empty_ham = {}
    result_empty = loss_function(test_params, empty_ham, mock_executor)
    assert result_empty == 0.0, "Empty Hamiltonian should give zero loss"
    
    # test case 3: single term should work
    single_ham = {"ZZ": 1.0}
    result_single = loss_function(test_params, single_ham, mock_executor)
    assert isinstance(result_single, float), "Single term should work correctly"


def test_loss_function_perceval_format():
    
    # Mock Perceval-style executor
    def mock_perceval_executor(params, pauli_string):
        return {
            'results': [
                pcvl.BasicState([1, 0, 1, 0]),  # |01⟩ 
                pcvl.BasicState([0, 1, 0, 1]),  # |11⟩
                pcvl.BasicState([1, 0, 0, 1]),  # |00⟩
            ]
        }
    
    hamiltonian = {"II": 0.5, "ZZ": 0.3}
    result = loss_function(np.array([0.1, 0.2]), hamiltonian, mock_perceval_executor)
    assert isinstance(result, float)

def test_loss_function_qiskit_bitstring_format():
    
    # Mock Qiskit bitstring executor
    def mock_qiskit_executor(params, pauli_string):
        return {'results': ['00', '01', '10', '11']}
    
    hamiltonian = {"II": 1.0, "ZZ": 0.5}
    result = loss_function(np.array([0.1, 0.2]), hamiltonian, mock_qiskit_executor)
    assert isinstance(result, float)

def test_loss_function_qiskit_counts_format():
    
    # Mock Qiskit counts executor
    def mock_counts_executor(params, pauli_string):
        return {'counts': {'00': 250, '01': 250, '10': 250, '11': 250}}
    
    hamiltonian = {"II": 1.0, "ZI": 0.2}
    result = loss_function(np.array([0.1, 0.2]), hamiltonian, mock_counts_executor)
    assert isinstance(result, float)

def test_loss_function_direct_list_format():
    
    # Mock direct list executor
    def mock_direct_executor(params, pauli_string):
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    hamiltonian = {"ZZ": 1.0, "XX": -0.5}
    result = loss_function(np.array([0.1, 0.2]), hamiltonian, mock_direct_executor)
    assert isinstance(result, float)

def test_loss_function_error_handling():

    # Mock invalid executor
    def invalid_executor(params, pauli_string):
        return "invalid_format"
    
    hamiltonian = {"ZZ": 1.0}
    
    try:
        loss_function(np.array([0.1]), hamiltonian, invalid_executor)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "unexpected format" in str(e).lower()

def test_loss_function_format_consistency():

    # Fixed samples for consistent comparison
    fixed_samples = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    def executor1(params, pauli_string):
        return {'results': fixed_samples}
    
    def executor2(params, pauli_string):
        return {'results': ['00', '01', '10', '11']}
    
    def executor3(params, pauli_string):
        return fixed_samples
    
    hamiltonian = {"ZZ": 1.0, "XX": -0.5}
    params = np.array([0.1, 0.2])
    
    result1 = loss_function(params, hamiltonian, executor1)
    result2 = loss_function(params, hamiltonian, executor2)
    result3 = loss_function(params, hamiltonian, executor3)
    
    # Allow small numerical differences
    tolerance = 1e-10
    assert abs(result1 - result2) < tolerance
    assert abs(result1 - result3) < tolerance