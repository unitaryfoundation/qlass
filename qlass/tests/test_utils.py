from qlass.utils import (
    compute_energy, 
    get_probabilities, 
    qubit_state_marginal, 
    is_qubit_state, 
    loss_function,
    compute_expectation_value_from_unitary,
    loss_function_matrix,
)
from qlass.quantum_chemistry import pauli_string_to_matrix

import perceval as pcvl
import numpy as np
import pytest

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

    energy, contributions = loss_function(test_params, simple_ham, mock_executor)
    assert isinstance(energy, float), "loss_function should return a float energy"
    assert isinstance(contributions, dict), "loss_function should return a contrib dict"
    
    empty_ham = {}
    energy_empty, contrib_empty = loss_function(test_params, empty_ham, mock_executor)
    assert energy_empty == 0.0, "Empty Hamiltonian should give zero loss"
    assert contrib_empty == {}, "Empty Hamiltonian should give empty contribs"

    single_ham = {"ZZ": 1.0}
    energy_single, contrib_single = loss_function(test_params, single_ham, mock_executor)
    assert isinstance(energy_single, float), "Single term should work correctly"
    assert "ZZ" in contrib_single


def test_loss_function_perceval_format():
    
    # Mock Perceval-style executor
    def mock_perceval_executor(params, pauli_string):
        return {
            'results': [
                pcvl.BasicState([1, 0, 1, 0]),  # |00⟩ 
                pcvl.BasicState([0, 1, 0, 1]),  # |11⟩
                pcvl.BasicState([1, 0, 0, 1]),  # |01⟩
            ]
        }
    
    hamiltonian = {"II": 0.5, "ZZ": 0.3}
    energy, contributions = loss_function(np.array([0.1, 0.2]), hamiltonian, mock_perceval_executor)
    assert isinstance(energy, float)
    assert isinstance(contributions, dict)
    # Manual check: States are |00⟩, |11⟩, |01⟩. All 1/3 prob.
    # II: 0.5 * 1.0 = 0.5
    # ZZ: 0.3 * ( (1/3)*(+1) + (1/3)*(+1) + (1/3)*(-1) ) = 0.3 * (1/3) = 0.1
    # Expected: 0.5 + 0.1 = 0.6
    assert np.isclose(energy, 0.6)

def test_loss_function_qiskit_bitstring_format():
    
    # Mock Qiskit bitstring executor
    def mock_qiskit_executor(params, pauli_string):
        return {'results': ['00', '01', '10', '11']}
    
    hamiltonian = {"II": 1.0, "ZZ": 0.5}
    energy, contributions = loss_function(np.array([0.1, 0.2]), hamiltonian, mock_qiskit_executor)
    assert isinstance(energy, float)
    # Manual check: States are |00⟩, |01⟩, |10⟩, |11⟩. All 1/4 prob.
    # II: 1.0 * 1.0 = 1.0
    # ZZ: 0.5 * ( (1/4)*(+1) + (1/4)*(-1) + (1/4)*(-1) + (1/4)*(+1) ) = 0.5 * 0 = 0
    # Expected: 1.0 + 0 = 1.0
    assert np.isclose(energy, 1.0)

def test_loss_function_qiskit_counts_format():
    
    # Mock Qiskit counts executor
    def mock_counts_executor(params, pauli_string):
        return {'counts': {'00': 250, '01': 250, '10': 250, '11': 250}}
    
    hamiltonian = {"II": 1.0, "ZI": 0.2}
    energy, contributions = loss_function(np.array([0.1, 0.2]), hamiltonian, mock_counts_executor)
    assert isinstance(energy, float)
    # Manual check: All states 1/4 prob.
    # II: 1.0 * 1.0 = 1.0
    # ZI: 0.2 * ( (1/4)*(+1) + (1/4)*(+1) + (1/4)*(-1) + (1/4)*(-1) ) = 0.2 * 0 = 0
    # Expected: 1.0 + 0 = 1.0
    assert np.isclose(energy, 1.0)

def test_loss_function_direct_list_format():
    
    # Mock direct list executor
    def mock_direct_executor(params, pauli_string):
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    hamiltonian = {"ZZ": 1.0, "XX": -0.5}
    energy, contributions = loss_function(np.array([0.1, 0.2]), hamiltonian, mock_direct_executor)
    assert isinstance(energy, float)
    # Manual check: All states 1/4 prob.
    # ZZ: 1.0 * ( (1/4)*(+1) + (1/4)*(-1) + (1/4)*(-1) + (1/4)*(+1) ) = 1.0 * 0 = 0
    # XX: -0.5 * ( (1/4)*(+1) + (1/4)*(-1) + (1/4)*(-1) + (1/4)*(+1) ) = -0.5 * 0 = 0
    # Expected: 0 + 0 = 0.0
    assert np.isclose(energy, 0.0)

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
    
    result1, _ = loss_function(params, hamiltonian, executor1)
    result2, _ = loss_function(params, hamiltonian, executor2)
    result3, _ = loss_function(params, hamiltonian, executor3)
    
    # Allow small numerical differences
    tolerance = 1e-10
    assert abs(result1 - result2) < tolerance
    assert abs(result1 - result3) < tolerance

def test_get_probabilities_string_format():
    # test case 1: Qiskit string format
    samples = ['00', '01', '00', '10', '01']
    expected = {(0, 0): 0.4, (0, 1): 0.4, (1, 0): 0.2}
    assert get_probabilities(samples) == expected

    # test case 2: single qubit strings  
    samples = ['0', '1', '0', '0']
    expected = {(0,): 0.75, (1,): 0.25}
    assert get_probabilities(samples) == expected

    # test case 3: 3-qubit strings
    samples = ['000', '001', '010', '000'] 
    expected = {(0, 0, 0): 0.5, (0, 0, 1): 0.25, (0, 1, 0): 0.25}
    assert get_probabilities(samples) == expected

def test_qubit_state_marginal_bitstring_input():
    # test case 1: input already as bitstring tuples
    prob_dist = {(0, 0): 0.5, (0, 1): 0.3, (1, 0): 0.2}
    expected = {(0, 0): 0.5, (0, 1): 0.3, (1, 0): 0.2}
    assert qubit_state_marginal(prob_dist) == expected

    # test case 2: single entry
    prob_dist = {(1, 1, 0): 1.0}
    expected = {(1, 1, 0): 1.0}
    assert qubit_state_marginal(prob_dist) == expected

    # test case 3: empty input
    assert qubit_state_marginal({}) == {}

def test_loss_function_fallback_without_grouping(mocker):
    """
    Tests the loss_function's fallback to individual term processing
    when the grouping utility is not available.
    """
    # 1. Mock the grouping function to trigger an ImportError
    mocker.patch(
        'qlass.quantum_chemistry.group_commuting_pauli_terms',
        side_effect=ImportError("Simulating grouping utility not found")
    )

    # 2. Define a simple mock executor that returns a consistent result
    def mock_executor(params, pauli_string):
        # Always returns a sample of |01>
        return {'counts': {'01': 1000}}

    # 3. Define a simple Hamiltonian
    hamiltonian = {"ZI": 0.5, "IZ": -0.5}
    params = np.array([0.1, 0.2]) # Dummy parameters

    # 4. Manually calculate the expected energy for the |01> state
    # For "ZI" (pauli_bin=(1,0)), sample=(0,1): inner dot = 0, sign = (-1)^0 = 1. Expectation = 1.0
    # For "IZ" (pauli_bin=(0,1)), sample=(0,1): inner dot = 1, sign = (-1)^1 = -1. Expectation = -1.0
    # Total expected loss = (0.5 * 1.0) + (-0.5 * -1.0) = 0.5 + 0.5 = 1.0
    expected_loss = 1.0

    # 5. Run the loss function and assert the result
    calculated_loss, contributions = loss_function(params, hamiltonian, mock_executor)

    assert np.isclose(calculated_loss, expected_loss)
    assert isinstance(contributions, dict)
    assert np.isclose(contributions["ZI"], 0.5)
    assert np.isclose(contributions["IZ"], 0.5)

def test_compute_expectation_value_from_unitary_identity():
    """Test expectation value computation with identity unitary and Pauli Z."""
    # For U = I and H = Z, <0|Z|0> = 1
    unitary = np.eye(2, dtype=complex)
    pauli_z = pauli_string_to_matrix("Z")
    
    expectation = compute_expectation_value_from_unitary(unitary, pauli_z)
    assert np.isclose(expectation, 1.0)

def test_compute_expectation_value_from_unitary_hadamard():
    """Test expectation value with Hadamard gate."""
    # H|0> = |+>, and <+|Z|+> = 0
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    pauli_z = pauli_string_to_matrix("Z")
    
    expectation = compute_expectation_value_from_unitary(H, pauli_z)
    assert np.isclose(expectation, 0.0, atol=1e-10)

def test_loss_function_matrix_multi_term_hamiltonian():
    """Test matrix-based loss function with multiple terms."""
    def identity_executor(params):
        return np.eye(2, dtype=complex)
    
    hamiltonian = {
        "I": 0.5,   # Identity term
        "Z": 1.0,   # <0|Z|0> = 1
        "X": -0.5   # <0|X|0> = 0
    }
    params = np.array([0.1])
    
    loss, contributions = loss_function_matrix(params, hamiltonian, identity_executor)
    
    assert np.isclose(loss, 1.5)
    assert isinstance(contributions, dict)
    assert np.isclose(contributions["I"], 0.5)
    assert np.isclose(contributions["Z"], 1.0)
    assert np.isclose(contributions["X"], 0.0)

def test_loss_function_matrix_two_qubit():
    """Test matrix-based loss function for 2-qubit system."""
    def identity_executor(params):
        return np.eye(4, dtype=complex)
    
    hamiltonian = {
        "II": 1.0,
        "ZZ": 0.5,
        "XX": -0.3
    }
    params = np.array([0.1, 0.2])
    
    loss, contributions = loss_function_matrix(params, hamiltonian, identity_executor)
    
    assert np.isclose(loss, 1.5)
    assert isinstance(contributions, dict)
    assert np.isclose(contributions["II"], 1.0)
    assert np.isclose(contributions["ZZ"], 0.5)
    assert np.isclose(contributions["XX"], 0.0)

def test_loss_function_matrix_parameterized_unitary():
    """Test with parameterized unitary (rotation)."""
    def rotation_executor(params):
        # Rotation around Y axis: Ry(theta)
        theta = params[0]
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    hamiltonian = {"Z": 1.0}
    
    loss_0, contrib_0 = loss_function_matrix(np.array([0.0]), hamiltonian, rotation_executor)
    assert np.isclose(loss_0, 1.0)
    assert np.isclose(contrib_0["Z"], 1.0)

    loss_pi, contrib_pi = loss_function_matrix(np.array([np.pi]), hamiltonian, rotation_executor)
    assert np.isclose(loss_pi, -1.0)
    assert np.isclose(contrib_pi["Z"], -1.0)

    loss_half, contrib_half = loss_function_matrix(
        np.array([np.pi/2]), hamiltonian, rotation_executor
    )
    assert np.isclose(loss_half, 0.0, atol=1e-10)
    assert np.isclose(contrib_half["Z"], 0.0, atol=1e-10)

def test_permanent():
    """Test permanent calculation for key cases."""
    from qlass.utils import permanent
    
    # Identity matrix
    I = np.eye(2, dtype=complex)
    assert np.isclose(permanent(I), 1.0)
    
    # Simple 2×2 matrix: permanent = 1*4 + 2*3 = 10
    A = np.array([[1, 2], [3, 4]], dtype=complex)
    assert np.isclose(permanent(A), 10.0)
    
    # Empty matrix
    E = np.zeros((0, 0), dtype=complex)
    assert np.isclose(permanent(E), 1.0)


def test_logical_state_to_modes():
    """Test logical state to modes conversion."""
    from qlass.utils import logical_state_to_modes
    
    # Single qubit
    assert logical_state_to_modes(0, 1) == [0]  # |0⟩ → mode 0
    assert logical_state_to_modes(1, 1) == [1]  # |1⟩ → mode 1
    
    # Two qubits
    assert logical_state_to_modes(0, 2) == [0, 2]  # |00⟩ → modes [0, 2]
    assert logical_state_to_modes(3, 2) == [1, 3]  # |11⟩ → modes [1, 3]


def test_photon_to_qubit_unitary():
    """Test photon to qubit unitary conversion."""
    from qlass.utils import photon_to_qubit_unitary
    
    # Identity photonic unitary should give identity qubit unitary
    U_photon = np.eye(4, dtype=complex)  # 2 qubits, 4 modes
    U_qubit = photon_to_qubit_unitary(U_photon)
    assert np.allclose(U_qubit, np.eye(4))
    
    # Mode swap for 1 qubit should give X gate
    U_photon_swap = np.array([[0, 1], [1, 0]], dtype=complex)
    U_qubit_swap = photon_to_qubit_unitary(U_photon_swap)
    expected_X = np.array([[0, 1], [1, 0]], dtype=complex)
    assert np.allclose(U_qubit_swap, expected_X)
    
    # Invalid dimension should raise error
    with pytest.raises(ValueError, match="even dimension"):
        photon_to_qubit_unitary(np.eye(3))


def test_loss_function_photonic_unitary():
    """Test photonic unitary loss function, including ancillary post-selection."""
    from qlass.utils import loss_function_photonic_unitary
    # --- (No Ancillas) ---
    def identity_executor_4x4(params):
        return np.eye(4, dtype=complex)
    
    hamiltonian = {"II": 0.5, "ZZ": 1.0}
    params = np.array([0.1, 0.2])
    
    loss, contributions = loss_function_photonic_unitary(params, hamiltonian, identity_executor_4x4)
    
    assert np.isclose(loss, 1.5), "Test failed for default state, no ancillas"
    assert isinstance(contributions, dict)
    assert np.isclose(contributions["II"], 0.5)
    assert np.isclose(contributions["ZZ"], 1.0)

    # --- (Custom Initial State, No Ancillas) ---
    initial_state_11 = np.array([0, 0, 0, 1], dtype=complex) # Logical |11⟩
    hamiltonian_z = {"ZZ": 1.0}
    loss_11, contrib_11 = loss_function_photonic_unitary(
        params, hamiltonian_z, identity_executor_4x4, initial_state_11
    )
    assert np.isclose(loss_11, 1.0), "Test failed for |11⟩ state, no ancillas"
    assert np.isclose(contrib_11["ZZ"], 1.0)

    # --- With Ancillas, Identity logical sub-matrix ---
    def identity_executor_6x6(params):
        return np.eye(6, dtype=complex)
        
    anc_modes = [0, 5]
    
    loss_ancilla, contrib_ancilla = loss_function_photonic_unitary(
        params, hamiltonian, identity_executor_6x6, ancillary_modes=anc_modes
    )
    assert np.isclose(loss_ancilla, 1.5), "Test failed for default state with ancillas"
    assert np.isclose(contrib_ancilla["II"], 0.5)
    assert np.isclose(contrib_ancilla["ZZ"], 1.0)

    # --- With Ancillas, Custom Initial State ---
    loss_ancilla_11, contrib_ancilla_11 = loss_function_photonic_unitary(
        params, hamiltonian_z, identity_executor_6x6, initial_state_11, anc_modes
    )
    assert np.isclose(loss_ancilla_11, 1.0), "Test failed for |11⟩ state with ancillas"
    assert np.isclose(contrib_ancilla_11["ZZ"], 1.0)
    
    # --- Post-selection failure (leakage to ancilla) ---
    def swap_1_5_executor(params):
        U = np.eye(6, dtype=complex)
        U[1, 1] = 0; U[5, 5] = 0; U[1, 5] = 1; U[5, 1] = 1
        return U
        
    loss_fail, contrib_fail = loss_function_photonic_unitary(
        params, hamiltonian, swap_1_5_executor, ancillary_modes=anc_modes
    )
    assert np.isclose(loss_fail, 1e6), "Test failed to return penalty"
    assert contrib_fail == {}, "Contributions should be empty on failure"
    
    with pytest.raises(ValueError, match="contain indices outside"):
        loss_function_photonic_unitary(
            params, hamiltonian, identity_executor_4x4, ancillary_modes=[0, 10]
        )
        
    with pytest.raises(ValueError, match="must be even"):
        loss_function_photonic_unitary(
            params, hamiltonian, identity_executor_4x4, ancillary_modes=[0]
        )
        
    with pytest.raises(ValueError, match="Initial state dimension"):
        initial_state_1q = np.array([0, 1], dtype=complex)
        loss_function_photonic_unitary(
            params, hamiltonian, identity_executor_6x6, initial_state_1q, anc_modes
        )


def test_ensemble_weights():
    from qlass.utils import ensemble_weights

    equal_weights = ensemble_weights('equi', 2)
    assert isinstance(equal_weights, list)
    dec_weights = ensemble_weights('weighted', 2)
    assert isinstance(dec_weights, list)
    ground_state = ensemble_weights('ground_state_only', 2)
    assert isinstance(ground_state, list)

    with pytest.raises(ValueError, match=f"Invalid weights_choice. Must be one of 'equi', 'weighted', or 'ground_state_only'."):
        invalid_weight_string = ensemble_weights('Invalid_weight', 2)

