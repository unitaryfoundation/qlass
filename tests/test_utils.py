import sys
from unittest.mock import MagicMock, patch

import exqalibur
import numpy as np
import perceval as pcvl
import pytest
from openfermion import BosonOperator
from openfermion.linalg import boson_operator_sparse

from qlass.quantum_chemistry import pauli_string_to_matrix
from qlass.utils import (
    compute_energy,
    compute_expectation_value_from_unitary,
    draw_circuit,
    get_probabilities,
    is_qubit_state,
    loss_function,
    loss_function_bose_hubbard,
    loss_function_matrix,
    qubit_state_marginal,
    rotate_modes,
    rotate_qubits,
)
from qlass.utils.utils import _extract_samples_from_executor_result
from qlass.vqe import le_ansatz


def test_compute_energy():
    # test case 1
    pauli_bin = (0, 0, 0)
    res = {(0, 0, 0): 0.5, (0, 0, 1): 0.3, (0, 1, 0): 0.1, (1, 0, 0): 0.1}
    assert compute_energy(pauli_bin, res) == 1.0

    # test case 2
    pauli_bin = (0, 0, 1)
    res = {(0, 0, 0): 0.5, (0, 0, 1): 0.3, (0, 1, 0): 0.1, (1, 0, 0): 0.1}
    assert compute_energy(pauli_bin, res) == 0.4

    # test case 3
    pauli_bin = (0, 1, 0)
    res = {(0, 0, 0): 0.5, (0, 0, 1): 0.3, (0, 1, 0): 0.1, (1, 0, 0): 0.1}
    assert compute_energy(pauli_bin, res) == 0.8

    # test case 4
    pauli_bin = (1, 0, 0)
    res = {(0, 0, 0): 0.45, (0, 0, 1): 0.23, (0, 1, 0): 0.1, (1, 0, 0): 0.32}
    assert compute_energy(pauli_bin, res) == 0.46


def test_rotate_modes_appends_beamsplitter():
    circuit = pcvl.Circuit(2)

    rotated = rotate_modes(circuit, 0, 1)

    assert rotated is circuit
    assert sum(1 for _ in circuit) == 1


def test_rotate_modes_rejects_invalid_modes():
    circuit = pcvl.Circuit(2)

    with pytest.raises(ValueError, match="distinct modes"):
        rotate_modes(circuit, 0, 0)

    with pytest.raises(ValueError, match="non-negative"):
        rotate_modes(circuit, -1, 0)


def test_loss_function_bose_hubbard_diagonal_matches_exact_matrix():
    hamiltonian = (
        BosonOperator("0^ 0^ 0 0", 0.5)
        + BosonOperator("1^ 1^ 1 1", 0.25)
        + BosonOperator("0^ 0", 1.3)
        + BosonOperator("1^ 1", 0.7)
        + BosonOperator("0^ 1^ 0 1", 0.2)
    )
    truncation = 3
    trial_state = np.zeros(truncation**2, dtype=complex)
    trial_state[2 * truncation + 1] = 1.0  # |n_0=2, n_1=1>
    exact_energy = np.vdot(
        trial_state, boson_operator_sparse(hamiltonian, truncation) @ trial_state
    ).real
    calls = []

    def executor(params, measurement):
        calls.append((measurement,))
        return {"results": [(2, 1)] * 10}

    sampled_energy = loss_function_bose_hubbard(np.array([0.0]), hamiltonian, executor)

    assert np.isclose(sampled_energy, exact_energy)
    assert calls == [("identity",)]


def test_loss_function_bose_hubbard_hopping_dimer_matches_exact_matrix():
    hopping_strength = -0.7
    hamiltonian = (
        BosonOperator("0^ 0", 0.4)
        + BosonOperator("1^ 1", 1.2)
        + BosonOperator("0^ 1", hopping_strength)
        + BosonOperator("1^ 0", hopping_strength)
    )
    truncation = 2
    trial_state = np.zeros(truncation**2, dtype=complex)
    trial_state[1] = 1 / np.sqrt(2)  # |n_0=0, n_1=1>
    trial_state[2] = 1 / np.sqrt(2)  # |n_0=1, n_1=0>
    exact_energy = np.vdot(
        trial_state, boson_operator_sparse(hamiltonian, truncation) @ trial_state
    ).real
    calls = []

    def executor(params, measurement, mode_1=None, mode_2=None):
        calls.append((measurement, mode_1, mode_2))
        if measurement == "identity":
            return {"results": [(1, 0)] * 50 + [(0, 1)] * 50}
        if measurement == "hop":
            assert (mode_1, mode_2) == (0, 1)
            return {"results": [(1, 0)] * 100}
        raise ValueError(f"Unexpected measurement: {measurement}")

    sampled_energy = loss_function_bose_hubbard(np.array([0.0]), hamiltonian, executor)

    assert np.isclose(sampled_energy, exact_energy)
    assert calls == [("identity", None, None), ("hop", 0, 1)]


def test_loss_function_bose_hubbard_hopping_with_perceval_simulator():
    hopping_strength = -0.7
    hamiltonian = (
        BosonOperator("0^ 0", 0.4)
        + BosonOperator("1^ 1", 1.2)
        + BosonOperator("0^ 1", hopping_strength)
        + BosonOperator("1^ 0", hopping_strength)
    )
    truncation = 2
    trial_state = np.zeros(truncation**2, dtype=complex)
    trial_state[1] = 1 / np.sqrt(2)  # |n_0=0, n_1=1>
    trial_state[2] = 1 / np.sqrt(2)  # |n_0=1, n_1=0>
    exact_energy = np.vdot(
        trial_state, boson_operator_sparse(hamiltonian, truncation) @ trial_state
    ).real
    calls = []

    def run_perceval(circuit, shots=1000):
        simulator = pcvl.Simulator(pcvl.SLOSBackend())
        simulator.set_circuit(circuit)
        samples = []
        for state, probability in simulator.probs(pcvl.BasicState([1, 0])).items():
            samples.extend([state] * int(round(probability * shots)))
        assert len(samples) == shots
        return {"results": samples}

    def executor(params, measurement, mode_1=None, mode_2=None):
        calls.append((measurement, mode_1, mode_2))
        circuit = pcvl.Circuit(2).add((0, 1), pcvl.BS.H())

        if measurement == "identity":
            return run_perceval(circuit)
        if measurement == "hop":
            assert (mode_1, mode_2) == (0, 1)
            return run_perceval(rotate_modes(circuit, mode_1, mode_2))
        raise ValueError(f"Unexpected measurement: {measurement}")

    sampled_energy = loss_function_bose_hubbard(np.array([0.0]), hamiltonian, executor)

    assert np.isclose(sampled_energy, exact_energy)
    assert calls == [("identity", None, None), ("hop", 0, 1)]


def test_loss_function_bose_hubbard_constant_term_does_not_call_executor():
    def executor(params, measurement, *modes):
        raise AssertionError("Constant-only Hamiltonians should not request samples.")

    hamiltonian = BosonOperator("", 2.5)

    sampled_energy = loss_function_bose_hubbard(np.array([0.0]), hamiltonian, executor)

    assert np.isclose(sampled_energy, 2.5)


def test_loss_function_bose_hubbard_accepts_fock_state_samples():
    hamiltonian = BosonOperator("0^ 0", 1.0)

    def executor(params, measurement):
        assert measurement == "identity"
        return {"results": [exqalibur.FockState([2])] * 10}

    sampled_energy = loss_function_bose_hubbard(np.array([0.0]), hamiltonian, executor)

    assert np.isclose(sampled_energy, 2.0)


def test_loss_function_bose_hubbard_rejects_invalid_inputs():
    def executor(params, measurement, *modes):
        raise AssertionError("Invalid Hamiltonians should fail before sampling.")

    with pytest.raises(TypeError, match="OpenFermion BosonOperator"):
        loss_function_bose_hubbard(np.array([0.0]), {"0^ 0": 1.0}, executor)

    hamiltonian = BosonOperator("0^ 1^", 1.0)

    with pytest.raises(ValueError, match="Unsupported Bose-Hubbard term"):
        loss_function_bose_hubbard(np.array([0.0]), hamiltonian, executor)

    non_normal_ordered_number = BosonOperator("0 0^", 1.0)

    with pytest.raises(ValueError, match="Unsupported Bose-Hubbard term"):
        loss_function_bose_hubbard(np.array([0.0]), non_normal_ordered_number, executor)

    complex_coefficient = BosonOperator("0^ 0", 1.0j)

    with pytest.raises(ValueError, match="real coefficients"):
        loss_function_bose_hubbard(np.array([0.0]), complex_coefficient, executor)

    non_hermitian_hop = BosonOperator("0^ 1", 1.0)

    with pytest.raises(ValueError, match="Hermitian pairs"):
        loss_function_bose_hubbard(np.array([0.0]), non_hermitian_hop, executor)

    asymmetric_hop = BosonOperator("0^ 1", 1.0) + BosonOperator("1^ 0", 2.0)

    with pytest.raises(ValueError, match="identical real coefficients"):
        loss_function_bose_hubbard(np.array([0.0]), asymmetric_hop, executor)


def test_draw_circuit_saves_each_output_format(tmp_path):
    processor = le_ansatz(np.zeros(4), "II")
    output_formats = {
        "mpl": "png",
        "html": "html",
        "latex": "tex",
        "text": "txt",
    }

    for output_format, extension in output_formats.items():
        save_path = tmp_path / f"ansatz.{extension}"
        draw_circuit(
            processor, output_format=output_format, save_path=str(save_path), recursive=True
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 0


def test_draw_circuit_displays_without_save_path(mocker):
    processor = le_ansatz(np.zeros(4), "II")
    pdisplay = mocker.patch("qlass.utils.utils.pcvl.pdisplay")

    draw_circuit(
        processor,
        output_format="text",
        skin="debug",
        compact=True,
        recursive=True,
    )

    pdisplay.assert_called_once()
    args, kwargs = pdisplay.call_args
    assert args[0] is processor
    assert kwargs["output_format"] == pcvl.Format.TEXT
    assert kwargs["skin"].__class__.__name__ == "DebugSkin"
    assert kwargs["recursive"] is True


def test_draw_circuit_normalizes_backend(mocker):
    processor = le_ansatz(np.zeros(4), "II")
    pdisplay = mocker.patch("qlass.utils.utils.pcvl.pdisplay")

    draw_circuit(processor, output_format="text", backend=" Perceval ")

    pdisplay.assert_called_once()


def test_draw_circuit_rejects_invalid_inputs():
    processor = le_ansatz(np.zeros(4), "II")

    with pytest.raises(ValueError, match="Invalid output_format"):
        draw_circuit(processor, output_format="pdf")

    with pytest.raises(ValueError, match="Invalid skin"):
        draw_circuit(processor, skin="minimal")

    with pytest.raises(NotImplementedError, match="perceval"):
        draw_circuit(processor, backend="piquasso")

    with pytest.raises(TypeError, match="Processor or Circuit"):
        draw_circuit(object())


def test_get_probabilities():
    # test case 1
    samples = [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert get_probabilities(samples) == {(0, 0, 0): 0.4, (0, 0, 1): 0.4, (0, 1, 0): 0.2}

    # test case 2
    samples = [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
    assert get_probabilities(samples) == {
        (0, 0, 0): 0.5,
        (0, 0, 1): 0.3333333333333333,
        (0, 1, 0): 0.16666666666666666,
    }

    # test case 3
    samples = [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0)]
    assert get_probabilities(samples) == {
        (0, 0, 0): 0.5714285714285714,
        (0, 0, 1): 0.2857142857142857,
        (0, 1, 0): 0.14285714285714285,
    }


def test_qubit_state_marginal():
    # test case 1
    prob_dist = {
        pcvl.BasicState([0, 0, 0, 0]): 0.4,
        pcvl.BasicState([0, 1, 0, 1]): 0.3,
        pcvl.BasicState([1, 0, 0, 1]): 0.3,
    }
    assert qubit_state_marginal(prob_dist) == {(1, 1): 0.5, (0, 1): 0.5}

    # test case 2
    prob_dist = {
        pcvl.BasicState([0, 1, 0, 1]): 0.4,
        pcvl.BasicState([0, 1, 1, 0]): 0.3,
        pcvl.BasicState([1, 0, 0, 1]): 0.3,
    }
    assert qubit_state_marginal(prob_dist) == {(1, 1): 0.4, (1, 0): 0.3, (0, 1): 0.3}

    # test case 3
    prob_dist = {
        pcvl.BasicState([0, 1, 0, 1, 0, 0]): 0.5,
        pcvl.BasicState([0, 1, 1, 0, 1, 0]): 0.4,
        pcvl.BasicState([1, 0, 0, 1, 0, 1]): 0.1,
    }
    assert qubit_state_marginal(prob_dist) == {(1, 0, 0): 0.8, (0, 1, 1): 0.2}


def test_is_qubit_state():
    # test case 1
    state = pcvl.BasicState([0, 1, 0, 1])
    assert is_qubit_state(state) == (1, 1)

    # test case 2
    state = pcvl.BasicState([1, 0, 0, 1])
    assert is_qubit_state(state) == (0, 1)

    # test case 3
    state = pcvl.BasicState([1, 0, 1, 0])
    assert is_qubit_state(state) == (0, 0)

    # test case 4
    state = pcvl.BasicState([0, 1, 1, 0])
    assert is_qubit_state(state) == (1, 0)

    # test case 5
    state = pcvl.BasicState([1, 1, 0, 1])
    assert not is_qubit_state(state)

    # test case 6
    state = pcvl.BasicState([0, 1, 1, 1])
    assert not is_qubit_state(state)

    # test case 7
    state = pcvl.BasicState([1, 1, 1, 1])
    assert not is_qubit_state(state)

    # test case 8
    state = pcvl.BasicState([0, 0, 0, 1])
    assert not is_qubit_state(state)


def test_loss_function_automatic_grouping():
    """
    Test that loss_function automatically uses Pauli grouping when available.
    This test verifies that the function can import and use the grouping functionality.
    """

    # Define a simple mock executor for testing
    def mock_executor(params, pauli_string):
        # Return a simple mock result that's consistent
        return {"results": [pcvl.BasicState([1, 0, 0, 1]), pcvl.BasicState([0, 1, 1, 0])]}

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
            "results": [
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
        return {"results": ["00", "01", "10", "11"]}

    hamiltonian = {"II": 1.0, "ZZ": 0.5}
    result = loss_function(np.array([0.1, 0.2]), hamiltonian, mock_qiskit_executor)
    assert isinstance(result, float)


def test_loss_function_qiskit_counts_format():
    # Mock Qiskit counts executor
    def mock_counts_executor(params, pauli_string):
        return {"counts": {"00": 250, "01": 250, "10": 250, "11": 250}}

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
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "unexpected format" in str(e).lower()


def test_loss_function_format_consistency():
    # Fixed samples for consistent comparison
    fixed_samples = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def executor1(params, pauli_string):
        return {"results": fixed_samples}

    def executor2(params, pauli_string):
        return {"results": ["00", "01", "10", "11"]}

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


def test_get_probabilities_string_format():
    # test case 1: Qiskit string format
    samples = ["00", "01", "00", "10", "01"]
    expected = {(0, 0): 0.4, (0, 1): 0.4, (1, 0): 0.2}
    assert get_probabilities(samples) == expected

    # test case 2: single qubit strings
    samples = ["0", "1", "0", "0"]
    expected = {(0,): 0.75, (1,): 0.25}
    assert get_probabilities(samples) == expected

    # test case 3: 3-qubit strings
    samples = ["000", "001", "010", "000"]
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
        "qlass.quantum_chemistry.group_commuting_pauli_terms",
        side_effect=ImportError("Simulating grouping utility not found"),
    )

    # 2. Define a simple mock executor that returns a consistent result
    def mock_executor(params, pauli_string):
        # Always returns a sample of |01>
        return {"counts": {"01": 1000}}

    # 3. Define a simple Hamiltonian
    hamiltonian = {"ZI": 0.5, "IZ": -0.5}
    params = np.array([0.1, 0.2])  # Dummy parameters

    # 4. Manually calculate the expected energy for the |01> state
    # For "ZI" (pauli_bin=(1,0)), sample=(0,1): inner dot = 0, sign = (-1)^0 = 1. Expectation = 1.0
    # For "IZ" (pauli_bin=(0,1)), sample=(0,1): inner dot = 1, sign = (-1)^1 = -1. Expectation = -1.0
    # Total expected loss = (0.5 * 1.0) + (-0.5 * -1.0) = 0.5 + 0.5 = 1.0
    expected_loss = 1.0

    # 5. Run the loss function and assert the result
    calculated_loss = loss_function(params, hamiltonian, mock_executor)

    assert np.isclose(calculated_loss, expected_loss)


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
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    pauli_z = pauli_string_to_matrix("Z")

    expectation = compute_expectation_value_from_unitary(H, pauli_z)
    assert np.isclose(expectation, 0.0, atol=1e-10)


def test_loss_function_matrix_multi_term_hamiltonian():
    """Test matrix-based loss function with multiple terms."""

    def identity_executor(params):
        return np.eye(2, dtype=complex)

    hamiltonian = {
        "I": 0.5,  # Identity term
        "Z": 1.0,  # <0|Z|0> = 1
        "X": -0.5,  # <0|X|0> = 0
    }
    params = np.array([0.1])

    loss = loss_function_matrix(params, hamiltonian, identity_executor)
    # Expected: 0.5 * 1 + 1.0 * 1 + (-0.5) * 0 = 1.5
    assert np.isclose(loss, 1.5)


def test_loss_function_matrix_two_qubit():
    """Test matrix-based loss function for 2-qubit system."""

    def identity_executor(params):
        return np.eye(4, dtype=complex)

    hamiltonian = {"II": 1.0, "ZZ": 0.5, "XX": -0.3}
    params = np.array([0.1, 0.2])

    loss = loss_function_matrix(params, hamiltonian, identity_executor)
    # |00> state: II=1, ZZ=1, XX=0
    # Expected: 1.0 * 1 + 0.5 * 1 + (-0.3) * 0 = 1.5
    assert np.isclose(loss, 1.5)


def test_loss_function_matrix_parameterized_unitary():
    """Test with parameterized unitary (rotation)."""

    def rotation_executor(params):
        # Rotation around Y axis: Ry(theta)
        theta = params[0]
        return np.array(
            [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]],
            dtype=complex,
        )

    hamiltonian = {"Z": 1.0}

    # At theta=0, should get <0|Z|0> = 1
    loss_0 = loss_function_matrix(np.array([0.0]), hamiltonian, rotation_executor)
    assert np.isclose(loss_0, 1.0)

    # At theta=pi, should get <1|Z|1> = -1
    loss_pi = loss_function_matrix(np.array([np.pi]), hamiltonian, rotation_executor)
    assert np.isclose(loss_pi, -1.0)

    # At theta=pi/2, should get 0
    loss_half = loss_function_matrix(np.array([np.pi / 2]), hamiltonian, rotation_executor)
    assert np.isclose(loss_half, 0.0, atol=1e-10)


def test_permanent():
    """Test permanent calculation for key cases."""
    from qlass.utils import permanent

    # Identity matrix
    I_matrix = np.eye(2, dtype=complex)
    assert np.isclose(permanent(I_matrix), 1.0)

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
        return np.eye(4, dtype=complex)  # 2 qubits, 4 modes

    hamiltonian = {"II": 0.5, "ZZ": 1.0}
    params = np.array([0.1, 0.2])

    loss = loss_function_photonic_unitary(params, hamiltonian, identity_executor_4x4)

    # Initial state |00⟩: II=1, ZZ=1
    # Expected: 0.5*1 + 1.0*1 = 1.5
    assert np.isclose(loss, 1.5), "Test failed for default state, no ancillas"

    # --- (Custom Initial State, No Ancillas) ---
    initial_state_11 = np.array([0, 0, 0, 1], dtype=complex)  # Logical |11⟩
    hamiltonian_z = {"ZZ": 1.0}
    loss_11 = loss_function_photonic_unitary(
        params, hamiltonian_z, identity_executor_4x4, initial_state_11
    )
    # <11|ZZ|11> = 1
    assert np.isclose(loss_11, 1.0), "Test failed for |11⟩ state, no ancillas"

    # --- With Ancillas, Identity logical sub-matrix ---
    # 6 total modes: 2 logical qubits (4 modes) + 2 ancillary modes
    def identity_executor_6x6(params):
        return np.eye(6, dtype=complex)

    # Ancillary modes are [0, 5]. Logical modes are [1, 2, 3, 4]
    # Logical Q0 -> physical modes [1, 2]
    # Logical Q1 -> physical modes [3, 4]
    anc_modes = [0, 5]

    # Use the same 2-qubit Hamiltonian
    # With a 6x6 identity matrix, the logical evolution is also identity.
    # <00|H|00> should be 1.5
    loss_ancilla = loss_function_photonic_unitary(
        params, hamiltonian, identity_executor_6x6, ancillary_modes=anc_modes
    )
    assert np.isclose(loss_ancilla, 1.5), "Test failed for default state with ancillas"

    # --- With Ancillas, Custom Initial State ---
    # Use the same 2-qubit |11⟩ initial state
    loss_ancilla_11 = loss_function_photonic_unitary(
        params, hamiltonian_z, identity_executor_6x6, initial_state_11, anc_modes
    )
    # <11|ZZ|11> = 1
    assert np.isclose(loss_ancilla_11, 1.0), "Test failed for |11⟩ state with ancillas"

    # --- Post-selection failure (leakage to ancilla) ---
    # Unitary swaps a logical mode (1) with an ancillary mode (5)
    def swap_1_5_executor(params):
        U = np.eye(6, dtype=complex)
        U[1, 1] = 0
        U[5, 5] = 0
        U[1, 5] = 1
        U[5, 1] = 1
        return U

    # Logical |00⟩ (physical modes [1, 3]) evolves to a state
    # where the photon from mode 1 goes to mode 5 (ancillary).
    # This should fail post-selection.
    # Success probability should be 0, loss should be penalty (1e6)
    loss_fail = loss_function_photonic_unitary(
        params, hamiltonian, swap_1_5_executor, ancillary_modes=anc_modes
    )
    assert np.isclose(loss_fail, 1e6), "Test failed to return penalty for post-selection failure"

    # --- Error - Invalid Ancilla Index ---
    with pytest.raises(ValueError, match="contain indices outside"):
        loss_function_photonic_unitary(
            params, hamiltonian, identity_executor_4x4, ancillary_modes=[0, 10]
        )

    # --- Error - Odd Logical Modes ---
    with pytest.raises(ValueError, match="must be even"):
        loss_function_photonic_unitary(
            params, hamiltonian, identity_executor_4x4, ancillary_modes=[0]
        )

    # --- Error - Mismatched Initial State Dim ---
    with pytest.raises(ValueError, match="Initial state dimension"):
        # 6 modes, 2 ancillas -> 4 logical modes -> 2 qubits -> dim 4
        # Pass a 1-qubit (dim 2) initial state
        initial_state_1q = np.array([0, 1], dtype=complex)
        loss_function_photonic_unitary(
            params, hamiltonian, identity_executor_6x6, initial_state_1q, anc_modes
        )


def test_ensemble_weights():
    from qlass.utils.utils import ensemble_weights

    equal_weights = ensemble_weights("equi", 2)
    assert isinstance(equal_weights, list)
    dec_weights = ensemble_weights("weighted", 2)
    assert isinstance(dec_weights, list)
    ground_state = ensemble_weights("ground_state_only", 2)
    assert isinstance(ground_state, list)

    with pytest.raises(
        ValueError,
        match="Invalid weights_choice. Must be one of 'equi', 'weighted', or 'ground_state_only'.",
    ):
        ensemble_weights("Invalid_weight", 2)


def test_loss_function_with_mitigator(mocker):
    """Test loss function with a mitigator."""
    from qlass.utils import loss_function

    # Mock executor results as counts
    def mock_executor(params, pauli_string):
        return {"counts": {"00": 100}}  # 100 shots of |00>

    # Mock mitigator
    mitigator = MagicMock()
    # Mitigator returns a probability distribution
    mitigator.mitigate.return_value = {(0, 0): 1.0}

    hamiltonian = {"ZZ": 1.0}
    params = np.array([0.1])

    # Run with mitigator
    loss = loss_function(params, hamiltonian, mock_executor, mitigator=mitigator)

    # Verify mitigator was called
    assert mitigator.mitigate.called
    assert np.isclose(loss, 1.0)


def test_loss_function_grouping_import_error(mocker):
    """Test that loss_function handles ImportError for grouping gracefully."""

    # Mock modules to simulate missing qlass.quantum_chemistry.hamiltonians
    # We use a patch on sys.modules to hide the module
    with patch.dict(sys.modules, {"qlass.quantum_chemistry.hamiltonians": None}):
        # Mock executor
        def mock_executor(params, pauli_string):
            return {"counts": {"00": 100}}

        hamiltonian = {"ZZ": 1.0}
        params = np.array([0.1])

        # This calls loss_function. Because qlass.quantum_chemistry.hamiltonians is None,
        # it raises ImportError on import, catching it and setting use_grouping=False.
        # Then it iterates over H directly.
        from qlass.utils import loss_function

        loss = loss_function(params, hamiltonian, mock_executor)

        # <00|ZZ|00> = 1.0
        assert np.isclose(loss, 1.0)


def test_rotate_qubits_qiskit_circuit():
    """The qiskit branch must map the measured Pauli onto Z: U† Z U == P."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    }
    for pauli, matrix in expected.items():
        rotated = rotate_qubits(pauli, QuantumCircuit(1))
        u = Operator(rotated).data
        assert np.allclose(u.conj().T @ Z @ u, matrix), f"{pauli} rotation is wrong"


def test_rotate_qubits_qiskit_bell_state_expectation():
    """Regression test for issue #207: <YY> on a Bell state is -1, not +1.

    For (|00> + |11>)/sqrt(2), <XX> = +1 while <YY> = -1, so this state
    distinguishes a Y measurement from an accidental X measurement.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    for pauli_string, exact in [("XX", 1.0), ("YY", -1.0)]:
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        rotated = rotate_qubits(pauli_string, bell)
        expectation = sum(
            prob * (-1) ** bin(int(bitstring, 2)).count("1")
            for bitstring, prob in Statevector(rotated).probabilities_dict().items()
        )
        assert np.isclose(expectation, exact), f"<{pauli_string}> = {expectation}, expected {exact}"


def test_loss_function_y_terms_match_exact_expectation():
    """End-to-end regression for issue #207: sampled <YY> through the photonic
    pipeline must match the exact statevector value.

    For these parameters <YY> = -0.271 while <XX> = +0.915, so measuring the
    wrong basis cannot pass.
    """
    from perceval.algorithm import Sampler
    from qiskit.circuit.library import n_local
    from qiskit.quantum_info import SparsePauliOp, Statevector

    params = np.array([0.4, 0.8, 1.1, 0.3])
    hamiltonian = {"YY": 1.0}

    ansatz = n_local(2, "ry", "cx", reps=1, entanglement="linear").assign_parameters(params)
    exact = Statevector(ansatz).expectation_value(SparsePauliOp("YY")).real

    def executor(p, pauli_string):
        processor = le_ansatz(p, pauli_string)
        return Sampler(processor).samples(20_000)

    sampled = loss_function(params, hamiltonian, executor)
    assert np.isclose(sampled, exact, atol=0.05), f"sampled {sampled}, exact {exact}"


def test_rotate_qubits_perceval_circuit():
    """The perceval branch must add a mode-pair unitary U with U† Z U == P."""
    import perceval as pcvl

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    }
    for pauli, matrix in expected.items():
        rotated = rotate_qubits(pauli, pcvl.Circuit(2))
        u = np.array(rotated.compute_unitary())
        assert np.allclose(u.conj().T @ Z @ u, matrix), f"{pauli} rotation is wrong"


def test_extract_samples_fallback():
    """Test _extract_samples_from_executor_result fallback logic."""
    # Input is a dict with a list value, but no "results" or "counts" key
    samples_dict = {"custom_key": [(0, 0), (0, 1)]}
    extracted = _extract_samples_from_executor_result(samples_dict)
    assert extracted == [(0, 0), (0, 1)]


def test_extract_samples_invalid_format():
    """Test _extract_samples_from_executor_result with invalid input."""
    # Input is a dict but no list values
    invalid_dict = {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="Could not extract sample list"):
        # The function logic actually raises "Expected dict with..." or so.
        # But if it reaches the end without finding list, it returns raises ValueError.
        # Wait, lines 375-378 raise ValueError if type is not dict/list.
        # If dict but no list found, line 381 raises ValueError.
        _extract_samples_from_executor_result(invalid_dict)

    # Input is not a valid type (e.g. an int)
    with pytest.raises(ValueError, match="Executor returned unexpected format"):
        _extract_samples_from_executor_result(123)


def test_loss_function_fallback_with_mitigator(mocker):
    """Test loss function fallback path (no grouping) with mitigator."""
    from qlass.utils import loss_function

    # Mock modules to simulate missing qlass.quantum_chemistry.hamiltonians
    with patch.dict(sys.modules, {"qlass.quantum_chemistry.hamiltonians": None}):
        # Mock executor
        def mock_executor(params, pauli_string):
            return {"counts": {"00": 100}}

        # Mock mitigator
        mitigator = MagicMock()
        mitigator.mitigate.return_value = {(0, 0): 1.0}

        hamiltonian = {"ZZ": 1.0}
        params = np.array([0.1])

        # Run with mitigator.
        # Import error will force use_grouping = False.
        # mitigator != None will hit the target block.
        loss = loss_function(params, hamiltonian, mock_executor, mitigator=mitigator)

        assert mitigator.mitigate.called
        assert np.isclose(loss, 1.0)
