#TODO: add some tests for the helper functions and hamiltonian functions

import unittest

from qlass.utils.utils import compute_energy, get_probabilities, qubit_state_marginal, is_qubit_state
import perceval as pcvl

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import TwoLocal
from perceval.algorithm import Sampler

from qlass import compile
from qlass.vqe import VQE, le_ansatz, custom_unitary_ansatz
from qlass.quantum_chemistry import LiH_hamiltonian

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

    # Define an executor function that uses the linear entangled ansatz
    def executor(params, pauli_string):
        processor = le_ansatz(params, pauli_string)
        sampler = Sampler(processor)
        samples = sampler.samples(10_000)
        return samples

    # Number of qubits
    num_qubits = 2

    # Generate a 2-qubit Hamiltonian
    hamiltonian = LiH_hamiltonian(num_electrons=2, num_orbitals=1)

    # Initialize the VQE solver with the custom executor
    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=executor,
        num_params=2*num_qubits, # Number of parameters in the linear entangled ansatz
    )

    # Run the VQE optimization
    vqe_energy = vqe.run(
        max_iterations=10,
        verbose=True
    )

    if not isinstance(vqe_energy, float):
        raise ValueError("Optimization result is not a valid float")


class TestCustomUnitaryAnsatz(unittest.TestCase):
    def test_valid_hadamard_2mode(self):
        """Test with a valid 2-mode Hadamard matrix."""
        H_matrix = (1 / np.sqrt(2)) * np.array([[1, 1],
                                               [1, -1]], dtype=complex)
        processor = custom_unitary_ansatz(U=H_matrix)
        self.assertIsInstance(processor, pcvl.Processor)
        self.assertEqual(processor.m, 2)

        # Retrieve the matrix from the processor
        # Based on previous findings, the component is in processor.components[0]
        # and might be a tuple (port_range, component_obj)
        self.assertTrue(hasattr(processor, 'components'))
        self.assertIsInstance(processor.components, list)
        self.assertGreater(len(processor.components), 0)

        component_wrapper = processor.components[0]
        actual_component = component_wrapper[1] if isinstance(component_wrapper, tuple) else component_wrapper

        self.assertIsInstance(actual_component, pcvl.Unitary)
        self.assertTrue(hasattr(actual_component, 'U'))

        retrieved_matrix_obj = actual_component.U
        retrieved_matrix_np = np.array(retrieved_matrix_obj, dtype=complex) # Ensure NumPy array

        np.testing.assert_array_almost_equal(retrieved_matrix_np, H_matrix, decimal=5)

    def test_valid_dft_3mode_with_dummy_args(self):
        """Test with a valid 3-mode DFT matrix and dummy le_ansatz arguments."""
        omega = np.exp(2 * np.pi * 1j / 3)
        DFT_matrix = (1 / np.sqrt(3)) * np.array([[1, 1, 1],
                                                  [1, omega, omega**2],
                                                  [1, omega**2, omega**4]], dtype=complex)

        # Pass dummy positional and keyword arguments for le_ansatz
        processor = custom_unitary_ansatz(1, "arg2", U=DFT_matrix, dummy_kwarg="value", num_modes=3)
        self.assertIsInstance(processor, pcvl.Processor)
        self.assertEqual(processor.m, 3)

        component_wrapper = processor.components[0]
        actual_component = component_wrapper[1] if isinstance(component_wrapper, tuple) else component_wrapper
        retrieved_matrix_obj = actual_component.U
        retrieved_matrix_np = np.array(retrieved_matrix_obj, dtype=complex)

        np.testing.assert_array_almost_equal(retrieved_matrix_np, DFT_matrix, decimal=5)

    def test_num_modes_kwarg_mismatch(self):
        """Test ValueError if 'num_modes' in kwargs mismatches U's dimension."""
        H_matrix = (1 / np.sqrt(2)) * np.array([[1, 1],
                                               [1, -1]], dtype=complex)
        with self.assertRaisesRegex(ValueError, "does not match the dimension of U"):
            custom_unitary_ansatz(U=H_matrix, num_modes=3) # num_modes=3, U is 2x2

    def test_non_unitary_matrix(self):
        """Test ValueError for a non-unitary matrix."""
        non_unitary_matrix = np.array([[1, 2], [3, 4]], dtype=complex)
        with self.assertRaisesRegex(ValueError, "The provided matrix U is not unitary"):
            custom_unitary_ansatz(U=non_unitary_matrix)

    def test_non_square_matrix(self):
        """Test ValueError for a non-square matrix."""
        non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=complex)
        with self.assertRaisesRegex(ValueError, "Unitary matrix U must be a square 2D array"):
            custom_unitary_ansatz(U=non_square_matrix)

    def test_non_numpy_array_input_for_U(self):
        """Test TypeError if U is not a NumPy ndarray."""
        not_a_matrix = [[1, 0], [0, 1]] # A list, not a NumPy array
        with self.assertRaisesRegex(TypeError, "Unitary matrix U must be a NumPy ndarray"):
            custom_unitary_ansatz(U=not_a_matrix)

    def test_empty_matrix_input(self):
        """Test behavior with an empty NumPy array (should fail square/dimension checks)."""
        empty_matrix = np.array([], dtype=complex)
        # This will likely be caught by the "square 2D array" check first,
        # or dimension checks if they are more specific for empty arrays.
        with self.assertRaises(ValueError): # General ValueError as specific message might vary
            custom_unitary_ansatz(U=empty_matrix)

    def test_1x1_unitary_matrix(self):
        """Test with a valid 1x1 unitary matrix."""
        U_1x1 = np.array([[np.exp(1j * 0.5)]], dtype=complex) # e.g., a phase
        processor = custom_unitary_ansatz(U=U_1x1)
        self.assertIsInstance(processor, pcvl.Processor)
        self.assertEqual(processor.m, 1)

        component_wrapper = processor.components[0]
        actual_component = component_wrapper[1] if isinstance(component_wrapper, tuple) else component_wrapper
        retrieved_matrix_obj = actual_component.U
        retrieved_matrix_np = np.array(retrieved_matrix_obj, dtype=complex)
        np.testing.assert_array_almost_equal(retrieved_matrix_np, U_1x1, decimal=5)

