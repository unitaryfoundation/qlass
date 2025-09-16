import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler
import pytest

from qlass.vqe import (
    VQE,
    le_ansatz,
    custom_unitary_ansatz,
)

from qlass.quantum_chemistry import (
    LiH_hamiltonian,
)

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


def mock_executor(params, pauli_string):
    """
    A mock executor that returns a predictable distribution.
    For this test, the actual computation doesn't matter, only the format.
    It returns a distribution where '00' and '11' are equally likely.
    """
    return {'counts': {'00': 500, '11': 500}}

@pytest.fixture
def simple_vqe():
    """Provides a simple VQE instance for testing."""
    hamiltonian = {"II": -0.5, "ZZ": 1.0, "XX": 0.5}
    num_params = 4
    return VQE(hamiltonian=hamiltonian, executor=mock_executor, num_params=num_params)

def test_vqe_init(simple_vqe):
    """Tests if the VQE class is initialized with the correct attributes."""
    assert simple_vqe.num_qubits == 2
    assert simple_vqe.num_params == 4
    assert simple_vqe.optimizer == "COBYLA"
    assert callable(simple_vqe.executor)
    assert simple_vqe.optimization_result is None

def test_vqe_run(simple_vqe):
    """Tests the main `run` method to ensure optimization completes."""
    # Run with a small number of iterations for speed
    final_energy = simple_vqe.run(max_iterations=3, verbose=False)
    
    assert isinstance(final_energy, float)
    assert simple_vqe.optimization_result is not None
    # The callback should populate the history
    assert len(simple_vqe.energy_history) > 0
    assert len(simple_vqe.parameter_history) > 0
    # The final energy should match the one in the result object
    assert np.isclose(final_energy, simple_vqe.optimization_result.fun)

def test_get_optimal_parameters(simple_vqe):
    """
    Tests that `get_optimal_parameters` returns correct parameters after a run
    and raises an error if run before.
    """
    # Should raise error before running
    with pytest.raises(ValueError, match="VQE optimization has not been run yet."):
        simple_vqe.get_optimal_parameters()

    # After running
    simple_vqe.run(max_iterations=3, verbose=False)
    optimal_params = simple_vqe.get_optimal_parameters()
    
    assert isinstance(optimal_params, np.ndarray)
    assert len(optimal_params) == simple_vqe.num_params
    assert np.allclose(optimal_params, simple_vqe.optimization_result.x)

def test_compare_with_exact(simple_vqe):
    """
    Tests the comparison with an exact energy value.
    """
    # Should raise error before running
    with pytest.raises(ValueError, match="VQE optimization has not been run yet."):
        simple_vqe.compare_with_exact(0.0)
    
    simple_vqe.run(max_iterations=3, verbose=False)
    vqe_energy = simple_vqe.optimization_result.fun
    exact_energy = 0.5
    
    comparison = simple_vqe.compare_with_exact(exact_energy)
    
    expected_abs_error = abs(vqe_energy - exact_energy)
    expected_rel_error = expected_abs_error / abs(exact_energy)
    
    assert comparison['vqe_energy'] == vqe_energy
    assert comparison['exact_energy'] == exact_energy
    assert np.isclose(comparison['absolute_error'], expected_abs_error)
    assert np.isclose(comparison['relative_error'], expected_rel_error)

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

def test_custom_unitary_ansatz():
    """
    Test that custom_unitary_ansatz correctly implements the Hadamard gate
    by checking the output probability distribution from the Perceval processor.
    """

    # Define 1-qubit Hadamard gate
    H = 1 / np.sqrt(2) * np.array([[1, 1],
                                   [1, -1]])
    lp_dummy = np.array([0.0])
    pauli_string = "Z"

    # Create processor
    processor = custom_unitary_ansatz(lp_dummy, pauli_string, H)

    # Sample from the processor using Perceval's Sampler
    sampler = pcvl.algorithm.Sampler(processor)
    samples = sampler.samples(10000)
    sample_count = sampler.sample_count(10000)
    prob_dist = sampler.probs()

    # Extract probabilities from BSDistribution
    prob_dict = {state: float(prob) for state, prob in prob_dist['results'].items()}

    # Assert both logical outcomes are present and roughly balanced
    assert len(prob_dict) == 2, f"Unexpected number of outcomes: {prob_dict}"
    keys = list(prob_dict.keys())
    assert all(state in prob_dict for state in [pcvl.BasicState('|1,0>'), pcvl.BasicState('|0,1>')]), \
        f"Expected states |1,0> and |0,1> not found in results: {prob_dict}"

    prob_0 = prob_dict[pcvl.BasicState('|1,0>')]
    prob_1 = prob_dict[pcvl.BasicState('|0,1>')]

    assert 0.45 <= prob_0 <= 0.55, f"Unexpected probability for |0⟩: {prob_0}"
    assert 0.45 <= prob_1 <= 0.55, f"Unexpected probability for |1⟩: {prob_1}"