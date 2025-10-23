
import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler
import pytest

from qlass.vqe import (
    VQE,
    le_ansatz,
    custom_unitary_ansatz,
    hf_ansatz
)

from qlass.quantum_chemistry import (
    LiH_hamiltonian,
    Hchain_KS_hamiltonian
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
    final_energy = simple_vqe.run(max_iterations=10, verbose=False)

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
    simple_vqe.run(max_iterations=10, verbose=False)
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

    simple_vqe.run(max_iterations=10, verbose=False)
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

def test_evqe_pipeline():

    # Define an executor function that uses the linear entangled ansatz
    def executor(params, pauli_string):

        processors = hf_ansatz(1, n_orbs, params, pauli_string, method="DFT", cost="e-VQE")
        samplers = [Sampler(p) for p in processors]
        samples = [sampler.samples(5) for sampler in samplers]

        return samples

    # Number of qubits
    num_qubits = 2

    ham, scf_mo_energy, n_orbs = Hchain_KS_hamiltonian(4, 1.2)

    vqe = VQE(
        hamiltonian=ham,
        executor=executor,
        num_params=4,  # Number of parameters in the linear entangled ansatz
    )

    # Run the VQE optimization
    vqe_energy = vqe.run(
        max_iterations=5,
        verbose=True,
        weight_option="weighted",
        cost="e-VQE"
    )

    if not isinstance(vqe_energy, float):
        raise ValueError("Optimization result is not a valid float")

def test_invalid_cost_type():
    """Test that invalid cost error."""

    # Define an executor function that uses the linear entangled ansatz
    def executor(params, pauli_string):

        processors = hf_ansatz(1, n_orbs, params, pauli_string, method="DFT", cost="e-VQE")
        samplers = [Sampler(p) for p in processors]
        samples = [sampler.samples(5) for sampler in samplers]

        return samples

    # Number of qubits
    num_qubits = 2

    ham, scf_mo_energy, n_orbs = Hchain_KS_hamiltonian(4, 1.2)

    vqe = VQE(
        hamiltonian=ham,
        executor=executor,
        num_params=4,  # Number of parameters in the linear entangled ansatz
    )

    with pytest.raises(ValueError, match="Invalid cost option. Use 'VQE' or 'e-VQE'."):
        vqe_energy = vqe.run(
            max_iterations=5,
            verbose=True,
            weight_option="weighted",
            cost="Invalid_cost"
        )

def test_invalid_evqe_executor_type():
    """Test that invalid executor error."""
    def unitary_exec():
        return np.eye(4, dtype=complex)

    # Number of qubits
    num_qubits = 2

    ham, scf_mo_energy, n_orbs = Hchain_KS_hamiltonian(4, 1.2)

    vqe = VQE(
        hamiltonian=ham,
        executor=unitary_exec(),
        executor_type="qubit_unitary",
        num_params=4,  # Number of parameters in the linear entangled ansatz
    )

    with pytest.raises(ValueError, match="option: e-VQE takes only executor_type: sampling"):
        vqe_energy = vqe.run(
            max_iterations=5,
            verbose=True,
            weight_option="weighted",
            cost="e-VQE"
        )

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

def test_hf_ansatz_vqe():
    proc = hf_ansatz(
        layers=1,
        n_orbs=1,
        lp=np.array([0.45674329, 0.91022972, 0.94590395, 0.58386885]),
        pauli_string="II",
        method="WFT",
        cost="VQE",
        noise_model=None
    )
    # Check that the returned object is not None
    assert proc is not None
    # Basic attribute check for Perceval processor
    assert isinstance(proc, pcvl.Processor)

def test_hf_ansatz_e_vqe():
    procs = hf_ansatz(
        layers=1,
        n_orbs=1,
        lp=np.array([0.45674329, 0.91022972, 0.94590395, 0.58386885]),
        pauli_string="II",
        method="WFT",
        cost="e-VQE",
        noise_model=None
    )
    assert isinstance(procs, list)
    assert len(procs) > 0
    # Each element should look like a processor
    for p in procs:
        assert isinstance(p, pcvl.Processor)

def test_plot_convergence(simple_vqe, mocker):
    """
    Tests the plot_convergence method.

    It verifies that a ValueError is raised if no optimization history exists,
    and that matplotlib plotting functions are called correctly when data is available.
    """
    # 1. Test ValueError when history is empty
    with pytest.raises(ValueError, match="No optimization history available."):
        simple_vqe.plot_convergence()

    # 2. Mock all pyplot functions to avoid GUI pop-ups
    mock_figure = mocker.patch('matplotlib.pyplot.figure')
    mock_plot = mocker.patch('matplotlib.pyplot.plot')
    mock_axhline = mocker.patch('matplotlib.pyplot.axhline')
    mock_xlabel = mocker.patch('matplotlib.pyplot.xlabel')
    mock_ylabel = mocker.patch('matplotlib.pyplot.ylabel')
    mock_title = mocker.patch('matplotlib.pyplot.title')
    mock_legend = mocker.patch('matplotlib.pyplot.legend')
    mock_grid = mocker.patch('matplotlib.pyplot.grid')
    mock_show = mocker.patch('matplotlib.pyplot.show')

    # 3. Run VQE to populate history and call the plot function
    simple_vqe.run(max_iterations=10, verbose=False)
    exact_energy_val = -1.0
    simple_vqe.plot_convergence(exact_energy=exact_energy_val)

    # 4. Assert that the plotting functions were called as expected
    mock_figure.assert_called_once()
    mock_plot.assert_called_once()
    mock_axhline.assert_called_once_with(y=exact_energy_val, color='r', linestyle='--', label='Exact Energy')
    mock_xlabel.assert_called_once()
    mock_ylabel.assert_called_once()
    mock_title.assert_called_once()
    mock_legend.assert_called_once()
    mock_grid.assert_called_once()
    mock_show.assert_called_once()

def identity_unitary_executor(params):
    """Simple unitary executor that returns identity matrix."""
    return np.eye(4, dtype=complex)


def parametrized_unitary_executor(params):
    """Unitary executor with actual parameters."""
    # Create a simple 2-qubit unitary using RY rotations
    theta1, theta2 = params[0], params[1]

    ry1 = np.array([
        [np.cos(theta1/2), -np.sin(theta1/2)],
        [np.sin(theta1/2), np.cos(theta1/2)]
    ], dtype=complex)

    ry2 = np.array([
        [np.cos(theta2/2), -np.sin(theta2/2)],
        [np.sin(theta2/2), np.cos(theta2/2)]
    ], dtype=complex)

    return np.kron(ry1, ry2)


def test_vqe_init_with_unitary_executor():
    """Test VQE initialization with unitary executor type."""
    hamiltonian = {"II": -0.5, "ZZ": 1.0}

    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=identity_unitary_executor,
        num_params=2,
        executor_type="qubit_unitary"
    )
    
    assert vqe.executor_type == "qubit_unitary"
    assert vqe.num_qubits == 2
    assert vqe.num_params == 2


def test_vqe_init_with_sampling_executor():
    """Test VQE initialization with sampling executor type."""
    hamiltonian = {"II": -0.5, "ZZ": 1.0}

    def sampling_exec(params, pauli_string):
        return {'results': [(0, 0)] * 100}

    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=sampling_exec,
        num_params=2,
        executor_type="sampling"
    )

    assert vqe.executor_type == "sampling"

def test_vqe_invalid_executor_type():
    """Test that invalid executor_type raises error."""
    hamiltonian = {"ZZ": 1.0}

    with pytest.raises(ValueError, match="Invalid executor_type"):
        VQE(
            hamiltonian=hamiltonian,
            executor=identity_unitary_executor,
            num_params=2,
            executor_type="invalid_type"
        )

def test_vqe_compare_unitary_vs_sampling():
    """
    Test that unitary and sampling executors give consistent results
    for the same problem (within statistical error).
    """
    hamiltonian = {"II": 0.5, "ZZ": 1.0}

    # Unitary executor
    def unitary_exec(params):
        return np.eye(4, dtype=complex)

    # Sampling executor (deterministic - always returns |00>)
    def sampling_exec(params, pauli_string):
        return {'results': [(0, 0)] * 10000}

    vqe_unitary = VQE(
        hamiltonian=hamiltonian,
        executor=unitary_exec,
        num_params=2,
        executor_type="qubit_unitary"
    )

    vqe_sampling = VQE(
        hamiltonian=hamiltonian,
        executor=sampling_exec,
        num_params=2,
        executor_type="sampling"
    )

    # Both should give same energy for identity circuit
    energy_unitary = vqe_unitary.run(
        initial_params=np.zeros(2),
        max_iterations=10,
        verbose=False
    )
    energy_sampling = vqe_sampling.run(
        initial_params=np.zeros(2),
        max_iterations=10,
        verbose=False
    )

    # Expected: 0.5 * 1 + 1.0 * 1 = 1.5
    assert np.isclose(energy_unitary, 1.5)
    assert np.isclose(energy_sampling, 1.5, atol=0.01)

def photonic_identity_executor(params):
    """Photonic unitary executor that returns identity."""
    return np.eye(4, dtype=complex)  # 2 qubits, 4 modes


def photonic_parametrized_executor(params):
    """Photonic unitary executor with actual parameters."""
    # Create a simple 4×4 unitary using rotation-like structure
    theta = params[0]
    phi = params[1] if len(params) > 1 else 0
    
    # Simple parameterized unitary for 2 qubits (4 modes)
    U = np.eye(4, dtype=complex)
    
    # Apply rotation-like transformation
    c, s = np.cos(theta), np.sin(theta)
    U[0, 0] = c * np.exp(1j * phi)
    U[0, 1] = -s
    U[1, 0] = s
    U[1, 1] = c * np.exp(-1j * phi)
    
    return U


def test_vqe_init_with_photonic_unitary_executor():
    """Test VQE initialization with photonic_unitary executor type."""
    hamiltonian = {"II": -0.5, "ZZ": 1.0}
    
    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=photonic_identity_executor,
        num_params=2,
        executor_type="photonic_unitary"
    )
    
    assert vqe.executor_type == "photonic_unitary"
    assert vqe.num_qubits == 2
    assert vqe.num_params == 2


def test_vqe_run_with_photonic_unitary():
    """Test VQE run with photonic unitary executor."""
    hamiltonian = {"II": 0.5, "ZZ": 1.0}
    
    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=photonic_identity_executor,
        num_params=2,
        executor_type="photonic_unitary"
    )
    
    energy = vqe.run(
        initial_params=np.zeros(2),
        max_iterations=5,
        verbose=False
    )
    
    # With identity executor and |00⟩ initial state: II=1, ZZ=1
    # Expected: 0.5*1 + 1.0*1 = 1.5
    assert np.isclose(energy, 1.5)
    assert len(vqe.energy_history) > 0


def test_vqe_photonic_unitary_with_custom_initial_state():
    """Test VQE with custom initial state."""
    hamiltonian = {"ZZ": 1.0}
    
    # Initial state |11⟩
    initial_state = np.array([0, 0, 0, 1], dtype=complex)
    
    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=photonic_identity_executor,
        num_params=2,
        executor_type="photonic_unitary",
        initial_state=initial_state
    )
    
    energy = vqe.run(
        initial_params=np.zeros(2),
        max_iterations=5,
        verbose=False
    )
    
    # <11|ZZ|11> = 1
    assert np.isclose(energy, 1.0)


def test_vqe_photonic_unitary_optimization():
    """Test that VQE optimization works with photonic unitary executor."""
    hamiltonian = {"II": 0.5, "ZZ": 1.0, "XX": -0.5}
    
    vqe = VQE(
        hamiltonian=hamiltonian,
        executor=photonic_parametrized_executor,
        num_params=2,
        executor_type="photonic_unitary"
    )
    
    energy = vqe.run(max_iterations=10, verbose=False)
    
    assert isinstance(energy, float)
    assert vqe.optimization_result is not None
    assert len(vqe.energy_history) > 0
    
    # Get optimal parameters
    optimal_params = vqe.get_optimal_parameters()
    assert len(optimal_params) == 2


def test_vqe_compare_executor_types():
    """Test that different executor types work correctly."""
    hamiltonian = {"ZZ": 1.0}
    
    # Sampling executor
    def sampling_exec(params, pauli_string):
        return {'results': [(0, 0)] * 100}
    
    # Unitary executor
    def unitary_exec(params):
        return np.eye(4, dtype=complex)
    
    # Photonic unitary executor
    def photonic_exec(params):
        return np.eye(4, dtype=complex)
    
    vqe_sampling = VQE(
        hamiltonian=hamiltonian,
        executor=sampling_exec,
        num_params=2,
        executor_type="sampling"
    )
    
    vqe_unitary = VQE(
        hamiltonian=hamiltonian,
        executor=unitary_exec,
        num_params=2,
        executor_type="qubit_unitary"
    )
    
    vqe_photonic = VQE(
        hamiltonian=hamiltonian,
        executor=photonic_exec,
        num_params=2,
        executor_type="photonic_unitary"
    )
    
    # All should initialize successfully
    assert vqe_sampling.executor_type == "sampling"
    assert vqe_unitary.executor_type == "qubit_unitary"
    assert vqe_photonic.executor_type == "photonic_unitary"
    
    # Run short optimizations
    energy_sampling = vqe_sampling.run(
        initial_params=np.zeros(2), max_iterations=3, verbose=False
    )
    energy_unitary = vqe_unitary.run(
        initial_params=np.zeros(2), max_iterations=3, verbose=False
    )
    energy_photonic = vqe_photonic.run(
        initial_params=np.zeros(2), max_iterations=3, verbose=False
    )
    
    # For identity operators and |00⟩ state, all should give similar results
    assert np.isclose(energy_unitary, energy_photonic, atol=0.1)
