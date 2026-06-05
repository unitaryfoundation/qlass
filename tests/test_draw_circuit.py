import warnings
from unittest.mock import patch

import numpy as np
import perceval as pcvl
import pytest

from qlass.utils import draw_circuit
from qlass.utils.draw_circuit import _OUTPUT_FORMATS, resolve_executor_circuit
from qlass.vqe import VQE, le_ansatz

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


@pytest.fixture
def two_qubit_processor():
    """Provides a 2-qubit linear entangled ansatz processor."""
    params = np.array([0.1, 0.2, 0.3, 0.4])
    return le_ansatz(params, "II")


@pytest.mark.parametrize("output_format", ["mpl", "html", "latex", "text"])
@patch("qlass.utils.draw_circuit.pcvl.pdisplay")
def test_draw_circuit_output_formats(mock_pdisplay, two_qubit_processor, output_format):
    """Tests that draw_circuit runs for each supported output format."""
    draw_circuit(two_qubit_processor, output_format=output_format)
    mock_pdisplay.assert_called_once()
    _, call_kwargs = mock_pdisplay.call_args
    assert call_kwargs["output_format"] == _OUTPUT_FORMATS[output_format]


@pytest.mark.parametrize("skin", ["phys", "symb", "debug"])
@patch("qlass.utils.draw_circuit.pcvl.pdisplay")
def test_draw_circuit_skins(mock_pdisplay, two_qubit_processor, skin):
    """Tests that draw_circuit accepts each supported skin."""
    draw_circuit(two_qubit_processor, skin=skin)
    _, call_kwargs = mock_pdisplay.call_args
    assert call_kwargs["skin"].__class__.__name__.lower().startswith(skin[:4])


@patch("qlass.utils.draw_circuit.pcvl.pdisplay")
def test_draw_circuit_accepts_circuit(mock_pdisplay, two_qubit_processor):
    """Tests that draw_circuit accepts a Perceval Circuit directly."""
    circuit = two_qubit_processor.linear_circuit()
    draw_circuit(circuit, output_format="text")
    mock_pdisplay.assert_called_once()
    assert isinstance(mock_pdisplay.call_args[0][0], pcvl.Circuit)


@patch("qlass.utils.draw_circuit.pcvl.pdisplay_to_file")
def test_draw_circuit_save_path(mock_pdisplay_to_file, two_qubit_processor, tmp_path):
    """Tests that draw_circuit saves figures when save_path is provided."""
    save_path = tmp_path / "circuit.png"
    draw_circuit(two_qubit_processor, save_path=str(save_path))
    mock_pdisplay_to_file.assert_called_once()
    assert mock_pdisplay_to_file.call_args[0][1] == str(save_path)


@patch("qlass.utils.draw_circuit.pcvl.pdisplay")
def test_draw_circuit_compact(mock_pdisplay, two_qubit_processor):
    """Tests that compact mode is forwarded to the Perceval skin."""
    draw_circuit(two_qubit_processor, compact=True)
    _, call_kwargs = mock_pdisplay.call_args
    assert call_kwargs["skin"]._compact is True


def test_draw_circuit_invalid_output_format(two_qubit_processor):
    """Tests that an invalid output format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid output_format"):
        draw_circuit(two_qubit_processor, output_format="svg")


def test_draw_circuit_invalid_skin(two_qubit_processor):
    """Tests that an invalid skin raises ValueError."""
    with pytest.raises(ValueError, match="Invalid skin"):
        draw_circuit(two_qubit_processor, skin="neon")


def test_draw_circuit_unsupported_backend(two_qubit_processor):
    """Tests that unsupported backends raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="piquasso"):
        draw_circuit(two_qubit_processor, backend="piquasso")


def test_resolve_executor_circuit_processor(two_qubit_processor):
    """Tests resolve_executor_circuit with a Processor input."""
    assert resolve_executor_circuit(two_qubit_processor) is two_qubit_processor


def test_resolve_executor_circuit_circuit(two_qubit_processor):
    """Tests resolve_executor_circuit with a Circuit input."""
    circuit = two_qubit_processor.linear_circuit()
    assert resolve_executor_circuit(circuit) is circuit


def test_resolve_executor_circuit_invalid():
    """Tests that invalid executor results raise TypeError."""
    with pytest.raises(TypeError, match="Executor did not return"):
        resolve_executor_circuit({"counts": {"00": 500}})


@patch("qlass.vqe.vqe.draw_circuit")
def test_vqe_draw_ansatz_with_ansatz_fn(mock_draw_circuit):
    """Tests VQE.draw_ansatz when ansatz_fn is provided."""
    hamiltonian = {"II": -0.5, "ZZ": 1.0}
    vqe = VQE(hamiltonian=hamiltonian, executor=lambda p, s: {"counts": {"00": 1}}, num_params=4)
    params = np.zeros(4)

    vqe.draw_ansatz(params, ansatz_fn=le_ansatz, output_format="text")

    mock_draw_circuit.assert_called_once()
    processor = mock_draw_circuit.call_args[0][0]
    assert hasattr(processor, "linear_circuit")


@patch("qlass.vqe.vqe.draw_circuit")
def test_vqe_draw_ansatz_default_pauli_string(mock_draw_circuit):
    """Tests that VQE.draw_ansatz forwards kwargs to draw_circuit."""
    hamiltonian = {"II": -0.5, "ZZ": 1.0}
    vqe = VQE(hamiltonian=hamiltonian, executor=lambda p, s: {"counts": {"00": 1}}, num_params=4)
    params = np.zeros(4)

    vqe.draw_ansatz(params, ansatz_fn=le_ansatz)

    assert mock_draw_circuit.call_args[1] == {}


@patch("qlass.vqe.vqe.draw_circuit")
def test_vqe_draw_ansatz_executor_returns_processor(mock_draw_circuit):
    """Tests VQE.draw_ansatz when the executor returns a Processor."""
    params = np.zeros(4)

    def executor(p, pauli_string):
        return le_ansatz(p, pauli_string)

    hamiltonian = {"II": -0.5, "ZZ": 1.0}
    vqe = VQE(hamiltonian=hamiltonian, executor=executor, num_params=4)

    vqe.draw_ansatz(params)

    mock_draw_circuit.assert_called_once()


def test_vqe_draw_ansatz_sampling_executor_raises():
    """Tests that sampling executors require ansatz_fn."""
    hamiltonian = {"II": -0.5, "ZZ": 1.0}
    vqe = VQE(hamiltonian=hamiltonian, executor=lambda p, s: {"counts": {"00": 1}}, num_params=4)

    with pytest.raises(TypeError, match="Executor did not return"):
        vqe.draw_ansatz(np.zeros(4))
