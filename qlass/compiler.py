from typing import Optional, Union
from perceval.converters import QiskitConverter
import perceval as pcvl
from qiskit import QuantumCircuit


def compile(circuit: QuantumCircuit, backend_name: str = "Naive", use_postselection: bool = True, 
            input_state: Optional[Union[pcvl.StateVector, pcvl.BasicState]] = None) -> pcvl.Processor:
    """
    Convert a Qiskit quantum circuit to a Perceval processor.
    
    Args:
        circuit (QuantumCircuit): The Qiskit quantum circuit to convert
        backend_name (str): The backend to use for the Perceval processor
                           Options are: "Naive", "SLOS"
        use_postselection (bool): Whether to use postselection for the processor
        input_state (Optional[Union[pcvl.StateVector, pcvl.BasicState]]): 
                    The input state for the processor. If None, the |0...0> state is used.
    
    Returns:
        pcvl.Processor: The quantum circuit as a Perceval processor
    """
    # Initialize the Qiskit converter
    qiskit_converter = QiskitConverter(backend_name=backend_name)
    
    # Convert the circuit to a Perceval processor
    processor = qiskit_converter.convert(circuit, use_postselection=use_postselection)
    
    # Set the input state if provided, otherwise use the |0...0> state
    if input_state is None:
        processor.with_input(pcvl.LogicalState([0] * circuit.num_qubits))
    else:
        processor.with_input(input_state)
    
    return processor